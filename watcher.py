from datetime import datetime
import logging
from pathlib import Path
import queue
import shutil
import threading
import time
import traceback

from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer

from src.DEMO.functional.xml_export import build_xml_from_results, parse_l4_codes
from src.DEMO.loading.keryx_loader import load_keryx_models
from src.DEMO.logger_config import setup_logging
from src.DEMO.paths_config import (
    ERRORS_DIR,
    HOT_DIR,
    INPUT_DIR,
    OUTPUT_DIR,
    PROCESSED_DIR,
    WATCHER_LOG_PATH,
)
from src.DEMO.running.run_single import classify_text_from_pdf

# Настройка двойного логирования
LOG_FILE = WATCHER_LOG_PATH
setup_logging(LOG_FILE)
logger = logging.getLogger("HotFolderWatcher")

for d in [INPUT_DIR, OUTPUT_DIR, PROCESSED_DIR, ERRORS_DIR]:
    try:
        d.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        logger.warning(f"Could not create directory {d}: {e}")


# Потокобезопасная очередь задач и отслеживание активных файлов
file_queue: queue.Queue = queue.Queue()
_seen_files = set()
_seen_lock = threading.Lock()
_stop_event = threading.Event()


def enqueue_file(file_path: Path):
    resolved = file_path.resolve()
    with _seen_lock:
        if resolved in _seen_files:
            return
        _seen_files.add(resolved)
    logger.info(f"Queued for processing: {file_path.name}")
    file_queue.put(file_path)


def release_file(file_path: Path):
    resolved = file_path.resolve()
    with _seen_lock:
        _seen_files.discard(resolved)


def wait_until_file_is_ready(file_path: Path, timeout: int = 30) -> bool:
    start_time = time.time()
    last_size = -1

    while time.time() - start_time < timeout:
        try:
            current_size = file_path.stat().st_size
            if current_size > 0 and current_size == last_size:
                with open(file_path, "rb") as f:
                    pass
                return True
            last_size = current_size
        except (PermissionError, OSError):
            pass
        time.sleep(1)

    return False


def safe_move(src_path: Path, dest_dir: Path, retries: int = 5, delay: float = 0.5) -> Path:
    dest_path = dest_dir / src_path.name
    for attempt in range(retries):
        try:
            return Path(shutil.move(str(src_path), str(dest_path)))
        except (PermissionError, OSError):
            if attempt < retries - 1:
                time.sleep(delay)
            else:
                raise


def write_error_report(file_path: Path, error_message: str, tb_str: str = ""):
    error_report_path = ERRORS_DIR / f"{file_path.name}.error.txt"
    try:
        content = (
            f"Timestamp: {datetime.now().isoformat()}\n"
            f"File: {file_path.name}\n"
            f"Original Path: {file_path}\n"
            f"Error: {error_message}\n"
        )
        if tb_str:
            content += f"\nTraceback:\n{tb_str}\n"
        error_report_path.write_text(content, encoding="utf-8")
        logger.info(f"Saved error report: {error_report_path.name}")
    except Exception as e:
        logger.error(f"Failed to write error report for {file_path.name}: {e}")


SUPPORTED_EXTENSIONS = {".pdf", ".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}


def is_supported_file(path_str: str) -> bool:
    return Path(path_str).suffix.lower() in SUPPORTED_EXTENSIONS


def process_file(file_path: Path):
    logger.info(f"Processing started: {file_path.name}")

    if not file_path.exists():
        logger.warning(f"File no longer exists: {file_path.name}. Skipping.")
        return

    if not wait_until_file_is_ready(file_path):
        err_msg = f"File {file_path.name} was not released within allowed time."
        logger.error(err_msg)
        write_error_report(file_path, err_msg)
        safe_move(file_path, ERRORS_DIR)
        return

    try:
        summary, l2, l3, l4, fields, entities = classify_text_from_pdf(file_path)
        l4_codes = parse_l4_codes(l4)

        out_xml = build_xml_from_results(
            summary=summary,
            l4_codes=l4_codes,
            field_predictions=fields,
            entities=entities,
            file_name=file_path.name,
            output_dir=OUTPUT_DIR,
        )

        logger.info(f"XML-file created successfully: {out_xml.name}")

        # Небольшая задержка перед перемещением, чтобы Windows гарантированно снял блокировку
        time.sleep(0.5)
        safe_move(file_path, PROCESSED_DIR)
        logger.info(f"File {file_path.name} moved to processed/")
        logger.info("\n" + "=" * 60 + "\n")

    except Exception as e:
        tb = traceback.format_exc()
        logger.exception(f"Error processing {file_path.name}: {e}")
        write_error_report(file_path, str(e), tb)
        try:
            safe_move(file_path, ERRORS_DIR)
        except Exception:
            logger.error(f"Failed to move {file_path.name} to errors/")


def worker_loop():
    logger.info("Worker thread started.")
    while not _stop_event.is_set():
        try:
            file_path = file_queue.get(timeout=1.0)
        except queue.Empty:
            continue

        if file_path is None:
            file_queue.task_done()
            break

        try:
            process_file(file_path)
        except Exception as e:
            logger.exception(f"Unexpected worker error on {file_path}: {e}")
        finally:
            release_file(file_path)
            file_queue.task_done()

    logger.info("Worker thread stopped.")


class FileHandler(FileSystemEventHandler):
    def on_created(self, event):
        if not event.is_directory and is_supported_file(event.src_path):
            enqueue_file(Path(event.src_path))

    def on_moved(self, event):
        if not event.is_directory and is_supported_file(event.dest_path):
            enqueue_file(Path(event.dest_path))


def run_watcher():
    logger.info("Preloading KERYX models for fast inference...")
    load_keryx_models()

    # Запуск фонового рабочего потока
    worker_thread = threading.Thread(target=worker_loop, name="WatcherWorker", daemon=True)
    worker_thread.start()

    # Добавление уже существующих файлов в очередь
    existing_files = [f for f in INPUT_DIR.iterdir() if f.is_file() and is_supported_file(str(f))]
    if existing_files:
        logger.info(f"Found {len(existing_files)} document(s) in folder. Adding to queue...")
        for f in existing_files:
            enqueue_file(f)

    event_handler = FileHandler()
    observer = Observer()
    observer.schedule(event_handler, str(INPUT_DIR), recursive=False)
    observer.start()

    logger.info("=" * 60)
    logger.info(f"Hot folder watching started (Multi-format XML-mode): {INPUT_DIR}")
    logger.info(f"Supported formats: {', '.join(sorted(SUPPORTED_EXTENSIONS))}")
    logger.info(f"Logging in: {LOG_FILE}")
    logger.info("=" * 60)

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("Shutdown signal received. Stopping observer...")
        observer.stop()
        _stop_event.set()
        file_queue.put(None)
        worker_thread.join(timeout=10)
    finally:
        observer.join()
        logger.info("Hot folder watcher stopped.")


if __name__ == "__main__":
    run_watcher()

