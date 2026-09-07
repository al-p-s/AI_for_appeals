import logging
import shutil
import time
from pathlib import Path
from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer

from src.DEMO.running.run_single import classify_text_from_pdf
from src.DEMO.functional.xml_export import build_xml_from_results, parse_l4_codes
from src.DEMO.paths_config import (
    WATCHER_LOG_PATH,
    HOT_DIR,
    INPUT_DIR,
    OUTPUT_DIR,
    PROCESSED_DIR,
    ERRORS_DIR,
)
from src.DEMO.logger_config import setup_logging

# Настройка двойного логирования
LOG_FILE = WATCHER_LOG_PATH
setup_logging(LOG_FILE)
logger = logging.getLogger("HotFolderWatcher")

for d in [INPUT_DIR, OUTPUT_DIR, PROCESSED_DIR, ERRORS_DIR]:
    d.mkdir(parents=True, exist_ok=True)


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


SUPPORTED_EXTENSIONS = {".pdf", ".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}


def is_supported_file(path_str: str) -> bool:
    return Path(path_str).suffix.lower() in SUPPORTED_EXTENSIONS


def process_file(file_path: Path):
    logger.info(f"New file detected: {file_path.name}")

    if not wait_until_file_is_ready(file_path):
        logger.error(f"File {file_path.name} was not released within allowed time. Skipping.")
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
        logger.info("\n" +"=" * 60 + "\n")

    except Exception as e:
        logger.exception(f"Error processing {file_path.name}: {e}")
        try:
            safe_move(file_path, ERRORS_DIR)
        except Exception:
            logger.error(f"Failed to move {file_path.name} to errors/")


class FileHandler(FileSystemEventHandler):
    def on_created(self, event):
        if not event.is_directory and is_supported_file(event.src_path):
            process_file(Path(event.src_path))

    def on_moved(self, event):
        if not event.is_directory and is_supported_file(event.dest_path):
            process_file(Path(event.dest_path))


def run_watcher():
    existing_files = [f for f in INPUT_DIR.iterdir() if f.is_file() and is_supported_file(str(f))]
    if existing_files:
        logger.info(f"Found {len(existing_files)} document(s) in folder. Processing...")
        for f in existing_files:
            process_file(f)

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
        observer.stop()
        logger.info("Hot folder stopping...")
    observer.join()


if __name__ == "__main__":
    run_watcher()
