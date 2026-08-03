import logging
import shutil
import time
from pathlib import Path
from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer

from src.DEMO.running.run_single import classify_text_from_pdf
from src.DEMO.functional.xml_export import build_xml_from_results, parse_l4_codes

PROJECT_ROOT = Path(__file__).resolve().parent

# Настройка двойного логирования
LOG_DIR = PROJECT_ROOT / "src" / "DEMO" / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = LOG_DIR / "watcher.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, encoding="utf-8"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("HotFolderWatcher")

# Рабочие папки горячей директории
HOT_DIR = Path(r"D:\Обращения\0_DEMO")

INPUT_DIR = HOT_DIR / "input"
OUTPUT_DIR = HOT_DIR / "output"
PROCESSED_DIR = HOT_DIR / "processed"
ERRORS_DIR = HOT_DIR / "errors"

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


def process_pdf(pdf_path: Path):
    logger.info(f"New file detected: {pdf_path.name}")

    if not wait_until_file_is_ready(pdf_path):
        logger.error(f"File {pdf_path.name} was not released within allowed time. Skipping.")
        safe_move(pdf_path, ERRORS_DIR)
        return

    try:
        summary, l2, l3, l4, fields, entities = classify_text_from_pdf(pdf_path)
        l4_codes = parse_l4_codes(l4)

        out_xml = build_xml_from_results(
            summary=summary,
            l4_codes=l4_codes,
            field_predictions=fields,
            entities=entities,
            file_name=pdf_path.name,
            output_dir=OUTPUT_DIR,
        )

        logger.info(f"XML-file created successfully: {out_xml.name}")

        # Небольшая задержка перед перемещением, чтобы Windows гарантированно снял блокировку
        time.sleep(0.5)
        safe_move(pdf_path, PROCESSED_DIR)
        logger.info(f"File {pdf_path.name} moved to processed/")
        logger.info("\n" +"=" * 60 + "\n")

    except Exception as e:
        logger.exception(f"Error processing {pdf_path.name}: {e}")
        try:
            safe_move(pdf_path, ERRORS_DIR)
        except Exception:
            logger.error(f"Failed to move {pdf_path.name} to errors/")


class PDFHandler(FileSystemEventHandler):
    def on_created(self, event):
        if not event.is_directory and event.src_path.lower().endswith(".pdf"):
            process_pdf(Path(event.src_path))

    def on_moved(self, event):
        if not event.is_directory and event.dest_path.lower().endswith(".pdf"):
            process_pdf(Path(event.dest_path))


def run_watcher():
    existing_files = list(INPUT_DIR.glob("*.pdf"))
    if existing_files:
        logger.info(f"Found {len(existing_files)} PDF in folder. Processing...")
        for pdf in existing_files:
            process_pdf(pdf)

    event_handler = PDFHandler()
    observer = Observer()
    observer.schedule(event_handler, str(INPUT_DIR), recursive=False)
    observer.start()

    logger.info("=" * 60)
    logger.info(f"Hot folder watching started (XML-mode): {INPUT_DIR}")
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
