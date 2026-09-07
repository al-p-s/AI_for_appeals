import logging
from pathlib import Path
from typing import Union


def setup_logging(
    log_file: Union[Path, str],
    console: bool = True,
    level: int = logging.INFO,
) -> logging.Logger:
    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    file_handler = logging.FileHandler(str(log_path), encoding="utf-8")
    file_handler.setFormatter(formatter)
    handlers = [file_handler]

    if console:
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        handlers.append(stream_handler)

    root = logging.getLogger()
    root.setLevel(level)

    # Clear previous handlers so imports from other modules don't leak handlers
    for h in list(root.handlers):
        try:
            h.close()
        except Exception:
            pass
        root.removeHandler(h)

    for h in handlers:
        root.addHandler(h)

    # Silence chatty third-party loggers
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("watchdog").setLevel(logging.WARNING)
    logging.getLogger("pdfminer").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)

    return root
