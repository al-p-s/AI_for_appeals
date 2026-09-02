"""
Centralised configuration for filesystem paths used by the DEMO code.

All absolute Windows paths are built from a single root directory.
The root can be overridden at runtime with the environment variable
BASE_DATA_DIR (e.g. `BASE_DATA_DIR="C:\\Data\\Appeals"`).

Only paths that are required by the DEMO package are defined here.
"""

import os
from pathlib import Path

# ----------------------------------------------------------------------
# Root data directory – default matches current hard‑coded layout.
# ----------------------------------------------------------------------
BASE_DATA_DIR: Path = Path(
    os.getenv(
        "BASE_DATA_DIR",
        Path(__file__).resolve().parents[2] / "data"
    )
)

# ----------------------------------------------------------------------
# Hot‑folder (watcher) sub‑directories
# ----------------------------------------------------------------------
HOT_DIR: Path = BASE_DATA_DIR / "0_DEMO"
INPUT_DIR: Path = HOT_DIR / "input"
OUTPUT_DIR: Path = HOT_DIR / "output"
PROCESSED_DIR: Path = HOT_DIR / "processed"
ERRORS_DIR: Path = HOT_DIR / "errors"

# ----------------------------------------------------------------------
# Batch / test PDFs used by run_batch.py
# ----------------------------------------------------------------------
PDF_DIR: Path = BASE_DATA_DIR / "100_test_appeals"

# ----------------------------------------------------------------------
# OCR testing dataset (dataset_for_OCR_testing.py)
# ----------------------------------------------------------------------
PDF_DIRECTORY: Path = Path(
    os.getenv(
        "OCR_PDF_DIR",
        BASE_DATA_DIR / "ocr_input"
    )
)

# ----------------------------------------------------------------------
# Form‑appeal category / copy‑folder helpers
# ----------------------------------------------------------------------
CATS_DIR: Path = BASE_DATA_DIR / "chel_cats"
ALL_SOURCE_DIR: Path = BASE_DATA_DIR / "ALL"
ALL_DEST_DIR: Path = BASE_DATA_DIR / "100_test_appeals"

# ----------------------------------------------------------------------
# XML‑with‑fields dataset input (create_fields_dataset.py)
# ----------------------------------------------------------------------
XML_FIELDS_INPUT_DIR: Path = BASE_DATA_DIR / "xml_w_fields"

# ----------------------------------------------------------------------
# End of config
# ----------------------------------------------------------------------
