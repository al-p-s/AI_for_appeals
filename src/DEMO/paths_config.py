import os
from pathlib import Path
from dotenv import load_dotenv

# Project root: AI_for_appeals/
PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]

# Automatically load .env from project root if present
load_dotenv(PROJECT_ROOT / ".env")

# Base directories
DATA_DIR: Path = Path(os.getenv("DATA_DIR", PROJECT_ROOT / "data"))
MODELS_DIR: Path = Path(os.getenv("MODELS_DIR", PROJECT_ROOT / "models"))
DEMO_DIR: Path = PROJECT_ROOT / "src" / "DEMO"
LOGS_DIR: Path = Path(os.getenv("LOGS_DIR", DEMO_DIR / "logs"))
XML_OUTPUT_DIR: Path = Path(os.getenv("XML_OUTPUT_DIR", DEMO_DIR / "xmls"))

# Ensure writable runtime directories exist
LOGS_DIR.mkdir(parents=True, exist_ok=True)
XML_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Classifier & Reference dictionaries (data/classifier)
CLASSIFIER_DIR: Path = DATA_DIR / "classifier"
CATS_L1_PATH: Path = CLASSIFIER_DIR / "cats1.json"
CATS_L2_PATH: Path = CLASSIFIER_DIR / "cats2.json"
CATS_L3_PATH: Path = CLASSIFIER_DIR / "cats3.json"
CATS_L4_PATH: Path = CLASSIFIER_DIR / "cats4.json"
FIELDS_CONFIG_PATH: Path = CLASSIFIER_DIR / "category_fields.json"
REFS_XML_PATH: Path = CLASSIFIER_DIR / "all_refs(but_orgs).xml"
ORGS_XML_PATH: Path = CLASSIFIER_DIR / "orgs.xml"

# Model checkpoints (models/)
KERYX_BASE_DIR: Path = MODELS_DIR / "KERYX_1340_G"
KERYX_PATH_L2: Path = KERYX_BASE_DIR / "L2"
KERYX_PATH_L3: Path = KERYX_BASE_DIR / "L3"
KERYX_PATH_L4: Path = KERYX_BASE_DIR / "L4"

KERYX_FIELDS_DIR_TEMPLATE: str = str(MODELS_DIR / "KERYXes_for_fields" / "KERYX_field_{}")
GIGACHAT_PATH: Path = MODELS_DIR / "GigaChat_Lite_NEW"

# Hot folder (watcher.py)
HOT_DIR: Path = Path(os.getenv("HOT_DIR", r"D:\Обращения\0_DEMO"))
INPUT_DIR: Path = Path(os.getenv("INPUT_DIR", HOT_DIR / "input"))
OUTPUT_DIR: Path = Path(os.getenv("OUTPUT_DIR", HOT_DIR / "output"))
PROCESSED_DIR: Path = Path(os.getenv("PROCESSED_DIR", HOT_DIR / "processed"))
ERRORS_DIR: Path = Path(os.getenv("ERRORS_DIR", HOT_DIR / "errors"))

# Testing & Batch evaluation datasets
TEST_DATASET_PATH: Path = DATA_DIR / "sets_to_learn" / "appeals_w_cats" / "100_for_test_G.json"
TARGET_FIELDS_PATH: Path = DATA_DIR / "sets_to_learn" / "fields" / "target_fields_test_100.json"
PDF_TEST_DIR: Path = Path(os.getenv("PDF_TEST_DIR", r"D:\Обращения\100_test_appeals"))
ERRORS_OUTPUT_PATH: Path = LOGS_DIR / "classification_errors.json"
DEFAULT_TEST_PDF: Path = DEMO_DIR / "337-9-1.pdf"
DEBUG_OCR_FILE: Path = LOGS_DIR / "ocr_150dpi_debug.txt"

# Log files (centralized log file paths for services and scripts)
RUN_SINGLE_LOG_PATH: Path = Path(os.getenv("RUN_SINGLE_LOG_PATH", LOGS_DIR / "run_single.log"))
RUN_BATCH_LOG_PATH: Path = Path(os.getenv("RUN_BATCH_LOG_PATH", LOGS_DIR / "run_batch_100_2NER_prompts.log"))
WATCHER_LOG_PATH: Path = Path(os.getenv("WATCHER_LOG_PATH", LOGS_DIR / "watcher.log"))
APP_LOG_PATH: Path = Path(os.getenv("APP_LOG_PATH", LOGS_DIR / "app.log"))
API_LOG_PATH: Path = Path(os.getenv("API_LOG_PATH", LOGS_DIR / "api.log"))
