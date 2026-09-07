import json
import logging
from pathlib import Path
from typing import Dict, List

from src.DEMO.paths_config import FIELDS_CONFIG_PATH

logger = logging.getLogger(__name__)


def load_field_config(config_path: str = FIELDS_CONFIG_PATH) -> Dict[str, List[str]]:
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        field_candidates = {}
        for field in data:
            field_name = field.get("field_name")
            candidates = field.get("field_candidates", [])
            if field_name and candidates:
                field_candidates[field_name] = candidates
            else:
                logger.warning(f"Field {field_name} skipped: no candidates")

        logger.info(f"Loaded fields to classify: {list(field_candidates.keys())}")
        return field_candidates

    except FileNotFoundError:
        logger.error(f"File not found: {config_path}")
        return {}
    except json.JSONDecodeError as e:
        logger.error(f"Parsing JSON error: {e}")
        return {}
    except Exception as e:
        logger.error(f"Config load error: {e}")
        return {}


_field_candidates_cache = None


def get_field_candidates() -> Dict[str, List[str]]:
    global _field_candidates_cache
    if _field_candidates_cache is None:
        _field_candidates_cache = load_field_config()
    return _field_candidates_cache


logger.info("Field config loaded")
