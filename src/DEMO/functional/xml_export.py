# generating an XML export based on the classification/NER results for a single request

import logging
import xml.etree.ElementTree as ET
from pathlib import Path

from src.DEMO.running.run_single import classify_text
from src.DEMO.functional.ner_inference import extract_entities

logger = logging.getLogger(__name__)

REFS_XML_PATH = "../../../data/classifier/all_refs(but_orgs).xml"
XML_OUTPUT_DIR = "../xmls"

# Справочные поля MainData/PetitionerData: field_name (в field_predictions) -> тег в XML.
# Значение резолвится через справочник (см. load_reference_dicts) - ищем RowID по Name,
# если не нашли - пишем текст как есть (для пополняемых справочников)
MAIN_REF_FIELDS = {
    "AppealKind": "AppealKind",
    "StatusId": "StatusId",
    "ItemID": "ItemID",
    "DeliveryTypeId": "DeliveryTypeId",
    "RegistrationPlaceId": "RegistrationPlaceId",
}
PETITIONER_REF_FIELDS = {
    "PetitionerCategory": "PetitionerCategory",
    "PetitionerDistrict": "PetitionerDistrict",
}

# NER-поля PetitionerData: NER-лейбл -> тег в XML (берём первого кандидата)
PETITIONER_NER_FIELDS = {
    "LAST_NAME": "PetitionerSurname",
    "FIRST_NAME": "PetitionerName",
    "MIDDLE_NAME": "PetitionerPatronymic",
    "ADDRESS": "PetitionerAddress",
    "PERSONAL_EMAIL": "PetitionerEmail",
    "PHONE_NUMBER": "PetitionerPhone",
}

# Плоские группы справочника (Name внутри Items -> RowID), 1:1 с полями выше
FIELD_TO_GROUP_NAME = {
    "ItemID": "Форма обращения",
    "DeliveryTypeId": "Источник поступления",
    "RegistrationPlaceId": "Место события",
    "PetitionerDistrict": "Район проживания заявителя",
    "PetitionerCategory": "Категория заявителя",
}

UNDEFINED_MARKERS = {"не определяется"}

_ref_dicts_cache = None


def _is_undefined(value):
    if value is None:
        return True
    v = str(value).strip()
    return not v or v.lower() in UNDEFINED_MARKERS


def load_reference_dicts(path=REFS_XML_PATH):
    global _ref_dicts_cache
    if _ref_dicts_cache is not None:
        return _ref_dicts_cache

    root = ET.parse(path).getroot()
    top_groups = {
        g.get("Name"): g
        for g in root.find("./RefBaseUniversal/ItemTypes").findall("./ItemTypesRow")
    }

    dicts = {
        field: {
            row.get("Name"): row.get("RowID")
            for row in top_groups[group_name].findall("./Items/ItemsRow")
        }
        for field, group_name in FIELD_TO_GROUP_NAME.items()
        if group_name in top_groups
    }

    # "Тип обращения": 4 вложенные ItemTypesRow (Заявление/Жалоба/Не обращение/
    # Предложение) -> AppealKind, их дети (ItemsRow) -> StatusId.
    # "Код вопроса" - отдельная вложенная группа -> QuestionCode
    appeal_kind, status_id, question_code = {}, {}, {}
    tip_obrasheniya = top_groups.get("Тип обращения")
    if tip_obrasheniya is not None:
        for nested in tip_obrasheniya.findall("./ItemTypesRow"):
            if nested.get("Name") == "Код вопроса":
                for row in nested.findall("./Items/ItemsRow"):
                    code = row.get("Name", "").split(" ", 1)[0].strip()
                    question_code[code] = row.get("RowID")
                continue
            appeal_kind[nested.get("Name")] = nested.get("RowID")
            for row in nested.findall("./Items/ItemsRow"):
                status_id[row.get("Name")] = row.get("RowID")

    dicts.update(AppealKind=appeal_kind, StatusId=status_id, QuestionCode=question_code)
    _ref_dicts_cache = dicts
    return dicts


def resolve_ref_value(field_name, predicted_text, ref_dicts):
    if _is_undefined(predicted_text):
        return None

    d = ref_dicts.get(field_name, {})
    if predicted_text in d:
        return d[predicted_text]

    norm = predicted_text.strip().lower()
    for name, rid in d.items():
        if name and name.strip().lower() == norm:
            return rid

    logger.warning(f"[{field_name}] значение '{predicted_text}' не найдено в справочнике, пишем текстом")
    return predicted_text


def resolve_consideration_type(predicted_text):
    if _is_undefined(predicted_text):
        return None
    text = predicted_text.strip()
    if text in {"0", "1", "2"}:
        return int(text)
    logger.warning(f"[ConsiderationType] значение '{predicted_text}' не распознано")
    return None


def resolve_question_codes(l4_codes, ref_dicts):
    q_map = ref_dicts.get("QuestionCode", {})
    result = []
    for code in l4_codes:
        row_id = q_map.get(code)
        if not row_id:
            logger.warning(f"[QuestionCode] код '{code}' не найден в справочнике, пишем текстом")
        result.append(row_id or code)
    return result


def first_or_none(values):
    if not values:
        return None
    return None if _is_undefined(values[0]) else values[0]


def add_text_element(parent, tag, value):
    if _is_undefined(value):
        return
    ET.SubElement(parent, tag).text = str(value)


def build_container(tag, fill_fn):
    el = ET.Element(tag)
    fill_fn(el)
    return el if len(el) > 0 else None


def parse_l4_codes(l4_text):
    return [line.split(" — ")[0].strip() for line in l4_text.split("\n") if line.strip()]


def build_appeal_xml(text, file_name, output_dir=XML_OUTPUT_DIR):
    ref_dicts = load_reference_dicts()
    summary, _, _, l4_text, field_predictions = classify_text(text)
    entities = extract_entities(text)

    root = ET.Element("Data")

    def fill_main_data(el):
        for field_name, tag in MAIN_REF_FIELDS.items():
            add_text_element(el, tag, resolve_ref_value(field_name, field_predictions.get(field_name), ref_dicts))
        add_text_element(el, "ConsiderationType", resolve_consideration_type(field_predictions.get("ConsiderationType")))
        add_text_element(el, "Content", summary)

    def fill_question_data(el):
        for row_id in resolve_question_codes(parse_l4_codes(l4_text), ref_dicts):
            add_text_element(el, "QuestionCode", row_id)

    def fill_petitioner_data(el):
        for ner_label, tag in PETITIONER_NER_FIELDS.items():
            add_text_element(el, tag, first_or_none(entities.get(ner_label)))
        for field_name, tag in PETITIONER_REF_FIELDS.items():
            add_text_element(el, tag, resolve_ref_value(field_name, field_predictions.get(field_name), ref_dicts))

    for tag, fill_fn in [
        ("MainData", fill_main_data),
        ("QuestionData", fill_question_data),
        ("PetitionerData", fill_petitioner_data),
    ]:
        container = build_container(tag, fill_fn)
        if container is not None:
            root.append(container)

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{Path(file_name).stem}.xml"

    tree = ET.ElementTree(root)
    ET.indent(tree, space="  ")
    tree.write(out_path, encoding="utf-16", xml_declaration=True)

    logger.info(f"XML saved: {out_path}")
    return out_path


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    from src.DEMO.running.run_single import HARDCODED_TEXT
    build_appeal_xml(HARDCODED_TEXT, "test_appeal.pdf")
