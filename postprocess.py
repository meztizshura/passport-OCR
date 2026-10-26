import csv
import datetime
import json
import os
import re
import sys
from collections import defaultdict

FIELDS = [
    "Last_name_ru",
    "First_name_ru",
    "Middle_name_ru",
    "Sex_ru",
    "Birth_date",
    "Birth_place_ru",
    "Issue_date",
    "Issue_organisation_code",
    "Licence_number",
    "Issue_organization_ru",
]

NAME_FIELDS = ["Last_name_ru", "First_name_ru", "Middle_name_ru"]

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_TRAIN = os.path.join(BASE_DIR, "data", "train_labels.jsonl")
DEFAULT_REFERENCE = os.path.join(BASE_DIR, "reference", "subdivisions.csv")
REFERENCE_DATE = datetime.date(2026, 10, 1)

CFG = {
    "free_conf": 0.99,
    "name_conf": 0.99,
    "fill_min": 0.80,
    "birth_onsets": True,
}

MRZ_MAP = {
    "A": "А", "B": "Б", "V": "В", "G": "Г", "D": "Д", "E": "Е",
    "J": "Ж", "Z": "З", "I": "И", "Q": "Й", "K": "К", "L": "Л",
    "M": "М", "N": "Н", "O": "О", "P": "П", "R": "Р", "S": "С",
    "T": "Т", "U": "У", "F": "Ф", "H": "Х", "C": "Ц", "3": "Ч",
    "4": "Ш", "W": "Щ", "X": "Ъ", "Y": "Ы", "9": "Ь", "6": "Э",
    "7": "Ю", "8": "Я", "2": "Ё",
}

CYR_RE = re.compile(r"[^А-ЯЁ]")
TOKEN_RE = re.compile(r"[А-ЯЁA-Z0-9]+(?:-[А-ЯЁA-Z0-9]+)*")
DATE_RE = re.compile(r"^(\d{2})\.(\d{2})\.(\d{4})$")
ADJ_END_RE = re.compile(r"(СКОГО|СКОЙ|СКОМ|СКОМУ|ЦКОГО|ЦКОЙ|ЦКОМ|ЦКОМУ)$")
BAD_END = {
    "В", "ВО", "НА", "ПО", "С", "СО", "И", "ИЗ", "ОТ", "ДО",
    "ПРИ", "ДЛЯ", "О", "ОБ", "ЗА", "У", "К", "КО", "НАД",
    "ПОД", "БЕЗ", "ЧЕРЕЗ",
}
PREPOSITIONS = BAD_END | {"РОССИИ"}
BIRTH_BAD_START = {
    "Р-НА", "Р-Н", "РАЙОНА", "РАЙОН", "ОБЛ", "ОБЛ.", "КР", "КР.",
    "РЕСП", "РЕСП.", "ОБЛАСТИ", "КРАЯ",
}
ORG_BAD_START = BIRTH_BAD_START | {
    "РОССИИ", "ПО", "В", "НА", "И", "Р-НЕ", "РАЙОНЕ", "ОБЛАСТЬ",
}


def norm(value):
    return " ".join(str(value or "").split()).upper()


def cyr_letters(value):
    return CYR_RE.sub("", value.upper())


def tokens(value):
    return TOKEN_RE.findall(value.upper())


def parse_date(value):
    if not value:
        return None
    text = norm(value)
    match = DATE_RE.fullmatch(text)
    if match:
        try:
            return datetime.date(int(match.group(3)), int(match.group(2)), int(match.group(1)))
        except ValueError:
            return None
    digits = re.sub(r"\D", "", text)
    if len(digits) == 8:
        try:
            return datetime.date(int(digits[4:8]), int(digits[2:4]), int(digits[0:2]))
        except ValueError:
            return None
    return None


def add_years(value, years):
    try:
        return value.replace(year=value.year + years)
    except ValueError:
        return value.replace(year=value.year + years, day=28)


def date_to_text(value):
    return value.strftime("%d.%m.%Y") if value else None


def canon_code(value):
    digits = re.sub(r"\D", "", norm(value))
    if len(digits) != 6:
        return None
    return digits[:3] + "-" + digits[3:]


def canon_licence(value):
    digits = re.sub(r"\D", "", norm(value))
    if len(digits) != 10:
        return None
    return digits[:2] + " " + digits[2:4] + " " + digits[4:]


def canon_sex(value):
    text = norm(value)
    if text.startswith("МУЖ"):
        return "МУЖ."
    if text.startswith("ЖЕН"):
        return "ЖЕН."
    return text


def canon(field, value):
    if field in ("Birth_date", "Issue_date"):
        parsed = parse_date(value)
        return date_to_text(parsed) if parsed else norm(value)
    if field == "Issue_organisation_code":
        return canon_code(value) or norm(value)
    if field == "Licence_number":
        return canon_licence(value) or norm(value)
    if field == "Sex_ru":
        return canon_sex(value)
    return norm(value)


def mrz_check_digit(value):
    total = 0
    for index, char in enumerate(value):
        if char.isdigit():
            digit = int(char)
        elif "A" <= char <= "Z":
            digit = ord(char) - 55
        else:
            return None
        total += digit * (7, 3, 1)[index % 3]
    return total % 10


def mrz_date(value, reference):
    if not re.fullmatch(r"\d{6}", value or ""):
        return None
    yy = int(value[0:2])
    month = int(value[2:4])
    day = int(value[4:6])
    candidates = []
    for year in (1900 + yy, 2000 + yy):
        try:
            candidates.append(datetime.date(year, month, day))
        except ValueError:
            continue
    if not candidates:
        return None
    if reference:
        return min(candidates, key=lambda item: abs((item - reference).days))
    return candidates[-1]


def field_text(prediction, field):
    item = (prediction.get("viz") or {}).get(field)
    if not item:
        return None
    lines = item.get("lines") or []
    text = " ".join((line.get("text") or "") for line in lines)
    return norm(text)


def field_min_conf(prediction, field):
    item = (prediction.get("viz") or {}).get(field)
    if not item:
        return 0.0
    values = []
    for line in item.get("lines") or []:
        values.extend(line.get("char_conf") or [])
    return min(values) if values else 0.0


def field_fill(prediction, field):
    item = (prediction.get("viz") or {}).get(field)
    if not item:
        return 0.0
    box = item.get("box") or [0, 0, 0, 0]
    height = box[3]
    lines = item.get("lines") or []
    if not height or not lines:
        return 0.0
    tops = [line["box"][1] for line in lines if line.get("box")]
    bottoms = [line["box"][1] + line["box"][3] for line in lines if line.get("box")]
    if not tops or not bottoms:
        return 0.0
    return (max(bottoms) - min(tops)) / height


def parse_mrz(prediction):
    mrz = prediction.get("mrz")
    if not mrz:
        return None
    lines = mrz.get("lines") or []
    name_line = None
    number_line = None
    for line in lines:
        text = line.get("text") or ""
        if len(text) != 44:
            continue
        if text.startswith("PNRUS"):
            name_line = line
        elif text[:9].isdigit():
            number_line = line
    if not name_line or not number_line:
        return None
    name_text = name_line["text"]
    decoded = []
    for char in name_text[5:]:
        if char == "<":
            continue
        if char not in MRZ_MAP:
            return None
        decoded.append(MRZ_MAP[char])
    text = number_line["text"]
    number = text[:9]
    number_check = text[9]
    birth_raw = text[13:19]
    birth_check = text[19]
    sex = text[20] if text[20] in ("M", "F") else None
    issue_raw = text[29:35]
    code_raw = text[35:41]
    if not re.fullmatch(r"\d{6}", birth_raw) or not re.fullmatch(r"\d{6}", issue_raw):
        return None
    if not re.fullmatch(r"\d{6}", code_raw):
        return None
    return {
        "names": "".join(decoded),
        "name_truncated": not name_text.endswith("<"),
        "number": number,
        "number_check_ok": number_check.isdigit()
        and mrz_check_digit(number) == int(number_check),
        "birth_raw": birth_raw,
        "birth_check_ok": birth_check.isdigit()
        and mrz_check_digit(birth_raw) == int(birth_check),
        "sex": sex,
        "issue_raw": issue_raw,
        "code": canon_code(code_raw),
    }


def build_context(labels, reference_path=DEFAULT_REFERENCE):
    known_birth = set()
    known_org = set()
    org_by_code = defaultdict(set)
    org_vocab = set()
    geo_vocab = set()
    for item in labels:
        fields = item.get("fields") or {}
        birth = norm(fields.get("Birth_place_ru"))
        organisation = norm(fields.get("Issue_organization_ru"))
        code = canon_code(fields.get("Issue_organisation_code"))
        if birth:
            known_birth.add(birth)
            geo_vocab.update(tokens(birth))
        if organisation:
            known_org.add(organisation)
            org_vocab.update(tokens(organisation))
        if code:
            org_by_code[code].add(organisation)
    with open(reference_path, encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            name = norm(row.get("name"))
            org_vocab.update(tokens(name))
            geo_vocab.update(tokens(name))
    geo_onsets = {token[:2] for token in geo_vocab if len(token) >= 2}
    return {
        "known_birth": known_birth,
        "known_org": known_org,
        "org_by_code": org_by_code,
        "org_vocab": org_vocab,
        "geo_vocab": geo_vocab,
        "geo_onsets": geo_onsets,
    }


def punctuation_ok(text, field):
    if not text:
        return False
    if text.count("(") != text.count(")"):
        return False
    if text.count("[") != text.count("]"):
        return False
    if re.search(r"[.,;:]{2,}", text):
        return False
    if "--" in text:
        return False
    if re.search(r"(ОБЛ|КР|РЕСП|ОКР|АДМ|АВТ)\.(?=[А-ЯЁ])", text):
        return False
    parts = tokens(text)
    if not parts:
        return False
    if field == "Birth_place_ru":
        if parts[0] in BIRTH_BAD_START:
            return False
    elif parts[0] in ORG_BAD_START:
        return False
    if parts[-1] in BAD_END:
        return False
    if ADJ_END_RE.search(parts[-1]):
        previous = parts[-2] if len(parts) > 1 else ""
        if previous in PREPOSITIONS:
            return False
    return True


def free_text_ok(prediction, field, known_pass, context, cfg):
    text = field_text(prediction, field)
    if text is None:
        return False
    if known_pass:
        return True
    if field_min_conf(prediction, field) < cfg["free_conf"]:
        return False
    if field_fill(prediction, field) < cfg["fill_min"]:
        return False
    if not punctuation_ok(text, field):
        return False
    if field == "Birth_place_ru" and cfg["birth_onsets"]:
        for token in tokens(text):
            if (
                len(token) >= 3
                and token not in context["geo_vocab"]
                and token[:2] not in context["geo_onsets"]
            ):
                return False
    if field == "Issue_organization_ru":
        if any(token not in context["org_vocab"] for token in tokens(text)):
            return False
    return True


def is_valid_passport(birth_date, issue_date):
    if not birth_date or not issue_date:
        return False
    if issue_date > REFERENCE_DATE or issue_date < birth_date:
        return False
    if issue_date < add_years(birth_date, 14):
        return False
    if REFERENCE_DATE < add_years(birth_date, 14):
        return False
    for milestone_years in (20, 45):
        milestone = add_years(birth_date, milestone_years)
        if REFERENCE_DATE >= milestone and issue_date < milestone:
            return False
    return True


def decide(prediction, context, cfg=None):
    cfg = cfg or CFG
    texts = {field: field_text(prediction, field) for field in FIELDS}
    if any(value is None for value in texts.values()):
        return "reject", {}, "missing"
    mrz = parse_mrz(prediction)
    if not mrz or not mrz["names"]:
        return "reject", {}, "mrz"
    if mrz["name_truncated"]:
        return "reject", {}, "name_truncated"
    for field in NAME_FIELDS:
        if field_min_conf(prediction, field) < cfg["name_conf"]:
            return "reject", {}, "name_conf"
    viz_names = "".join(cyr_letters(texts[field]) for field in NAME_FIELDS)
    if viz_names != mrz["names"]:
        return "reject", {}, "names"

    sex_viz = None
    if texts["Sex_ru"].startswith("МУЖ"):
        sex_viz = "M"
    elif texts["Sex_ru"].startswith("ЖЕН"):
        sex_viz = "F"
    if not sex_viz or mrz["sex"] != sex_viz:
        return "reject", {}, "sex"
    middle = texts["Middle_name_ru"]
    if re.search(r"(ОВИЧ|ЕВИЧ|ИЧ)$", middle):
        expected_sex = "M"
    elif re.search(r"(ОВНА|ЕВНА|ИЧНА|ИНИЧНА)$", middle):
        expected_sex = "F"
    else:
        expected_sex = None
    if expected_sex and expected_sex != sex_viz:
        return "reject", {}, "sex_suffix"

    birth_date = parse_date(texts["Birth_date"])
    issue_date = parse_date(texts["Issue_date"])
    mrz_birth = mrz_date(mrz["birth_raw"], birth_date)
    mrz_issue = mrz_date(mrz["issue_raw"], issue_date)
    if not birth_date or not issue_date or not mrz_birth or not mrz_issue:
        return "reject", {}, "dates"
    if birth_date != mrz_birth or issue_date != mrz_issue:
        return "reject", {}, "dates_mrz"
    if not mrz["birth_check_ok"]:
        return "reject", {}, "birth_check"

    code = canon_code(texts["Issue_organisation_code"])
    if not code or code != mrz["code"]:
        return "reject", {}, "code"

    licence = canon_licence(texts["Licence_number"])
    if not licence:
        return "reject", {}, "licence"
    licence_digits = licence.replace(" ", "")
    if licence_digits[:3] + licence_digits[4:] != mrz["number"]:
        return "reject", {}, "licence_mrz"
    if not mrz["number_check_ok"]:
        return "reject", {}, "licence_check"

    if not is_valid_passport(birth_date, issue_date):
        return "reject", {}, "invalid"

    org_text = texts["Issue_organization_ru"]
    if code in context["org_by_code"]:
        org_known = org_text in context["org_by_code"][code]
    else:
        org_known = org_text in context["known_org"]
    if not free_text_ok(
        prediction,
        "Birth_place_ru",
        texts["Birth_place_ru"] in context["known_birth"],
        context,
        cfg,
    ):
        return "reject", {}, "birth_place"
    if not free_text_ok(
        prediction,
        "Issue_organization_ru",
        org_known,
        context,
        cfg,
    ):
        return "reject", {}, "organization"

    fields = {field: canon(field, texts[field]) for field in FIELDS}
    return "accept", fields, "ok"


def load_jsonl(path):
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def main():
    if len(sys.argv) != 3:
        sys.exit("usage: python postprocess.py <predictions.jsonl> <answers.jsonl>")
    predictions = load_jsonl(sys.argv[1])
    labels = load_jsonl(DEFAULT_TRAIN)
    context = build_context(labels, DEFAULT_REFERENCE)
    with open(sys.argv[2], "w", encoding="utf-8") as handle:
        for prediction in predictions:
            decision, fields, _ = decide(prediction, context, CFG)
            row = {
                "doc_id": prediction["doc_id"],
                "decision": decision,
                "fields": fields if decision == "accept" else {},
            }
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()

