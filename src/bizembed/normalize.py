from __future__ import annotations

import re
from typing import Optional


FIELD_LABELS = {
    "merchant": "가맹점명",
    "supplier": "공급업체명",
    "company": "업체명",
}

CORPORATE_PREFIX_PATTERNS = (
    r"^주식회사\s*",
    r"^유한회사\s*",
    r"^농업회사법인\s*",
    r"^사회적협동조합\s*",
    r"^\(?주\)?\s*",
    r"^㈜\s*",
)

CORPORATE_SUFFIX_PATTERNS = (
    r"\s*\(?주\)?$",
    r"\s*㈜$",
    r"\s*주식회사$",
    r"\s*유한회사$",
)


def normalize_text(value: object) -> str:
    """Normalize noisy Korean business strings without removing semantic words."""
    if value is None:
        return ""

    text = str(value).strip()
    if not text or text.lower() == "nan":
        return ""

    text = re.sub(r"[()]", " ", text)
    text = re.sub(r"[|,/\\;:\t\r\n]+", " ", text)
    text = re.sub(r"[\[\]{}<>]+", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def normalize_entity_name(value: object) -> str:
    """Normalize entity names for grouping while preserving the displayed name elsewhere."""
    text = normalize_text(value)
    for pattern in CORPORATE_PREFIX_PATTERNS:
        text = re.sub(pattern, "", text)
    for pattern in CORPORATE_SUFFIX_PATTERNS:
        text = re.sub(pattern, "", text)
    return re.sub(r"\s+", "", text).strip()


def clean_optional(value: Optional[object]) -> str:
    return normalize_text(value) if value is not None else ""


def format_record_text(
    *,
    entity_name: object,
    entity_type: str = "company",
    industry_name: Optional[object] = None,
    item_name: Optional[object] = None,
) -> str:
    """Build a field-aware text representation for embedding models."""
    name = normalize_text(entity_name)
    industry = clean_optional(industry_name)
    item = clean_optional(item_name)
    label = FIELD_LABELS.get(entity_type, FIELD_LABELS["company"])

    fields = []
    if name:
        fields.append(f"{label}: {name}")
    if industry:
        fields.append(f"업종명: {industry}")
    if item:
        fields.append(f"품목명: {item}")
    return " | ".join(fields)
