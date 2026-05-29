from bizembed.normalize import format_record_text, normalize_entity_name, normalize_text


def test_normalize_text_compacts_business_strings():
    assert normalize_text("  (주) 보나비 / 일반한식  ") == "주 보나비 일반한식"


def test_normalize_entity_name_removes_common_corporate_suffixes():
    assert normalize_entity_name("(주) 대성푸드 ") == "대성푸드"
    assert normalize_entity_name("주식회사 대한항공") == "대한항공"


def test_format_record_text_keeps_field_boundaries():
    text = format_record_text(
        entity_name="보나비",
        entity_type="merchant",
        industry_name="일반한식",
        item_name="외식 서비스",
    )

    assert text == "가맹점명: 보나비 | 업종명: 일반한식 | 품목명: 외식 서비스"
