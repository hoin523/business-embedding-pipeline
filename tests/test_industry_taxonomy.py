from bizembed.industry_taxonomy import (
    build_semantic_text,
    canonicalize_industry,
    industry_concept,
)


def test_canonicalize_industry_maps_korean_food_synonyms():
    assert canonicalize_industry("일반한식") == "한식"
    assert canonicalize_industry("한식음식점") == "한식"
    assert canonicalize_industry("일반음식점") == "음식점"


def test_industry_concept_returns_parent_for_synonym():
    concept = industry_concept("식자재 도매업")

    assert concept is not None
    assert concept.canonical == "식자재 도매"
    assert concept.parent == "도소매"


def test_build_semantic_text_keeps_raw_and_canonical_industry():
    text = build_semantic_text(
        entity_name="보나비",
        raw_industry="일반한식",
        context="내부회계관리팀 천호랄 출근 국내 평일",
    )

    assert text == (
        "가맹점명: 보나비 | 표준업종명: 한식 | 상위업종명: 음식점 | "
        "원업종명: 일반한식 | 문맥: 내부회계관리팀 천호랄 출근 국내 평일"
    )
