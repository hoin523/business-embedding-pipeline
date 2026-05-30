from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Dict, Iterable, List


@dataclass(frozen=True)
class IndustryConcept:
    canonical: str
    parent: str
    aliases: tuple[str, ...]


INDUSTRY_CONCEPTS: tuple[IndustryConcept, ...] = (
    IndustryConcept("한식", "음식점", ("일반한식", "한식음식점", "한식", "한식당", "백반", "국밥", "분식")),
    IndustryConcept("음식점", "음식점", ("일반음식점", "식당", "음식점", "외식업", "요식업")),
    IndustryConcept("카페", "음식점", ("커피전문점", "카페", "커피숍", "음료점", "다과점")),
    IndustryConcept("제과점", "음식점", ("제과점", "베이커리", "빵집", "제빵", "제과제빵")),
    IndustryConcept("중식", "음식점", ("중식", "중식음식점", "중국집", "중화요리")),
    IndustryConcept("일식", "음식점", ("일식", "일식음식점", "초밥", "스시", "횟집")),
    IndustryConcept("항공사", "교통/여행", ("항공사", "항공권", "항공운송", "국내선항공", "국제선항공", "저비용항공사")),
    IndustryConcept("여행사", "교통/여행", ("여행사", "여행예약", "출장예약", "관광서비스")),
    IndustryConcept("호텔", "숙박", ("호텔", "숙박업", "비즈니스호텔", "리조트", "출장숙박", "객실")),
    IndustryConcept("병원", "의료", ("병원", "종합병원", "의료기관", "의원", "내과", "정형외과", "안과")),
    IndustryConcept("치과", "의료", ("치과", "치과의원", "치과진료")),
    IndustryConcept("약국", "의료", ("약국", "의약품소매", "의약품", "약품")),
    IndustryConcept("식자재 도매", "도소매", ("식자재 도매업", "식자재 유통업", "급식재료 도매업", "급식재료 유통", "식품 도매업", "식품 유통업", "식자재 납품업")),
    IndustryConcept("대형마트", "도소매", ("대형마트", "종합소매", "창고형 할인점", "할인점", "마트")),
    IndustryConcept("슈퍼마켓", "도소매", ("슈퍼마켓", "식료품소매", "식료품점", "동네마트")),
    IndustryConcept("편의점", "도소매", ("편의점", "24시편의점", "종합소매점")),
    IndustryConcept("전자상거래", "도소매", ("전자상거래 소매업", "온라인쇼핑", "온라인몰", "통신판매")),
    IndustryConcept("전기공사", "공사/설비", ("전기공사업", "전기공사", "전기설비공사", "전기설비공사업", "전기설비", "배선공사")),
    IndustryConcept("건설공사", "공사/설비", ("건설업", "건축공사", "시설공사", "인테리어공사", "실내건축")),
    IndustryConcept("소프트웨어", "IT서비스", ("소프트웨어", "소프트웨어 개발", "소프트웨어 개발업", "업무 소프트웨어", "전산 개발", "시스템 개발")),
    IndustryConcept("클라우드", "IT서비스", ("클라우드 서비스", "호스팅 서비스", "IT서비스", "시스템통합", "전산서비스")),
    IndustryConcept("보안 소프트웨어", "IT서비스", ("보안 소프트웨어", "정보보안", "보안서비스")),
    IndustryConcept("문구/사무용품", "사무/전문서비스", ("문구점", "사무용품", "사무용품 소매", "문구", "오피스용품")),
    IndustryConcept("서점", "사무/전문서비스", ("서점", "도서소매", "온라인 서점", "도서")),
    IndustryConcept("출력/복사", "사무/전문서비스", ("복사 출력 서비스", "사무서비스", "인쇄", "출력")),
    IndustryConcept("주유소", "차량/연료", ("주유소", "셀프주유소", "유류판매", "주유", "연료판매")),
    IndustryConcept("충전소", "차량/연료", ("LPG충전소", "전기차 충전소", "충전서비스", "충전소")),
)


def normalize_industry_key(value: str) -> str:
    return re.sub(r"\s+", "", str(value or "").strip()).lower()


def _alias_index(concepts: Iterable[IndustryConcept]) -> Dict[str, IndustryConcept]:
    index: Dict[str, IndustryConcept] = {}
    for concept in concepts:
        index[normalize_industry_key(concept.canonical)] = concept
        for alias in concept.aliases:
            index[normalize_industry_key(alias)] = concept
    return index


INDUSTRY_ALIAS_INDEX = _alias_index(INDUSTRY_CONCEPTS)


def industry_concept(raw_industry: str) -> IndustryConcept | None:
    return INDUSTRY_ALIAS_INDEX.get(normalize_industry_key(raw_industry))


def canonicalize_industry(raw_industry: str) -> str:
    concept = industry_concept(raw_industry)
    return concept.canonical if concept else str(raw_industry or "").strip()


def parent_industry(raw_industry: str) -> str:
    concept = industry_concept(raw_industry)
    return concept.parent if concept else ""


def build_semantic_text(
    *,
    entity_name: str,
    raw_industry: str,
    context: str = "",
    entity_label: str = "가맹점명",
) -> str:
    canonical = canonicalize_industry(raw_industry)
    parent = parent_industry(raw_industry)
    segments: List[str] = [
        f"{entity_label}: {str(entity_name).strip()}",
        f"표준업종명: {canonical}",
    ]
    if parent:
        segments.append(f"상위업종명: {parent}")
    segments.append(f"원업종명: {str(raw_industry).strip()}")
    if str(context).strip():
        segments.append(f"문맥: {str(context).strip()}")
    return " | ".join(segments)
