from __future__ import annotations

from itertools import combinations
from typing import Dict, Iterable, List

import pandas as pd

from bizembed.industry_taxonomy import INDUSTRY_CONCEPTS, IndustryConcept, build_semantic_text
from bizembed.pairs import PAIR_COLUMNS


ENTITY_NAMES: Dict[str, tuple[str, ...]] = {
    "한식": ("보나비", "동해횟집", "본우리집밥", "청담순대국", "명동칼국수"),
    "음식점": ("맛나식당", "서울식당", "행복식당", "우리식당", "중앙식당"),
    "카페": ("스타벅스", "투썸플레이스", "이디야커피", "커피빈", "메가커피"),
    "제과점": ("파리바게뜨", "뚜레쥬르", "성심당", "베이커리팩토리", "브레댄코"),
    "중식": ("홍콩반점", "교동짬뽕", "북경반점", "차이나타운", "중화루"),
    "일식": ("스시로", "미소야", "독도참치", "스시마켓", "동해횟집"),
    "항공사": ("대한항공", "아시아나항공", "제주항공", "티웨이항공", "진에어"),
    "여행사": ("하나투어", "모두투어", "노랑풍선", "참좋은여행", "인터파크투어"),
    "호텔": ("신라스테이", "롯데호텔", "토요코인", "그랜드하얏트", "라마다호텔"),
    "병원": ("세브란스병원", "서울아산병원", "삼성서울병원", "강남연세의원", "바른정형외과"),
    "치과": ("서울치과", "연세치과", "미소치과", "화이트치과", "바른치과"),
    "약국": ("온누리약국", "메디팜약국", "건강약국", "우리약국", "중앙약국"),
    "식자재 도매": ("한국식품유통", "대한식자재유통", "푸드마켓코리아", "우리급식유통", "농협식품"),
    "대형마트": ("이마트", "롯데마트", "홈플러스", "코스트코", "하나로마트"),
    "슈퍼마켓": ("GS더프레시", "롯데슈퍼", "이마트에브리데이", "홈플러스익스프레스", "우리마트"),
    "편의점": ("CU", "GS25", "세븐일레븐", "이마트24", "미니스톱"),
    "전자상거래": ("쿠팡", "네이버쇼핑", "11번가", "G마켓", "옥션"),
    "전기공사": ("대성전기", "한빛전기", "태양전설", "신우이엔씨", "광명전기"),
    "건설공사": ("대우건설", "현대건설", "한샘리하우스", "우리인테리어", "중앙건설"),
    "소프트웨어": ("더존비즈온", "한글과컴퓨터", "비즈플레이", "카카오엔터프라이즈", "안랩"),
    "클라우드": ("네이버클라우드", "메가존클라우드", "가비아", "NHN클라우드", "아이티센"),
    "보안 소프트웨어": ("안랩", "이글루코퍼레이션", "SK쉴더스", "시큐아이", "라온시큐어"),
    "문구/사무용품": ("오피스디포", "알파문구", "모닝글로리", "드림디포", "오피스넥스"),
    "서점": ("교보문고", "영풍문고", "예스24", "알라딘", "인터파크도서"),
    "출력/복사": ("프린트뱅크", "킨코스", "카피프린트", "문서마당", "프린트샵"),
    "주유소": ("SK주유소", "GS칼텍스", "현대오일뱅크", "S-OIL", "알뜰주유소"),
    "충전소": ("서울충전소", "그린충전소", "EV충전소", "하이차저", "LPG충전소"),
}


DEFAULT_CONTEXTS = (
    "내부회계관리팀 천호랄 출근 국내 평일",
    "재무팀 월말 결산 검토 회의 후 법인카드 처리",
    "경영지원팀 정기 감사 준비 관련 지출 내역",
    "영업본부 고객사 방문 일정에 따른 국내 평일 사용",
    "인사팀 교육 운영 준비 과정에서 발생한 비용",
    "구매팀 정기 계약 검토 후 비용 정산",
    "품질관리팀 현장 점검 일정 중 법인카드 결제",
    "기획팀 워크숍 준비 및 참석자 지원 비용",
    "정보보안팀 내부통제 점검 관련 지출",
    "총무팀 사무실 운영 목적의 평일 사용 내역",
)


def _pair(text_a: str, text_b: str, label: float, relation: str) -> Dict[str, object]:
    return {"text_a": text_a, "text_b": text_b, "label": label, "relation": relation}


def _entity_names(concept: IndustryConcept, count: int) -> tuple[str, ...]:
    names = ENTITY_NAMES.get(concept.canonical, ())
    if names:
        return names[:count]
    return tuple(f"{concept.canonical}{idx + 1}" for idx in range(count))


def build_industry_semantic_pairs(
    *,
    contexts: Iterable[str] = DEFAULT_CONTEXTS,
    entities_per_concept: int = 5,
) -> pd.DataFrame:
    context_list = tuple(contexts)
    pairs: List[Dict[str, object]] = []

    concept_rows = []
    for concept in INDUSTRY_CONCEPTS:
        aliases = (concept.canonical,) + concept.aliases
        names = _entity_names(concept, entities_per_concept)
        for name_index, entity_name in enumerate(names):
            concept_rows.append((concept, entity_name, context_list[name_index % len(context_list)], aliases))

            for context in context_list:
                for left_alias, right_alias in combinations(aliases[: min(len(aliases), 5)], 2):
                    pairs.append(
                        _pair(
                            build_semantic_text(entity_name=entity_name, raw_industry=left_alias, context=context),
                            build_semantic_text(entity_name=entity_name, raw_industry=right_alias, context=context),
                            0.95,
                            "same_entity_same_canonical_industry",
                        )
                    )

        for context in context_list:
            for left_name, right_name in combinations(names[: min(len(names), 4)], 2):
                pairs.append(
                    _pair(
                        build_semantic_text(entity_name=left_name, raw_industry=aliases[0], context=context),
                        build_semantic_text(entity_name=right_name, raw_industry=aliases[1], context=context),
                        0.88,
                        "different_entity_same_canonical_industry",
                    )
                )

    for left, right in combinations(concept_rows, 2):
        left_concept, left_name, left_context, left_aliases = left
        right_concept, right_name, _, right_aliases = right
        if left_context != context_list[0]:
            continue

        if left_concept.parent == right_concept.parent and left_concept.canonical != right_concept.canonical:
            label = 0.62
            relation = "same_parent_different_canonical_industry"
        elif left_concept.parent != right_concept.parent:
            label = 0.08
            relation = "different_parent_industry_same_context"
        else:
            continue

        pairs.append(
            _pair(
                build_semantic_text(entity_name=left_name, raw_industry=left_aliases[0], context=left_context),
                build_semantic_text(entity_name=right_name, raw_industry=right_aliases[0], context=left_context),
                label,
                relation,
            )
        )

    return pd.DataFrame(pairs, columns=PAIR_COLUMNS).drop_duplicates().reset_index(drop=True)
