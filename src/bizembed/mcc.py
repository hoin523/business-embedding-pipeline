from __future__ import annotations

from itertools import combinations
from typing import Dict, Iterable, List, Tuple

import pandas as pd

from bizembed.ingest import standardize_dataframe
from bizembed.pairs import PAIR_COLUMNS


MCC_KOREAN_ALIASES: Dict[str, Tuple[str, ...]] = {
    "4511": ("항공사", "항공권", "국내선 항공권", "해외 항공권"),
    "4582": ("공항", "공항 서비스"),
    "4722": ("여행사", "여행 예약", "출장 예약"),
    "4789": ("교통 서비스", "운송 서비스"),
    "4812": ("통신장비", "전화 서비스"),
    "4814": ("통신요금", "인터넷 통신 서비스"),
    "4829": ("송금 서비스", "자금 이체"),
    "5044": ("사무용품", "복사기 사무기기"),
    "5045": ("컴퓨터 장비", "전산 장비"),
    "5111": ("문구 도매", "사무용품 도매"),
    "5192": ("서적 도매", "도서 유통"),
    "5200": ("건축자재", "철물점"),
    "5411": ("슈퍼마켓", "식료품점", "마트"),
    "5499": ("식품점", "기타 식료품"),
    "5511": ("자동차 판매", "차량 구매"),
    "5533": ("자동차 부품", "차량 부품"),
    "5541": ("주유소", "유류비", "주유"),
    "5542": ("자동 주유소", "셀프 주유"),
    "5651": ("의류점", "복장 구매"),
    "5691": ("남녀의류점", "의류 소매"),
    "5712": ("가구점", "사무 가구"),
    "5732": ("전자제품", "가전제품"),
    "5734": ("소프트웨어", "컴퓨터 소프트웨어"),
    "5812": ("일반음식점", "한식", "식사", "식당"),
    "5813": ("주점", "음주점"),
    "5814": ("패스트푸드", "간편식"),
    "5912": ("약국", "의약품"),
    "5942": ("서점", "도서 구입"),
    "5943": ("문구점", "사무 문구"),
    "5977": ("화장품", "미용용품"),
    "5999": ("기타 소매", "잡화점"),
    "7011": ("호텔", "숙박업", "출장 숙박"),
    "7210": ("세탁소", "세탁 서비스"),
    "7230": ("미용실", "미용 서비스"),
    "7299": ("개인서비스", "기타 개인서비스"),
    "7311": ("광고 서비스", "마케팅 대행"),
    "7372": ("전산 개발", "시스템 개발", "소프트웨어 개발"),
    "7392": ("컨설팅", "경영 컨설팅"),
    "7399": ("비즈니스 서비스", "업무 대행"),
    "7512": ("렌터카", "자동차 대여"),
    "7523": ("주차장", "주차비"),
    "7538": ("자동차 정비", "차량 정비"),
    "7832": ("영화관", "영화 관람"),
    "7911": ("공연장", "문화 공연"),
    "7991": ("관광지", "놀이공원"),
    "7995": ("도박", "사행성 업종"),
    "8011": ("의원", "외래 진료"),
    "8021": ("치과", "치과 진료"),
    "8041": ("물리치료", "재활 치료"),
    "8062": ("병원", "입원 진료"),
    "8099": ("의료 서비스", "기타 의료"),
    "8211": ("초중고 교육", "학교"),
    "8220": ("대학교", "고등교육"),
    "8299": ("학원", "교육 서비스"),
    "8911": ("건축 설계", "엔지니어링 서비스"),
    "8931": ("회계 세무", "세무 서비스"),
    "8999": ("전문 서비스", "기타 전문서비스"),
    "9399": ("공공 서비스", "정부 서비스"),
}


def mcc_category(code: str) -> str:
    value = int(str(code).zfill(4))
    if value < 1500:
        return "농업 서비스"
    if value < 3000:
        return "공사 용역"
    if value < 4800:
        return "교통 여행"
    if value < 5000:
        return "통신 공공요금"
    if value < 5600:
        return "도소매 식료품"
    if value < 5700:
        return "의류 소매"
    if value < 7300:
        return "생활 소매 개인서비스"
    if value < 8000:
        return "비즈니스 서비스"
    if value < 9000:
        return "의료 교육 전문서비스"
    return "정부 공공서비스"


def _first(row: pd.Series, names: Iterable[str]) -> str:
    for name in names:
        value = row.get(name)
        if pd.notna(value) and str(value).strip():
            return str(value).strip()
    return ""


def _pair(text_a: str, text_b: str, label: float, relation: str) -> Dict[str, object]:
    return {"text_a": text_a, "text_b": text_b, "label": label, "relation": relation}


def build_mcc_training_tables(raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: List[Dict[str, str]] = []
    for _, row in raw.iterrows():
        code = _first(row, ("MCC", "MCC코드", "mcc"))
        if not code:
            continue
        code = code.zfill(4)
        base_name = _first(row, ("MCC업종명", "edited_description", "combined_description"))
        if not base_name:
            continue
        category = mcc_category(code)

        rows.append({"MCC": code, "MCC업종명": base_name, "MCC카테고리": category})
        for alias in MCC_KOREAN_ALIASES.get(code, ()):
            rows.append({"MCC": code, "MCC업종명": alias, "MCC카테고리": category})

    records = standardize_dataframe(pd.DataFrame(rows), source="mcc").drop_duplicates().reset_index(drop=True)

    pairs: List[Dict[str, object]] = []
    for _, group in records.groupby("industry_code"):
        unique = group.drop_duplicates("text").head(12)
        for left, right in combinations(unique["text"].tolist(), 2):
            pairs.append(_pair(left, right, 0.90, "same_mcc_alias"))

    category_heads = records.drop_duplicates(["industry_code"]).head(200).copy()
    category_heads["mcc_category"] = category_heads["industry_code"].map(mcc_category)
    for _, group in category_heads.groupby("mcc_category"):
        unique = group.drop_duplicates("text").head(8)
        for left, right in combinations(unique["text"].tolist(), 2):
            pairs.append(_pair(left, right, 0.70, "same_mcc_category"))

    sampled = records.drop_duplicates("industry_code").head(80)
    for left, right in combinations(sampled.to_dict("records"), 2):
        if left["industry_code"] == right["industry_code"]:
            continue
        if mcc_category(left["industry_code"]) == mcc_category(right["industry_code"]):
            continue
        pairs.append(_pair(left["text"], right["text"], 0.10, "different_mcc_category"))
        if len(pairs) > 2000:
            break

    return records, pd.DataFrame(pairs, columns=PAIR_COLUMNS).drop_duplicates().reset_index(drop=True)
