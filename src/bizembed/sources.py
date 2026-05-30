from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


@dataclass(frozen=True)
class DataSource:
    name: str
    title: str
    method: str
    page_url: str
    fields: Tuple[str, ...]
    source: str
    license: str
    requires_service_key: bool = False
    api_base_url: Optional[str] = None
    operation: Optional[str] = None
    notes: str = ""


DATA_SOURCES: Tuple[DataSource, ...] = (
    DataSource(
        name="sbiz_store_api",
        title="소상공인시장진흥공단_상가(상권)정보_API",
        method="openapi",
        page_url="https://www.data.go.kr/data/15012005/openapi.do?recommendDataYn=Y",
        api_base_url="https://apis.data.go.kr/B553077/api/open/sdsc2",
        operation="storeListInUpjong",
        fields=("상호명", "상권업종대분류명", "상권업종중분류명", "상권업종소분류명", "표준산업분류명"),
        source="sbiz",
        license="이용허락범위 제한 없음",
        requires_service_key=True,
        notes="업종코드별 storeListInUpjong 조회. 전체 수집은 업종분류 코드 목록을 순회한다.",
    ),
    DataSource(
        name="sbiz_store_file",
        title="소상공인시장진흥공단_상가(상권)정보_파일",
        method="file",
        page_url="https://www.data.go.kr/tcs/dss/selectFileDataDetailView.do?publicDataPk=15083033",
        fields=("상호명", "상권업종소분류코드", "상권업종소분류명", "시도명", "시군구명"),
        source="sbiz",
        license="이용허락범위 제한 없음",
        notes="전국 영업 중 상가업소 원문 CSV. 포털 로그인 또는 원문파일 다운로드가 필요할 수 있다.",
    ),
    DataSource(
        name="nara_user_api",
        title="조달청_나라장터 사용자정보 서비스",
        method="openapi",
        page_url="https://www.data.go.kr/data/15129466/openapi.do",
        fields=("사업자등록번호", "업체명", "업체주소", "등록업종정보", "공급물품정보"),
        source="nara",
        license="이용허락범위 제한 없음",
        requires_service_key=True,
        notes="Swagger/활용가이드에서 업체정보 상세 operation을 확인한 뒤 generic-openapi 모드로 수집한다.",
    ),
    DataSource(
        name="nara_supplier_license_file",
        title="조달청_조달업체 면허 업종 등록 내역",
        method="file",
        page_url="https://www.data.go.kr/data/15053476/fileData.do?recommendDataYn=Y",
        fields=("업체명", "업체사업자등록번호", "업종명", "업종코드", "대표업종여부"),
        source="nara",
        license="이용허락범위 제한 없음",
        notes="업체명-업종명 매핑의 우선 원천. 원문 CSV 다운로드 후 prepare_dataset.py로 변환한다.",
    ),
    DataSource(
        name="nara_supplier_registration_file",
        title="조달청_조달업체 등록 내역",
        method="file",
        page_url="https://www.data.go.kr/data/15053474/fileData.do",
        fields=("업체명", "업체사업자등록번호", "대표업종", "대표품명", "대표세부품명"),
        source="nara",
        license="이용허락범위 제한 없음",
        notes="업체명-대표업종-대표품명 매핑 원천. 조달데이터허브 보고서ID UI-ADOAAA-008R.",
    ),
    DataSource(
        name="nara_supplier_item_file",
        title="조달청_조달업체 물품 내역",
        method="file",
        page_url="https://www.data.go.kr/dataset/3070349/openapi.do",
        fields=("업체명", "사업자등록번호", "물품분류명", "세부품명", "대표물품여부", "제조공급구분"),
        source="nara",
        license="이용허락범위 제한 없음",
        notes="공급업체명-공급물품 매핑 원천. 조달데이터허브 보고서ID UI-ADOAAA-011R.",
    ),
    DataSource(
        name="ksic_classification",
        title="한국표준산업분류 제11차 업종 라벨",
        method="file",
        page_url="https://kssc.kostat.go.kr",
        fields=("대분류", "중분류", "소분류", "세분류", "세세분류"),
        source="ksic",
        license="공식 분류 자료",
        notes="통계청 기준 표준산업분류. scripts/download_ksic11.py는 R-universe KSIC 패키지의 제11차 계층 데이터를 자동 추출한다.",
    ),
    DataSource(
        name="ksic_r_universe",
        title="KSIC R 패키지 제11차 한국표준산업분류 데이터",
        method="file",
        page_url="https://urbanjj.r-universe.dev/KSIC",
        fields=("cd", "nm", "eng_nm", "digit", "ksic_C", "ksic1~5_cd", "ksic1~5_nm"),
        source="ksic",
        license="GPL-3 R package; derived from Korean Standard Industrial Classification",
        notes="9/10/11차 KSIC 코드와 계층 테이블을 포함한다. 자동 수집 가능한 실제 국내 업종 계층 원천으로 사용한다.",
    ),
    DataSource(
        name="ksure_industry_code_file",
        title="한국무역보험공사_업종코드",
        method="file",
        page_url="https://www.data.go.kr/data/15064297/fileData.do",
        fields=("업종코드", "업종한글명", "업종영문명", "대분류", "중분류", "소분류"),
        source="ksure",
        license="공공데이터포털 제공 조건 확인 필요",
        notes="공공데이터포털 설명 기준 제11차 KSIC 기반 2,003개 업종코드. KSIC 자동 데이터의 검증/보강 후보.",
    ),
    DataSource(
        name="localdata_license_download",
        title="LOCALDATA 지방행정인허가 데이터 다운로드",
        method="file",
        page_url="https://www.localdata.go.kr/",
        fields=("사업장명", "업태구분명", "인허가업종", "영업상태명", "도로명주소"),
        source="localdata",
        license="공공 인허가 데이터",
        notes="전체자료는 다운로드 방식, 변동분은 OpenAPI 방식으로 제공된다. 음식점/숙박/의료/미용/체육 등 카드 가맹점 업종 보강에 사용한다.",
    ),
    DataSource(
        name="fairtrade_franchise_api",
        title="공정거래위원회 가맹정보 정보공개서",
        method="openapi",
        page_url="https://www.data.go.kr/data/15125529/openapi.do?recommendDataYn=Y",
        fields=("영업표지", "브랜드명", "가맹본부", "업종", "업종명"),
        source="franchise",
        license="이용허락범위 제한 없음",
        requires_service_key=True,
        notes="프랜차이즈 브랜드명과 업종명을 가맹점명 유사도 보강에 사용한다. 공공데이터포털 인증키가 필요하다.",
    ),
    DataSource(
        name="dart_company_overview_api",
        title="DART 기업개황",
        method="openapi",
        page_url="https://dart.fss.or.kr/guide/main.jsp?menu=130",
        fields=("corp_name", "corp_code", "induty_code", "induty_name"),
        source="dart",
        license="전자공시 공개 데이터",
        requires_service_key=True,
        notes="법인명과 표준산업분류 기반 업종명을 공급업체/법인명 정규화 보강에 사용한다. OpenDART 인증키가 필요하다.",
    ),
    DataSource(
        name="mcc_codes_csv",
        title="Merchant Category Codes CSV",
        method="file",
        page_url="https://github.com/greggles/mcc-codes",
        fields=("MCC", "MCC업종명", "MCC카테고리", "MCC설명"),
        source="mcc",
        license="공개 MCC CSV",
        notes="카드 승인 데이터의 MCC 업종 축을 보강한다. scripts/download_mcc_codes.py로 키 없이 내려받을 수 있다.",
    ),
    DataSource(
        name="public_card_usage_files",
        title="공공기관 업무추진비/법인카드 사용내역 파일",
        method="file",
        page_url="https://www.data.go.kr/",
        fields=("가맹점", "가맹점명", "사용처명", "사용내역", "집행목적", "업종명"),
        source="public_card",
        license="기관별 공개 조건 확인 필요",
        notes="실제 법인카드형 메모 문장을 만들기 위한 보조 원천이다. 업종은 SBIZ/LOCALDATA와 후처리 매칭한다.",
    ),
)


SOURCE_BY_NAME: Dict[str, DataSource] = {source.name: source for source in DATA_SOURCES}


def get_source(name: str) -> DataSource:
    try:
        return SOURCE_BY_NAME[name]
    except KeyError as exc:
        known = ", ".join(sorted(SOURCE_BY_NAME))
        raise KeyError(f"Unknown source '{name}'. Known sources: {known}") from exc


def sources_requiring_service_key() -> List[DataSource]:
    return [source for source in DATA_SOURCES if source.requires_service_key]
