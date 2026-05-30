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
        title="한국표준산업분류 업종 라벨",
        method="file",
        page_url="https://kssc.kostat.go.kr",
        fields=("산업분류코드", "산업분류명", "설명"),
        source="ksic",
        license="공식 분류 자료",
        notes="업종 라벨 사전과 계층형 hard negative 생성에 사용한다.",
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
