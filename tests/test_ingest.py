import pandas as pd
import zipfile

from bizembed.ingest import read_table, source_usecols, standardize_dataframe


def test_standardize_sbiz_store_columns():
    raw = pd.DataFrame(
        [
            {
                "상호명": "보나비",
                "상권업종소분류명": "일반한식",
                "상권업종소분류코드": "I20101",
            }
        ]
    )

    result = standardize_dataframe(raw, source="sbiz")

    assert result.loc[0, "entity_name"] == "보나비"
    assert result.loc[0, "entity_type"] == "merchant"
    assert result.loc[0, "industry_name"] == "일반한식"
    assert result.loc[0, "industry_code"] == "I20101"
    assert result.loc[0, "source"] == "sbiz"
    assert result.loc[0, "text"] == "가맹점명: 보나비 | 업종명: 일반한식"


def test_standardize_supplier_columns_with_item_name():
    raw = pd.DataFrame(
        [
            {
                "업체명": "한국식품유통",
                "업종명": "식자재 도매업",
                "업종코드": "G463",
                "공급물품명": "급식재료",
            }
        ]
    )

    result = standardize_dataframe(raw, source="nara")

    assert result.loc[0, "entity_name"] == "한국식품유통"
    assert result.loc[0, "entity_type"] == "supplier"
    assert result.loc[0, "industry_name"] == "식자재 도매업"
    assert result.loc[0, "item_name"] == "급식재료"
    assert result.loc[0, "text"] == "공급업체명: 한국식품유통 | 업종명: 식자재 도매업 | 품목명: 급식재료"


def test_standardize_card_merchant_columns():
    raw = pd.DataFrame(
        [
            {
                "카드가맹점명": "보나비",
                "가맹점업종명": "일반한식",
                "MCC코드": "5812",
            }
        ]
    )

    result = standardize_dataframe(raw, source="card")

    assert result.loc[0, "entity_name"] == "보나비"
    assert result.loc[0, "entity_type"] == "merchant"
    assert result.loc[0, "industry_name"] == "일반한식"
    assert result.loc[0, "industry_code"] == "5812"
    assert result.loc[0, "text"] == "가맹점명: 보나비 | 업종명: 일반한식"


def test_standardize_corporate_card_supplier_columns():
    raw = pd.DataFrame(
        [
            {
                "공급업체명": "한국식품유통",
                "카드업종명": "식자재 도매업",
                "품목명": "급식재료",
            }
        ]
    )

    result = standardize_dataframe(raw, source="corporate_card")

    assert result.loc[0, "entity_name"] == "한국식품유통"
    assert result.loc[0, "entity_type"] == "supplier"
    assert result.loc[0, "industry_name"] == "식자재 도매업"
    assert result.loc[0, "item_name"] == "급식재료"
    assert result.loc[0, "text"] == "공급업체명: 한국식품유통 | 업종명: 식자재 도매업 | 품목명: 급식재료"


def test_standardize_public_card_usage_columns():
    raw = pd.DataFrame(
        [
            {
                "가맹점": "두부사랑",
                "사용내역": "업무회의 식사",
                "사용방법": "카드",
            }
        ]
    )

    result = standardize_dataframe(raw, source="public_card")

    assert result.loc[0, "entity_name"] == "두부사랑"
    assert result.loc[0, "entity_type"] == "merchant"
    assert result.loc[0, "item_name"] == "업무회의 식사"
    assert result.loc[0, "text"] == "가맹점명: 두부사랑 | 품목명: 업무회의 식사"


def test_standardize_localdata_license_columns():
    raw = pd.DataFrame(
        [
            {
                "사업장명": "서울정형외과",
                "업태구분명": "의원",
                "인허가업종": "의료기관",
            }
        ]
    )

    result = standardize_dataframe(raw, source="localdata")

    assert result.loc[0, "entity_name"] == "서울정형외과"
    assert result.loc[0, "entity_type"] == "merchant"
    assert result.loc[0, "industry_name"] == "의원"
    assert result.loc[0, "text"] == "가맹점명: 서울정형외과 | 업종명: 의원"


def test_standardize_franchise_brand_columns():
    raw = pd.DataFrame(
        [
            {
                "영업표지": "메가MGC커피",
                "업종": "커피",
                "가맹본부": "앤하우스",
            }
        ]
    )

    result = standardize_dataframe(raw, source="franchise")

    assert result.loc[0, "entity_name"] == "메가MGC커피"
    assert result.loc[0, "entity_type"] == "merchant"
    assert result.loc[0, "industry_name"] == "커피"


def test_standardize_dart_company_columns():
    raw = pd.DataFrame(
        [
            {
                "corp_name": "대한항공",
                "induty_code": "51100",
                "induty_name": "항공 여객 운송업",
            }
        ]
    )

    result = standardize_dataframe(raw, source="dart")

    assert result.loc[0, "entity_name"] == "대한항공"
    assert result.loc[0, "entity_type"] == "company"
    assert result.loc[0, "industry_name"] == "항공 여객 운송업"
    assert result.loc[0, "industry_code"] == "51100"


def test_standardize_mcc_columns():
    raw = pd.DataFrame(
        [
            {
                "MCC": "4511",
                "MCC업종명": "항공사",
                "MCC설명": "Airlines, Air Carriers",
            }
        ]
    )

    result = standardize_dataframe(raw, source="mcc")

    assert result.loc[0, "entity_type"] == "merchant"
    assert result.loc[0, "industry_name"] == "항공사"
    assert result.loc[0, "industry_code"] == "4511"
    assert result.loc[0, "item_name"] == "Airlines Air Carriers"
    assert result.loc[0, "text"] == "업종명: 항공사 | 품목명: Airlines Air Carriers"


def test_standardize_nara_supplier_item_export_columns():
    raw = pd.DataFrame(
        [
            {
                "업체명": "경원농자재",
                "물품분류명": "온실설치공사",
                "세부품명": "온실설치공사",
            }
        ]
    )

    result = standardize_dataframe(raw, source="nara")

    assert result.loc[0, "entity_name"] == "경원농자재"
    assert result.loc[0, "item_name"] == "온실설치공사"
    assert result.loc[0, "text"] == "공급업체명: 경원농자재 | 품목명: 온실설치공사"


def test_standardize_nara_registration_export_columns():
    raw = pd.DataFrame(
        [
            {
                "업체명": "한국식품유통",
                "대표업종": "식자재 도매업",
                "대표세부품명": "급식재료",
            }
        ]
    )

    result = standardize_dataframe(raw, source="nara")

    assert result.loc[0, "industry_name"] == "식자재 도매업"
    assert result.loc[0, "item_name"] == "급식재료"


def test_read_table_concatenates_zip_csv_files(tmp_path):
    zip_path = tmp_path / "stores.zip"
    with zipfile.ZipFile(zip_path, "w") as archive:
        archive.writestr("a.csv", "상호명,상권업종소분류명\n보나비,일반한식\n")
        archive.writestr("b.csv", "상호명,상권업종소분류명\n다빈치모텔,여관/모텔\n")
        archive.writestr("readme.txt", "ignore")

    result = read_table(zip_path)

    assert result["상호명"].tolist() == ["보나비", "다빈치모텔"]


def test_source_usecols_limits_sbiz_to_training_fields():
    assert "상호명" in source_usecols("sbiz")
    assert "도로명주소" not in source_usecols("sbiz")


def test_source_usecols_includes_card_training_fields():
    assert "카드가맹점명" in source_usecols("card")
    assert "MCC코드" in source_usecols("card")
    assert "도로명주소" not in source_usecols("card")


def test_read_table_detects_mstr_excel_header_row(tmp_path):
    path = tmp_path / "mstr.xlsx"
    raw = pd.DataFrame(
        [
            ["검색조건", None, None],
            [None, None, None],
            ["업체명", "물품분류명", "세부품명"],
            ["경원농자재", "온실설치공사", "온실설치공사"],
        ]
    )
    raw.to_excel(path, index=False, header=False)

    result = read_table(path, usecols=source_usecols("nara"))

    assert result.columns.tolist() == ["업체명", "물품분류명", "세부품명"]
    assert result.loc[0, "업체명"] == "경원농자재"
