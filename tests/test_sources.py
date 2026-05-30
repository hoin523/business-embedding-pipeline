from bizembed.sources import DATA_SOURCES, get_source, sources_requiring_service_key


def test_catalog_contains_primary_business_sources():
    names = {source.name for source in DATA_SOURCES}

    assert "sbiz_store_api" in names
    assert "nara_user_api" in names
    assert "nara_supplier_license_file" in names
    assert "localdata_license_download" in names
    assert "fairtrade_franchise_api" in names
    assert "dart_company_overview_api" in names
    assert "mcc_codes_csv" in names
    assert "ksic_r_universe" in names
    assert "ksure_industry_code_file" in names


def test_source_describes_public_access_constraints():
    source = get_source("sbiz_store_api")

    assert source.requires_service_key is True
    assert source.license == "이용허락범위 제한 없음"
    assert "상호명" in source.fields
    assert source.method == "openapi"


def test_sources_requiring_service_key_are_listed():
    names = [source.name for source in sources_requiring_service_key()]

    assert "sbiz_store_api" in names
    assert "nara_user_api" in names
    assert "fairtrade_franchise_api" in names
    assert "dart_company_overview_api" in names
