from bizembed.collect import (
    DataGoKrClient,
    build_request_params,
    detect_industry_codes,
    extract_items,
    should_continue,
)


class FakeResponse:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload


class FakeSession:
    def __init__(self):
        self.calls = []

    def get(self, url, params, timeout):
        self.calls.append((url, params, timeout))
        return FakeResponse(
            {
                "header": {"columns": ["상호명", "상권업종소분류명"]},
                "body": {
                    "items": [
                        {"상호명": "보나비", "상권업종소분류명": "일반한식"},
                    ],
                    "totalCount": 1,
                    "pageNo": 1,
                    "numOfRows": 100,
                },
            }
        )


def test_build_request_params_uses_standard_service_key_and_json_type():
    params = build_request_params(
        service_key="abc",
        page_no=2,
        num_rows=500,
        extra={"divId": "indsLclsCd", "key": "I2"},
    )

    assert params["serviceKey"] == "abc"
    assert params["pageNo"] == 2
    assert params["numOfRows"] == 500
    assert params["type"] == "json"
    assert params["divId"] == "indsLclsCd"


def test_extract_items_handles_data_go_kr_json_shape():
    payload = {"body": {"items": {"상호명": "보나비"}}}

    assert extract_items(payload) == [{"상호명": "보나비"}]


def test_client_fetch_page_calls_endpoint_with_params():
    session = FakeSession()
    client = DataGoKrClient(service_key="abc", session=session)

    items = client.fetch_page(
        "https://apis.data.go.kr/B553077/api/open/sdsc2/storeListInUpjong",
        page_no=1,
        num_rows=100,
        extra={"divId": "indsLclsCd", "key": "I2"},
    )

    assert items == [{"상호명": "보나비", "상권업종소분류명": "일반한식"}]
    assert session.calls[0][0].endswith("/storeListInUpjong")
    assert session.calls[0][1]["serviceKey"] == "abc"


def test_should_continue_uses_total_count():
    assert should_continue(page_no=1, num_rows=100, total_count=101) is True
    assert should_continue(page_no=2, num_rows=100, total_count=101) is False


def test_detect_industry_codes_accepts_korean_and_api_column_names():
    items = [
        {"상권업종소분류코드": "I20101", "상권업종소분류명": "일반한식"},
        {"indsSclsCd": "G20405", "indsSclsNm": "문구용품"},
        {"indsSclsCd": "I20101", "indsSclsNm": "중복"},
    ]

    assert detect_industry_codes(items) == ["G20405", "I20101"]
