from __future__ import annotations

from typing import Any, Dict, Iterable, List, Mapping, Optional
from urllib.parse import unquote

import requests


DEFAULT_TIMEOUT = 30


def normalize_service_key(service_key: str) -> str:
    """Accept the portal's encoded display key and hand requests a raw value."""
    return unquote(service_key.strip())


def build_request_params(
    *,
    service_key: str,
    page_no: int,
    num_rows: int,
    extra: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    params: Dict[str, Any] = {
        "serviceKey": service_key,
        "pageNo": page_no,
        "numOfRows": num_rows,
        "type": "json",
    }
    if extra:
        params.update(dict(extra))
    return params


def extract_items(payload: Mapping[str, Any]) -> List[Dict[str, Any]]:
    body = payload.get("body") or payload.get("response", {}).get("body") or {}
    items = body.get("items", [])
    if isinstance(items, dict) and "item" in items:
        items = items["item"]
    if isinstance(items, dict):
        return [dict(items)]
    return [dict(item) for item in items or []]


def extract_total_count(payload: Mapping[str, Any]) -> Optional[int]:
    body = payload.get("body") or payload.get("response", {}).get("body") or {}
    for key in ("totalCount", "totalCnt", "total_count"):
        if key in body:
            try:
                return int(body[key])
            except (TypeError, ValueError):
                return None
    return None


def should_continue(*, page_no: int, num_rows: int, total_count: Optional[int]) -> bool:
    if total_count is None:
        return False
    return page_no * num_rows < total_count


def detect_industry_codes(items: Iterable[Mapping[str, Any]]) -> List[str]:
    """Extract SBIZ small industry codes from Korean or API-style response columns."""
    preferred_keys = (
        "indsSclsCd",
        "상권업종소분류코드",
        "indsMclsCd",
        "상권업종중분류코드",
        "indsLclsCd",
        "상권업종대분류코드",
    )
    codes = set()
    for item in items:
        for key in preferred_keys:
            value = item.get(key)
            if value:
                codes.add(str(value).strip())
                break
    return sorted(code for code in codes if code)


class DataGoKrClient:
    def __init__(self, *, service_key: str, session: Optional[Any] = None, timeout: int = DEFAULT_TIMEOUT):
        self.service_key = normalize_service_key(service_key)
        self.session = session or requests.Session()
        self.timeout = timeout

    def fetch_page(
        self,
        endpoint: str,
        *,
        page_no: int,
        num_rows: int,
        extra: Optional[Mapping[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        params = build_request_params(
            service_key=self.service_key,
            page_no=page_no,
            num_rows=num_rows,
            extra=extra,
        )
        response = self.session.get(endpoint, params=params, timeout=self.timeout)
        response.raise_for_status()
        return extract_items(response.json())

    def iter_pages(
        self,
        endpoint: str,
        *,
        num_rows: int = 1000,
        extra: Optional[Mapping[str, Any]] = None,
        max_pages: Optional[int] = None,
    ) -> Iterable[List[Dict[str, Any]]]:
        page_no = 1
        while True:
            params = build_request_params(
                service_key=self.service_key,
                page_no=page_no,
                num_rows=num_rows,
                extra=extra,
            )
            response = self.session.get(endpoint, params=params, timeout=self.timeout)
            response.raise_for_status()
            payload = response.json()
            items = extract_items(payload)
            if not items:
                break
            yield items

            total_count = extract_total_count(payload)
            if max_pages is not None and page_no >= max_pages:
                break
            if not should_continue(page_no=page_no, num_rows=num_rows, total_count=total_count):
                break
            page_no += 1
