# Korean Business Embedding Design

## Goal

Build a reusable pipeline that turns Korean merchant, supplier, company, industry, and item-name fields into training data for a BGE-M3 based similarity model.

## Problem Shape

The final transaction text is not available yet. The system therefore learns from structured public-data fields rather than memorizing full text combinations. It uses field-aware templates such as `공급업체명: 한국식품유통 | 업종명: 식자재 도매업 | 품목명: 급식재료`.

## Data Sources

- 소상공인시장진흥공단 상가(상권)정보 for merchant/store names and industry labels.
- 조달청 나라장터 user/company information for suppliers and supply items.
- 조달업체 면허 업종 등록 내역 for supplier-to-industry mapping.
- 한국표준산업분류 for industry label vocabulary and hierarchy.

## Acquisition Strategy

The project uses every legitimate public acquisition route:

- File datasets are ingested from downloaded CSV, ZIP, XLSX, or Parquet files.
- OpenAPI datasets are collected with paginated data.go.kr requests using `DATA_GO_KR_SERVICE_KEY`.
- Source metadata is listed in code so each data source has a documented page URL, license note, method, and expected fields.
- Public pages may be used to discover official download/API locations, but the pipeline does not bypass login, access controls, rate limits, or terms of use.

## Architecture

The pipeline has three layers. `bizembed.ingest` maps source-specific columns into one canonical schema. `bizembed.normalize` cleans entity strings and creates field-aware model text. `bizembed.pairs` generates weak-supervised positive and hard-negative similarity pairs.

Training uses BGE-M3 through SentenceTransformers. The first training stage uses positive pairs with `MultipleNegativesRankingLoss`. A later calibration stage can use scored pairs with `CoSENTLoss` or `CosineSimilarityLoss` once production feedback labels are available.

## Canonical Schema

```text
entity_name
entity_type
industry_name
industry_code
item_name
source
text
```

## Pair Labels

- `0.85`: different entities in the same detailed industry bucket.
- `0.20`: same normalized entity name but different industry, used as a hard negative.
- `0.05`: clearly different industry buckets.

These labels are conservative defaults for bootstrapping. They should be recalibrated after a held-out review set is created.

## Limits

The pipeline cannot fetch public-data records until a 공공데이터포털 service key or downloaded source files are provided. It also cannot learn unknown business-process terms such as department, commute type, country, or weekday until those fields appear in actual logs.
