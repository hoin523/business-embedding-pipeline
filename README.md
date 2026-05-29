# Korean Business Embedding

한국어 `업종명`, `공급업체명`, `업체명`, `가맹점명`, `품목명`을 통합 임베딩 공간에서 비교하기 위한 데이터 수집, 정규화, pair 생성, BGE-M3 파인튜닝 프로젝트입니다.

## 목표

전체 거래/문장 텍스트가 없어도 구조화 필드만으로 통용 가능한 유사도 모델을 만듭니다.

```text
가맹점명: 보나비 | 업종명: 일반한식
공급업체명: 한국식품유통 | 업종명: 식자재 도매업 | 품목명: 급식재료
업체명: 대성 | 업종명: 전기공사업 | 품목명: 전기공사
```

모델은 위 텍스트들을 임베딩하고, cosine similarity 또는 벡터 검색으로 유사한 업체/업종/공급 역할을 찾습니다.

## 데이터 원천

권장 원천은 공식 공공데이터입니다.

- 소상공인시장진흥공단 상가(상권)정보: 가맹점명/상호명, 업종코드, 업종명
- 조달청 나라장터 사용자정보/업체정보: 공급업체명, 등록업종, 공급물품
- 조달업체 면허 업종 등록 내역: 업체명, 사업자번호, 업종코드, 업종명
- 한국표준산업분류: 업종 라벨 사전

대량 수집에는 공공데이터포털 인증키 또는 원문 CSV/ZIP 파일이 필요합니다.

## 빠른 실행

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
pytest -q
```

## 데이터 수집

현재 프로젝트는 비공개 접근이나 인증 우회 없이, 공개 원천을 최대한 자동화해서 가져옵니다.

수집 가능한 원천 목록을 확인합니다.

```bash
python scripts/list_sources.py
```

공공데이터포털 인증키를 환경변수로 넣습니다.

```bash
export DATA_GO_KR_SERVICE_KEY='공공데이터포털에서 발급받은 일반 인증키'
```

소상공인 상가정보 API를 업종코드 기준으로 수집합니다.

```bash
python scripts/download_sbiz_by_industry.py \
  --industry-code I20101 \
  --out data/raw/sbiz_I20101.csv
```

소상공인 상가정보 전체 수집은 소분류 업종코드 목록을 먼저 받은 뒤 전체 코드를 순회합니다.

```bash
python scripts/download_sbiz_all.py \
  --out data/raw/sbiz_all.csv
```

공공데이터포털의 임의 OpenAPI endpoint는 범용 수집기로 받을 수 있습니다.

```bash
python scripts/download_openapi.py \
  --endpoint 'https://apis.data.go.kr/B553077/api/open/sdsc2/storeListInUpjong' \
  --param divId=indsSclsCd \
  --param key=I20101 \
  --out data/raw/sbiz_I20101.csv
```

파일데이터는 포털 로그인 또는 원문파일 버튼이 필요한 경우가 있어, 원문 CSV/ZIP을 `data/raw/`에 내려받은 뒤 변환합니다.

로컬 CSV를 표준 스키마로 변환합니다.

```bash
python scripts/prepare_dataset.py \
  --source sbiz \
  --input data/raw/sbiz.csv \
  --records-out data/processed/records.parquet \
  --pairs-out data/processed/pairs.parquet
```

BGE-M3를 파인튜닝합니다.

```bash
python scripts/train_bge_m3.py \
  --pairs data/processed/pairs.parquet \
  --output models/bge-m3-business-lora
```

## 학습 전략

1. 구조화 필드를 `필드명: 값` 템플릿으로 변환합니다.
2. 같은 세부 업종/품목 조합은 positive pair로 만듭니다.
3. 이름은 같거나 비슷하지만 업종이 다른 조합은 hard negative로 만듭니다.
4. BGE-M3 기반 SentenceTransformer를 `MultipleNegativesRankingLoss`로 1차 학습합니다.
5. 점수형 라벨이 충분히 쌓이면 `CoSENTLoss` 또는 `CosineSimilarityLoss`로 보정합니다.

## Hugging Face

현재 연결된 Hugging Face 계정은 `hoin1218`입니다. `HF_TOKEN`이 있는 환경에서 아래 스크립트로 데이터셋이나 모델을 업로드할 수 있습니다.

```bash
python scripts/upload_to_hf.py \
  --repo-id hoin1218/korean-business-embedding-data \
  --repo-type dataset \
  --path data/processed
```
