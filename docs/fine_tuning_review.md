# Fine-Tuning Review

## Current Recommendation

Keep `BAAI/bge-m3` as the main production base model for now.

Reasons:

- It supports multilingual dense embeddings and long text up to 8192 tokens, which covers short card text and longer accounting memos.
- The current project already has KSIC, compact card, and hard-negative tuning checkpoints on top of BGE-M3.
- The evaluation target is Korean merchant/industry semantic similarity, not general English retrieval, so continuity with the tuned model matters more than switching bases immediately.

## Next Fine-Tuning Tracks

1. Company-sample threshold calibration
   - Collect 500-1000 real card rows or 200-500 labeled A/B pairs.
   - Run `scripts/evaluate_company_samples.py`.
   - Pick a working threshold for automatic match, review zone, and reject.

2. Hard-negative mining
   - Pull high-scoring pairs where 업종명 differs strongly.
   - Add those as low-label pairs.
   - Re-run short cosine fine-tuning.

3. Field dropout training
   - Train variants with missing merchant name, missing industry, or noisy memo tails.
   - This helps when card data is incomplete or inconsistent.

4. Alternative base-model bakeoff
   - Fine-tune the same compact training pairs on one alternate base model.
   - Compare Korean card-like evaluation, not generic MTEB alone.

## Candidate Base Models

| Model | Why Consider | Risk |
| --- | --- | --- |
| `BAAI/bge-m3` | Current model; multilingual, long-context, dense/sparse/multi-vector capable | Large model, slower local tuning |
| `intfloat/multilingual-e5-large-instruct` | Strong multilingual embedding baseline | Requires instruction-style formatting for best behavior |
| `jinaai/jina-embeddings-v3` | Multilingual, long-context, task LoRA design | Different inference/trust-remote-code behavior may complicate deployment |
| Korean SBERT/KoSimCSE family | Lightweight Korean sentence similarity | Usually weaker for mixed merchant/code/memo retrieval and less current |

## Decision

Do not switch the production base yet. First run real company data calibration on the current compact v2 model. If failure cases remain systematic after two hard-negative loops, run a controlled bakeoff with `multilingual-e5-large-instruct` and `jina-embeddings-v3`.
