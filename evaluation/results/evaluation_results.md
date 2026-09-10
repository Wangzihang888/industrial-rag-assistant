# Industrial RAG Assistant - Evaluation Results

## 1. Evaluation Overview

This project uses an offline evaluation dataset to evaluate the retrieval
pipeline and evidence-based answerability decision mechanism.

The current evaluation contains:

- 17 retrieval test cases
- 13 answerability test cases

The evaluation dataset is manually constructed based on the current
industrial equipment maintenance knowledge base.

---

## 2. Retrieval Evaluation

The retrieval evaluation compares four retrieval strategies:

1. Dense Retrieval
2. BM25 Retrieval
3. Hybrid Retrieval (Dense + BM25 + RRF)
4. Hybrid Retrieval + Reranker

The main evaluation metrics are:

- Top1 Accuracy
- Recall@3

### Results

| Method | Top1 Accuracy | Recall@3 |
|---|---:|---:|
| Dense Retrieval | 100.00% | 100.00% |
| BM25 Retrieval | 94.12% | 100.00% |
| Hybrid Retrieval | 100.00% | 100.00% |
| Hybrid + Reranker | 100.00% | 100.00% |

---

## 3. Retrieval Failure Analysis

BM25 produced one Top1 ranking failure in the current 17-case
retrieval evaluation dataset.

### Failure Case

Query:

`控制柜太热应该排查什么？`

Query Type:

`semantic`

Expected Chunk:

`cabinet_cooling`

BM25 Top3:

1. `cabinet_power`
2. `cabinet_cooling`
3. `servo_e103`

The expected chunk was retrieved within the Top3 results but was not
ranked at Top1.

### Analysis

The query uses the expression:

`控制柜太热`

while the target knowledge chunk uses:

`控制柜温度过高`

These expressions are semantically similar but have limited exact
lexical overlap.

BM25 mainly relies on lexical term matching and term statistics, so it
ranked `cabinet_power` above the expected `cabinet_cooling` chunk.

Dense Retrieval correctly ranked `cabinet_cooling` at Top1 because the
embedding model can better capture the semantic similarity between
"太热" and "温度过高".

This case demonstrates the complementary characteristics of lexical
retrieval and semantic retrieval.

---

## 4. Hybrid Retrieval Design

The project combines Dense Retrieval and BM25 Retrieval.

Dense Retrieval is used to capture semantic similarity and natural
language paraphrases.

BM25 Retrieval provides lexical matching capability for exact terms,
fault codes, identifiers, and domain-specific expressions such as:

- `E-103`
- `AXIS-4`
- `PLC-ALARM-003`

The ranking results from Dense Retrieval and BM25 are fused using
Reciprocal Rank Fusion (RRF).

RRF operates on ranking positions instead of directly adding BM25
scores and cosine similarity scores, avoiding direct comparison between
scores with different scales.

---

## 5. Answerability Evaluation

The project uses an Evidence Gate before answer generation.

The Evidence Gate decides whether the retrieved evidence is sufficient
for answering the user query.

Current threshold:

`EVIDENCE_THRESHOLD = 0.8`

The evaluation dataset contains both answerable and unanswerable
queries.

The unanswerable cases are intentionally designed to be related to the
knowledge-base domain while requiring information that is not actually
present in the knowledge base.

Examples include:

- lubricant brand recommendation
- maintenance cost
- alarm occurrence frequency
- repair duration
- cooling fan brand

### Results

| Metric | Result |
|---|---:|
| Total Cases | 13 |
| Correct | 13 |
| Accuracy | 100.00% |
| False Accept | 0 |
| False Reject | 0 |

---

## 6. Evidence Score Observation

In the current evaluation set, answerable queries received high Top1
reranker relevance scores.

Observed answerable examples include scores approximately between:

`0.9851 - 1.0000`

The tested unanswerable queries received lower scores, approximately
between:

`0.2409 - 0.5577`

With the current threshold:

`0.8`

all 13 answerability test cases were classified correctly.

---

## 7. Current Conclusions

The current evaluation indicates that:

1. Dense Retrieval performs well on semantic paraphrase queries.
2. BM25 provides strong lexical retrieval capability but produced one
   Top1 ranking error on a semantic paraphrase query.
3. Hybrid Retrieval preserved 100% Top1 accuracy on the current
   retrieval evaluation set.
4. Hybrid + Reranker also achieved 100% Top1 accuracy and Recall@3 on
   the current evaluation set.
5. The Evidence Gate correctly separated the tested answerable and
   unanswerable queries using the current threshold.

---

## 8. Limitations

The current evaluation is a small, manually constructed offline
evaluation set.

The knowledge base currently contains only 20 chunks, and the retrieval
evaluation contains 17 queries.

Therefore, the current 100% results should not be interpreted as
production-level performance or generalization guarantees.

The Evidence Gate threshold of `0.8` is calibrated only against the
current small evaluation dataset and should be re-evaluated when the
knowledge base and evaluation dataset are expanded.

Future evaluation can include:

- larger evaluation datasets
- more difficult paraphrase queries
- adversarial or ambiguous queries
- multiple relevant chunks
- MRR
- nDCG
- end-to-end answer quality evaluation