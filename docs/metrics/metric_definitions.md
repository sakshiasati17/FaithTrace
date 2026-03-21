# DriftLens — Metric Definitions

## Standard RAG Metrics (via Ragas)

### Answer Correctness
Semantic similarity between the generated answer and the ground-truth answer.
Computed using embedding cosine similarity or an LLM judge.
Range: 0–1. Higher is better.

### Faithfulness
Fraction of factual claims in the generated answer that are supported by the retrieved context.
Computed by decomposing the answer into atomic claims and verifying each against retrieved chunks.
Range: 0–1. Higher is better. Low faithfulness indicates hallucination.

### Context Precision
Fraction of retrieved chunks that were actually relevant to the question.
Measures retrieval precision: were the retrieved documents on-topic?
Range: 0–1. Higher is better.

### Context Recall
Fraction of relevant information (present in ground truth) that was captured in the retrieved context.
Measures retrieval recall: did we surface everything we needed?
Range: 0–1. Higher is better.

### Answer Relevance
How directly and completely the generated answer addresses the question.
Computed via LLM judge or question regeneration approach.
Range: 0–1. Higher is better.

---

## Operational Metrics

### Latency (p50 / p95)
Wall-clock time from query submission to final answer, measured in milliseconds.
p50 = median latency. p95 = 95th-percentile latency.
Lower is better. Relevant for SLA compliance.

### Average Token Usage
Mean total tokens (input + output) consumed per query.
Directly proportional to cost. Lower is better under cost constraints.

### Cost per Query (USD)
Estimated cost based on model pricing and token usage.
Useful for cost-vs-quality tradeoff analysis.

### Reranker Overhead
Latency delta introduced by the reranking step.
Computed as mean(latency with reranker) − mean(latency without reranker) for matched configs.

---

## DriftLens Custom Metrics

### Freshness Validity
**Definition**: The fraction of answers that drew from the document version valid at query time.

**Computation**:
1. For each query, determine the `query_date` (from eval set `valid_from`/`valid_to` fields).
2. Identify which document version should be used at that date.
3. Check whether the retrieved chunks come from that version.
4. Score = (correctly versioned queries) / (total queries with a temporal constraint).

**Interpretation**: Low freshness validity means the pipeline retrieved outdated knowledge — the most common form of temporal drift failure.

---

### Temporal Citation Accuracy
**Definition**: The fraction of cited documents that were not only relevant but also time-correct.

**Computation**:
1. For each cited chunk, check whether its `effective_from`–`effective_to` range includes the `query_date`.
2. Score = (time-correct citations) / (total citations).

**Interpretation**: Distinguishes between retrieving a document that is topically relevant but expired versus one that is both relevant and temporally valid.

---

### Multimodal Grounding Rate
**Definition**: For questions labeled with modality `table`, `chart`, `spreadsheet`, or `mixed`, the fraction of answers that retrieved and used the correct non-plain-text evidence.

**Computation**:
1. Filter eval set to non-text modality questions.
2. For each, check whether the retrieved chunks include at least one chunk of the correct non-text type (table, image, or spreadsheet cell) that overlaps with the ground-truth source.
3. Score = (correctly grounded multimodal answers) / (total multimodal questions).

**Interpretation**: Low multimodal grounding rate indicates that the pipeline is using text-only fallback for evidence that lives in structured or visual form.

---

### Root-Cause Diagnostic Accuracy
**Definition**: The accuracy of the diagnostics engine in classifying the primary failure category, evaluated against manually labeled ground-truth failure annotations in the eval set.

**Computation**:
`accuracy = (correctly classified failure queries) / (total failure queries with ground-truth labels)`

**Interpretation**: Measures how reliable the diagnostic module is as a signal for pipeline debugging. Only meaningful for eval sets that include `failure_type` labels.
