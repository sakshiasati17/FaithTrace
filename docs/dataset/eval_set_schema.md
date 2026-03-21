# FaithTrace — Evaluation Set Schema

## Question Item Format

Each entry in an evaluation set is a JSON object with the following fields:

```json
{
  "id": "q_001",
  "question": "What is the maximum order exception threshold effective after July 2024?",
  "ground_truth": "The threshold is $15,000 per order as of the revised procurement policy effective July 1, 2024.",
  "source_docs": ["procurement_policy_v2.pdf"],
  "valid_from": "2024-07-01",
  "valid_to": "2024-12-31",
  "modality": "text",
  "difficulty": "medium",
  "answerable": true,
  "failure_type": null
}
```

## Field Definitions

| Field | Type | Description |
|---|---|---|
| `id` | string | Unique question identifier |
| `question` | string | Natural language query |
| `ground_truth` | string | Correct answer text |
| `source_docs` | string[] | Document filenames that contain the supporting evidence |
| `valid_from` | ISO date or null | Start of the effective period for the correct answer |
| `valid_to` | ISO date or null | End of the effective period (null = still valid) |
| `modality` | enum | `text`, `table`, `chart`, `spreadsheet`, `mixed` |
| `difficulty` | enum | `easy`, `medium`, `hard` |
| `answerable` | boolean | Whether the corpus contains a correct answer |
| `failure_type` | enum or null | Expected failure category for negative test cases |

## Modality Labels

| Value | Meaning |
|---|---|
| `text` | Answer lives in plain prose |
| `table` | Answer requires reading a structured table |
| `chart` | Answer requires interpreting a chart or graph |
| `spreadsheet` | Answer requires reading spreadsheet cells or formulas |
| `mixed` | Answer requires combining text with at least one non-text source |

## Failure Type Labels (for negative / adversarial test cases)

| Value | Meaning |
|---|---|
| `STALE_ANSWER` | Only older document versions contain the answer |
| `WRONG_VERSION` | Multiple versions exist; only the version-correct one applies |
| `TABLE_RETRIEVAL_MISS` | Answer is only in a table not easily retrieved by text search |
| `CHART_LAYOUT_BLINDNESS` | Answer is only in a chart or image |
| `CHUNKING_BOUNDARY_ERROR` | Answer spans a typical chunk boundary |
| `LOW_RECALL_RETRIEVAL` | Answer is buried and unlikely to surface in top-k |
| `UNANSWERABLE` | Not in corpus; tests hallucination resistance |

## Sample Corpus Versioning

```
Query date: 2024-03-01 → use Policy v1 (effective Jan 2024 – Jun 2024)
Query date: 2024-09-01 → use Policy v2 (effective Jul 2024 – Dec 2024)
Query date: 2025-02-01 → use Policy v3 (effective Jan 2025 – present)
```

Questions with `valid_from`/`valid_to` values test whether the pipeline
retrieves the version valid at query time rather than the most recent version.
