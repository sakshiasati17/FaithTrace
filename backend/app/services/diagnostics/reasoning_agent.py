"""
LLM-powered diagnostic reasoning agent.

Uses GPT-4o to reason step-by-step about why a RAG query failed.
Results are cached in diagnosis_evidence["reasoning"] to avoid repeated API calls.

Design: on-demand only (expensive). Called via POST /diagnostics/.../reason.
"""

from __future__ import annotations

import json
import logging
from typing import Optional

from app.core.config import settings

logger = logging.getLogger(__name__)

# Failure category → plain-English description for the prompt
_FAILURE_DESCRIPTIONS = {
    "STALE_ANSWER": "retrieved chunks are temporally outdated (past their validity window)",
    "WRONG_VERSION": "retrieved chunks are from a different document version than expected",
    "TABLE_RETRIEVAL_MISS": "a table/spreadsheet chunk was needed but not retrieved",
    "CHART_LAYOUT_BLINDNESS": "a chart/diagram needed vision parsing but was not processed correctly",
    "CHUNKING_BOUNDARY_ERROR": "the relevant information was split across chunk boundaries and lost",
    "LOW_RECALL_RETRIEVAL": "the retriever failed to surface the correct context chunks",
    "IRRELEVANT_CONTEXT_POLLUTION": "irrelevant chunks diluted the context and confused the model",
    "UNSUPPORTED_SYNTHESIS": "the model generated an answer not supported by the retrieved context",
    "NO_FAILURE": "no significant failure detected — the answer quality is acceptable",
}

_SYSTEM_PROMPT = """You are a senior RAG (Retrieval-Augmented Generation) systems engineer.
Your job is to diagnose exactly why a RAG pipeline produced a poor answer.

You will be given:
- The original question
- The generated answer from the RAG pipeline
- The retrieved context chunks (with their types and metadata)
- Evaluation metrics (faithfulness, context_recall, context_precision, answer_correctness)
- The pre-classified failure category with a brief description

Your task:
1. Reason step by step about what went wrong
2. Identify the true root cause clearly
3. Give ONE concrete, actionable fix that an engineer can implement today
4. Write a plain-English summary for a non-technical stakeholder

Respond ONLY with a JSON object in this exact format:
{
  "reasoning_steps": [
    "Step 1: <observation>",
    "Step 2: <deduction>",
    "Step 3: <conclusion>"
  ],
  "root_cause": "<1-2 sentence plain English root cause>",
  "fix_suggestion": "<1 concrete actionable fix for the engineering team>",
  "stakeholder_summary": "<2-3 sentence non-technical explanation of what went wrong and impact>",
  "confidence": <float 0.0-1.0>
}"""


def _build_prompt(
    question: str,
    generated_answer: str,
    retrieved_chunks: list[dict],
    metrics: dict,
    failure_category: str,
) -> str:
    failure_desc = _FAILURE_DESCRIPTIONS.get(failure_category, failure_category)

    chunks_text = ""
    for i, chunk in enumerate(retrieved_chunks[:5]):  # cap at 5 to stay within token limits
        ctype = chunk.get("chunk_type", "text")
        content = (chunk.get("content") or "")[:400]
        meta = []
        if chunk.get("filename"):
            meta.append(f"file={chunk['filename']}")
        if chunk.get("doc_version"):
            meta.append(f"version={chunk['doc_version']}")
        if chunk.get("page"):
            meta.append(f"page={chunk['page']}")
        meta_str = f" [{', '.join(meta)}]" if meta else ""
        chunks_text += f"\nChunk {i+1} (type={ctype}{meta_str}):\n{content}\n"

    metrics_text = "\n".join(
        f"  {k}: {v:.3f}" for k, v in metrics.items() if isinstance(v, (int, float))
    )

    return f"""=== QUESTION ===
{question}

=== GENERATED ANSWER ===
{generated_answer}

=== RETRIEVED CHUNKS ===
{chunks_text}

=== EVALUATION METRICS ===
{metrics_text}

=== PRE-CLASSIFIED FAILURE ===
Category: {failure_category}
Description: {failure_desc}

Now reason step by step about the root cause and provide your diagnosis in the required JSON format."""


async def reason(
    question: str,
    generated_answer: str,
    retrieved_chunks: list[dict],
    metrics: dict,
    failure_category: str,
) -> dict:
    """
    Call GPT-4o to reason about why a query failed.

    Returns a structured dict with reasoning_steps, root_cause,
    fix_suggestion, stakeholder_summary, and confidence.
    Raises RuntimeError on API failure.
    """
    try:
        from openai import AsyncOpenAI
    except ImportError as exc:
        raise RuntimeError("openai package is required") from exc

    client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)
    prompt = _build_prompt(question, generated_answer, retrieved_chunks, metrics, failure_category)

    response = await client.chat.completions.create(
        model=settings.OPENAI_MODEL,
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        temperature=0.2,
        max_tokens=800,
        response_format={"type": "json_object"},
    )

    raw = response.choices[0].message.content or "{}"
    try:
        result = json.loads(raw)
    except json.JSONDecodeError:
        logger.warning("Reasoning agent returned invalid JSON: %s", raw[:200])
        result = {
            "reasoning_steps": ["Could not parse structured response from model."],
            "root_cause": raw[:500],
            "fix_suggestion": "Review the raw model output above.",
            "stakeholder_summary": "Automated analysis was unable to produce a structured result.",
            "confidence": 0.3,
        }

    # Normalise — ensure all expected keys exist
    result.setdefault("reasoning_steps", [])
    result.setdefault("root_cause", "")
    result.setdefault("fix_suggestion", "")
    result.setdefault("stakeholder_summary", "")
    result.setdefault("confidence", 0.5)

    logger.info(
        "Reasoning agent completed: category=%s confidence=%.2f",
        failure_category, result.get("confidence", 0),
    )
    return result
