"""
Shared test fixtures for FaithTrace backend tests.
"""

import pytest
from unittest.mock import MagicMock, AsyncMock, patch
from pathlib import Path


@pytest.fixture
def sample_eval_set():
    """Load the frozen golden regression eval set."""
    import json
    path = Path(__file__).parent.parent.parent / "eval_sets" / "golden_regression_set.json"
    with open(path) as f:
        return json.load(f)


@pytest.fixture
def mock_db_session():
    """Mock async SQLAlchemy session."""
    session = AsyncMock()
    session.commit = AsyncMock()
    session.add = MagicMock()
    session.delete = AsyncMock()
    session.execute = AsyncMock()
    return session


@pytest.fixture
def mock_qdrant_client():
    """Mock Qdrant client that simulates normal operation."""
    client = MagicMock()
    client.get_collections.return_value = MagicMock(
        collections=[MagicMock(name="faithtrace_chunks")]
    )
    client.upsert.return_value = None
    client.delete.return_value = None
    client.scroll.return_value = ([], None)
    return client


@pytest.fixture
def sample_query_result():
    """A sample QueryResult-like dict for testing."""
    return {
        "query_id": "q_001",
        "question": "What is the maximum order exception threshold?",
        "generated_answer": "The threshold is $15,000 per order.",
        "retrieved_chunks": [
            {
                "content": "The threshold is $15,000 per order as revised.",
                "chunk_type": "text",
                "page": 3,
                "filename": "corporate-procurement-policy-2.pdf",
                "doc_version": "v2",
                "effective_from": 1719792000,
                "effective_to": None,
            }
        ],
        "latency_ms": 734.4,
        "input_tokens": 512,
        "output_tokens": 48,
        "cost_usd": 0.001,
    }
