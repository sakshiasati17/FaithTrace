"""
Focused eval subsets generated from eval_sets/faithtrace_v1.json by
scripts/make_eval_subset.py.

Covers:
  - each subset file exists, passes scripts/validate_eval_set.py --check-evidence
  - ids are unique and every item is identical to its v1 entry, in v1 order
  - temporal pairs are never split
  - counts and compositions of smoke / temporal / multimodal / core40
  - the committed files match the generator (--check)
  - the subsets are listed as built-ins and faithtrace_v1.json stays the default
"""

import json
import subprocess
import sys
from collections import Counter
from pathlib import Path

import pytest

from app.services.evaluation import eval_sets as es

REPO_ROOT = Path(__file__).resolve().parents[2]
EVAL_SETS = REPO_ROOT / "eval_sets"
SCRIPTS = REPO_ROOT / "scripts"
SUBSET_FILES = [
    "faithtrace_smoke.json",
    "faithtrace_temporal.json",
    "faithtrace_multimodal.json",
    "faithtrace_core40.json",
]


def _load(name):
    return json.loads((EVAL_SETS / name).read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def v1():
    return _load("faithtrace_v1.json")


@pytest.fixture(scope="module")
def v1_by_id(v1):
    return {item["id"]: item for item in v1}


def _pair_sizes(items):
    return Counter(i["temporal_pair_id"] for i in items if i.get("temporal_pair_id"))


def _composition(items):
    return {
        "count": len(items),
        "temporal": sum(1 for i in items if i.get("temporal_pair_id")),
        "unanswerable": sum(1 for i in items if i.get("answerable") is False),
        "modality": dict(Counter(i["modality"] for i in items)),
    }


@pytest.mark.parametrize("name", SUBSET_FILES)
class TestSubsetFile:
    def test_exists_with_trailing_newline(self, name):
        text = (EVAL_SETS / name).read_text(encoding="utf-8")
        assert text.endswith("\n")
        assert isinstance(json.loads(text), list)

    def test_passes_validator_with_evidence(self, name):
        proc = subprocess.run(
            [sys.executable, str(SCRIPTS / "validate_eval_set.py"), f"eval_sets/{name}", "--check-evidence"],
            cwd=REPO_ROOT, capture_output=True, text=True,
        )
        assert proc.returncode == 0, proc.stdout + proc.stderr
        assert "OK:" in proc.stdout

    def test_items_unique_identical_and_in_v1_order(self, name, v1, v1_by_id):
        items = _load(name)
        ids = [i["id"] for i in items]
        assert len(ids) == len(set(ids))
        assert set(ids) <= set(v1_by_id)
        for item in items:
            assert item == v1_by_id[item["id"]]
        v1_order = [i["id"] for i in v1 if i["id"] in set(ids)]
        assert ids == v1_order

    def test_pairs_intact(self, name, v1):
        full = _pair_sizes(v1)
        for pid, size in _pair_sizes(_load(name)).items():
            assert size == full[pid] == 2, pid


def test_smoke_composition(v1):
    items = _load("faithtrace_smoke.json")
    first_pair = next(i["temporal_pair_id"] for i in v1 if i.get("temporal_pair_id"))
    first_table = next(i for i in v1 if i["modality"] == "table" and i["answerable"] is True)
    first_chart = next(i for i in v1 if i["modality"] == "chart" and i["answerable"] is True)
    first_unans = next(i for i in v1 if i["answerable"] is False)
    expected = {i["id"] for i in v1 if i.get("temporal_pair_id") == first_pair}
    expected |= {first_table["id"], first_chart["id"], first_unans["id"]}
    assert len(items) == 5
    assert {i["id"] for i in items} == expected


def test_temporal_composition(v1):
    items = _load("faithtrace_temporal.json")
    assert [i["id"] for i in items] == [i["id"] for i in v1 if i.get("temporal_pair_id")]
    assert _composition(items) == {"count": 30, "temporal": 30, "unanswerable": 0,
                                   "modality": {"text": 30}}
    assert len(_pair_sizes(items)) == 15


def test_multimodal_composition():
    items = _load("faithtrace_multimodal.json")
    assert _composition(items) == {"count": 38, "temporal": 0, "unanswerable": 0,
                                   "modality": {"table": 16, "spreadsheet": 13, "chart": 9}}


def test_core40_composition(v1):
    items = _load("faithtrace_core40.json")
    assert _composition(items) == {"count": 40, "temporal": 16, "unanswerable": 6,
                                   "modality": {"text": 18, "table": 9, "chart": 7, "spreadsheet": 6}}
    pairs = list(dict.fromkeys(i["temporal_pair_id"] for i in v1 if i.get("temporal_pair_id")))
    assert set(_pair_sizes(items)) == set(pairs[:8])


def test_generator_check_passes():
    proc = subprocess.run(
        [sys.executable, str(SCRIPTS / "make_eval_subset.py"), "--check"],
        cwd=REPO_ROOT, capture_output=True, text=True,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr


def test_subsets_listed_as_builtin_and_default_unchanged(monkeypatch):
    monkeypatch.delenv("EVAL_SETS_DIR", raising=False)
    listed = {e["filename"]: e for e in es.list_builtin()}
    for name in SUBSET_FILES:
        assert listed[name]["id"] == f"builtin:{name}"
        assert listed[name]["item_count"] == len(_load(name))
        assert listed[name]["is_default"] is False
    assert es.DEFAULT_EVAL_SET_PATH == "eval_sets/faithtrace_v1.json"
    assert [n for n, e in listed.items() if e["is_default"]] == ["faithtrace_v1.json"]
