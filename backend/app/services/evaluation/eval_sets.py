"""
Eval set validation, parsing and built-in file resolution.

An eval set is a list of question items (see docs/dataset/eval_set_schema.md).
Sets come from two places:
  - uploads (JSON or CSV), validated here and stored in the eval_sets table;
  - built-in JSON files in the repo's eval_sets/ folder.

Paths are only ever resolved inside the eval_sets/ folder.
"""

import csv
import io
import json
import logging
import os
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

from app.services.diagnostics.classifier import FailureCategory

logger = logging.getLogger(__name__)

# Single default used by the API schema, classifier training, workers and frontend.
DEFAULT_EVAL_SET_PATH = "eval_sets/faithtrace_v1.json"

# Built-in sets are addressed in the API as "builtin:<filename>".
BUILTIN_ID_PREFIX = "builtin:"

MAX_EVAL_SET_BYTES = 5 * 1024 * 1024
MAX_EVAL_SET_ITEMS = 5000

REQUIRED_FIELDS = ("id", "question", "ground_truth")
OPTIONAL_FIELDS = (
    "source_docs", "valid_from", "valid_to", "modality", "difficulty",
    "answerable", "failure_type", "evidence", "temporal_pair_id",
)
MODALITIES = {"text", "table", "chart", "spreadsheet", "mixed"}
DIFFICULTIES = {"easy", "medium", "hard"}
FAILURE_TYPES = {c.value for c in FailureCategory} | {"UNANSWERABLE"}

_TRUE = {"true", "1", "yes", "y"}
_FALSE = {"false", "0", "no", "n"}


class EvalSetError(ValueError):
    """Raised when an eval set file cannot be parsed or a path is not allowed."""


@dataclass
class ValidationResult:
    items: list[dict] = field(default_factory=list)
    errors: list[dict] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors


# ─── Built-in folder ──────────────────────────────────────────────────────────

def eval_sets_dir() -> Path:
    """The repo eval_sets/ folder (EVAL_SETS_DIR env var overrides)."""
    override = os.environ.get("EVAL_SETS_DIR")
    if override:
        return Path(override)
    here = Path(__file__).resolve()
    candidates = [
        here.parents[4] / "eval_sets",  # repo checkout: <repo>/backend/app/services/evaluation
        here.parents[3] / "eval_sets",  # container: /app/eval_sets mounted next to app/
    ]
    for c in candidates:
        if c.is_dir():
            return c
    return candidates[0]


def resolve_eval_set_path(path: str) -> tuple[str, Path]:
    """
    Resolve a legacy eval set path to a file inside eval_sets/.

    Accepts "eval_sets/<name>.json" or "<name>.json". Rejects absolute paths,
    "..", sub-folders, other folders and missing files.
    Returns (normalised "eval_sets/<name>.json", absolute Path).
    """
    if not isinstance(path, str) or not path.strip():
        raise EvalSetError("eval_set_path is empty")
    raw = path.strip()
    if "\\" in raw or raw.startswith("/") or (len(raw) > 1 and raw[1] == ":"):
        raise EvalSetError(f"eval_set_path must be relative to eval_sets/: {path!r}")
    parts = raw.split("/")
    if any(p in ("..", ".", "") for p in parts):
        raise EvalSetError(f"eval_set_path may not contain '..', '.' or empty segments: {path!r}")
    if parts[0] == "eval_sets":
        parts = parts[1:]
    if len(parts) != 1:
        raise EvalSetError(f"eval_set_path must name a file directly inside eval_sets/: {path!r}")
    name = parts[0]
    if not name.endswith(".json"):
        raise EvalSetError(f"eval_set_path must be a .json file: {path!r}")

    base = eval_sets_dir().resolve()
    target = (base / name).resolve()
    if not target.is_relative_to(base):
        raise EvalSetError(f"eval_set_path resolves outside eval_sets/: {path!r}")
    if not target.is_file():
        raise EvalSetError(f"eval set not found: eval_sets/{name}")
    return f"eval_sets/{name}", target


def load_builtin(path: str) -> list[dict]:
    """Load a built-in eval set by path (restricted to eval_sets/)."""
    _, target = resolve_eval_set_path(path)
    with open(target, encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise EvalSetError(f"{path} is not a JSON list of items")
    return data


def list_builtin() -> list[dict]:
    """Describe every *.json file in eval_sets/ (unreadable files are logged and skipped)."""
    out = []
    base = eval_sets_dir()
    if not base.is_dir():
        logger.warning("Built-in eval set folder not found: %s", base)
        return out
    for p in sorted(base.glob("*.json")):
        try:
            with open(p, encoding="utf-8") as f:
                data = json.load(f)
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("Skipping unreadable built-in eval set %s: %s", p.name, exc)
            continue
        if not isinstance(data, list):
            logger.warning("Skipping built-in eval set %s: not a JSON list", p.name)
            continue
        out.append({
            "id": f"{BUILTIN_ID_PREFIX}{p.name}",
            "name": p.stem,
            "description": "",
            "source": "builtin",
            "filename": p.name,
            "path": f"eval_sets/{p.name}",
            "item_count": len(data),
            "created_at": None,
            "is_default": f"eval_sets/{p.name}" == DEFAULT_EVAL_SET_PATH,
        })
    return out


# ─── Parsing ──────────────────────────────────────────────────────────────────

def parse_upload(filename: str, content: bytes) -> list:
    """Parse an uploaded JSON or CSV file into raw item dicts (not yet validated)."""
    if len(content) > MAX_EVAL_SET_BYTES:
        raise EvalSetError(f"file is larger than {MAX_EVAL_SET_BYTES // (1024 * 1024)} MB")
    try:
        text = content.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise EvalSetError(f"file is not UTF-8 text: {exc}") from exc

    lower = (filename or "").lower()
    if lower.endswith(".csv"):
        return parse_csv(text)
    if lower.endswith(".json"):
        return parse_json(text)
    raise EvalSetError("file must be .json or .csv")


def parse_json(text: str) -> list:
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise EvalSetError(f"invalid JSON: {exc}") from exc
    if not isinstance(data, list):
        raise EvalSetError("JSON eval set must be a list of objects")
    return data


def parse_csv(text: str) -> list[dict]:
    """CSV with a header row. source_docs is ';'-separated; empty cells are omitted."""
    reader = csv.DictReader(io.StringIO(text))
    if not reader.fieldnames:
        raise EvalSetError("CSV has no header row")
    items = []
    for row in reader:
        item: dict = {}
        for key, value in row.items():
            if key is None:
                # More cells than header columns.
                item.setdefault("_extra_cells", value)
                continue
            key = key.strip()
            value = (value or "").strip()
            if value == "":
                continue
            if key == "source_docs":
                item[key] = [s.strip() for s in value.split(";") if s.strip()]
            elif key == "answerable":
                low = value.lower()
                item[key] = True if low in _TRUE else False if low in _FALSE else value
            elif key == "evidence" and value.startswith("{"):
                try:
                    item[key] = json.loads(value)
                except json.JSONDecodeError:
                    item[key] = value
            else:
                item[key] = value
        items.append(item)
    return items


# ─── Validation ───────────────────────────────────────────────────────────────

def _is_iso_date(value) -> bool:
    if not isinstance(value, str):
        return False
    try:
        date.fromisoformat(value)
        return True
    except ValueError:
        return False


def validate_items(raw_items: list) -> ValidationResult:
    """
    Validate items against docs/dataset/eval_set_schema.md.

    Errors are per row: {"row": 1-based index, "id": item id or None,
    "field": field name or None, "message": str}.
    """
    result = ValidationResult()
    if not isinstance(raw_items, list):
        result.errors.append({"row": None, "id": None, "field": None,
                              "message": "eval set must be a list of items"})
        return result
    if not raw_items:
        result.errors.append({"row": None, "id": None, "field": None,
                              "message": "eval set has no items"})
        return result
    if len(raw_items) > MAX_EVAL_SET_ITEMS:
        result.errors.append({"row": None, "id": None, "field": None,
                              "message": f"eval set has more than {MAX_EVAL_SET_ITEMS} items"})
        return result

    seen_ids: dict[str, int] = {}
    for idx, raw in enumerate(raw_items, start=1):
        def err(fld, msg, _idx=idx, _raw=raw):
            item_id = _raw.get("id") if isinstance(_raw, dict) else None
            result.errors.append({"row": _idx, "id": item_id if isinstance(item_id, str) else None,
                                  "field": fld, "message": msg})

        if not isinstance(raw, dict):
            err(None, "item must be an object")
            continue
        if "_extra_cells" in raw:
            err(None, "row has more cells than the header")

        for fld in REQUIRED_FIELDS:
            v = raw.get(fld)
            if not isinstance(v, str) or not v.strip():
                err(fld, f"'{fld}' is required and must be a non-empty string")

        item_id = raw.get("id")
        if isinstance(item_id, str) and item_id.strip():
            if item_id in seen_ids:
                err("id", f"duplicate id '{item_id}' (first seen in row {seen_ids[item_id]})")
            else:
                seen_ids[item_id] = idx

        if "source_docs" in raw and raw["source_docs"] is not None:
            docs = raw["source_docs"]
            if not isinstance(docs, list) or not all(isinstance(d, str) and d.strip() for d in docs):
                err("source_docs", "'source_docs' must be a list of filenames")

        for fld in ("valid_from", "valid_to"):
            v = raw.get(fld)
            if v is not None and not _is_iso_date(v):
                err(fld, f"'{fld}' must be an ISO date (YYYY-MM-DD) or null, got {v!r}")
        vf, vt = raw.get("valid_from"), raw.get("valid_to")
        if _is_iso_date(vf) and _is_iso_date(vt) and date.fromisoformat(vf) > date.fromisoformat(vt):
            err("valid_to", "'valid_to' is before 'valid_from'")

        v = raw.get("modality")
        if v is not None and v not in MODALITIES:
            err("modality", f"'modality' must be one of {sorted(MODALITIES)}, got {v!r}")

        v = raw.get("difficulty")
        if v is not None and v not in DIFFICULTIES:
            err("difficulty", f"'difficulty' must be one of {sorted(DIFFICULTIES)}, got {v!r}")

        v = raw.get("answerable")
        if v is not None and not isinstance(v, bool):
            err("answerable", f"'answerable' must be true or false, got {v!r}")

        v = raw.get("failure_type")
        if v is not None and v not in FAILURE_TYPES:
            err("failure_type", f"'failure_type' must be one of {sorted(FAILURE_TYPES)} or null, got {v!r}")

        v = raw.get("evidence")
        if v is not None and not isinstance(v, (dict, str)):
            err("evidence", "'evidence' must be an object or a string")

        v = raw.get("temporal_pair_id")
        if v is not None and not isinstance(v, str):
            err("temporal_pair_id", "'temporal_pair_id' must be a string or null")

    if not result.errors:
        result.items = [dict(item) for item in raw_items]
    return result


def unmatched_source_docs(items: list[dict], known_filenames: set[str]) -> list[str]:
    """source_docs named by items that match no uploaded Document.filename."""
    missing = set()
    for item in items:
        for doc in item.get("source_docs") or []:
            if doc not in known_filenames:
                missing.add(doc)
    return sorted(missing)
