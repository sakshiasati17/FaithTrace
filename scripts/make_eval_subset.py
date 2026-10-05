#!/usr/bin/env python3
"""Build deterministic focused subsets of eval_sets/faithtrace_v1.json.

Usage:
    python scripts/make_eval_subset.py            # (re)write the subset files
    python scripts/make_eval_subset.py --check    # exit 1 if a committed file differs

Items are copied unchanged and kept in their faithtrace_v1.json order.
Temporal pairs (items sharing a temporal_pair_id) are never split.

  faithtrace_smoke.json       5 items: the first temporal pair, the first
                              answerable table item, the first answerable
                              chart item, the first unanswerable item.
  faithtrace_temporal.json    every item with a temporal_pair_id.
  faithtrace_multimodal.json  answerable, non-temporal table/spreadsheet/chart items.
  faithtrace_core40.json      the first 8 temporal pairs, then from the rest
                              (v1 order): 6 unanswerable, 8 table, 4 spreadsheet
                              and 6 chart items (the last three answerable).
"""

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
EVAL_SETS_DIR = REPO_ROOT / "eval_sets"
SOURCE = EVAL_SETS_DIR / "faithtrace_v1.json"

CORE_PAIRS = 8
CORE_PICKS = (
    # (label, count, predicate) — applied in this order to the non-pair items.
    ("unanswerable", 6, lambda i: i.get("answerable") is False),
    ("table", 8, lambda i: i.get("modality") == "table" and i.get("answerable") is not False),
    ("spreadsheet", 4, lambda i: i.get("modality") == "spreadsheet" and i.get("answerable") is not False),
    ("chart", 6, lambda i: i.get("modality") == "chart" and i.get("answerable") is not False),
)


class SubsetError(RuntimeError):
    """Raised when the source set cannot satisfy a subset rule."""


def _pair_ids(items):
    """temporal_pair_ids in order of first appearance."""
    seen = []
    for item in items:
        pid = item.get("temporal_pair_id")
        if pid and pid not in seen:
            seen.append(pid)
    return seen


def _first(items, predicate, label):
    for item in items:
        if predicate(item):
            return item
    raise SubsetError(f"no item for {label}")


def _in_source_order(items, selected):
    """Return the selected items in source order (selection by identity)."""
    chosen = {id(i) for i in selected}
    return [i for i in items if id(i) in chosen]


def smoke(items):
    pairs = _pair_ids(items)
    if not pairs:
        raise SubsetError("no temporal pair for smoke set")
    picked = [i for i in items if i.get("temporal_pair_id") == pairs[0]]
    picked.append(_first(items, lambda i: i.get("modality") == "table" and i.get("answerable") is True,
                         "answerable table item"))
    picked.append(_first(items, lambda i: i.get("modality") == "chart" and i.get("answerable") is True,
                         "answerable chart item"))
    picked.append(_first(items, lambda i: i.get("answerable") is False, "unanswerable item"))
    return _in_source_order(items, picked)


def temporal(items):
    return [i for i in items if i.get("temporal_pair_id")]


def multimodal(items):
    return [
        i for i in items
        if i.get("answerable") is True
        and not i.get("temporal_pair_id")
        and i.get("modality") in ("table", "spreadsheet", "chart")
    ]


def core40(items):
    pairs = _pair_ids(items)
    if len(pairs) < CORE_PAIRS:
        raise SubsetError(f"need {CORE_PAIRS} temporal pairs, found {len(pairs)}")
    keep_pairs = set(pairs[:CORE_PAIRS])
    keep = [i for i in items if i.get("temporal_pair_id") in keep_pairs]
    rest = [i for i in items if i.get("temporal_pair_id") not in keep_pairs]
    picked = list(keep)
    for label, count, predicate in CORE_PICKS:
        chosen_ids = {id(i) for i in picked}
        matches = [i for i in rest if predicate(i) and id(i) not in chosen_ids][:count]
        if len(matches) < count:
            raise SubsetError(f"core40 needs {count} {label} items, found {len(matches)}")
        picked.extend(matches)
    return _in_source_order(items, picked)


SUBSETS = {
    "faithtrace_smoke.json": smoke,
    "faithtrace_temporal.json": temporal,
    "faithtrace_multimodal.json": multimodal,
    "faithtrace_core40.json": core40,
}


def _check_pairs(name, subset, items):
    """Every temporal pair in the subset must be complete."""
    for pid in _pair_ids(subset):
        want = sum(1 for i in items if i.get("temporal_pair_id") == pid)
        got = sum(1 for i in subset if i.get("temporal_pair_id") == pid)
        if got != want:
            raise SubsetError(f"{name}: temporal pair {pid} split ({got} of {want} items)")


def render(items):
    """Return {filename: file text} for every subset."""
    out = {}
    for name, build in SUBSETS.items():
        subset = build(items)
        _check_pairs(name, subset, items)
        out[name] = json.dumps(subset, indent=2, ensure_ascii=False) + "\n"
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--check", action="store_true",
                        help="regenerate in memory and exit 1 if any committed file differs")
    args = parser.parse_args(argv)

    try:
        items = json.loads(SOURCE.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        print(f"ERROR: cannot load {SOURCE}: {exc}")
        return 1
    if not isinstance(items, list):
        print(f"ERROR: {SOURCE} is not a JSON list")
        return 1

    try:
        rendered = render(items)
    except SubsetError as exc:
        print(f"ERROR: {exc}")
        return 1

    stale = []
    for name, text in rendered.items():
        path = EVAL_SETS_DIR / name
        count = len(json.loads(text))
        if args.check:
            try:
                current = path.read_text(encoding="utf-8")
            except OSError as exc:
                print(f"STALE: {path.relative_to(REPO_ROOT)} cannot be read: {exc}")
                stale.append(name)
                continue
            if current != text:
                print(f"STALE: {path.relative_to(REPO_ROOT)} differs from the generated set")
                stale.append(name)
            else:
                print(f"OK: {path.relative_to(REPO_ROOT)} ({count} items)")
        else:
            path.write_text(text, encoding="utf-8")
            print(f"wrote {path.relative_to(REPO_ROOT)} ({count} items)")

    if stale:
        print("Run: python scripts/make_eval_subset.py")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
