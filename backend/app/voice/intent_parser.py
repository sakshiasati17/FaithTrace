"""
Intent parser — maps natural language voice commands to FaithTrace API calls.

Design choice: deterministic regex patterns, NOT an NLU model.
Rationale:
  - Command vocabulary is small (~20 intents) and domain-specific.
  - Pattern matching is deterministic, debuggable, and adds zero latency.
  - A probabilistic NLU model can hallucinate intents; for a developer
    tool, predictable failures with clear error messages are better UX.
  - Upgrade path: replace parse() with a fine-tuned small intent classifier
    if commands grow more conversational.
"""

import re
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ParsedIntent:
    action: str
    parameters: dict[str, Any]
    api_endpoint: str
    api_method: str
    raw_text: str
    confidence: float


class IntentParser:
    def __init__(self):
        # (regex, action, endpoint_template, http_method, param_extractor)
        self._patterns = [
            (
                r"run experiment (\d+)",
                "run_experiment", "/experiments/{id}/run", "POST",
                lambda m: {"id": int(m.group(1))},
            ),
            (
                r"(?:compare|diff) (?:experiment|config)s? (\d+) (?:and|with|vs\.?) (\d+)",
                "compare_experiments", "/experiments/compare", "GET",
                lambda m: {"exp_a": int(m.group(1)), "exp_b": int(m.group(2))},
            ),
            (
                r"(?:show|get|list) (?:all )?experiments",
                "list_experiments", "/experiments", "GET",
                lambda m: {},
            ),
            (
                r"(?:what|which) (?:went wrong|failed|is the (?:problem|issue))",
                "diagnose_latest", "/diagnostics/latest", "GET",
                lambda m: {},
            ),
            (
                r"diagnose (?:experiment )?(\d+)",
                "diagnose_experiment", "/diagnostics/experiment/{id}", "GET",
                lambda m: {"id": int(m.group(1))},
            ),
            (
                r"(?:show|get) (?:worst|failed|bad) (?:queries|results|answers)",
                "get_failures", "/diagnostics/failures", "GET",
                lambda m: {},
            ),
            (
                r"(?:suggest|recommend|get) (?:improvements|optimizations|recommendations)",
                "get_recommendations", "/recommendations/", "GET",
                lambda m: {},
            ),
            (
                r"(?:run|start) (?:benchmark|profiling|performance test)",
                "run_benchmark", "/optimization/benchmark", "POST",
                lambda m: {},
            ),
            (
                r"(?:show|get) (?:gpu|hardware) (?:info|profile|status)",
                "gpu_profile", "/optimization/gpu-profile", "GET",
                lambda m: {},
            ),
            (
                r"(?:show|get) (?:benchmark|performance) (?:results|report)",
                "get_benchmark", "/optimization/benchmark/latest", "GET",
                lambda m: {},
            ),
            (
                r"(?:status|what.s running|any jobs)",
                "check_status", "/status", "GET",
                lambda m: {},
            ),
            (
                r"(?:help|what can you do|commands)",
                "help", "", "",
                lambda m: {},
            ),
        ]

    def parse(self, text: str) -> ParsedIntent:
        cleaned = re.sub(r"[^\w\s]", "", text.lower().strip())
        cleaned = re.sub(r"\s+", " ", cleaned)

        for pattern, action, endpoint, method, extractor in self._patterns:
            m = re.search(pattern, cleaned)
            if m:
                params = extractor(m)
                formatted = endpoint
                for k, v in params.items():
                    formatted = formatted.replace(f"{{{k}}}", str(v))
                return ParsedIntent(
                    action=action,
                    parameters=params,
                    api_endpoint=formatted,
                    api_method=method,
                    raw_text=text,
                    confidence=0.95,
                )

        return ParsedIntent(
            action="unknown",
            parameters={},
            api_endpoint="",
            api_method="",
            raw_text=text,
            confidence=0.0,
        )

    def help_text(self) -> str:
        return (
            "Available commands: "
            "run experiment [number]. "
            "Compare experiments [A] and [B]. "
            "What went wrong. "
            "Show failed queries. "
            "Suggest improvements. "
            "Run benchmark. "
            "Show GPU info. "
            "Check status."
        )
