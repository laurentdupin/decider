"""Compare Python reference output with native --diagnostics output."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def maximum_delta(left, right) -> float:
    if len(left) != len(right):
        raise AssertionError(f"vector lengths differ: {len(left)} != {len(right)}")
    return max((abs(float(a) - float(b)) for a, b in zip(left, right)), default=0.0)


def response_delta(expected, actual, tolerance: float, path: str = "response") -> float:
    if isinstance(expected, dict):
        if not isinstance(actual, dict) or expected.keys() != actual.keys():
            raise AssertionError(f"{path} object keys differ")
        return max(
            (response_delta(value, actual[key], tolerance, f"{path}.{key}")
             for key, value in expected.items()),
            default=0.0,
        )
    if isinstance(expected, list):
        if not isinstance(actual, list) or len(expected) != len(actual):
            raise AssertionError(f"{path} list shape differs")
        return max(
            (response_delta(left, right, tolerance, f"{path}[{index}]")
             for index, (left, right) in enumerate(zip(expected, actual))),
            default=0.0,
        )
    if isinstance(expected, (int, float)) and not isinstance(expected, bool):
        delta = abs(float(expected) - float(actual))
        if path.startswith("response.usage.") and delta != 0:
            raise AssertionError(f"{path} differs: {expected} != {actual}")
        if delta > tolerance:
            raise AssertionError(f"{path} delta {delta} exceeds {tolerance}")
        return delta
    if expected != actual:
        raise AssertionError(f"{path} differs: {expected!r} != {actual!r}")
    return 0.0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("python_fixture", type=Path)
    parser.add_argument("native_response", type=Path)
    parser.add_argument("--logit-atol", type=float, default=0.1)
    parser.add_argument("--probability-atol", type=float, default=0.02)
    parser.add_argument("--response-atol", type=float, default=0.02)
    args = parser.parse_args()

    reference = json.loads(args.python_fixture.read_text(encoding="utf-8"))
    native = json.loads(args.native_response.read_text(encoding="utf-8"))
    diagnostics = native.pop("_diagnostics")
    if reference["label_token_ids"] != diagnostics["label_token_ids"]:
        raise AssertionError("label token IDs differ")
    if len(reference["rows"]) != len(diagnostics["rows"]):
        raise AssertionError("physical row counts differ")

    max_logit = 0.0
    max_probability = 0.0
    for index, (expected, actual) in enumerate(
        zip(reference["rows"], diagnostics["rows"])
    ):
        for key in ("token_ids", "slots", "option_counts", "types"):
            if expected[key] != actual[key]:
                raise AssertionError(f"row {index} {key} differs")
        for left, right in zip(expected["selected_logits"], actual["selected_logits"]):
            max_logit = max(max_logit, maximum_delta(left, right))
        for left, right in zip(expected["probabilities"], actual["probabilities"]):
            max_probability = max(max_probability, maximum_delta(left, right))

    if max_logit > args.logit_atol:
        raise AssertionError(f"maximum logit delta {max_logit} exceeds {args.logit_atol}")
    if max_probability > args.probability_atol:
        raise AssertionError(
            f"maximum probability delta {max_probability} exceeds {args.probability_atol}"
        )
    max_response = response_delta(reference["response"], native, args.response_atol)
    print(
        f"PARITY_OK rows={len(reference['rows'])} "
        f"max_logit_delta={max_logit:.8g} max_probability_delta={max_probability:.8g} "
        f"max_response_delta={max_response:.8g}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
