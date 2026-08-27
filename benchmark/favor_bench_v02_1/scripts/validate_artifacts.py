#!/usr/bin/env python3
"""Validate a FAVoR continuation run directory against lightweight checks."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


REQUIRED_FILES = [
    "generations.jsonl",
    "per_example_metrics.jsonl",
    "summary_metrics.csv",
    "training_budget.csv",
    "quality_filter_report.json",
    "method_config_resolved.yaml",
    "seed_manifest.json",
]


def iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if line.strip():
                yield line_no, json.loads(line)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", required=True)
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    archive_root = Path(__file__).resolve().parents[1]
    method_names = json.loads((archive_root / "method_names.json").read_text(encoding="utf-8"))
    valid_methods = set(method_names["official_names_used_in_paper"])
    failures = []

    for name in REQUIRED_FILES:
        if not (run_dir / name).exists():
            failures.append(f"missing {name}")

    if failures:
        print("FAIL")
        print("\n".join(failures))
        return 1

    generation_ids = set()
    for line_no, record in iter_jsonl(run_dir / "generations.jsonl"):
        generation_ids.add(record.get("generation_id"))
        if record.get("dataset") not in {"BlogText", "Mythos-Reddit"}:
            failures.append(f"generations line {line_no}: invalid dataset")
        if record.get("method") not in valid_methods:
            failures.append(f"generations line {line_no}: invalid method {record.get('method')}")
        if record.get("max_new_tokens") != 220:
            failures.append(f"generations line {line_no}: max_new_tokens must be 220")
        if record.get("prefix_token_count") not in {96, 192}:
            failures.append(f"generations line {line_no}: invalid prefix_token_count")
        if record.get("output_word_count", 0) < 20:
            failures.append(f"generations line {line_no}: toy output below paper quality threshold")
        for old_field in ["task_" + "family", "style_" + "spec_" + "type"]:
            if old_field in record:
                failures.append(f"generations line {line_no}: obsolete field {old_field}")

    for line_no, record in iter_jsonl(run_dir / "per_example_metrics.jsonl"):
        if record.get("generation_id") not in generation_ids:
            failures.append(f"metrics line {line_no}: unknown generation_id")
        if record.get("method") not in valid_methods:
            failures.append(f"metrics line {line_no}: invalid method {record.get('method')}")
        if record.get("passed_quality_filter") and record.get("distinct_bigram_ratio", 0) < 0.15:
            failures.append(f"metrics line {line_no}: passed example below distinct-bigram threshold")
        if record.get("asce_embedding_dim") not in {None, 256}:
            failures.append(f"metrics line {line_no}: ASCE embedding dim should be 256")

    with (run_dir / "summary_metrics.csv").open("r", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        failures.append("summary_metrics.csv is empty")
    for row in rows:
        if row.get("method") not in valid_methods:
            failures.append(f"summary row invalid method {row.get('method')}")

    qf = json.loads((run_dir / "quality_filter_report.json").read_text(encoding="utf-8"))
    config = qf.get("filter_config", {})
    if config.get("min_output_words") != 20 or config.get("min_distinct_bigram_ratio") != 0.15:
        failures.append("quality_filter_report thresholds must be 20 and 0.15")

    if failures:
        print("FAIL")
        print("\n".join(failures))
        return 1

    print("PASS: run directory matches FAVoR continuation artifact checks")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
