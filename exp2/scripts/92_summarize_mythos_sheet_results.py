#!/usr/bin/env python3
"""Summarize Mythos non-FL baseline runs and optional metric outputs."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from mythos_sheet_utils import config_get, load_yaml_config


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default=None, help="Optional YAML config; CLI values take precedence.")
    p.add_argument("--run_root", default=None)
    p.add_argument("--methods", nargs="+", default=None)
    p.add_argument("--output_csv", default=None)
    p.add_argument("--favor_summary_csv", default=None, help="Optional existing FAVoR/FL summary to append for comparison.")
    args = p.parse_args()
    if args.config:
        cfg = load_yaml_config(args.config)
        args.run_root = args.run_root or config_get(cfg, "paths", "runs_dir")
        args.methods = args.methods or config_get(cfg, "methods")
        if not args.output_csv and args.run_root:
            args.output_csv = str(Path(args.run_root) / "summary_mythos_sheet.csv")
    if not args.run_root:
        raise ValueError("Missing --run_root; pass it directly or via paths.runs_dir in --config.")
    if not args.methods:
        raise ValueError("Missing --methods; pass them directly or via methods in --config.")
    if not args.output_csv:
        raise ValueError("Missing --output_csv; pass it directly or provide --config with paths.runs_dir.")
    return args


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def read_summary_csv(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return rows[0] if rows else {}


def read_method_metrics(method_dir: Path) -> dict[str, Any]:
    candidates = [
        method_dir / "summary_metrics.csv",
        method_dir / "summary_metrics_continuation.csv",
        method_dir / "summary_metrics.continuation.csv",
    ]
    candidates.extend(sorted(method_dir.glob("summary_metrics.continuation.*.csv")))
    for path in candidates:
        metrics = read_summary_csv(path)
        if metrics:
            return metrics
    return {}


def count_jsonl(path: Path) -> int:
    if not path.exists():
        return 0
    with open(path, "r", encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())


def main() -> None:
    args = parse_args()
    run_root = Path(args.run_root)
    rows = []
    for method in args.methods:
        method_dir = run_root / method
        qf = read_json(method_dir / "quality_filter_report.json")
        audit = read_json(method_dir / "leakage_audit.json")
        metrics = read_method_metrics(method_dir)
        sheet_report = read_json(method_dir / "sheet_quality_report.json")
        if not sheet_report:
            sheet_report = read_json(run_root / "mythos_sheet" / "sheet_quality_report.json")
        if not sheet_report:
            sheet_report = read_json(run_root / "sheet_quality_report.json")
        total = qf["total"] if "total" in qf else count_jsonl(method_dir / "generations.jsonl")
        passed = qf["passed"] if "passed" in qf else count_jsonl(method_dir / "generations_filtered.jsonl")
        uses_rag_demo = method in {"rag_only", "mythos_sheet", "mythos_sheet_np"}
        uses_distilled_artifact = method in {"mythos_sheet", "mythos_sheet_np", "mythos_sheet_no_rag"}
        row = {
            "method": method,
            "FL": "No",
            "uses_raw_author_history_at_inference": "Yes" if uses_rag_demo else "No",
            "uses_rag_demo_at_inference": "Yes" if uses_rag_demo else "No",
            "uses_distilled_author_artifact_at_inference": "Yes" if uses_distilled_artifact else "No",
            "training": "No",
            "num_generations": total,
            "pass_rate": (float(passed) / total) if total else "",
            "leakage_passed": audit.get("passed", ""),
            "num_retrieval_violations": audit.get("num_retrieval_violations", ""),
            "num_gold_string_prompt_hits": audit.get("num_gold_string_prompt_hits", ""),
            "num_bm25_no_positive_match": audit.get("num_bm25_no_positive_match", ""),
            "sheet_coverage": sheet_report.get("sheet_coverage", ""),
            "valid_evidence_rate": sheet_report.get("valid_evidence_rate", ""),
            "avg_claims_per_author": sheet_report.get("avg_claims_per_author", ""),
        }
        metric_aliases = {
            "authorship_accuracy": ("authorship_accuracy", "separability_acc"),
            "authorship_macro_f1": ("authorship_macro_f1", "separability_macro_f1"),
            "macro_f1": ("macro_f1", "separability_macro_f1"),
            "silhouette_score": ("silhouette_score", "silhouette"),
            "mean_centroid_cosine_distance": ("mean_centroid_cosine_distance", "centroid_distance"),
            "assistant_score_mean": ("assistant_score_mean",),
            "bertscore_f1": ("bertscore_f1", "continuation_bertscore_f1"),
            "semantic_similarity": ("semantic_similarity", "continuation_semantic_similarity"),
            "mauve": ("mauve", "mauve_score"),
        }
        for out_key, aliases in metric_aliases.items():
            for key in aliases:
                if key in metrics:
                    row[out_key] = metrics[key]
                    break
        for key in (
            "n_texts",
            "n_authors",
            "n_continuation_ids",
            "collapse_index",
            "signed_gap_silhouette",
            "signed_gap_centroid_distance",
            "authorship_accuracy",
            "authorship_macro_f1",
            "authorship_margin_mean",
            "authorship_top1_cosine_mean",
            "assistant_phrase_rate",
            "assistant_phrase_hit_ratio",
        ):
            if key in metrics:
                row[key] = metrics[key]
        rows.append(row)

    if args.favor_summary_csv:
        favor_path = Path(args.favor_summary_csv)
        if favor_path.exists():
            with open(favor_path, "r", encoding="utf-8") as f:
                for rec in csv.DictReader(f):
                    rec = dict(rec)
                    rec.setdefault("FL", "Yes")
                    rec.setdefault("uses_raw_author_history_at_inference", "No raw sharing")
                    rec.setdefault("uses_rag_demo_at_inference", "No")
                    rec.setdefault("uses_distilled_author_artifact_at_inference", "No")
                    rec.setdefault("training", "Yes")
                    rows.append(rec)

    fieldnames = sorted({key for row in rows for key in row.keys()})
    preferred = [
        "method",
        "FL",
        "uses_raw_author_history_at_inference",
        "uses_rag_demo_at_inference",
        "uses_distilled_author_artifact_at_inference",
        "training",
        "num_generations",
        "pass_rate",
        "leakage_passed",
        "num_retrieval_violations",
        "num_gold_string_prompt_hits",
        "num_bm25_no_positive_match",
        "sheet_coverage",
        "valid_evidence_rate",
        "avg_claims_per_author",
        "n_texts",
        "n_authors",
        "n_continuation_ids",
        "authorship_accuracy",
        "authorship_macro_f1",
        "macro_f1",
        "authorship_margin_mean",
        "authorship_top1_cosine_mean",
        "silhouette_score",
        "mean_centroid_cosine_distance",
        "collapse_index",
        "signed_gap_silhouette",
        "signed_gap_centroid_distance",
        "assistant_score_mean",
        "assistant_phrase_rate",
        "assistant_phrase_hit_ratio",
        "bertscore_f1",
        "semantic_similarity",
        "mauve",
    ]
    fieldnames = [f for f in preferred if f in fieldnames] + [f for f in fieldnames if f not in preferred]
    Path(args.output_csv).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote summary to {args.output_csv}")


if __name__ == "__main__":
    main()
