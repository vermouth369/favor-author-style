#!/usr/bin/env python3
"""build_phaseb_main_table.py — Build the curated Exp2 main table from a full summary.

This keeps the full summary as a backend artifact while writing the paper-facing
main table with a smaller, more interpretable metric roster.
"""

import argparse
import os
import sys

import pandas as pd


DEFAULT_ROSTER = [
    "method",
    "n_texts",
    "n_authors",
    "separability_macro_f1",
    "verification_auc",
    "signed_gap_centroid_distance",
    "assistant_score_mean",
    "assistant_phrase_hit_ratio",
    "semantic_similarity",
]


def main():
    parser = argparse.ArgumentParser(description="Build curated Exp2 main table")
    parser.add_argument("--source-csv", type=str, default=None)
    parser.add_argument("--output-csv", type=str, default=None)
    parser.add_argument("--keep-extra", nargs="*", default=[])
    parser.add_argument("--round", type=int, default=None)
    args = parser.parse_args()

    exp2_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    source_csv = args.source_csv or os.path.join(exp2_dir, "results", "summary_metrics_PhaseB_Full.csv")
    output_csv = args.output_csv or os.path.join(exp2_dir, "results", "summary_metrics_PhaseB_Main.csv")

    if not os.path.exists(source_csv):
        print(f"FATAL: source CSV not found at {source_csv}")
        sys.exit(1)

    df = pd.read_csv(source_csv)
    roster = list(DEFAULT_ROSTER)
    for col in args.keep_extra:
        if col not in roster:
            roster.append(col)

    curated = pd.DataFrame()
    for col in roster:
        curated[col] = df[col] if col in df.columns else None

    if args.round is not None:
        numeric_cols = curated.select_dtypes(include=["number"]).columns
        curated.loc[:, numeric_cols] = curated.loc[:, numeric_cols].round(args.round)

    curated.to_csv(output_csv, index=False)
    print("=" * 60)
    print("Curated Exp2 Main Table")
    print("=" * 60)
    print(f"  Source: {source_csv}")
    print(f"  Output: {output_csv}")
    print(f"  Columns: {list(curated.columns)}")
    print(curated.to_string(index=False))


if __name__ == "__main__":
    main()
