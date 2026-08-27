#!/usr/bin/env python3
"""compute_verification_metrics.py — Patch Exp2 summary with verification AUC/EER.

Verification now uses the shared style scorer runtime so same-author / diff-author
pairs are measured in the same style space used elsewhere in Phase B.
"""

import argparse
import itertools
import json
import os
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import roc_auc_score, roc_curve


EXP2_ROOT = Path(__file__).resolve().parents[1]
EXP1_ROOT = Path(__file__).resolve().parents[2] / "exp1"
REPO_ROOT = Path(__file__).resolve().parents[3]
EXP1_SCRIPTS = EXP1_ROOT / "scripts"
if str(EXP1_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(EXP1_SCRIPTS))

from style_asce_runtime import ARC_FACE_BACKEND, classifier_artifact_exists, load_style_scorer


def compute_eer(fpr, tpr):
    fnr = 1.0 - tpr
    idx = np.nanargmin(np.abs(fpr - fnr))
    return float((fpr[idx] + fnr[idx]) / 2.0)


def load_generations(method_dir):
    for fname in ("generations_filtered.jsonl", "generations.jsonl"):
        path = os.path.join(method_dir, fname)
        if os.path.exists(path):
            records = []
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        records.append(json.loads(line))
            print(f"  Loaded {len(records)} records from {fname}")
            return records, fname
    return [], None


def build_author_texts(records):
    author_texts = defaultdict(list)
    for rec in records:
        author_id = rec.get("author_id", rec.get("client_id", ""))
        text = rec.get("output_text", "")
        if author_id and text and text.strip():
            author_texts[author_id].append(text)
    return {author_id: texts for author_id, texts in author_texts.items() if len(texts) >= 2}


def sample_pairs(text_indices, max_pairs_per_author, max_negative_pairs, rng):
    same_pairs = []
    for _author_id, indices in text_indices.items():
        pairs = list(itertools.combinations(indices, 2))
        if len(pairs) > max_pairs_per_author:
            pairs = rng.sample(pairs, max_pairs_per_author)
        same_pairs.extend(pairs)

    authors = list(text_indices.keys())
    diff_pairs = []
    attempts = 0
    max_attempts = max_negative_pairs * 5 if max_negative_pairs > 0 else 0
    while len(diff_pairs) < max_negative_pairs and attempts < max_attempts:
        a1, a2 = rng.sample(authors, 2)
        i1 = rng.choice(text_indices[a1])
        i2 = rng.choice(text_indices[a2])
        diff_pairs.append((i1, i2))
        attempts += 1

    return same_pairs, diff_pairs


def expand_candidate_paths(raw_path):
    path = Path(raw_path)
    candidates = [path]
    if not path.is_absolute():
        candidates.append(REPO_ROOT / path)
        candidates.append(EXP1_ROOT / path)

    deduped = []
    seen = set()
    for candidate in candidates:
        key = str(candidate)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(candidate)
    return deduped


def resolve_style_scorer(cfg, explicit_model_dir=None, legacy_embedding_arg=None):
    metrics_cfg = cfg.get("metrics", {})
    style_cfg = metrics_cfg.get("style_space", {})
    use_empirical = bool(style_cfg.get("use_empirical_prototypes", True))
    require_asce = bool(style_cfg.get("asce_required", False)) or str(
        style_cfg.get("backend", "")
    ).strip().lower() == ARC_FACE_BACKEND

    raw_candidates = []
    if explicit_model_dir:
        raw_candidates.append(explicit_model_dir)
    if legacy_embedding_arg:
        raw_candidates.append(legacy_embedding_arg)
    if style_cfg.get("authorship_model_dir"):
        raw_candidates.append(style_cfg["authorship_model_dir"])
    raw_candidates.extend(
        [
            EXP1_ROOT / "runs" / "exp1" / "K=50" / "author_classifier",
            EXP1_ROOT / "runs" / "exp1_rerun" / "K=50" / "author_classifier",
        ]
    )

    for raw_candidate in raw_candidates:
        for candidate in expand_candidate_paths(raw_candidate):
            if not classifier_artifact_exists(candidate):
                continue
            scorer = load_style_scorer(
                candidate,
                task="authorship",
                use_empirical_prototypes=use_empirical,
            )
            if require_asce and scorer.backend != ARC_FACE_BACKEND:
                print(
                    "  [style scorer] skipping non-ArcFace candidate "
                    f"backend={scorer.backend} dir={candidate}"
                )
                continue
            return scorer, str(candidate)

    if require_asce:
        raise FileNotFoundError(
            "ArcFace authorship scorer required, but no ArcFace-ready author_classifier "
            f"was found. Checked candidates: {[str(x) for x in raw_candidates]}"
        )
    raise FileNotFoundError(
        "No authorship style scorer artifacts found. "
        "Expected an ArcFace-ready author_classifier directory."
    )


def compute_method_metrics(records, style_scorer, max_pairs_per_author, max_negative_pairs, pair_seed):
    author_texts = build_author_texts(records)
    if len(author_texts) < 2:
        return None

    all_texts = []
    text_indices = defaultdict(list)
    for author_id, texts in author_texts.items():
        for text in texts:
            text_indices[author_id].append(len(all_texts))
            all_texts.append(text)

    print(f"  Encoding {len(all_texts)} texts from {len(author_texts)} authors ...")
    embeddings = style_scorer.encode_texts(all_texts, batch_size=64)
    embeddings = np.asarray(embeddings, dtype=np.float32)
    embeddings = embeddings / np.clip(np.linalg.norm(embeddings, axis=1, keepdims=True), 1.0e-8, None)

    rng = random.Random(pair_seed)
    same_pairs, diff_pairs = sample_pairs(
        text_indices,
        max_pairs_per_author=max_pairs_per_author,
        max_negative_pairs=max_negative_pairs,
        rng=rng,
    )
    if not same_pairs or not diff_pairs:
        return None

    same_sims = [float(np.dot(embeddings[i], embeddings[j])) for i, j in same_pairs]
    diff_sims = [float(np.dot(embeddings[i], embeddings[j])) for i, j in diff_pairs]

    labels = [1] * len(same_sims) + [0] * len(diff_sims)
    scores = same_sims + diff_sims

    auc = float(roc_auc_score(labels, scores))
    fpr, tpr, _ = roc_curve(labels, scores)
    eer = compute_eer(fpr, tpr)

    return {
        "verification_auc": round(auc, 4),
        "verification_eer": round(eer, 4),
        "verification_num_authors": len(author_texts),
        "verification_num_same_pairs": len(same_pairs),
        "verification_num_diff_pairs": len(diff_pairs),
        "verification_mean_same_similarity": round(float(np.mean(same_sims)), 4),
        "verification_mean_diff_similarity": round(float(np.mean(diff_sims)), 4),
    }


def patch_method_summary(method_dir, metrics):
    summary_path = os.path.join(method_dir, "summary_metrics.csv")
    if not os.path.exists(summary_path):
        return

    df = pd.read_csv(summary_path)
    for col, value in metrics.items():
        if col not in df.columns:
            df[col] = None
        df.loc[:, col] = value
    df.to_csv(summary_path, index=False)


def main():
    parser = argparse.ArgumentParser(description="Patch Exp2 summary with verification AUC/EER")
    parser.add_argument("--config", type=str, default="config/favor_main.yaml")
    parser.add_argument("--summary-csv", type=str, default=None)
    parser.add_argument("--methods", nargs="+", default=None)
    parser.add_argument("--style-model-dir", type=str, default=None)
    parser.add_argument(
        "--embedding-model",
        type=str,
        default=None,
        help="Deprecated legacy arg. If this points to a classifier dir, it will be used.",
    )
    parser.add_argument("--max-pairs-per-author", type=int, default=None)
    parser.add_argument("--max-negative-pairs", type=int, default=None)
    parser.add_argument("--pair-seed", type=int, default=42)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    runs_dir = cfg["paths"]["runs_dir"]
    summary_csv = args.summary_csv or os.path.join(EXP2_ROOT, "results", "summary_metrics_PhaseB_Full.csv")
    if not os.path.exists(summary_csv):
        print(f"FATAL: summary CSV not found at {summary_csv}")
        raise SystemExit(1)

    metrics_cfg = cfg.get("metrics", {})
    verification_cfg = metrics_cfg.get("verification", {})
    max_pairs_per_author = args.max_pairs_per_author or verification_cfg.get("max_pairs_per_author", 100)
    max_negative_pairs = args.max_negative_pairs or verification_cfg.get("max_negative_pairs", 5000)

    style_scorer, style_model_dir = resolve_style_scorer(
        cfg,
        explicit_model_dir=args.style_model_dir,
        legacy_embedding_arg=args.embedding_model,
    )

    df = pd.read_csv(summary_csv)
    methods = args.methods or list(df["method"])

    print("=" * 60)
    print(f"Verification Metrics — Patching {summary_csv}")
    print("=" * 60)
    print(f"  Style scorer backend: {style_scorer.backend}")
    print(f"  Style scorer dir: {style_model_dir}")
    print(f"  max_pairs_per_author: {max_pairs_per_author}")
    print(f"  max_negative_pairs: {max_negative_pairs}")

    results = {}
    for method in methods:
        print(f"\n{'=' * 40} {method} {'=' * 40}")
        method_dir = os.path.join(runs_dir, method)
        if not os.path.isdir(method_dir):
            print(f"  WARNING: method dir not found at {method_dir}")
            continue

        records, source_name = load_generations(method_dir)
        if not records:
            print("  No generations found, skipping")
            continue

        metrics = compute_method_metrics(
            records,
            style_scorer=style_scorer,
            max_pairs_per_author=max_pairs_per_author,
            max_negative_pairs=max_negative_pairs,
            pair_seed=args.pair_seed,
        )
        if metrics is None:
            print("  Not enough usable data for verification, skipping")
            continue

        print(
            "  verification_auc={verification_auc:.4f}, verification_eer={verification_eer:.4f} "
            "(authors={verification_num_authors}, same={verification_num_same_pairs}, "
            "diff={verification_num_diff_pairs})".format(**metrics)
        )
        results[method] = metrics

        metrics_path = os.path.join(method_dir, "metrics_verification.json")
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    **metrics,
                    "source_generations": source_name,
                    "style_scorer_backend": style_scorer.backend,
                    "style_model_dir": style_model_dir,
                    "style_reference_source": getattr(style_scorer, "reference_source", None),
                    "max_pairs_per_author": max_pairs_per_author,
                    "max_negative_pairs": max_negative_pairs,
                    "pair_seed": args.pair_seed,
                },
                f,
                indent=2,
            )
        patch_method_summary(method_dir, metrics)

    if not results:
        print("\nNo verification metrics computed.")
        return

    if args.dry_run:
        print("\n[DRY RUN] Not modifying CSV.")
        return

    patch_cols = [
        "verification_auc",
        "verification_eer",
        "verification_num_authors",
        "verification_num_same_pairs",
        "verification_num_diff_pairs",
    ]
    for col in patch_cols:
        if col not in df.columns:
            df[col] = None

    for method, metrics in results.items():
        mask = df["method"] == method
        if mask.sum() == 0:
            print(f"  WARNING: {method} not found in CSV, skipping patch")
            continue
        for col in patch_cols:
            df.loc[mask, col] = metrics[col]

    df.to_csv(summary_csv, index=False)
    print(f"\n✓ Patched CSV saved to {summary_csv}")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
