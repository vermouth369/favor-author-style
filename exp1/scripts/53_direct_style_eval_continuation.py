#!/usr/bin/env python3
"""53_direct_style_eval_continuation.py — Phase B Direct Style Eval Pack.

Implements components B1–B6 from phaseA_phaseB_execution_spec.revised.md §4.5.
Runs on Phase-A-cleaned held-out-continuation generations.

Components:
    B1: Style/content disentanglement table
    B2: Content-controlled style retrieval
    B3: Continuation-slice breakdown
    B4: Same-author vs different-author margin analysis
    B5: Assistantness disentanglement
    B6: Content-leakage stress test slices

Outputs:
    direct_style_eval_summary.csv
    direct_style_eval_by_slice.csv
    style_content_disentanglement_report.md
    assistantness_vs_style_report.md
"""

import argparse
import json
import os
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy import stats
from sklearn.metrics.pairwise import cosine_similarity

from favor_helpers import resolve_run_client_roster
from style_asce_runtime import (
    ARC_FACE_BACKEND,
    classifier_artifact_exists,
    load_style_scorer,
)


# ============================================================
# Scorer helpers
# ============================================================

def expand_candidate_paths(base_dir, raw_path):
    """Resolve config-declared model dirs against repo-relative anchors."""
    if raw_path is None:
        return []
    path = Path(raw_path)
    candidates = [path]
    if not path.is_absolute():
        candidates.extend([
            Path(base_dir) / path,
            Path(base_dir).parent / path,
            Path(base_dir).parent.parent / path,
        ])
    deduped = []
    seen = set()
    for candidate in candidates:
        key = str(candidate)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(str(candidate))
    return deduped


def encode_style_texts(scorer, texts, batch_size=64):
    """Encode texts for style similarity using the loaded scorer.

    Uses style_asce_runtime.BaseStyleScorer.encode_texts when available.
    """
    return scorer.encode_texts(texts, batch_size=batch_size)


def asce_required(cfg_section: dict) -> bool:
    """Whether this config section requires an ArcFace-backed scorer."""
    backend = str(cfg_section.get("backend", "")).strip().lower()
    return bool(cfg_section.get("asce_required", False)) or backend == ARC_FACE_BACKEND


# ============================================================
# Data loading
# ============================================================

def load_continuation_generations(method_dir):
    """Load continuation-protocol generation records."""
    filtered_path = os.path.join(method_dir, "generations_filtered.jsonl")
    raw_path = os.path.join(method_dir, "generations.jsonl")
    path = filtered_path if os.path.exists(filtered_path) else raw_path
    records = []
    if os.path.exists(path):
        with open(path, "r") as f:
            for line in f:
                if line.strip():
                    records.append(json.loads(line))
    return records


def load_author_history(pooled_dir, split="train"):
    """Load author texts from pooled split for history anchors."""
    path = os.path.join(pooled_dir, f"{split}.jsonl")
    author_texts = defaultdict(list)
    if os.path.exists(path):
        with open(path, "r") as f:
            for line in f:
                if line.strip():
                    rec = json.loads(line)
                    author_texts[rec.get("author_id", "")].append(rec.get("text", ""))
    return dict(author_texts)


# ============================================================
# Lexical overlap utilities
# ============================================================

def compute_lexical_overlap(text_a, text_b):
    """Compute word-level Jaccard overlap between two texts."""
    words_a = set(text_a.lower().split())
    words_b = set(text_b.lower().split())
    if not words_a or not words_b:
        return 0.0
    return len(words_a & words_b) / len(words_a | words_b)


def compute_ngram_overlap(text_a, text_b, n=2):
    """Compute n-gram overlap (Jaccard) between two texts."""
    words_a = text_a.lower().split()
    words_b = text_b.lower().split()
    if len(words_a) < n or len(words_b) < n:
        return 0.0
    ngrams_a = set(tuple(words_a[i:i+n]) for i in range(len(words_a) - n + 1))
    ngrams_b = set(tuple(words_b[i:i+n]) for i in range(len(words_b) - n + 1))
    if not ngrams_a or not ngrams_b:
        return 0.0
    return len(ngrams_a & ngrams_b) / len(ngrams_a | ngrams_b)


# ============================================================
# B1: Style/content disentanglement table
# ============================================================

def compute_b1_disentanglement(records, scorer, author_anchors, semantic_model):
    """Compute style/content/assistantness separately for each record.

    Returns a DataFrame with per-example columns:
        style_sim_to_author, semantic_sim_to_reference, lexical_overlap_to_reference,
        continuation_fidelity
    """
    from sentence_transformers import SentenceTransformer

    rows = []
    output_texts = [r["output_text"] for r in records]
    reference_texts = [r.get("reference_continuation_text", "") for r in records]

    # Style embeddings for outputs (via ArcFace scorer)
    output_style_embs = encode_style_texts(scorer, output_texts)

    # Semantic embeddings for outputs and references
    output_sem_embs = semantic_model.encode(output_texts, show_progress_bar=False, batch_size=64)
    ref_sem_embs = semantic_model.encode(reference_texts, show_progress_bar=False, batch_size=64)

    for idx, rec in enumerate(records):
        author_id = rec.get("owner_author_id", "")
        ref_text = rec.get("reference_continuation_text", "")
        out_text = rec["output_text"]

        # Style similarity to author history anchor
        style_sim = None
        if author_id in author_anchors:
            anchor = author_anchors[author_id].reshape(1, -1)
            style_sim = float(cosine_similarity(
                output_style_embs[idx:idx+1], anchor
            )[0, 0])

        # Semantic similarity to reference continuation
        semantic_sim = float(cosine_similarity(
            output_sem_embs[idx:idx+1], ref_sem_embs[idx:idx+1]
        )[0, 0])

        # Lexical overlap with reference
        lexical_overlap = compute_lexical_overlap(out_text, ref_text)
        bigram_overlap = compute_ngram_overlap(out_text, ref_text, n=2)

        rows.append({
            "continuation_id": rec.get("continuation_id"),
            "method": rec.get("method"),
            "owner_author_id": author_id,
            "owner_client_id": rec.get("owner_client_id"),
            "prefix_token_count": rec.get("prefix_token_count"),
            "reference_token_count": rec.get("reference_token_count"),
            "style_sim_to_author": round(style_sim, 6) if style_sim is not None else None,
            "semantic_sim_to_reference": round(semantic_sim, 6),
            "lexical_overlap_to_reference": round(lexical_overlap, 6),
            "bigram_overlap_to_reference": round(bigram_overlap, 6),
        })

    return pd.DataFrame(rows)


# ============================================================
# B3: Continuation-slice breakdown
# ============================================================

def assign_slices(df):
    """Assign slice labels to each example based on prefix length, continuation length, overlap."""
    # Prefix length slices
    prefix_median = df["prefix_token_count"].median()
    df["prefix_slice"] = df["prefix_token_count"].apply(
        lambda x: "long_prefix" if x >= prefix_median else "short_prefix"
    )

    # Continuation length slices
    ref_median = df["reference_token_count"].median()
    df["continuation_slice"] = df["reference_token_count"].apply(
        lambda x: "long_continuation" if x >= ref_median else "short_continuation"
    )

    # Lexical overlap slices
    overlap_median = df["lexical_overlap_to_reference"].median()
    df["overlap_slice"] = df["lexical_overlap_to_reference"].apply(
        lambda x: "high_overlap" if x >= overlap_median else "low_overlap"
    )

    return df


def compute_b3_slice_breakdown(df):
    """Compute metrics by slice."""
    slice_columns = ["prefix_slice", "continuation_slice", "overlap_slice"]
    results = []

    for slice_col in slice_columns:
        for slice_val, group in df.groupby(slice_col):
            row = {
                "slice_type": slice_col,
                "slice_value": slice_val,
                "n": len(group),
            }
            if "style_sim_to_author" in group.columns:
                valid = group["style_sim_to_author"].dropna()
                row["style_sim_mean"] = round(float(valid.mean()), 6) if len(valid) else None
                row["style_sim_std"] = round(float(valid.std()), 6) if len(valid) else None
            if "semantic_sim_to_reference" in group.columns:
                row["semantic_sim_mean"] = round(float(group["semantic_sim_to_reference"].mean()), 6)
            if "lexical_overlap_to_reference" in group.columns:
                row["lexical_overlap_mean"] = round(float(group["lexical_overlap_to_reference"].mean()), 6)
            results.append(row)

    return pd.DataFrame(results)


# ============================================================
# B4: Same-author vs different-author margin analysis
# ============================================================

def compute_b4_author_margin(records, scorer, author_anchors):
    """For each method, estimate same-author vs different-author style margin."""
    all_authors = sorted(author_anchors.keys())
    if len(all_authors) < 2:
        return {}

    anchor_matrix = np.stack([author_anchors[a] for a in all_authors])
    author_idx = {a: i for i, a in enumerate(all_authors)}

    output_texts = [r["output_text"] for r in records]
    output_embs = encode_style_texts(scorer, output_texts)

    same_sims = []
    diff_sims = []
    ranks = []

    for idx, rec in enumerate(records):
        author_id = rec.get("owner_author_id", "")
        if author_id not in author_idx:
            continue

        sims = cosine_similarity(output_embs[idx:idx+1], anchor_matrix)[0]
        true_idx = author_idx[author_id]
        same_sim = float(sims[true_idx])
        same_sims.append(same_sim)

        other_sims = [sims[j] for j in range(len(all_authors)) if j != true_idx]
        diff_sims.append(float(np.mean(other_sims)))

        ranking = np.argsort(-sims)
        rank = int(np.where(ranking == true_idx)[0][0]) + 1
        ranks.append(rank)

    same_arr = np.array(same_sims)
    diff_arr = np.array(diff_sims)
    margin = same_arr - diff_arr

    result = {
        "same_author_sim_mean": round(float(same_arr.mean()), 6),
        "diff_author_sim_mean": round(float(diff_arr.mean()), 6),
        "margin_mean": round(float(margin.mean()), 6),
        "margin_std": round(float(margin.std()), 6),
        "target_author_mean_rank": round(float(np.mean(ranks)), 4),
        "target_author_top1_rate": round(float(np.mean([1 if r == 1 else 0 for r in ranks])), 6),
        "n_examples": len(same_sims),
        "n_authors": len(all_authors),
    }
    return result


# ============================================================
# B5: Assistantness disentanglement
# ============================================================

def compute_b5_assistantness(records, df_disentangle, assistant_scorer, assistant_phrases_path):
    """Separate style-retention vs assistantness scores.

    Preferred assistantness signal is the calibrated assistant scorer. Phrase hits are
    retained as a lexical side-channel diagnostic.
    """
    phrases = []
    if os.path.exists(assistant_phrases_path):
        with open(assistant_phrases_path, "r") as f:
            phrases = [line.strip().lower() for line in f if line.strip()]

    assistant_scores = [None] * len(records)
    assistant_raw_margins = [None] * len(records)
    assistant_backend = None
    assistant_calibrated = False
    if assistant_scorer is not None and records:
        output_texts = [rec["output_text"] for rec in records]
        assistant_pred = assistant_scorer.predict_assistant(output_texts, batch_size=64)
        score_arr = assistant_pred.get("assistant_score")
        if score_arr is not None and len(score_arr) == len(records):
            assistant_scores = [round(float(x), 6) for x in score_arr.tolist()]
        raw_margin_arr = assistant_pred.get("raw_margin")
        if raw_margin_arr is not None and len(raw_margin_arr) == len(records):
            assistant_raw_margins = [round(float(x), 6) for x in raw_margin_arr.tolist()]
        assistant_backend = assistant_scorer.backend
        assistant_calibrated = bool(getattr(assistant_scorer, "calibrator", None) is not None)

    rows = []
    for idx, rec in enumerate(records):
        text = rec["output_text"].lower()
        words = text.split()
        hit_count = sum(text.count(ph) for ph in phrases)

        style_sim = None
        if idx < len(df_disentangle):
            style_sim = df_disentangle.iloc[idx].get("style_sim_to_author")

        rows.append({
            "continuation_id": rec.get("continuation_id"),
            "method": rec.get("method"),
            "owner_author_id": rec.get("owner_author_id"),
            "style_sim_to_author": style_sim,
            "assistant_score": assistant_scores[idx],
            "assistant_raw_margin": assistant_raw_margins[idx],
            "assistant_score_backend": assistant_backend,
            "assistant_score_calibrated": assistant_calibrated,
            "assistant_phrase_hits": hit_count,
            "assistant_phrase_hit": hit_count > 0,
            "num_words": len(words),
        })

    df = pd.DataFrame(rows)

    return df


# ============================================================
# B6: Content leakage stress tests
# ============================================================

def compute_b6_leakage_slices(df):
    """Identify stress-test slices where content leakage is especially likely."""
    results = {}

    if "lexical_overlap_to_reference" not in df.columns:
        return results

    # Very high lexical overlap
    p90 = df["lexical_overlap_to_reference"].quantile(0.9)
    high_overlap = df[df["lexical_overlap_to_reference"] >= p90]
    low_overlap = df[df["lexical_overlap_to_reference"] < df["lexical_overlap_to_reference"].median()]

    if len(high_overlap) > 0 and "style_sim_to_author" in df.columns:
        valid_high = high_overlap["style_sim_to_author"].dropna()
        valid_low = low_overlap["style_sim_to_author"].dropna()

        results["high_overlap_slice"] = {
            "n_examples": len(high_overlap),
            "overlap_threshold_p90": round(float(p90), 4),
            "style_sim_mean": round(float(valid_high.mean()), 6) if len(valid_high) else None,
        }
        results["low_overlap_slice"] = {
            "n_examples": len(low_overlap),
            "style_sim_mean": round(float(valid_low.mean()), 6) if len(valid_low) else None,
        }
        if len(valid_high) > 0 and len(valid_low) > 0:
            results["style_sim_delta_high_minus_low"] = round(
                float(valid_high.mean() - valid_low.mean()), 6
            )

    # High embedding similarity to reference (semantic)
    if "semantic_sim_to_reference" in df.columns:
        sem_p90 = df["semantic_sim_to_reference"].quantile(0.9)
        high_sem = df[df["semantic_sim_to_reference"] >= sem_p90]
        results["high_semantic_overlap_slice"] = {
            "n_examples": len(high_sem),
            "semantic_threshold_p90": round(float(sem_p90), 4),
        }
        if len(high_sem) > 0 and "style_sim_to_author" in df.columns:
            valid = high_sem["style_sim_to_author"].dropna()
            results["high_semantic_overlap_slice"]["style_sim_mean"] = (
                round(float(valid.mean()), 6) if len(valid) else None
            )

    return results


# ============================================================
# Report generation
# ============================================================

def generate_disentanglement_report(method_summaries, slice_df, leakage_results, output_path):
    """Generate style_content_disentanglement_report.md."""
    lines = [
        "# Style / Content Disentanglement Report",
        "",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## Method Summary",
        "",
        "| Method | Style Sim (author) | Semantic Sim (ref) | Lexical Overlap | Margin |",
        "|--------|-------------------|--------------------|-----------------|--------|",
    ]

    for method, summary in sorted(method_summaries.items()):
        lines.append(
            f"| {method} "
            f"| {summary.get('style_sim_mean', '-')} "
            f"| {summary.get('semantic_sim_mean', '-')} "
            f"| {summary.get('lexical_overlap_mean', '-')} "
            f"| {summary.get('margin_mean', '-')} |"
        )

    lines.extend(["", "## Continuation-Slice Breakdown", ""])
    if slice_df is not None and len(slice_df) > 0:
        lines.append(slice_df.to_markdown(index=False))
    else:
        lines.append("No slice data available.")

    lines.extend(["", "## Content Leakage Stress Tests", ""])
    if leakage_results:
        for key, val in leakage_results.items():
            lines.append(f"### {key}")
            lines.append(f"```json\n{json.dumps(val, indent=2)}\n```")
            lines.append("")

    lines.extend([
        "",
        "## Questions Answered",
        "",
        "1. **Which style conclusions remain stable under continuation-content control?** "
        "See method summary and slice breakdown above.",
        "2. **Which methods gain mainly on continuation matching rather than style?** "
        "Compare style_sim vs semantic_sim columns.",
        "3. **Which continuation slices are reliable for direct style evaluation?** "
        "Low-overlap slices are more reliable.",
        "4. **Where do assistantness and style retention diverge?** "
        "See assistantness_vs_style_report.md.",
        "5. **Does M1 still show an advantage after content-sensitive checks?** "
        "Compare M1 row vs other methods in the summary table.",
        "",
        "---",
        "",
        "*Generated by 53_direct_style_eval_continuation.py*",
    ])

    with open(output_path, "w") as f:
        f.write("\n".join(lines))
    return output_path


def generate_assistantness_report(asst_dfs, output_path):
    """Generate assistantness_vs_style_report.md."""
    lines = [
        "# Assistantness vs Style Report",
        "",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
    ]

    for method, df in sorted(asst_dfs.items()):
        lines.extend([f"## {method}", ""])

        if df.empty:
            lines.append("No data available.")
            lines.append("")
            continue

        valid = df.dropna(subset=["style_sim_to_author"])
        if valid.empty:
            lines.append("No valid style similarity data.")
            lines.append("")
            continue

        if "assistant_score" in valid.columns:
            score_valid = valid.dropna(subset=["assistant_score"])
        else:
            score_valid = pd.DataFrame()

        if len(score_valid) > 2:
            corr, p_val = stats.pearsonr(
                score_valid["style_sim_to_author"],
                score_valid["assistant_score"],
            )
            backend = score_valid["assistant_score_backend"].dropna().iloc[0] if score_valid["assistant_score_backend"].notna().any() else None
            calibrated = (
                bool(score_valid["assistant_score_calibrated"].dropna().iloc[0])
                if score_valid["assistant_score_calibrated"].notna().any()
                else False
            )
            lines.append(
                f"- Style-assistantness correlation: r={corr:.4f}, p={p_val:.6f} "
                f"(signal=assistant_score, backend={backend}, calibrated={calibrated})"
            )
            lines.append(
                f"- Mean assistant_score: {score_valid['assistant_score'].mean():.6f} "
                f"(n={len(score_valid)})"
            )
        elif "assistant_phrase_hits" in valid.columns and len(valid) > 2:
            corr, p_val = stats.pearsonr(
                valid["style_sim_to_author"],
                valid["assistant_phrase_hits"],
            )
            lines.append(
                f"- Style-assistantness correlation: r={corr:.4f}, p={p_val:.6f} "
                "(signal=assistant_phrase_hits fallback)"
            )

        # Split by assistant hit
        with_hit = valid[valid["assistant_phrase_hit"] == True]
        without_hit = valid[valid["assistant_phrase_hit"] == False]
        if len(with_hit) > 0 and len(without_hit) > 0:
            lines.append(
                f"- Style sim (with assistant phrases): "
                f"{with_hit['style_sim_to_author'].mean():.6f} (n={len(with_hit)})"
            )
            lines.append(
                f"- Style sim (without assistant phrases): "
                f"{without_hit['style_sim_to_author'].mean():.6f} (n={len(without_hit)})"
            )

        lines.append("")

    lines.extend([
        "---",
        "",
        "*Generated by 53_direct_style_eval_continuation.py*",
    ])

    with open(output_path, "w") as f:
        f.write("\n".join(lines))
    return output_path


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Phase B Direct Style Eval Pack (continuation protocol)"
    )
    parser.add_argument("--config", type=str, default="config/favor_main.yaml")
    parser.add_argument(
        "--methods", nargs="+",
        default=["B0_base", "B1_local_only", "F1_fedavg_shared",
                 "F2_fedprox", "F3_ditto_peft", "M1_favor"],
    )
    parser.add_argument("--device", type=str, default=None,
                        help="Device for model inference (e.g. cuda:0)")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    runs_dir = cfg["paths"]["runs_dir"]
    pooled_dir = cfg["paths"].get("pooled_dir", "../exp1/data/pooled/K=50")
    phrases_file = cfg["paths"].get("assistant_phrases_file", "../exp1/metrics/assistant_phrases.txt")

    print("=" * 70)
    print("Step 53 — Phase B Direct Style Eval Pack (Continuation Protocol)")
    print("=" * 70)

    # ---- Load style scorer via style_asce_runtime ----
    exp1_base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    metrics_cfg = cfg.get("metrics", {})
    style_space_cfg = metrics_cfg.get("style_space", {})
    use_empirical_prototypes = bool(style_space_cfg.get("use_empirical_prototypes", True))
    asce_required_style = asce_required(style_space_cfg)

    authorship_candidates = []
    if style_space_cfg.get("authorship_model_dir"):
        authorship_candidates.extend(
            expand_candidate_paths(exp1_base, style_space_cfg["authorship_model_dir"])
        )
    authorship_candidates.extend([
        os.path.join(exp1_base, "runs", "exp1_asce_full", "K=50", "author_classifier"),
        os.path.join(exp1_base, "runs", "exp1", "K=50", "author_classifier"),
        os.path.join(exp1_base, "runs", "exp1_rerun", "K=50", "author_classifier"),
    ])

    scorer = None
    scorer_dir = None
    for candidate_dir in authorship_candidates:
        if not classifier_artifact_exists(candidate_dir):
            continue
        try:
            scorer = load_style_scorer(
                candidate_dir, task="authorship",
                use_empirical_prototypes=use_empirical_prototypes,
            )
            if asce_required_style and scorer.backend != ARC_FACE_BACKEND:
                print(
                    "  [Style scorer] skipping non-ArcFace candidate "
                    f"backend={scorer.backend} dir={candidate_dir}"
                )
                scorer = None
                continue
            scorer_dir = candidate_dir
            break
        except Exception as exc:
            print(f"  WARNING: failed to load scorer from {candidate_dir}: {exc}")

    if scorer is None:
        requirement_msg = "ArcFace authorship scorer" if asce_required_style else "authorship scorer"
        raise SystemExit(f"FATAL: No {requirement_msg} found. Cannot run Phase B.")

    print(f"  [Style scorer] backend={scorer.backend} dir={scorer_dir}")

    # ---- Load assistant scorer via style_asce_runtime ----
    assistant_scoring_cfg = metrics_cfg.get("assistant_scoring", {})
    asce_required_assistant = asce_required(assistant_scoring_cfg)
    assistant_candidates = []
    if assistant_scoring_cfg.get("assistant_model_dir"):
        assistant_candidates.extend(
            expand_candidate_paths(exp1_base, assistant_scoring_cfg["assistant_model_dir"])
        )
    assistant_candidates.extend([
        os.path.join(exp1_base, "runs", "exp1_asce_full", "assistant_classifier"),
        os.path.join(exp1_base, "runs", "exp1", "assistant_classifier"),
        os.path.join(exp1_base, "runs", "exp1_rerun", "assistant_classifier"),
    ])

    assistant_scorer = None
    assistant_scorer_dir = None
    for candidate_dir in assistant_candidates:
        if not classifier_artifact_exists(candidate_dir):
            continue
        try:
            candidate_scorer = load_style_scorer(candidate_dir, task="assistant")
            if asce_required_assistant and candidate_scorer.backend != ARC_FACE_BACKEND:
                print(
                    "  [Assistant scorer] skipping non-ArcFace candidate "
                    f"backend={candidate_scorer.backend} dir={candidate_dir}"
                )
                continue
            assistant_scorer = candidate_scorer
            assistant_scorer_dir = candidate_dir
            break
        except Exception as exc:
            print(f"  WARNING: failed to load assistant scorer from {candidate_dir}: {exc}")

    if assistant_scorer is None and asce_required_assistant:
        raise SystemExit(
            "FATAL: No ArcFace assistant scorer found. Cannot run assistantness "
            f"disentanglement. Checked candidates: {assistant_candidates}"
        )

    if assistant_scorer is not None:
        print(f"  [Assistant scorer] backend={assistant_scorer.backend} dir={assistant_scorer_dir}")
    else:
        print("  [Assistant scorer] unavailable; B5 will fall back to phrase diagnostics only.")

    # Load semantic model (MiniLM — only for content/faithfulness similarity)
    from sentence_transformers import SentenceTransformer
    semantic_model_name = (
        metrics_cfg.get("semantic_space", {}).get("embedding_model")
        or metrics_cfg.get("embedding_model")
        or "sentence-transformers/all-MiniLM-L6-v2"
    )
    semantic_model = SentenceTransformer(semantic_model_name)
    print(f"  [Semantic model] {semantic_model_name} (content/faithfulness only)")

    # Load author history for anchors
    print(f"  Loading author history from {pooled_dir}/train.jsonl ...")
    author_history = load_author_history(pooled_dir, split="train")
    print(f"    {len(author_history)} authors loaded")

    # Build author anchors using the STYLE scorer (ArcFace when available)
    print(f"  Building author style anchors (backend={scorer.backend}) ...")
    author_anchors = {}
    for author_id, texts in author_history.items():
        if texts:
            embs = encode_style_texts(scorer, texts[:50], batch_size=64)
            author_anchors[author_id] = embs.mean(axis=0)
    print(f"    {len(author_anchors)} author anchors built")

    # Process each method
    method_summaries = {}
    all_slice_dfs = []
    all_asst_dfs = {}
    all_leakage_results = {}
    all_disentangle_dfs = []

    for method in args.methods:
        print(f"\n{'=' * 40} {method} {'=' * 40}")

        method_dir = os.path.join(runs_dir, method)
        records = load_continuation_generations(method_dir)

        if not records:
            print(f"  No generations found for {method}, skipping")
            continue

        # Validate protocol
        non_continuation = sum(
            1 for r in records if r.get("prompt_protocol") != "heldout_continuation_v1"
        )
        if non_continuation > 0:
            print(f"  WARNING: {non_continuation} records not using continuation protocol")

        # B1: Disentanglement table
        print(f"  [B1] Computing style/content disentanglement ...")
        df_dis = compute_b1_disentanglement(records, scorer, author_anchors, semantic_model)
        df_dis["method"] = method
        all_disentangle_dfs.append(df_dis)

        summary = {
            "method": method,
            "n_examples": len(df_dis),
        }
        if "style_sim_to_author" in df_dis.columns:
            valid = df_dis["style_sim_to_author"].dropna()
            summary["style_sim_mean"] = round(float(valid.mean()), 6) if len(valid) else None
        summary["semantic_sim_mean"] = round(float(df_dis["semantic_sim_to_reference"].mean()), 6)
        summary["lexical_overlap_mean"] = round(float(df_dis["lexical_overlap_to_reference"].mean()), 6)

        # B3: Slice breakdown
        print(f"  [B3] Computing continuation-slice breakdown ...")
        df_sliced = assign_slices(df_dis)
        slice_df = compute_b3_slice_breakdown(df_sliced)
        slice_df["method"] = method
        all_slice_dfs.append(slice_df)

        # B4: Author margin
        print(f"  [B4] Computing same-vs-different author margin ...")
        margin_result = compute_b4_author_margin(records, scorer, author_anchors)
        summary.update({
            "margin_mean": margin_result.get("margin_mean"),
            "target_author_mean_rank": margin_result.get("target_author_mean_rank"),
            "target_author_top1_rate": margin_result.get("target_author_top1_rate"),
        })

        # B5: Assistantness
        print(f"  [B5] Computing assistantness disentanglement ...")
        asst_df = compute_b5_assistantness(records, df_dis, assistant_scorer, phrases_file)
        all_asst_dfs[method] = asst_df

        # B6: Leakage slices
        print(f"  [B6] Computing content-leakage stress slices ...")
        leakage = compute_b6_leakage_slices(df_dis)
        all_leakage_results[method] = leakage

        method_summaries[method] = summary
        summary["style_scorer_backend"] = scorer.backend
        summary["style_scorer_dir"] = scorer_dir
        print(f"  ✓ {method} complete")

    # Write outputs
    os.makedirs(runs_dir, exist_ok=True)

    # Summary CSV
    summary_df = pd.DataFrame(list(method_summaries.values()))
    summary_path = os.path.join(runs_dir, "direct_style_eval_summary.csv")
    summary_df.to_csv(summary_path, index=False)
    print(f"\n✓ Summary: {summary_path}")

    # Slice CSV
    if all_slice_dfs:
        combined_slices = pd.concat(all_slice_dfs, ignore_index=True)
        slice_path = os.path.join(runs_dir, "direct_style_eval_by_slice.csv")
        combined_slices.to_csv(slice_path, index=False)
        print(f"✓ Slices: {slice_path}")
    else:
        combined_slices = None

    # Disentanglement report
    report_path = os.path.join(runs_dir, "style_content_disentanglement_report.md")
    generate_disentanglement_report(
        method_summaries, combined_slices, all_leakage_results, report_path,
    )
    print(f"✓ Disentanglement report: {report_path}")

    # Assistantness report
    asst_report_path = os.path.join(runs_dir, "assistantness_vs_style_report.md")
    generate_assistantness_report(all_asst_dfs, asst_report_path)
    print(f"✓ Assistantness report: {asst_report_path}")

    # Full disentanglement data
    if all_disentangle_dfs:
        full_df = pd.concat(all_disentangle_dfs, ignore_index=True)
        full_path = os.path.join(runs_dir, "direct_style_eval_full.csv")
        full_df.to_csv(full_path, index=False)
        print(f"✓ Full data: {full_path}")

    print("\n✓ Phase B Direct Style Eval Pack complete.")


if __name__ == "__main__":
    main()
