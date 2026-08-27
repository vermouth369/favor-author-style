#!/usr/bin/env python3
"""47b_compute_metrics_continuation.py — Metrics for held-out continuation protocol.

Adapted from 47_compute_metrics_prototype.py for the Phase A held-out-continuation
protocol.  Key differences:

  - Joins on continuation_id (not prompt_id)
  - Reads exact conditioning_text and reference_continuation_text from records
  - Does NOT assume seed_id, prompt_id, or prompt-suite metadata
  - Faithfulness = output vs reference_continuation_text (not vs prompt instruction)
  - Does not compute prompt-conditioned metrics (no prompt_id in this protocol)

Output: runs/exp2_phaseB/{method}/summary_metrics_continuation.csv
"""

import argparse
import itertools
import json
import os
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from sentence_transformers import SentenceTransformer
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, roc_curve, silhouette_score
from sklearn.metrics.pairwise import cosine_distances, cosine_similarity

from favor_helpers import canonical_author_id, candidate_client_ids, resolve_run_client_roster
from style_asce_runtime import ARC_FACE_BACKEND, classifier_artifact_exists, load_style_scorer

# Optional: BERTScore
try:
    from bert_score import score as bert_score_fn
    BERTSCORE_AVAILABLE = True
except ImportError:
    BERTSCORE_AVAILABLE = False

PROTOCOL_VERSION = "heldout_continuation_v1"
PROTOTYPE_SOURCE_ENCODER_NATIVE = "encoder_native"
PROTOTYPE_SOURCE_EMPIRICAL_MAIN_ROSTER = "empirical_main_roster"
SUPPORTED_PROTOTYPE_SOURCES = (
    PROTOTYPE_SOURCE_ENCODER_NATIVE,
    PROTOTYPE_SOURCE_EMPIRICAL_MAIN_ROSTER,
)
_EMPIRICAL_PROTOTYPE_CACHE = {}


# ============================================================
# Data loading
# ============================================================

def load_continuation_generations(method_dir):
    """Load continuation-protocol generation records."""
    filtered_path = os.path.join(method_dir, "generations_filtered.jsonl")
    raw_path = os.path.join(method_dir, "generations.jsonl")
    path = filtered_path if os.path.exists(filtered_path) else raw_path

    records = []
    if not os.path.exists(path):
        return records, path
    with open(path, "r") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records, path


def author_id_aliases(author_id):
    """Return all accepted bare/blog_pa variants for one author/client id."""
    aliases = {str(author_id)}
    aliases.update(candidate_client_ids(author_id))
    canonical = canonical_author_id(author_id)
    aliases.add(canonical)
    aliases.add(f"blog_pa_{canonical}")
    return {alias for alias in aliases if alias}


def build_author_to_idx(authors):
    """Build a label map that tolerates bare IDs and blog_pa-prefixed IDs."""
    author_to_idx = {}
    for idx, author in enumerate(authors):
        for alias in author_id_aliases(author):
            author_to_idx.setdefault(alias, idx)
    return author_to_idx


def resolve_author_idx(author_to_idx, author_id):
    """Resolve one author id through exact and canonical aliases."""
    for alias in author_id_aliases(author_id):
        if alias in author_to_idx:
            return author_to_idx[alias]
    return -1


# ============================================================
# Embedding and metric helpers (reused from 47)
# ============================================================

def l2_normalize_rows(values):
    """Return row-wise L2-normalized float32 values."""
    arr = np.asarray(values, dtype=np.float32)
    if arr.size == 0:
        return arr.astype(np.float32)
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms = np.clip(norms, 1.0e-8, None)
    return (arr / norms).astype(np.float32)


def encode_texts_normalized(scorer, texts, batch_size=64):
    """Encode text and guarantee row-wise unit vectors."""
    if hasattr(scorer, "encode_texts_normalized"):
        return scorer.encode_texts_normalized(texts, batch_size=batch_size)
    return l2_normalize_rows(scorer.encode_texts(texts, batch_size=batch_size))


def source_record_text(record):
    """Best-effort source text accessor for pooled train records."""
    return str(
        record.get("text")
        or record.get("source_text")
        or record.get("conditioning_text")
        or ""
    )


def build_empirical_main_roster_prototypes(
    scorer,
    scorer_dir,
    pooled_dir,
    roster_authors,
    author_to_idx,
    batch_size=64,
    max_texts_per_author=50,
    max_chars_per_text=1500,
):
    """Build nearest-neighbor prototypes for the main FL roster."""
    source_path = os.path.join(pooled_dir, "train.jsonl")
    if not os.path.exists(source_path):
        raise FileNotFoundError(
            f"empirical_main_roster requires pooled train split at {source_path}"
        )

    roster_by_idx = {idx: author for idx, author in enumerate(roster_authors)}
    expected_authors = sorted({canonical_author_id(author) for author in roster_authors})
    cache_key = (
        os.path.abspath(str(scorer_dir or "")),
        os.path.abspath(source_path),
        tuple(expected_authors),
        int(max_texts_per_author),
        int(max_chars_per_text),
    )
    if cache_key in _EMPIRICAL_PROTOTYPE_CACHE:
        return _EMPIRICAL_PROTOTYPE_CACHE[cache_key]

    buckets = defaultdict(list)
    with open(source_path, "r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            raw_author = record.get("author_id", record.get("client_id", ""))
            author_idx = resolve_author_idx(author_to_idx, raw_author)
            if author_idx < 0:
                continue
            author = canonical_author_id(roster_by_idx[author_idx])
            if len(buckets[author]) >= int(max_texts_per_author):
                continue
            text = source_record_text(record).strip()
            if not text:
                continue
            buckets[author].append(text[: int(max_chars_per_text)])

    missing = [author for author in expected_authors if not buckets.get(author)]
    if missing:
        preview = ", ".join(missing[:10])
        raise RuntimeError(
            "empirical_main_roster could not build prototypes for "
            f"{len(missing)} roster authors from {source_path}: {preview}"
        )

    prototype_authors = []
    source_texts = []
    source_text_authors = []
    for author in expected_authors:
        prototype_authors.append(author)
        for text in buckets[author]:
            source_texts.append(text)
            source_text_authors.append(author)

    source_embeddings = encode_texts_normalized(
        scorer, source_texts, batch_size=batch_size
    )
    embedding_buckets = defaultdict(list)
    for author, embedding in zip(source_text_authors, source_embeddings):
        embedding_buckets[author].append(embedding)

    prototypes = []
    author_manifest = []
    for author in prototype_authors:
        stacked = np.stack(embedding_buckets[author], axis=0)
        prototypes.append(np.mean(stacked, axis=0))
        author_manifest.append({
            "author_id": author,
            "source_text_count": len(embedding_buckets[author]),
        })
    prototypes = l2_normalize_rows(np.stack(prototypes, axis=0))

    manifest = {
        "prototype_source": PROTOTYPE_SOURCE_EMPIRICAL_MAIN_ROSTER,
        "decision_rule": "nearest_cosine",
        "classifier_head_used": False,
        "authorship_model_dir": os.path.abspath(str(scorer_dir or "")),
        "pooled_train_path": os.path.abspath(source_path),
        "num_authors": len(prototype_authors),
        "total_source_texts": len(source_texts),
        "max_texts_per_author": int(max_texts_per_author),
        "max_chars_per_text": int(max_chars_per_text),
        "authors": author_manifest,
    }
    bundle = {
        "authors": prototype_authors,
        "prototypes": prototypes.astype(np.float32),
        "manifest": manifest,
    }
    _EMPIRICAL_PROTOTYPE_CACHE[cache_key] = bundle
    return bundle


def predict_empirical_main_roster_authorship(
    scorer,
    texts,
    prototype_bundle,
    author_to_idx,
    batch_size=64,
):
    """Predict main-roster authors with nearest empirical prototype cosine."""
    embeddings = encode_texts_normalized(scorer, texts, batch_size=batch_size)
    prototypes = np.asarray(prototype_bundle["prototypes"], dtype=np.float32)
    prototype_authors = list(prototype_bundle["authors"])
    if embeddings.size == 0:
        return {
            "backend": getattr(scorer, "backend", "unknown"),
            "embeddings": embeddings,
            "score_matrix": np.zeros((0, len(prototype_authors)), dtype=np.float32),
            "pred_indices": np.asarray([], dtype=np.int64),
            "pred_labels": [],
            "top1_scores": np.asarray([], dtype=np.float32),
            "top2_scores": np.asarray([], dtype=np.float32),
            "margins": np.asarray([], dtype=np.float32),
            "confidence_like": np.asarray([], dtype=np.float32),
        }

    scores = embeddings @ prototypes.T
    pred_positions = np.argmax(scores, axis=1)
    sorted_scores = np.sort(scores, axis=1)
    top1_scores = sorted_scores[:, -1]
    top2_scores = (
        sorted_scores[:, -2]
        if scores.shape[1] > 1
        else np.zeros_like(top1_scores)
    )
    pred_labels = [prototype_authors[int(idx)] for idx in pred_positions.tolist()]
    pred_indices = np.asarray(
        [resolve_author_idx(author_to_idx, label) for label in pred_labels],
        dtype=np.int64,
    )
    margins = top1_scores - top2_scores
    confidence_like = np.clip((top1_scores + 1.0) / 2.0, 0.0, 1.0)
    return {
        "backend": getattr(scorer, "backend", "unknown"),
        "embeddings": embeddings.astype(np.float32),
        "score_matrix": scores.astype(np.float32),
        "pred_indices": pred_indices,
        "pred_labels": pred_labels,
        "top1_scores": top1_scores.astype(np.float32),
        "top2_scores": top2_scores.astype(np.float32),
        "margins": margins.astype(np.float32),
        "confidence_like": confidence_like.astype(np.float32),
    }


def resolve_style_embeddings(texts, authorship_scorer, embedding_model_name):
    """Resolve style embeddings, preferring ArcFace when available."""
    if authorship_scorer is not None and authorship_scorer.backend == ARC_FACE_BACKEND:
        embeddings = authorship_scorer.encode_texts(texts, batch_size=64)
        return embeddings, "authorship_asce"
    model = SentenceTransformer(embedding_model_name)
    embeddings = model.encode(texts, show_progress_bar=True, batch_size=64)
    return np.asarray(embeddings, dtype=np.float32), "sentence_transformer_fallback"


def compute_embedding_metrics(embeddings, labels):
    """Compute style-geometry metrics from precomputed embeddings."""
    embeddings = np.asarray(embeddings, dtype=np.float32)
    unique_labels = sorted(set(labels))
    if len(unique_labels) < 2:
        return {"mean_centroid_cosine_distance": 0.0, "silhouette_score": 0.0}

    centroids = []
    for lbl in unique_labels:
        mask = [i for i, l in enumerate(labels) if l == lbl]
        centroid = embeddings[mask].mean(axis=0)
        centroids.append(centroid)
    centroids = np.array(centroids)

    dist_matrix = cosine_distances(centroids)
    n = len(centroids)
    pairwise_dists = [dist_matrix[i, j] for i in range(n) for j in range(i + 1, n)]
    mean_centroid_dist = float(np.mean(pairwise_dists)) if pairwise_dists else 0.0

    labels_arr = np.array(labels)
    sil = float(silhouette_score(embeddings, labels_arr, metric="cosine"))

    return {
        "mean_centroid_cosine_distance": mean_centroid_dist,
        "silhouette_score": sil,
    }


def compute_eer(fpr, tpr):
    """Compute equal-error rate from ROC false-positive/true-positive arrays."""
    fnr = 1.0 - tpr
    idx = np.nanargmin(np.abs(fpr - fnr))
    return float((fpr[idx] + fnr[idx]) / 2.0)


def compute_verification_metrics(
    embeddings,
    labels,
    max_pairs_per_author=100,
    max_negative_pairs=5000,
    pair_seed=42,
):
    """Compute same-author vs different-author verification AUC/EER.

    This is the pairwise counterpart to closed-set separability. Positive
    examples are two generated continuations from the same author; negative
    examples are generated continuations from different authors. Scores are
    cosine similarities in the style-embedding space.
    """
    embeddings = np.asarray(embeddings, dtype=np.float32)
    labels = list(labels)
    if len(embeddings) != len(labels):
        raise ValueError(
            f"verification embeddings/labels length mismatch: {len(embeddings)} != {len(labels)}"
        )

    label_to_indices = defaultdict(list)
    for idx, label in enumerate(labels):
        label_to_indices[label].append(idx)

    eligible_labels = [
        label for label, indices in label_to_indices.items() if len(indices) >= 2
    ]
    if len(eligible_labels) < 2:
        return {
            "verification_auc": None,
            "verification_eer": None,
            "verification_num_authors": len(eligible_labels),
            "verification_num_same_pairs": 0,
            "verification_num_diff_pairs": 0,
            "verification_mean_same_similarity": None,
            "verification_mean_diff_similarity": None,
        }

    rng = random.Random(pair_seed)
    same_pairs = []
    for label in eligible_labels:
        pairs = list(itertools.combinations(label_to_indices[label], 2))
        if max_pairs_per_author is not None and len(pairs) > int(max_pairs_per_author):
            pairs = rng.sample(pairs, int(max_pairs_per_author))
        same_pairs.extend(pairs)

    target_negative_pairs = int(max_negative_pairs)
    diff_pairs = []
    seen_diff_pairs = set()
    max_attempts = max(target_negative_pairs * 10, 100)
    attempts = 0
    while len(diff_pairs) < target_negative_pairs and attempts < max_attempts:
        label_a, label_b = rng.sample(eligible_labels, 2)
        idx_a = rng.choice(label_to_indices[label_a])
        idx_b = rng.choice(label_to_indices[label_b])
        pair = (idx_a, idx_b) if idx_a < idx_b else (idx_b, idx_a)
        if pair not in seen_diff_pairs:
            seen_diff_pairs.add(pair)
            diff_pairs.append(pair)
        attempts += 1

    if not same_pairs or not diff_pairs:
        return {
            "verification_auc": None,
            "verification_eer": None,
            "verification_num_authors": len(eligible_labels),
            "verification_num_same_pairs": len(same_pairs),
            "verification_num_diff_pairs": len(diff_pairs),
            "verification_mean_same_similarity": None,
            "verification_mean_diff_similarity": None,
        }

    norm_embeddings = embeddings / np.clip(
        np.linalg.norm(embeddings, axis=1, keepdims=True),
        1.0e-8,
        None,
    )
    same_sims = [
        float(np.dot(norm_embeddings[i], norm_embeddings[j])) for i, j in same_pairs
    ]
    diff_sims = [
        float(np.dot(norm_embeddings[i], norm_embeddings[j])) for i, j in diff_pairs
    ]

    verification_labels = [1] * len(same_sims) + [0] * len(diff_sims)
    verification_scores = same_sims + diff_sims
    auc = float(roc_auc_score(verification_labels, verification_scores))
    fpr, tpr, _ = roc_curve(verification_labels, verification_scores)
    eer = compute_eer(fpr, tpr)

    return {
        "verification_auc": round(auc, 4),
        "verification_eer": round(eer, 4),
        "verification_num_authors": len(eligible_labels),
        "verification_num_same_pairs": len(same_pairs),
        "verification_num_diff_pairs": len(diff_pairs),
        "verification_mean_same_similarity": round(float(np.mean(same_sims)), 4),
        "verification_mean_diff_similarity": round(float(np.mean(diff_sims)), 4),
    }


def compute_homogenization_metrics(embeddings, labels, human_embeddings=None, human_labels=None):
    """Compute homogenization metrics (collapse_index, signed_gap)."""
    labels_arr = np.array(labels)
    unique_labels = sorted(set(labels))

    if len(unique_labels) < 2:
        return {
            "collapse_index": None,
            "between_within_dispersion_ratio": None,
            "signed_gap_silhouette": None,
            "signed_gap_centroid_distance": None,
        }

    centroids = {}
    for lbl in unique_labels:
        mask = [i for i, l in enumerate(labels) if l == lbl]
        centroids[lbl] = embeddings[mask].mean(axis=0)

    within_dists = []
    for lbl in unique_labels:
        mask = [i for i, l in enumerate(labels) if l == lbl]
        if len(mask) > 1:
            centroid = centroids[lbl]
            for idx in mask:
                within_dists.append(np.linalg.norm(embeddings[idx] - centroid))
    within_dispersion = float(np.mean(within_dists)) if within_dists else 1e-8

    centroid_arr = np.array([centroids[lbl] for lbl in unique_labels])
    n = len(centroid_arr)
    between_dists = []
    for i in range(n):
        for j in range(i + 1, n):
            between_dists.append(np.linalg.norm(centroid_arr[i] - centroid_arr[j]))
    between_dispersion = float(np.mean(between_dists)) if between_dists else 0.0

    collapse_index = between_dispersion / max(within_dispersion, 1e-8)

    result = {
        "collapse_index": round(float(collapse_index), 4),
        "between_within_dispersion_ratio": round(float(collapse_index), 4),
    }

    if human_embeddings is not None and human_labels is not None:
        human_unique = sorted(set(human_labels))
        if len(human_unique) >= 2:
            human_sil = float(silhouette_score(
                human_embeddings, np.array(human_labels), metric="cosine"
            ))
            gen_sil = float(silhouette_score(embeddings, labels_arr, metric="cosine"))
            result["signed_gap_silhouette"] = round(gen_sil - human_sil, 4)

            human_centroids = []
            for lbl in human_unique:
                hmask = [i for i, l in enumerate(human_labels) if l == lbl]
                human_centroids.append(human_embeddings[hmask].mean(axis=0))
            human_centroid_arr = np.array(human_centroids)
            h_dists = cosine_distances(human_centroid_arr)
            h_n = len(human_centroid_arr)
            h_pairs = [h_dists[i, j] for i in range(h_n) for j in range(i + 1, h_n)]
            h_mean = float(np.mean(h_pairs)) if h_pairs else 0.0

            g_centroids = np.array([centroids[lbl] for lbl in unique_labels])
            g_dists = cosine_distances(g_centroids)
            g_pairs = [g_dists[i, j] for i in range(n) for j in range(i + 1, n)]
            g_mean = float(np.mean(g_pairs)) if g_pairs else 0.0
            result["signed_gap_centroid_distance"] = round(g_mean - h_mean, 4)
        else:
            result["signed_gap_silhouette"] = None
            result["signed_gap_centroid_distance"] = None
    else:
        result["signed_gap_silhouette"] = None
        result["signed_gap_centroid_distance"] = None

    return result


def compute_continuation_faithfulness(
    references,
    outputs,
    embedding_model,
    bertscore_model_type=None,
    bertscore_num_layers=None,
):
    """Compute faithfulness between output and gold reference continuation.

    This is the continuation-protocol equivalent of compute_faithfulness_metrics
    from 47_compute_metrics_prototype.py, but uses reference_continuation_text
    instead of prompt instruction text.
    """
    result = {}

    # Semantic similarity
    ref_embs = embedding_model.encode(references, show_progress_bar=False, batch_size=64)
    out_embs = embedding_model.encode(outputs, show_progress_bar=False, batch_size=64)
    sims = []
    for r_emb, o_emb in zip(ref_embs, out_embs):
        sim = float(cosine_similarity([r_emb], [o_emb])[0, 0])
        sims.append(sim)
    result["continuation_semantic_similarity_mean"] = round(float(np.mean(sims)), 4)
    result["continuation_semantic_similarity_std"] = round(float(np.std(sims)), 4)

    per_example = {
        "continuation_semantic_similarity": [round(float(s), 4) for s in sims],
        "continuation_bertscore_f1": [None] * len(outputs),
    }

    # BERTScore
    if BERTSCORE_AVAILABLE:
        bertscore_kwargs = {
            "verbose": False,
            "batch_size": 32,
        }
        if bertscore_model_type:
            bertscore_kwargs["model_type"] = bertscore_model_type
            if bertscore_num_layers is not None:
                bertscore_kwargs["num_layers"] = int(bertscore_num_layers)
        else:
            bertscore_kwargs["lang"] = "en"
        P, R, F1 = bert_score_fn(
            outputs, references, **bertscore_kwargs,
        )
        result["continuation_bertscore_precision"] = round(float(P.mean()), 4)
        result["continuation_bertscore_recall"] = round(float(R.mean()), 4)
        result["continuation_bertscore_f1"] = round(float(F1.mean()), 4)
        per_example["continuation_bertscore_f1"] = [round(float(x), 4) for x in F1.tolist()]
    else:
        result["continuation_bertscore_precision"] = None
        result["continuation_bertscore_recall"] = None
        result["continuation_bertscore_f1"] = None

    return result, per_example


def compute_phrase_hits(texts, phrases):
    """Compute per-example assistant phrase hit counts."""
    details = []
    for text in texts:
        text_lower = text.lower()
        words = text_lower.split()
        hit_count = sum(text_lower.count(phrase) for phrase in phrases)
        details.append({
            "assistant_phrase_hits": hit_count,
            "assistant_phrase_hit": hit_count > 0,
            "num_words": len(words),
        })
    return details


def load_assistant_phrases(phrases_file):
    """Load assistant phrase lexicon."""
    with open(phrases_file, "r", encoding="utf-8") as f:
        return [line.strip().lower() for line in f if line.strip()]


def expand_candidate_paths(base_dir, raw_path):
    """Resolve config-declared model dirs against a few repo-relative anchors."""
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


def semantic_embedding_model_name(cfg):
    metrics_cfg = cfg.get("metrics", {})
    semantic_cfg = metrics_cfg.get("semantic_space", {})
    return (
        semantic_cfg.get("embedding_model")
        or metrics_cfg.get("embedding_model")
        or "sentence-transformers/all-MiniLM-L6-v2"
    )


def asce_required(cfg_section: dict) -> bool:
    """Whether this config section requires an ArcFace-backed scorer."""
    backend = str(cfg_section.get("backend", "")).strip().lower()
    return bool(cfg_section.get("asce_required", False)) or backend == ARC_FACE_BACKEND


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Compute metrics for Exp2 held-out continuation protocol"
    )
    parser.add_argument("--config", type=str, default="config/favor_main.yaml")
    parser.add_argument(
        "--methods", nargs="+",
        default=[
            "FedAvg",
            "FedProx",
            "pFedMe",
            "Ditto",
            "FedDPA",
            "Pooled PEFT",
            "FAVoR",
        ],
    )
    parser.add_argument(
        "--include_pooled", action="store_true",
        help="Include Exp1 K=50 pooled baseline in metrics if available",
    )
    parser.add_argument(
        "--verification-max-pairs-per-author",
        type=int,
        default=None,
        help="Maximum same-author pairs sampled per author for verification AUC/EER.",
    )
    parser.add_argument(
        "--verification-max-negative-pairs",
        type=int,
        default=None,
        help="Maximum different-author pairs sampled for verification AUC/EER.",
    )
    parser.add_argument(
        "--verification-pair-seed",
        type=int,
        default=None,
        help="Pair-sampling seed for verification AUC/EER.",
    )
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    runs_dir = cfg["paths"]["runs_dir"]
    embedding_model_name = semantic_embedding_model_name(cfg)
    phrases_file = cfg["paths"]["assistant_phrases_file"]
    assistant_phrases = load_assistant_phrases(phrases_file) if os.path.exists(phrases_file) else []
    metrics_cfg = cfg.get("metrics", {})
    style_space_cfg = metrics_cfg.get("style_space", {})
    assistant_scoring_cfg = metrics_cfg.get("assistant_scoring", {})
    verification_cfg = metrics_cfg.get("verification", {})
    prototype_source = str(
        style_space_cfg.get("prototype_source", PROTOTYPE_SOURCE_ENCODER_NATIVE)
        or PROTOTYPE_SOURCE_ENCODER_NATIVE
    )
    if prototype_source not in SUPPORTED_PROTOTYPE_SOURCES:
        raise ValueError(
            f"Unsupported metrics.style_space.prototype_source={prototype_source!r}; "
            f"expected one of {SUPPORTED_PROTOTYPE_SOURCES}"
        )
    verification_max_pairs_per_author = (
        args.verification_max_pairs_per_author
        if args.verification_max_pairs_per_author is not None
        else int(verification_cfg.get("max_pairs_per_author", 100))
    )
    verification_max_negative_pairs = (
        args.verification_max_negative_pairs
        if args.verification_max_negative_pairs is not None
        else int(verification_cfg.get("max_negative_pairs", 5000))
    )
    verification_pair_seed = (
        args.verification_pair_seed
        if args.verification_pair_seed is not None
        else int(verification_cfg.get("pair_seed", 42))
    )

    print("=" * 60)
    print("Step 47b — Metrics for Held-Out Continuation Protocol")
    print("=" * 60)
    print(f"  Protocol: {PROTOCOL_VERSION}")
    print(f"  Join key: continuation_id")
    print(
        "  Verification: "
        f"max_same_pairs/author={verification_max_pairs_per_author}, "
        f"max_diff_pairs={verification_max_negative_pairs}, "
        f"pair_seed={verification_pair_seed}"
    )

    # Resolve client roster
    prototype_dir = cfg["paths"]["prototype_dir"]
    roster_info = resolve_run_client_roster(
        prototype_dir, runs_dir, cfg, fail_fast=True,
    )
    all_authors = sorted(roster_info["client_ids"])
    author_to_idx = build_author_to_idx(all_authors)
    print(f"  [roster] source={roster_info['source']} client_count={len(all_authors)}")

    bertscore_cfg = metrics_cfg.get("bertscore", {})
    bertscore_model_type = (
        bertscore_cfg.get("model_dir")
        or bertscore_cfg.get("model_type")
    )
    bertscore_num_layers = bertscore_cfg.get("num_layers")
    if bertscore_model_type:
        print(f"  [BERTScore] model_type={bertscore_model_type}")

    # Load scorers
    exp1_base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    authorship_scorer = None
    authorship_scorer_dir = None
    authorship_candidates = []
    if style_space_cfg.get("authorship_model_dir"):
        authorship_candidates.extend(
            expand_candidate_paths(exp1_base, style_space_cfg.get("authorship_model_dir"))
        )
    authorship_candidates.extend([
        os.path.join(exp1_base, "runs", "exp1", "K=50", "author_classifier"),
        os.path.join(exp1_base, "runs", "exp1_rerun", "K=50", "author_classifier"),
    ])
    use_empirical_prototypes = bool(style_space_cfg.get("use_empirical_prototypes", True))
    asce_required_authorship = asce_required(style_space_cfg)
    for candidate_dir in authorship_candidates:
        if not classifier_artifact_exists(candidate_dir):
            continue
        try:
            candidate_scorer = load_style_scorer(
                candidate_dir, task="authorship",
                use_empirical_prototypes=use_empirical_prototypes,
            )
            if asce_required_authorship and candidate_scorer.backend != ARC_FACE_BACKEND:
                print(
                    "  [authorship scorer] skipping non-ArcFace candidate "
                    f"backend={candidate_scorer.backend} dir={candidate_dir}"
                )
                continue
            authorship_scorer = candidate_scorer
            authorship_scorer_dir = candidate_dir
            print(f"  [authorship scorer] backend={authorship_scorer.backend} dir={candidate_dir}")
            break
        except Exception as exc:
            print(f"  WARNING: failed to load authorship scorer from {candidate_dir}: {exc}")
    if authorship_scorer is None and asce_required_authorship:
        raise SystemExit(
            "FATAL: ArcFace authorship scorer required but no ArcFace-ready artifact "
            f"was found. Checked candidates: {authorship_candidates}"
        )

    assistant_scorer = None
    assistant_scorer_dir = None
    assistant_candidates = []
    if assistant_scoring_cfg.get("assistant_model_dir"):
        assistant_candidates.extend(
            expand_candidate_paths(exp1_base, assistant_scoring_cfg.get("assistant_model_dir"))
        )
    assistant_candidates.extend([
        os.path.join(exp1_base, "runs", "exp1", "assistant_classifier"),
        os.path.join(exp1_base, "runs", "exp1_rerun", "assistant_classifier"),
    ])
    asce_required_assistant = asce_required(assistant_scoring_cfg)
    for candidate_dir in assistant_candidates:
        if not classifier_artifact_exists(candidate_dir):
            continue
        try:
            candidate_scorer = load_style_scorer(candidate_dir, task="assistant")
            if asce_required_assistant and candidate_scorer.backend != ARC_FACE_BACKEND:
                print(
                    "  [assistant scorer] skipping non-ArcFace candidate "
                    f"backend={candidate_scorer.backend} dir={candidate_dir}"
                )
                continue
            assistant_scorer = candidate_scorer
            assistant_scorer_dir = candidate_dir
            print(f"  [assistant scorer] backend={assistant_scorer.backend} dir={candidate_dir}")
            break
        except Exception as exc:
            print(f"  WARNING: failed to load assistant scorer from {candidate_dir}: {exc}")
    if assistant_scorer is None and asce_required_assistant:
        raise SystemExit(
            "FATAL: ArcFace assistant scorer required but no ArcFace-ready artifact "
            f"was found. Checked candidates: {assistant_candidates}"
        )

    # Human baseline
    pooled_dir = cfg["paths"].get("pooled_dir", "data/pooled/K=50")
    empirical_prototype_bundle = None
    if prototype_source == PROTOTYPE_SOURCE_EMPIRICAL_MAIN_ROSTER:
        if authorship_scorer is None:
            raise SystemExit(
                "FATAL: metrics.style_space.prototype_source=empirical_main_roster "
                "requires an authorship scorer."
            )
        print(
            "\n  [style prototypes] Building empirical main-roster prototypes "
            f"from {pooled_dir}/train.jsonl ..."
        )
        empirical_prototype_bundle = build_empirical_main_roster_prototypes(
            scorer=authorship_scorer,
            scorer_dir=authorship_scorer_dir,
            pooled_dir=pooled_dir,
            roster_authors=all_authors,
            author_to_idx=author_to_idx,
            batch_size=64,
            max_texts_per_author=int(
                style_space_cfg.get("prototype_max_texts_per_author", 50)
            ),
            max_chars_per_text=int(
                style_space_cfg.get("prototype_max_chars_per_text", 1500)
            ),
        )
        proto_manifest = empirical_prototype_bundle["manifest"]
        print(
            "    ✓ empirical prototypes ready "
            f"authors={proto_manifest['num_authors']} "
            f"source_texts={proto_manifest['total_source_texts']}"
        )

    human_ref_path = os.path.join(pooled_dir, "test.jsonl")
    human_embeddings = None
    human_labels = None
    if os.path.exists(human_ref_path):
        print(f"\n  [Human Baseline] Loading from {human_ref_path} ...")
        human_texts_raw = []
        human_author_ids = []
        with open(human_ref_path, "r") as f:
            for line in f:
                if line.strip():
                    rec = json.loads(line)
                    txt = rec.get("text", "")
                    aid = rec.get("author_id", "")
                    author_idx = resolve_author_idx(author_to_idx, aid)
                    if txt.strip() and author_idx >= 0:
                        human_texts_raw.append(txt)
                        human_author_ids.append(author_idx)
        if len(human_texts_raw) >= 2 and len(set(human_author_ids)) >= 2:
            human_embeddings, _ = resolve_style_embeddings(
                human_texts_raw, authorship_scorer, embedding_model_name,
            )
            human_labels = human_author_ids
            print(f"    ✓ Human baseline ready: {human_embeddings.shape}")

    # Process each method
    summary_rows = []
    faithfulness_model = None

    for method in args.methods:
        print(f"\n{'=' * 40} {method} {'=' * 40}")

        method_dir = os.path.join(runs_dir, method)
        records, records_path = load_continuation_generations(method_dir)

        if not records:
            print(f"  No generations found for {method}, skipping")
            continue

        # Validate protocol
        protocol_ok = all(
            r.get("prompt_protocol") == PROTOCOL_VERSION for r in records
        )
        if not protocol_ok:
            protocols = set(r.get("prompt_protocol") for r in records)
            print(f"  WARNING: non-continuation records detected: {protocols}")

        # Extract fields using continuation-protocol schema
        author_ids_all = [r.get("owner_author_id", r.get("author_id", "")) for r in records]
        labels_all = [resolve_author_idx(author_to_idx, a) for a in author_ids_all]
        valid_records = [r for r, label in zip(records, labels_all) if label >= 0]
        labels = [label for label in labels_all if label >= 0]
        valid_author_ids = [a for a, label in zip(author_ids_all, labels_all) if label >= 0]

        if not valid_records:
            print(f"  No valid labeled records for {method}")
            continue

        texts = [r["output_text"] for r in valid_records]
        continuation_ids = [r.get("continuation_id") for r in valid_records]

        print(f"  {len(texts)} texts from {len(set(labels))} authors")
        print(f"  {len(set(continuation_ids))} unique continuation_ids")

        # --- Authorship separability ---
        sep_metrics = {
            "style_space_prototype_source": prototype_source,
            "style_space_prototype_manifest_path": None,
        }
        authorship_preds = [None] * len(texts)
        authorship_pred_author_ids = [None] * len(texts)
        authorship_conf = [None] * len(texts)
        authorship_margin = [None] * len(texts)
        authorship_top1_cosine = [None] * len(texts)
        authorship_correct = [None] * len(texts)
        style_embeddings = None
        style_embedding_backend = None
        authorship_backend_used = None
        authorship_margins = None
        authorship_top1_cosines = None

        if authorship_scorer is not None:
            if prototype_source == PROTOTYPE_SOURCE_EMPIRICAL_MAIN_ROSTER:
                print(
                    "  [Separability] Running empirical main-roster "
                    "nearest-prototype scorer ..."
                )
                auth_pred = predict_empirical_main_roster_authorship(
                    authorship_scorer,
                    texts,
                    empirical_prototype_bundle,
                    author_to_idx,
                    batch_size=64,
                )
                sep_metrics["authorship_decision_rule"] = (
                    "empirical_main_roster_nearest_cosine"
                )
                sep_metrics["authorship_classifier_head_used"] = False
            else:
                print(f"  [Separability] Running authorship classifier ...")
                auth_pred = authorship_scorer.predict_authorship(texts, batch_size=64)
                sep_metrics["authorship_decision_rule"] = "encoder_native_head"
                sep_metrics["authorship_classifier_head_used"] = True
            authorship_backend_used = authorship_scorer.backend
            if "pred_indices" in auth_pred:
                authorship_preds = [int(x) for x in auth_pred["pred_indices"].tolist()]
            pred_author_ids = list(auth_pred.get("pred_labels") or [None] * len(texts))
            authorship_pred_author_ids = pred_author_ids
            if "confidence_like" in auth_pred and auth_pred["confidence_like"].size > 0:
                authorship_conf = [round(float(x), 4) for x in auth_pred["confidence_like"].tolist()]
            if "margins" in auth_pred and auth_pred["margins"].size > 0:
                authorship_margin = [round(float(x), 4) for x in auth_pred["margins"].tolist()]
            if "top1_scores" in auth_pred and auth_pred["top1_scores"].size > 0:
                authorship_top1_cosine = [round(float(x), 4) for x in auth_pred["top1_scores"].tolist()]

            if all(p is not None for p in pred_author_ids):
                valid_author_ids_norm = [
                    canonical_author_id(author_id) for author_id in valid_author_ids
                ]
                pred_author_ids_norm = [
                    canonical_author_id(pred_id) for pred_id in pred_author_ids
                ]
                acc = float(accuracy_score(valid_author_ids_norm, pred_author_ids_norm))
                f1 = float(f1_score(valid_author_ids_norm, pred_author_ids_norm, average="macro", zero_division=0))
                authorship_correct = [
                    int(pred_id == author_id)
                    for pred_id, author_id in zip(pred_author_ids_norm, valid_author_ids_norm)
                ]
            else:
                acc = float(accuracy_score(labels, authorship_preds))
                f1 = float(f1_score(labels, authorship_preds, average="macro", zero_division=0))
                authorship_correct = [
                    int(pred_idx == label)
                    for pred_idx, label in zip(authorship_preds, labels)
                ]

            sep_metrics["authorship_accuracy"] = acc
            sep_metrics["authorship_macro_f1"] = f1
            print(f"    Accuracy: {acc:.4f}, Macro-F1: {f1:.4f}")

            # Extract ArcFace-specific margin and top-1 cosine fields
            if "margins" in auth_pred and auth_pred["margins"].size > 0:
                authorship_margins = auth_pred["margins"]
                sep_metrics["authorship_margin_mean"] = round(float(np.mean(authorship_margins)), 6)
                sep_metrics["authorship_margin_std"] = round(float(np.std(authorship_margins)), 6)
                print(f"    Margin mean: {sep_metrics['authorship_margin_mean']:.4f}")
            if "top1_scores" in auth_pred and auth_pred["top1_scores"].size > 0:
                authorship_top1_cosines = auth_pred["top1_scores"]
                sep_metrics["authorship_top1_cosine_mean"] = round(float(np.mean(authorship_top1_cosines)), 6)

            if (
                authorship_scorer.backend == ARC_FACE_BACKEND
                or prototype_source == PROTOTYPE_SOURCE_EMPIRICAL_MAIN_ROSTER
            ):
                style_embeddings = np.asarray(auth_pred.get("embeddings"), dtype=np.float32)
                style_embedding_backend = (
                    "authorship_asce"
                    if authorship_scorer.backend == ARC_FACE_BACKEND
                    else "authorship_empirical_main_roster"
                )
        else:
            print("  WARNING: no authorship classifier found")
            sep_metrics["authorship_accuracy"] = None
            sep_metrics["authorship_macro_f1"] = None

        # Embedding metrics
        if style_embeddings is None:
            style_embeddings, style_embedding_backend = resolve_style_embeddings(
                texts, authorship_scorer, embedding_model_name,
            )
        emb_metrics = compute_embedding_metrics(style_embeddings, labels)
        sep_metrics.update(emb_metrics)
        print(f"    Centroid distance: {emb_metrics['mean_centroid_cosine_distance']:.4f}")
        print(f"    Silhouette: {emb_metrics['silhouette_score']:.4f}")

        # Pairwise verification: same-author vs different-author generated continuations.
        verification_metrics = compute_verification_metrics(
            style_embeddings,
            labels,
            max_pairs_per_author=verification_max_pairs_per_author,
            max_negative_pairs=verification_max_negative_pairs,
            pair_seed=verification_pair_seed,
        )
        sep_metrics.update(verification_metrics)
        if verification_metrics.get("verification_auc") is not None:
            print(
                "    Verification AUC/EER: "
                f"{verification_metrics['verification_auc']:.4f} / "
                f"{verification_metrics['verification_eer']:.4f} "
                f"(same={verification_metrics['verification_num_same_pairs']}, "
                f"diff={verification_metrics['verification_num_diff_pairs']})"
            )
        else:
            print("    Verification AUC/EER: N/A (not enough same/different-author pairs)")

        # Homogenization
        homo_metrics = compute_homogenization_metrics(
            style_embeddings, labels, human_embeddings, human_labels,
        )
        sep_metrics.update(homo_metrics)
        if homo_metrics.get("collapse_index") is not None:
            print(f"    Collapse index: {homo_metrics['collapse_index']:.4f}")

        # --- Continuation faithfulness ---
        references = [
            r.get("reference_continuation_text") or r.get("gold_continuation_text", "")
            for r in valid_records
        ]
        faith_metrics = {}
        faith_details = {
            "continuation_semantic_similarity": [None] * len(texts),
            "continuation_bertscore_f1": [None] * len(texts),
        }
        if any(ref.strip() for ref in references):
            print(f"  [Faithfulness] Computing vs reference continuation ...")
            if faithfulness_model is None:
                faithfulness_model = SentenceTransformer(embedding_model_name)
            faith_metrics, faith_details = compute_continuation_faithfulness(
                references,
                texts,
                faithfulness_model,
                bertscore_model_type=bertscore_model_type,
                bertscore_num_layers=bertscore_num_layers,
            )
            if faith_metrics.get("continuation_bertscore_f1") is not None:
                print(f"    Continuation BERTScore F1: {faith_metrics['continuation_bertscore_f1']:.4f}")
            print(f"    Continuation semantic sim: {faith_metrics['continuation_semantic_similarity_mean']:.4f}")
        else:
            print(f"  [Faithfulness] Skipped: no reference_continuation_text")
            faith_metrics = {
                "continuation_bertscore_precision": None,
                "continuation_bertscore_recall": None,
                "continuation_bertscore_f1": None,
                "continuation_semantic_similarity_mean": None,
                "continuation_semantic_similarity_std": None,
            }
        sep_metrics.update(faith_metrics)

        # --- Assistant-register diagnostics ---
        asst_metrics = {}
        assistant_backend_used = None
        assistant_calibrated = False
        assistant_scores = [None] * len(texts)
        assistant_raw_margin = [None] * len(texts)
        assistant_preds = [None] * len(texts)
        if assistant_scorer is not None:
            print(f"  [Assistant-register] Running assistant classifier ...")
            assistant_pred = assistant_scorer.predict_assistant(texts, batch_size=64)
            score_arr = assistant_pred["assistant_score"]
            assistant_scores = [round(float(x), 4) for x in score_arr.tolist()]
            assistant_backend_used = assistant_scorer.backend
            assistant_calibrated = bool(getattr(assistant_scorer, "calibrator", None) is not None)
            asst_metrics["assistant_score_mean"] = round(float(np.mean(score_arr)), 4)
            asst_metrics["assistant_score_std"] = round(float(np.std(score_arr)), 4)
            asst_metrics["assistant_score_backend"] = assistant_backend_used
            asst_metrics["assistant_score_calibrated"] = assistant_calibrated
            if "raw_margin" in assistant_pred:
                assistant_raw_margin = [
                    round(float(x), 4) for x in assistant_pred["raw_margin"].tolist()
                ]
                asst_metrics["assistant_raw_margin_mean"] = round(
                    float(np.mean(assistant_pred["raw_margin"])), 4
                )
            if "pred_indices" in assistant_pred:
                assistant_preds = [int(x) for x in assistant_pred["pred_indices"].tolist()]
            print(
                f"    Mean assistant score: {asst_metrics['assistant_score_mean']:.4f} "
                f"(backend={assistant_backend_used}, calibrated={assistant_calibrated})"
            )
        else:
            asst_metrics["assistant_score_mean"] = None
            asst_metrics["assistant_score_std"] = None
            asst_metrics["assistant_score_backend"] = None
            asst_metrics["assistant_score_calibrated"] = None

        # Phrase hits
        phrase_details = compute_phrase_hits(texts, assistant_phrases)
        total_words = sum(d["num_words"] for d in phrase_details)
        total_hits = sum(d["assistant_phrase_hits"] for d in phrase_details)
        texts_with_hit = sum(1 for d in phrase_details if d["assistant_phrase_hit"])
        asst_metrics["assistant_phrase_rate_per_1k_tokens"] = round(
            (total_hits / max(total_words, 1)) * 1000, 4
        )
        asst_metrics["assistant_phrase_hit_ratio"] = round(
            texts_with_hit / max(len(texts), 1), 4
        )

        sep_metrics.update(asst_metrics)

        # Save per-method outputs
        if empirical_prototype_bundle is not None:
            prototype_manifest_path = os.path.join(
                method_dir, "style_space_prototype_manifest_continuation.json"
            )
            with open(prototype_manifest_path, "w", encoding="utf-8") as f:
                json.dump(empirical_prototype_bundle["manifest"], f, indent=2)
            sep_metrics["style_space_prototype_manifest_path"] = os.path.abspath(
                prototype_manifest_path
            )
            sep_metrics["style_space_prototype_num_authors"] = (
                empirical_prototype_bundle["manifest"].get("num_authors")
            )
            sep_metrics["style_space_prototype_total_source_texts"] = (
                empirical_prototype_bundle["manifest"].get("total_source_texts")
            )

        metrics_path = os.path.join(method_dir, "metrics_continuation.json")
        with open(metrics_path, "w") as f:
            json.dump(sep_metrics, f, indent=2)

        per_example_path = os.path.join(method_dir, "per_example_metrics.continuation.jsonl")
        qf_applied = os.path.basename(records_path).startswith("generations_filtered")
        with open(per_example_path, "w", encoding="utf-8") as f:
            for idx, rec in enumerate(valid_records):
                owner_author_id = rec.get(
                    "owner_author_id",
                    rec.get("author_id", rec.get("client_id")),
                )
                semantic_similarity = (
                    faith_details["continuation_semantic_similarity"][idx]
                    if idx < len(faith_details["continuation_semantic_similarity"])
                    else None
                )
                bertscore_f1 = (
                    faith_details["continuation_bertscore_f1"][idx]
                    if idx < len(faith_details["continuation_bertscore_f1"])
                    else None
                )
                example = {
                    "method": method,
                    "protocol": PROTOCOL_VERSION,
                    "client_id": rec.get("client_id", owner_author_id),
                    "author_id": rec.get("author_id", owner_author_id),
                    "owner_author_id": owner_author_id,
                    "prompt_id": rec.get("prompt_id"),
                    "seed_id": rec.get("seed_id"),
                    "continuation_id": rec.get("continuation_id"),
                    "conditioning_text": rec.get("conditioning_text"),
                    "reference_continuation_text": rec.get(
                        "reference_continuation_text",
                        rec.get("gold_continuation_text"),
                    ),
                    "gold_continuation_text": rec.get(
                        "gold_continuation_text",
                        rec.get("reference_continuation_text"),
                    ),
                    "output_text": rec.get("output_text"),
                    "output_token_count": rec.get(
                        "output_token_count",
                        rec.get("num_output_tokens"),
                    ),
                    "num_output_tokens": rec.get("num_output_tokens"),
                    "style_embedding_backend": style_embedding_backend,
                    "style_space_prototype_source": sep_metrics.get(
                        "style_space_prototype_source"
                    ),
                    "quality_filter_applied": qf_applied,
                    "passed_quality_filter": True if qf_applied else None,
                    "assistant_score": (
                        assistant_scores[idx] if idx < len(assistant_scores) else None
                    ),
                    "assistant_prob": (
                        assistant_scores[idx] if idx < len(assistant_scores) else None
                    ),
                    "assistant_raw_margin": (
                        assistant_raw_margin[idx] if idx < len(assistant_raw_margin) else None
                    ),
                    "assistant_score_backend": assistant_backend_used,
                    "assistant_score_calibrated": assistant_calibrated,
                    "assistant_pred": (
                        assistant_preds[idx] if idx < len(assistant_preds) else None
                    ),
                    "assistant_phrase_count": (
                        phrase_details[idx]["assistant_phrase_hits"]
                        if idx < len(phrase_details)
                        else None
                    ),
                    "assistant_phrase_hits": (
                        phrase_details[idx]["assistant_phrase_hits"]
                        if idx < len(phrase_details)
                        else None
                    ),
                    "assistant_phrase_hit": (
                        phrase_details[idx]["assistant_phrase_hit"]
                        if idx < len(phrase_details)
                        else None
                    ),
                    "semantic_similarity": semantic_similarity,
                    "continuation_semantic_similarity": semantic_similarity,
                    "bertscore_f1": bertscore_f1,
                    "continuation_bertscore_f1": bertscore_f1,
                    "authorship_pred": (
                        authorship_preds[idx] if idx < len(authorship_preds) else None
                    ),
                    "authorship_pred_author_id": (
                        authorship_pred_author_ids[idx]
                        if idx < len(authorship_pred_author_ids)
                        else None
                    ),
                    "authorship_top1_cosine": (
                        authorship_top1_cosine[idx]
                        if idx < len(authorship_top1_cosine)
                        else None
                    ),
                    "authorship_margin": (
                        authorship_margin[idx] if idx < len(authorship_margin) else None
                    ),
                    "authorship_confidence": (
                        authorship_conf[idx] if idx < len(authorship_conf) else None
                    ),
                    "authorship_correct": (
                        authorship_correct[idx] if idx < len(authorship_correct) else None
                    ),
                }
                f.write(json.dumps(example) + "\n")
        print(f"  ✓ per-example metrics saved to {per_example_path}")

        # Summary row
        row = {
            "method": method,
            "protocol": PROTOCOL_VERSION,
            "n_texts": len(texts),
            "n_authors": len(set(labels)),
            "n_continuation_ids": len(set(continuation_ids)),
            # --- Authorship (paper-facing: use margin, not max-softmax) ---
            "authorship_accuracy": sep_metrics.get("authorship_accuracy"),
            "authorship_macro_f1": sep_metrics.get("authorship_macro_f1"),
            "authorship_margin_mean": sep_metrics.get("authorship_margin_mean"),
            "authorship_margin_std": sep_metrics.get("authorship_margin_std"),
            "authorship_top1_cosine_mean": sep_metrics.get("authorship_top1_cosine_mean"),
            "authorship_backend": authorship_backend_used,
            "authorship_decision_rule": sep_metrics.get("authorship_decision_rule"),
            "authorship_classifier_head_used": sep_metrics.get(
                "authorship_classifier_head_used"
            ),
            # --- Pairwise verification ---
            "verification_auc": sep_metrics.get("verification_auc"),
            "verification_eer": sep_metrics.get("verification_eer"),
            "verification_num_authors": sep_metrics.get("verification_num_authors"),
            "verification_num_same_pairs": sep_metrics.get("verification_num_same_pairs"),
            "verification_num_diff_pairs": sep_metrics.get("verification_num_diff_pairs"),
            "verification_mean_same_similarity": sep_metrics.get("verification_mean_same_similarity"),
            "verification_mean_diff_similarity": sep_metrics.get("verification_mean_diff_similarity"),
            # --- Geometry ---
            "centroid_distance": sep_metrics.get("mean_centroid_cosine_distance"),
            "silhouette": sep_metrics.get("silhouette_score"),
            "collapse_index": sep_metrics.get("collapse_index"),
            "signed_gap_silhouette": sep_metrics.get("signed_gap_silhouette"),
            "signed_gap_centroid_distance": sep_metrics.get("signed_gap_centroid_distance"),
            # --- Faithfulness ---
            "continuation_bertscore_f1": sep_metrics.get("continuation_bertscore_f1"),
            "continuation_semantic_similarity": sep_metrics.get("continuation_semantic_similarity_mean"),
            # --- Assistant-register score (paper-facing: calibrated P(asst) when ArcFace) ---
            "assistant_score_mean": asst_metrics.get("assistant_score_mean"),
            "assistant_score_std": asst_metrics.get("assistant_score_std"),
            "assistant_raw_margin_mean": asst_metrics.get("assistant_raw_margin_mean"),
            "assistant_score_backend": asst_metrics.get("assistant_score_backend"),
            "assistant_score_calibrated": asst_metrics.get("assistant_score_calibrated"),
            "assistant_phrase_rate": asst_metrics.get("assistant_phrase_rate_per_1k_tokens"),
            "assistant_phrase_hit_ratio": asst_metrics.get("assistant_phrase_hit_ratio"),
            # --- Meta ---
            "style_embedding_backend": style_embedding_backend,
            "style_space_prototype_source": sep_metrics.get("style_space_prototype_source"),
            "style_space_prototype_manifest_path": sep_metrics.get(
                "style_space_prototype_manifest_path"
            ),
            "authorship_model_dir": authorship_scorer_dir,
            "assistant_model_dir": assistant_scorer_dir,
        }
        summary_rows.append(row)

        method_summary_path = os.path.join(method_dir, "summary_metrics_continuation.csv")
        pd.DataFrame([row]).to_csv(method_summary_path, index=False)
        print(f"  ✓ saved to {method_summary_path}")

    # Write overall summary
    if summary_rows:
        summary_df = pd.DataFrame(summary_rows)
        summary_path = os.path.join(runs_dir, "summary_metrics_continuation.csv")
        summary_df.to_csv(summary_path, index=False)
        print(f"\n✓ Summary metrics saved to {summary_path}")
        print(summary_df.to_string(index=False))

    print("\n✓ Continuation metrics computation complete.")


if __name__ == "__main__":
    main()
