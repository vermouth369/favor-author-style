#!/usr/bin/env python3
"""asce_private_alignment.py — ArcFace-guided private-stage training helpers.

Provides:
  - ASCEStyleProjector: learnable linear projection from LM hidden space
    into the ArcFace style embedding space.
  - Target caching: per-client frozen ArcFace target vectors from reference texts.
  - Pooling: configurable mean-pooled or per-token supervised hidden-state views.
  - Loss: cosine alignment loss against frozen targets.
  - Config resolver: reads the `asce_private_alignment` YAML block.

Used by 44_train_favor.py during private-stage training.
This module is imported lazily (only when asce_private_alignment.enabled = true).
"""

from __future__ import annotations

import json
import math
import os
from collections import OrderedDict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# Projection head
# ============================================================

class ASCEStyleProjector(nn.Module):
    """Learnable projection from LM hidden space into ArcFace style space.

    Supported architectures (selected via `projector.type` in YAML):
      - "linear" (default / phase-1):
            Linear(hidden_size -> embedding_dim)
            optional LayerNorm, optional Dropout, L2-normalize
      - "mlp_2":
            Linear(hidden_size -> hidden_dim)
            GELU
            optional Dropout
            Linear(hidden_dim -> embedding_dim)
            optional LayerNorm, L2-normalize
        `hidden_dim` defaults to `max(hidden_size // 2, embedding_dim)` so the
        bottleneck is not tighter than the target embedding. This gives the
        projector enough capacity to learn per-author discrimination instead of
        collapsing every client toward the mean of the author-classifier's
        reference embeddings (which was the observed failure mode of the
        linear projector under moderate ArcFace weight).

    The projector is attached to the LM as a submodule during private-stage
    training so that HF Trainer's optimizer automatically picks up its
    parameters.  It is detached and deleted after each client's private stage.
    """

    def __init__(
        self,
        hidden_size: int,
        embedding_dim: int,
        architecture: str = "linear",
        use_layer_norm: bool = False,
        dropout: float = 0.0,
        hidden_dim: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.hidden_size = int(hidden_size)
        self.embedding_dim = int(embedding_dim)
        self.architecture = str(architecture).lower()
        self.use_layer_norm = bool(use_layer_norm)
        self.dropout_p = float(dropout)

        if self.architecture not in ("linear", "mlp_2"):
            raise ValueError(
                f"ASCEStyleProjector: unknown architecture '{architecture}'; "
                f"expected one of ['linear', 'mlp_2']"
            )

        if self.architecture == "linear":
            self.linear = nn.Linear(self.hidden_size, self.embedding_dim)
            self.act = None
            self.linear2 = None
        else:  # mlp_2
            if hidden_dim is None:
                hidden_dim = max(self.hidden_size // 2, self.embedding_dim)
            self.hidden_dim = int(hidden_dim)
            self.linear = nn.Linear(self.hidden_size, self.hidden_dim)
            self.act = nn.GELU()
            self.linear2 = nn.Linear(self.hidden_dim, self.embedding_dim)

        self.layer_norm = nn.LayerNorm(self.embedding_dim) if self.use_layer_norm else None
        self.dropout = nn.Dropout(self.dropout_p) if self.dropout_p > 0 else None

    def forward(self, pooled_hidden: torch.Tensor) -> torch.Tensor:
        """Project and L2-normalize.

        Args:
            pooled_hidden: (batch, hidden_size) mean-pooled hidden states.

        Returns:
            (batch, embedding_dim) L2-normalized style projections.
        """
        x = self.linear(pooled_hidden)
        if self.act is not None:
            x = self.act(x)
            if self.dropout is not None:
                x = self.dropout(x)
            x = self.linear2(x)
        if self.layer_norm is not None:
            x = self.layer_norm(x)
        if self.dropout is not None and self.act is None:
            # For the linear head, dropout stays after the (optional) LN,
            # preserving the phase-1 ordering.
            x = self.dropout(x)
        return F.normalize(x, p=2, dim=-1, eps=1e-8)


def build_projector(
    hidden_size: int,
    embedding_dim: int,
    projector_cfg: Optional[dict] = None,
    device: Optional[torch.device] = None,
) -> ASCEStyleProjector:
    """Construct projector from YAML config block."""
    projector_cfg = projector_cfg or {}
    architecture = str(projector_cfg.get("type", "linear")).lower()
    proj = ASCEStyleProjector(
        hidden_size=hidden_size,
        embedding_dim=embedding_dim,
        architecture=architecture,
        use_layer_norm=bool(projector_cfg.get("use_layer_norm", False)),
        dropout=float(projector_cfg.get("dropout", 0.0)),
        hidden_dim=projector_cfg.get("hidden_dim"),
    )
    if device is not None:
        proj = proj.to(device)
    return proj


# ============================================================
# Target computation and caching
# ============================================================

def compute_client_asce_target(
    scorer,
    texts: List[str],
    normalize: bool = True,
    strategy: str = "mean",
    batch_size: int = 32,
) -> np.ndarray:
    """Compute a frozen ArcFace target vector from client reference texts.

    Args:
        scorer: an ASCEScorer (or BaseStyleScorer with encode_texts).
        texts: list of reference text strings.
        normalize: if True, L2-normalize the final vector.
        strategy: aggregation strategy ("mean" only for phase 1).
        batch_size: encoding batch size.

    Returns:
        (embedding_dim,) float32 numpy array.
    """
    if not texts:
        raise ValueError("compute_client_asce_target: no reference texts provided")

    embeddings = scorer.encode_texts(texts, batch_size=batch_size)
    # embeddings: (n_texts, embedding_dim), already L2-normalized per-text by ArcFace

    strategy = str(strategy or "mean").lower()

    if strategy in (
        "mean",
        "reference_mean",
        "empirical_per_client",
        "client_embedding_mean",
    ):
        target = embeddings.mean(axis=0)
    else:
        raise ValueError(f"Unsupported target strategy: {strategy!r}")

    if normalize:
        norm = np.linalg.norm(target)
        if norm > 1e-8:
            target = target / norm

    return target.astype(np.float32)


def cache_client_asce_target(
    client_id: str,
    texts: List[str],
    scorer,
    cache_dir: str,
    normalize: bool = True,
    strategy: str = "mean",
    model_dir: str = "",
    batch_size: int = 32,
) -> Tuple[np.ndarray, dict]:
    """Compute and cache a per-client ArcFace target vector.

    Returns:
        (target_vector, metadata_dict)
    """
    os.makedirs(cache_dir, exist_ok=True)
    npy_path = os.path.join(cache_dir, f"client={client_id}.npy")
    json_path = os.path.join(cache_dir, f"client={client_id}.json")

    if os.path.exists(npy_path):
        target = np.load(npy_path)
        meta = {}
        if os.path.exists(json_path):
            with open(json_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
        print(f"      [asce_target] loaded cached target for {client_id} "
              f"(dim={target.shape[0]})")
        return target, meta

    target = compute_client_asce_target(
        scorer, texts,
        normalize=normalize,
        strategy=strategy,
        batch_size=batch_size,
    )

    np.save(npy_path, target)
    meta = {
        "client_id": str(client_id),
        "n_texts": len(texts),
        "embedding_dim": int(target.shape[0]),
        "normalize": normalize,
        "strategy": str(strategy),
        "model_dir": str(model_dir),
        "target_norm": float(np.linalg.norm(target)),
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"      [asce_target] computed and cached for {client_id} "
          f"(dim={target.shape[0]}, n_texts={len(texts)}, strategy={strategy})")

    return target, meta


# ============================================================
# Pooling helper
# ============================================================

def pool_supervised_hidden_states(
    hidden_states: torch.Tensor,
    labels: torch.Tensor,
    min_tokens: int = 8,
) -> Optional[torch.Tensor]:
    """Mean-pool final-layer hidden states over supervised token positions.

    Args:
        hidden_states: (batch, seq_len, hidden_size) from final layer.
        labels: (batch, seq_len) with -100 for non-supervised positions.
        min_tokens: minimum supervised tokens required per example.

    Returns:
        (batch, hidden_size) or None if too few supervised tokens.
    """
    # mask: 1 where supervised, 0 otherwise
    mask = labels.ne(-100).to(hidden_states.dtype)  # (batch, seq_len)
    token_counts = mask.sum(dim=1)  # (batch,)

    # Check minimum token count (use batch mean)
    mean_tokens = token_counts.mean().item()
    if mean_tokens < min_tokens:
        return None

    # Expand mask for broadcasting
    mask_expanded = mask.unsqueeze(-1)  # (batch, seq_len, 1)
    summed = (hidden_states * mask_expanded).sum(dim=1)  # (batch, hidden_size)
    counts = token_counts.clamp_min(1.0).unsqueeze(-1)  # (batch, 1)
    return summed / counts


def gather_supervised_hidden_states(
    hidden_states: torch.Tensor,
    labels: torch.Tensor,
    min_tokens: int = 8,
) -> Optional[torch.Tensor]:
    """Gather final-layer hidden states at supervised token positions."""
    mask = labels.ne(-100)
    token_counts = mask.sum(dim=1)
    mean_tokens = token_counts.float().mean().item()
    if mean_tokens < min_tokens:
        return None
    if int(mask.sum().item()) <= 0:
        return None
    return hidden_states[mask]


def select_supervised_token_hidden_states(
    hidden_states: torch.Tensor,
    labels: torch.Tensor,
    min_tokens: int = 8,
) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
    """Return flat supervised-token hidden states plus their batch indices."""
    mask = labels.ne(-100)
    token_counts = mask.sum(dim=1)
    mean_tokens = token_counts.float().mean().item()
    if mean_tokens < min_tokens:
        return None
    if int(mask.sum().item()) <= 0:
        return None

    token_hidden = hidden_states[mask]
    batch_arange = torch.arange(
        mask.size(0), device=mask.device, dtype=torch.long,
    ).unsqueeze(1).expand_as(mask)
    batch_index = batch_arange[mask]
    return token_hidden, batch_index


def prepare_asce_projector_inputs(
    hidden_states: torch.Tensor,
    labels: torch.Tensor,
    pooling: str = "mean_supervised_tokens",
    min_tokens: int = 8,
) -> Optional[torch.Tensor]:
    """Prepare projector inputs under the requested supervised-token view."""
    pooling = str(pooling or "mean_supervised_tokens").lower()
    if pooling in ("mean_supervised_tokens", "mean"):
        return pool_supervised_hidden_states(
            hidden_states,
            labels,
            min_tokens=min_tokens,
        )
    if pooling in ("per_token_supervised", "per_token"):
        return gather_supervised_hidden_states(
            hidden_states,
            labels,
            min_tokens=min_tokens,
        )
    raise ValueError(
        f"Unsupported arcface pooling={pooling!r}; "
        "expected one of ['mean_supervised_tokens', 'per_token_supervised']"
    )


# ============================================================
# Loss computation
# ============================================================

def compute_asce_style_loss(
    projected: torch.Tensor,
    target: torch.Tensor,
    loss_type: str = "cosine",
) -> Tuple[torch.Tensor, float]:
    """Compute alignment loss between projected LM embedding and frozen target.

    Args:
        projected: (batch, embedding_dim) L2-normalized projected embeddings.
        target: (embedding_dim,) frozen ArcFace target vector.
        loss_type: "cosine" (default) or "mse".

    Returns:
        (loss_scalar, mean_cosine_float)

    Note: for classifier-cross-entropy (the "KL divergence" formulation that
    uses the full K-way softmax against all reference classes), see
    `compute_arcface_ce_loss` below, which is dispatched separately by the
    trainer because it needs the full reference matrix, a class index, and
    the scale factor.
    """
    # Expand target to match batch
    if target.dim() == 1:
        target = target.unsqueeze(0).expand_as(projected)

    if loss_type == "cosine":
        cosine = F.cosine_similarity(projected, target, dim=-1)  # (batch,)
        loss = (1.0 - cosine).mean()
        mean_cosine = cosine.mean().item()
    elif loss_type == "mse":
        loss = F.mse_loss(projected, target)
        cosine = F.cosine_similarity(projected, target, dim=-1)
        mean_cosine = cosine.mean().item()
    else:
        raise ValueError(f"Unsupported arcface loss_type: {loss_type!r}")

    return loss, mean_cosine


def _mean_per_example_loss(
    per_item_loss: torch.Tensor,
    batch_index: torch.Tensor,
) -> torch.Tensor:
    """Average losses within each example, then average across active examples."""
    batch_size = int(batch_index.max().item()) + 1
    example_loss_sum = torch.zeros(
        batch_size, device=per_item_loss.device, dtype=per_item_loss.dtype,
    )
    example_count = torch.zeros(
        batch_size, device=per_item_loss.device, dtype=per_item_loss.dtype,
    )
    example_loss_sum.scatter_add_(0, batch_index, per_item_loss)
    example_count.scatter_add_(0, batch_index, torch.ones_like(per_item_loss))
    active_examples = example_count > 0
    if not torch.any(active_examples):
        return per_item_loss.mean()
    per_example_loss = example_loss_sum[active_examples] / example_count[active_examples]
    return per_example_loss.mean()


def compute_asce_style_loss_per_token(
    projected_per_token: torch.Tensor,
    batch_index: torch.Tensor,
    target: torch.Tensor,
    loss_type: str = "cosine",
) -> Tuple[torch.Tensor, float]:
    """Per-token style loss with per-example normalization."""
    if target.dim() == 1:
        target_expanded = target.unsqueeze(0).expand_as(projected_per_token)
    else:
        target_expanded = target[batch_index]

    if loss_type == "cosine":
        cosine = F.cosine_similarity(
            projected_per_token, target_expanded, dim=-1,
        )
        per_token_loss = 1.0 - cosine
        mean_cosine = cosine.mean().item()
    elif loss_type == "mse":
        per_token_loss = F.mse_loss(
            projected_per_token,
            target_expanded,
            reduction="none",
        ).mean(dim=-1)
        cosine = F.cosine_similarity(
            projected_per_token, target_expanded, dim=-1,
        )
        mean_cosine = cosine.mean().item()
    else:
        raise ValueError(
            f"Unsupported per-token arcface loss_type: {loss_type!r}"
        )

    loss = _mean_per_example_loss(per_token_loss, batch_index)
    return loss, mean_cosine


def compute_arcface_ce_loss(
    projected: torch.Tensor,
    reference_vectors: torch.Tensor,
    class_index: int,
    scale_s: float = 30.0,
    margin_m: float = 0.0,
) -> Tuple[torch.Tensor, float, float]:
    """Cross-entropy loss against the full K-way ArcFace classifier.

    Equivalent in objective to the KL divergence from the one-hot author
    label to the softmax over scaled cosine similarities against all K
    reference vectors. Unlike `compute_asce_style_loss(loss_type='cosine')`
    -- which only pulls `projected` toward the *own* author's reference
    vector and has a trivial degenerate solution (collapse every client's
    hidden representation toward the global mean of all reference vectors)
    -- this loss pushes `projected` away from all K-1 other authors'
    references via the softmax denominator. It is therefore a properly
    discriminative signal and uses the trained ArcFace classifier head
    (which achieves ~0.89 authorship accuracy on held-out human text in
    this project), not just its embedding layer.

    Args:
        projected: (batch, embedding_dim) L2-normalized projector output.
        reference_vectors: (K, embedding_dim) L2-normalized classifier
            reference matrix (either `class_weight_vectors.npy` — the
            trained ArcFace head weights — or `class_prototypes.npy` — the
            empirical per-class embedding means). See
            `load_classifier_head_vectors`.
        class_index: integer in [0, K-1] — the author's class index in the
            classifier's label map.
        scale_s: ArcFace scale factor (typically 30.0, read from the
            classifier's `backend_meta.json`).
        margin_m: angular margin applied only to the target logit. 0.0
            recovers vanilla softmax-CE on scaled cosines; 0.35 recovers
            the full ArcFace (Deng et al., 2019) training objective.

    Returns:
        (loss_scalar, mean_target_cosine, top1_accuracy)
            loss_scalar: CE loss averaged over the batch.
            mean_target_cosine: mean cosine similarity between `projected`
                and the target class's reference vector. Directly comparable
                to `mean_cosine` from `compute_asce_style_loss`, so
                `round_metrics.json` stays apples-to-apples across modes.
            top1_accuracy: fraction of batch items whose argmax over the
                K logits equals `class_index`. A useful health signal:
                values near 1/K mean the projector is chance-level; values
                climbing toward 1.0 mean it has learned the author.
    """
    # Both projected and reference_vectors are assumed unit-norm in their
    # last dim (projector ends with F.normalize; ASCEScorer normalizes
    # its reference_vectors on load).
    cosine = projected @ reference_vectors.T  # (B, K) in [-1, 1]

    batch_size = projected.size(0)
    target = torch.full(
        (batch_size,), int(class_index),
        dtype=torch.long, device=projected.device,
    )

    if margin_m > 0.0:
        # ArcFace-proper: apply angular margin only to target logit.
        # Clamp away from {-1, 1} for acos numerical stability.
        cos_clamped = cosine.clamp(-1.0 + 1.0e-7, 1.0 - 1.0e-7)
        target_cos = cos_clamped.gather(1, target.unsqueeze(1))  # (B, 1)
        target_theta = torch.acos(target_cos)
        target_cos_m = torch.cos(target_theta + margin_m)
        logits = cosine.clone()
        logits.scatter_(1, target.unsqueeze(1), target_cos_m)
        logits = logits * scale_s
    else:
        logits = cosine * scale_s

    loss = F.cross_entropy(logits, target)

    with torch.no_grad():
        target_cos_per_item = cosine.gather(1, target.unsqueeze(1)).squeeze(1)
        mean_target_cosine = float(target_cos_per_item.mean().item())
        pred = logits.argmax(dim=1)
        top1_acc = float((pred == target).float().mean().item())

    return loss, mean_target_cosine, top1_acc


def compute_arcface_ce_loss_per_token(
    projected_per_token: torch.Tensor,
    batch_index: torch.Tensor,
    reference_vectors: torch.Tensor,
    class_index: int,
    scale_s: float = 30.0,
    margin_m: float = 0.0,
) -> Tuple[torch.Tensor, float, float]:
    """Per-token ArcFace CE with per-example normalization."""
    cosine = projected_per_token @ reference_vectors.T  # (N_kept, K)
    target = torch.full(
        (projected_per_token.size(0),),
        int(class_index),
        dtype=torch.long,
        device=projected_per_token.device,
    )

    if margin_m > 0.0:
        cos_clamped = cosine.clamp(-1.0 + 1.0e-7, 1.0 - 1.0e-7)
        target_cos = cos_clamped.gather(1, target.unsqueeze(1))
        target_theta = torch.acos(target_cos)
        target_cos_m = torch.cos(target_theta + margin_m)
        logits = cosine.clone()
        logits.scatter_(1, target.unsqueeze(1), target_cos_m)
        logits = logits * scale_s
    else:
        logits = cosine * scale_s

    per_token_loss = F.cross_entropy(logits, target, reduction="none")
    loss = _mean_per_example_loss(per_token_loss, batch_index)

    with torch.no_grad():
        target_cos_per_token = cosine.gather(1, target.unsqueeze(1)).squeeze(1)
        mean_target_cosine = float(target_cos_per_token.mean().item())
        pred = logits.argmax(dim=1)
        top1_acc = float((pred == target).float().mean().item())

    return loss, mean_target_cosine, top1_acc


def load_classifier_head_vectors(
    model_dir,
    source: str = "weight_vectors",
) -> np.ndarray:
    """Load K x D reference vectors from an ArcFace classifier artifact dir.

    For CE / ArcFace-proper loss we want the **trained classifier head
    weights** (`class_weight_vectors.npy`) because those are the W matrix
    the classifier actually optimized against during training. Falling
    back to `class_prototypes.npy` (empirical per-class embedding means)
    still works but produces a different -- typically smoother, less
    discriminative -- loss surface.

    Args:
        model_dir: path to the ArcFace classifier directory (the one that
            contains `backend_meta.json`, `class_weight_vectors.npy`, etc.)
        source: "weight_vectors" (default) or "prototypes".

    Returns:
        (K, embedding_dim) L2-normalized float32 numpy array.
    """
    model_dir = Path(model_dir)
    primary_name = (
        "class_weight_vectors.npy" if source == "weight_vectors"
        else "class_prototypes.npy" if source == "prototypes"
        else None
    )
    if primary_name is None:
        raise ValueError(
            f"load_classifier_head_vectors: unknown source={source!r}; "
            f"expected one of ['weight_vectors', 'prototypes']"
        )
    fallback_name = (
        "class_prototypes.npy" if primary_name == "class_weight_vectors.npy"
        else "class_weight_vectors.npy"
    )

    primary_path = model_dir / primary_name
    fallback_path = model_dir / fallback_name
    if primary_path.exists():
        used_path = primary_path
    elif fallback_path.exists():
        used_path = fallback_path
    else:
        raise FileNotFoundError(
            f"No classifier head vectors in {model_dir}; expected "
            f"{primary_name} or {fallback_name}."
        )

    vectors = np.load(used_path).astype(np.float32)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    vectors = vectors / np.clip(norms, 1.0e-8, None)
    return vectors


# ============================================================
# Warmup schedule
# ============================================================

def compute_asce_schedule_weight(
    current_step: int,
    total_steps: int,
    base_weight: float,
    warmup_fraction: float,
    schedule_cfg: Optional[dict] = None,
) -> float:
    """Compute ArcFace lambda under linear warmup or a curriculum schedule."""
    if total_steps < 5:
        return base_weight

    schedule_cfg = schedule_cfg or {}
    schedule_type = str(schedule_cfg.get("type", "linear_warmup")).lower()

    warmup_steps = 0
    if warmup_fraction > 0:
        warmup_steps = max(1, math.ceil(warmup_fraction * total_steps))
        if current_step < warmup_steps:
            ramp = min(1.0, float(current_step) / float(warmup_steps))
            return base_weight * ramp

    if schedule_type == "linear_warmup":
        return base_weight

    if schedule_type != "warmup_peak_cooldown":
        raise ValueError(
            f"Unsupported arcface schedule type={schedule_type!r}; "
            "expected one of ['linear_warmup', 'warmup_peak_cooldown']"
        )

    peak_multiplier = float(schedule_cfg.get("peak_multiplier", 1.0))
    cooldown_fraction = float(schedule_cfg.get("cooldown_fraction", 0.0))
    peak_fraction = schedule_cfg.get("peak_fraction")
    if peak_fraction is None:
        peak_fraction = max(0.0, 1.0 - warmup_fraction - cooldown_fraction)
    peak_fraction = float(peak_fraction)

    peak_steps = max(0, int(round(peak_fraction * total_steps)))
    cooldown_steps = max(0, int(round(cooldown_fraction * total_steps)))
    peak_end = warmup_steps + peak_steps
    cooldown_end = peak_end + cooldown_steps
    peak_weight = base_weight * peak_multiplier

    if current_step < peak_end:
        return peak_weight
    if current_step < cooldown_end and cooldown_steps > 0:
        decay_progress = float(current_step - peak_end) / float(cooldown_steps)
        decay_progress = min(max(decay_progress, 0.0), 1.0)
        return peak_weight + (base_weight - peak_weight) * decay_progress
    return base_weight


def compute_asce_warmup_weight(
    current_step: int,
    total_steps: int,
    base_weight: float,
    warmup_fraction: float,
) -> float:
    """Backward-compatible alias for the legacy linear warmup behavior."""
    return compute_asce_schedule_weight(
        current_step=current_step,
        total_steps=total_steps,
        base_weight=base_weight,
        warmup_fraction=warmup_fraction,
        schedule_cfg={"type": "linear_warmup"},
    )


# ============================================================
# Config resolver
# ============================================================

_DEFAULTS = {
    "enabled": False,
    "model_dir": "runs/exp1/K=50/author_classifier",
    "target_source": "reference_mean",
    "normalize_targets": True,
    # loss_type:
    #   "cosine"        (phase 1, original) -- 1 - cos(proj, own_reference_mean)
    #   "mse"           (phase 1)           -- mse(proj, own_reference_mean)
    #   "classifier_ce" (phase 2, v4+)      -- softmax CE over all K author
    #       reference vectors; equivalent to KL from one-hot author label.
    "loss_type": "cosine",
    "weight": 0.05,
    "warmup_fraction": 0.2,
    "min_supervised_tokens": 8,
    "pooling": "mean_supervised_tokens",
    "schedule": {
        "type": "linear_warmup",
        "peak_multiplier": 1.0,
        "peak_fraction": None,
        "cooldown_fraction": 0.0,
    },
    "cache_dir": "asce_target_cache",
    "fail_policy": "error",
    # Optional off-the-shelf target encoders used as drop-in alignment targets.
    # When external_backend is None, the standard ASCE/ArcFace scorer path is
    # used and these fields are ignored.
    "external_backend": None,  # None | "luar" | "star"
    "max_seq_len": 512,
    "tokenizer_name": None,
    "trust_remote_code": True,
    "external_batch_size": 32,
    # Only consumed when loss_type == "classifier_ce".
    #   ce_reference_source: "weight_vectors" (trained classifier head
    #       weights, most discriminative) or "prototypes" (empirical
    #       class means, smoother).
    #   margin_m: angular margin on target logit. 0.0 = plain softmax-CE
    #       over scaled cosines; 0.35 = full ArcFace (Deng et al., 2019).
    "ce_reference_source": "weight_vectors",
    "margin_m": 0.0,
    "projector": {
        "type": "linear",
        "use_layer_norm": False,
        "dropout": 0.0,
        "hidden_dim": None,  # only consumed when type == "mlp_2"; None -> auto
    },
    "logging": {
        "save_projector_state": True,
        "debug_artifacts": True,
    },
}


def resolve_asce_alignment_config(raw_cfg: Optional[dict] = None) -> dict:
    """Resolve and validate the asce_private_alignment config block.

    Returns a dict with all defaults filled in.
    """
    if raw_cfg is None:
        raw_cfg = {}

    resolved = {}
    for key, default in _DEFAULTS.items():
        if isinstance(default, dict):
            resolved[key] = dict(default)
            if key in raw_cfg and isinstance(raw_cfg[key], dict):
                resolved[key].update(raw_cfg[key])
        else:
            resolved[key] = raw_cfg.get(key, default)

    # Type coercions
    resolved["enabled"] = bool(resolved["enabled"])
    resolved["normalize_targets"] = bool(resolved["normalize_targets"])
    resolved["weight"] = float(resolved["weight"])
    resolved["warmup_fraction"] = float(resolved["warmup_fraction"])
    resolved["min_supervised_tokens"] = int(resolved["min_supervised_tokens"])
    resolved["pooling"] = str(resolved["pooling"]).lower()
    if resolved["pooling"] == "mean":
        resolved["pooling"] = "mean_supervised_tokens"
    elif resolved["pooling"] == "per_token":
        resolved["pooling"] = "per_token_supervised"
    resolved["margin_m"] = float(resolved["margin_m"])
    resolved["loss_type"] = str(resolved["loss_type"]).lower()
    resolved["ce_reference_source"] = str(resolved["ce_reference_source"]).lower()
    resolved["target_source"] = str(resolved["target_source"]).lower()
    external_backend = resolved.get("external_backend")
    resolved["external_backend"] = (
        None if external_backend in (None, "", "none", "None")
        else str(external_backend).lower()
    )
    tokenizer_name = resolved.get("tokenizer_name")
    resolved["tokenizer_name"] = (
        None if tokenizer_name in (None, "", "none", "None")
        else str(tokenizer_name)
    )
    resolved["max_seq_len"] = int(resolved["max_seq_len"])
    resolved["trust_remote_code"] = bool(resolved["trust_remote_code"])
    resolved["external_batch_size"] = int(resolved["external_batch_size"])
    schedule_cfg = dict(resolved.get("schedule") or {})
    schedule_cfg["type"] = str(schedule_cfg.get("type", "linear_warmup")).lower()
    schedule_cfg["peak_multiplier"] = float(schedule_cfg.get("peak_multiplier", 1.0))
    peak_fraction = schedule_cfg.get("peak_fraction")
    schedule_cfg["peak_fraction"] = (
        None if peak_fraction is None else float(peak_fraction)
    )
    schedule_cfg["cooldown_fraction"] = float(
        schedule_cfg.get("cooldown_fraction", 0.0)
    )
    if schedule_cfg["peak_fraction"] is None:
        schedule_cfg["peak_fraction"] = max(
            0.0,
            1.0 - resolved["warmup_fraction"] - schedule_cfg["cooldown_fraction"],
        )
    resolved["schedule"] = schedule_cfg

    if resolved["loss_type"] not in ("cosine", "mse", "classifier_ce"):
        raise ValueError(
            f"asce_private_alignment.loss_type must be one of "
            f"['cosine', 'mse', 'classifier_ce']; got {resolved['loss_type']!r}"
        )
    if resolved["external_backend"] not in (None, "luar", "star"):
        raise ValueError(
            "asce_private_alignment.external_backend must be one of "
            "[None, 'luar', 'star']; "
            f"got {resolved['external_backend']!r}"
        )
    if resolved["max_seq_len"] <= 0:
        raise ValueError(
            f"asce_private_alignment.max_seq_len must be > 0; "
            f"got {resolved['max_seq_len']}"
        )
    if resolved["external_batch_size"] <= 0:
        raise ValueError(
            f"asce_private_alignment.external_batch_size must be > 0; "
            f"got {resolved['external_batch_size']}"
        )
    if resolved["pooling"] not in ("mean_supervised_tokens", "per_token_supervised"):
        raise ValueError(
            "asce_private_alignment.pooling must be one of "
            "['mean_supervised_tokens', 'per_token_supervised']; "
            f"got {resolved['pooling']!r}"
        )
    if resolved["ce_reference_source"] not in ("weight_vectors", "prototypes"):
        raise ValueError(
            f"asce_private_alignment.ce_reference_source must be one of "
            f"['weight_vectors', 'prototypes']; got "
            f"{resolved['ce_reference_source']!r}"
        )
    if resolved["target_source"] not in (
        "mean",
        "reference_mean",
        "empirical_per_client",
        "client_embedding_mean",
    ):
        raise ValueError(
            "asce_private_alignment.target_source must be one of "
            "['mean', 'reference_mean', 'empirical_per_client', "
            "'client_embedding_mean']; "
            f"got {resolved['target_source']!r}"
        )
    if resolved["margin_m"] < 0.0 or resolved["margin_m"] > 0.5:
        raise ValueError(
            f"asce_private_alignment.margin_m must be in [0.0, 0.5]; "
            f"got {resolved['margin_m']}"
        )
    if resolved["schedule"]["type"] not in ("linear_warmup", "warmup_peak_cooldown"):
        raise ValueError(
            "asce_private_alignment.schedule.type must be one of "
            "['linear_warmup', 'warmup_peak_cooldown']; "
            f"got {resolved['schedule']['type']!r}"
        )
    if resolved["schedule"]["peak_multiplier"] <= 0.0:
        raise ValueError(
            "asce_private_alignment.schedule.peak_multiplier must be > 0; "
            f"got {resolved['schedule']['peak_multiplier']}"
        )
    peak_fraction = float(resolved["schedule"]["peak_fraction"])
    cooldown_fraction = float(resolved["schedule"]["cooldown_fraction"])
    if peak_fraction < 0.0 or peak_fraction > 1.0:
        raise ValueError(
            "asce_private_alignment.schedule.peak_fraction must be in [0, 1]; "
            f"got {peak_fraction}"
        )
    if cooldown_fraction < 0.0 or cooldown_fraction > 1.0:
        raise ValueError(
            "asce_private_alignment.schedule.cooldown_fraction must be in [0, 1]; "
            f"got {cooldown_fraction}"
        )
    total_fraction = resolved["warmup_fraction"] + peak_fraction + cooldown_fraction
    if total_fraction > 1.0 + 1.0e-6:
        raise ValueError(
            "asce_private_alignment schedule fractions must sum to <= 1.0; "
            f"got warmup + peak + cooldown = {total_fraction:.4f}"
        )

    return resolved
