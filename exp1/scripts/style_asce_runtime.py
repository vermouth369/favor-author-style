#!/usr/bin/env python3
"""Shared runtime helpers for legacy classifiers and ArcFace style scorers."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoModelForSequenceClassification, AutoTokenizer


ARC_FACE_BACKEND = "arcface"
LEGACY_BACKEND = "legacy_softmax_classifier"
DEFAULT_MAX_SEQ_LEN = 256
LOCAL_CLASSIFIER_DIR_HINTS = {
    "exp1_asce_full": ("exp1",),
}


def _resolve_device(device: Optional[str] = None) -> torch.device:
    if device:
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def resolve_encoder_reference(encoder_name, model_dir) -> str:
    """Resolve repo-relative encoder references stored in ArcFace metadata."""
    raw = str(encoder_name)
    path = Path(raw).expanduser()
    candidates = [path]
    if not path.is_absolute():
        model_path = Path(model_dir)
        candidates.append(model_path / path)
        candidates.extend(parent / path for parent in model_path.parents)

    seen = set()
    for candidate in candidates:
        key = str(candidate)
        if key in seen:
            continue
        seen.add(key)
        if candidate.is_dir() and (candidate / "config.json").exists():
            return str(candidate)
        if candidate.exists():
            return str(candidate)
    return raw


def _has_classifier_artifacts(model_path: Path) -> bool:
    """Return True when a directory has persisted classifier artifacts."""
    return (model_path / "backend_meta.json").exists() or (model_path / "config.json").exists()


def _looks_like_local_model_reference(model_dir) -> bool:
    """Heuristic to distinguish local filesystem paths from HF repo ids."""
    raw = str(model_dir).strip()
    path = Path(raw).expanduser()
    if path.is_absolute():
        return True
    if raw.startswith((".", "~")):
        return True

    parts = [part for part in path.parts if part and part != path.anchor]
    if len(parts) > 2:
        return True
    if parts and parts[0] in {"runs", "data", "config", "checkpoints", "models"}:
        return True
    return any("=" in part for part in parts)


def resolve_classifier_model_dir(model_dir) -> tuple[str, List[str]]:
    """Normalize a classifier directory reference."""
    normalized = str(Path(str(model_dir)).expanduser())
    return normalized, [normalized]


def _detect_backend_from_artifact_dir(model_path: Path) -> str:
    """Infer backend for an already-existing local artifact directory."""
    backend_meta_path = model_path / "backend_meta.json"
    if backend_meta_path.exists():
        return str(_read_json(backend_meta_path).get("backend", ARC_FACE_BACKEND))

    config_path = model_path / "config.json"
    if config_path.exists():
        try:
            config = _read_json(config_path)
        except Exception:
            config = {}
        model_type = str(config.get("model_type", ""))
        backend = str(config.get("backend", ""))
        if model_type == "style_asce" or backend == ARC_FACE_BACKEND:
            return ARC_FACE_BACKEND

    return LEGACY_BACKEND


def _suggest_nearby_classifier_artifacts(model_dir) -> List[str]:
    """List nearby classifier dirs that may explain a missing-path failure."""
    raw_path = Path(str(model_dir)).expanduser()
    raw_parts = list(raw_path.parts)
    suggestions = []
    seen = set()

    for idx, part in enumerate(raw_parts):
        for replacement in LOCAL_CLASSIFIER_DIR_HINTS.get(part, ()):
            alias_parts = list(raw_parts)
            alias_parts[idx] = replacement
            candidate = Path(*alias_parts)
            if not _has_classifier_artifacts(candidate):
                continue
            key = str(candidate)
            if key in seen:
                continue
            seen.add(key)
            backend = _detect_backend_from_artifact_dir(candidate)
            suggestions.append(f"{candidate} (backend={backend})")

    return suggestions


def l2_normalize(tensor: torch.Tensor, eps: float = 1.0e-8) -> torch.Tensor:
    """L2-normalize the final dimension."""
    return F.normalize(tensor, p=2, dim=-1, eps=eps)


def mean_pool_last_hidden_state(
    last_hidden_state: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """Mean-pool token embeddings with attention-mask weighting."""
    if attention_mask is None:
        return last_hidden_state.mean(dim=1)

    mask = attention_mask.unsqueeze(-1).to(last_hidden_state.dtype)
    summed = (last_hidden_state * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp_min(1.0)
    return summed / counts


@dataclass
class BinaryCalibrator:
    """Simple binary calibrator over a raw scalar score."""

    method: str = "temperature"
    scale: float = 1.0
    bias: float = 0.0

    def predict_proba(self, raw_scores) -> np.ndarray:
        scores = np.asarray(raw_scores, dtype=np.float32)
        logits = (self.scale * scores) + self.bias
        probs = 1.0 / (1.0 + np.exp(-np.clip(logits, -50.0, 50.0)))
        return probs.astype(np.float32)

    def to_dict(self) -> dict:
        return {
            "method": self.method,
            "scale": float(self.scale),
            "bias": float(self.bias),
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "BinaryCalibrator":
        return cls(
            method=str(payload.get("method", "temperature")),
            scale=float(payload.get("scale", 1.0)),
            bias=float(payload.get("bias", 0.0)),
        )


def fit_binary_calibrator(
    raw_scores,
    labels,
    method: str = "temperature",
    max_iter: int = 250,
) -> BinaryCalibrator:
    """Fit a binary score calibrator with a tiny torch optimization loop."""
    score_arr = np.asarray(raw_scores, dtype=np.float32)
    label_arr = np.asarray(labels, dtype=np.float32)

    if score_arr.size == 0 or len(np.unique(label_arr)) < 2:
        return BinaryCalibrator(method=method, scale=1.0, bias=0.0)

    x = torch.tensor(score_arr, dtype=torch.float32)
    y = torch.tensor(label_arr, dtype=torch.float32)

    if method == "temperature":
        log_temperature = nn.Parameter(torch.zeros(1))
        params = [log_temperature]

        def forward_logits():
            temperature = torch.exp(log_temperature).clamp_min(1.0e-4)
            return x / temperature

    else:
        scale = nn.Parameter(torch.ones(1))
        bias = nn.Parameter(torch.zeros(1))
        params = [scale, bias]

        def forward_logits():
            return (x * scale) + bias

    optimizer = torch.optim.LBFGS(
        params,
        lr=0.1,
        max_iter=max_iter,
        line_search_fn="strong_wolfe",
    )

    def closure():
        optimizer.zero_grad()
        logits = forward_logits()
        loss = F.binary_cross_entropy_with_logits(logits, y)
        loss.backward()
        return loss

    optimizer.step(closure)

    if method == "temperature":
        temperature = float(torch.exp(log_temperature.detach()).cpu().item())
        return BinaryCalibrator(
            method="temperature",
            scale=1.0 / max(temperature, 1.0e-4),
            bias=0.0,
        )

    return BinaryCalibrator(
        method=str(method),
        scale=float(scale.detach().cpu().item()),
        bias=float(bias.detach().cpu().item()),
    )


class ASCEEncoderModel(nn.Module):
    """Encoder -> projection -> optional layer norm -> L2 normalize."""

    def __init__(
        self,
        encoder_name: str,
        embedding_dim: int,
        dropout: float = 0.1,
        use_layer_norm: bool = True,
        normalize_embeddings: bool = True,
    ) -> None:
        super().__init__()
        self.encoder_name = encoder_name
        self.embedding_dim = int(embedding_dim)
        self.dropout = nn.Dropout(float(dropout))
        self.use_layer_norm = bool(use_layer_norm)
        self.normalize_embeddings = bool(normalize_embeddings)

        self.encoder = AutoModel.from_pretrained(encoder_name)
        hidden_size = int(self.encoder.config.hidden_size)
        self.projection = nn.Linear(hidden_size, self.embedding_dim)
        self.layer_norm = nn.LayerNorm(self.embedding_dim) if self.use_layer_norm else None

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        pooled = mean_pool_last_hidden_state(outputs.last_hidden_state, attention_mask)
        projected = self.projection(self.dropout(pooled))
        if self.layer_norm is not None:
            projected = self.layer_norm(projected)
        if self.normalize_embeddings:
            projected = l2_normalize(projected)
        return projected


class ArcMarginProduct(nn.Module):
    """ArcFace margin head."""

    def __init__(
        self,
        embedding_dim: int,
        num_classes: int,
        scale_s: float = 30.0,
        margin_m: float = 0.35,
    ) -> None:
        super().__init__()
        self.embedding_dim = int(embedding_dim)
        self.num_classes = int(num_classes)
        self.scale_s = float(scale_s)
        self.margin_m = float(margin_m)
        self.weight = nn.Parameter(torch.empty(self.num_classes, self.embedding_dim))
        nn.init.xavier_uniform_(self.weight)

    def cosine_scores(self, embeddings: torch.Tensor) -> torch.Tensor:
        return F.linear(l2_normalize(embeddings), l2_normalize(self.weight))

    def forward(
        self,
        embeddings: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        apply_margin: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        cosine = self.cosine_scores(embeddings).clamp(-1.0 + 1.0e-7, 1.0 - 1.0e-7)
        if labels is None or not apply_margin:
            return cosine * self.scale_s, cosine

        one_hot = F.one_hot(labels, num_classes=self.num_classes).to(cosine.dtype)
        theta = torch.acos(cosine)
        target = torch.cos(theta + self.margin_m)
        logits = (one_hot * target) + ((1.0 - one_hot) * cosine)
        return logits * self.scale_s, cosine

    def normalized_weights(self) -> np.ndarray:
        weights = l2_normalize(self.weight.detach()).cpu().numpy().astype(np.float32)
        return weights


def detect_classifier_backend(model_dir) -> str:
    """Infer whether a directory contains legacy or ArcFace artifacts."""
    resolved_model_dir, _ = resolve_classifier_model_dir(model_dir)
    model_path = Path(resolved_model_dir)
    backend_meta_path = model_path / "backend_meta.json"
    if backend_meta_path.exists():
        return str(_read_json(backend_meta_path).get("backend", ARC_FACE_BACKEND))

    config_path = model_path / "config.json"
    if config_path.exists():
        try:
            config = _read_json(config_path)
        except Exception:
            config = {}
        model_type = str(config.get("model_type", ""))
        backend = str(config.get("backend", ""))
        if model_type == "style_asce" or backend == ARC_FACE_BACKEND:
            return ARC_FACE_BACKEND

    return LEGACY_BACKEND


def classifier_artifact_exists(model_dir) -> bool:
    """Check whether a classifier directory has loadable artifacts."""
    resolved_model_dir, _ = resolve_classifier_model_dir(model_dir)
    return _has_classifier_artifacts(Path(resolved_model_dir))


def load_label_map(model_dir) -> Dict[str, str]:
    """Load integer-label -> class-name mapping when available."""
    path = Path(model_dir) / "label_map.json"
    if not path.exists():
        return {}
    payload = _read_json(path)
    return {str(key): str(value) for key, value in payload.items()}


def load_binary_calibrator(model_dir) -> Optional[BinaryCalibrator]:
    """Load calibration metadata if present."""
    path = Path(model_dir) / "calibration.json"
    if not path.exists():
        return None
    payload = _read_json(path)
    return BinaryCalibrator.from_dict(payload)


def write_asce_artifacts(
    output_dir,
    model: ASCEEncoderModel,
    tokenizer,
    backend_meta: dict,
    label_map: Optional[Dict[str, str]] = None,
    class_weight_vectors: Optional[np.ndarray] = None,
    class_prototypes: Optional[np.ndarray] = None,
    calibrator: Optional[BinaryCalibrator] = None,
) -> None:
    """Persist ArcFace model state and runtime metadata."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    tokenizer.save_pretrained(output_path)
    torch.save(model.state_dict(), output_path / "model_state.pt")

    if class_weight_vectors is not None:
        np.save(output_path / "class_weight_vectors.npy", class_weight_vectors.astype(np.float32))
    if class_prototypes is not None:
        np.save(output_path / "class_prototypes.npy", class_prototypes.astype(np.float32))
    if label_map is not None:
        _write_json(output_path / "label_map.json", label_map)
    if calibrator is not None:
        _write_json(output_path / "calibration.json", calibrator.to_dict())

    meta = dict(backend_meta)
    meta.setdefault("backend", ARC_FACE_BACKEND)
    meta.setdefault("model_type", "style_asce")
    _write_json(output_path / "backend_meta.json", meta)
    _write_json(output_path / "config.json", meta)


class BaseStyleScorer:
    """Common scorer interface."""

    def __init__(self, model_dir, task: Optional[str] = None, device: Optional[str] = None) -> None:
        self.model_dir = Path(model_dir)
        self.task = task
        self.device = _resolve_device(device)
        self.backend = LEGACY_BACKEND
        self.label_map = load_label_map(self.model_dir)
        self.max_seq_len = DEFAULT_MAX_SEQ_LEN

    @property
    def embedding_dim(self) -> int:
        """Return embedding dimension, or -1 if not available."""
        return -1

    def encode_texts(self, texts, batch_size: int = 32) -> np.ndarray:
        raise NotImplementedError

    def predict_authorship(self, texts, batch_size: int = 32) -> dict:
        raise NotImplementedError

    def predict_assistant(self, texts, batch_size: int = 32) -> dict:
        raise NotImplementedError


class LegacySequenceClassifierScorer(BaseStyleScorer):
    """Runtime wrapper for existing AutoModelForSequenceClassification artifacts."""

    def __init__(self, model_dir, task: Optional[str] = None, device: Optional[str] = None) -> None:
        super().__init__(model_dir=model_dir, task=task, device=device)
        self.backend = LEGACY_BACKEND
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_dir)
        self.model.to(self.device).eval()
        self.max_seq_len = int(getattr(self.model.config, "max_position_embeddings", DEFAULT_MAX_SEQ_LEN))
        self.max_seq_len = min(self.max_seq_len, DEFAULT_MAX_SEQ_LEN)
        self.calibrator = load_binary_calibrator(self.model_dir)

    def _forward_batches(self, texts, batch_size: int = 32) -> tuple[np.ndarray, np.ndarray]:
        all_logits: List[np.ndarray] = []
        all_probs: List[np.ndarray] = []
        for start in range(0, len(texts), batch_size):
            batch_texts = list(texts[start : start + batch_size])
            inputs = self.tokenizer(
                batch_texts,
                return_tensors="pt",
                truncation=True,
                max_length=self.max_seq_len,
                padding=True,
            ).to(self.device)
            with torch.no_grad():
                logits = self.model(**inputs).logits
                probs = torch.softmax(logits, dim=-1)
            all_logits.append(logits.cpu().numpy())
            all_probs.append(probs.cpu().numpy())
        logits_arr = np.concatenate(all_logits, axis=0) if all_logits else np.zeros((0, 0), dtype=np.float32)
        probs_arr = np.concatenate(all_probs, axis=0) if all_probs else np.zeros((0, 0), dtype=np.float32)
        return logits_arr, probs_arr

    def encode_texts(self, texts, batch_size: int = 32) -> np.ndarray:
        embeddings: List[np.ndarray] = []
        base_model = self.model.base_model
        for start in range(0, len(texts), batch_size):
            batch_texts = list(texts[start : start + batch_size])
            inputs = self.tokenizer(
                batch_texts,
                return_tensors="pt",
                truncation=True,
                max_length=self.max_seq_len,
                padding=True,
            ).to(self.device)
            with torch.no_grad():
                outputs = base_model(**inputs)
                pooled = mean_pool_last_hidden_state(outputs.last_hidden_state, inputs.get("attention_mask"))
                pooled = l2_normalize(pooled)
            embeddings.append(pooled.cpu().numpy().astype(np.float32))
        return np.concatenate(embeddings, axis=0) if embeddings else np.zeros((0, 1), dtype=np.float32)

    def predict_authorship(self, texts, batch_size: int = 32) -> dict:
        logits, probs = self._forward_batches(texts, batch_size=batch_size)
        if probs.size == 0:
            return {
                "backend": self.backend,
                "score_matrix": probs,
                "pred_indices": np.asarray([], dtype=np.int64),
                "pred_labels": [],
                "top1_scores": np.asarray([], dtype=np.float32),
                "top2_scores": np.asarray([], dtype=np.float32),
                "margins": np.asarray([], dtype=np.float32),
                "confidence_like": np.asarray([], dtype=np.float32),
            }

        pred_indices = np.argmax(probs, axis=-1)
        top1_scores = np.max(probs, axis=-1)
        top2_scores = np.partition(probs, -2, axis=-1)[:, -2] if probs.shape[1] > 1 else np.zeros_like(top1_scores)
        pred_labels = [self.label_map.get(str(idx)) for idx in pred_indices.tolist()]
        margins = top1_scores - top2_scores
        return {
            "backend": self.backend,
            "score_matrix": probs,
            "logits": logits,
            "pred_indices": pred_indices,
            "pred_labels": pred_labels,
            "top1_scores": top1_scores.astype(np.float32),
            "top2_scores": top2_scores.astype(np.float32),
            "margins": margins.astype(np.float32),
            "confidence_like": top1_scores.astype(np.float32),
        }

    def predict_assistant(self, texts, batch_size: int = 32) -> dict:
        logits, probs = self._forward_batches(texts, batch_size=batch_size)
        if probs.size == 0:
            return {
                "backend": self.backend,
                "assistant_score": np.asarray([], dtype=np.float32),
                "raw_margin": np.asarray([], dtype=np.float32),
                "pred_indices": np.asarray([], dtype=np.int64),
            }

        if probs.shape[1] > 1:
            default_scores = probs[:, 1]
            raw_margin = logits[:, 1] - logits[:, 0]
        else:
            default_scores = probs[:, 0]
            raw_margin = logits[:, 0]

        if self.calibrator is not None:
            assistant_score = self.calibrator.predict_proba(raw_margin)
            pred_indices = (assistant_score >= 0.5).astype(np.int64)
        else:
            assistant_score = default_scores.astype(np.float32)
            pred_indices = np.argmax(probs, axis=-1)

        return {
            "backend": self.backend,
            "assistant_score": assistant_score.astype(np.float32),
            "raw_margin": raw_margin.astype(np.float32),
            "pred_indices": pred_indices,
            "score_matrix": probs.astype(np.float32),
        }


class ASCEScorer(BaseStyleScorer):
    """Runtime wrapper for ArcFace-based style scorers."""

    def __init__(
        self,
        model_dir,
        task: Optional[str] = None,
        device: Optional[str] = None,
        use_empirical_prototypes: Optional[bool] = None,
    ) -> None:
        super().__init__(model_dir=model_dir, task=task, device=device)
        self.backend = ARC_FACE_BACKEND
        self.meta = _read_json(self.model_dir / "backend_meta.json")
        self.task = task or str(self.meta.get("task", "authorship"))
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_dir)
        self.max_seq_len = int(self.meta.get("max_seq_len", DEFAULT_MAX_SEQ_LEN))
        self.encoder_name = resolve_encoder_reference(self.meta["encoder_name"], self.model_dir)

        self.model = ASCEEncoderModel(
            encoder_name=self.encoder_name,
            embedding_dim=int(self.meta["embedding_dim"]),
            dropout=float(self.meta.get("dropout", 0.1)),
            use_layer_norm=bool(self.meta.get("use_layer_norm", True)),
            normalize_embeddings=bool(self.meta.get("normalize_embeddings", True)),
        )
        state_dict = torch.load(self.model_dir / "model_state.pt", map_location="cpu")
        self.model.load_state_dict(state_dict)
        self.model.to(self.device).eval()

        self.scale_s = float(self.meta.get("scale_s", 1.0))
        prototype_path = self.model_dir / "class_prototypes.npy"
        weight_path = self.model_dir / "class_weight_vectors.npy"
        use_prototypes = use_empirical_prototypes
        if use_prototypes is None:
            use_prototypes = bool(self.meta.get("use_empirical_prototypes_default", True))

        if use_prototypes and prototype_path.exists():
            vectors = np.load(prototype_path)
            self.reference_source = "class_prototypes.npy"
        elif weight_path.exists():
            vectors = np.load(weight_path)
            self.reference_source = "class_weight_vectors.npy"
        else:
            raise FileNotFoundError(
                f"No class vectors found in {self.model_dir}. "
                "Expected class_prototypes.npy or class_weight_vectors.npy."
            )

        vectors = vectors.astype(np.float32)
        vectors = vectors / np.clip(np.linalg.norm(vectors, axis=1, keepdims=True), 1.0e-8, None)
        self.reference_vectors = torch.tensor(vectors, dtype=torch.float32, device=self.device)
        self.calibrator = load_binary_calibrator(self.model_dir)

    @property
    def embedding_dim(self) -> int:
        """Return the ArcFace embedding dimension."""
        return int(self.model.embedding_dim)

    def encode_texts(self, texts, batch_size: int = 32) -> np.ndarray:
        embeddings: List[np.ndarray] = []
        for start in range(0, len(texts), batch_size):
            batch_texts = list(texts[start : start + batch_size])
            inputs = self.tokenizer(
                batch_texts,
                return_tensors="pt",
                truncation=True,
                max_length=self.max_seq_len,
                padding=True,
            ).to(self.device)
            with torch.no_grad():
                batch_embeddings = self.model(**inputs)
            embeddings.append(batch_embeddings.cpu().numpy().astype(np.float32))
        return np.concatenate(embeddings, axis=0) if embeddings else np.zeros((0, self.reference_vectors.shape[1]), dtype=np.float32)

    def encode_texts_normalized(self, texts, batch_size: int = 32) -> np.ndarray:
        """Encode texts and explicitly L2-normalize each row.

        The underlying ArcFace model already normalizes, but this method
        makes the contract explicit for training-time target computation.
        """
        embeddings = self.encode_texts(texts, batch_size=batch_size)
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.clip(norms, 1e-8, None)
        return (embeddings / norms).astype(np.float32)

    def _score_matrix(self, embeddings: np.ndarray) -> np.ndarray:
        if embeddings.size == 0:
            return np.zeros((0, self.reference_vectors.shape[0]), dtype=np.float32)
        emb_tensor = torch.tensor(embeddings, dtype=torch.float32, device=self.device)
        score_matrix = emb_tensor @ self.reference_vectors.T
        return score_matrix.cpu().numpy().astype(np.float32)

    def predict_authorship(self, texts, batch_size: int = 32) -> dict:
        embeddings = self.encode_texts(texts, batch_size=batch_size)
        score_matrix = self._score_matrix(embeddings)
        if score_matrix.size == 0:
            return {
                "backend": self.backend,
                "embeddings": embeddings,
                "score_matrix": score_matrix,
                "pred_indices": np.asarray([], dtype=np.int64),
                "pred_labels": [],
                "top1_scores": np.asarray([], dtype=np.float32),
                "top2_scores": np.asarray([], dtype=np.float32),
                "margins": np.asarray([], dtype=np.float32),
                "confidence_like": np.asarray([], dtype=np.float32),
            }

        pred_indices = np.argmax(score_matrix, axis=-1)
        sorted_scores = np.sort(score_matrix, axis=-1)
        top1_scores = sorted_scores[:, -1]
        top2_scores = sorted_scores[:, -2] if score_matrix.shape[1] > 1 else np.zeros_like(top1_scores)
        pred_labels = [self.label_map.get(str(idx)) for idx in pred_indices.tolist()]
        margins = top1_scores - top2_scores
        confidence_like = np.clip((top1_scores + 1.0) / 2.0, 0.0, 1.0)
        return {
            "backend": self.backend,
            "embeddings": embeddings,
            "score_matrix": score_matrix,
            "pred_indices": pred_indices,
            "pred_labels": pred_labels,
            "top1_scores": top1_scores.astype(np.float32),
            "top2_scores": top2_scores.astype(np.float32),
            "margins": margins.astype(np.float32),
            "confidence_like": confidence_like.astype(np.float32),
        }

    def predict_assistant(self, texts, batch_size: int = 32) -> dict:
        embeddings = self.encode_texts(texts, batch_size=batch_size)
        score_matrix = self._score_matrix(embeddings)
        if score_matrix.size == 0:
            return {
                "backend": self.backend,
                "embeddings": embeddings,
                "assistant_score": np.asarray([], dtype=np.float32),
                "raw_margin": np.asarray([], dtype=np.float32),
                "pred_indices": np.asarray([], dtype=np.int64),
                "score_matrix": score_matrix,
            }

        if score_matrix.shape[1] > 1:
            raw_margin = score_matrix[:, 1] - score_matrix[:, 0]
        else:
            raw_margin = score_matrix[:, 0]

        if self.calibrator is not None:
            assistant_score = self.calibrator.predict_proba(raw_margin)
        else:
            scaled_margin = raw_margin * self.scale_s
            assistant_score = 1.0 / (1.0 + np.exp(-np.clip(scaled_margin, -50.0, 50.0)))

        pred_indices = (assistant_score >= 0.5).astype(np.int64)
        return {
            "backend": self.backend,
            "embeddings": embeddings,
            "assistant_score": assistant_score.astype(np.float32),
            "raw_margin": raw_margin.astype(np.float32),
            "pred_indices": pred_indices,
            "score_matrix": score_matrix,
        }


def load_style_scorer(
    model_dir,
    task: Optional[str] = None,
    device: Optional[str] = None,
    use_empirical_prototypes: Optional[bool] = None,
) -> BaseStyleScorer:
    """Load either a legacy or ArcFace classifier through a shared interface."""
    resolved_model_dir, attempted_candidates = resolve_classifier_model_dir(model_dir)
    if _has_classifier_artifacts(Path(resolved_model_dir)):
        model_dir = resolved_model_dir
    elif _looks_like_local_model_reference(model_dir):
        attempts = ", ".join(attempted_candidates) if attempted_candidates else str(model_dir)
        message = (
            f"No classifier artifacts found for local model_dir={str(model_dir)!r}. "
            f"Tried: {attempts}."
        )
        nearby = _suggest_nearby_classifier_artifacts(model_dir)
        if nearby:
            message += " Nearby classifier artifacts: " + "; ".join(nearby) + "."
        if "exp1_asce_full" in str(model_dir):
            message += (
                " Hint: `exp1_asce_full` is a separate ArcFace training output. "
                "Run `exp2/scripts/train_asce_full_k50.sh` to materialize it, "
                "or override the config if you intentionally want a different scorer."
            )
        raise FileNotFoundError(message)

    backend = detect_classifier_backend(model_dir)
    if backend == ARC_FACE_BACKEND:
        return ASCEScorer(
            model_dir=model_dir,
            task=task,
            device=device,
            use_empirical_prototypes=use_empirical_prototypes,
        )
    return LegacySequenceClassifierScorer(model_dir=model_dir, task=task, device=device)
