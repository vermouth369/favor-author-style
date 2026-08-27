#!/usr/bin/env python3
"""External authorship encoders as private-alignment scorers.

This module adapts the non-ArcFace evaluator encoders to the small scorer API
consumed by ``44_train_favor.py`` for private-branch alignment.
"""

from __future__ import annotations

from typing import Iterable, Optional


class ExternalAlignmentScorer:
    """Expose LUAR/STAR encoders through the alignment-scorer API."""

    def __init__(self, encoder, *, backend: str) -> None:
        self._enc = encoder
        self.backend = str(backend).lower()
        probe = self._enc.encode_texts(["probe sentence."], batch_size=1)
        if probe.ndim != 2 or probe.shape[1] <= 0:
            raise RuntimeError(
                f"{self.backend} probe returned invalid shape {getattr(probe, 'shape', None)}"
            )
        self.embedding_dim = int(probe.shape[1])
        self.scale_s = 30.0  # API parity with ArcFace scorers; unused for cosine loss.

    def encode_texts(self, texts: Iterable[str], batch_size: int = 32):
        return self._enc.encode_texts(list(texts), batch_size=batch_size)

    def eval(self):
        model = getattr(self._enc, "model", None)
        if model is not None and hasattr(model, "eval"):
            model.eval()
        return self

    def to(self, device):
        model = getattr(self._enc, "model", None)
        if model is not None and hasattr(model, "to"):
            model.to(device)
            self._enc.device = str(device)
        return self


def load_external_alignment_scorer(
    backend: str,
    model_name: str,
    *,
    max_seq_len: int = 512,
    tokenizer_name: Optional[str] = None,
    trust_remote_code: bool = True,
    device: Optional[str] = None,
) -> ExternalAlignmentScorer:
    """Load a frozen external authorship encoder for private alignment."""

    normalized = str(backend).lower()
    if normalized == "luar":
        from non_asce_eval.encoders import LUAREncoder

        encoder = LUAREncoder(
            model_name,
            device=device,
            max_seq_len=max_seq_len,
            trust_remote_code=trust_remote_code,
        )
    elif normalized == "star":
        from non_asce_eval.encoders import STAREncoder

        encoder = STAREncoder(
            model_name,
            tokenizer_name=(tokenizer_name or "roberta-large"),
            device=device,
            max_seq_len=max_seq_len,
            trust_remote_code=trust_remote_code,
        )
    else:
        raise ValueError(f"Unknown external alignment backend: {backend!r}")
    return ExternalAlignmentScorer(encoder, backend=normalized)
