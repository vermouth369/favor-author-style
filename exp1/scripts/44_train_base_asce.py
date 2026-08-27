#!/usr/bin/env python3
"""44_train_base_asce.py — No-PFL ASCE baseline for Phase 1.

Base ASCE keeps the ASCE/ArcFace alignment objective from FAVoR, but trains
only a single global shared adapter. There is no private residual pack or
personalized generation path.

Each client minimizes:
  L_task + lambda_arc * L_asce_style + optional (mu/2) * ||A_local - A_global||^2

Only LoRA adapter deltas are uploaded/aggregated. The per-client ASCE projector
head is local-only and discarded before upload.

Usage:
  python 44_train_base_asce.py --config exp2/config/phase1/phase1_medium_seed2026_runtime.yaml \\
      --method-name "Base ASCE" --prox-mu 0.0
"""

import argparse
import copy
import hashlib
import json
import os
import sys
import time
from collections import OrderedDict

sys.setrecursionlimit(10000)

import numpy as np
import torch
import yaml
from datasets import Dataset
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType,
)
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

from favor_helpers import (
    active_roster_snapshot_path,
    canonical_author_id,
    estimate_gradient_steps,
    get_lora_state,
    load_lora_state,
    diff_lora_state,
    lora_delta_norm,
    log_adapter_state,
    count_trainable_params,
    load_client_manifest_with_noniid,
    record_matches_client_id,
)


# ============================================================
# BaseASCETrainer — Trainer with ASCE alignment + optional prox
# ============================================================

class BaseASCETrainer(Trainer):
    """Single-stage local trainer for Base ASCE.

    The projector is attached to the model as a local submodule before the
    trainer is constructed, so HF Trainer includes it in the optimizer. It is
    intentionally absent from the LoRA delta extracted after training.
    """

    def __init__(
        self, *args,
        server_state=None,
        prox_mu=0.0,
        asce_target_tensor=None,
        asce_weight=0.0,
        asce_projector=None,
        arcface_loss_type="cosine",
        asce_min_supervised_tokens=8,
        asce_warmup_fraction=0.0,
        asce_total_steps=0,
        asce_pooling="mean_supervised_tokens",
        asce_schedule_cfg=None,
        arcface_reference_vectors=None,
        arcface_class_index=None,
        arcface_scale_s=30.0,
        arcface_margin_m=0.0,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.server_state = server_state or {}
        self.prox_mu = float(prox_mu or 0.0)
        self._prox_loss_accum = 0.0
        self._prox_loss_count = 0

        self.asce_target_tensor = asce_target_tensor
        self.asce_weight = float(asce_weight or 0.0)
        self.asce_projector = asce_projector
        self.arcface_loss_type = str(arcface_loss_type or "cosine").lower()
        self.asce_min_supervised_tokens = int(asce_min_supervised_tokens)
        self.asce_warmup_fraction = float(asce_warmup_fraction or 0.0)
        self.asce_total_steps = int(asce_total_steps or 0)
        self.asce_pooling = str(asce_pooling or "mean_supervised_tokens").lower()
        self.asce_schedule_cfg = copy.deepcopy(asce_schedule_cfg or {})
        self.arcface_reference_vectors = arcface_reference_vectors
        self.arcface_class_index = arcface_class_index
        self.arcface_scale_s = float(arcface_scale_s)
        self.arcface_margin_m = float(arcface_margin_m)

        if self.arcface_loss_type == "classifier_ce":
            self._asce_enabled = bool(
                self.asce_weight > 0
                and self.asce_projector is not None
                and self.arcface_reference_vectors is not None
                and self.arcface_class_index is not None
            )
        else:
            self._asce_enabled = bool(
                self.asce_weight > 0
                and self.asce_projector is not None
                and self.asce_target_tensor is not None
            )

        self._asce_debug_emitted = False
        self._asce_loss_accum = 0.0
        self._asce_cosine_accum = 0.0
        self._arcface_ce_top1_accum = 0.0
        self._asce_steps = 0
        self._asce_weighted_accum = 0.0
        self._asce_active_steps = 0
        self._total_loss_when_asce_active_accum = 0.0
        self._task_loss_accum = 0.0
        self._total_loss_accum = 0.0
        self._loss_accum_steps = 0
        self._steps = 0

    def _compute_prox_loss(self, model, task_loss):
        if self.prox_mu <= 0 or not self.server_state:
            return torch.tensor(0.0, device=task_loss.device, dtype=task_loss.dtype)

        prox_raw = torch.tensor(0.0, device=task_loss.device, dtype=task_loss.dtype)
        for name, param in model.named_parameters():
            if not param.requires_grad or "lora_" not in name:
                continue
            if name not in self.server_state:
                continue
            ref = self.server_state[name].to(param.device, dtype=param.dtype)
            prox_raw = prox_raw + torch.sum((param - ref) ** 2)
        return (self.prox_mu / 2.0) * prox_raw

    def _compute_asce_style_loss(self, outputs, inputs, task_loss):
        if not self._asce_enabled:
            return torch.tensor(0.0, device=task_loss.device, dtype=task_loss.dtype)

        from asce_private_alignment import (
            select_supervised_token_hidden_states,
            prepare_asce_projector_inputs,
            compute_asce_style_loss,
            compute_asce_style_loss_per_token,
            compute_arcface_ce_loss,
            compute_arcface_ce_loss_per_token,
            compute_asce_schedule_weight,
        )

        if not hasattr(outputs, "hidden_states") or outputs.hidden_states is None:
            if not self._asce_debug_emitted:
                print("      [base_asce] WARNING: hidden_states unavailable; skipping ASCE")
                self._asce_debug_emitted = True
            return torch.tensor(0.0, device=task_loss.device, dtype=task_loss.dtype)

        labels = inputs.get("labels")
        if labels is None:
            if not self._asce_debug_emitted:
                print("      [base_asce] WARNING: labels unavailable; skipping ASCE")
                self._asce_debug_emitted = True
            return torch.tensor(0.0, device=task_loss.device, dtype=task_loss.dtype)

        final_hidden = outputs.hidden_states[-1]
        ce_top1_acc = 0.0

        if self.asce_pooling in ("per_token_supervised", "per_token"):
            selected = select_supervised_token_hidden_states(
                final_hidden,
                labels,
                min_tokens=self.asce_min_supervised_tokens,
            )
            if selected is None:
                return torch.tensor(0.0, device=task_loss.device, dtype=task_loss.dtype)
            token_hidden, batch_index = selected
            projected = self.asce_projector(token_hidden)

            if self.arcface_loss_type == "classifier_ce":
                ref_vecs = self.arcface_reference_vectors.to(
                    projected.device, dtype=projected.dtype,
                )
                loss, mean_cosine, ce_top1_acc = compute_arcface_ce_loss_per_token(
                    projected_per_token=projected,
                    batch_index=batch_index,
                    reference_vectors=ref_vecs,
                    class_index=int(self.arcface_class_index),
                    scale_s=self.arcface_scale_s,
                    margin_m=self.arcface_margin_m,
                )
            else:
                target = self.asce_target_tensor.to(projected.device)
                loss, mean_cosine = compute_asce_style_loss_per_token(
                    projected_per_token=projected,
                    batch_index=batch_index,
                    target=target,
                    loss_type=self.arcface_loss_type,
                )
        else:
            projector_inputs = prepare_asce_projector_inputs(
                final_hidden,
                labels,
                pooling=self.asce_pooling,
                min_tokens=self.asce_min_supervised_tokens,
            )
            if projector_inputs is None:
                return torch.tensor(0.0, device=task_loss.device, dtype=task_loss.dtype)
            projected = self.asce_projector(projector_inputs)

            if self.arcface_loss_type == "classifier_ce":
                ref_vecs = self.arcface_reference_vectors.to(
                    projected.device, dtype=projected.dtype,
                )
                loss, mean_cosine, ce_top1_acc = compute_arcface_ce_loss(
                    projected=projected,
                    reference_vectors=ref_vecs,
                    class_index=int(self.arcface_class_index),
                    scale_s=self.arcface_scale_s,
                    margin_m=self.arcface_margin_m,
                )
            else:
                target = self.asce_target_tensor.to(projected.device)
                loss, mean_cosine = compute_asce_style_loss(
                    projected, target, loss_type=self.arcface_loss_type,
                )

        effective_weight = compute_asce_schedule_weight(
            current_step=self._steps + 1,
            total_steps=self.asce_total_steps,
            base_weight=1.0,
            warmup_fraction=self.asce_warmup_fraction,
            schedule_cfg=self.asce_schedule_cfg,
        )
        loss = loss * effective_weight

        self._asce_loss_accum += float(loss.item())
        self._asce_cosine_accum += float(mean_cosine)
        self._arcface_ce_top1_accum += float(ce_top1_acc)
        self._asce_steps += 1

        if not self._asce_debug_emitted:
            extra = ""
            if self.arcface_loss_type == "classifier_ce":
                extra = (
                    f", class_index={self.arcface_class_index}, "
                    f"K={self.arcface_reference_vectors.size(0)}, "
                    f"scale_s={self.arcface_scale_s}, margin_m={self.arcface_margin_m}"
                )
            print(
                f"      [base_asce] active: projector "
                f"{self.asce_projector.hidden_size}->{self.asce_projector.embedding_dim}, "
                f"loss_type={self.arcface_loss_type}, pooling={self.asce_pooling}, "
                f"weight={self.asce_weight}, warmup={self.asce_warmup_fraction}, "
                f"schedule={self.asce_schedule_cfg.get('type', 'linear_warmup')}"
                f"{extra}"
            )
            self._asce_debug_emitted = True

        return loss

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        if self._asce_enabled:
            inputs = dict(inputs)
            inputs["output_hidden_states"] = True

        outputs = model(**inputs)
        task_loss = outputs.loss

        prox_loss = self._compute_prox_loss(model, task_loss)
        asce_loss = self._compute_asce_style_loss(outputs, inputs, task_loss)
        asce_weighted = self.asce_weight * asce_loss
        total_loss = task_loss + prox_loss + asce_weighted

        self._prox_loss_accum += float(prox_loss.item())
        self._prox_loss_count += 1
        self._task_loss_accum += float(task_loss.item())
        self._total_loss_accum += float(total_loss.item())
        self._loss_accum_steps += 1

        if self._asce_enabled:
            asce_contribution = float(asce_weighted.item())
            self._asce_weighted_accum += asce_contribution
            if asce_contribution != 0.0:
                self._total_loss_when_asce_active_accum += float(total_loss.item())
                self._asce_active_steps += 1

        self._steps += 1

        if return_outputs:
            return total_loss, outputs
        return total_loss

    @property
    def mean_prox_loss(self):
        if self._prox_loss_count == 0:
            return 0.0
        return self._prox_loss_accum / self._prox_loss_count

    def get_training_diagnostics(self):
        diag = {
            "mean_task_loss": (
                self._task_loss_accum / self._loss_accum_steps
                if self._loss_accum_steps else 0.0
            ),
            "mean_total_loss": (
                self._total_loss_accum / self._loss_accum_steps
                if self._loss_accum_steps else 0.0
            ),
            "mean_prox_loss": self.mean_prox_loss,
            "effective_steps": int(self._steps),
            "asce_style_status": (
                "enabled_but_no_steps" if self._asce_enabled else "disabled"
            ),
        }

        if self._asce_enabled:
            if self._asce_steps > 0:
                diag["asce_style_status"] = "active"
                diag["asce_style_loss"] = self._asce_loss_accum / self._asce_steps
                diag["asce_style_mean_cosine"] = (
                    self._asce_cosine_accum / self._asce_steps
                )
                diag["arcface_ce_top1_acc"] = (
                    self._arcface_ce_top1_accum / self._asce_steps
                    if self.arcface_loss_type == "classifier_ce" else 0.0
                )
            else:
                diag["asce_style_loss"] = 0.0
                diag["asce_style_mean_cosine"] = 0.0
                diag["arcface_ce_top1_acc"] = 0.0

            diag["asce_style_effective_steps"] = int(self._asce_steps)
            diag["asce_style_weight"] = float(self.asce_weight)
            diag["asce_style_loss_type"] = str(self.arcface_loss_type)
            diag["asce_style_pooling"] = str(self.asce_pooling)
            diag["asce_style_schedule_type"] = str(
                self.asce_schedule_cfg.get("type", "linear_warmup")
            )
            if self._loss_accum_steps:
                diag["asce_weighted_loss_mean"] = (
                    self._asce_weighted_accum / self._loss_accum_steps
                )
                if self._asce_active_steps > 0:
                    denom = abs(self._total_loss_when_asce_active_accum) + 1e-12
                else:
                    denom = abs(self._total_loss_accum) + 1e-12
                diag["asce_contribution_ratio"] = (
                    abs(self._asce_weighted_accum) / denom
                )
                diag["asce_active_steps"] = int(self._asce_active_steps)
            else:
                diag["asce_weighted_loss_mean"] = 0.0
                diag["asce_contribution_ratio"] = 0.0
                diag["asce_active_steps"] = 0
        else:
            diag["asce_style_loss"] = 0.0
            diag["asce_style_mean_cosine"] = 0.0
            diag["arcface_ce_top1_acc"] = 0.0
            diag["asce_style_effective_steps"] = 0
            diag["asce_style_weight"] = 0.0
            diag["asce_style_loss_type"] = "disabled"
            diag["asce_style_pooling"] = "disabled"
            diag["asce_style_schedule_type"] = "disabled"
            diag["asce_weighted_loss_mean"] = 0.0
            diag["asce_contribution_ratio"] = 0.0
            diag["asce_active_steps"] = 0

        return diag


# ============================================================
# Utility functions
# ============================================================

def load_base_model_and_tokenizer(model_cfg):
    """Load quantized base model and tokenizer."""
    model_name = model_cfg["model"]["name"]
    quant_cfg = model_cfg.get("quantization", {})
    bnb_config = None
    if quant_cfg.get("load_in_4bit", False):
        compute_dtype = getattr(
            torch, quant_cfg.get("bnb_4bit_compute_dtype", "bfloat16")
        )
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=quant_cfg.get("bnb_4bit_quant_type", "nf4"),
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=quant_cfg.get(
                "bnb_4bit_use_double_quant", True
            ),
        )
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=model_cfg["model"].get("trust_remote_code", False),
        torch_dtype=torch.bfloat16 if bnb_config is None else None,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=model_cfg["model"].get("trust_remote_code", False),
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if bnb_config is not None:
        model = prepare_model_for_kbit_training(model)
    return model, tokenizer


def build_lora_config(peft_cfg):
    """Build LoRA config from YAML peft section."""
    return LoraConfig(
        r=peft_cfg["r"],
        lora_alpha=peft_cfg["alpha"],
        lora_dropout=peft_cfg["dropout"],
        target_modules=peft_cfg["target_modules"],
        bias=peft_cfg.get("bias", "none"),
        task_type=TaskType.CAUSAL_LM,
    )


def stable_client_round_seed(seed, round_idx, client_id):
    """Deterministic per-client seed independent of Python hash randomization."""
    digest = hashlib.sha256(str(client_id).encode("utf-8")).hexdigest()
    client_offset = int(digest[:8], 16) % 100000
    return int(seed) + int(round_idx) * 100000 + client_offset


def resolve_round_client_sample(
    all_client_ids,
    clients_per_round,
    rng,
    sample_mode="uniform",
    seen_clients=None,
):
    """Select round participants with optional uncovered-first sampling."""
    all_client_ids = list(all_client_ids)
    if clients_per_round >= len(all_client_ids):
        return list(all_client_ids)

    sample_mode = str(sample_mode or "uniform").lower()
    seen_clients = set(seen_clients or [])

    if sample_mode == "uniform":
        return list(rng.choice(all_client_ids, clients_per_round, replace=False))

    if sample_mode != "stratified_uncovered":
        raise ValueError(
            f"Unknown fed.sample_mode={sample_mode!r}; "
            "expected 'uniform' or 'stratified_uncovered'."
        )

    uncovered = [cid for cid in all_client_ids if cid not in seen_clients]
    covered = [cid for cid in all_client_ids if cid in seen_clients]

    selected = []
    if uncovered:
        take = min(len(uncovered), clients_per_round)
        selected.extend(list(rng.choice(uncovered, take, replace=False)))

    remaining = clients_per_round - len(selected)
    if remaining > 0:
        pool = [cid for cid in covered if cid not in selected]
        if len(pool) < remaining:
            pool = [cid for cid in all_client_ids if cid not in selected]
        selected.extend(list(rng.choice(pool, remaining, replace=False)))

    if len(selected) > 1:
        selected = list(np.asarray(selected)[rng.permutation(len(selected))])
    return list(selected)


def load_all_client_texts(pooled_dir, client_ids):
    """Scan pooled train.jsonl once and collect texts for each active client."""
    train_path = os.path.join(pooled_dir, "train.jsonl")
    client_ids = list(client_ids)
    texts_by_client = {cid: [] for cid in client_ids}
    with open(train_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            text = rec.get("text", "")
            if not text:
                continue
            for cid in client_ids:
                if record_matches_client_id(rec, cid):
                    texts_by_client[cid].append(text)
                    break
    return texts_by_client


def dataset_from_texts(texts, tokenizer, max_seq_len):
    """Tokenize raw client texts into a causal-LM training Dataset."""
    if not texts:
        return None
    ds = Dataset.from_dict({"text": texts})

    def tokenize_fn(examples):
        return tokenizer(
            examples["text"], truncation=True,
            max_length=max_seq_len, padding=False,
        )

    ds = ds.map(tokenize_fn, batched=True, remove_columns=["text"])
    return ds


def build_asce_target_cache(
    client_texts,
    asce_align_cfg,
    scorer,
    cache_dir,
):
    """Build frozen per-client ASCE targets once before federated rounds."""
    from asce_private_alignment import cache_client_asce_target

    target_cache = {}
    target_meta = {}
    for cid, texts in client_texts.items():
        if not texts:
            print(f"  [base_asce] WARNING: no reference texts for {cid}; ASCE disabled for client")
            continue
        target_vector, meta = cache_client_asce_target(
            client_id=cid,
            texts=texts,
            scorer=scorer,
            cache_dir=cache_dir,
            normalize=asce_align_cfg.get("normalize_targets", True),
            strategy=asce_align_cfg.get("target_source", "reference_mean"),
            model_dir=asce_align_cfg.get("model_dir", ""),
        )
        target_cache[cid] = target_vector
        target_meta[cid] = meta
    return target_cache, target_meta


def write_target_collision_report(target_cache, output_dir):
    """Log cosine overlap among frozen client targets as a conflict diagnostic."""
    if len(target_cache) < 2:
        return None
    os.makedirs(output_dir, exist_ok=True)
    client_ids = list(target_cache.keys())
    matrix = np.stack([target_cache[cid] for cid in client_ids]).astype(np.float32)
    matrix = matrix / np.clip(np.linalg.norm(matrix, axis=1, keepdims=True), 1e-8, None)
    cosine = matrix @ matrix.T
    off_diag = cosine[~np.eye(len(client_ids), dtype=bool)]
    report = {
        "num_clients": len(client_ids),
        "mean_offdiag_cosine": float(np.mean(off_diag)),
        "max_offdiag_cosine": float(np.max(off_diag)),
        "min_offdiag_cosine": float(np.min(off_diag)),
        "p95_offdiag_cosine": float(np.percentile(off_diag, 95)),
    }
    path = os.path.join(output_dir, "asce_target_collisions.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(
        "  [base_asce] target collision diag: "
        f"mean={report['mean_offdiag_cosine']:.4f}, "
        f"p95={report['p95_offdiag_cosine']:.4f}, "
        f"max={report['max_offdiag_cosine']:.4f}"
    )
    return path


# ============================================================
# Client-side training with ASCE alignment
# ============================================================

def train_client_local_base_asce(
    model, tokenizer, global_adapter_dict, client_ds,
    fed_cfg, output_dir, seed, prox_mu=0.0, asce_bundle=None,
):
    """Train one client locally with task CE + ASCE + optional proximal penalty.

    Returns:
        content_delta: OrderedDict of Δ for the shared adapter
        train_loss: average training loss
        delta_norm: L2 norm of the delta
        diagnostics: auxiliary loss and ASCE health fields
    """
    os.makedirs(output_dir, exist_ok=True)

    model.set_adapter("default")
    load_lora_state(model, global_adapter_dict)
    log_adapter_state(model, "CLIENT_TRAIN_START")

    before_state = get_lora_state(model)

    # Build server reference state for optional proximal penalty. Keep this
    # LoRA-only so the local ASCE projector is never regularized/uploaded.
    server_state = {}
    for name, param in model.named_parameters():
        if param.requires_grad and "lora_" in name:
            server_state[name] = param.detach().clone()

    asce_bundle = asce_bundle or {}
    asce_enabled = bool(asce_bundle.get("enabled", False))
    asce_projector = None
    asce_target_tensor = None
    arcface_reference_vectors_tensor = None
    arcface_class_index = None
    arcface_scale_s = 30.0
    arcface_margin_m = 0.0
    asce_pooling = "mean_supervised_tokens"
    asce_schedule_cfg = {}

    if asce_enabled:
        from asce_private_alignment import build_projector

        target_np = asce_bundle.get("target_vector")
        if target_np is not None:
            asce_target_tensor = torch.as_tensor(target_np, dtype=torch.float32)

        asce_pooling = str(
            asce_bundle.get("pooling", "mean_supervised_tokens")
        ).lower()
        asce_schedule_cfg = copy.deepcopy(asce_bundle.get("schedule") or {})
        asce_projector = build_projector(
            hidden_size=int(asce_bundle["hidden_size"]),
            embedding_dim=int(asce_bundle["embedding_dim"]),
            projector_cfg=asce_bundle.get("projector_cfg"),
            device=next(model.parameters()).device,
        )
        model.asce_base_asce_projector = asce_projector

        if str(asce_bundle.get("loss_type", "cosine")).lower() == "classifier_ce":
            ref_np = asce_bundle.get("reference_vectors")
            if ref_np is None or asce_bundle.get("class_index") is None:
                raise RuntimeError(
                    "Base ASCE classifier_ce requested but reference_vectors or "
                    "class_index is missing from the client bundle"
                )
            arcface_reference_vectors_tensor = torch.as_tensor(
                ref_np, dtype=torch.float32,
            )
            arcface_class_index = int(asce_bundle["class_index"])
            arcface_scale_s = float(asce_bundle.get("scale_s", 30.0))
            arcface_margin_m = float(asce_bundle.get("margin_m", 0.0))
            print(
                f"    [base_asce] projector attached (CE): "
                f"{asce_projector.hidden_size}->{asce_projector.embedding_dim}, "
                f"weight={asce_bundle['weight']}, K={arcface_reference_vectors_tensor.size(0)}, "
                f"class_index={arcface_class_index}, pooling={asce_pooling}"
            )
        else:
            target_dim = int(target_np.shape[0]) if target_np is not None else 0
            print(
                f"    [base_asce] projector attached: "
                f"{asce_projector.hidden_size}->{asce_projector.embedding_dim}, "
                f"weight={asce_bundle['weight']}, target_dim={target_dim}, "
                f"pooling={asce_pooling}"
            )

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=False
    )
    training_args = TrainingArguments(
        output_dir=os.path.join(output_dir, "_train_tmp"),
        per_device_train_batch_size=fed_cfg["micro_batch_size"],
        gradient_accumulation_steps=fed_cfg["gradient_accumulation_steps"],
        learning_rate=fed_cfg["local_lr"],
        warmup_ratio=fed_cfg["warmup_ratio"],
        num_train_epochs=fed_cfg["local_epochs"],
        optim="adamw_torch",
        bf16=True,
        gradient_checkpointing=True,
        logging_steps=5,
        save_strategy="no",
        seed=seed,
        report_to="none",
        remove_unused_columns=False,
        max_grad_norm=1.0,
    )

    effective_steps = estimate_gradient_steps(len(client_ds), fed_cfg)

    trainer = BaseASCETrainer(
        model=model,
        args=training_args,
        train_dataset=client_ds,
        data_collator=data_collator,
        server_state=server_state,
        prox_mu=prox_mu,
        asce_target_tensor=asce_target_tensor,
        asce_weight=float(asce_bundle.get("weight", 0.0)) if asce_enabled else 0.0,
        asce_projector=asce_projector,
        arcface_loss_type=asce_bundle.get("loss_type", "cosine") if asce_enabled else "cosine",
        asce_min_supervised_tokens=int(asce_bundle.get("min_supervised_tokens", 8)) if asce_enabled else 8,
        asce_warmup_fraction=float(asce_bundle.get("warmup_fraction", 0.0)) if asce_enabled else 0.0,
        asce_total_steps=int(effective_steps),
        asce_pooling=asce_pooling,
        asce_schedule_cfg=asce_schedule_cfg,
        arcface_reference_vectors=arcface_reference_vectors_tensor,
        arcface_class_index=arcface_class_index,
        arcface_scale_s=arcface_scale_s,
        arcface_margin_m=arcface_margin_m,
    )
    result = trainer.train()
    train_loss = result.training_loss
    diagnostics = trainer.get_training_diagnostics()

    after_state = get_lora_state(model)
    content_delta = diff_lora_state(before_state, after_state)
    d_norm = lora_delta_norm(before_state, after_state)

    log_adapter_state(model, "CLIENT_TRAIN_END")
    print(
        f"    delta_norm: {d_norm:.6f}, "
        f"prox_loss: {diagnostics['mean_prox_loss']:.6f} (mu={prox_mu}), "
        f"asce_status={diagnostics['asce_style_status']}"
    )
    if asce_enabled:
        print(
            f"    asce_loss={diagnostics['asce_style_loss']:.6f}, "
            f"cosine={diagnostics['asce_style_mean_cosine']:.4f}, "
            f"contrib_ratio={diagnostics['asce_contribution_ratio']:.4f}"
        )

    for p in model.parameters():
        p.grad = None
    if hasattr(model, "asce_base_asce_projector"):
        del model.asce_base_asce_projector
    del trainer
    del server_state
    torch.cuda.empty_cache()

    return content_delta, train_loss, d_norm, diagnostics


# ============================================================
# Uniform FedAvg aggregation
# ============================================================

def aggregate_deltas_uniform(client_deltas, client_ids, data_sizes):
    """Uniform FedAvg aggregation (weighted by data size only)."""
    param_names = list(next(iter(client_deltas.values())).keys())
    agg_delta = OrderedDict()
    for name in param_names:
        agg_delta[name] = torch.zeros_like(
            next(iter(client_deltas.values()))[name]
        )
    total_weight = 0.0
    for cid in client_ids:
        if cid not in client_deltas:
            continue
        delta = client_deltas[cid]
        w = float(data_sizes.get(cid, 1))
        total_weight += w
        for name in param_names:
            agg_delta[name] += w * delta[name]
    if total_weight > 0:
        for name in param_names:
            agg_delta[name] /= total_weight
    return agg_delta


# ============================================================
# Main training loop
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="No-PFL ASCE baseline training"
    )
    parser.add_argument("--config", type=str, default="exp2/config/phase1/phase1_medium_seed2026_runtime.yaml")
    parser.add_argument("--gpu", type=int, default=None)
    parser.add_argument(
        "--method-name", type=str, default="Base ASCE",
        help="Method directory name (default: Base ASCE)"
    )
    parser.add_argument(
        "--prox-mu", type=float, default=None,
        help="Proximal penalty weight mu (overrides config)"
    )
    parser.add_argument(
        "--resume-round", type=int, default=0,
        help="Resume from a specific round (0 = start fresh)"
    )
    args = parser.parse_args()

    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)
    with open(cfg["paths"]["base_model_config"], "r") as f:
        model_cfg = yaml.safe_load(f)

    # ---- Config extraction ----
    prototype_dir = cfg["paths"]["prototype_dir"]
    pooled_dir = cfg["paths"]["pooled_dir"]
    runs_dir = cfg["paths"]["runs_dir"]
    fed_cfg = cfg["fed"]
    favor_cfg = cfg["favor"]
    peft_cfg = cfg["peft"]
    seed = cfg["seeds"]["training"]

    total_rounds = fed_cfg["rounds"]
    clients_per_round = fed_cfg["clients_per_round"]
    sample_mode = str(fed_cfg.get("sample_mode", "uniform")).lower()

    # Proximal strength: CLI > config > default
    prox_mu = args.prox_mu
    if prox_mu is None:
        prox_block = cfg.get("losses", {}).get("prox_shared", {})
        prox_mu = prox_block.get("weight", 0.0) if prox_block.get("enabled", False) else 0.0
    method_name = args.method_name

    method_dir = os.path.join(runs_dir, method_name)
    os.makedirs(method_dir, exist_ok=True)

    trainable_count = None

    print("=" * 60)
    print(f"No-PFL ASCE Training — {method_name}")
    print(f"  ASCE enabled: {bool(cfg.get('asce_private_alignment', {}).get('enabled', False))}")
    print(f"  Proximal mu: {prox_mu}")
    print(f"  Rounds: {total_rounds}")
    print(f"  Clients/round: {clients_per_round}")
    print(f"  Sample mode: {sample_mode}")
    print("=" * 60)

    # ---- Save resolved config ----
    resolved = {
        "config": cfg,
        "model_config": model_cfg,
        "method_name": method_name,
        "prox_mu": prox_mu,
        "implementation": "base_asce_global_adapter_only",
        "personalization": {
            "private_residual_pack": False,
        },
    }
    with open(os.path.join(method_dir, "config_resolved.yaml"), "w") as f:
        yaml.dump(resolved, f, default_flow_style=False)

    # ---- Load client manifest ----
    manifest_info = load_client_manifest_with_noniid(prototype_dir, cfg)
    all_client_ids = manifest_info["all_client_ids"]
    client_data_sizes = manifest_info["client_data_sizes"]
    noniid_level = manifest_info["non_iid_level"]
    resolved["noniid_manifest_path"] = manifest_info["non_iid_manifest_path"]
    resolved["noniid_level"] = noniid_level
    resolved["non_iid_downgrade"] = manifest_info.get("non_iid_downgrade", False)
    resolved["sample_mode"] = sample_mode
    with open(os.path.join(method_dir, "config_resolved.yaml"), "w") as f:
        yaml.dump(resolved, f, default_flow_style=False)
    print(f"\n  Total clients: {len(all_client_ids)} (non-IID: {noniid_level})")

    roster_snapshot = {
        "client_ids": list(all_client_ids),
        "non_iid_manifest_path": manifest_info["non_iid_manifest_path"],
        "non_iid_level": noniid_level,
        "non_iid_downgrade": manifest_info.get("non_iid_downgrade", False),
        "method": method_name,
        "seed": seed,
    }
    snapshot_path = active_roster_snapshot_path(runs_dir)
    os.makedirs(os.path.dirname(snapshot_path), exist_ok=True)
    with open(snapshot_path, "w", encoding="utf-8") as f:
        json.dump(roster_snapshot, f, indent=2)
    print(f"  Active roster snapshot: {snapshot_path}")

    # ---- Scan client training texts once (used for ASCE targets and datasets) ----
    print("\n  Loading client texts from pooled train split ...")
    client_texts = load_all_client_texts(pooled_dir, all_client_ids)
    nonempty_clients = sum(1 for texts in client_texts.values() if texts)
    print(f"  Client text cache: {nonempty_clients}/{len(all_client_ids)} clients have data")

    # ---- ArcFace / ASCE alignment setup ----
    asce_align_cfg = None
    asce_target_cache = {}
    arcface_ce_reference_vectors = None
    arcface_ce_label_map = {}
    arcface_ce_scale_s = 30.0
    arcface_ce_margin_m = 0.0
    asce_embedding_dim = None

    asce_align_raw = cfg.get("asce_private_alignment", {})
    if asce_align_raw.get("enabled", False):
        from asce_private_alignment import resolve_asce_alignment_config
        from style_asce_runtime import load_style_scorer as load_asce_scorer

        asce_align_cfg = resolve_asce_alignment_config(asce_align_raw)
        asce_model_dir = asce_align_cfg["model_dir"]
        print(f"\n  Loading ArcFace scorer for Base ASCE: {asce_model_dir}")
        try:
            asce_alignment_scorer = load_asce_scorer(
                asce_model_dir, task="authorship",
            )
            asce_embedding_dim = int(asce_alignment_scorer.embedding_dim)
            print(
                f"  ArcFace scorer loaded: backend={asce_alignment_scorer.backend}, "
                f"embedding_dim={asce_embedding_dim}"
            )
        except Exception as exc:
            if asce_align_cfg.get("fail_policy", "error") == "error":
                raise RuntimeError(
                    f"ArcFace private alignment enabled but scorer failed to load "
                    f"from {asce_model_dir}: {exc}"
                ) from exc
            print(f"  WARNING: ArcFace scorer failed ({exc}); disabling ASCE")
            asce_align_cfg["enabled"] = False
            asce_alignment_scorer = None

        if asce_align_cfg.get("enabled", False):
            asce_target_cache_dir = os.path.join(
                method_dir,
                "optional",
                asce_align_cfg.get("cache_dir", "asce_target_cache"),
            )
            os.makedirs(asce_target_cache_dir, exist_ok=True)
            asce_target_cache, asce_target_meta = build_asce_target_cache(
                client_texts=client_texts,
                asce_align_cfg=asce_align_cfg,
                scorer=asce_alignment_scorer,
                cache_dir=asce_target_cache_dir,
            )
            write_target_collision_report(
                asce_target_cache,
                os.path.join(method_dir, "optional"),
            )

            if str(asce_align_cfg.get("loss_type", "cosine")).lower() == "classifier_ce":
                from asce_private_alignment import load_classifier_head_vectors
                from style_asce_runtime import load_label_map

                ce_source = str(
                    asce_align_cfg.get("ce_reference_source", "weight_vectors")
                ).lower()
                vectors_np = load_classifier_head_vectors(
                    asce_model_dir, source=ce_source,
                )
                arcface_ce_reference_vectors = torch.as_tensor(
                    vectors_np, dtype=torch.float32,
                )
                label_map_raw = load_label_map(asce_model_dir)
                if not label_map_raw:
                    raise RuntimeError(
                        "asce_private_alignment.loss_type='classifier_ce' "
                        f"requires label_map.json in {asce_model_dir}"
                    )
                arcface_ce_label_map = {str(v): int(k) for k, v in label_map_raw.items()}
                arcface_ce_scale_s = float(asce_alignment_scorer.scale_s)
                arcface_ce_margin_m = float(asce_align_cfg.get("margin_m", 0.0))
                print(
                    f"  [base_asce_ce] reference_source={ce_source} "
                    f"K={arcface_ce_reference_vectors.size(0)} "
                    f"scale_s={arcface_ce_scale_s} margin_m={arcface_ce_margin_m}"
                )

            resolved["asce_private_alignment"] = {
                "enabled": True,
                "model_dir": asce_model_dir,
                "embedding_dim": asce_embedding_dim,
                "weight": asce_align_cfg["weight"],
                "loss_type": asce_align_cfg["loss_type"],
                "warmup_fraction": asce_align_cfg["warmup_fraction"],
                "pooling": asce_align_cfg["pooling"],
                "schedule": copy.deepcopy(asce_align_cfg.get("schedule", {})),
                "target_source": asce_align_cfg["target_source"],
                "projector": asce_align_cfg["projector"],
                "target_cache_dir": asce_target_cache_dir,
                "targets_cached": len(asce_target_cache),
            }
            del asce_alignment_scorer
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    else:
        resolved["asce_private_alignment"] = {"enabled": False}

    with open(os.path.join(method_dir, "config_resolved.yaml"), "w") as f:
        yaml.dump(resolved, f, default_flow_style=False)

    # ---- Load base model ----
    print("\n  Loading base model ...")
    base_model, tokenizer = load_base_model_and_tokenizer(model_cfg)

    # ---- Build PEFT config and model ----
    lora_config = build_lora_config(peft_cfg)
    model = get_peft_model(base_model, lora_config)
    global_adapter_dict = get_lora_state(model)

    trainable_count, total_count = count_trainable_params(model)
    print(f"  Trainable params: {trainable_count:,} / {total_count:,}")

    asce_lm_hidden_size = int(model.config.hidden_size)
    if asce_align_cfg is not None and asce_align_cfg.get("enabled", False):
        resolved["asce_private_alignment"]["lm_hidden_size"] = asce_lm_hidden_size
        with open(os.path.join(method_dir, "config_resolved.yaml"), "w") as f:
            yaml.dump(resolved, f, default_flow_style=False)

    # Save initial global adapter
    init_adapter_dir = os.path.join(method_dir, "round=0", "server", "global_content_adapter")
    os.makedirs(init_adapter_dir, exist_ok=True)
    model.save_pretrained(init_adapter_dir)
    tokenizer.save_pretrained(init_adapter_dir)

    round_start = max(1, args.resume_round)
    history = []
    total_gradient_steps = 0
    wall_time_total = 0.0

    # ===========================================================
    # FEDERATED TRAINING LOOP (FedAvg + ASCE)
    # ===========================================================
    seen_clients = set()
    for t in range(round_start, total_rounds + 1):
        round_start_time = time.time()
        print(f"\n{'='*60}")
        print(f" ROUND {t}/{total_rounds} (Base ASCE, prox_mu={prox_mu})")
        print(f"{'='*60}")

        round_dir = os.path.join(method_dir, f"round={t}")
        server_dir = os.path.join(round_dir, "server")
        clients_dir = os.path.join(round_dir, "clients")

        # ---- Step 1: Select clients ----
        rng = np.random.RandomState(seed + t)
        selected_clients = resolve_round_client_sample(
            all_client_ids,
            clients_per_round,
            rng,
            sample_mode=sample_mode,
            seen_clients=seen_clients,
        )
        print(
            f"\n  Selected {len(selected_clients)} clients "
            f"(sample_mode={sample_mode}, seen={len(seen_clients)})"
        )

        # ---- Step 2: Client-side local training (global adapter + ASCE) ----
        client_deltas = {}
        client_losses = {}
        client_delta_norms = {}
        client_diagnostics = {}

        for cid in selected_clients:
            print(f"\n  --- Client {cid} ---")
            client_dir = os.path.join(clients_dir, f"client={cid}")
            os.makedirs(client_dir, exist_ok=True)

            client_ds = dataset_from_texts(
                client_texts.get(cid, []),
                tokenizer,
                fed_cfg["max_seq_len"],
            )
            if client_ds is None or len(client_ds) == 0:
                print(f"    WARNING: no training data for {cid}, skipping")
                continue

            client_asce_bundle = None
            if (
                asce_align_cfg is not None
                and asce_align_cfg.get("enabled", False)
                and cid in asce_target_cache
            ):
                client_asce_bundle = {
                    "enabled": True,
                    "target_vector": asce_target_cache[cid],
                    "hidden_size": asce_lm_hidden_size,
                    "embedding_dim": asce_embedding_dim,
                    "weight": asce_align_cfg["weight"],
                    "loss_type": asce_align_cfg["loss_type"],
                    "warmup_fraction": asce_align_cfg["warmup_fraction"],
                    "min_supervised_tokens": asce_align_cfg["min_supervised_tokens"],
                    "pooling": asce_align_cfg["pooling"],
                    "schedule": copy.deepcopy(asce_align_cfg.get("schedule", {})),
                    "target_source": asce_align_cfg.get("target_source", "reference_mean"),
                    "projector_cfg": asce_align_cfg.get("projector", {}),
                    "logging": asce_align_cfg.get("logging", {}),
                }
                if str(asce_align_cfg.get("loss_type", "cosine")).lower() == "classifier_ce":
                    canonical_cid = str(canonical_author_id(cid))
                    class_idx = arcface_ce_label_map.get(canonical_cid)
                    if class_idx is None:
                        sample = list(arcface_ce_label_map.keys())[:5]
                        raise KeyError(
                            f"classifier_ce: client_id={cid!r} "
                            f"(canonical={canonical_cid!r}) not in ArcFace "
                            f"label_map; first few keys: {sample}"
                        )
                    client_asce_bundle["reference_vectors"] = arcface_ce_reference_vectors
                    client_asce_bundle["class_index"] = int(class_idx)
                    client_asce_bundle["scale_s"] = arcface_ce_scale_s
                    client_asce_bundle["margin_m"] = arcface_ce_margin_m

            print(
                f"    Training on {len(client_ds)} samples "
                f"(ASCE={'on' if client_asce_bundle else 'off'}, prox_mu={prox_mu}) ..."
            )

            delta, loss, d_norm, diagnostics = train_client_local_base_asce(
                model=model,
                tokenizer=tokenizer,
                global_adapter_dict=global_adapter_dict,
                client_ds=client_ds,
                fed_cfg=fed_cfg,
                output_dir=client_dir,
                seed=stable_client_round_seed(seed, t, cid),
                prox_mu=prox_mu,
                asce_bundle=client_asce_bundle,
            )

            client_deltas[cid] = delta
            client_losses[cid] = loss
            client_delta_norms[cid] = d_norm
            client_diagnostics[cid] = diagnostics
            seen_clients.add(cid)
            print(f"    Loss: {loss:.4f}")

            # Estimate gradient steps for this client
            total_gradient_steps += estimate_gradient_steps(len(client_ds), fed_cfg)

            upload_stats = {
                "client_id": cid,
                "round": t,
                "method": method_name,
                "implementation": "base_asce_global_adapter_only",
                "train_samples": len(client_ds),
                "train_loss": loss,
                "delta_norm": d_norm,
                "prox_mu": prox_mu,
                "mean_prox_loss": diagnostics.get("mean_prox_loss", 0.0),
                "delta_params": len(delta),
                "active_adapter": "default",
                "trainable_params": trainable_count,
                "asce_alignment_enabled": bool(client_asce_bundle),
                "asce_style_status": diagnostics.get("asce_style_status", "disabled"),
                "asce_style_loss": diagnostics.get("asce_style_loss", 0.0),
                "asce_style_mean_cosine": diagnostics.get("asce_style_mean_cosine", 0.0),
                "asce_style_effective_steps": diagnostics.get("asce_style_effective_steps", 0),
                "asce_style_weight": diagnostics.get("asce_style_weight", 0.0),
                "asce_style_loss_type": diagnostics.get("asce_style_loss_type", "disabled"),
                "asce_weighted_loss_mean": diagnostics.get("asce_weighted_loss_mean", 0.0),
                "asce_contribution_ratio": diagnostics.get("asce_contribution_ratio", 0.0),
                "arcface_ce_top1_acc": diagnostics.get("arcface_ce_top1_acc", 0.0),
                "mean_task_loss": diagnostics.get("mean_task_loss", 0.0),
                "mean_total_loss": diagnostics.get("mean_total_loss", 0.0),
            }
            with open(os.path.join(client_dir, "upload_stats.json"), "w") as f:
                json.dump(upload_stats, f, indent=2)

            try:
                torch._dynamo.reset()
            except Exception:
                pass

        if not client_deltas:
            print("  WARNING: no client deltas produced, skipping round")
            continue

        # ---- Step 3: Uniform aggregation ----
        print(f"\n  Aggregating {len(client_deltas)} client deltas (FedAvg-style) ...")
        active_client_ids = [cid for cid in selected_clients if cid in client_deltas]
        agg_delta = aggregate_deltas_uniform(
            client_deltas, active_client_ids, client_data_sizes
        )

        # ---- Step 4: Update global adapter ----
        for name in global_adapter_dict:
            if name in agg_delta:
                global_adapter_dict[name] += agg_delta[name]

        # Save updated global adapter
        adapter_save_dir = os.path.join(server_dir, "global_content_adapter")
        os.makedirs(adapter_save_dir, exist_ok=True)
        load_lora_state(model, global_adapter_dict)
        model.save_pretrained(adapter_save_dir)
        tokenizer.save_pretrained(adapter_save_dir)
        torch.cuda.empty_cache()

        # ---- Round summary ----
        round_time = time.time() - round_start_time
        wall_time_total += round_time
        avg_loss = np.mean(list(client_losses.values()))
        avg_delta_norm = np.mean(list(client_delta_norms.values()))
        avg_prox_loss = np.mean([
            d.get("mean_prox_loss", 0.0) for d in client_diagnostics.values()
        ])
        asce_losses = [
            float(d.get("asce_style_loss", 0.0))
            for d in client_diagnostics.values()
            if d.get("asce_style_status") == "active"
        ]
        asce_cosines = [
            float(d.get("asce_style_mean_cosine", 0.0))
            for d in client_diagnostics.values()
            if d.get("asce_style_status") == "active"
        ]
        asce_weighted_losses = [
            float(d.get("asce_weighted_loss_mean", 0.0))
            for d in client_diagnostics.values()
            if d.get("asce_style_status") == "active"
        ]
        asce_contrib_ratios = [
            float(d.get("asce_contribution_ratio", 0.0))
            for d in client_diagnostics.values()
            if d.get("asce_style_status") == "active"
        ]
        asce_effective_steps = [
            int(d.get("asce_style_effective_steps", 0))
            for d in client_diagnostics.values()
        ]
        arcface_ce_top1s = [
            float(d.get("arcface_ce_top1_acc", 0.0))
            for d in client_diagnostics.values()
            if d.get("asce_style_status") == "active"
        ]
        asce_round_enabled = bool(
            asce_align_cfg is not None and asce_align_cfg.get("enabled", False)
        )
        if asce_round_enabled:
            if not asce_losses:
                asce_round_status = "enabled_but_no_active_clients"
            elif sum(asce_effective_steps) == 0:
                asce_round_status = "enabled_but_zero_steps"
            else:
                asce_round_status = "active"
        else:
            asce_round_status = "disabled"

        round_metrics = {
            "round": t,
            "method": method_name,
            "implementation": "base_asce_global_adapter_only",
            "prox_mu": prox_mu,
            "selected_clients": len(selected_clients),
            "participating_clients": len(client_deltas),
            "avg_train_loss": float(avg_loss),
            "avg_delta_norm": float(avg_delta_norm),
            "avg_prox_loss": float(avg_prox_loss),
            "active_adapter": "default",
            "trainable_params": trainable_count,
            "round_time_seconds": round_time,
            "sample_mode": sample_mode,
            "asce_alignment_enabled": asce_round_enabled,
            "asce_alignment_status": asce_round_status,
            "asce_alignment_weight": float(
                (asce_align_cfg or {}).get("weight", 0.0)
                if asce_round_enabled else 0.0
            ),
            "avg_asce_alignment_loss": float(
                np.mean(asce_losses) if asce_losses else 0.0
            ),
            "avg_asce_style_cosine": float(
                np.mean(asce_cosines) if asce_cosines else 0.0
            ),
            "avg_asce_weighted_loss": float(
                np.mean(asce_weighted_losses) if asce_weighted_losses else 0.0
            ),
            "avg_asce_contribution_ratio": float(
                np.mean(asce_contrib_ratios) if asce_contrib_ratios else 0.0
            ),
            "avg_asce_effective_steps": float(
                np.mean(asce_effective_steps) if asce_effective_steps else 0.0
            ),
            "num_clients_asce_active": int(
                sum(1 for d in client_diagnostics.values() if d.get("asce_style_status") == "active")
            ),
            "avg_arcface_ce_top1_acc": float(
                np.mean(arcface_ce_top1s) if arcface_ce_top1s else 0.0
            ),
            "arcface_loss_type": str(
                (asce_align_cfg or {}).get("loss_type", "cosine")
                if asce_round_enabled else "disabled"
            ),
            "asce_pooling": str(
                (asce_align_cfg or {}).get("pooling", "mean_supervised_tokens")
                if asce_round_enabled else "disabled"
            ),
            "asce_schedule_type": str(
                ((asce_align_cfg or {}).get("schedule", {}) or {}).get(
                    "type", "linear_warmup"
                )
                if asce_round_enabled else "disabled"
            ),
        }
        os.makedirs(server_dir, exist_ok=True)
        with open(os.path.join(server_dir, "round_metrics.json"), "w") as f:
            json.dump(round_metrics, f, indent=2)

        history.append(round_metrics)

        print(f"\n  Round {t} complete:")
        print(f"    Avg loss: {avg_loss:.4f}")
        print(f"    Avg prox_loss: {avg_prox_loss:.6f}")
        print(
            f"    ArcFace: status={asce_round_status}, "
            f"active_clients={round_metrics['num_clients_asce_active']}/{len(client_diagnostics)}, "
            f"avg_loss={round_metrics['avg_asce_alignment_loss']:.6f}, "
            f"avg_cosine={round_metrics['avg_asce_style_cosine']:.4f}"
        )
        print(f"    Avg delta_norm: {avg_delta_norm:.6f}")
        print(f"    Time: {round_time:.1f}s")

    # ===========================================================
    # Save final artifacts
    # ===========================================================
    final_dir = os.path.join(method_dir, "final")
    os.makedirs(final_dir, exist_ok=True)

    with open(os.path.join(final_dir, "training_history.json"), "w") as f:
        json.dump(history, f, indent=2)

    final_adapter_dir = os.path.join(final_dir, "global_content_adapter")
    os.makedirs(final_adapter_dir, exist_ok=True)
    load_lora_state(model, global_adapter_dict)
    model.save_pretrained(final_adapter_dir)
    tokenizer.save_pretrained(final_adapter_dir)

    # Convenience alias for shared-adapter generation.
    adapter_alias_dir = os.path.join(method_dir, "adapter")
    os.makedirs(adapter_alias_dir, exist_ok=True)
    model.save_pretrained(adapter_alias_dir)
    tokenizer.save_pretrained(adapter_alias_dir)

    training_summary = {
        "method": method_name,
        "implementation": "base_asce_global_adapter_only",
        "total_rounds": total_rounds,
        "clients_per_round": clients_per_round,
        "sample_mode": sample_mode,
        "prox_mu": prox_mu,
        "asce_alignment_enabled": bool(
            asce_align_cfg is not None and asce_align_cfg.get("enabled", False)
        ),
        "asce_alignment_weight": float(
            (asce_align_cfg or {}).get("weight", 0.0)
            if asce_align_cfg is not None else 0.0
        ),
        "arcface_loss_type": str(
            (asce_align_cfg or {}).get("loss_type", "disabled")
            if asce_align_cfg is not None else "disabled"
        ),
        "targets_cached": len(asce_target_cache),
        "total_gradient_steps": total_gradient_steps,
        "wall_time_seconds": round(wall_time_total, 1),
        "final_adapter_dir": final_adapter_dir,
        "adapter_alias_dir": adapter_alias_dir,
    }
    with open(os.path.join(final_dir, "training_summary.json"), "w") as f:
        json.dump(training_summary, f, indent=2)

    # ---- Emit training_budget.csv (Task 3 artifact contract) ----
    import csv
    budget_path = os.path.join(method_dir, "training_budget.csv")
    with open(budget_path, "w", newline="") as csvf:
        writer = csv.writer(csvf)
        writer.writerow([
            "method", "total_rounds", "clients_per_round", "local_epochs",
            "total_gradient_steps", "wall_time_seconds", "prox_mu",
            "asce_enabled", "asce_weight",
        ])
        writer.writerow([
            method_name, total_rounds, clients_per_round,
            fed_cfg["local_epochs"], total_gradient_steps,
            round(wall_time_total, 1), prox_mu,
            training_summary["asce_alignment_enabled"],
            training_summary["asce_alignment_weight"],
        ])
    print(f"  ✓ training_budget.csv saved to {budget_path}")

    print(f"\n{'='*60}")
    print(f"✓ Base ASCE training complete ({total_rounds} rounds, prox_mu={prox_mu})")
    print(f"  Method: {method_name}")
    print(f"  Final adapter: {final_adapter_dir}")
    print(f"  Adapter alias: {adapter_alias_dir}")
    print(f"  Artifacts: {method_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
