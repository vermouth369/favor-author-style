#!/usr/bin/env python3
"""49_train_ditto_peft.py — Ditto personalized federated baseline.

Implements Ditto (Li et al., 2021) with PEFT/QLoRA:
  - Global adapter A_g: trained via FedAvg
  - Personalized adapter A_i: trained locally with proximal regularization
    loss = task_loss + λ * ||A_i - A_g||²
  - Only A_g deltas are uploaded; A_i stays local

This produces a personalized FL baseline to compare with FAVoR.

Usage:
  python 49_train_ditto_peft.py --config exp2/config/phase1/phase1_medium_seed2026_runtime.yaml \\
      --method-name Ditto --lambda-ditto 0.01
"""

import argparse
import copy
import hashlib
import importlib.util
import json
import os
import shutil
import sys
import time
from collections import OrderedDict
from pathlib import Path

# Raise recursion limit early — bitsandbytes dequantize_4bit can trigger
# deep recursion through PyTorch dispatch hooks after many client iterations.
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
    get_lora_state,
    load_lora_state,
    diff_lora_state,
    lora_delta_norm,
    log_adapter_state,
    count_trainable_params,
    estimate_gradient_steps,
    load_client_manifest_with_noniid,
    canonical_author_id,
    record_matches_client_id,
)


def load_script_module(module_name, filename):
    """Load a sibling training script whose filename is not importable normally."""
    script_dir = Path(__file__).resolve().parent
    module_path = script_dir / filename
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_base_asce_module():
    return load_script_module("train_base_asce_impl", "44_train_base_asce.py")


def build_client_asce_bundle(
    *,
    cid,
    asce_align_cfg,
    asce_target_cache,
    asce_lm_hidden_size,
    asce_embedding_dim,
    arcface_ce_reference_vectors,
    arcface_ce_label_map,
    arcface_ce_scale_s,
    arcface_ce_margin_m,
    base_asce,
):
    if (
        asce_align_cfg is None
        or not asce_align_cfg.get("enabled", False)
        or cid not in asce_target_cache
    ):
        return None

    bundle = {
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
        canonical_cid = str(base_asce.canonical_author_id(cid))
        class_idx = arcface_ce_label_map.get(canonical_cid)
        if class_idx is None:
            sample = list(arcface_ce_label_map.keys())[:5]
            raise KeyError(
                f"classifier_ce: client_id={cid!r} (canonical={canonical_cid!r}) "
                f"not in ArcFace label_map; first few keys: {sample}"
            )
        bundle["reference_vectors"] = arcface_ce_reference_vectors
        bundle["class_index"] = int(class_idx)
        bundle["scale_s"] = arcface_ce_scale_s
        bundle["margin_m"] = arcface_ce_margin_m
    return bundle


def disabled_personal_diagnostics(loss):
    return {
        "mean_task_loss": float(loss),
        "mean_total_loss": float(loss),
        "mean_prox_loss": 0.0,
        "effective_steps": 0,
        "asce_style_status": "disabled",
        "asce_style_loss": 0.0,
        "asce_style_mean_cosine": 0.0,
        "arcface_ce_top1_acc": 0.0,
        "asce_style_effective_steps": 0,
        "asce_style_weight": 0.0,
        "asce_style_loss_type": "disabled",
        "asce_style_pooling": "disabled",
        "asce_style_schedule_type": "disabled",
        "asce_weighted_loss_mean": 0.0,
        "asce_contribution_ratio": 0.0,
        "asce_active_steps": 0,
    }


# ============================================================
# Proximal-regularized Trainer for Ditto personalized branch
# ============================================================

class DittoProximalTrainer(Trainer):
    """Trainer that adds proximal regularization toward a reference adapter.

    loss_total = loss_task + lambda_ditto * sum_p ||A_i_p - A_g_p||^2

    The reference (global) state is frozen and passed at init.
    """

    def __init__(self, *args, global_ref_state=None, lambda_ditto=0.01, **kwargs):
        super().__init__(*args, **kwargs)
        self.lambda_ditto = lambda_ditto
        # Store reference state tensors on the same device as model
        self.global_ref_state = global_ref_state or {}

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        """Override compute_loss to add proximal regularization."""
        outputs = model(**inputs)
        task_loss = outputs.loss

        # Proximal term: λ * Σ ||A_i_p - A_g_p||²
        prox_loss = torch.tensor(0.0, device=task_loss.device, dtype=task_loss.dtype)
        for name, param in model.named_parameters():
            if "lora_" in name and name in self.global_ref_state:
                ref = self.global_ref_state[name].to(param.device)
                prox_loss = prox_loss + (param - ref).pow(2).sum()

        total_loss = task_loss + self.lambda_ditto * prox_loss

        return (total_loss, outputs) if return_outputs else total_loss


# ============================================================
# Model loading
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


def load_client_data(pooled_dir, client_id, tokenizer, max_seq_len):
    """Load training data for a single client from the pooled JSONL."""
    train_path = os.path.join(pooled_dir, "train.jsonl")
    texts = []
    with open(train_path, "r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            if record_matches_client_id(rec, client_id):
                texts.append(rec["text"])
    if not texts:
        return None, 0
    ds = Dataset.from_dict({"text": texts})
    def tokenize_fn(examples):
        return tokenizer(
            examples["text"], truncation=True,
            max_length=max_seq_len, padding=False,
        )
    ds = ds.map(tokenize_fn, batched=True, remove_columns=["text"])
    return ds, len(texts)


def resolve_ditto_client_artifacts_cfg(cfg):
    """Resolve Ditto client artifact policy.

    Generation only needs `personalized_adapter/`; raw states and upload
    deltas are large debugging/resume artifacts, so the default is lean.
    """
    raw = copy.deepcopy(
        cfg.get("ditto", {}).get("client_artifacts", {}) or {}
    )
    return {
        "save_global_upload_delta": bool(
            raw.get("save_global_upload_delta", False)
        ),
        "save_personalized_raw_state": bool(
            raw.get("save_personalized_raw_state", False)
        ),
        "save_tokenizer_with_personalized_adapter": bool(
            raw.get("save_tokenizer_with_personalized_adapter", False)
        ),
        "cleanup_trainer_tmp": bool(raw.get("cleanup_trainer_tmp", True)),
    }


# ============================================================
# Client-side training: two branches (Ditto)
# ============================================================

def train_client_global_branch(
    model, tokenizer, client_ds, global_state, fed_cfg, output_dir, seed,
    client_artifact_cfg=None,
):
    """Branch A: Train the global adapter for aggregation (standard FedAvg step).

    Returns:
        global_delta: OrderedDict of parameter deltas (uploaded)
        updated_global_state: LoRA state after training
        loss: training loss
    """
    os.makedirs(output_dir, exist_ok=True)
    client_artifact_cfg = client_artifact_cfg or {}

    model.set_adapter("default")
    load_lora_state(model, global_state)
    log_adapter_state(model, "DITTO_GLOBAL_START")

    before_state = get_lora_state(model)

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=False
    )
    training_args = TrainingArguments(
        output_dir=os.path.join(output_dir, "_global_tmp"),
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
    trainer = Trainer(
        model=model, args=training_args,
        train_dataset=client_ds, data_collator=data_collator,
    )
    result = trainer.train()
    loss = result.training_loss

    after_state = get_lora_state(model)
    global_delta = diff_lora_state(before_state, after_state)
    delta_norm = lora_delta_norm(before_state, after_state)

    log_adapter_state(model, "DITTO_GLOBAL_END")
    print(f"    global_delta_norm: {delta_norm:.6f}")

    delta_path = os.path.join(output_dir, "global_upload_delta.pt")
    if client_artifact_cfg.get("save_global_upload_delta", False):
        torch.save(global_delta, delta_path)
    elif os.path.exists(delta_path):
        os.remove(delta_path)

    for p in model.parameters():
        p.grad = None
    del trainer
    torch.cuda.empty_cache()
    if client_artifact_cfg.get("cleanup_trainer_tmp", True):
        shutil.rmtree(os.path.join(output_dir, "_global_tmp"), ignore_errors=True)

    return global_delta, after_state, loss


def train_client_personalized_branch(
    model, tokenizer, client_ds, personalized_state, global_state,
    fed_cfg, lambda_ditto, output_dir, seed, client_artifact_cfg=None,
    asce_bundle=None, base_asce=None,
):
    """Branch B: Train personalized adapter with proximal regularization.

    loss = task_loss + λ * ||A_i - A_g||²

    The personalized adapter is initialized from its previous state
    (or from A_g on the first round), and regularized toward A_g.

    Returns:
        personalized_ckpt_path: path where personalized adapter is saved
        loss: training loss
    """
    os.makedirs(output_dir, exist_ok=True)
    client_artifact_cfg = client_artifact_cfg or {}

    model.set_adapter("default")
    load_lora_state(model, personalized_state)
    log_adapter_state(model, "DITTO_PERSONAL_START")

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=False
    )
    training_args = TrainingArguments(
        output_dir=os.path.join(output_dir, "_personal_tmp"),
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
        seed=seed + 9999,  # different seed from global branch
        report_to="none",
        remove_unused_columns=False,
        max_grad_norm=1.0,
    )

    asce_bundle = asce_bundle or {}
    asce_enabled = bool(asce_bundle.get("enabled", False))
    asce_projector = None
    asce_target_tensor = None
    arcface_reference_vectors_tensor = None

    if asce_enabled:
        if base_asce is None:
            raise ValueError("ASCE alignment requested but base_asce module is unavailable")
        from asce_private_alignment import build_projector

        asce_projector = build_projector(
            hidden_size=int(asce_bundle["hidden_size"]),
            embedding_dim=int(asce_bundle["embedding_dim"]),
            projector_cfg=asce_bundle.get("projector_cfg", {}),
            device=next(model.parameters()).device,
        )
        model.asce_ditto_projector = asce_projector
        asce_target_tensor = torch.as_tensor(
            asce_bundle["target_vector"], dtype=torch.float32
        )
        if asce_bundle.get("reference_vectors") is not None:
            arcface_reference_vectors_tensor = torch.as_tensor(
                asce_bundle["reference_vectors"], dtype=torch.float32
            )

        # BaseASCETrainer uses (prox_mu / 2) * ||A_i - A_g||^2, while
        # DittoProximalTrainer uses lambda_ditto * ||A_i - A_g||^2.
        # Set prox_mu=2*lambda_ditto to keep the Ditto penalty unchanged.
        trainer = base_asce.BaseASCETrainer(
            model=model,
            args=training_args,
            train_dataset=client_ds,
            data_collator=data_collator,
            server_state=global_state,
            prox_mu=2.0 * float(lambda_ditto),
            asce_target_tensor=asce_target_tensor,
            asce_weight=float(asce_bundle.get("weight", 0.0)),
            asce_projector=asce_projector,
            arcface_loss_type=asce_bundle.get("loss_type", "cosine"),
            asce_min_supervised_tokens=int(
                asce_bundle.get("min_supervised_tokens", 8)
            ),
            asce_warmup_fraction=float(
                asce_bundle.get("warmup_fraction", 0.0)
            ),
            asce_total_steps=int(estimate_gradient_steps(len(client_ds), fed_cfg)),
            asce_pooling=asce_bundle.get("pooling", "mean_supervised_tokens"),
            asce_schedule_cfg=asce_bundle.get("schedule", {}),
            arcface_reference_vectors=arcface_reference_vectors_tensor,
            arcface_class_index=asce_bundle.get("class_index"),
            arcface_scale_s=float(asce_bundle.get("scale_s", 30.0)),
            arcface_margin_m=float(asce_bundle.get("margin_m", 0.0)),
        )
    else:
        # Use proximal trainer with reference to global adapter.
        trainer = DittoProximalTrainer(
            model=model,
            args=training_args,
            train_dataset=client_ds,
            data_collator=data_collator,
            global_ref_state=global_state,
            lambda_ditto=lambda_ditto,
        )
    result = trainer.train()
    loss = result.training_loss
    diagnostics = (
        trainer.get_training_diagnostics()
        if hasattr(trainer, "get_training_diagnostics")
        else disabled_personal_diagnostics(loss)
    )

    # Save personalized adapter checkpoint
    personalized_ckpt_path = os.path.join(output_dir, "personalized_adapter")
    os.makedirs(personalized_ckpt_path, exist_ok=True)

    if asce_projector is not None and hasattr(model, "asce_ditto_projector"):
        delattr(model, "asce_ditto_projector")

    # CRITICAL: Disable gradient_checkpointing before saving.
    model.gradient_checkpointing_disable()
    model.eval()

    model.save_pretrained(personalized_ckpt_path)
    if client_artifact_cfg.get("save_tokenizer_with_personalized_adapter", False):
        tokenizer.save_pretrained(personalized_ckpt_path)
    else:
        for name in (
            "tokenizer.json",
            "tokenizer_config.json",
            "special_tokens_map.json",
            "vocab.json",
            "merges.txt",
            "added_tokens.json",
        ):
            path = os.path.join(personalized_ckpt_path, name)
            if os.path.exists(path):
                os.remove(path)

    # Validate that weight file was actually saved
    safetensors_path = os.path.join(personalized_ckpt_path, "adapter_model.safetensors")
    bin_path = os.path.join(personalized_ckpt_path, "adapter_model.bin")
    if not os.path.exists(safetensors_path) and not os.path.exists(bin_path):
        print(f"    WARNING: save_pretrained did not produce weight file, using fallback torch.save")
        fallback_state = get_lora_state(model)
        torch.save(fallback_state, os.path.join(personalized_ckpt_path, "adapter_state.pt"))

    raw_state = get_lora_state(model)
    raw_path = os.path.join(output_dir, "personalized_raw.pt")
    if client_artifact_cfg.get("save_personalized_raw_state", False):
        torch.save(raw_state, raw_path)
    elif os.path.exists(raw_path):
        os.remove(raw_path)

    log_adapter_state(model, "DITTO_PERSONAL_END")

    for p in model.parameters():
        p.grad = None
    del trainer
    torch.cuda.empty_cache()
    if client_artifact_cfg.get("cleanup_trainer_tmp", True):
        shutil.rmtree(os.path.join(output_dir, "_personal_tmp"), ignore_errors=True)

    return personalized_ckpt_path, raw_state, loss, diagnostics


# ============================================================
# FedAvg aggregation
# ============================================================

def aggregate_deltas_uniform(client_deltas, client_ids, data_sizes):
    """Uniform FedAvg aggregation (weighted by data size)."""
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


def stable_seed_offset(identifier, modulo=1000):
    """Return a process-stable hash bucket for per-client seeding."""
    canonical_id = str(canonical_author_id(identifier))
    digest = hashlib.sha1(canonical_id.encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big") % int(modulo)


def stable_client_round_seed(base_seed, round_id, client_id):
    """Derive a deterministic seed for one client in one federated round."""
    return int(base_seed) + int(round_id) * 1000 + stable_seed_offset(client_id)


# ============================================================
# Main training loop
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Ditto Personalized Federated Training"
    )
    parser.add_argument("--config", type=str, default="exp2/config/phase1/phase1_medium_seed2026_runtime.yaml")
    parser.add_argument("--gpu", type=int, default=None)
    parser.add_argument(
        "--method-name", type=str, default="Ditto",
        help="Method directory name"
    )
    parser.add_argument(
        "--lambda-ditto", type=float, default=0.01,
        help="Proximal regularization strength (default: 0.01)"
    )
    parser.add_argument(
        "--resume-round", type=int, default=0,
        help="Resume from a specific round (0 = start fresh)"
    )
    parser.add_argument(
        "--enable-asce-alignment", action="store_true",
        help="Add ASCE style alignment to the Ditto personalized branch only"
    )
    args = parser.parse_args()

    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)
    with open(cfg["paths"]["base_model_config"], "r") as f:
        model_cfg = yaml.safe_load(f)

    # ---- Config ----
    prototype_dir = cfg["paths"]["prototype_dir"]
    pooled_dir = cfg["paths"]["pooled_dir"]
    runs_dir = cfg["paths"]["runs_dir"]
    fed_cfg = cfg["fed"]
    peft_cfg = cfg["peft"]
    seed = cfg["seeds"]["training"]

    total_rounds = fed_cfg["rounds"]
    clients_per_round = fed_cfg["clients_per_round"]
    sample_mode = str(fed_cfg.get("sample_mode", "uniform")).lower()
    lambda_ditto = args.lambda_ditto
    method_name = args.method_name
    client_artifact_cfg = resolve_ditto_client_artifacts_cfg(cfg)
    asce_alignment_enabled = bool(args.enable_asce_alignment)
    implementation_name = "ditto_peft_asce" if asce_alignment_enabled else "ditto_peft"
    personalized_branch_name = (
        "proximal_regularization_plus_asce"
        if asce_alignment_enabled
        else "proximal_regularization"
    )

    method_dir = os.path.join(runs_dir, method_name)
    os.makedirs(method_dir, exist_ok=True)

    print("=" * 60)
    print(f"Ditto Personalized FL Training — {method_name}")
    print(f"  λ_ditto: {lambda_ditto}")
    print(f"  Rounds: {total_rounds}")
    print(f"  Clients/round: {clients_per_round}")
    print(f"  Sample mode: {sample_mode}")
    print(f"  ASCE alignment: {'on' if asce_alignment_enabled else 'off'}")
    print(f"  Client artifacts: {client_artifact_cfg}")
    print(f"  Two branches: global (FedAvg) + personalized (proximal)")
    print("=" * 60)

    # ---- Save resolved config ----
    resolved = {
        "config": cfg,
        "model_config": model_cfg,
        "method_name": method_name,
        "lambda_ditto": lambda_ditto,
        "implementation": implementation_name,
        "global_branch": "fedavg",
        "personalized_branch": personalized_branch_name,
        "sample_mode": sample_mode,
        "client_artifacts": client_artifact_cfg,
        "seed": seed,
    }

    # ---- Load client manifest ----
    manifest_info = load_client_manifest_with_noniid(prototype_dir, cfg)
    all_client_ids = manifest_info["all_client_ids"]
    client_data_sizes = manifest_info["client_data_sizes"]
    noniid_level = manifest_info["non_iid_level"]
    resolved["noniid_manifest_path"] = manifest_info["non_iid_manifest_path"]
    resolved["noniid_level"] = noniid_level
    resolved["noniid_downgrade"] = manifest_info["non_iid_downgrade"]

    base_asce = None
    asce_align_cfg = None
    asce_target_cache = {}
    arcface_ce_reference_vectors = None
    arcface_ce_label_map = {}
    arcface_ce_scale_s = 30.0
    arcface_ce_margin_m = 0.0
    asce_embedding_dim = None

    if asce_alignment_enabled:
        asce_align_raw = cfg.get("asce_private_alignment", {})
        if not asce_align_raw.get("enabled", False):
            raise ValueError(
                "--enable-asce-alignment was requested, but "
                "cfg['asce_private_alignment'].enabled is false/missing"
            )
        base_asce = load_base_asce_module()
        from asce_private_alignment import resolve_asce_alignment_config
        from style_asce_runtime import load_label_map, load_style_scorer

        asce_align_cfg = resolve_asce_alignment_config(asce_align_raw)
        asce_model_dir = asce_align_cfg["model_dir"]
        print(f"\n  Loading ArcFace scorer for Ditto+ASCE: {asce_model_dir}")
        scorer = load_style_scorer(asce_model_dir, task="authorship")
        asce_embedding_dim = int(scorer.embedding_dim)
        print(
            f"  ArcFace scorer loaded: backend={scorer.backend}, "
            f"embedding_dim={asce_embedding_dim}"
        )

        print("\n  Loading client texts from pooled train split for ASCE targets ...")
        client_texts = base_asce.load_all_client_texts(pooled_dir, all_client_ids)
        print(
            "  Client text cache: "
            f"{sum(1 for v in client_texts.values() if v)}/{len(all_client_ids)} clients"
        )
        cache_dir = os.path.join(
            method_dir,
            "optional",
            asce_align_cfg.get("cache_dir", "asce_target_cache"),
        )
        os.makedirs(cache_dir, exist_ok=True)
        asce_target_cache, target_meta = base_asce.build_asce_target_cache(
            client_texts=client_texts,
            asce_align_cfg=asce_align_cfg,
            scorer=scorer,
            cache_dir=cache_dir,
        )
        base_asce.write_target_collision_report(
            asce_target_cache,
            os.path.join(method_dir, "optional"),
        )

        if str(asce_align_cfg.get("loss_type", "cosine")).lower() == "classifier_ce":
            from asce_private_alignment import load_classifier_head_vectors

            ce_source = str(
                asce_align_cfg.get("ce_reference_source", "weight_vectors")
            ).lower()
            vectors_np = load_classifier_head_vectors(asce_model_dir, source=ce_source)
            arcface_ce_reference_vectors = torch.as_tensor(vectors_np, dtype=torch.float32)
            label_map_raw = load_label_map(asce_model_dir)
            arcface_ce_label_map = {str(v): int(k) for k, v in label_map_raw.items()}
            arcface_ce_scale_s = float(scorer.scale_s)
            arcface_ce_margin_m = float(asce_align_cfg.get("margin_m", 0.0))

        resolved["asce_private_alignment"] = {
            "enabled": True,
            "scope": "personalized_branch_only",
            "model_dir": asce_model_dir,
            "embedding_dim": asce_embedding_dim,
            "weight": asce_align_cfg["weight"],
            "loss_type": asce_align_cfg["loss_type"],
            "warmup_fraction": asce_align_cfg["warmup_fraction"],
            "pooling": asce_align_cfg["pooling"],
            "schedule": copy.deepcopy(asce_align_cfg.get("schedule", {})),
            "target_source": asce_align_cfg["target_source"],
            "projector": asce_align_cfg["projector"],
            "target_cache_dir": cache_dir,
            "targets_cached": len(asce_target_cache),
        }
        del scorer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    else:
        resolved["asce_private_alignment"] = {
            "enabled": False,
            "scope": "none",
        }

    with open(os.path.join(method_dir, "config_resolved.yaml"), "w") as f:
        yaml.dump(resolved, f, default_flow_style=False)

    print(f"\n  Total clients: {len(all_client_ids)} (non-IID: {noniid_level})")

    # ---- Load base model ----
    print("\n  Loading base model ...")
    base_model, tokenizer = load_base_model_and_tokenizer(model_cfg)

    # ---- Build PEFT config and model ----
    lora_config = build_lora_config(peft_cfg)
    model = get_peft_model(base_model, lora_config)
    trainable, total = count_trainable_params(model)
    print(f"  Trainable params: {trainable:,} / {total:,}")

    asce_lm_hidden_size = int(model.config.hidden_size)
    if asce_alignment_enabled:
        resolved["asce_private_alignment"]["lm_hidden_size"] = asce_lm_hidden_size
        with open(os.path.join(method_dir, "config_resolved.yaml"), "w") as f:
            yaml.dump(resolved, f, default_flow_style=False)

    # ---- Initialize global adapter ----
    global_adapter_dict = get_lora_state(model)

    # ---- Per-client personalized state storage ----
    # Each client's personalized adapter state (initialized from global on first use)
    client_personalized_states = {}

    # ---- Handle resume ----
    if args.resume_round > 0:
        if not client_artifact_cfg.get("save_personalized_raw_state", False):
            print(
                "  WARNING: save_personalized_raw_state is false; "
                "resume can restore the global adapter but may not restore "
                "per-client personalized raw states from lean runs."
            )
        # Try to load global adapter from the last completed round
        for r in range(args.resume_round - 1, 0, -1):
            adapter_path = os.path.join(
                method_dir, "server", f"round={r}", "global_content_adapter"
            )
            safetensors_path = os.path.join(adapter_path, "adapter_model.safetensors")
            bin_path = os.path.join(adapter_path, "adapter_model.bin")
            if os.path.exists(safetensors_path) or os.path.exists(bin_path):
                from peft import PeftModel
                # Load adapter and extract state
                model.load_adapter(adapter_path, adapter_name="resumed_global")
                model.set_adapter("resumed_global")
                global_adapter_dict = get_lora_state(model)
                model.set_adapter("default")
                load_lora_state(model, global_adapter_dict)
                try:
                    model.delete_adapter("resumed_global")
                except Exception:
                    pass
                print(f"  ✓ Resumed global adapter from round={r}")
                break

        # Try to load personalized states
        clients_base = os.path.join(method_dir, "clients")
        if os.path.exists(clients_base):
            for cid in all_client_ids:
                client_base = os.path.join(clients_base, f"client={cid}")
                if not os.path.exists(client_base):
                    continue
                # Find latest round with a personalized state
                raw_path_latest = os.path.join(client_base, "latest", "personalized_raw.pt")
                if os.path.exists(raw_path_latest):
                    client_personalized_states[cid] = torch.load(
                        raw_path_latest, map_location="cpu", weights_only=True
                    )
                else:
                    # Fallback to round=X for compatibility with old runs
                    latest = -1
                    for d in os.listdir(client_base):
                        if d.startswith("round="):
                            try:
                                r = int(d.split("=")[1])
                                raw_path = os.path.join(client_base, d, "personalized_raw.pt")
                                if r > latest and os.path.exists(raw_path):
                                    latest = r
                            except ValueError:
                                pass
                    if latest > 0:
                        raw_path = os.path.join(
                            client_base, f"round={latest}", "personalized_raw.pt"
                        )
                        client_personalized_states[cid] = torch.load(
                            raw_path, map_location="cpu", weights_only=True
                        )
            print(f"  ✓ Resumed personalized states for {len(client_personalized_states)} clients")

        print(f"  Resuming from round {args.resume_round}")

    # Save initial global adapter
    init_dir = os.path.join(method_dir, "server", "round=0", "global_content_adapter")
    os.makedirs(init_dir, exist_ok=True)
    load_lora_state(model, global_adapter_dict)
    model.save_pretrained(init_dir)
    tokenizer.save_pretrained(init_dir)

    round_start = max(1, args.resume_round)
    client_personalized_states_seen = set(client_personalized_states.keys())
    history = []
    total_gradient_steps = 0
    wall_time_total = 0.0

    # ===========================================================
    # FEDERATED TRAINING LOOP
    # ===========================================================
    for t in range(round_start, total_rounds + 1):
        round_start_time = time.time()
        print(f"\n{'='*60}")
        print(f" ROUND {t}/{total_rounds}")
        print(f"{'='*60}")

        round_server_dir = os.path.join(method_dir, "server", f"round={t}")
        round_clients_dir = os.path.join(method_dir, "clients")
        os.makedirs(round_server_dir, exist_ok=True)

        # ---- Step 1: Select clients ----
        rng = np.random.RandomState(seed + t)
        selected_clients = resolve_round_client_sample(
            all_client_ids,
            clients_per_round,
            rng,
            sample_mode=sample_mode,
            seen_clients=client_personalized_states_seen,
        )
        print(
            f"\n  Selected {len(selected_clients)} clients "
            f"(sample_mode={sample_mode}, seen_personalized={len(client_personalized_states_seen)})"
        )

        # ---- Step 2-3: Client training (global + personalized branches) ----
        client_deltas = {}
        client_losses_global = {}
        client_losses_personal = {}
        client_personal_diagnostics = {}

        for cid in selected_clients:
            print(f"\n  --- Client {cid} ---")
            client_seed = stable_client_round_seed(seed, t, cid)
            # Space-optimized: use 'latest' instead of 'round={t}' to prevent 100GB+ usage
            client_dir = os.path.join(
                round_clients_dir, f"client={cid}", "latest"
            )
            os.makedirs(client_dir, exist_ok=True)

            # Load client data
            client_ds, n_texts = load_client_data(
                pooled_dir, cid, tokenizer, fed_cfg["max_seq_len"]
            )
            if client_ds is None or len(client_ds) == 0:
                print(f"    WARNING: no training data, skipping")
                continue
            print(f"    Training on {n_texts} samples ...")

            # --- Branch A: Global (FedAvg) ---
            global_delta, _, loss_global = train_client_global_branch(
                model=model,
                tokenizer=tokenizer,
                client_ds=client_ds,
                global_state=global_adapter_dict,
                fed_cfg=fed_cfg,
                output_dir=client_dir,
                seed=client_seed,
                client_artifact_cfg=client_artifact_cfg,
            )
            client_deltas[cid] = global_delta
            client_losses_global[cid] = loss_global
            print(f"    Global loss: {loss_global:.4f}")

            # --- Branch B: Personalized (proximal) ---
            # Initialize from previous personalized state, or from global if first time
            if cid in client_personalized_states:
                p_state = client_personalized_states[cid]
            else:
                p_state = OrderedDict(
                    (name, param.clone()) for name, param in global_adapter_dict.items()
                )

            asce_bundle = None
            if asce_alignment_enabled:
                asce_bundle = build_client_asce_bundle(
                    cid=cid,
                    asce_align_cfg=asce_align_cfg,
                    asce_target_cache=asce_target_cache,
                    asce_lm_hidden_size=asce_lm_hidden_size,
                    asce_embedding_dim=asce_embedding_dim,
                    arcface_ce_reference_vectors=arcface_ce_reference_vectors,
                    arcface_ce_label_map=arcface_ce_label_map,
                    arcface_ce_scale_s=arcface_ce_scale_s,
                    arcface_ce_margin_m=arcface_ce_margin_m,
                    base_asce=base_asce,
                )

            _, p_raw_state, loss_personal, personal_diag = train_client_personalized_branch(
                model=model,
                tokenizer=tokenizer,
                client_ds=client_ds,
                personalized_state=p_state,
                global_state=global_adapter_dict,
                fed_cfg=fed_cfg,
                lambda_ditto=lambda_ditto,
                output_dir=client_dir,
                seed=client_seed,
                client_artifact_cfg=client_artifact_cfg,
                asce_bundle=asce_bundle,
                base_asce=base_asce,
            )
            # Update stored personalized state
            client_personalized_states[cid] = p_raw_state
            client_personalized_states_seen.add(cid)
            client_losses_personal[cid] = loss_personal
            client_personal_diagnostics[cid] = personal_diag
            print(
                f"    Personal loss: {loss_personal:.4f} | "
                f"asce_status={personal_diag.get('asce_style_status', 'disabled')}"
            )

            # Save client metrics
            client_metrics = {
                "client_id": cid,
                "round": t,
                "train_samples": n_texts,
                "global_loss": loss_global,
                "personal_loss": loss_personal,
                "lambda_ditto": lambda_ditto,
                "global_delta_norm": lora_delta_norm(
                    global_adapter_dict,
                    OrderedDict(
                        (name, global_adapter_dict[name] + global_delta[name])
                        for name in global_delta
                    ),
                ),
                "artifact_policy": client_artifact_cfg,
                "asce_alignment_enabled": bool(asce_bundle),
                "asce_style_status": personal_diag.get(
                    "asce_style_status", "disabled"
                ),
                "asce_style_loss": personal_diag.get("asce_style_loss", 0.0),
                "asce_style_mean_cosine": personal_diag.get(
                    "asce_style_mean_cosine", 0.0
                ),
                "asce_contribution_ratio": personal_diag.get(
                    "asce_contribution_ratio", 0.0
                ),
                "mean_personal_task_loss": personal_diag.get("mean_task_loss", 0.0),
                "mean_personal_prox_loss": personal_diag.get("mean_prox_loss", 0.0),
                "mean_personal_total_loss": personal_diag.get("mean_total_loss", 0.0),
            }
            with open(os.path.join(client_dir, "client_metrics.json"), "w") as f:
                json.dump(client_metrics, f, indent=2)

            # Track gradient steps
            total_gradient_steps += estimate_gradient_steps(len(client_ds), fed_cfg)

            # Reset TorchDynamo state between clients
            try:
                torch._dynamo.reset()
            except Exception:
                pass

        if not client_deltas:
            print("  WARNING: no client deltas produced, skipping round")
            continue

        # ---- Step 4: Aggregate global deltas (FedAvg) ----
        print(f"\n  Aggregating {len(client_deltas)} client deltas (FedAvg) ...")
        active_client_ids = [cid for cid in selected_clients if cid in client_deltas]
        agg_delta = aggregate_deltas_uniform(
            client_deltas, active_client_ids, client_data_sizes
        )

        # ---- Step 5: Update global adapter ----
        for name in global_adapter_dict:
            if name in agg_delta:
                global_adapter_dict[name] += agg_delta[name]

        # Save updated global adapter
        adapter_save_dir = os.path.join(round_server_dir, "global_content_adapter")
        os.makedirs(adapter_save_dir, exist_ok=True)
        load_lora_state(model, global_adapter_dict)
        model.save_pretrained(adapter_save_dir)
        tokenizer.save_pretrained(adapter_save_dir)
        torch.cuda.empty_cache()

        # ---- Round summary ----
        round_time = time.time() - round_start_time
        wall_time_total += round_time
        avg_loss_global = np.mean(list(client_losses_global.values()))
        avg_loss_personal = np.mean(list(client_losses_personal.values()))
        active_asce_diags = [
            diag for diag in client_personal_diagnostics.values()
            if diag.get("asce_style_status") == "active"
        ]

        round_metrics = {
            "round": t,
            "method": method_name,
            "lambda_ditto": lambda_ditto,
            "implementation": implementation_name,
            "sample_mode": sample_mode,
            "selected_clients": len(selected_clients),
            "selected_client_ids": list(selected_clients),
            "participating_clients": len(client_deltas),
            "participating_client_ids": list(active_client_ids),
            "clients_with_personalized_state_seen": int(
                len(client_personalized_states_seen)
            ),
            "clients_without_personalized_state_remaining": int(
                len(all_client_ids) - len(client_personalized_states_seen)
            ),
            "avg_global_loss": float(avg_loss_global),
            "avg_personal_loss": float(avg_loss_personal),
            "avg_train_loss": float(avg_loss_global),
            "trainable_params": trainable,
            "round_time_seconds": round_time,
            "asce_alignment_enabled": asce_alignment_enabled,
            "asce_active_clients": int(len(active_asce_diags)),
            "asce_avg_style_loss": float(
                np.mean([d.get("asce_style_loss", 0.0) for d in active_asce_diags])
            ) if active_asce_diags else 0.0,
            "asce_avg_mean_cosine": float(
                np.mean([
                    d.get("asce_style_mean_cosine", 0.0)
                    for d in active_asce_diags
                ])
            ) if active_asce_diags else 0.0,
            "asce_avg_contribution_ratio": float(
                np.mean([
                    d.get("asce_contribution_ratio", 0.0)
                    for d in active_asce_diags
                ])
            ) if active_asce_diags else 0.0,
        }
        with open(os.path.join(round_server_dir, "round_metrics.json"), "w") as f:
            json.dump(round_metrics, f, indent=2)

        history.append(round_metrics)

        print(f"\n  Round {t} complete:")
        print(f"    Avg global loss:   {avg_loss_global:.4f}")
        print(f"    Avg personal loss: {avg_loss_personal:.4f}")
        print(
            f"    Personalized coverage: "
            f"{len(client_personalized_states_seen)}/{len(all_client_ids)}"
        )
        if asce_alignment_enabled:
            print(
                f"    ASCE active clients: {round_metrics['asce_active_clients']} | "
                f"avg cosine: {round_metrics['asce_avg_mean_cosine']:.4f}"
            )
        print(f"    Time: {round_time:.1f}s")

    # ===========================================================
    # Save final artifacts
    # ===========================================================
    final_dir = os.path.join(method_dir, "final")
    os.makedirs(final_dir, exist_ok=True)

    with open(os.path.join(final_dir, "training_history.json"), "w") as f:
        json.dump(history, f, indent=2)

    # Save final global adapter
    final_adapter_dir = os.path.join(final_dir, "global_content_adapter")
    os.makedirs(final_adapter_dir, exist_ok=True)
    load_lora_state(model, global_adapter_dict)
    model.save_pretrained(final_adapter_dir)
    tokenizer.save_pretrained(final_adapter_dir)

    # ---- Emit training_budget.csv (artifact contract) ----
    import csv
    budget_path = os.path.join(method_dir, "training_budget.csv")
    with open(budget_path, "w", newline="") as csvf:
        writer = csv.writer(csvf)
        writer.writerow([
            "method", "seed", "non_iid_level", "non_iid_manifest_path",
            "total_rounds", "clients_per_round", "local_epochs",
            "total_gradient_steps", "wall_time_seconds", "lambda_ditto",
            "peft_rank_shared", "peft_alpha_shared",
            "peft_rank_private", "peft_alpha_private",
            "trainable_params_shared", "trainable_params_private",
            "peft_target_modules", "asce_alignment_enabled",
            "asce_scope", "asce_weight", "asce_loss_type",
        ])
        writer.writerow([
            method_name,
            seed,
            noniid_level,
            manifest_info["non_iid_manifest_path"] or "",
            total_rounds,
            clients_per_round,
            fed_cfg["local_epochs"],
            total_gradient_steps,
            round(wall_time_total, 1),
            lambda_ditto,
            peft_cfg["r"],
            peft_cfg.get("alpha", 32),
            peft_cfg["r"],
            peft_cfg.get("alpha", 32),
            trainable,
            trainable,
            "|".join(peft_cfg.get("target_modules", [])),
            bool(asce_alignment_enabled),
            "personalized_branch_only" if asce_alignment_enabled else "",
            asce_align_cfg["weight"] if asce_align_cfg else "",
            asce_align_cfg["loss_type"] if asce_align_cfg else "",
        ])
    print(f"  ✓ training_budget.csv saved to {budget_path}")

    print(f"\n{'='*60}")
    print(f"✓ Ditto training complete ({total_rounds} rounds)")
    print(f"  Method: {method_name}")
    print(f"  λ_ditto: {lambda_ditto}")
    print(f"  ASCE alignment: {'on' if asce_alignment_enabled else 'off'}")
    print(f"  Artifacts: {method_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
