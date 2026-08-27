#!/usr/bin/env python3
"""Shared dual-adapter trainer for the FedDPA baseline.

This script is the implementation source for the FedDPA wrapper
(`53_train_feddpa_peft.py`) so the baseline shares the same data split,
sampling, LoRA budget, and artifact contract as the other reported methods.

Outputs follow the Ditto-style layout expected by the continuation generator:

  runs/<method>/final/global_content_adapter
  runs/<method>/clients/client=<id>/latest/personalized_adapter

For FedDPA-T, the trainer keeps the local adapter and gate-reference
embeddings, then exports a deterministic gate-summary adapter for the existing
generation path while recording the dynamic-gate metadata.
"""

from __future__ import annotations

import argparse
import copy
import csv
import importlib.util
import json
import os
import shutil
import sys
import time
from collections import OrderedDict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

sys.setrecursionlimit(10000)

import numpy as np
import torch
import yaml
from datasets import Dataset
from torch.utils.data import DataLoader
from transformers import DataCollatorForLanguageModeling

from favor_helpers import (
    canonical_author_id,
    count_trainable_params,
    diff_lora_state,
    get_lora_state,
    hash_state_dict,
    load_client_manifest_with_noniid,
    load_lora_state,
    lora_delta_norm,
    record_matches_client_id,
    save_peft_adapter_checkpoint,
)


def load_script_module(module_name: str, filename: str):
    script_dir = Path(__file__).resolve().parent
    module_path = script_dir / filename
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ditto = load_script_module("train_ditto_peft_impl", "49_train_ditto_peft.py")


def clone_state(state: Dict[str, torch.Tensor]) -> OrderedDict:
    return OrderedDict((name, tensor.detach().clone().cpu()) for name, tensor in state.items())


def load_client_texts(pooled_dir: str, client_id: str) -> List[str]:
    train_path = os.path.join(pooled_dir, "train.jsonl")
    texts: List[str] = []
    with open(train_path, "r", encoding="utf-8") as handle:
        for line in handle:
            rec = json.loads(line)
            if record_matches_client_id(rec, client_id):
                texts.append(rec["text"])
    return texts


def tokenize_texts(tokenizer, texts: List[str], max_seq_len: int) -> Optional[Dataset]:
    if not texts:
        return None
    ds = Dataset.from_dict({"text": texts})

    def tokenize_fn(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=max_seq_len,
            padding=False,
        )

    return ds.map(tokenize_fn, batched=True, remove_columns=["text"])


def split_client_data(
    *,
    pooled_dir: str,
    client_id: str,
    tokenizer,
    max_seq_len: int,
    validation_k: int,
    gate_k: int,
    seed: int,
) -> Dict[str, object]:
    """Deterministically hold out local validation examples from LoRA training."""
    texts = load_client_texts(pooled_dir, client_id)
    if not texts:
        return {
            "train_ds": None,
            "val_ds": None,
            "train_texts": [],
            "val_texts": [],
            "gate_ref_texts": [],
            "n_texts": 0,
        }

    validation_k = max(0, min(int(validation_k), len(texts) - 1))
    if validation_k > 0:
        train_texts = texts[:-validation_k]
        val_texts = texts[-validation_k:]
    else:
        train_texts = list(texts)
        val_texts = []

    rng = np.random.RandomState(int(seed))
    ref_pool = train_texts if train_texts else texts
    if len(ref_pool) <= gate_k:
        gate_ref_texts = list(ref_pool)
    else:
        indices = sorted(rng.choice(len(ref_pool), size=int(gate_k), replace=False).tolist())
        gate_ref_texts = [ref_pool[i] for i in indices]

    return {
        "train_ds": tokenize_texts(tokenizer, train_texts, max_seq_len),
        "val_ds": tokenize_texts(tokenizer, val_texts, max_seq_len),
        "train_texts": train_texts,
        "val_texts": val_texts,
        "gate_ref_texts": gate_ref_texts,
        "n_texts": len(texts),
    }


def local_branch_fed_cfg(fed_cfg: Dict[str, object]) -> Dict[str, object]:
    cfg = copy.deepcopy(fed_cfg)
    cfg["local_epochs"] = int(fed_cfg.get("local_epochs_private", fed_cfg.get("local_epochs", 1)))
    cfg["local_lr"] = float(fed_cfg.get("local_lr_private", fed_cfg.get("local_lr", 2e-4)))
    return cfg


def linear_fuse_states(
    global_state: Dict[str, torch.Tensor],
    local_state: Dict[str, torch.Tensor],
    w_global: float,
) -> OrderedDict:
    w_global = float(w_global)
    w_local = 1.0 - w_global
    fused = OrderedDict()
    for name in global_state:
        if name not in local_state:
            fused[name] = global_state[name].detach().clone().cpu()
            continue
        fused[name] = (
            w_global * global_state[name].detach().cpu()
            + w_local * local_state[name].detach().cpu()
        )
    return fused


@torch.no_grad()
def evaluate_state_loss(model, tokenizer, dataset, state, fed_cfg) -> float:
    if dataset is None or len(dataset) == 0:
        return float("inf")

    model.set_adapter("default")
    load_lora_state(model, state)
    model.eval()
    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    loader = DataLoader(
        dataset,
        batch_size=max(1, int(fed_cfg.get("micro_batch_size", 1))),
        shuffle=False,
        collate_fn=collator,
    )
    total_loss = 0.0
    total_items = 0
    device = next(model.parameters()).device
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        bs = int(batch["input_ids"].shape[0])
        total_loss += float(outputs.loss.detach().float().cpu()) * bs
        total_items += bs
    return total_loss / max(1, total_items)


def adaptive_fusion_search(
    *,
    model,
    tokenizer,
    val_ds,
    global_state,
    local_state,
    fed_cfg,
    l1_lambda: float,
    candidate_weights: Iterable[float],
) -> Tuple[OrderedDict, Dict[str, object]]:
    """Gradient-free AdaFusion over a small black-box coefficient budget."""
    candidates = []
    for w_global in candidate_weights:
        fused_state = linear_fuse_states(global_state, local_state, float(w_global))
        val_loss = evaluate_state_loss(model, tokenizer, val_ds, fused_state, fed_cfg)
        objective = val_loss + float(l1_lambda) * (
            abs(float(w_global)) + abs(1.0 - float(w_global))
        )
        candidates.append(
            {
                "w_global": float(w_global),
                "w_local": float(1.0 - float(w_global)),
                "val_loss": float(val_loss),
                "objective": float(objective),
            }
        )
    selected = min(candidates, key=lambda row: row["objective"])
    fused = linear_fuse_states(global_state, local_state, selected["w_global"])
    return fused, {
        "fusion_type": "gradient_free_adaptive_fusion",
        "candidate_weights_global": [float(w) for w in candidate_weights],
        "l1_lambda": float(l1_lambda),
        "selected": selected,
        "candidates": candidates,
    }


@torch.no_grad()
def embed_texts_last_token(model, tokenizer, texts: List[str], max_seq_len: int) -> torch.Tensor:
    if not texts:
        return torch.empty(0, int(model.config.hidden_size), dtype=torch.float32)

    model.eval()
    device = next(model.parameters()).device
    embeddings = []
    for text in texts:
        encoded = tokenizer(
            text,
            truncation=True,
            max_length=max_seq_len,
            padding=False,
            return_tensors="pt",
        )
        encoded = {k: v.to(device) for k, v in encoded.items()}
        outputs = model(**encoded, output_hidden_states=True, use_cache=False)
        hidden = outputs.hidden_states[-1][0]
        attn = encoded.get("attention_mask")
        if attn is None:
            idx = hidden.shape[0] - 1
        else:
            idx = int(attn[0].sum().item()) - 1
            idx = max(0, min(idx, hidden.shape[0] - 1))
        vec = hidden[idx].detach().float().cpu()
        vec = vec / torch.clamp(vec.norm(p=2), min=1e-8)
        embeddings.append(vec)
    return torch.stack(embeddings, dim=0)


def compute_feddpa_gate_summary(
    *,
    model,
    tokenizer,
    global_state,
    val_texts: List[str],
    ref_texts: List[str],
    max_seq_len: int,
    lambda_gate: float,
) -> Tuple[float, Dict[str, object], torch.Tensor]:
    model.set_adapter("default")
    load_lora_state(model, global_state)
    ref_emb = embed_texts_last_token(model, tokenizer, ref_texts, max_seq_len)
    if ref_emb.numel() == 0:
        return 0.5, {"status": "no_gate_refs", "weights": []}, ref_emb

    eval_texts = val_texts if val_texts else ref_texts
    prompt_emb = embed_texts_last_token(model, tokenizer, eval_texts, max_seq_len)
    weights = []
    for vec in prompt_emb:
        sims = torch.matmul(ref_emb, vec)
        score = float(sims.mean().item())
        weights.append(float(np.clip(float(lambda_gate) * score, 0.0, 1.0)))
    mean_weight = float(np.mean(weights)) if weights else 0.5
    return mean_weight, {
        "status": "active",
        "weights": weights,
        "mean_w_global": mean_weight,
        "min_w_global": float(np.min(weights)) if weights else None,
        "max_w_global": float(np.max(weights)) if weights else None,
    }, ref_emb


def save_adapter_from_state(
    *,
    model,
    tokenizer,
    state,
    template_dir: str,
    output_dir: str,
    extra_json: Dict[str, object],
    save_tokenizer: bool = False,
) -> None:
    save_peft_adapter_checkpoint(
        state,
        template_dir=template_dir,
        output_dir=output_dir,
        extra_json=extra_json,
        adapter_name="default",
    )
    if save_tokenizer:
        tokenizer.save_pretrained(output_dir)
    load_lora_state(model, state)


def prepare_asce_if_needed(
    *,
    cfg,
    method_dir: str,
    pooled_dir: str,
    all_client_ids: List[str],
) -> Dict[str, object]:
    asce_raw = cfg.get("asce_private_alignment", {})
    if not asce_raw.get("enabled", False):
        return {"enabled": False}

    base_asce = ditto.load_base_asce_module()
    from asce_private_alignment import resolve_asce_alignment_config
    from style_asce_runtime import load_label_map, load_style_scorer

    asce_cfg = resolve_asce_alignment_config(asce_raw)
    scorer = load_style_scorer(asce_cfg["model_dir"], task="authorship")
    client_texts = base_asce.load_all_client_texts(pooled_dir, all_client_ids)
    cache_dir = os.path.join(
        method_dir,
        "optional",
        asce_cfg.get("cache_dir", "asce_target_cache"),
    )
    os.makedirs(cache_dir, exist_ok=True)
    target_cache, _ = base_asce.build_asce_target_cache(
        client_texts=client_texts,
        asce_align_cfg=asce_cfg,
        scorer=scorer,
        cache_dir=cache_dir,
    )
    base_asce.write_target_collision_report(
        target_cache,
        os.path.join(method_dir, "optional"),
    )

    ce_reference_vectors = None
    ce_label_map = {}
    ce_scale_s = 30.0
    ce_margin_m = 0.0
    if str(asce_cfg.get("loss_type", "cosine")).lower() == "classifier_ce":
        from asce_private_alignment import load_classifier_head_vectors

        vectors_np = load_classifier_head_vectors(
            asce_cfg["model_dir"],
            source=str(asce_cfg.get("ce_reference_source", "weight_vectors")).lower(),
        )
        ce_reference_vectors = torch.as_tensor(vectors_np, dtype=torch.float32)
        label_map_raw = load_label_map(asce_cfg["model_dir"])
        ce_label_map = {str(v): int(k) for k, v in label_map_raw.items()}
        ce_scale_s = float(scorer.scale_s)
        ce_margin_m = float(asce_cfg.get("margin_m", 0.0))

    embedding_dim = int(scorer.embedding_dim)
    del scorer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return {
        "enabled": True,
        "base_asce": base_asce,
        "asce_cfg": asce_cfg,
        "target_cache": target_cache,
        "embedding_dim": embedding_dim,
        "ce_reference_vectors": ce_reference_vectors,
        "ce_label_map": ce_label_map,
        "ce_scale_s": ce_scale_s,
        "ce_margin_m": ce_margin_m,
        "cache_dir": cache_dir,
    }


def resolve_baseline_cfg(cfg: Dict[str, object], family: str) -> Dict[str, object]:
    block = cfg.get("closest_global_local_baseline", {}) or {}
    if family != "feddpa":
        raise ValueError("This public package exposes the FedDPA baseline only.")
    return block.get("feddpa", {}) or {}


def aggregate_client_deltas(client_deltas, active_client_ids, client_data_sizes):
    return ditto.aggregate_deltas_uniform(client_deltas, active_client_ids, client_data_sizes)


def write_json(path: str, payload: Dict[str, object]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def main(default_family: str = "feddpa") -> int:
    parser = argparse.ArgumentParser(description="FedDPA-T dual-adapter trainer")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--gpu", type=int, default=None)
    parser.add_argument("--method-name", type=str, default=None)
    parser.add_argument("--family", choices=["feddpa"], default=default_family)
    parser.add_argument("--resume-round", type=int, default=0)
    parser.add_argument("--enable-asce-alignment", action="store_true")
    args = parser.parse_args()

    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    with open(args.config, "r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    with open(cfg["paths"]["base_model_config"], "r", encoding="utf-8") as handle:
        model_cfg = yaml.safe_load(handle)

    family = args.family
    method_name = args.method_name or cfg.get("closest_global_local_baseline", {}).get(
        "method_name",
        "FedDPA",
    )
    seed = int(cfg["seeds"]["training"])
    fed_cfg = cfg["fed"]
    peft_cfg = cfg["peft"]
    pooled_dir = cfg["paths"]["pooled_dir"]
    prototype_dir = cfg["paths"]["prototype_dir"]
    runs_dir = cfg["paths"]["runs_dir"]
    method_dir = os.path.join(runs_dir, method_name)
    os.makedirs(method_dir, exist_ok=True)

    baseline_cfg = resolve_baseline_cfg(cfg, family)
    validation_k = int(baseline_cfg.get("fusion_budget_examples", 16))
    gate_k = int(baseline_cfg.get("gate_reference_budget", 5))
    seed_gate = int(baseline_cfg.get("seed_gate", seed))
    alpha_gate = float(baseline_cfg.get("alpha_gate", 0.5))
    lambda_gate = float(baseline_cfg.get("lambda_gate", alpha_gate))
    asce_enabled = bool(cfg.get("asce_private_alignment", {}).get("enabled", False)) or bool(
        args.enable_asce_alignment
    )

    print("=" * 60)
    print(f"FedDPA-T training - {method_name}")
    print(f"  seed: {seed}")
    print(f"  rounds: {fed_cfg['rounds']}")
    print(f"  clients/round: {fed_cfg['clients_per_round']}")
    print(f"  local epochs: global={fed_cfg.get('local_epochs', 1)} local={fed_cfg.get('local_epochs_private', 2)}")
    print(f"  validation_k: {validation_k}")
    print(f"  ASCE alignment: {'on' if asce_enabled else 'off'}")
    print(f"  FedDPA gate: S={gate_k} alpha={alpha_gate} lambda={lambda_gate} seed_gate={seed_gate}")
    print("=" * 60)

    manifest_info = load_client_manifest_with_noniid(prototype_dir, cfg)
    all_client_ids = manifest_info["all_client_ids"]
    client_data_sizes = manifest_info["client_data_sizes"]
    sample_mode = str(fed_cfg.get("sample_mode", "uniform")).lower()

    asce_bundle_global = prepare_asce_if_needed(
        cfg=cfg,
        method_dir=method_dir,
        pooled_dir=pooled_dir,
        all_client_ids=all_client_ids,
    ) if asce_enabled else {"enabled": False}

    print("\n  Loading base model ...")
    base_model, tokenizer = ditto.load_base_model_and_tokenizer(model_cfg)
    lora_config = ditto.build_lora_config(peft_cfg)
    from peft import get_peft_model

    model = get_peft_model(base_model, lora_config)
    trainable, total = count_trainable_params(model)
    print(f"  Trainable params: {trainable:,} / {total:,}")

    global_state = get_lora_state(model)
    local_states: Dict[str, OrderedDict] = {}
    clients_seen = set()
    history = []
    total_gradient_steps = 0
    wall_time_total = 0.0

    init_dir = os.path.join(method_dir, "server", "round=0", "global_content_adapter")
    os.makedirs(init_dir, exist_ok=True)
    load_lora_state(model, global_state)
    model.save_pretrained(init_dir)
    tokenizer.save_pretrained(init_dir)

    method_meta = {
        "method_name": method_name,
        "family": family,
        "implementation": "controlled_dual_lora",
        "paper_fidelity": "FedDPA-T dual adapters with cosine dynamic-gate metadata",
        "controlled_equalization": {
            "rank": peft_cfg.get("r"),
            "alpha": peft_cfg.get("alpha"),
            "target_modules": peft_cfg.get("target_modules", []),
            "local_epochs_global": fed_cfg.get("local_epochs"),
            "local_epochs_private_or_local": fed_cfg.get("local_epochs_private"),
        },
        "asce_alignment_enabled": asce_enabled,
    }
    if family == "feddpa":
        method_meta.update(
            {
                "gate_type": "instance_wise_dynamic",
                "gate_variant": "FedDPA-T",
                "gate_reference_budget": gate_k,
                "seed_gate": seed_gate,
                "gate_embedding_source": "global_adapter_last_nonpadding_prompt_token_hidden",
                "alpha_gate": alpha_gate,
                "lambda_gate": lambda_gate,
            }
        )
    write_json(os.path.join(method_dir, "method_meta.json"), method_meta)

    private_fed_cfg = local_branch_fed_cfg(fed_cfg)

    for t in range(max(1, args.resume_round), int(fed_cfg["rounds"]) + 1):
        round_start = time.time()
        print(f"\n{'=' * 60}\n ROUND {t}/{fed_cfg['rounds']}\n{'=' * 60}")

        rng = np.random.RandomState(seed + t)
        selected_clients = ditto.resolve_round_client_sample(
            all_client_ids,
            int(fed_cfg["clients_per_round"]),
            rng,
            sample_mode=sample_mode,
            seen_clients=clients_seen,
        )
        print(f"  Selected {len(selected_clients)} clients")

        client_deltas = {}
        client_losses_global = {}
        client_losses_local = {}
        client_fusion_meta = {}

        for cid in selected_clients:
            client_seed = ditto.stable_client_round_seed(seed, t, cid)
            client_dir = os.path.join(method_dir, "clients", f"client={cid}", "latest")
            os.makedirs(client_dir, exist_ok=True)
            split = split_client_data(
                pooled_dir=pooled_dir,
                client_id=cid,
                tokenizer=tokenizer,
                max_seq_len=int(fed_cfg["max_seq_len"]),
                validation_k=validation_k,
                gate_k=gate_k,
                seed=seed_gate + ditto.stable_seed_offset(cid),
            )
            train_ds = split["train_ds"]
            val_ds = split["val_ds"]
            if train_ds is None or len(train_ds) == 0:
                print(f"  WARNING: no training data for client {cid}, skipping")
                continue
            print(
                f"\n  --- Client {cid} --- "
                f"train={len(train_ds)} val={len(val_ds) if val_ds is not None else 0}"
            )

            global_delta, _, loss_global = ditto.train_client_global_branch(
                model=model,
                tokenizer=tokenizer,
                client_ds=train_ds,
                global_state=global_state,
                fed_cfg=fed_cfg,
                output_dir=client_dir,
                seed=client_seed,
                client_artifact_cfg={"cleanup_trainer_tmp": True},
            )
            client_deltas[cid] = global_delta
            client_losses_global[cid] = float(loss_global)

            local_init = local_states.get(cid)
            if local_init is None:
                local_init = clone_state(global_state)

            asce_bundle = None
            if asce_enabled and asce_bundle_global.get("enabled", False):
                asce_bundle = ditto.build_client_asce_bundle(
                    cid=cid,
                    asce_align_cfg=asce_bundle_global["asce_cfg"],
                    asce_target_cache=asce_bundle_global["target_cache"],
                    asce_lm_hidden_size=int(model.config.hidden_size),
                    asce_embedding_dim=asce_bundle_global["embedding_dim"],
                    arcface_ce_reference_vectors=asce_bundle_global["ce_reference_vectors"],
                    arcface_ce_label_map=asce_bundle_global["ce_label_map"],
                    arcface_ce_scale_s=asce_bundle_global["ce_scale_s"],
                    arcface_ce_margin_m=asce_bundle_global["ce_margin_m"],
                    base_asce=asce_bundle_global["base_asce"],
                )

            local_adapter_path, local_state, loss_local, local_diag = ditto.train_client_personalized_branch(
                model=model,
                tokenizer=tokenizer,
                client_ds=train_ds,
                personalized_state=local_init,
                global_state=global_state,
                fed_cfg=private_fed_cfg,
                lambda_ditto=0.0,
                output_dir=client_dir,
                seed=client_seed,
                client_artifact_cfg={
                    "save_personalized_raw_state": True,
                    "cleanup_trainer_tmp": True,
                    "save_tokenizer_with_personalized_adapter": False,
                },
                asce_bundle=asce_bundle,
                base_asce=asce_bundle_global.get("base_asce"),
            )
            local_states[cid] = local_state
            clients_seen.add(cid)
            client_losses_local[cid] = float(loss_local)

            w_global, gate_meta, ref_embeddings = compute_feddpa_gate_summary(
                model=model,
                tokenizer=tokenizer,
                global_state=global_state,
                val_texts=list(split["val_texts"]),
                ref_texts=list(split["gate_ref_texts"]),
                max_seq_len=int(fed_cfg["max_seq_len"]),
                lambda_gate=lambda_gate,
            )
            torch.save(
                {
                    "client_id": cid,
                    "ref_text_count": len(split["gate_ref_texts"]),
                    "embeddings": ref_embeddings.cpu(),
                    "embedding_source": "global_adapter_last_nonpadding_prompt_token_hidden",
                    "lambda_gate": lambda_gate,
                    "alpha_gate": alpha_gate,
                },
                os.path.join(client_dir, "feddpa_gate_refs.pt"),
            )
            fused_state = linear_fuse_states(global_state, local_state, w_global)
            fusion_meta = {
                "fusion_type": "feddpa_t_dynamic_gate_summary_export",
                "canonical_gate_type": "instance_wise_dynamic",
                "export_note": (
                    "This adapter uses the mean validation gate for compatibility "
                    "with the existing generation path; gate references are saved "
                    "for prompt-wise dynamic generation/probing."
                ),
                "w_global_export": w_global,
                "w_local_export": 1.0 - w_global,
                "gate_summary": gate_meta,
                "gate_reference_budget": gate_k,
                "seed_gate": seed_gate,
                "alpha_gate": alpha_gate,
                "lambda_gate": lambda_gate,
            }

            save_adapter_from_state(
                model=model,
                tokenizer=tokenizer,
                state=fused_state,
                template_dir=local_adapter_path,
                output_dir=os.path.join(client_dir, "personalized_adapter"),
                extra_json={
                    "closest_family_fusion_meta.json": {
                        "client_id": cid,
                        "round": t,
                        "method": method_name,
                        "family": family,
                        "global_state_hash": hash_state_dict(global_state),
                        "local_state_hash": hash_state_dict(local_state),
                        "fused_state_hash": hash_state_dict(fused_state),
                        "validation_examples": len(val_ds) if val_ds is not None else 0,
                        "training_examples": len(train_ds),
                        "asce_alignment_enabled": asce_enabled,
                        **fusion_meta,
                    }
                },
            )
            client_fusion_meta[cid] = fusion_meta

            write_json(
                os.path.join(client_dir, "client_metrics.json"),
                {
                    "client_id": cid,
                    "round": t,
                    "train_samples_total": int(split["n_texts"]),
                    "train_samples_after_validation_holdout": len(train_ds),
                    "validation_samples_held_out": len(val_ds) if val_ds is not None else 0,
                    "global_loss": float(loss_global),
                    "local_loss": float(loss_local),
                    "global_delta_norm": lora_delta_norm(
                        global_state,
                        OrderedDict(
                            (name, global_state[name] + global_delta[name])
                            for name in global_delta
                        ),
                    ),
                    "fusion": fusion_meta,
                    "asce_alignment_enabled": bool(asce_bundle),
                    "asce_style_status": local_diag.get("asce_style_status", "disabled"),
                    "asce_style_loss": local_diag.get("asce_style_loss", 0.0),
                    "asce_style_mean_cosine": local_diag.get("asce_style_mean_cosine", 0.0),
                },
            )
            total_gradient_steps += ditto.estimate_gradient_steps(len(train_ds), fed_cfg)
            total_gradient_steps += ditto.estimate_gradient_steps(len(train_ds), private_fed_cfg)
            try:
                torch._dynamo.reset()
            except Exception:
                pass

        if not client_deltas:
            print("  WARNING: no client deltas produced, skipping aggregation")
            continue

        active_client_ids = [cid for cid in selected_clients if cid in client_deltas]
        agg_delta = aggregate_client_deltas(client_deltas, active_client_ids, client_data_sizes)
        for name in global_state:
            if name in agg_delta:
                global_state[name] += agg_delta[name]

        round_server_dir = os.path.join(method_dir, "server", f"round={t}")
        adapter_save_dir = os.path.join(round_server_dir, "global_content_adapter")
        os.makedirs(adapter_save_dir, exist_ok=True)
        load_lora_state(model, global_state)
        model.save_pretrained(adapter_save_dir)
        tokenizer.save_pretrained(adapter_save_dir)
        torch.cuda.empty_cache()

        round_time = time.time() - round_start
        wall_time_total += round_time
        round_metrics = {
            "round": t,
            "method": method_name,
            "family": family,
            "sample_mode": sample_mode,
            "selected_client_ids": list(selected_clients),
            "participating_client_ids": list(active_client_ids),
            "clients_with_local_state_seen": len(clients_seen),
            "avg_global_loss": float(np.mean(list(client_losses_global.values()))),
            "avg_local_loss": float(np.mean(list(client_losses_local.values()))),
            "round_time_seconds": round_time,
            "asce_alignment_enabled": asce_enabled,
        }
        write_json(os.path.join(round_server_dir, "round_metrics.json"), round_metrics)
        history.append(round_metrics)
        print(
            f"  Round {t} complete: global={round_metrics['avg_global_loss']:.4f} "
            f"local={round_metrics['avg_local_loss']:.4f} "
            f"coverage={len(clients_seen)}/{len(all_client_ids)} "
            f"time={round_time:.1f}s"
        )

    final_dir = os.path.join(method_dir, "final")
    os.makedirs(final_dir, exist_ok=True)
    write_json(os.path.join(final_dir, "training_history.json"), {"history": history})
    final_adapter_dir = os.path.join(final_dir, "global_content_adapter")
    os.makedirs(final_adapter_dir, exist_ok=True)
    load_lora_state(model, global_state)
    model.save_pretrained(final_adapter_dir)
    tokenizer.save_pretrained(final_adapter_dir)

    budget_path = os.path.join(method_dir, "training_budget.csv")
    with open(budget_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "method",
                "seed",
                "family",
                "non_iid_level",
                "total_rounds",
                "clients_per_round",
                "local_epochs_global",
                "local_epochs_private_or_local",
                "total_gradient_steps",
                "wall_time_seconds",
                "peft_rank_shared",
                "peft_alpha_shared",
                "peft_rank_private",
                "peft_alpha_private",
                "trainable_params_shared",
                "trainable_params_private",
                "peft_target_modules",
                "asce_alignment_enabled",
                "validation_k",
                "gate_reference_budget",
                "alpha_gate",
                "lambda_gate",
            ]
        )
        writer.writerow(
            [
                method_name,
                seed,
                family,
                manifest_info.get("non_iid_level"),
                fed_cfg["rounds"],
                fed_cfg["clients_per_round"],
                fed_cfg.get("local_epochs"),
                fed_cfg.get("local_epochs_private"),
                total_gradient_steps,
                round(wall_time_total, 1),
                peft_cfg["r"],
                peft_cfg.get("alpha", 32),
                peft_cfg.get("rank_private", peft_cfg["r"]),
                peft_cfg.get("alpha_private", peft_cfg.get("alpha", 32)),
                trainable,
                trainable,
                "|".join(peft_cfg.get("target_modules", [])),
                bool(asce_enabled),
                validation_k,
                gate_k if family == "feddpa" else "",
                alpha_gate if family == "feddpa" else "",
                lambda_gate if family == "feddpa" else "",
            ]
        )

    print(f"\n{'=' * 60}")
    print(f"✓ {method_name} complete")
    print(f"  Artifacts: {method_dir}")
    print(f"  training_budget.csv: {budget_path}")
    print(f"{'=' * 60}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
