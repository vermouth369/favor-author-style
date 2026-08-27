#!/usr/bin/env python3
"""FAVoR main training driver.

This is the paper-facing implementation of FAVoR: each selected client first
trains a shared LoRA adapter update that is uploaded for FedAvg aggregation,
then trains a client-local private residual pack with ASCE alignment. The
private residual pack is never aggregated; generation reconstructs the
client adapter as shared endpoint + private residual.
"""

from __future__ import annotations

import argparse
import copy
import csv
import importlib.util
import json
import os
import sys
import time
from collections import OrderedDict
from pathlib import Path

sys.setrecursionlimit(10000)

import numpy as np
import torch
import yaml
from peft import get_peft_model

from favor_helpers import (
    active_roster_snapshot_path,
    canonical_author_id,
    diff_adapter_states,
    get_lora_state,
    hash_state_dict,
    load_client_manifest_with_noniid,
    load_lora_state,
    save_peft_adapter_checkpoint,
    strip_adapter_name_infix,
)


def load_base_asce_module():
    script_dir = Path(__file__).resolve().parent
    module_path = script_dir / "44_train_base_asce.py"
    spec = importlib.util.spec_from_file_location("train_base_asce_impl", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def resolve_path(path_value, config_path):
    if path_value is None:
        return None
    path = Path(str(path_value)).expanduser()
    if path.is_absolute() or path.exists():
        return str(path)
    config_relative = Path(config_path).resolve().parent / path
    if config_relative.exists():
        return str(config_relative)
    return str(path)


def load_runtime_config(config_path):
    with open(config_path, "r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)

    # Allow the paper-facing metadata config to be used directly by normalizing
    # it into the runtime keys consumed by the training scripts.
    if "fed" not in cfg and "federated" in cfg:
        fed_src = cfg["federated"]
        cfg["fed"] = {
            "rounds": fed_src["rounds"],
            "clients_per_round": fed_src["clients_per_round"],
            "local_epochs": fed_src.get("local_epochs_shared", 1),
            "local_epochs_private": fed_src.get("local_epochs_private", 2),
            "local_lr": fed_src.get("local_lr_shared", 2.0e-4),
            "local_lr_private": fed_src.get("local_lr_private", 2.0e-4),
            "warmup_ratio": fed_src.get("warmup_ratio", 0.03),
            "max_seq_len": fed_src.get("max_seq_len", 1024),
            "micro_batch_size": fed_src.get("micro_batch_size", 4),
            "gradient_accumulation_steps": fed_src.get("gradient_accumulation_steps", 8),
            "sample_mode": "stratified_uncovered",
        }

    if "peft" in cfg and "r" not in cfg["peft"]:
        cfg["peft"]["r"] = cfg["peft"].get("rank", 16)

    cfg.setdefault("paths", {})
    cfg["paths"].setdefault("pooled_dir", cfg.get("data", {}).get("pooled_dir", "exp1/data/pooled/K=50"))
    cfg["paths"].setdefault("runs_dir", "runs/phase1_medium/seed2026")
    cfg["paths"].setdefault("prototype_dir", "benchmark/favor_bench_v02_1")

    if "data" in cfg and "non_iid" not in cfg["data"]:
        cfg["data"]["non_iid"] = {
            "level": cfg["data"].get("non_iid_level", "medium"),
            "manifest_path": cfg["data"].get(
                "manifest_path",
                "benchmark/favor_bench_v02_1/non_iid_medium_v02.json",
            ),
        }

    if "asce_private_alignment" not in cfg:
        asce = cfg.get("favor", {}).get("asce_alignment", {})
        cfg["asce_private_alignment"] = {
            "enabled": bool(asce.get("enabled", False)),
            "model_dir": asce.get(
                "model_dir",
                "exp1/runs/exp1_asce_full/K=50/author_classifier",
            ),
            "target_source": asce.get("target_source", "empirical_per_client_reference_texts"),
            "normalize_targets": True,
            "loss_type": asce.get("loss", "cosine"),
            "weight": asce.get("weight", 0.3),
            "warmup_fraction": 0.05,
            "min_supervised_tokens": 8,
            "pooling": "mean_supervised_tokens",
            "fail_policy": "error",
            "projector": asce.get(
                "projector",
                {"type": "mlp_2", "hidden_dim": 1024, "dropout": 0.05},
            ),
        }

    seeds = cfg.setdefault("seeds", {})
    if isinstance(seeds.get("training"), list):
        seeds["training_replicates"] = list(seeds["training"])
        seeds["training"] = int(seeds["training"][0])

    # Paths are intentionally left as archive-root-relative strings unless the
    # caller provided paths relative to the config file and those files exist.
    for section, keys in {
        "paths": ["base_model_config", "pooled_dir", "runs_dir", "prototype_dir"],
        "data.non_iid": ["manifest_path"],
    }.items():
        if section == "paths":
            block = cfg.get("paths", {})
        else:
            block = cfg.get("data", {}).get("non_iid", {})
        for key in keys:
            if key in block:
                block[key] = resolve_path(block[key], config_path)

    return cfg


def load_model_config(cfg):
    model_config_path = cfg.get("paths", {}).get("base_model_config")
    if model_config_path and os.path.exists(model_config_path):
        with open(model_config_path, "r", encoding="utf-8") as handle:
            return yaml.safe_load(handle)

    model_src = cfg.get("model", {})
    base_model = model_src.get("base_model") or model_src.get("name") or "Qwen/Qwen2.5-3B-Instruct"
    return {
        "model": {
            "name": base_model,
            "trust_remote_code": True,
        },
        "quantization": {
            "load_in_4bit": str(model_src.get("quantization", "nf4_4bit")).lower() in {"nf4_4bit", "4bit", "true"},
            "bnb_4bit_quant_type": "nf4",
            "bnb_4bit_compute_dtype": model_src.get("compute_dtype", "bfloat16"),
            "bnb_4bit_use_double_quant": True,
        },
    }


def private_stage_fed_cfg(fed_cfg):
    private_cfg = copy.deepcopy(fed_cfg)
    private_cfg["local_epochs"] = int(fed_cfg.get("local_epochs_private", 2))
    private_cfg["local_lr"] = float(fed_cfg.get("local_lr_private", fed_cfg.get("local_lr", 2.0e-4)))
    return private_cfg


def build_client_asce_bundle(
    *,
    cid,
    asce_align_cfg,
    target_cache,
    lm_hidden_size,
    embedding_dim,
    ce_reference_vectors,
    ce_label_map,
    ce_scale_s,
    ce_margin_m,
    base_asce,
):
    if not asce_align_cfg or not asce_align_cfg.get("enabled", False):
        return None
    if cid not in target_cache:
        return None

    bundle = {
        "enabled": True,
        "target_vector": target_cache[cid],
        "hidden_size": int(lm_hidden_size),
        "embedding_dim": int(embedding_dim),
        "weight": float(asce_align_cfg["weight"]),
        "loss_type": asce_align_cfg["loss_type"],
        "warmup_fraction": float(asce_align_cfg["warmup_fraction"]),
        "min_supervised_tokens": int(asce_align_cfg["min_supervised_tokens"]),
        "pooling": asce_align_cfg["pooling"],
        "schedule": copy.deepcopy(asce_align_cfg.get("schedule", {})),
        "target_source": asce_align_cfg.get("target_source", "reference_mean"),
        "projector_cfg": asce_align_cfg.get("projector", {}),
        "logging": asce_align_cfg.get("logging", {}),
    }

    if str(asce_align_cfg.get("loss_type", "cosine")).lower() == "classifier_ce":
        canonical_cid = str(base_asce.canonical_author_id(cid))
        class_idx = ce_label_map.get(canonical_cid)
        if class_idx is None:
            sample = list(ce_label_map.keys())[:5]
            raise KeyError(
                f"ASCE classifier_ce: client_id={cid!r} "
                f"(canonical={canonical_cid!r}) not in label map; sample={sample}"
            )
        bundle["reference_vectors"] = ce_reference_vectors
        bundle["class_index"] = int(class_idx)
        bundle["scale_s"] = float(ce_scale_s)
        bundle["margin_m"] = float(ce_margin_m)

    return bundle


def prepare_asce_alignment(cfg, method_dir, model, client_texts, base_asce):
    asce_align_raw = cfg.get("asce_private_alignment", {})
    if not asce_align_raw.get("enabled", False):
        return {"enabled": False}

    from asce_private_alignment import load_classifier_head_vectors, resolve_asce_alignment_config
    from style_asce_runtime import load_label_map, load_style_scorer

    asce_align_cfg = resolve_asce_alignment_config(asce_align_raw)
    asce_model_dir = asce_align_cfg["model_dir"]
    print(f"\n  Loading ASCE scorer: {asce_model_dir}")
    scorer = load_style_scorer(asce_model_dir, task="authorship")
    lm_hidden_size = int(model.config.hidden_size)
    cache_dir = os.path.join(method_dir, "optional", asce_align_cfg.get("cache_dir", "asce_target_cache"))
    target_cache, target_meta = base_asce.build_asce_target_cache(
        client_texts,
        asce_align_cfg,
        scorer,
        cache_dir,
    )
    base_asce.write_target_collision_report(
        target_cache,
        os.path.join(method_dir, "optional"),
    )

    ce_reference_vectors = None
    ce_label_map = {}
    ce_scale_s = float(getattr(scorer, "scale_s", 30.0))
    ce_margin_m = float(asce_align_cfg.get("margin_m", 0.0))
    if str(asce_align_cfg.get("loss_type", "cosine")).lower() == "classifier_ce":
        ce_source = asce_align_cfg.get("ce_reference_source", "weight_vectors")
        ce_reference_vectors = load_classifier_head_vectors(asce_model_dir, source=ce_source)
        label_map_raw = load_label_map(asce_model_dir)
        ce_label_map = {str(v): int(k) for k, v in label_map_raw.items()}

    return {
        "enabled": True,
        "cfg": asce_align_cfg,
        "target_cache": target_cache,
        "target_meta": target_meta,
        "embedding_dim": int(scorer.embedding_dim),
        "lm_hidden_size": lm_hidden_size,
        "ce_reference_vectors": ce_reference_vectors,
        "ce_label_map": ce_label_map,
        "ce_scale_s": ce_scale_s,
        "ce_margin_m": ce_margin_m,
    }


def write_json(path, payload):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def save_active_roster_snapshot(runs_dir, manifest_info, client_ids):
    path = active_roster_snapshot_path(runs_dir)
    write_json(
        path,
        {
            "client_ids": list(client_ids),
            "non_iid_manifest_path": manifest_info.get("non_iid_manifest_path"),
            "non_iid_level": manifest_info.get("non_iid_level"),
            "downgraded_to_clients_json": bool(manifest_info.get("non_iid_downgrade", False)),
        },
    )


def save_residual_pack(
    *,
    shared_state,
    private_state,
    shared_endpoint_dir,
    residual_dir,
    client_id,
    round_idx,
    method_name,
):
    residual_state = diff_adapter_states(shared_state, private_state)
    shared_hash = hash_state_dict(strip_adapter_name_infix(shared_state))
    residual_hash = hash_state_dict(strip_adapter_name_infix(residual_state))
    final_hash = hash_state_dict(strip_adapter_name_infix(private_state))
    save_peft_adapter_checkpoint(
        residual_state,
        template_dir=shared_endpoint_dir,
        output_dir=residual_dir,
        extra_json={
            "private_residual_pack_meta.json": {
                "client_id": client_id,
                "round": int(round_idx),
                "method": method_name,
                "residual_semantics": True,
                "reconstruction": "shared_endpoint_adapter_plus_private_residual_pack",
                "shared_adapter_hash": shared_hash,
                "residual_adapter_hash": residual_hash,
                "final_adapter_hash": final_hash,
            }
        },
    )
    return {
        "shared_adapter_hash": shared_hash,
        "residual_adapter_hash": residual_hash,
        "final_adapter_hash": final_hash,
    }


def main():
    parser = argparse.ArgumentParser(description="FAVoR shared-private residual training")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--gpu", type=int, default=None)
    parser.add_argument("--method-name", type=str, default="FAVoR")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--client-id", type=str, default=None)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    cfg = load_runtime_config(args.config)
    if args.seed is not None:
        cfg["seeds"]["training"] = int(args.seed)
        cfg["paths"]["runs_dir"] = os.path.join("runs", "phase1_medium", f"seed{args.seed}")
    model_cfg = load_model_config(cfg)
    base_asce = load_base_asce_module()

    paths = cfg["paths"]
    pooled_dir = paths["pooled_dir"]
    runs_dir = paths["runs_dir"]
    prototype_dir = paths["prototype_dir"]
    fed_cfg = cfg["fed"]
    peft_cfg = cfg["peft"]
    seed = int(cfg["seeds"]["training"])
    method_name = args.method_name
    method_dir = os.path.join(runs_dir, method_name)
    os.makedirs(method_dir, exist_ok=True)

    manifest_info = load_client_manifest_with_noniid(prototype_dir, cfg)
    all_client_ids = list(manifest_info["all_client_ids"])
    if args.client_id:
        all_client_ids = [args.client_id]
    client_data_sizes = manifest_info["client_data_sizes"]
    save_active_roster_snapshot(runs_dir, manifest_info, all_client_ids)

    print("=" * 60)
    print(f"FAVoR Training - {method_name}")
    print(f"  seed: {seed}")
    print(f"  clients: {len(all_client_ids)}")
    print(f"  non-IID: {manifest_info['non_iid_level']}")
    print(f"  rounds: {fed_cfg['rounds']}")
    print(f"  clients/round: {fed_cfg['clients_per_round']}")
    print(f"  shared/private epochs: {fed_cfg.get('local_epochs', 1)} / {fed_cfg.get('local_epochs_private', 2)}")
    print("=" * 60)

    resolved = {
        "config": cfg,
        "model_config": model_cfg,
        "method_name": method_name,
        "implementation": "favor_shared_private_residual_asce",
        "noniid_manifest_path": manifest_info["non_iid_manifest_path"],
        "noniid_level": manifest_info["non_iid_level"],
        "shared_adapter_uploaded": True,
        "private_residual_pack": {
            "local_only": True,
            "reconstruction": "shared_endpoint_adapter_plus_private_residual_pack",
        },
        "asce_alignment_enabled": bool(cfg.get("asce_private_alignment", {}).get("enabled", False)),
    }
    write_json(os.path.join(method_dir, "config_resolved.json"), resolved)

    print("\n  Loading base model ...")
    base_model, tokenizer = base_asce.load_base_model_and_tokenizer(model_cfg)
    lora_config = base_asce.build_lora_config(peft_cfg)
    model = get_peft_model(base_model, lora_config)
    trainable, total = base_asce.count_trainable_params(model)
    print(f"  Trainable params: {trainable:,} / {total:,}")

    print("\n  Loading client training texts ...")
    client_texts = base_asce.load_all_client_texts(pooled_dir, all_client_ids)
    print(f"  Client text cache: {sum(1 for texts in client_texts.values() if texts)}/{len(all_client_ids)}")

    asce = prepare_asce_alignment(cfg, method_dir, model, client_texts, base_asce)
    global_state = get_lora_state(model)
    clients_seen = set()
    history = []
    total_gradient_steps = 0
    wall_time_total = 0.0

    init_dir = os.path.join(method_dir, "server", "round=0", "global_content_adapter")
    os.makedirs(init_dir, exist_ok=True)
    load_lora_state(model, global_state)
    model.save_pretrained(init_dir)
    tokenizer.save_pretrained(init_dir)

    shared_prox_mu = float(cfg.get("losses", {}).get("prox_shared", {}).get("weight", 0.0))
    sample_mode = str(fed_cfg.get("sample_mode", "uniform")).lower()
    total_rounds = int(fed_cfg["rounds"])
    clients_per_round = int(fed_cfg["clients_per_round"])
    private_cfg = private_stage_fed_cfg(fed_cfg)

    for round_idx in range(1, total_rounds + 1):
        round_start = time.time()
        rng = np.random.RandomState(seed + round_idx)
        selected_clients = base_asce.resolve_round_client_sample(
            all_client_ids,
            clients_per_round,
            rng,
            sample_mode=sample_mode,
            seen_clients=clients_seen,
        )
        print(f"\n{'=' * 60}\n ROUND {round_idx}/{total_rounds}\n{'=' * 60}")
        print(f"  Selected clients: {selected_clients}")

        client_deltas = {}
        client_summaries = []

        for cid in selected_clients:
            texts = client_texts.get(cid, [])
            client_ds = base_asce.dataset_from_texts(texts, tokenizer, int(fed_cfg["max_seq_len"]))
            if client_ds is None or len(client_ds) == 0:
                print(f"  WARNING: no training data for client {cid}, skipping")
                continue

            client_seed = base_asce.stable_client_round_seed(seed, round_idx, cid)
            round_client_dir = os.path.join(method_dir, "clients", f"client={cid}", f"round={round_idx}")
            os.makedirs(round_client_dir, exist_ok=True)

            print(f"\n  --- Client {cid}: shared stage ---")
            shared_delta, shared_loss, shared_norm, shared_diag = base_asce.train_client_local_base_asce(
                model=model,
                tokenizer=tokenizer,
                global_adapter_dict=global_state,
                client_ds=client_ds,
                fed_cfg=fed_cfg,
                output_dir=os.path.join(round_client_dir, "shared_stage"),
                seed=client_seed,
                prox_mu=shared_prox_mu,
                asce_bundle=None,
            )
            shared_endpoint_state = get_lora_state(model)
            shared_endpoint_dir = os.path.join(round_client_dir, "shared_endpoint_adapter")
            os.makedirs(shared_endpoint_dir, exist_ok=True)
            load_lora_state(model, shared_endpoint_state)
            model.save_pretrained(shared_endpoint_dir)
            tokenizer.save_pretrained(shared_endpoint_dir)

            print(f"\n  --- Client {cid}: private residual stage ---")
            client_asce_bundle = None
            if asce.get("enabled", False):
                client_asce_bundle = build_client_asce_bundle(
                    cid=cid,
                    asce_align_cfg=asce["cfg"],
                    target_cache=asce["target_cache"],
                    lm_hidden_size=asce["lm_hidden_size"],
                    embedding_dim=asce["embedding_dim"],
                    ce_reference_vectors=asce["ce_reference_vectors"],
                    ce_label_map=asce["ce_label_map"],
                    ce_scale_s=asce["ce_scale_s"],
                    ce_margin_m=asce["ce_margin_m"],
                    base_asce=base_asce,
                )

            _, private_loss, private_norm, private_diag = base_asce.train_client_local_base_asce(
                model=model,
                tokenizer=tokenizer,
                global_adapter_dict=shared_endpoint_state,
                client_ds=client_ds,
                fed_cfg=private_cfg,
                output_dir=os.path.join(round_client_dir, "private_stage"),
                seed=client_seed + 17,
                prox_mu=0.0,
                asce_bundle=client_asce_bundle,
            )
            private_state = get_lora_state(model)
            residual_dir = os.path.join(round_client_dir, "private_residual_pack")
            residual_meta = save_residual_pack(
                shared_state=shared_endpoint_state,
                private_state=private_state,
                shared_endpoint_dir=shared_endpoint_dir,
                residual_dir=residual_dir,
                client_id=cid,
                round_idx=round_idx,
                method_name=method_name,
            )

            client_deltas[cid] = shared_delta
            clients_seen.add(cid)
            total_gradient_steps += base_asce.estimate_gradient_steps(len(client_ds), fed_cfg)
            total_gradient_steps += base_asce.estimate_gradient_steps(len(client_ds), private_cfg)
            client_summaries.append(
                {
                    "client_id": cid,
                    "round": round_idx,
                    "shared_loss": float(shared_loss),
                    "private_loss": float(private_loss),
                    "shared_delta_norm": float(shared_norm),
                    "private_delta_norm": float(private_norm),
                    "asce_status": private_diag.get("asce_style_status", "disabled"),
                    **residual_meta,
                }
            )
            write_json(os.path.join(round_client_dir, "client_summary.json"), client_summaries[-1])

        if not client_deltas:
            print("  WARNING: no client shared deltas produced; skipping server update")
            continue

        active_clients = [cid for cid in selected_clients if cid in client_deltas]
        aggregated_delta = base_asce.aggregate_deltas_uniform(
            client_deltas,
            active_clients,
            client_data_sizes,
        )
        for name in global_state:
            if name in aggregated_delta:
                global_state[name] = global_state[name] + aggregated_delta[name]

        round_server_dir = os.path.join(method_dir, "server", f"round={round_idx}")
        adapter_save_dir = os.path.join(round_server_dir, "global_content_adapter")
        os.makedirs(adapter_save_dir, exist_ok=True)
        load_lora_state(model, global_state)
        model.save_pretrained(adapter_save_dir)
        tokenizer.save_pretrained(adapter_save_dir)

        round_time = time.time() - round_start
        wall_time_total += round_time
        round_metrics = {
            "round": round_idx,
            "method": method_name,
            "selected_client_ids": list(selected_clients),
            "participating_client_ids": list(active_clients),
            "coverage": len(clients_seen),
            "avg_shared_loss": float(np.mean([row["shared_loss"] for row in client_summaries])),
            "avg_private_loss": float(np.mean([row["private_loss"] for row in client_summaries])),
            "asce_active_clients": int(sum(row["asce_status"] == "active" for row in client_summaries)),
            "wall_time_seconds": round_time,
        }
        write_json(os.path.join(round_server_dir, "round_metrics.json"), round_metrics)
        history.append(round_metrics)
        print(
            f"  Round {round_idx} complete: "
            f"shared={round_metrics['avg_shared_loss']:.4f}, "
            f"private={round_metrics['avg_private_loss']:.4f}, "
            f"coverage={len(clients_seen)}/{len(all_client_ids)}"
        )
        torch.cuda.empty_cache()

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
                "non_iid_level",
                "total_rounds",
                "clients_per_round",
                "local_epochs_shared",
                "local_epochs_private",
                "total_gradient_steps",
                "wall_time_seconds",
                "peft_rank_shared",
                "peft_alpha_shared",
                "peft_rank_private",
                "peft_alpha_private",
                "asce_alignment_enabled",
                "shared_adapter_uploaded",
                "private_residual_pack_local_only",
            ]
        )
        writer.writerow(
            [
                method_name,
                seed,
                manifest_info.get("non_iid_level"),
                total_rounds,
                clients_per_round,
                fed_cfg.get("local_epochs", 1),
                private_cfg.get("local_epochs", 2),
                total_gradient_steps,
                round(wall_time_total, 1),
                peft_cfg["r"],
                peft_cfg.get("alpha", 32),
                peft_cfg.get("rank_private", peft_cfg["r"]),
                peft_cfg.get("alpha_private", peft_cfg.get("alpha", 32)),
                bool(asce.get("enabled", False)),
                True,
                True,
            ]
        )

    print(f"\n{'=' * 60}")
    print(f"FAVoR training complete ({total_rounds} rounds)")
    print(f"  Method: {method_name}")
    print(f"  Artifacts: {method_dir}")
    print(f"  training_budget.csv: {budget_path}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
