#!/usr/bin/env python3
"""45_train_local_only_asce.py — Local-only QLoRA + ASCE alignment.

Local-only + ASCE trains an independent adapter for each client from the same
base initialization, with no federation, aggregation, or shared-adapter
communication. It adds the same ArcFace/ASCE alignment objective used by the
FAVoR private stage.

Output: runs/.../Local-only + ASCE/client={CID}/adapter/
"""

import argparse
import copy
import importlib.util
import json
import os
import time
from pathlib import Path

import torch
import yaml
from peft import get_peft_model


def load_base_asce_module():
    script_dir = Path(__file__).resolve().parent
    module_path = script_dir / "44_train_base_asce.py"
    spec = importlib.util.spec_from_file_location("train_base_asce_impl", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


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


def main():
    parser = argparse.ArgumentParser(description="Local-only QLoRA + ASCE")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--gpu", type=int, default=None)
    parser.add_argument("--client-id", type=str, default=None)
    parser.add_argument("--method-name", type=str, default="Local-only + ASCE")
    parser.add_argument("--force", action="store_true", help="Retrain even if adapter exists")
    args = parser.parse_args()

    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    base_asce = load_base_asce_module()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    with open(cfg["paths"]["base_model_config"], "r", encoding="utf-8") as f:
        model_cfg = yaml.safe_load(f)

    prototype_dir = cfg["paths"]["prototype_dir"]
    pooled_dir = cfg["paths"]["pooled_dir"]
    runs_dir = cfg["paths"]["runs_dir"]
    fed_cfg = cfg["fed"]
    peft_cfg = cfg["peft"]
    seed = cfg["seeds"]["training"]

    method_name = args.method_name
    method_dir = os.path.join(runs_dir, method_name)
    os.makedirs(method_dir, exist_ok=True)

    print("=" * 60)
    print(f"Local-Only QLoRA + ASCE Training — {method_name}")
    print(f"  ASCE enabled: {bool(cfg.get('asce_private_alignment', {}).get('enabled', False))}")
    print(f"  B1 compute epochs/client: {fed_cfg['local_epochs']} * {fed_cfg['rounds']}")
    print("=" * 60)

    manifest_info = base_asce.load_client_manifest_with_noniid(prototype_dir, cfg)
    all_client_ids = manifest_info["all_client_ids"]
    if args.client_id:
        all_client_ids = [args.client_id]
    print(f"  Total clients: {len(all_client_ids)} (non-IID: {manifest_info['non_iid_level']})")

    resolved = {
        "config": cfg,
        "model_config": model_cfg,
        "method_name": method_name,
        "implementation": "local_only_asce_independent_adapters",
        "noniid_manifest_path": manifest_info["non_iid_manifest_path"],
        "noniid_level": manifest_info["non_iid_level"],
        "non_iid_downgrade": manifest_info.get("non_iid_downgrade", False),
        "personalization": {
            "local_only": True,
            "private_residual_pack": False,
            "aggregation": False,
        },
    }
    with open(os.path.join(method_dir, "config_resolved.yaml"), "w", encoding="utf-8") as f:
        yaml.dump(resolved, f, default_flow_style=False)

    print("\n  Loading client texts from pooled train split ...")
    client_texts = base_asce.load_all_client_texts(pooled_dir, all_client_ids)
    print(f"  Client text cache: {sum(1 for v in client_texts.values() if v)}/{len(all_client_ids)} clients")

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
        from style_asce_runtime import load_label_map, load_style_scorer

        asce_align_cfg = resolve_asce_alignment_config(asce_align_raw)
        asce_model_dir = asce_align_cfg["model_dir"]
        print(f"\n  Loading ArcFace scorer for local ASCE: {asce_model_dir}")
        scorer = load_style_scorer(asce_model_dir, task="authorship")
        asce_embedding_dim = int(scorer.embedding_dim)
        print(f"  ArcFace scorer loaded: backend={scorer.backend}, embedding_dim={asce_embedding_dim}")

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

            ce_source = str(asce_align_cfg.get("ce_reference_source", "weight_vectors")).lower()
            vectors_np = load_classifier_head_vectors(asce_model_dir, source=ce_source)
            arcface_ce_reference_vectors = torch.as_tensor(vectors_np, dtype=torch.float32)
            label_map_raw = load_label_map(asce_model_dir)
            arcface_ce_label_map = {str(v): int(k) for k, v in label_map_raw.items()}
            arcface_ce_scale_s = float(scorer.scale_s)
            arcface_ce_margin_m = float(asce_align_cfg.get("margin_m", 0.0))

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
            "target_cache_dir": cache_dir,
            "targets_cached": len(asce_target_cache),
        }
        del scorer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    else:
        resolved["asce_private_alignment"] = {"enabled": False}

    print("\n  Loading base model ...")
    base_model, tokenizer = base_asce.load_base_model_and_tokenizer(model_cfg)
    lora_config = base_asce.build_lora_config(peft_cfg)
    model = get_peft_model(base_model, lora_config)
    initial_lora_state = base_asce.get_lora_state(model)
    trainable, total = base_asce.count_trainable_params(model)
    print(f"  Trainable params: {trainable:,} / {total:,}")

    asce_lm_hidden_size = int(model.config.hidden_size)
    if asce_align_cfg is not None and asce_align_cfg.get("enabled", False):
        resolved["asce_private_alignment"]["lm_hidden_size"] = asce_lm_hidden_size
    with open(os.path.join(method_dir, "config_resolved.yaml"), "w", encoding="utf-8") as f:
        yaml.dump(resolved, f, default_flow_style=False)

    local_fed_cfg = copy.deepcopy(fed_cfg)
    local_fed_cfg["local_epochs"] = float(fed_cfg["local_epochs"]) * float(fed_cfg["rounds"])

    summary = []
    total_gradient_steps = 0
    start_time = time.time()

    for i, cid in enumerate(all_client_ids, 1):
        print(f"\n--- Client {i}/{len(all_client_ids)}: {cid} ---")
        client_dir = os.path.join(method_dir, f"client={cid}")
        adapter_dir = os.path.join(client_dir, "adapter")
        if (
            not args.force
            and os.path.exists(os.path.join(adapter_dir, "adapter_config.json"))
            and os.path.exists(os.path.join(adapter_dir, "adapter_model.safetensors"))
        ):
            print("  Already trained, skipping")
            continue

        texts = client_texts.get(cid, [])
        if not texts:
            print(f"  WARNING: no data for {cid}, skipping")
            continue
        ds = base_asce.dataset_from_texts(texts, tokenizer, fed_cfg["max_seq_len"])

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
        print(
            f"  Training on {len(ds)} samples "
            f"(ASCE={'on' if asce_bundle else 'off'}, epochs={local_fed_cfg['local_epochs']}) ..."
        )

        _, loss, delta_norm, diagnostics = base_asce.train_client_local_base_asce(
            model=model,
            tokenizer=tokenizer,
            global_adapter_dict=initial_lora_state,
            client_ds=ds,
            fed_cfg=local_fed_cfg,
            output_dir=client_dir,
            seed=base_asce.stable_client_round_seed(seed, 0, cid),
            prox_mu=0.0,
            asce_bundle=asce_bundle,
        )

        os.makedirs(adapter_dir, exist_ok=True)
        model.save_pretrained(adapter_dir)
        tokenizer.save_pretrained(adapter_dir)

        steps = base_asce.estimate_gradient_steps(len(ds), local_fed_cfg)
        total_gradient_steps += steps
        row = {
            "client_id": cid,
            "num_train_docs": len(texts),
            "train_loss": loss,
            "epochs": local_fed_cfg["local_epochs"],
            "delta_norm": delta_norm,
            "trainable_params": trainable,
            "seed": seed,
            "gradient_steps_est": steps,
            "asce_alignment_enabled": bool(asce_bundle),
            "asce_style_status": diagnostics.get("asce_style_status", "disabled"),
            "asce_style_loss": diagnostics.get("asce_style_loss", 0.0),
            "asce_style_mean_cosine": diagnostics.get("asce_style_mean_cosine", 0.0),
            "asce_contribution_ratio": diagnostics.get("asce_contribution_ratio", 0.0),
            "mean_task_loss": diagnostics.get("mean_task_loss", 0.0),
            "mean_total_loss": diagnostics.get("mean_total_loss", 0.0),
        }
        summary.append(row)
        with open(os.path.join(client_dir, "training_summary.json"), "w", encoding="utf-8") as f:
            json.dump(row, f, indent=2)
        print(
            f"  Loss: {loss:.4f} | delta_norm: {delta_norm:.6f} | "
            f"asce_status={row['asce_style_status']}"
        )

    wall_time = time.time() - start_time
    with open(os.path.join(method_dir, "training_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    with open(os.path.join(method_dir, "training_budget.csv"), "w", encoding="utf-8") as f:
        f.write("method,seed,total_clients,total_gradient_steps,wall_time_seconds,epochs_per_client\n")
        f.write(
            f"{method_name},{seed},{len(summary)},{total_gradient_steps},"
            f"{wall_time:.1f},{local_fed_cfg['local_epochs']}\n"
        )

    print(f"\n✓ Local-only ASCE training complete for {len(summary)} clients")
    print(f"  Artifacts in {method_dir}")


if __name__ == "__main__":
    main()
