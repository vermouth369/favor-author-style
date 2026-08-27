#!/usr/bin/env python3
"""50_train_pfedme_peft.py — pFedMe-style personalized PEFT baseline.

This baseline keeps a per-client personalized adapter, then derives a
server-uploaded local-meta adapter from that personalized solution. The goal is
to provide a personalized FL comparator whose upload semantics differ from
Ditto's explicit global branch.

Artifact layout intentionally mirrors Ditto so generation and validators can
reuse the same per-client personalized adapter path pattern:

  runs/<method>/
    config_resolved.yaml
    training_budget.csv
    server/round=<t>/global_content_adapter/
    server/round=<t>/round_metrics.json
    clients/client=<cid>/latest/personalized_adapter/
    clients/client=<cid>/latest/personalized_raw.pt
    final/global_content_adapter/
"""

import argparse
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
    TaskType,
    get_peft_model,
    prepare_model_for_kbit_training,
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
    count_trainable_params,
    diff_lora_state,
    estimate_gradient_steps,
    get_lora_state,
    load_client_manifest_with_noniid,
    load_lora_state,
    log_adapter_state,
    lora_delta_norm,
    record_matches_client_id,
)


class PFedMeProximalTrainer(Trainer):
    """Trainer with proximal regularization toward a local meta state."""

    def __init__(self, *args, anchor_state=None, pfedme_lambda=0.01, **kwargs):
        super().__init__(*args, **kwargs)
        self.anchor_state = anchor_state or {}
        self.pfedme_lambda = pfedme_lambda

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        outputs = model(**inputs)
        task_loss = outputs.loss

        prox_loss = torch.tensor(0.0, device=task_loss.device, dtype=task_loss.dtype)
        for name, param in model.named_parameters():
            if "lora_" in name and name in self.anchor_state:
                ref = self.anchor_state[name].to(param.device)
                prox_loss = prox_loss + (param - ref).pow(2).sum()

        total_loss = task_loss + self.pfedme_lambda * prox_loss
        return (total_loss, outputs) if return_outputs else total_loss


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
            examples["text"],
            truncation=True,
            max_length=max_seq_len,
            padding=False,
        )

    ds = ds.map(tokenize_fn, batched=True, remove_columns=["text"])
    return ds, len(texts)


def train_client_personalized_branch(
    model,
    tokenizer,
    client_ds,
    personalized_state,
    anchor_state,
    fed_cfg,
    pfedme_lambda,
    output_dir,
    seed,
):
    """Train the personalized adapter toward a pFedMe proximal objective."""
    os.makedirs(output_dir, exist_ok=True)

    model.set_adapter("default")
    load_lora_state(model, personalized_state)
    log_adapter_state(model, "PFEDME_PERSONAL_START")

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
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
        seed=seed,
        report_to="none",
        remove_unused_columns=False,
        max_grad_norm=1.0,
    )

    trainer = PFedMeProximalTrainer(
        model=model,
        args=training_args,
        train_dataset=client_ds,
        data_collator=data_collator,
        anchor_state=anchor_state,
        pfedme_lambda=pfedme_lambda,
    )
    result = trainer.train()
    loss = result.training_loss

    personalized_ckpt_path = os.path.join(output_dir, "personalized_adapter")
    os.makedirs(personalized_ckpt_path, exist_ok=True)

    model.gradient_checkpointing_disable()
    model.eval()
    model.save_pretrained(personalized_ckpt_path)
    tokenizer.save_pretrained(personalized_ckpt_path)

    safetensors_path = os.path.join(personalized_ckpt_path, "adapter_model.safetensors")
    bin_path = os.path.join(personalized_ckpt_path, "adapter_model.bin")
    if not os.path.exists(safetensors_path) and not os.path.exists(bin_path):
        print(
            "    WARNING: save_pretrained did not produce weight file, using fallback torch.save"
        )
        fallback_state = get_lora_state(model)
        torch.save(fallback_state, os.path.join(personalized_ckpt_path, "adapter_state.pt"))

    raw_state = get_lora_state(model)
    torch.save(raw_state, os.path.join(output_dir, "personalized_raw.pt"))

    log_adapter_state(model, "PFEDME_PERSONAL_END")

    for p in model.parameters():
        p.grad = None
    del trainer
    torch.cuda.empty_cache()

    return personalized_ckpt_path, raw_state, loss


def derive_local_meta_state(global_state, personalized_state, meta_step_size, pfedme_lambda):
    """Project a personalized solution back into the uploaded local-meta state."""
    shrink = meta_step_size * pfedme_lambda
    local_meta = OrderedDict()
    for name, g_param in global_state.items():
        p_param = personalized_state[name]
        local_meta[name] = g_param + shrink * (p_param - g_param)
    return local_meta


def aggregate_deltas_uniform(client_deltas, client_ids, data_sizes):
    """Uniform FedAvg aggregation weighted by client data size."""
    param_names = list(next(iter(client_deltas.values())).keys())
    agg_delta = OrderedDict()
    for name in param_names:
        agg_delta[name] = torch.zeros_like(next(iter(client_deltas.values()))[name])

    total_weight = 0.0
    for cid in client_ids:
        if cid not in client_deltas:
            continue
        delta = client_deltas[cid]
        weight = float(data_sizes.get(cid, 1))
        total_weight += weight
        for name in param_names:
            agg_delta[name] += weight * delta[name]

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
    """Select round participants, optionally prioritizing clients not yet personalized."""
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
    canonical_id = str(identifier)
    digest = hashlib.sha1(canonical_id.encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big") % int(modulo)


def stable_client_round_seed(base_seed, round_id, client_id):
    """Derive a deterministic seed for one client in one federated round."""
    return int(base_seed) + int(round_id) * 1000 + stable_seed_offset(client_id)


def main():
    parser = argparse.ArgumentParser(
        description="pFedMe-style personalized federated training (PEFT baseline)"
    )
    parser.add_argument("--config", type=str, default="exp2/config/phase1/phase1_medium_seed2026_runtime.yaml")
    parser.add_argument("--gpu", type=int, default=None)
    parser.add_argument(
        "--method-name",
        type=str,
        default="pFedMe",
        help="Method directory name",
    )
    parser.add_argument(
        "--pfedme-lambda",
        type=float,
        default=0.01,
        help="Proximal strength toward the local-meta anchor",
    )
    parser.add_argument(
        "--meta-step-size",
        type=float,
        default=1.0,
        help="How far the uploaded local meta state moves toward the personalized state",
    )
    parser.add_argument(
        "--resume-round",
        type=int,
        default=0,
        help="Resume from a specific round (0 = start fresh)",
    )
    args = parser.parse_args()

    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

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

    total_rounds = fed_cfg["rounds"]
    clients_per_round = fed_cfg["clients_per_round"]
    sample_mode = str(fed_cfg.get("sample_mode", "uniform")).lower()
    pfedme_lambda = args.pfedme_lambda
    meta_step_size = args.meta_step_size
    method_name = args.method_name

    method_dir = os.path.join(runs_dir, method_name)
    os.makedirs(method_dir, exist_ok=True)

    print("=" * 60)
    print(f"pFedMe-style Personalized FL Training — {method_name}")
    print(f"  pfedme_lambda: {pfedme_lambda}")
    print(f"  meta_step_size: {meta_step_size}")
    print(f"  Rounds: {total_rounds}")
    print(f"  Clients/round: {clients_per_round}")
    print(f"  Sample mode: {sample_mode}")
    print("=" * 60)

    resolved = {
        "config": cfg,
        "model_config": model_cfg,
        "method_name": method_name,
        "implementation": "pfedme_peft",
        "pfedme_lambda": pfedme_lambda,
        "meta_step_size": meta_step_size,
        "sample_mode": sample_mode,
        "seed": seed,
        "server_upload_semantics": "local_meta_from_personalized_solution",
    }

    manifest_info = load_client_manifest_with_noniid(prototype_dir, cfg)
    all_client_ids = manifest_info["all_client_ids"]
    client_data_sizes = manifest_info["client_data_sizes"]
    noniid_level = manifest_info["non_iid_level"]
    resolved["noniid_manifest_path"] = manifest_info["non_iid_manifest_path"]
    resolved["noniid_level"] = noniid_level
    resolved["noniid_downgrade"] = manifest_info["non_iid_downgrade"]

    with open(os.path.join(method_dir, "config_resolved.yaml"), "w", encoding="utf-8") as f:
        yaml.dump(resolved, f, default_flow_style=False)

    print(f"\n  Total clients: {len(all_client_ids)} (non-IID: {noniid_level})")

    print("\n  Loading base model ...")
    base_model, tokenizer = load_base_model_and_tokenizer(model_cfg)

    lora_config = build_lora_config(peft_cfg)
    model = get_peft_model(base_model, lora_config)
    trainable, total = count_trainable_params(model)
    print(f"  Trainable params: {trainable:,} / {total:,}")

    global_adapter_dict = get_lora_state(model)
    client_personalized_states = {}

    if args.resume_round > 0:
        for r in range(args.resume_round - 1, 0, -1):
            adapter_path = os.path.join(
                method_dir, "server", f"round={r}", "global_content_adapter"
            )
            safetensors_path = os.path.join(adapter_path, "adapter_model.safetensors")
            bin_path = os.path.join(adapter_path, "adapter_model.bin")
            if os.path.exists(safetensors_path) or os.path.exists(bin_path):
                from peft import PeftModel

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

        clients_base = os.path.join(method_dir, "clients")
        if os.path.exists(clients_base):
            for cid in all_client_ids:
                raw_path_latest = os.path.join(
                    clients_base, f"client={cid}", "latest", "personalized_raw.pt"
                )
                if os.path.exists(raw_path_latest):
                    client_personalized_states[cid] = torch.load(
                        raw_path_latest, map_location="cpu", weights_only=True
                    )
        print(f"  ✓ Resumed personalized states for {len(client_personalized_states)} clients")
        print(f"  Resuming from round {args.resume_round}")

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

    for t in range(round_start, total_rounds + 1):
        round_start_time = time.time()
        print(f"\n{'=' * 60}")
        print(f" ROUND {t}/{total_rounds}")
        print(f"{'=' * 60}")

        round_server_dir = os.path.join(method_dir, "server", f"round={t}")
        round_clients_dir = os.path.join(method_dir, "clients")
        os.makedirs(round_server_dir, exist_ok=True)

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

        client_deltas = {}
        client_losses_personal = {}
        client_meta_delta_norms = {}

        for cid in selected_clients:
            print(f"\n  --- Client {cid} ---")
            client_dir = os.path.join(round_clients_dir, f"client={cid}", "latest")
            os.makedirs(client_dir, exist_ok=True)

            client_ds, n_texts = load_client_data(
                pooled_dir, cid, tokenizer, fed_cfg["max_seq_len"]
            )
            if client_ds is None or len(client_ds) == 0:
                print("    WARNING: no training data, skipping")
                continue
            print(f"    Training on {n_texts} samples ...")

            if cid in client_personalized_states:
                personalized_state = client_personalized_states[cid]
            else:
                personalized_state = OrderedDict(
                    (name, param.clone()) for name, param in global_adapter_dict.items()
                )

            _, p_raw_state, loss_personal = train_client_personalized_branch(
                model=model,
                tokenizer=tokenizer,
                client_ds=client_ds,
                personalized_state=personalized_state,
                anchor_state=global_adapter_dict,
                fed_cfg=fed_cfg,
                pfedme_lambda=pfedme_lambda,
                output_dir=client_dir,
                seed=stable_client_round_seed(seed, t, cid),
            )

            local_meta_state = derive_local_meta_state(
                global_adapter_dict,
                p_raw_state,
                meta_step_size=meta_step_size,
                pfedme_lambda=pfedme_lambda,
            )
            client_delta = diff_lora_state(global_adapter_dict, local_meta_state)
            client_delta_norm = lora_delta_norm(global_adapter_dict, local_meta_state)

            torch.save(client_delta, os.path.join(client_dir, "local_upload_delta.pt"))
            client_personalized_states[cid] = p_raw_state
            client_personalized_states_seen.add(cid)
            client_deltas[cid] = client_delta
            client_losses_personal[cid] = loss_personal
            client_meta_delta_norms[cid] = client_delta_norm
            total_gradient_steps += estimate_gradient_steps(len(client_ds), fed_cfg)

            client_metrics = {
                "client_id": cid,
                "round": t,
                "train_samples": n_texts,
                "personal_loss": loss_personal,
                "pfedme_lambda": pfedme_lambda,
                "meta_step_size": meta_step_size,
                "local_meta_delta_norm": client_delta_norm,
            }
            with open(os.path.join(client_dir, "client_metrics.json"), "w", encoding="utf-8") as f:
                json.dump(client_metrics, f, indent=2)

            print(f"    Personal loss: {loss_personal:.4f}")
            print(f"    Local-meta delta norm: {client_delta_norm:.6f}")

            try:
                torch._dynamo.reset()
            except Exception:
                pass

        if not client_deltas:
            print("  WARNING: no client deltas produced, skipping round")
            continue

        print(f"\n  Aggregating {len(client_deltas)} client deltas ...")
        active_client_ids = [cid for cid in selected_clients if cid in client_deltas]
        agg_delta = aggregate_deltas_uniform(
            client_deltas, active_client_ids, client_data_sizes
        )

        for name in global_adapter_dict:
            if name in agg_delta:
                global_adapter_dict[name] += agg_delta[name]

        adapter_save_dir = os.path.join(round_server_dir, "global_content_adapter")
        os.makedirs(adapter_save_dir, exist_ok=True)
        load_lora_state(model, global_adapter_dict)
        model.save_pretrained(adapter_save_dir)
        tokenizer.save_pretrained(adapter_save_dir)
        torch.cuda.empty_cache()

        round_time = time.time() - round_start_time
        wall_time_total += round_time
        avg_personal_loss = np.mean(list(client_losses_personal.values()))
        avg_meta_delta_norm = np.mean(list(client_meta_delta_norms.values()))

        round_metrics = {
            "round": t,
            "method": method_name,
            "implementation": "pfedme_peft",
            "pfedme_lambda": pfedme_lambda,
            "meta_step_size": meta_step_size,
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
            "avg_personal_loss": float(avg_personal_loss),
            "avg_train_loss": float(avg_personal_loss),
            "avg_local_meta_delta_norm": float(avg_meta_delta_norm),
            "trainable_params": trainable,
            "round_time_seconds": round_time,
        }
        with open(os.path.join(round_server_dir, "round_metrics.json"), "w", encoding="utf-8") as f:
            json.dump(round_metrics, f, indent=2)

        history.append(round_metrics)

        print(f"\n  Round {t} complete:")
        print(f"    Avg personal loss: {avg_personal_loss:.4f}")
        print(f"    Avg local-meta delta norm: {avg_meta_delta_norm:.6f}")
        print(f"    Time: {round_time:.1f}s")

    final_dir = os.path.join(method_dir, "final")
    os.makedirs(final_dir, exist_ok=True)

    with open(os.path.join(final_dir, "training_history.json"), "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    final_adapter_dir = os.path.join(final_dir, "global_content_adapter")
    os.makedirs(final_adapter_dir, exist_ok=True)
    load_lora_state(model, global_adapter_dict)
    model.save_pretrained(final_adapter_dir)
    tokenizer.save_pretrained(final_adapter_dir)

    import csv

    budget_path = os.path.join(method_dir, "training_budget.csv")
    with open(budget_path, "w", newline="", encoding="utf-8") as csvf:
        writer = csv.writer(csvf)
        writer.writerow([
            "method", "seed", "non_iid_level", "non_iid_manifest_path",
            "total_rounds", "clients_per_round", "sample_mode", "local_epochs",
            "total_gradient_steps", "wall_time_seconds",
            "pfedme_lambda", "meta_step_size",
            "peft_rank_shared", "peft_alpha_shared",
            "peft_rank_private", "peft_alpha_private",
            "trainable_params_shared", "trainable_params_private",
            "peft_target_modules",
        ])
        writer.writerow([
            method_name,
            seed,
            noniid_level,
            manifest_info["non_iid_manifest_path"] or "",
            total_rounds,
            clients_per_round,
            sample_mode,
            fed_cfg["local_epochs"],
            total_gradient_steps,
            round(wall_time_total, 1),
            pfedme_lambda,
            meta_step_size,
            peft_cfg["r"],
            peft_cfg.get("alpha", 32),
            peft_cfg["r"],
            peft_cfg.get("alpha", 32),
            trainable,
            trainable,
            "|".join(peft_cfg.get("target_modules", [])),
        ])
    print(f"  ✓ training_budget.csv saved to {budget_path}")

    print(f"\n{'=' * 60}")
    print(f"✓ pFedMe-style training complete ({total_rounds} rounds)")
    print(f"  Method: {method_name}")
    print(f"  Artifacts: {method_dir}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
