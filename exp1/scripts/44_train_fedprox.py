#!/usr/bin/env python3
"""44_train_fedprox.py — FedProx baseline for Phase B.

FedProx = FedAvg + proximal regularization on the shared adapter.
Each client minimizes: L_task + (mu/2) * ||A_local - A_global||^2

Uses the shared-adapter federated path with a proximal L2 penalty.

Usage:
  python 44_train_fedprox.py --config exp2/config/phase1/phase1_medium_seed2026_runtime.yaml \\
      --method-name FedProx --prox-mu 0.001
"""

import argparse
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
from sentence_transformers import SentenceTransformer
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
    load_client_manifest_with_noniid,
    record_matches_client_id,
)


# ============================================================
# ProxTrainer — Trainer with proximal penalty (FedProx)
# ============================================================

class ProxTrainer(Trainer):
    """Custom Trainer that adds a proximal L2 penalty toward the global adapter.

    Loss = L_task + (mu / 2) * sum_p ||p_local - p_global||^2
    """

    def __init__(self, *args, server_state=None, prox_mu=0.01, **kwargs):
        super().__init__(*args, **kwargs)
        self.server_state = server_state or {}
        self.prox_mu = prox_mu
        self._prox_loss_accum = 0.0
        self._prox_loss_count = 0

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        outputs = model(**inputs)
        task_loss = outputs.loss

        # Proximal penalty
        prox_loss = torch.tensor(0.0, device=task_loss.device, dtype=task_loss.dtype)
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.server_state:
                ref = self.server_state[name].to(param.device)
                prox_loss = prox_loss + torch.sum((param - ref) ** 2)
        prox_loss = (self.prox_mu / 2.0) * prox_loss

        total_loss = task_loss + prox_loss

        # Track proximal loss
        self._prox_loss_accum += prox_loss.item()
        self._prox_loss_count += 1

        if return_outputs:
            return total_loss, outputs
        return total_loss

    @property
    def mean_prox_loss(self):
        if self._prox_loss_count == 0:
            return 0.0
        return self._prox_loss_accum / self._prox_loss_count


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
        return None
    ds = Dataset.from_dict({"text": texts})
    def tokenize_fn(examples):
        return tokenizer(
            examples["text"], truncation=True,
            max_length=max_seq_len, padding=False,
        )
    ds = ds.map(tokenize_fn, batched=True, remove_columns=["text"])
    return ds


# ============================================================
# Client-side training with proximal penalty
# ============================================================

def train_client_local_prox(
    model, tokenizer, global_adapter_dict, client_ds,
    fed_cfg, output_dir, seed, prox_mu=0.01,
):
    """Train one client locally with FedProx proximal penalty.

    Returns:
        content_delta: OrderedDict of Δ for the shared adapter
        train_loss: average training loss
        delta_norm: L2 norm of the delta
        mean_prox_loss: average proximal loss component
    """
    os.makedirs(output_dir, exist_ok=True)

    model.set_adapter("default")
    load_lora_state(model, global_adapter_dict)
    log_adapter_state(model, "CLIENT_TRAIN_START")

    before_state = get_lora_state(model)

    # Build server reference state for proximal penalty
    server_state = {}
    for name, param in model.named_parameters():
        if param.requires_grad:
            server_state[name] = param.detach().clone()

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=False
    )
    training_args = TrainingArguments(
        output_dir=os.path.join(output_dir, "_train_tmp"),
        overwrite_output_dir=True,
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

    trainer = ProxTrainer(
        model=model,
        args=training_args,
        train_dataset=client_ds,
        data_collator=data_collator,
        server_state=server_state,
        prox_mu=prox_mu,
    )
    result = trainer.train()
    train_loss = result.training_loss
    mean_prox = trainer.mean_prox_loss

    after_state = get_lora_state(model)
    content_delta = diff_lora_state(before_state, after_state)
    d_norm = lora_delta_norm(before_state, after_state)

    log_adapter_state(model, "CLIENT_TRAIN_END")
    print(f"    delta_norm: {d_norm:.6f}, prox_loss: {mean_prox:.6f} (mu={prox_mu})")

    for p in model.parameters():
        p.grad = None
    del trainer
    del server_state
    torch.cuda.empty_cache()

    return content_delta, train_loss, d_norm, mean_prox


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
        description="FedProx baseline training"
    )
    parser.add_argument("--config", type=str, default="exp2/config/phase1/phase1_medium_seed2026_runtime.yaml")
    parser.add_argument("--gpu", type=int, default=None)
    parser.add_argument(
        "--method-name", type=str, default="FedProx",
        help="Method directory name (default: FedProx)"
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

    # Proximal strength: CLI > config > default
    prox_mu = args.prox_mu
    if prox_mu is None:
        prox_mu = cfg.get("losses", {}).get("prox_shared", {}).get("weight", 0.01)
    method_name = args.method_name

    method_dir = os.path.join(runs_dir, method_name)
    os.makedirs(method_dir, exist_ok=True)

    trainable_count = None

    print("=" * 60)
    print(f"FedProx Training — {method_name}")
    print(f"  Proximal mu: {prox_mu}")
    print(f"  Rounds: {total_rounds}")
    print(f"  Clients/round: {clients_per_round}")
    print("=" * 60)

    # ---- Save resolved config ----
    resolved = {
        "config": cfg,
        "model_config": model_cfg,
        "method_name": method_name,
        "prox_mu": prox_mu,
        "implementation": "fedprox_shared_only",
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
    resolved["noniid_downgrade"] = manifest_info.get("non_iid_downgrade", False)
    with open(os.path.join(method_dir, "config_resolved.yaml"), "w") as f:
        yaml.dump(resolved, f, default_flow_style=False)
    print(f"\n  Total clients: {len(all_client_ids)} (non-IID: {noniid_level})")

    # ---- Load base model ----
    print("\n  Loading base model ...")
    base_model, tokenizer = load_base_model_and_tokenizer(model_cfg)

    # ---- Build PEFT config and model ----
    lora_config = build_lora_config(peft_cfg)
    model = get_peft_model(base_model, lora_config)
    global_adapter_dict = get_lora_state(model)

    trainable_count, total_count = count_trainable_params(model)
    print(f"  Trainable params: {trainable_count:,} / {total_count:,}")

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
    # FEDERATED TRAINING LOOP (FedProx)
    # ===========================================================
    for t in range(round_start, total_rounds + 1):
        round_start_time = time.time()
        print(f"\n{'='*60}")
        print(f" ROUND {t}/{total_rounds} (FedProx, mu={prox_mu})")
        print(f"{'='*60}")

        round_dir = os.path.join(method_dir, f"round={t}")
        server_dir = os.path.join(round_dir, "server")
        clients_dir = os.path.join(round_dir, "clients")

        # ---- Step 1: Select clients ----
        rng = np.random.RandomState(seed + t)
        if clients_per_round >= len(all_client_ids):
            selected_clients = all_client_ids
        else:
            selected_clients = list(
                rng.choice(all_client_ids, clients_per_round, replace=False)
            )
        print(f"\n  Selected {len(selected_clients)} clients")

        # ---- Step 2: Client-side local training (FedProx) ----
        client_deltas = {}
        client_losses = {}
        client_delta_norms = {}
        client_prox_losses = {}

        for cid in selected_clients:
            print(f"\n  --- Client {cid} ---")
            client_dir = os.path.join(clients_dir, f"client={cid}")
            os.makedirs(client_dir, exist_ok=True)

            client_ds = load_client_data(
                pooled_dir, cid, tokenizer, fed_cfg["max_seq_len"]
            )
            if client_ds is None or len(client_ds) == 0:
                print(f"    WARNING: no training data for {cid}, skipping")
                continue

            print(f"    Training on {len(client_ds)} samples (FedProx, mu={prox_mu}) ...")

            delta, loss, d_norm, prox_loss = train_client_local_prox(
                model=model,
                tokenizer=tokenizer,
                global_adapter_dict=global_adapter_dict,
                client_ds=client_ds,
                fed_cfg=fed_cfg,
                output_dir=client_dir,
                seed=seed + t * 1000 + hash(cid) % 1000,
                prox_mu=prox_mu,
            )

            client_deltas[cid] = delta
            client_losses[cid] = loss
            client_delta_norms[cid] = d_norm
            client_prox_losses[cid] = prox_loss
            print(f"    Loss: {loss:.4f}")

            # Estimate gradient steps for this client
            n_samples = len(client_ds)
            bs = fed_cfg["micro_batch_size"]
            ga = fed_cfg["gradient_accumulation_steps"]
            epochs = fed_cfg["local_epochs"]
            steps = max(1, n_samples // (bs * ga)) * epochs
            total_gradient_steps += steps

            upload_stats = {
                "client_id": cid,
                "round": t,
                "method": method_name,
                "train_samples": len(client_ds),
                "train_loss": loss,
                "delta_norm": d_norm,
                "prox_mu": prox_mu,
                "mean_prox_loss": prox_loss,
                "delta_params": len(delta),
                "active_adapter": "default",
                "trainable_params": trainable_count,
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
        avg_prox_loss = np.mean(list(client_prox_losses.values()))

        round_metrics = {
            "round": t,
            "method": method_name,
            "implementation": "fedprox_shared_only",
            "prox_mu": prox_mu,
            "selected_clients": len(selected_clients),
            "participating_clients": len(client_deltas),
            "avg_train_loss": float(avg_loss),
            "avg_delta_norm": float(avg_delta_norm),
            "avg_prox_loss": float(avg_prox_loss),
            "active_adapter": "default",
            "trainable_params": trainable_count,
            "round_time_seconds": round_time,
        }
        os.makedirs(server_dir, exist_ok=True)
        with open(os.path.join(server_dir, "round_metrics.json"), "w") as f:
            json.dump(round_metrics, f, indent=2)

        history.append(round_metrics)

        print(f"\n  Round {t} complete:")
        print(f"    Avg loss: {avg_loss:.4f}")
        print(f"    Avg prox_loss: {avg_prox_loss:.6f}")
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

    # ---- Emit training_budget.csv (Task 3 artifact contract) ----
    import csv
    budget_path = os.path.join(method_dir, "training_budget.csv")
    with open(budget_path, "w", newline="") as csvf:
        writer = csv.writer(csvf)
        writer.writerow([
            "method", "total_rounds", "clients_per_round", "local_epochs",
            "total_gradient_steps", "wall_time_seconds", "prox_mu",
        ])
        writer.writerow([
            method_name, total_rounds, clients_per_round,
            fed_cfg["local_epochs"], total_gradient_steps,
            round(wall_time_total, 1), prox_mu,
        ])
    print(f"  ✓ training_budget.csv saved to {budget_path}")

    print(f"\n{'='*60}")
    print(f"✓ FedProx training complete ({total_rounds} rounds, mu={prox_mu})")
    print(f"  Method: {method_name}")
    print(f"  Artifacts: {method_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
