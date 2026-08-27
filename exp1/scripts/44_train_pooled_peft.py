#!/usr/bin/env python3
"""44_train_pooled_peft.py - centralized pooled PEFT baseline.

This baseline trains one shared LoRA adapter on the pooled train split for the
active Phase 1 client roster. It does not do federation, private residual
packs, or ASCE. The output layout matches shared-adapter methods so generation
can reuse the standard path.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import sys
import time
from pathlib import Path

sys.setrecursionlimit(10000)

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
    load_client_manifest_with_noniid,
    record_matches_client_id,
)


def load_base_model_and_tokenizer(model_cfg):
    """Load the base model/tokenizer using the repo's QLoRA conventions."""
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
    if hasattr(model, "config"):
        model.config.use_cache = False
    return model, tokenizer


def build_lora_config(peft_cfg):
    return LoraConfig(
        r=peft_cfg["r"],
        lora_alpha=peft_cfg["alpha"],
        lora_dropout=peft_cfg["dropout"],
        target_modules=peft_cfg["target_modules"],
        bias=peft_cfg.get("bias", "none"),
        task_type=TaskType.CAUSAL_LM,
    )


def load_pooled_texts(pooled_dir, client_ids):
    """Load all pooled training texts belonging to the active client roster."""
    train_path = os.path.join(pooled_dir, "train.jsonl")
    allowed = set(client_ids)
    texts = []
    per_client_counts = {cid: 0 for cid in client_ids}

    with open(train_path, "r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            rec = json.loads(line)
            matched_client = None
            for cid in allowed:
                if record_matches_client_id(rec, cid):
                    matched_client = cid
                    break
            if matched_client is None:
                continue
            text = rec.get("text")
            if text:
                texts.append(text)
                per_client_counts[matched_client] += 1

    return texts, per_client_counts


def compute_default_epochs(fed_cfg, n_clients):
    """Match the average client exposure of the federated schedule."""
    rounds = float(fed_cfg["rounds"])
    clients_per_round = float(fed_cfg["clients_per_round"])
    local_epochs = float(fed_cfg.get("local_epochs", 1))
    return rounds * clients_per_round * local_epochs / max(float(n_clients), 1.0)


def copy_adapter_alias(final_adapter_dir, alias_dir):
    if os.path.exists(alias_dir):
        shutil.rmtree(alias_dir)
    shutil.copytree(final_adapter_dir, alias_dir)


def main():
    parser = argparse.ArgumentParser(
        description="Centralized pooled PEFT baseline training."
    )
    parser.add_argument("--config", type=str, default="exp2/config/phase1/phase1_medium_seed2026_runtime.yaml")
    parser.add_argument("--gpu", type=int, default=None)
    parser.add_argument("--method-name", type=str, default="Pooled PEFT")
    parser.add_argument(
        "--num-train-epochs",
        type=float,
        default=None,
        help=(
            "Override centralized pooled epochs. Default compute-matches the "
            "federated schedule: rounds * clients_per_round * local_epochs / n_clients."
        ),
    )
    args = parser.parse_args()

    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    with open(args.config, "r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    with open(cfg["paths"]["base_model_config"], "r", encoding="utf-8") as handle:
        model_cfg = yaml.safe_load(handle)

    prototype_dir = cfg["paths"]["prototype_dir"]
    pooled_dir = cfg["paths"]["pooled_dir"]
    runs_dir = cfg["paths"]["runs_dir"]
    fed_cfg = cfg["fed"]
    peft_cfg = cfg["peft"]
    seed = int(cfg["seeds"]["training"])
    method_name = args.method_name
    method_dir = os.path.join(runs_dir, method_name)
    os.makedirs(method_dir, exist_ok=True)

    manifest_info = load_client_manifest_with_noniid(prototype_dir, cfg)
    client_ids = manifest_info["all_client_ids"]
    noniid_level = manifest_info["non_iid_level"]
    num_train_epochs = (
        float(args.num_train_epochs)
        if args.num_train_epochs is not None
        else compute_default_epochs(fed_cfg, len(client_ids))
    )

    resolved = {
        "config": cfg,
        "model_config": model_cfg,
        "method_name": method_name,
        "implementation": "centralized_pooled_peft",
        "pooled_dir": pooled_dir,
        "n_clients": len(client_ids),
        "non_iid_level": noniid_level,
        "non_iid_manifest_path": manifest_info.get("non_iid_manifest_path"),
        "non_iid_downgrade": manifest_info.get("non_iid_downgrade", False),
        "num_train_epochs": num_train_epochs,
        "epoch_policy": (
            "override"
            if args.num_train_epochs is not None
            else "fed_exposure_matched"
        ),
    }
    with open(os.path.join(method_dir, "config_resolved.yaml"), "w", encoding="utf-8") as handle:
        yaml.dump(resolved, handle, default_flow_style=False, sort_keys=False)

    print("=" * 60)
    print(f"Centralized Pooled PEFT Training - {method_name}")
    print(f"  Non-IID: {noniid_level}")
    print(f"  Clients: {len(client_ids)}")
    print(f"  Pooled dir: {pooled_dir}")
    print(f"  Epochs: {num_train_epochs:.4f}")
    print(f"  LR: {fed_cfg['local_lr']}")
    print("=" * 60)

    print("\n  Loading pooled train texts ...")
    texts, per_client_counts = load_pooled_texts(pooled_dir, client_ids)
    if not texts:
        raise RuntimeError(f"No pooled training texts found in {pooled_dir}")
    print(f"  Training samples: {len(texts)}")
    missing = [cid for cid, count in per_client_counts.items() if count == 0]
    if missing:
        print(f"  WARNING: {len(missing)} clients have zero pooled train texts")

    print("\n  Loading base model ...")
    base_model, tokenizer = load_base_model_and_tokenizer(model_cfg)
    model = get_peft_model(base_model, build_lora_config(peft_cfg))
    trainable, total = count_trainable_params(model)
    print(f"  Trainable params: {trainable:,} / {total:,}")

    dataset = Dataset.from_dict({"text": texts})

    def tokenize_fn(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=fed_cfg["max_seq_len"],
            padding=False,
        )

    dataset = dataset.map(tokenize_fn, batched=True, remove_columns=["text"])
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    training_args = TrainingArguments(
        output_dir=os.path.join(method_dir, "_train_tmp"),
        overwrite_output_dir=True,
        per_device_train_batch_size=fed_cfg["micro_batch_size"],
        gradient_accumulation_steps=fed_cfg["gradient_accumulation_steps"],
        learning_rate=fed_cfg["local_lr"],
        warmup_ratio=fed_cfg["warmup_ratio"],
        num_train_epochs=num_train_epochs,
        optim="adamw_torch",
        bf16=True,
        gradient_checkpointing=True,
        logging_steps=5,
        save_strategy="no",
        seed=seed,
        data_seed=seed,
        report_to="none",
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=data_collator,
    )

    started = time.time()
    result = trainer.train()
    wall_time = time.time() - started
    total_gradient_steps = int(trainer.state.global_step)

    final_adapter_dir = os.path.join(method_dir, "final", "global_content_adapter")
    os.makedirs(final_adapter_dir, exist_ok=True)
    model.save_pretrained(final_adapter_dir)
    tokenizer.save_pretrained(final_adapter_dir)
    alias_dir = os.path.join(method_dir, "adapter")
    copy_adapter_alias(final_adapter_dir, alias_dir)

    summary = {
        "method": method_name,
        "implementation": "centralized_pooled_peft",
        "n_clients": len(client_ids),
        "n_train_texts": len(texts),
        "num_train_epochs": round(num_train_epochs, 6),
        "learning_rate": fed_cfg["local_lr"],
        "total_gradient_steps": total_gradient_steps,
        "train_loss": float(result.training_loss),
        "wall_time_seconds": round(wall_time, 1),
        "final_adapter_dir": os.path.relpath(final_adapter_dir, method_dir),
        "adapter_alias_dir": os.path.relpath(alias_dir, method_dir),
    }
    with open(os.path.join(method_dir, "training_summary.json"), "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    budget_path = os.path.join(method_dir, "training_budget.csv")
    with open(budget_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "method",
                "n_clients",
                "n_train_texts",
                "num_train_epochs",
                "learning_rate",
                "total_gradient_steps",
                "wall_time_seconds",
            ]
        )
        writer.writerow(
            [
                method_name,
                len(client_ids),
                len(texts),
                round(num_train_epochs, 6),
                fed_cfg["local_lr"],
                total_gradient_steps,
                round(wall_time, 1),
            ]
        )

    print(f"\n✓ {method_name} training complete")
    print(f"  Final adapter: {final_adapter_dir}")
    print(f"  Adapter alias: {alias_dir}")
    print(f"  Training summary: {os.path.join(method_dir, 'training_summary.json')}")


if __name__ == "__main__":
    main()
