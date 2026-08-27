#!/usr/bin/env python3
"""46b_generate_exp2_continuation.py — Exp2 held-out continuation generation.

Dedicated generation entrypoint for the held-out continuation protocol.

Key properties:
  - No style seeds, no prompt shells, no instruction wrappers
  - Conditioning is raw held-out author prefix text
  - Batch structure: client → owned continuation_items (not client → seed → prompt)
  - Records exact conditioning_text and reference_continuation_text
  - Enforces owner_client_id == client_id for every row
  - Emits Phase A protocol/coverage/roster reports

Output: runs/exp2_phaseB/{method}/generations.jsonl
"""

import argparse
import gc
import json
import os
import re
from collections import defaultdict

import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from favor_helpers import (
    materialize_private_runtime_adapter,
    canonical_author_id,
    resolve_run_client_roster,
)


# ============================================================
# Constants
# ============================================================

PROTOCOL_VERSION = "heldout_continuation_v1"
AUTHOR_CONDITIONING_MODE = "model_only"


# ============================================================
# Continuation roster loading
# ============================================================

def load_continuation_roster(roster_path):
    """Load and validate the continuation roster JSON."""
    with open(roster_path, "r", encoding="utf-8") as f:
        roster = json.load(f)

    assert roster.get("protocol_version") == PROTOCOL_VERSION, (
        f"Roster protocol mismatch: expected {PROTOCOL_VERSION}, "
        f"got {roster.get('protocol_version')}"
    )

    items = roster["items"]
    # Verify global uniqueness
    ids = [item["continuation_id"] for item in items]
    assert len(set(ids)) == len(ids), "continuation_id collision in roster"

    return roster


def group_roster_by_client(roster):
    """Group roster items by owner_client_id for per-client generation."""
    client_items = defaultdict(list)
    for item in roster["items"]:
        client_items[item["owner_client_id"]].append(item)
    return dict(client_items)


# ============================================================
# Adapter resolution
# ============================================================

def is_favor_method(method_name):
    """Return True for FAVoR shared-private residual runs."""
    return method_name == "FAVoR"


def read_adapter_signature(adapter_dir):
    """Return a small signature describing the adapter architecture."""
    config_path = os.path.join(adapter_dir, "adapter_config.json")
    if not os.path.exists(config_path):
        return None
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    return (
        cfg.get("r"),
        cfg.get("lora_alpha"),
        cfg.get("lora_dropout"),
        tuple(cfg.get("target_modules", [])),
    )


def materialize_favor_runtime_adapter(private_residual_pack_path, output_dir):
    """Materialize the canonical FAVoR adapter from shared + residual."""
    runtime_dir, artifacts = materialize_private_runtime_adapter(
        private_residual_pack_path,
        output_dir,
        require_residual=True,
    )
    print(
        "  [FAVoR runtime] materialized residual-native adapter: "
        f"{artifacts['full_dir']} -> {runtime_dir}"
    )
    return runtime_dir


# ============================================================
# Quality filters (minimal, adapted for continuation outputs)
# ============================================================

def distinct_bigram_ratio(text):
    """Compute distinct-2-gram ratio."""
    words = text.lower().split()
    if len(words) < 2:
        return 0.0
    bigrams = [(words[i], words[i + 1]) for i in range(len(words) - 1)]
    return len(set(bigrams)) / len(bigrams) if bigrams else 0.0


# ============================================================
# Transformers generation backend
# ============================================================

def build_bnb_config(model_cfg):
    """Build optional 4-bit quantization config."""
    quant_cfg = model_cfg.get("quantization", {})
    if not quant_cfg.get("load_in_4bit", False):
        return None
    compute_dtype = getattr(
        torch,
        quant_cfg.get("bnb_4bit_compute_dtype", "bfloat16"),
    )
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type=quant_cfg.get("bnb_4bit_quant_type", "nf4"),
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=quant_cfg.get("bnb_4bit_use_double_quant", True),
    )


def load_transformers_stack(model_cfg):
    """Load a transformers generation stack."""
    model_name = model_cfg["model"]["name"]
    bnb_config = build_bnb_config(model_cfg)
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
    tokenizer.padding_side = "left"
    model.eval()
    return model, tokenizer


def generate_batch_transformers(model, tokenizer, prompts, gen_cfg):
    """Generate a batch with transformers/PEFT and return texts plus token counts."""
    max_new_tokens = gen_cfg.get("max_new_tokens", 220)
    max_prompt_tokens = max(64, 1280 - max_new_tokens)
    batch_size = 8
    all_texts = []
    all_token_counts = []

    generate_kwargs = {
        "max_new_tokens": max_new_tokens,
        "repetition_penalty": gen_cfg.get("repetition_penalty", 1.1),
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    temperature = gen_cfg.get("temperature", 0.0)
    if temperature and temperature > 0:
        generate_kwargs.update({
            "do_sample": True,
            "temperature": temperature,
            "top_p": gen_cfg.get("top_p", 1.0),
        })
    else:
        generate_kwargs["do_sample"] = False

    device = next(model.parameters()).device

    for i in range(0, len(prompts), batch_size):
        chunk_prompts = prompts[i:i + batch_size]
        inputs = tokenizer(
            chunk_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_prompt_tokens,
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.inference_mode():
            outputs = model.generate(**inputs, **generate_kwargs)

        prompt_width = inputs["input_ids"].shape[1]
        generated_ids = outputs[:, prompt_width:]
        texts = [
            tokenizer.decode(row, skip_special_tokens=True).strip()
            for row in generated_ids
        ]
        token_counts = [int(row.numel()) for row in generated_ids]

        all_texts.extend(texts)
        all_token_counts.extend(token_counts)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return all_texts, all_token_counts


# ============================================================
# Core generation logic
# ============================================================

def generate_continuation_for_client(
    model, tokenizer, items, gen_cfg, method_name, client_id,
):
    """Generate continuations for a list of roster items using one client model.

    Args:
        model: The loaded (possibly adapter-augmented) model.
        tokenizer: Tokenizer.
        items: List of roster items owned by this client.
        gen_cfg: Generation config dict.
        method_name: Method label.
        client_id: Client ID used for generation.

    Returns:
        List of generation records.
    """
    # Conditioning is just the raw prefix text — no shells, no seeds
    prompts = [item["prefix_text"] for item in items]

    texts, token_counts = generate_batch_transformers(
        model, tokenizer, prompts, gen_cfg,
    )

    records = []
    for item, output_text, num_tokens in zip(items, texts, token_counts):
        record = {
            "method": method_name,
            "client_id": client_id,
            "author_id": item["owner_author_id"],
            "continuation_id": item["continuation_id"],
            "owner_client_id": item["owner_client_id"],
            "owner_author_id": item["owner_author_id"],
            "source_split": item["source_split"],
            "source_doc_id": item["source_doc_id"],
            "prefix_token_count": item["prefix_token_count"],
            "reference_token_count": item["reference_token_count"],
            "conditioning_text": item["prefix_text"],
            "reference_continuation_text": item["reference_continuation_text"],
            "prompt_protocol": PROTOCOL_VERSION,
            "author_conditioning_mode": AUTHOR_CONDITIONING_MODE,
            "style_seed_used": False,
            "seed_id": None,
            "output_text": output_text,
            "num_output_tokens": num_tokens,
        }
        records.append(record)
    return records


def generate_for_method_continuation(
    method_name, cfg, model_cfg, roster, client_items,
    adapter_path=None, per_client_adapters=None,
    backend="transformers",
):
    """Generate held-out continuations for one method.

    Args:
        method_name: e.g. "B0_base", "FAVoR"
        cfg: Full Exp2 config.
        model_cfg: Base model config.
        roster: Full continuation roster dict.
        client_items: dict client_id -> list of roster items.
        adapter_path: Shared adapter path.
        per_client_adapters: dict client_id -> adapter_path for personalized methods.
        backend: Generation backend (only "transformers" supported).
    """
    from peft import PeftModel

    gen_cfg = cfg["generation"]
    runs_dir = cfg["paths"]["runs_dir"]
    method_dir = os.path.join(runs_dir, method_name)
    os.makedirs(method_dir, exist_ok=True)

    all_generations = []
    all_client_ids = sorted(client_items.keys())

    print(f"\n  [{method_name}] Generating continuations for {len(all_client_ids)} clients ...")
    print(f"  [{method_name}] backend={backend}, protocol={PROTOCOL_VERSION}")

    base_model, tokenizer = load_transformers_stack(model_cfg)

    if per_client_adapters:
        # Per-client personalized generation
        active_adapter_name = "active_runtime"
        active_adapter_dir = None
        active_adapter_signature = None
        active_model = None

        for idx, cid in enumerate(all_client_ids, start=1):
            items = client_items.get(cid, [])
            if not items:
                continue

            adapter_dir = per_client_adapters.get(cid)
            if adapter_dir is None or not os.path.exists(adapter_dir):
                print(f"  WARNING: Adapter not found for {cid} at {adapter_dir}, skipping")
                continue

            print(
                f"  [{method_name}|transformers] client {idx}/{len(all_client_ids)} "
                f"cid={cid} items={len(items)}"
            )

            adapter_signature = read_adapter_signature(adapter_dir)

            if active_model is None or adapter_signature != active_adapter_signature:
                if active_model is not None and active_model is not base_model:
                    del active_model
                    del base_model
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    base_model, _ = load_transformers_stack(model_cfg)
                active_model = PeftModel.from_pretrained(
                    base_model,
                    adapter_dir,
                    adapter_name=active_adapter_name,
                    is_trainable=False,
                )
                active_model.eval()
                active_model.set_adapter(active_adapter_name)
                active_adapter_dir = adapter_dir
                active_adapter_signature = adapter_signature
            elif adapter_dir != active_adapter_dir:
                if active_adapter_name in active_model.peft_config:
                    active_model.delete_adapter(active_adapter_name)
                active_model.load_adapter(
                    adapter_dir,
                    adapter_name=active_adapter_name,
                    is_trainable=False,
                )
                active_model.set_adapter(active_adapter_name)
                active_adapter_dir = adapter_dir
                active_adapter_signature = adapter_signature
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            records = generate_continuation_for_client(
                active_model, tokenizer, items, gen_cfg, method_name, cid,
            )
            all_generations.extend(records)
            print(
                f"    [{method_name}] done client {cid}: "
                f"outputs={len(records)} cumulative={len(all_generations)}"
            )

        if active_model is not None and active_model is not base_model:
            del active_model
    elif adapter_path:
        # Shared adapter methods.
        active_model = PeftModel.from_pretrained(
            base_model, adapter_path,
            adapter_name="active_runtime",
            is_trainable=False,
        )
        active_model.eval()

        for idx, cid in enumerate(all_client_ids, start=1):
            items = client_items.get(cid, [])
            if not items:
                continue
            print(
                f"  [{method_name}|transformers] client {idx}/{len(all_client_ids)} "
                f"cid={cid} items={len(items)}"
            )
            records = generate_continuation_for_client(
                active_model, tokenizer, items, gen_cfg, method_name, cid,
            )
            all_generations.extend(records)

        del active_model
    else:
        # Base model (B0)
        for idx, cid in enumerate(all_client_ids, start=1):
            items = client_items.get(cid, [])
            if not items:
                continue
            print(
                f"  [{method_name}|transformers] client {idx}/{len(all_client_ids)} "
                f"cid={cid} items={len(items)}"
            )
            records = generate_continuation_for_client(
                base_model, tokenizer, items, gen_cfg, method_name, cid,
            )
            all_generations.extend(records)

    del base_model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Write outputs
    gen_path = os.path.join(method_dir, "generations.jsonl")
    with open(gen_path, "w", encoding="utf-8") as f:
        for rec in all_generations:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"  → {len(all_generations)} generations written to {gen_path}")

    # Quality filter
    qf_cfg = cfg.get("quality_filters", {})
    if qf_cfg.get("enabled", False):
        filtered = []
        stats = {"total": len(all_generations), "short": 0, "repetitive": 0, "passed": 0}
        min_tokens = qf_cfg.get("min_output_tokens", 20)
        min_bigram = qf_cfg.get("min_distinct_bigram_ratio", 0.15)

        for rec in all_generations:
            text = rec["output_text"]
            words = text.split()
            if len(words) < min_tokens:
                stats["short"] += 1
                continue
            if distinct_bigram_ratio(text) < min_bigram:
                stats["repetitive"] += 1
                continue
            filtered.append(rec)
            stats["passed"] += 1

        filtered_path = os.path.join(method_dir, "generations_filtered.jsonl")
        with open(filtered_path, "w", encoding="utf-8") as f:
            for rec in filtered:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

        report_path = os.path.join(method_dir, "quality_filter_report.json")
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(stats, f, indent=2)
        print(f"  Quality filter: {stats}")

    return gen_path


# ============================================================
# Protocol reports
# ============================================================

def write_protocol_report(runs_dir, roster, all_methods, method_row_counts):
    """Write the generation protocol report required by Phase A §3.6.G."""
    report = {
        "protocol_version": PROTOCOL_VERSION,
        "seed_usage_disabled": True,
        "prompt_shell_wrappers_disabled": True,
        "continuation_id_globally_unique": True,
        "author_conditioning_mode": AUTHOR_CONDITIONING_MODE,
        "continuation_roster_total_items": roster["total_items"],
        "continuations_per_client": roster["continuations_per_client"],
        "balanced_roster": roster["balanced"],
        "n_active_clients": roster["n_active_clients"],
        "methods": {},
    }

    for method_name in all_methods:
        row_count = method_row_counts.get(method_name, 0)
        expected = roster["total_items"]
        report["methods"][method_name] = {
            "row_count": row_count,
            "expected_rows": expected,
            "cardinality_match": row_count == expected,
        }

    report_path = os.path.join(runs_dir, "generation_protocol_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"\n  [protocol report] written to {report_path}")

    # Also write the roster report
    roster_report_path = os.path.join(runs_dir, "continuation_roster_report.json")
    roster_summary = {k: v for k, v in roster.items() if k != "items"}
    roster_summary["item_count"] = len(roster["items"])
    roster_summary["sample_items"] = roster["items"][:3]
    with open(roster_report_path, "w", encoding="utf-8") as f:
        json.dump(roster_summary, f, indent=2)
    print(f"  [roster report] written to {roster_report_path}")

    return report


def verify_generation_invariants(method_dir, method_name, roster):
    """Quick post-generation invariant checks for one method."""
    gen_path = os.path.join(method_dir, "generations.jsonl")
    if not os.path.exists(gen_path):
        return {"method": method_name, "status": "MISSING", "issues": ["generations.jsonl not found"]}

    records = []
    with open(gen_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))

    issues = []

    # Check protocol fields
    for rec in records:
        if rec.get("prompt_protocol") != PROTOCOL_VERSION:
            issues.append(f"wrong prompt_protocol: {rec.get('prompt_protocol')}")
            break
        if rec.get("style_seed_used") is not False:
            issues.append(f"style_seed_used not False: {rec.get('style_seed_used')}")
            break
        if rec.get("seed_id") is not None:
            issues.append(f"seed_id not None: {rec.get('seed_id')}")
            break
        if rec.get("client_id") != rec.get("owner_client_id"):
            issues.append(
                f"client_id != owner_client_id: "
                f"{rec.get('client_id')} != {rec.get('owner_client_id')}"
            )
            break
        cond = rec.get("conditioning_text", "")
        if "[TASK]" in cond or "PASSAGE:" in cond or "[OUTPUT]" in cond:
            issues.append("conditioning_text contains template shell markers")
            break
        if not rec.get("conditioning_text"):
            issues.append("conditioning_text is empty")
            break
        if not rec.get("reference_continuation_text"):
            issues.append("reference_continuation_text is empty")
            break

    # Cardinality
    expected = roster["total_items"]
    if len(records) != expected:
        issues.append(f"row_count={len(records)}, expected={expected}")

    # continuation_id uniqueness
    cids = [r["continuation_id"] for r in records]
    if len(set(cids)) != len(cids):
        issues.append("duplicate continuation_ids in output")

    status = "PASS" if not issues else "FAIL"
    return {
        "method": method_name,
        "status": status,
        "row_count": len(records),
        "expected_rows": expected,
        "issues": issues,
    }


# ============================================================
# Method-specific adapter resolution
# ============================================================

def resolve_adapters_for_method(method_name, cfg, client_ids):
    """Resolve adapter paths for a given method.

    Returns (adapter_path, per_client_adapters) — one or both may be None.
    """
    runs_dir = cfg["paths"]["runs_dir"]

    if method_name == "B0_base":
        return None, None

    elif method_name in ["FedAvg", "FedProx", "Pooled PEFT", "Base ASCE"]:
        adapter_path = os.path.join(runs_dir, method_name, "final", "global_content_adapter")
        if not os.path.exists(adapter_path):
            print(f"  ERROR: Shared adapter not found at {adapter_path}")
            return None, None
        return adapter_path, None

    elif is_favor_method(method_name):
        per_client_adapters = {}
        runtime_root = os.path.join(runs_dir, method_name, "_runtime_materialized_adapters")

        for cid in client_ids:
            client_base_dir = os.path.join(runs_dir, method_name, "clients", f"client={cid}")
            latest_round = -1
            latest_adapter = None

            if os.path.exists(client_base_dir):
                for d in os.listdir(client_base_dir):
                    if d.startswith("round="):
                        try:
                            r = int(d.split("=")[1])
                            if r > latest_round:
                                round_dir = os.path.join(client_base_dir, d)
                                residual_dir = os.path.join(round_dir, "private_residual_pack")
                                if os.path.exists(residual_dir):
                                    materialized_dir = os.path.join(
                                        runtime_root, f"client={cid}", f"round={r}",
                                    )
                                    latest_adapter = materialize_favor_runtime_adapter(
                                        residual_dir, materialized_dir,
                                    )
                                    latest_round = r
                        except ValueError:
                            pass

            if latest_adapter:
                per_client_adapters[cid] = latest_adapter
            else:
                print(f"  WARNING: No private FAVoR artifact for {cid}, skipping")

        return None, per_client_adapters if per_client_adapters else None

    elif (
        method_name in ["Ditto", "pFedMe", "FedDPA"]
    ):
        # Personalized FL baselines share the Ditto-style client layout.
        # FedDPA materializes fused runtime adapters at the same path.
        per_client_adapters = {}
        global_fallback = os.path.join(runs_dir, method_name, "final", "global_content_adapter")

        for cid in client_ids:
            client_base_dir = os.path.join(runs_dir, method_name, "clients", f"client={cid}")
            latest_adapter = None

            # Check for 'latest' folder first
            latest_path = os.path.join(client_base_dir, "latest", "personalized_adapter")
            has_latest = (
                os.path.exists(os.path.join(latest_path, "adapter_model.safetensors"))
                or os.path.exists(os.path.join(latest_path, "adapter_model.bin"))
            )
            if has_latest:
                latest_adapter = latest_path
            elif os.path.exists(client_base_dir):
                latest_round = -1
                for d in os.listdir(client_base_dir):
                    if d.startswith("round="):
                        try:
                            r = int(d.split("=")[1])
                            if r > latest_round:
                                cand = os.path.join(client_base_dir, d, "personalized_adapter")
                                has_weights = (
                                    os.path.exists(os.path.join(cand, "adapter_model.safetensors"))
                                    or os.path.exists(os.path.join(cand, "adapter_model.bin"))
                                )
                                if has_weights:
                                    latest_round = r
                                    latest_adapter = cand
                        except ValueError:
                            pass

            if latest_adapter:
                per_client_adapters[cid] = latest_adapter
            else:
                print(f"  WARNING: No personalized adapter for {cid} under {method_name}, using global fallback")
                per_client_adapters[cid] = global_fallback

        return None, per_client_adapters

    elif method_name in ["Local-only + ASCE"]:
        per_client_adapters = {}
        for cid in client_ids:
            adapter_dir = os.path.join(runs_dir, method_name, f"client={cid}", "adapter")
            per_client_adapters[cid] = adapter_dir
        return None, per_client_adapters

    else:
        print(f"  WARNING: Unknown method {method_name}, treating as base model")
        return None, None


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Generate held-out continuations for Exp2 (Phase A protocol)"
    )
    parser.add_argument(
        "--config", type=str, default="config/favor_main.yaml",
        help="Exp2 config YAML",
    )
    parser.add_argument(
        "--roster", type=str, default=None,
        help="Path to continuation roster JSON (default: from config)",
    )
    parser.add_argument(
        "--methods", nargs="+",
        default=[
            "FedAvg",
            "FedProx",
            "pFedMe",
            "Ditto",
            "FedDPA",
            "Pooled PEFT",
            "FAVoR",
        ],
        help="Methods to generate for",
    )
    parser.add_argument("--gpu", type=int, default=None)
    parser.add_argument(
        "--backend", choices=["transformers"], default="transformers",
        help="Generation backend (default: transformers)",
    )
    args = parser.parse_args()

    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    with open(cfg["paths"]["base_model_config"], "r", encoding="utf-8") as f:
        model_cfg = yaml.safe_load(f)

    # Load continuation roster
    roster_path = args.roster or cfg.get("generation", {}).get(
        "continuation_roster_path", "data/continuation_roster.json"
    )
    roster = load_continuation_roster(roster_path)
    client_items = group_roster_by_client(roster)

    runs_dir = cfg["paths"]["runs_dir"]
    os.makedirs(runs_dir, exist_ok=True)

    print("=" * 60)
    print("Step 46b — Exp2 Held-Out Continuation Generation")
    print("=" * 60)
    print(f"  Protocol:       {PROTOCOL_VERSION}")
    print(f"  Roster:         {roster_path}")
    print(f"  Total items:    {roster['total_items']}")
    print(f"  Clients:        {len(client_items)}")
    print(f"  C per client:   {roster['continuations_per_client']}")
    print(f"  Balanced:       {roster['balanced']}")
    print(f"  Methods:        {args.methods}")
    print(f"  Backend:        {args.backend}")

    method_row_counts = {}

    for method in args.methods:
        print(f"\n{'=' * 40} {method} {'=' * 40}")

        # Resolve adapters
        adapter_path, per_client_adapters = resolve_adapters_for_method(
            method, cfg, list(client_items.keys()),
        )

        # Filter client_items to clients with adapters
        if per_client_adapters is not None:
            active_items = {
                cid: items for cid, items in client_items.items()
                if cid in per_client_adapters
            }
        else:
            active_items = client_items

        if not active_items:
            print(f"  ERROR: No eligible clients for {method}")
            method_row_counts[method] = 0
            continue

        gen_path = generate_for_method_continuation(
            method, cfg, model_cfg, roster, active_items,
            adapter_path=adapter_path,
            per_client_adapters=per_client_adapters,
            backend=args.backend,
        )

        # Count rows
        row_count = 0
        if os.path.exists(gen_path):
            with open(gen_path, "r") as f:
                row_count = sum(1 for line in f if line.strip())
        method_row_counts[method] = row_count

        # Verify invariants
        method_dir = os.path.join(runs_dir, method)
        check = verify_generation_invariants(method_dir, method, roster)
        print(f"  [invariant check] {check['status']}: {check}")

    # Write protocol report
    write_protocol_report(runs_dir, roster, args.methods, method_row_counts)

    print(f"\n✓ Continuation generation complete for methods: {args.methods}")


if __name__ == "__main__":
    main()
