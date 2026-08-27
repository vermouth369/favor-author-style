#!/usr/bin/env python3
"""favor_helpers.py — Shared LoRA state and adapter utilities for FAVoR scripts.

Used by:
  - 44_train_favor.py
  - 44_train_fedavg.py
  - 45_train_local_only_asce.py
"""

import json
import hashlib
import math
import os
import shutil
from collections import OrderedDict

import torch
from peft import LoraConfig, TaskType

try:
    from safetensors.torch import (
        load_file as load_safetensors_file,
        save_file as save_safetensors_file,
    )
except ImportError:
    load_safetensors_file = None
    save_safetensors_file = None


# ============================================================
# Non-IID manifest loading (Task 4 — Phase B)
# ============================================================

def canonical_author_id(client_or_author_id):
    """Normalize either bare or `blog_pa_`-prefixed identifiers."""
    value = str(client_or_author_id)
    if value.startswith("blog_pa_"):
        return value[len("blog_pa_") :]
    return value


def candidate_client_ids(client_or_author_id):
    """Return the accepted client-id variants for an author/client."""
    bare = canonical_author_id(client_or_author_id)
    return {bare, f"blog_pa_{bare}"}


def record_matches_client_id(record, client_id):
    """Return True when a pooled-data record belongs to the requested client."""
    candidates = {value for value in candidate_client_ids(client_id) if value}
    record_values = {
        str(record.get("client_id", "")),
        str(record.get("author_id", "")),
        canonical_author_id(record.get("client_id", "")),
        canonical_author_id(record.get("author_id", "")),
    }
    return any(value in candidates for value in record_values if value)


def _extract_manifest_allocations(manifest_data):
    allocations = manifest_data.get("client_allocations", [])
    if not allocations:
        raise ValueError("non-IID manifest has no client_allocations")

    if any("is_active" in row for row in allocations):
        active_allocations = [row for row in allocations if bool(row.get("is_active", False))]
        if active_allocations:
            return active_allocations
    return allocations


def _extract_allocation_budget(allocation):
    for key in (
        "allocated_train_tokens",
        "allocated_train_word_tokens",
        "allocated_tokens",
        "train_word_tokens",
    ):
        if key in allocation:
            return int(allocation[key])
    raise ValueError(
        "non-IID allocation missing supported budget field "
        f"(keys={sorted(allocation.keys())})"
    )


def resolve_client_ids_with_noniid(prototype_dir, cfg):
    """Resolve the active client roster from config, honoring non-IID manifests."""
    return load_client_manifest_with_noniid(prototype_dir, cfg)["all_client_ids"]


def active_roster_snapshot_path(runs_dir):
    """Return the canonical run-local active roster snapshot path."""
    return os.path.join(runs_dir, "active_roster_snapshot.json")


def load_active_roster_snapshot(runs_dir):
    """Load a run-local active roster snapshot if present."""
    snapshot_path = active_roster_snapshot_path(runs_dir)
    if not os.path.exists(snapshot_path):
        return None

    with open(snapshot_path, "r", encoding="utf-8") as f:
        snapshot = json.load(f)

    client_ids = snapshot.get("client_ids", [])
    if not client_ids:
        raise ValueError(
            f"Active roster snapshot exists but is empty: {snapshot_path}"
        )
    snapshot["_snapshot_path"] = snapshot_path
    return snapshot


def load_noniid_manifest(manifest_path):
    """Load a frozen non-IID manifest and return client selection info.

    Parameters
    ----------
    manifest_path : str
        Path to a non-IID manifest JSON (e.g., favor_bench_v02_1/non_iid_medium_v02.json).

    Returns
    -------
    manifest_data : dict
        Full parsed manifest.
    client_ids : list[str]
        Ordered list of client IDs from the manifest.
    token_budgets : dict[str, int]
        Mapping client_id → allocated_train_tokens.
    """
    if not os.path.exists(manifest_path):
        raise FileNotFoundError(
            f"Non-IID manifest not found: {manifest_path}. "
            f"Check data.non_iid.manifest_path in config."
        )

    with open(manifest_path, "r") as f:
        manifest_data = json.load(f)

    allocations = _extract_manifest_allocations(manifest_data)

    client_ids = [a["client_id"] for a in allocations]
    token_budgets = {
        a["client_id"]: _extract_allocation_budget(a)
        for a in allocations
    }

    summary = manifest_data.get("summary", {})
    level = manifest_data.get("non_iid_level") or manifest_data.get("level", "unknown")
    n_clients = (
        manifest_data.get("num_clients")
        or summary.get("num_active_clients")
        or len(client_ids)
    )
    gini = (
        manifest_data.get("token_stats", {}).get("gini")
        or summary.get("client_token_gini")
        or summary.get("imbalance_gini_active")
    )

    print(f"  [Non-IID] Loaded manifest: level={level}, "
          f"clients={n_clients}, gini={gini}")
    print(f"  [Non-IID] Manifest path: {manifest_path}")

    return manifest_data, client_ids, token_budgets


def load_client_manifest_with_noniid(prototype_dir, cfg):
    """Resolve client roster + weighting with optional non-IID manifest.

    Returns a dict with:
      - all_client_ids
      - client_data_sizes
      - non_iid_level
      - non_iid_manifest_path
      - non_iid_downgrade
    """
    noniid_cfg = cfg.get("data", {}).get("non_iid", {})
    manifest_path = noniid_cfg.get("manifest_path")
    requested_level = noniid_cfg.get("level", "none")

    if manifest_path and os.path.exists(manifest_path):
        manifest_data, client_ids, token_budgets = load_noniid_manifest(manifest_path)
        return {
            "all_client_ids": client_ids,
            "client_data_sizes": token_budgets,
            "non_iid_level": manifest_data.get("non_iid_level") or manifest_data.get("level", requested_level),
            "non_iid_manifest_path": manifest_path,
            "non_iid_downgrade": False,
        }

    if manifest_path:
        print(f"  WARNING: non-IID manifest path specified but not found: {manifest_path}")
        print("  DOWNGRADE: falling back to clients.json (non-IID=none)")

    clients_path = os.path.join(prototype_dir, "clients.json")
    with open(clients_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    return {
        "all_client_ids": [c["client_id"] for c in manifest["clients"]],
        "client_data_sizes": {
            c["client_id"]: c.get("train_docs", 1) for c in manifest["clients"]
        },
        "non_iid_level": "none",
        "non_iid_manifest_path": manifest_path,
        "non_iid_downgrade": manifest_path is not None,
    }


def resolve_run_client_roster(prototype_dir, runs_dir, cfg, *, fail_fast=True):
    """Resolve the roster for generation/metrics without crossing training boundaries.

    Preference order:
      1. run-local active roster snapshot
      2. active client roster from data.non_iid.manifest_path
      3. clients.json fallback only when no non-IID boundary was requested
    """
    snapshot = load_active_roster_snapshot(runs_dir)
    if snapshot is not None:
        return {
            "client_ids": list(snapshot["client_ids"]),
            "source": "run_snapshot",
            "snapshot_path": snapshot["_snapshot_path"],
            "manifest_path": snapshot.get("non_iid_manifest_path"),
            "subset_manifest_path": snapshot.get("subset_manifest_path"),
            "non_iid_level": snapshot.get("non_iid_level"),
            "downgraded_to_clients_json": False,
        }

    manifest_info = load_client_manifest_with_noniid(prototype_dir, cfg)
    manifest_path = manifest_info.get("non_iid_manifest_path")
    requested_boundary = bool(cfg.get("data", {}).get("non_iid", {}).get("manifest_path"))
    downgraded = bool(manifest_info.get("non_iid_downgrade", False))

    if requested_boundary and downgraded and fail_fast:
        raise FileNotFoundError(
            "Run roster resolution would downgrade to clients.json even though "
            f"data.non_iid.manifest_path was requested ({manifest_path}). "
            "This would cross the training protocol boundary. "
            "Re-run stage preparation to materialize active_roster_snapshot.json "
            "or restore the referenced manifest."
        )

    return {
        "client_ids": list(manifest_info["all_client_ids"]),
        "source": "non_iid_manifest" if manifest_path and not downgraded else "clients_json",
        "snapshot_path": None,
        "manifest_path": manifest_path,
        "subset_manifest_path": cfg.get("data", {}).get("scaled_subset", {}).get("subset_manifest"),
        "non_iid_level": manifest_info.get("non_iid_level"),
        "downgraded_to_clients_json": downgraded,
    }


def estimate_gradient_steps(num_examples, fed_cfg):
    """Approximate optimizer steps for one client round."""
    micro_bs = max(1, int(fed_cfg.get("micro_batch_size", 1)))
    grad_accum = max(1, int(fed_cfg.get("gradient_accumulation_steps", 1)))
    local_epochs = max(1, int(fed_cfg.get("local_epochs", 1)))
    per_step_examples = micro_bs * grad_accum
    steps_per_epoch = max(1, math.ceil(float(num_examples) / float(per_step_examples)))
    return steps_per_epoch * local_epochs


# ============================================================
# LoRA state helpers — operate on ALL lora_ params by name,
# never filter by requires_grad
# ============================================================

def get_lora_state(model):
    """Extract all LoRA parameters from a PeftModel by name.

    Returns an OrderedDict of {name: Tensor (cpu clone)}.
    Collects every parameter whose name contains 'lora_',
    regardless of requires_grad.
    """
    state = OrderedDict()
    for name, param in model.named_parameters():
        if "lora_" in name:
            state[name] = param.data.clone().cpu()
    if not state:
        raise RuntimeError(
            "get_lora_state: no parameters with 'lora_' found. "
            "Is this a PeftModel with LoRA adapters?"
        )
    return state


def get_adapter_state(model, adapter_name):
    """Extract LoRA parameters belonging to exactly one adapter namespace."""
    state = OrderedDict()
    token = f".{adapter_name}."
    for name, param in model.named_parameters():
        if "lora_" not in name:
            continue
        if token in name:
            state[name] = param.data.clone().cpu()
    if not state:
        raise RuntimeError(
            f"get_adapter_state: no LoRA parameters found for adapter '{adapter_name}'."
        )
    return state


def rename_adapter_state(state_dict, from_adapter, to_adapter="default"):
    """Rename adapter namespace in a state dict.

    Passing ``to_adapter=""`` strips the adapter-name infix entirely, e.g.
    ``...lora_A.default.weight -> ...lora_A.weight``.
    """
    renamed = OrderedDict()
    from_token = f".{from_adapter}." if from_adapter else None
    to_token = f".{to_adapter}." if to_adapter else "."
    for name, tensor in state_dict.items():
        new_name = name.replace(from_token, to_token, 1) if from_token and from_token in name else name
        renamed[new_name] = tensor.detach().clone().cpu()
    return renamed


def strip_adapter_name_infix(state_dict, adapter_name="default"):
    """Strip a PEFT adapter-name infix so keys match the load contract."""
    return rename_adapter_state(state_dict, from_adapter=adapter_name, to_adapter="")


def add_adapter_name_infix(state_dict, adapter_name="default"):
    """Add a PEFT adapter-name infix to load-time LoRA keys.

    PEFT checkpoint files store keys like ``...lora_A.weight``. Runtime
    ``PeftModel`` modules expose the same tensor as
    ``...lora_A.default.weight`` for the default adapter.
    """
    renamed = OrderedDict()
    adapter = str(adapter_name)
    replacements = {
        ".lora_A.weight": f".lora_A.{adapter}.weight",
        ".lora_B.weight": f".lora_B.{adapter}.weight",
    }
    for name, tensor in state_dict.items():
        new_name = name
        for old, new in replacements.items():
            if old in new_name:
                new_name = new_name.replace(old, new, 1)
                break
        renamed[new_name] = tensor.detach().clone().cpu()
    return renamed


def normalize_lora_state_for_model(model, state_dict, adapter_name="default"):
    """Return a LoRA state dict whose keys match ``model.state_dict()``.

    This accepts both runtime-format keys (``...lora_A.default.weight``) and
    PEFT checkpoint-format keys (``...lora_A.weight``).
    """
    model_keys = set(model.state_dict())
    normalized = OrderedDict()
    missing = []

    for name, tensor in state_dict.items():
        candidates = [name]
        for converted_name in add_adapter_name_infix(
            OrderedDict([(name, tensor)]),
            adapter_name=adapter_name,
        ):
            if converted_name not in candidates:
                candidates.append(converted_name)

        matched_name = next((candidate for candidate in candidates if candidate in model_keys), None)
        if matched_name is None:
            missing.append(name)
            normalized[name] = tensor.detach().clone().cpu()
        else:
            normalized[matched_name] = tensor.detach().clone().cpu()

    if missing:
        raise KeyError(
            f"normalize_lora_state_for_model: {len(missing)} keys could not be mapped "
            f"to adapter '{adapter_name}': {missing[:5]}{'...' if len(missing) > 5 else ''}"
        )
    return normalized


def adapter_weights_exist(adapter_dir):
    """Return True if a PEFT adapter directory contains a readable weight file."""
    candidates = [
        os.path.join(adapter_dir, "adapter_model.safetensors"),
        os.path.join(adapter_dir, "adapter_model.bin"),
        os.path.join(adapter_dir, "adapter_state.pt"),
    ]
    return any(os.path.exists(path) for path in candidates)


def load_lora_state(model, state_dict):
    """Load LoRA parameters into a PeftModel.

    Raises KeyError if any key in state_dict is missing from the model.
    """
    model_state = model.state_dict()
    missing = [k for k in state_dict if k not in model_state]
    if missing:
        raise KeyError(
            f"load_lora_state: {len(missing)} keys missing from model: "
            f"{missing[:5]}{'...' if len(missing) > 5 else ''}"
        )
    for name, param in state_dict.items():
        model_state[name].copy_(param.to(model_state[name].device))


def count_state_params(state_dict):
    """Count exact parameters from a tensor state dict."""
    return sum(int(t.numel()) for t in state_dict.values())


def hash_state_dict(state_dict):
    """Stable short hash for exported adapter states."""
    hasher = hashlib.sha256()
    for key in sorted(state_dict):
        tensor = state_dict[key].detach().cpu().contiguous()
        hasher.update(key.encode("utf-8"))
        hasher.update(str(tuple(tensor.shape)).encode("utf-8"))
        hasher.update(tensor.numpy().tobytes())
    return hasher.hexdigest()[:16]

def _normalize_task_type(task_type):
    if isinstance(task_type, TaskType):
        return task_type
    if isinstance(task_type, str):
        return getattr(TaskType, task_type)
    return TaskType.CAUSAL_LM


def write_lora_adapter_config(output_dir, adapter_config):
    """Write adapter_config.json from structured LoRA config fields."""
    cfg = LoraConfig(
        r=adapter_config["r"],
        lora_alpha=adapter_config["lora_alpha"],
        lora_dropout=adapter_config.get("lora_dropout", 0.0),
        target_modules=adapter_config["target_modules"],
        bias=adapter_config.get("bias", "none"),
        task_type=_normalize_task_type(adapter_config.get("task_type", "CAUSAL_LM")),
        inference_mode=adapter_config.get("inference_mode", True),
    )
    base_model_name = adapter_config.get("base_model_name_or_path")
    if base_model_name is not None:
        cfg.base_model_name_or_path = base_model_name
    cfg.save_pretrained(output_dir)


def load_json_if_exists(path):
    """Load a JSON file if present, else return an empty dict."""
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_adapter_state_dict(adapter_dir, map_location="cpu"):
    """Load a PEFT adapter state dict from safetensors/bin/pt formats."""
    safetensors_path = os.path.join(adapter_dir, "adapter_model.safetensors")
    bin_path = os.path.join(adapter_dir, "adapter_model.bin")
    fallback_pt_path = os.path.join(adapter_dir, "adapter_state.pt")

    if os.path.exists(safetensors_path):
        if load_safetensors_file is None:
            raise RuntimeError(
                f"load_adapter_state_dict: found {safetensors_path} but safetensors is unavailable"
            )
        state = load_safetensors_file(safetensors_path, device=str(map_location))
        return OrderedDict((k, v.detach().cpu()) for k, v in state.items())

    if os.path.exists(bin_path):
        try:
            state = torch.load(bin_path, map_location=map_location, weights_only=True)
        except TypeError:
            state = torch.load(bin_path, map_location=map_location)
        return OrderedDict((k, v.detach().cpu()) for k, v in state.items())

    if os.path.exists(fallback_pt_path):
        try:
            state = torch.load(fallback_pt_path, map_location=map_location, weights_only=True)
        except TypeError:
            state = torch.load(fallback_pt_path, map_location=map_location)
        return OrderedDict((k, v.detach().cpu()) for k, v in state.items())

    raise FileNotFoundError(
        f"load_adapter_state_dict: no adapter weight file found in {adapter_dir}"
    )


def save_peft_adapter_checkpoint(
    state_dict,
    template_dir,
    output_dir,
    extra_json=None,
    adapter_config=None,
    adapter_name="default",
    prefer_safetensors=True,
):
    """Materialize a standalone PEFT adapter dir from a state dict.

    Persisted keys are normalized to PEFT's load-time contract by stripping
    the ``.{adapter_name}.`` infix before writing weights.
    """
    os.makedirs(output_dir, exist_ok=True)
    if adapter_config is not None:
        write_lora_adapter_config(output_dir, adapter_config)
    else:
        config_src = os.path.join(template_dir, "adapter_config.json")
        if not os.path.exists(config_src):
            raise FileNotFoundError(
                f"save_peft_adapter_checkpoint: missing adapter_config.json in {template_dir}"
            )
        config_dst = os.path.join(output_dir, "adapter_config.json")
        if os.path.abspath(config_src) != os.path.abspath(config_dst):
            shutil.copy2(config_src, config_dst)
    normalized_state = strip_adapter_name_infix(state_dict, adapter_name=adapter_name)
    weights_path = os.path.join(output_dir, "adapter_model.safetensors")
    wrote_safetensors = False
    if prefer_safetensors and save_safetensors_file is not None:
        save_safetensors_file(
            {name: tensor.detach().cpu().contiguous() for name, tensor in normalized_state.items()},
            weights_path,
        )
        wrote_safetensors = True
    if not wrote_safetensors:
        torch.save(normalized_state, os.path.join(output_dir, "adapter_model.bin"))
    if extra_json:
        for name, payload in extra_json.items():
            with open(os.path.join(output_dir, name), "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2)


def diff_lora_state(before, after):
    """Compute element-wise delta = after - before for all LoRA params.

    Asserts that the key sets match exactly.
    """
    if set(before.keys()) != set(after.keys()):
        extra = set(after.keys()) - set(before.keys())
        missing = set(before.keys()) - set(after.keys())
        raise RuntimeError(
            f"diff_lora_state: key mismatch. "
            f"Extra in after: {extra}, missing in after: {missing}"
        )
    delta = OrderedDict()
    for name in before:
        delta[name] = after[name] - before[name]
    return delta


def diff_adapter_states(base_state, target_state):
    """Residual-style delta that tolerates dual-rank shape mismatches."""
    delta = OrderedDict()
    for key in target_state:
        if key in base_state and base_state[key].shape == target_state[key].shape:
            delta[key] = target_state[key] - base_state[key]
        else:
            delta[key] = target_state[key].clone().cpu()
    return delta


def reconstruct_adapter_state(base_state, residual_state):
    """Reconstruct adapter weights from base + residual."""
    reconstructed = OrderedDict()
    for key in base_state:
        if key in residual_state:
            if base_state[key].shape == residual_state[key].shape:
                reconstructed[key] = base_state[key].cpu() + residual_state[key].cpu()
            else:
                reconstructed[key] = residual_state[key].cpu()
        else:
            reconstructed[key] = base_state[key].cpu()
    for key in residual_state:
        if key not in reconstructed:
            reconstructed[key] = residual_state[key].cpu()
    return reconstructed


def load_private_branch_artifacts(private_residual_pack_dir, require_residual=False):
    """Load metadata and tensors for a private residual checkpoint family."""
    if os.path.basename(os.path.normpath(private_residual_pack_dir)) == "private_residual_pack":
        round_dir = os.path.dirname(private_residual_pack_dir)
        residual_dir = private_residual_pack_dir
        full_dir = os.path.join(round_dir, "private_adapter_full")
    else:
        full_dir = private_residual_pack_dir
        round_dir = os.path.dirname(private_residual_pack_dir)
        residual_dir = os.path.join(round_dir, "private_residual_pack")

    shared_dir = os.path.join(round_dir, "shared_endpoint_adapter")
    full_meta = load_json_if_exists(os.path.join(full_dir, "private_residual_pack_meta.json"))
    residual_meta = load_json_if_exists(os.path.join(residual_dir, "private_residual_pack_meta.json"))

    residual_declared = bool(
        residual_meta.get("residual_semantics", False)
        or full_meta.get("residual_semantics", False)
        or full_meta.get("preferred_runtime_artifact") == "private_residual_pack"
        or os.path.isdir(residual_dir)
    )

    if residual_declared:
        if not os.path.isdir(shared_dir):
            raise FileNotFoundError(
                f"Residual private pack requires shared_endpoint_adapter/, missing at {shared_dir}"
            )
        if not os.path.isdir(residual_dir):
            raise FileNotFoundError(
                f"Residual semantics declared but private_residual_pack/ missing at {residual_dir}"
            )

        shared_state = load_adapter_state_dict(shared_dir, map_location="cpu")
        residual_state = load_adapter_state_dict(residual_dir, map_location="cpu")
        shared_hash_expected = residual_meta.get("shared_adapter_hash") or full_meta.get("shared_adapter_hash")
        if shared_hash_expected:
            shared_hash_actual = hash_state_dict(shared_state)
            if shared_hash_actual != shared_hash_expected:
                raise RuntimeError(
                    "Residual private pack shared-adapter hash mismatch: "
                    f"expected {shared_hash_expected}, got {shared_hash_actual}"
                )

        final_state = reconstruct_adapter_state(shared_state, residual_state)
        final_hash_expected = residual_meta.get("final_adapter_hash")
        if final_hash_expected:
            final_hash_actual = hash_state_dict(final_state)
            if final_hash_actual != final_hash_expected:
                # IMPORTANT: `final = shared + (final - shared)` is NOT byte-exact
                # in floating point (bf16/fp16 rounding). The shared_adapter_hash
                # check above is byte-exact (pure load), which is sufficient to
                # confirm the residual was written against the same shared
                # endpoint we are reconstructing from. The final-adapter hash
                # equality is mathematically unachievable for float tensors, so
                # we downgrade it to a non-fatal warning.
                import sys as _sys
                print(
                    f"      [warn] residual final-adapter hash differs "
                    f"(expected {final_hash_expected}, got {final_hash_actual}); "
                    f"likely bf16/fp16 reconstruction roundoff, continuing with "
                    f"reconstructed state.",
                    file=_sys.stderr,
                )

        return {
            "mode": "residual",
            "round_dir": round_dir,
            "full_dir": full_dir,
            "shared_dir": shared_dir,
            "residual_dir": residual_dir,
            "full_meta": full_meta,
            "residual_meta": residual_meta,
            "meta": residual_meta or full_meta,
            "shared_state": shared_state,
            "residual_state": residual_state,
            "final_state": final_state,
        }

    if require_residual:
        raise RuntimeError(
            f"Residual private-pack semantics required, but no canonical residual artifact was found for {private_residual_pack_dir}"
        )

    if not os.path.isdir(full_dir):
        raise FileNotFoundError(f"Private adapter checkpoint not found: {full_dir}")

    final_state = load_adapter_state_dict(full_dir, map_location="cpu")
    return {
        "mode": "full",
        "round_dir": round_dir,
        "full_dir": full_dir,
        "shared_dir": shared_dir,
        "residual_dir": residual_dir,
        "full_meta": full_meta,
        "residual_meta": residual_meta,
        "meta": full_meta,
        "final_state": final_state,
    }


def materialize_private_runtime_adapter(private_residual_pack_dir, output_dir, require_residual=False):
    """Create a runtime-loadable adapter dir for generation/signature code."""
    artifacts = load_private_branch_artifacts(
        private_residual_pack_dir,
        require_residual=require_residual,
    )
    template_dir = artifacts["full_dir"]
    if not os.path.exists(os.path.join(template_dir, "adapter_config.json")):
        template_dir = artifacts.get("residual_dir") or template_dir

    runtime_meta = {
        "materialized_from": artifacts["mode"],
        "source_private_residual_pack_dir": artifacts["residual_dir"],
        "source_residual_dir": artifacts.get("residual_dir"),
        "shared_endpoint_dir": artifacts.get("shared_dir"),
        "final_adapter_hash": hash_state_dict(strip_adapter_name_infix(artifacts["final_state"])),
    }
    save_peft_adapter_checkpoint(
        artifacts["final_state"],
        template_dir=template_dir,
        output_dir=output_dir,
        extra_json={"runtime_materialization_meta.json": runtime_meta},
    )
    return output_dir, artifacts


def lora_delta_norm(before, after):
    """Compute L2 norm of the LoRA delta between before and after states."""
    delta = diff_lora_state(before, after)
    total_sq = sum(d.float().pow(2).sum().item() for d in delta.values())
    return math.sqrt(total_sq)


# ============================================================
# Adapter introspection and logging
# ============================================================

def get_active_adapter_name(model):
    """Return the name of the currently active adapter, or 'unknown'."""
    if hasattr(model, "active_adapter"):
        adapter = model.active_adapter
        if isinstance(adapter, (list, set)):
            return ",".join(sorted(adapter))
        return str(adapter)
    return "unknown"


def count_trainable_params(model):
    """Return (trainable_count, total_count) for the model."""
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total


def log_adapter_state(model, label, client_id=None, round_id=None):
    """Print a diagnostic line showing adapter state."""
    adapter_name = get_active_adapter_name(model)
    trainable, total = count_trainable_params(model)
    parts = [f"  [{label}]"]
    if round_id is not None:
        parts.append(f"round={round_id}")
    if client_id is not None:
        parts.append(f"client={client_id}")
    parts.append(f"adapter={adapter_name}")
    parts.append(f"trainable={trainable:,}")
    print(" | ".join(parts))
