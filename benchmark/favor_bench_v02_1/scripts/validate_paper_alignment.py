#!/usr/bin/env python3
"""Check that the archive follows the paper protocol."""

from __future__ import annotations

import json
import sys
from pathlib import Path


REQUIRED_FILES = [
    "clients_manifest_v02.json",
    "non_iid_medium_v02.json",
    "continuation_eval_protocol.yaml",
    "method_config_resolved.yaml",
    "asce_encoder_spec.json",
    "quality_filter_v2.yaml",
    "mythos_reddit_protocol.json",
    "method_names.json",
]

DISALLOWED_FILES = [
    "non_iid_" + "mi" + "ld_v02.json",
    "non_iid_" + "str" + "ong_v02.json",
]


def assembled_obsolete_terms() -> list[str]:
    return [
        "rout" + "ing_method",
        "rout" + "ing_temperature",
        "num_" + "style_" + "proto" + "types",
        "style_" + "rou" + "ted",
        "voice_" + "anchor",
        "private_" + "pack_" + "rank",
        "E" + "simple",
        "E" + "hybrid",
        "E" + "cont",
        "G" + "YAFC",
        "prompt_" + "suite",
        "assistant" + "ization",
        "task_" + "family",
    ]


def main() -> int:
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(".")
    failures = []
    for rel in REQUIRED_FILES:
        if not (root / rel).exists():
            failures.append(f"missing required file: {rel}")
    for rel in DISALLOWED_FILES:
        if (root / rel).exists():
            failures.append(f"unreported split metadata should not be included: {rel}")

    if failures:
        print("FAIL")
        print("\n".join(failures))
        return 1

    all_text = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.name == "validate_paper_alignment.py" or path.suffix in {".tgz", ".gz", ".zip"}:
            continue
        try:
            all_text.append((path.relative_to(root).as_posix(), path.read_text(encoding="utf-8")))
        except UnicodeDecodeError:
            continue

    for rel, text in all_text:
        for term in assembled_obsolete_terms():
            if term in text:
                failures.append(f"obsolete term found in {rel}: {term}")

    qf = (root / "quality_filter_v2.yaml").read_text(encoding="utf-8")
    if "fail_if_output_word_count_lt: 20" not in qf or "fail_if_distinct_bigram_ratio_lt: 0.15" not in qf:
        failures.append("quality filter does not match 20 words and 0.15 distinct-bigram threshold")

    protocol = (root / "continuation_eval_protocol.yaml").read_text(encoding="utf-8")
    if "prefix_tokens: 96" not in protocol or "prefix_tokens: 192" not in protocol:
        failures.append("continuation protocol must include BlogText 96-token and Mythos-Reddit 192-token prefixes")

    method_names = json.loads((root / "method_names.json").read_text(encoding="utf-8"))
    names = set(method_names.get("official_names_used_in_paper", []))
    for method in ["FedAvg", "FedProx", "pFedMe", "Ditto", "FedDPA", "FAVoR"]:
        if method not in names:
            failures.append(f"canonical method missing: {method}")
    if "prompt_" + "only" in names:
        failures.append("non-paper canonical method found")

    asce = json.loads((root / "asce_encoder_spec.json").read_text(encoding="utf-8"))
    if asce.get("backbone") != "distilroberta-base":
        failures.append("ASCE backbone must be distilroberta-base")
    if asce.get("embedding_dim") != 256:
        failures.append("ASCE embedding_dim must be 256")
    if asce.get("training_epochs") != 3:
        failures.append("ASCE training_epochs must be 3")
    if asce.get("scale_gamma") != 30.0:
        failures.append("ASCE scale_gamma must be 30.0")
    if asce.get("angular_margin") != 0.35:
        failures.append("ASCE angular_margin must be 0.35 radians")
    architecture = asce.get("architecture", {})
    expected_architecture = {
        "transformer_layers": 6,
        "hidden_size": 768,
        "attention_heads": 12,
        "feed_forward_size": 3072,
        "projection": "Linear(768, 256)",
    }
    for key, expected in expected_architecture.items():
        if architecture.get(key) != expected:
            failures.append(f"ASCE architecture.{key} must be {expected}")
    if asce.get("num_disjoint_training_authors") != 50:
        failures.append("ASCE should report 50 disjoint training authors")
    if asce.get("main_roster_overlap_count") != 0:
        failures.append("ASCE training authors should be disjoint from the main roster")

    if failures:
        print("FAIL")
        print("\n".join(failures))
        return 1
    print("PASS: archive matches paper-aligned protocol checks")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
