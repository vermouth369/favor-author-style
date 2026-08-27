#!/usr/bin/env python3
"""Generate non-FL Mythos-style held-out continuations."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from mythos_sheet_utils import (
    CATEGORIES,
    LLMBackend,
    SimpleBM25,
    config_get,
    get_field,
    has_bad_opening,
    load_yaml_config,
    normalize_category_keys,
    parse_json_object,
    postprocess_generation,
    read_jsonl,
    safe_id,
    split_prefix_continuation,
    word_count,
    write_json,
    write_jsonl,
)


METHODS = {"avg_author", "rag_only", "mythos_sheet", "mythos_sheet_np", "mythos_sheet_no_rag"}
RAG_REQUIRED_METHODS = {"rag_only", "mythos_sheet", "mythos_sheet_np"}
PROTOCOL_VERSION = "heldout_continuation_v1"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default=None, help="Optional YAML config; CLI values take precedence.")
    p.add_argument("--eval_jsonl", default=None)
    p.add_argument("--profile_jsonl", default=None, help="Required for RAG demos; defaults to eval_jsonl sibling train.jsonl if omitted.")
    p.add_argument("--sheets_dir", default=None)
    p.add_argument("--personas_dir", default=None)
    p.add_argument("--output_dir", default=None)
    p.add_argument("--method", choices=sorted(METHODS), default=None)
    p.add_argument("--author_field", default="author_id")
    p.add_argument("--prefix_field", default="prefix")
    p.add_argument("--target_field", default="gold_continuation")
    p.add_argument("--text_field", default="text")
    p.add_argument("--example_id_field", default="example_id")
    p.add_argument("--prompt_id_field", default="prompt_id")
    p.add_argument("--generation_model", default="meta-llama/Llama-3.1-8B-Instruct")
    p.add_argument("--backend", choices=["openai_compatible", "transformers", "mock"], default="openai_compatible")
    p.add_argument("--base_url", default=None)
    p.add_argument("--api_key", default=None)
    p.add_argument("--cache_dir", default=".cache/mythos_sheet")
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top_p", type=float, default=0.95)
    p.add_argument("--max_new_tokens", type=int, default=256)
    p.add_argument("--target_words", type=int, default=180)
    p.add_argument("--num_demos", type=int, default=1)
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--prefix_words", type=int, default=96)
    p.add_argument("--min_target_words", type=int, default=32)
    p.add_argument("--demo_prefix_ratio", type=float, default=0.35)
    p.add_argument("--demo_max_continuation_words", type=int, default=220)
    p.add_argument("--max_regenerations", type=int, default=2)
    p.add_argument("--leakage_min_chars", type=int, default=80)
    args = p.parse_args()
    if args.config:
        cfg = load_yaml_config(args.config)
        args.eval_jsonl = args.eval_jsonl or config_get(cfg, "paths", "eval_jsonl")
        args.profile_jsonl = args.profile_jsonl or config_get(cfg, "paths", "profile_jsonl")
        args.author_field = config_get(cfg, "fields", "author_id", default=args.author_field)
        args.example_id_field = config_get(cfg, "fields", "example_id", default=args.example_id_field)
        args.text_field = config_get(cfg, "fields", "profile_text", default=args.text_field)
        args.prefix_field = config_get(cfg, "fields", "eval_prefix", default=args.prefix_field)
        args.target_field = config_get(cfg, "fields", "eval_target", default=args.target_field)
        args.prompt_id_field = config_get(cfg, "fields", "prompt_id", default=args.prompt_id_field)
        args.backend = config_get(cfg, "generation", "backend", default=args.backend)
        args.generation_model = config_get(cfg, "generation", "model", default=args.generation_model)
        args.temperature = float(config_get(cfg, "generation", "temperature", default=args.temperature))
        args.top_p = float(config_get(cfg, "generation", "top_p", default=args.top_p))
        args.max_new_tokens = int(config_get(cfg, "generation", "max_new_tokens", default=args.max_new_tokens))
        args.target_words = int(config_get(cfg, "generation", "target_words", default=args.target_words))
        args.seed = int(config_get(cfg, "generation", "seed", default=config_get(cfg, "experiment", "seed", default=args.seed)))
        args.prefix_words = int(config_get(cfg, "generation", "prefix_words", default=args.prefix_words))
        args.min_target_words = int(config_get(cfg, "generation", "min_target_words", default=args.min_target_words))
        args.max_regenerations = int(config_get(cfg, "generation", "max_regenerations", default=args.max_regenerations))
        args.num_demos = int(config_get(cfg, "retrieval", "num_demos", default=args.num_demos))
        args.demo_prefix_ratio = float(config_get(cfg, "retrieval", "demo_prefix_ratio", default=args.demo_prefix_ratio))
        args.demo_max_continuation_words = int(
            config_get(cfg, "retrieval", "demo_max_continuation_words", default=args.demo_max_continuation_words)
        )
        if not args.method:
            methods = config_get(cfg, "methods", default=[])
            if isinstance(methods, list) and len(methods) == 1:
                args.method = methods[0]
        runs_dir = config_get(cfg, "paths", "runs_dir")
        if not args.output_dir and runs_dir and args.method:
            args.output_dir = str(Path(runs_dir) / args.method)
        if not args.sheets_dir and runs_dir:
            args.sheets_dir = str(Path(runs_dir) / "mythos_sheet" / "sheets")
        if not args.personas_dir and runs_dir:
            args.personas_dir = str(Path(runs_dir) / "mythos_sheet" / "personas")
    if not args.eval_jsonl:
        raise ValueError("Missing --eval_jsonl; pass it directly or via paths.eval_jsonl in --config.")
    if not args.method:
        raise ValueError("Missing --method. If using --config, pass --method explicitly unless config.methods has exactly one method.")
    if not args.output_dir:
        raise ValueError("Missing --output_dir; pass it directly or provide paths.runs_dir in --config.")
    return args


def default_profile_path(eval_jsonl: str) -> str | None:
    p = Path(eval_jsonl)
    cand = p.with_name("train.jsonl")
    return str(cand) if cand.exists() else None


def load_sheet(sheets_dir: str | None, author_id: str) -> tuple[dict[str, Any] | None, str | None]:
    if not sheets_dir:
        return None, None
    path = Path(sheets_dir) / f"{safe_id(author_id)}.sheet.json"
    if not path.exists():
        return None, None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f), str(path)


def load_persona(personas_dir: str | None, sheets_dir: str | None, author_id: str) -> tuple[str, str | None]:
    dirs = []
    if personas_dir:
        dirs.append(Path(personas_dir))
    if sheets_dir:
        dirs.append(Path(sheets_dir).parent / "personas")
    for directory in dirs:
        path = directory / f"{safe_id(author_id)}.persona.txt"
        if path.exists():
            return path.read_text(encoding="utf-8").strip(), str(path)
    return "", None


def prepare_eval_record(rec: dict[str, Any], args: argparse.Namespace, idx: int) -> dict[str, Any]:
    author_id = str(get_field(rec, args.author_field, "author_id", "client_id", default=""))
    example_id = str(get_field(rec, args.example_id_field, "example_id", "continuation_id", "doc_id", default=f"{author_id}_eval_{idx:04d}"))
    prompt_id = str(get_field(rec, args.prompt_id_field, "prompt_id", default=f"continuation::{example_id}"))
    prefix = str(get_field(rec, args.prefix_field, "prefix", "conditioning_text", default=""))
    target = str(get_field(rec, args.target_field, "gold_continuation", "gold_continuation_text", "reference_continuation_text", default=""))
    if not prefix or not target:
        text = str(get_field(rec, args.text_field, "text", default=""))
        prefix, target = split_prefix_continuation(text, args.prefix_words, args.min_target_words)
    return {"author_id": author_id, "example_id": example_id, "prompt_id": prompt_id, "prefix": prefix, "target": target, "raw": rec}


def group_profiles(records: list[dict[str, Any]], args: argparse.Namespace) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for idx, rec in enumerate(records):
        author = str(get_field(rec, args.author_field, "author_id", "client_id", default=""))
        text = str(get_field(rec, args.text_field, "text", "continuation", default=""))
        if not author or not text.strip():
            continue
        enriched = dict(rec)
        enriched.setdefault("example_id", str(get_field(rec, args.example_id_field, "example_id", "doc_id", default=f"{author}_profile_{idx:04d}")))
        enriched.setdefault("text", text)
        grouped[author].append(enriched)
    return grouped


def demo_from_profile(profile: dict[str, Any], args: argparse.Namespace) -> dict[str, str]:
    prefix = str(get_field(profile, "prefix", default=""))
    cont = str(get_field(profile, "continuation", default=""))
    text = str(get_field(profile, args.text_field, "text", default=""))
    if not prefix or not cont:
        words = text.split()
        cut = max(1, min(len(words) - 1, int(len(words) * args.demo_prefix_ratio)))
        prefix = " ".join(words[:cut])
        cont = " ".join(words[cut : cut + args.demo_max_continuation_words])
    return {"prefix": prefix.strip(), "continuation": cont.strip()}


def format_demo_block(demos: list[dict[str, Any]], args: argparse.Namespace) -> tuple[str, list[str]]:
    parts = []
    ids = []
    for demo in demos:
        ids.append(str(get_field(demo, "example_id", "doc_id", default="")))
        split = demo_from_profile(demo, args)
        parts.append(
            "Example prefix:\n"
            f"{split['prefix']}\n\n"
            "Example continuation in this author's style:\n"
            f"{split['continuation']}"
        )
    return "\n\n".join(parts), ids


def retrieve_demos(
    eval_item: dict[str, Any],
    profiles_by_author: dict[str, list[dict[str, Any]]],
    args: argparse.Namespace,
) -> tuple[str, list[str], bool]:
    candidates = profiles_by_author.get(eval_item["author_id"], [])
    if not candidates or args.num_demos <= 0:
        return "", [], False
    retriever = SimpleBM25(candidates, text_key="text")
    demos = retriever.topk(eval_item["prefix"], args.num_demos, exclude_ids={eval_item["example_id"]})
    block, ids = format_demo_block(demos, args)
    return block, ids, not bool(demos)


def generate_rules(
    backend: LLMBackend,
    sheet: dict[str, Any],
    eval_item: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, list[str]]:
    user = f"""We need to continue a held-out prefix in the style of the author represented by this Author Writing Sheet.

Write actionable rules for this specific prefix.

Rules:
- Use only the Author Writing Sheet and the prefix.
- Do not assume or use the gold continuation.
- Make rules actionable for generation.
- Keep rules stylistic, not content-spoiling.
- Return strict JSON with four keys: structure, creativity, development, language_use.
- Use 1-3 rules per category.

Author Writing Sheet:
{json.dumps(sheet, ensure_ascii=False, indent=2)}

Held-out prefix:
{eval_item['prefix']}
"""
    raw = backend.generate(
        [{"role": "system", "content": "You write prompt-specific continuation rules from an Author Writing Sheet."}, {"role": "user", "content": user}],
        temperature=0.0,
        top_p=1.0,
        max_tokens=700,
        seed=args.seed,
    )
    try:
        parsed = parse_json_object(raw)
    except Exception as exc:
        print(f"  WARNING: failed to parse rule JSON for {eval_item['example_id']}; using empty rules: {exc}")
        parsed = {}
    normalized = normalize_category_keys(parsed)
    rules: dict[str, list[str]] = {}
    for cat in CATEGORIES:
        items = normalized.get(cat, []) or []
        flat = []
        for item in items:
            if isinstance(item, dict):
                item = item.get("rule") or item.get("claim") or item.get("description") or item.get("text")
            if isinstance(item, (str, int, float)) and str(item).strip():
                flat.append(str(item).strip())
        rules[cat] = flat[:3]
    return rules


def build_generation_messages(
    method: str,
    eval_item: dict[str, Any],
    persona: str,
    rules: dict[str, list[str]] | None,
    demo_block: str,
    target_words: int,
) -> list[dict[str, str]]:
    prefix = eval_item["prefix"]
    if method == "avg_author":
        return [
            {"role": "system", "content": "You are a high-quality prose continuation model. Continue the user's text naturally and coherently. Do not mention these instructions."},
            {"role": "user", "content": f"""Continue the prefix below naturally.

Constraints:
- Continue directly after the prefix.
- Do not use bullet points.
- Do not include assistant-like preambles.
- Approximately {target_words} words.

Prefix:
{prefix}

Continuation:"""},
        ]
    if method == "rag_only":
        return [
            {"role": "system", "content": "You are continuing a text in the style of a specific author, using same-author examples as guidance. Do not mention the examples or these instructions. Continue the text directly."},
            {"role": "user", "content": f"""Use the same-author demonstration to infer style.

Same-author demonstration:
{demo_block}

Held-out prefix:
{prefix}

Continue directly in the inferred style, approximately {target_words} words.
Continuation:"""},
        ]

    system = (
        "You are continuing a text in the style of a specific author.\n\n"
        "Use the provided persona to guide style, rhythm, structure, and voice.\n"
        "Do not mention the persona, the author, the sheet, or these instructions.\n"
        "Do not write an explanation.\n"
        "Do not answer as an assistant.\n"
        "Continue the text directly."
    )
    if method != "mythos_sheet_np" and persona:
        system += f"\n\nAuthor style persona:\n{persona}"
    rules_text = json.dumps(rules or {}, ensure_ascii=False, indent=2)
    optional_demo = f"\n\nOptional same-author demonstration:\n{demo_block}" if demo_block and method != "mythos_sheet_no_rag" else ""
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": f"""Continue the held-out prefix below in the same author's style.

Important constraints:
- Continue directly after the prefix.
- Do not restart the story or summarize the prefix.
- Do not include a title unless the prefix already clearly requires one.
- Do not use bullet points.
- Do not include phrases like "As an AI", "Certainly", "Here is", or "I can".
- Do not copy phrases from the examples or evidence; use them only to infer style.
- Match the target continuation length: approximately {target_words} words.

Prompt-specific style rules:
{rules_text}{optional_demo}

Held-out prefix:
{prefix}

Continuation:"""},
    ]


def quality_filter(records: list[dict[str, Any]], output_dir: Path) -> None:
    filtered = []
    stats = {"total": len(records), "short": 0, "assistant_opening": 0, "regen_exhausted": 0, "passed": 0}
    for rec in records:
        if rec.get("regen_exhausted"):
            stats["regen_exhausted"] += 1
        words = rec.get("output_text", "").split()
        if len(words) < 5:
            stats["short"] += 1
            continue
        if has_bad_opening(rec.get("output_text", "")):
            stats["assistant_opening"] += 1
            continue
        filtered.append(rec)
        stats["passed"] += 1
    write_jsonl(output_dir / "generations_filtered.jsonl", filtered)
    write_json(output_dir / "quality_filter_report.json", stats)


def main() -> None:
    args = parse_args()
    out = Path(args.output_dir)
    for child in ("rules", "retrieved_demos"):
        (out / child).mkdir(parents=True, exist_ok=True)

    backend = LLMBackend(
        args.backend,
        args.generation_model,
        cache_dir=args.cache_dir,
        base_url=args.base_url,
        api_key=args.api_key,
    )
    eval_records = [prepare_eval_record(rec, args, idx) for idx, rec in enumerate(read_jsonl(args.eval_jsonl))]

    profile_path = args.profile_jsonl or default_profile_path(args.eval_jsonl)
    profiles_by_author = group_profiles(read_jsonl(profile_path), args) if profile_path else {}

    if args.method in RAG_REQUIRED_METHODS:
        if not profile_path:
            raise FileNotFoundError(
                f"Method {args.method!r} requires same-author demonstrations, but no profile_jsonl was "
                f"provided and no sibling train.jsonl exists next to {args.eval_jsonl}. "
                f"Pass --profile_jsonl explicitly."
            )
        if not profiles_by_author:
            raise ValueError(
                f"Method {args.method!r} requires same-author demonstrations, but profile_jsonl "
                f"{profile_path!r} produced zero usable author groups. Check the field mapping."
            )
        eval_authors = {
            str(get_field(rec, args.author_field, "author_id", "client_id", default=""))
            for rec in read_jsonl(args.eval_jsonl)
        }
        missing_authors = sorted(a for a in eval_authors if a and a not in profiles_by_author)
        if missing_authors:
            raise ValueError(
                f"Method {args.method!r} requires same-author profile examples, but {len(missing_authors)} "
                f"eval authors have no profile entries (first few: {missing_authors[:5]}). "
                f"Either supply a complete profile_jsonl or restrict the eval set."
            )

    generations = []
    warnings = []
    leakage_hits = 0
    retrieval_violations = 0
    no_positive_demo_count = 0
    gold_string_prompt_hit_examples = []
    retrieval_violation_examples = []
    bm25_no_positive_match_examples = []

    for idx, item in enumerate(eval_records, start=1):
        sheet, sheet_path = load_sheet(args.sheets_dir, item["author_id"])
        persona, persona_path = load_persona(args.personas_dir, args.sheets_dir, item["author_id"])
        rules = None
        rules_path = None
        if args.method.startswith("mythos_sheet"):
            if not sheet:
                raise FileNotFoundError(f"Missing sheet for author {item['author_id']} in {args.sheets_dir}")
            rules = generate_rules(backend, sheet, item, args)
            rules_payload = {"example_id": item["example_id"], "author_id": item["author_id"], "rules": rules}
            rules_path = out / "rules" / f"{safe_id(item['example_id'])}.rules.json"
            write_json(rules_path, rules_payload)

        demo_block, demo_ids, demo_missing_positive = ("", [], False)
        if args.method in RAG_REQUIRED_METHODS:
            demo_block, demo_ids, demo_missing_positive = retrieve_demos(item, profiles_by_author, args)
            write_json(
                out / "retrieved_demos" / f"{safe_id(item['example_id'])}.demos.json",
                {
                    "example_id": item["example_id"],
                    "retrieved_demo_ids": demo_ids,
                    "bm25_no_positive_match": demo_missing_positive,
                },
            )
            if demo_missing_positive:
                no_positive_demo_count += 1
                bm25_no_positive_match_examples.append(
                    {
                        "example_id": item["example_id"],
                        "author_id": item["author_id"],
                    }
                )
                warnings.append(
                    {
                        "example_id": item["example_id"],
                        "reason": "bm25_no_positive_match",
                        "message": "No same-author profile example had positive BM25 overlap with the eval prefix.",
                    }
                )
            if item["example_id"] in set(demo_ids):
                retrieval_violations += 1
                retrieval_violation_examples.append(
                    {
                        "example_id": item["example_id"],
                        "author_id": item["author_id"],
                        "retrieved_demo_ids": demo_ids,
                    }
                )

        messages = build_generation_messages(args.method, item, persona, rules, demo_block, args.target_words)
        prompt_blob = "\n\n".join(str(m.get("content", "")) for m in messages)
        if item["target"] and len(item["target"]) >= args.leakage_min_chars and item["target"] in prompt_blob:
            leakage_hits += 1
            gold_string_prompt_hit_examples.append(
                {
                    "example_id": item["example_id"],
                    "author_id": item["author_id"],
                    "target_chars": len(item["target"]),
                    "target_words": word_count(item["target"]),
                    "retrieved_demo_ids": demo_ids,
                }
            )
        text = ""
        success = False
        attempts_used = 0
        for attempt in range(args.max_regenerations + 1):
            attempts_used = attempt + 1
            raw = backend.generate(messages, args.temperature, args.top_p, args.max_new_tokens, seed=args.seed + attempt)
            text = postprocess_generation(raw, item["prefix"])
            if text and not has_bad_opening(text):
                success = True
                break
            warnings.append({"example_id": item["example_id"], "attempt": attempt, "reason": "empty_or_assistant_opening", "raw": raw[:200]})
        if not success:
            warnings.append(
                {
                    "example_id": item["example_id"],
                    "attempts": attempts_used,
                    "reason": "regen_exhausted",
                }
            )

        rec = {
            "generation_id": f"{args.method}::{item['example_id']}",
            "example_id": item["example_id"],
            "author_id": item["author_id"],
            "client_id": item["author_id"],
            "prompt_id": item["prompt_id"],
            "method": args.method,
            "prefix": item["prefix"],
            "generation": text,
            "gold_continuation": item["target"],
            "model_name": args.generation_model,
            "decode_config": {
                "temperature": args.temperature,
                "top_p": args.top_p,
                "max_new_tokens": args.max_new_tokens,
                "seed": args.seed,
            },
            "sheet_path": sheet_path,
            "persona_path": persona_path,
            "rules_path": str(rules_path) if rules_path else None,
            "retrieved_demo_ids": demo_ids,
            "retrieved_demo_count": len(demo_ids),
            "bm25_no_positive_match": demo_missing_positive,
            "regen_attempts": attempts_used,
            "regen_exhausted": not success,
            "prompt_protocol": PROTOCOL_VERSION,
            "author_conditioning_mode": "profile_prompting",
            "conditioning_text": item["prefix"],
            "reference_continuation_text": item["target"],
            "gold_continuation_text": item["target"],
            "output_text": text,
            "num_output_tokens": word_count(text),
            "source_split": str(item["raw"].get("split", "test")),
            "source_doc_id": item["raw"].get("doc_id"),
            "continuation_id": item["example_id"],
        }
        generations.append(rec)
        print(f"[{idx}/{len(eval_records)}] {item['author_id']} {item['example_id']}: {word_count(text)} words")

    write_jsonl(out / "generations.jsonl", generations)
    quality_filter(generations, out)
    if warnings:
        write_jsonl(out / "generation_warnings.jsonl", warnings)
    write_json(
        out / "leakage_audit.json",
        {
            "passed": leakage_hits == 0 and retrieval_violations == 0,
            "num_eval_examples": len(eval_records),
            "num_retrieval_violations": retrieval_violations,
            "num_bm25_no_positive_match": no_positive_demo_count,
            "num_gold_string_prompt_hits": leakage_hits,
            "retrieval_violation_examples": retrieval_violation_examples,
            "bm25_no_positive_match_examples": bm25_no_positive_match_examples,
            "gold_string_prompt_hit_examples": gold_string_prompt_hit_examples,
            "profile_jsonl": profile_path,
        },
    )
    write_json(
        out / "generation_config.json",
        {
            "method": args.method,
            "backend": args.backend,
            "generation_model": args.generation_model,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "max_new_tokens": args.max_new_tokens,
            "seed": args.seed,
            "eval_jsonl": args.eval_jsonl,
            "profile_jsonl": profile_path,
        },
    )
    print(f"Wrote {len(generations)} generations to {out / 'generations.jsonl'}")


if __name__ == "__main__":
    main()
