#!/usr/bin/env python3
"""Build Mythos-style Author Writing Sheets from author-local profile texts."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from mythos_sheet_utils import (
    CATEGORIES,
    EVIDENCE_TIERS,
    LLMBackend,
    coerce_to_categories,
    config_get,
    get_field,
    load_yaml_config,
    merge_candidate_claims_python,
    parse_json_object,
    read_jsonl,
    safe_id,
    truncate_chars,
    validate_evidence,
    write_json,
)


EXTRACTION_SYSTEM = (
    "You are an expert literary and linguistic analyst. Your task is to infer transferable "
    "writing-style characteristics from an author's past writing. Focus on how the author "
    "writes, not what specific events or facts they mention."
)

MERGE_SYSTEM = "You merge writing-style observations into a compact Author Writing Sheet."

PERSONA_SYSTEM = "You convert an Author Writing Sheet into a concise second-person writing persona for generation."


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default=None, help="Optional YAML config; CLI values take precedence.")
    p.add_argument("--profile_jsonl", default=None)
    p.add_argument("--author_field", default="author_id")
    p.add_argument("--text_field", default="text")
    p.add_argument("--example_id_field", default="example_id")
    p.add_argument("--output_dir", default=None)
    p.add_argument("--sheet_build_mode", choices=["direct", "contrastive"], default="direct")
    p.add_argument("--max_profile_examples_per_author", type=int, default=24)
    p.add_argument("--max_chars_per_profile_example", type=int, default=2500)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--max_claims_per_batch_per_category", type=int, default=3)
    p.add_argument("--max_final_claims_per_category", type=int, default=10)
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--backend", choices=["openai_compatible", "transformers", "mock"], default="openai_compatible")
    p.add_argument("--model_name", default="meta-llama/Llama-3.1-8B-Instruct")
    p.add_argument("--base_url", default=None)
    p.add_argument("--api_key", default=None)
    p.add_argument("--cache_dir", default=".cache/mythos_sheet")
    p.add_argument("--keep_invalid_evidence", action="store_true")
    p.add_argument("--skip_personas", action="store_true")
    p.add_argument(
        "--strict_evidence",
        action="store_true",
        help="Only count exact-substring evidence as valid. Default also accepts whitespace/case/punct/5-gram tiers, "
        "which is essential when the extractor is a small model that paraphrases quotes.",
    )
    p.add_argument(
        "--no_llm_merge",
        action="store_true",
        help="Skip the LLM merge step entirely and use deterministic Python-side dedup. "
        "Recommended when the extractor model is small or unreliable at JSON output.",
    )
    p.add_argument(
        "--max_extract_retries",
        type=int,
        default=1,
        help="If a batch produces no parseable claims, retry the extractor with a JSON-only reminder. "
        "0 disables retries.",
    )
    args = p.parse_args()
    if args.config:
        cfg = load_yaml_config(args.config)
        args.profile_jsonl = args.profile_jsonl or config_get(cfg, "paths", "profile_jsonl")
        args.output_dir = args.output_dir or config_get(cfg, "paths", "output_dir")
        args.author_field = config_get(cfg, "fields", "author_id", default=args.author_field)
        args.text_field = config_get(cfg, "fields", "profile_text", default=args.text_field)
        args.example_id_field = config_get(cfg, "fields", "example_id", default=args.example_id_field)
        args.sheet_build_mode = config_get(cfg, "sheet", "build_mode", default=args.sheet_build_mode)
        args.max_profile_examples_per_author = int(
            config_get(
                cfg,
                "sheet",
                "max_profile_examples_per_author",
                default=args.max_profile_examples_per_author,
            )
        )
        args.max_chars_per_profile_example = int(
            config_get(
                cfg,
                "sheet",
                "max_chars_per_profile_example",
                default=args.max_chars_per_profile_example,
            )
        )
        args.batch_size = int(config_get(cfg, "sheet", "batch_size", default=args.batch_size))
        args.max_claims_per_batch_per_category = int(
            config_get(
                cfg,
                "sheet",
                "max_claims_per_batch_per_category",
                default=args.max_claims_per_batch_per_category,
            )
        )
        args.max_final_claims_per_category = int(
            config_get(
                cfg,
                "sheet",
                "max_final_claims_per_category",
                default=args.max_final_claims_per_category,
            )
        )
        args.seed = int(config_get(cfg, "experiment", "seed", default=args.seed))
        args.backend = config_get(cfg, "generation", "backend", default=args.backend)
        args.model_name = config_get(cfg, "generation", "model", default=args.model_name)
        if not args.strict_evidence:
            args.strict_evidence = bool(config_get(cfg, "sheet", "strict_evidence", default=False))
        if not args.no_llm_merge:
            args.no_llm_merge = bool(config_get(cfg, "sheet", "no_llm_merge", default=False))
        args.max_extract_retries = int(
            config_get(cfg, "sheet", "max_extract_retries", default=args.max_extract_retries)
        )
    if not args.profile_jsonl:
        raise ValueError("Missing --profile_jsonl; pass it directly or via paths.profile_jsonl in --config.")
    if not args.output_dir:
        cfg = load_yaml_config(args.config) if args.config else {}
        runs_dir = config_get(cfg, "paths", "runs_dir")
        if runs_dir:
            args.output_dir = str(Path(runs_dir) / "mythos_sheet")
        else:
            raise ValueError("Missing --output_dir; pass it directly or provide paths.runs_dir in --config.")
    return args


def select_examples(records: list[dict[str, Any]], max_count: int, seed: int) -> list[dict[str, Any]]:
    records = list(records)
    if not records:
        return []
    if any(rec.get("timestamp") for rec in records):
        records.sort(key=lambda rec: str(rec.get("timestamp", "")))
    if len(records) <= max_count:
        return records
    if max_count <= 0:
        return []
    if max_count == 1:
        return [records[0]]
    positions = [
        round(i * (len(records) - 1) / (max_count - 1))
        for i in range(max_count)
    ]
    return [records[i] for i in positions]


def profile_excerpt_block(
    examples: list[dict[str, Any]],
    text_field: str,
    example_id_field: str,
    max_chars: int,
) -> str:
    parts = []
    for rec in examples:
        ex_id = safe_id(get_field(rec, example_id_field, "example_id", "doc_id", default="unknown"))
        text = truncate_chars(str(get_field(rec, text_field, "text", "continuation", default="")), max_chars)
        parts.append(f"[source_example_id={ex_id}]\n{text}")
    return "\n\n".join(parts)


JSON_ONLY_REMINDER = (
    "Reminder: reply with a single JSON object only. "
    "Do NOT emit any text, code fences, markdown, or <think> blocks before or after the JSON. "
    "Begin with `{` and end with `}`. Use exactly the four top-level keys: "
    "structure, creativity, development, language_use."
)


def _build_extract_prompt(examples: list[dict[str, Any]], args: argparse.Namespace) -> str:
    excerpts = profile_excerpt_block(
        examples, args.text_field, args.example_id_field, args.max_chars_per_profile_example
    )
    return f"""We need an Author Writing Sheet for one author.

Extract style characteristics from the profile excerpts below.

Use exactly four categories:
1. structure: discourse organization, opening/closing habits, pacing, paragraph movement, prompt engagement
2. creativity: unusual framing, genre blending, imagery, conceptual moves, topic handling
3. development: stance, emotional movement, character/argument development, degree of introspection
4. language_use: diction, syntax, punctuation, sentence length, rhythm, dialogue, figurative language

For each category, output Claim-Evidence pairs.
- A Claim must be transferable to future prompts (about HOW the author writes).
- Evidence MUST be a verbatim substring (5-25 words) copied EXACTLY from one of the profile excerpts.
  Do not paraphrase, expand contractions, fix typos, or alter punctuation.
  If you cannot find an exact short quote that supports a claim, omit that claim.
- Do not mention the author's name.
- Keep at most {args.max_claims_per_batch_per_category} claims per category.

Output format (CRITICAL — small deviations break downstream parsing):
- Reply with a single JSON object only. No prose, no code fences, no <think> blocks.
- Begin your reply with `{{` and end with `}}`.
- Use exactly the four keys below; if a category has no qualifying claims, return an empty list.

Schema:
{{
  "structure": [{{"claim": "...", "evidence": "...", "source_example_id": "..."}}],
  "creativity": [{{"claim": "...", "evidence": "...", "source_example_id": "..."}}],
  "development": [{{"claim": "...", "evidence": "...", "source_example_id": "..."}}],
  "language_use": [{{"claim": "...", "evidence": "...", "source_example_id": "..."}}]
}}

Profile excerpts:
{excerpts}
"""


def extract_batch_claims(
    backend: LLMBackend,
    examples: list[dict[str, Any]],
    args: argparse.Namespace,
) -> dict[str, list[dict[str, Any]]]:
    user = _build_extract_prompt(examples, args)
    base_messages = [
        {"role": "system", "content": EXTRACTION_SYSTEM},
        {"role": "user", "content": user},
    ]
    last_err: str | None = None
    for attempt in range(max(1, args.max_extract_retries + 1)):
        if attempt == 0:
            messages = base_messages
        else:
            messages = base_messages + [
                {"role": "user", "content": JSON_ONLY_REMINDER},
            ]
        raw = backend.generate(
            messages,
            temperature=0.0,
            top_p=1.0,
            max_tokens=1400,
            seed=args.seed + attempt,
        )
        try:
            parsed = parse_json_object(raw)
        except Exception as exc:
            last_err = f"parse error: {exc}"
            continue
        coerced = coerce_to_categories(parsed, args.max_claims_per_batch_per_category)
        if any(coerced[cat] for cat in CATEGORIES):
            return coerced
        last_err = "parsed but produced 0 claims after schema coercion"
    print(
        f"  WARNING: extract_batch_claims yielded 0 claims after "
        f"{max(1, args.max_extract_retries + 1)} attempt(s): {last_err}"
    )
    return {cat: [] for cat in CATEGORIES}


def merge_claims(
    backend: LLMBackend,
    candidates: dict[str, list[dict[str, Any]]],
    args: argparse.Namespace,
) -> dict[str, list[dict[str, Any]]]:
    if args.no_llm_merge:
        return merge_candidate_claims_python(candidates, args.max_final_claims_per_category)
    if not any(candidates.get(cat) for cat in CATEGORIES):
        # Nothing to merge; skip the LLM call entirely.
        return merge_candidate_claims_python(candidates, args.max_final_claims_per_category)

    user = f"""Merge the following candidate Claim-Evidence pairs.

Rules:
- Keep only transferable style claims.
- Merge duplicate or near-duplicate claims.
- Prefer claims supported by concrete evidence.
- Keep at most {args.max_final_claims_per_category} claims per category.
- Preserve evidence quotes verbatim and the matching source_example_id.
- Do not invent evidence.

Output format (CRITICAL):
- Reply with a single JSON object only. No prose, no code fences, no <think> blocks.
- Begin your reply with `{{` and end with `}}`.
- Use exactly the four keys: structure, creativity, development, language_use.

Candidate claims:
{json.dumps(candidates, ensure_ascii=False, indent=2)}
"""
    raw = backend.generate(
        [{"role": "system", "content": MERGE_SYSTEM}, {"role": "user", "content": user}],
        temperature=0.0,
        top_p=1.0,
        max_tokens=1800,
        seed=args.seed,
    )
    try:
        parsed = parse_json_object(raw)
        coerced = coerce_to_categories(parsed, args.max_final_claims_per_category)
        if any(coerced[cat] for cat in CATEGORIES):
            return coerced
        print(
            "  WARNING: LLM merge returned an empty/unrecognized schema; "
            "using deterministic Python-side dedup."
        )
    except Exception as exc:
        print(
            f"  WARNING: failed to parse merged sheet JSON; using deterministic Python-side dedup: {exc}"
        )
    return merge_candidate_claims_python(candidates, args.max_final_claims_per_category)


def build_persona(backend: LLMBackend, sheet: dict[str, Any], args: argparse.Namespace) -> str:
    user = f"""Given this Author Writing Sheet, write a second-person persona description that helps a language model continue text in this author's style.

Requirements:
- 120-180 words.
- Use second person: "You tend to..."
- Describe style, not biography.
- Do not mention the author's name.
- Do not copy long phrases from evidence.
- Include structure, creativity, development, and language use.
- Avoid generic advice such as "write vividly" unless grounded in the sheet.

Author Writing Sheet:
{json.dumps(sheet, ensure_ascii=False, indent=2)}
"""
    return backend.generate(
        [{"role": "system", "content": PERSONA_SYSTEM}, {"role": "user", "content": user}],
        temperature=0.0,
        top_p=1.0,
        max_tokens=260,
        seed=args.seed,
    ).strip()


def main() -> None:
    args = parse_args()
    if args.sheet_build_mode == "contrastive":
        raise NotImplementedError(
            "sheet_build_mode='contrastive' is reserved for a later implementation; "
            "use 'direct' for the current fair non-FL baseline."
        )
    out = Path(args.output_dir)
    sheets_dir = out / "sheets"
    personas_dir = out / "personas"
    sheets_dir.mkdir(parents=True, exist_ok=True)
    personas_dir.mkdir(parents=True, exist_ok=True)

    backend = LLMBackend(
        args.backend,
        args.model_name,
        cache_dir=args.cache_dir,
        base_url=args.base_url,
        api_key=args.api_key,
    )

    records = read_jsonl(args.profile_jsonl)
    by_author: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for rec in records:
        author = str(get_field(rec, args.author_field, "author_id", "client_id", default=""))
        text = str(get_field(rec, args.text_field, "text", default=""))
        if author and text.strip():
            by_author[author].append(rec)

    reports = []
    total_stats: dict[str, Any] = {
        "total": 0,
        "valid": 0,
        "invalid": 0,
        "tier_counts": {tier: 0 for tier in EVIDENCE_TIERS},
        "source_attribution_corrected": 0,
    }
    authors_with_sheet = 0

    for author_idx, (author_id, author_records) in enumerate(sorted(by_author.items()), start=1):
        selected = select_examples(author_records, args.max_profile_examples_per_author, args.seed + author_idx)
        profile_lookup = {
            safe_id(get_field(rec, args.example_id_field, "example_id", "doc_id", default=f"{author_id}_{i}")): str(
                get_field(rec, args.text_field, "text", default="")
            )
            for i, rec in enumerate(selected)
        }
        candidates = {cat: [] for cat in CATEGORIES}
        for start in range(0, len(selected), args.batch_size):
            batch = selected[start : start + args.batch_size]
            batch_claims = extract_batch_claims(backend, batch, args)
            for cat in CATEGORIES:
                candidates[cat].extend(batch_claims.get(cat, []))

        merged = merge_claims(backend, candidates, args)
        sheet = {
            "author_id": author_id,
            "sheet_version": "mythos_sheet_v1",
            "source_profile_count": len(selected),
            "created_from_split": "train",
            "categories": merged,
        }
        sheet, stats = validate_evidence(
            sheet,
            profile_lookup,
            keep_invalid=args.keep_invalid_evidence,
            strict=args.strict_evidence,
        )
        for key in ("total", "valid", "invalid", "source_attribution_corrected"):
            total_stats[key] += stats.get(key, 0)
        for tier, count in stats.get("tier_counts", {}).items():
            total_stats["tier_counts"][tier] = total_stats["tier_counts"].get(tier, 0) + count
        claim_count = sum(len(sheet["categories"].get(cat, [])) for cat in CATEGORIES)
        if claim_count > 0:
            authors_with_sheet += 1
        write_json(sheets_dir / f"{safe_id(author_id)}.sheet.json", sheet)

        persona_path = None
        if not args.skip_personas:
            persona = build_persona(backend, sheet, args)
            persona_path = personas_dir / f"{safe_id(author_id)}.persona.txt"
            persona_path.write_text(persona + "\n", encoding="utf-8")

        reports.append(
            {
                "author_id": author_id,
                "source_profile_count": len(selected),
                "claim_count": claim_count,
                "evidence_stats": stats,
                "sheet_path": str(sheets_dir / f"{safe_id(author_id)}.sheet.json"),
                "persona_path": str(persona_path) if persona_path else None,
            }
        )
        print(f"[{author_idx}/{len(by_author)}] {author_id}: {claim_count} claims")

    total_claims = sum(rep["claim_count"] for rep in reports)
    avg_by_cat = {}
    for cat in CATEGORIES:
        vals = []
        for rep in reports:
            with open(rep["sheet_path"], "r", encoding="utf-8") as f:
                vals.append(len(json.load(f)["categories"].get(cat, [])))
        avg_by_cat[cat] = sum(vals) / max(len(vals), 1)

    quality = {
        "num_authors": len(by_author),
        "sheet_coverage": authors_with_sheet / max(len(by_author), 1),
        "total_claims": total_claims,
        "valid_evidence_rate": total_stats["valid"] / max(total_stats["total"], 1),
        "invalid_evidence_count": total_stats["invalid"],
        "avg_claims_per_author": total_claims / max(len(by_author), 1),
        "avg_claims_per_category": avg_by_cat,
        "evidence_match_tier_counts": total_stats["tier_counts"],
        "source_attribution_corrected": total_stats["source_attribution_corrected"],
        "evidence_validation_mode": "strict" if args.strict_evidence else "lenient",
        "llm_merge": not args.no_llm_merge,
        "max_extract_retries": args.max_extract_retries,
        "backend": args.backend,
        "model_name": args.model_name,
        "author_reports": reports,
    }
    write_json(out / "sheet_quality_report.json", quality)
    print(f"Wrote sheets to {sheets_dir}")
    print(f"Sheet coverage={quality['sheet_coverage']:.3f}, valid evidence={quality['valid_evidence_rate']:.3f}")


if __name__ == "__main__":
    main()
