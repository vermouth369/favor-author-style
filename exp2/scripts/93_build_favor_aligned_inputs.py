#!/usr/bin/env python3
"""Materialize FAVoR-aligned profile + eval JSONLs for the Mythos baseline.

We pull the prefix/gold continuations FAVoR itself saw (from
``phase1_medium/seed{S}/continuation_roster.json``), filter to the eval-subset
authors (e.g. ``eval_subset_22.json``), derive same-author profile texts from
the K=50 train pool, and run a no-leak filter that drops any train doc whose
text contains any same-author gold continuation as a >= ``--leakage_min_chars``
substring.

Outputs written under ``--output_dir``:
  - profile_train.jsonl  (author-local profile after no-leak filter)
  - eval.jsonl           (flattened roster items; prefix + gold pre-split)
  - split_audit.json     (per-author counts, leak hits, dropped doc summary)

The eval JSONL fields match the format expected by 91_generate_mythos_sheet_baseline.py
when invoked with --prefix_field prefix --target_field gold_continuation.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--continuation_roster",
        required=True,
        help="Path to phase1_medium/seed{S}/continuation_roster.json from FAVoR.",
    )
    p.add_argument(
        "--pooled_train_jsonl",
        required=True,
        help="Path to FAVoR pooled K=50 train.jsonl (e.g. exp1/data/pooled/K=50/train.jsonl).",
    )
    p.add_argument(
        "--eval_subset_json",
        default=None,
        help="Optional eval_subset_22.json restricting roster to those authors.",
    )
    p.add_argument("--output_dir", required=True)
    p.add_argument(
        "--leakage_min_chars",
        type=int,
        default=80,
        help="Drop any train doc whose text contains a same-author gold continuation "
        "as a substring of at least this many characters.",
    )
    p.add_argument(
        "--leakage_min_words",
        type=int,
        default=12,
        help="Skip leak check for golds shorter than this many words; treat them as too short "
        "to be a fair leak signal.",
    )
    return p.parse_args()


def canon_author(value: str | None) -> str:
    """Strip the ``blog_pa_`` prefix that some pipeline stages add and trim."""
    text = str(value or "").strip()
    if text.startswith("blog_pa_"):
        text = text[len("blog_pa_") :]
    return text.lstrip("_")


def load_eval_subset(path: str | None) -> set[str] | None:
    if not path:
        return None
    with open(path, "r", encoding="utf-8") as f:
        d = json.load(f)
    ids = d.get("author_ids") or d.get("authors") or []
    return {canon_author(a) for a in ids if str(a).strip()}


def load_roster_items(roster_path: str, allowed: set[str] | None) -> list[dict]:
    with open(roster_path, "r", encoding="utf-8") as f:
        roster = json.load(f)
    items_by_client = roster.get("items_by_client", {})
    out: list[dict] = []
    for client_id, lst in items_by_client.items():
        for item in lst:
            author_raw = item.get("author_id") or item.get("client_id") or client_id
            author = canon_author(author_raw)
            if allowed is not None and author not in allowed:
                continue
            cont_id = item.get("continuation_id") or f"{author}_{item.get('source_record_idx')}"
            out.append(
                {
                    "example_id": cont_id,
                    "continuation_id": cont_id,
                    "author_id": author,
                    "client_id": item.get("client_id") or client_id,
                    "raw_author_id": author_raw,
                    "prefix": item.get("prefix_text", ""),
                    "gold_continuation": item.get("gold_continuation_text", ""),
                    "prefix_token_count": item.get("prefix_token_count"),
                    "gold_token_count": item.get("gold_token_count"),
                    "total_doc_tokens": item.get("total_doc_tokens"),
                    "source_split": item.get("source_split"),
                    "source_record_idx": item.get("source_record_idx"),
                    "split": "test",
                    "prompt_id": f"continuation::{cont_id}",
                }
            )
    return out


def load_train_records(pool_path: str, allowed: set[str]) -> list[dict]:
    out: list[dict] = []
    with open(pool_path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            author_raw = rec.get("author_id") or rec.get("client_id")
            author = canon_author(author_raw)
            if author not in allowed:
                continue
            doc_id = rec.get("doc_id") or f"{author}_train_{idx:06d}"
            out.append(
                {
                    "example_id": doc_id,
                    "doc_id": doc_id,
                    "author_id": author,
                    "client_id": rec.get("client_id") or author,
                    "raw_author_id": author_raw,
                    "text": rec.get("text", ""),
                    "split": "train",
                }
            )
    return out


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    eval_authors = load_eval_subset(args.eval_subset_json)
    items = load_roster_items(args.continuation_roster, eval_authors)
    if not items:
        raise SystemExit(
            f"No eval items survived after intersecting roster with eval_subset_json "
            f"({args.eval_subset_json}); aborting."
        )

    items_by_author: dict[str, list[dict]] = defaultdict(list)
    for it in items:
        items_by_author[it["author_id"]].append(it)
    item_authors = set(items_by_author)

    profile = load_train_records(args.pooled_train_jsonl, item_authors)

    audit: dict = {
        "leakage_min_chars": args.leakage_min_chars,
        "leakage_min_words": args.leakage_min_words,
        "continuation_roster": str(args.continuation_roster),
        "pooled_train_jsonl": str(args.pooled_train_jsonl),
        "eval_subset_json": str(args.eval_subset_json) if args.eval_subset_json else None,
        "n_eval_authors_requested": len(eval_authors) if eval_authors is not None else None,
        "n_eval_authors_with_items": len(item_authors),
        "n_eval_items": len(items),
        "n_profile_records_pre_filter": len(profile),
        "per_author": {},
    }

    golds_by_author: dict[str, list[str]] = defaultdict(list)
    skipped_short_golds = 0
    for it in items:
        gold = (it.get("gold_continuation") or "").strip()
        if not gold:
            continue
        if (
            len(gold.split()) < args.leakage_min_words
            or len(gold) < args.leakage_min_chars
        ):
            skipped_short_golds += 1
            continue
        golds_by_author[it["author_id"]].append(gold)
    audit["skipped_short_golds"] = skipped_short_golds

    kept: list[dict] = []
    leak_examples_global: list[dict] = []
    for rec in profile:
        author = rec["author_id"]
        text = rec.get("text", "")
        leak_hit: dict | None = None
        for gold in golds_by_author.get(author, []):
            if gold and gold in text:
                leak_hit = {
                    "doc_id": rec["doc_id"],
                    "author_id": author,
                    "matched_gold_chars": len(gold),
                }
                break
        per = audit["per_author"].setdefault(
            author,
            {
                "profile_pre": 0,
                "profile_kept": 0,
                "leaks_dropped": 0,
                "n_eval_items": len(items_by_author.get(author, [])),
                "leak_examples": [],
            },
        )
        per["profile_pre"] += 1
        if leak_hit is not None:
            per["leaks_dropped"] += 1
            if len(per["leak_examples"]) < 3:
                per["leak_examples"].append(leak_hit)
            if len(leak_examples_global) < 10:
                leak_examples_global.append(leak_hit)
        else:
            per["profile_kept"] += 1
            kept.append(rec)

    audit["n_profile_records_post_filter"] = len(kept)
    audit["n_leaks_dropped"] = sum(p["leaks_dropped"] for p in audit["per_author"].values())
    audit["leak_examples_global"] = leak_examples_global
    audit["authors_with_zero_kept_profile"] = sorted(
        a for a, p in audit["per_author"].items() if p["profile_kept"] == 0
    )

    profile_path = out_dir / "profile_train.jsonl"
    eval_path = out_dir / "eval.jsonl"
    audit_path = out_dir / "split_audit.json"

    with open(profile_path, "w", encoding="utf-8") as f:
        for rec in kept:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    with open(eval_path, "w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")
    with open(audit_path, "w", encoding="utf-8") as f:
        json.dump(audit, f, indent=2, ensure_ascii=False)

    print(
        f"profile: {len(kept)} kept / {len(profile)} pre-filter "
        f"({audit['n_leaks_dropped']} leaks dropped)"
    )
    print(f"eval:    {len(items)} items across {len(item_authors)} authors")
    print(f"wrote {profile_path}")
    print(f"wrote {eval_path}")
    print(f"wrote {audit_path}")
    if audit["authors_with_zero_kept_profile"]:
        print(
            f"WARNING: {len(audit['authors_with_zero_kept_profile'])} authors have 0 "
            f"kept profile docs — they will fail the RAG_REQUIRED_METHODS guard in 91. "
            f"Consider lowering --leakage_min_chars or restricting eval to authors with profile."
        )
        print(f"  authors: {audit['authors_with_zero_kept_profile']}")


if __name__ == "__main__":
    main()
