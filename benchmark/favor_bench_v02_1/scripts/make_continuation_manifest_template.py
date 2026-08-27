#!/usr/bin/env python3
"""Create local continuation metadata from a raw corpus file.

This template is for users who have separately obtained the raw corpus.
By default it emits only IDs and token counts, not raw text.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def token_count(text: str) -> int:
    return len(text.split())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-jsonl", required=True, help="Local raw JSONL with author_id, document_id, split, text.")
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--dataset", default="BlogText")
    parser.add_argument("--prefix-tokens", type=int, default=96)
    parser.add_argument("--min-continuation-tokens", type=int, default=64)
    parser.add_argument("--include-text", action="store_true", help="For local use only; do not add raw text to the public repository.")
    args = parser.parse_args()

    output = Path(args.output_jsonl)
    output.parent.mkdir(parents=True, exist_ok=True)
    kept = 0
    with Path(args.input_jsonl).open("r", encoding="utf-8") as src, output.open("w", encoding="utf-8") as dst:
        for line in src:
            record = json.loads(line)
            if record.get("split") != "test":
                continue
            tokens = str(record.get("text", "")).split()
            if len(tokens) < args.prefix_tokens + args.min_continuation_tokens:
                continue
            kept += 1
            out = {
                "dataset": args.dataset,
                "continuation_id": f"{args.dataset.lower()}_continuation_{kept:06d}",
                "author_id": record.get("author_id"),
                "document_id": record.get("document_id"),
                "prefix_token_count": args.prefix_tokens,
                "gold_continuation_token_count": len(tokens) - args.prefix_tokens,
                "raw_text_included": bool(args.include_text),
            }
            if args.include_text:
                out["prefix_text"] = " ".join(tokens[: args.prefix_tokens])
                out["gold_continuation_text"] = " ".join(tokens[args.prefix_tokens :])
            dst.write(json.dumps(out, ensure_ascii=False) + "\n")
    print(f"wrote {kept} continuation records to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
