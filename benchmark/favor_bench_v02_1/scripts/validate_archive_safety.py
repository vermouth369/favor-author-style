#!/usr/bin/env python3
"""Scan the public benchmark metadata for high-risk raw-data leakage patterns."""

from __future__ import annotations

import re
import sys
from pathlib import Path


PATTERNS = {
    "original_blog_client_id": re.compile("blog" + "_pa_" + r"\d+"),
    "numeric_author_id_field": re.compile(r'"(?:author_id|origin_author|authorship_pred_author_id)"\s*:\s*"\d+"'),
    "numeric_source_doc_id": re.compile(r'"source_doc_id"\s*:\s*\d+'),
    "private_key_file_reference": re.compile(r"private_key\.jsonl"),
    "known_raw_source_snippet": re.compile(
        "|".join(
            [
                "robbie" + "writer",
                "Meme" + "Gen",
                "PowerPC" + " G4",
                "h-" + "dropping occurs",
                "billing info" + " has changed",
            ]
        ),
        re.IGNORECASE,
    ),
}

SKIP_NAMES = {"validate_archive_safety.py"}
SKIP_SUFFIXES = {".tgz", ".zip", ".gz", ".png", ".pdf", ".pyc"}


def should_scan(path: Path) -> bool:
    return path.is_file() and path.name not in SKIP_NAMES and path.suffix not in SKIP_SUFFIXES


def main() -> int:
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(".")
    failures = []
    for path in sorted(root.rglob("*")):
        if not should_scan(path):
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        for name, pattern in PATTERNS.items():
            if pattern.search(text):
                failures.append((str(path.relative_to(root)), name))

    if failures:
        print("FAIL: possible privacy leakage patterns found")
        for rel_path, name in failures[:50]:
            print(f"  {rel_path}: {name}")
        if len(failures) > 50:
            print(f"  ... {len(failures) - 50} more")
        return 1

    print("PASS: no high-risk leakage patterns found")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
