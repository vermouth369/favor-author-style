# Local data preparation

This repository intentionally contains protocol metadata rather than raw
author text. Prepare all corpus-derived files locally and keep them out of Git.

## Privacy boundary

Do not commit raw BlogText or Mythos-Reddit text, author handles, original
document identifiers, the anonymous-ID reverse mapping, author-reference sets,
author targets, generated continuations linked to source authors, or private
human-evaluation keys. The checked-in `.gitignore` excludes `exp1/data/` except
for its documentation.

Obtain each raw corpus from its original provider and follow its applicable
license, platform terms, consent constraints, and deletion requests. This
repository does not redistribute or grant rights to the underlying text.

## Expected layout

From the repository root, the training and evaluation scripts expect this
local-only structure:

```text
exp1/data/
├── raw/
│   ├── blog_authorship_corpus/   # Hugging Face Dataset saved with save_to_disk
│   └── oasst1/                   # needed only for the assistant classifier
├── authors_50.json
├── splits/
│   └── <author_id>.json
├── pooled/
│   └── K=50/
│       ├── train.jsonl
│       ├── val.jsonl
│       └── test.jsonl
└── continuation_roster.json
```

`authors_50.json` has the form:

```json
{"authors": ["<local-author-id>", "<local-author-id>"]}
```

Each `splits/<author_id>.json` stores integer indices into the locally saved
BlogText dataset:

```json
{"train": [0, 1], "val": [2], "test": [3]}
```

Each pooled JSONL record must contain `text` and either `client_id` or
`author_id`. The client identifiers must correspond to the active allocation
in `benchmark/favor_bench_v02_1/non_iid_medium_v02.json`. The public manifest
uses anonymous IDs; the private local mapping is not distributed.

## Continuation roster

`continuation_roster.json` is local because it contains prompt prefixes and
reference continuations. The generation script expects:

```json
{
  "protocol_version": "heldout_continuation_v1",
  "total_items": 1,
  "continuations_per_client": 4,
  "balanced": true,
  "items": [
    {
      "continuation_id": "local_continuation_000001",
      "owner_client_id": "<local-client-id>",
      "owner_author_id": "<local-author-id>",
      "source_split": "test",
      "source_doc_id": "<local-document-id>",
      "prefix_token_count": 96,
      "reference_token_count": 64,
      "prefix_text": "<local prompt prefix>",
      "reference_continuation_text": "<local held-out continuation>"
    }
  ]
}
```

For BlogText, the paper protocol uses four held-out test continuations per
client, a 96-token prefix, at least 64 reference-continuation tokens, and 220
generated tokens. Mythos-Reddit uses a 192-token prefix. See
`benchmark/favor_bench_v02_1/continuation_eval_protocol.yaml` for the complete
protocol.

## ASCE training inputs

`exp1/scripts/07_train_classifiers.py` loads `raw/blog_authorship_corpus` and,
unless `--skip-assistant` is supplied, `raw/oasst1`. To train only the
author-disjoint authorship ASCE:

```bash
cd exp1
python scripts/07_train_classifiers.py \
  --config config/exp1.yaml \
  --skip-assistant
```

The 50 ASCE-training authors must be disjoint from the 50-author federated
roster. Do not commit their raw identifiers or the reverse mapping.

## Validation before use

Run the metadata and artifact checks from the repository root:

```bash
bash scripts/run_toy_validation.sh
python benchmark/favor_bench_v02_1/scripts/validate_archive_safety.py \
  benchmark/favor_bench_v02_1
```
