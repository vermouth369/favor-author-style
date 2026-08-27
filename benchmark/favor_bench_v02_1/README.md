# FAVoR Privacy-Preserving Benchmark Metadata

This directory is the privacy-preserving data and protocol supplement for the
FAVoR paper.

## Included

- Anonymized BlogText client and split specifications using labels such as `client_001`, `author_001`, and `client_001_doc_0001`.
- Non-IID allocation metadata, including the medium 50-client split and 100-client stress-test manifests.
- Paper-aligned held-out continuation protocol metadata for BlogText and Mythos-Reddit.
- FAVoR method configuration for shared adapter aggregation, client-local private residual packs, and ASCE alignment.
- ASCE training/evaluation specification with aggregate counts and validation metrics only.
- JSON schemas, validation scripts, and a tiny artificial toy run for schema checks.

## Excluded

- Raw Blog Authorship / BlogText texts.
- Raw Mythos-Reddit texts, author handles, author-reference sets, author targets, and deletion-sensitive platform data.
- Original author IDs, original document IDs, and the private reverse mapping from anonymous labels to raw-corpus entities.
- Author-linkable generated continuations, human-evaluation private keys, model checkpoints, adapters, server logs, and caches.

## Raw Data Access

Raw corpora must be obtained from their original providers and used only under the relevant licenses, platform terms, consent constraints, and deletion requests. This repository does not redistribute or grant rights to raw text.

## Validation

From this directory:

```bash
python scripts/validate_archive_safety.py .
python scripts/validate_paper_alignment.py .
python scripts/validate_artifacts.py --run_dir fixtures/toy_valid_run
shasum -a 256 -c MANIFEST_SHA256.txt
```
