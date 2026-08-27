# FAVoR: Measuring and Mitigating Author-Style Homogenization in Federated Personalized Generation

Official implementation of **FAVoR (Federated Authorial Voice Retention)**, a
shared--private residual PEFT method for retaining author style in federated
personalized generation.

FAVoR aggregates only shared-adapter updates. Each client's author-specific
residual remains local and can be optimized against a frozen Angular Style
Classification Encoder (ASCE) target. The repository contains the training and
evaluation code, paper-aligned configurations, privacy-preserving benchmark
metadata, validation utilities, and artificial toy fixtures.

**Lu Han\*, Jingyao Zhang\*, Katy Ilonka Gero, and Nguyen H. Tran**<br>
School of Computer Science, The University of Sydney<br>
Sydney, NSW 2006, Australia

\* Corresponding authors:
[`lhan0123@sydney.edu.au`](mailto:lhan0123@sydney.edu.au) and
[`jzha0544@sydney.edu.au`](mailto:jzha0544@sydney.edu.au).

Accepted as a Main Conference paper at the
[2026 Conference on Empirical Methods in Natural Language Processing
(EMNLP 2026)](https://2026.emnlp.org/), Budapest, Hungary.

## Repository layout

- `exp1/scripts/`: FAVoR, federated baselines, held-out generation, and metrics.
- `exp1/config/exp1.yaml`: public ASCE/classifier training configuration.
- `exp1/metrics/assistant_phrases.txt`: assistant-style phrase lexicon used by
  the evaluation scripts.
- `exp2/config/`: paper-aligned FAVoR and base-model configurations.
- `exp2/scripts/`: ASCE training and Mythos-Reddit evaluation utilities.
- `benchmark/favor_bench_v02_1/`: anonymized split specifications, protocol
  metadata, schemas, aggregate reports, and validators.
- `examples/toy_run/`: artificial artifacts for a fast validation smoke test.
- `docs/DATA_PREPARATION.md`: local data layout and privacy boundary.

The repository does **not** contain raw BlogText or Mythos-Reddit text,
platform handles, the private mapping from anonymous IDs to source records,
author targets, model checkpoints, adapters, or private human-evaluation data.

## Installation

Python 3.10 and a CUDA-capable environment are recommended for full training.

```bash
conda env create -f environment.yml
conda activate favor
```

Alternatively:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The dependency files record compatible lower bounds. For exact reproduction,
record the resolved package versions and CUDA stack used in your environment.

## Quick validation

The validation path uses only artificial toy data and does not require a GPU:

```bash
bash scripts/run_toy_validation.sh
python benchmark/favor_bench_v02_1/scripts/validate_archive_safety.py \
  benchmark/favor_bench_v02_1
```

## Data preparation

Obtain the raw corpora from their original providers and comply with their
licenses, platform terms, consent constraints, and deletion requests. Then
materialize the local files described in
[`docs/DATA_PREPARATION.md`](docs/DATA_PREPARATION.md). Raw or author-linkable
text must not be committed to this repository.

The checked-in benchmark uses stable anonymous labels such as `client_001` and
`author_001`. The private reverse mapping is intentionally not distributed.

## ASCE

The production authorship ASCE uses:

- `distilroberta-base` (6 Transformer layers, hidden size 768, 12 attention
  heads, feed-forward size 3072);
- attention-mask-aware mean pooling over final-layer token states;
- dropout 0.1, a 768-to-256 linear projection, LayerNorm, and L2
  normalization;
- an ArcFace authorship head with scale `gamma = 30.0` and angular margin
  `m_s = 0.35` radians;
- 3 training epochs over 4,959 training and 742 validation texts from 50
  authors disjoint from the federated roster.

The classifier head is discarded after training and the encoder is frozen
during federated training. Train the public configuration with:

```bash
bash exp2/scripts/train_asce_full_k50.sh
```

The resulting authorship encoder is expected at:

```text
exp1/runs/exp1_asce_full/K=50/author_classifier
```

## FAVoR training

After preparing the local corpora and ASCE artifact:

```bash
python exp1/scripts/44_train_favor.py \
  --config exp2/config/phase1/phase1_medium_seed2026_runtime.yaml \
  --method-name FAVoR
```

Paper-facing settings are summarized in
`exp2/config/phase1/phase1_medium_paper_aligned.yaml`. The runtime template is
provided for seed 2026; the reported experiment uses seeds 2026, 2027, and
2028.

## Baselines

```bash
python exp1/scripts/44_train_fedavg.py --config exp2/config/phase1/phase1_medium_seed2026_runtime.yaml
python exp1/scripts/44_train_fedprox.py --config exp2/config/phase1/phase1_medium_seed2026_runtime.yaml
python exp1/scripts/49_train_ditto_peft.py --config exp2/config/phase1/phase1_medium_seed2026_runtime.yaml
python exp1/scripts/50_train_pfedme_peft.py --config exp2/config/phase1/phase1_medium_seed2026_runtime.yaml
python exp1/scripts/53_train_feddpa_peft.py --config exp2/config/phase1/phase1_medium_seed2026_runtime.yaml
python exp1/scripts/44_train_pooled_peft.py --config exp2/config/phase1/phase1_medium_seed2026_runtime.yaml
python exp1/scripts/45_train_local_only_asce.py --config exp2/config/phase1/phase1_medium_seed2026_runtime.yaml
```

## Held-out continuation evaluation

```bash
python exp1/scripts/46b_generate_exp2_continuation.py \
  --config exp2/config/phase1/phase1_medium_seed2026_runtime.yaml

python exp1/scripts/47b_compute_metrics_continuation.py \
  --config exp2/config/phase1/phase1_medium_seed2026_runtime.yaml
```

The repository includes ASCE-space diagnostics and ASCE-independent external
verification with StyleDistance, MiniLM, and stylometric features. Raw
continuations and author-reference material remain local.

## Citation

If you use this repository, please cite the paper. The ACL Anthology URL, DOI,
and page range will be added after the proceedings record is published.

```bibtex
@inproceedings{han2026favor,
  title     = {{FAVoR}: Measuring and Mitigating Author-Style Homogenization in Federated Personalized Generation},
  author    = {Han, Lu and Zhang, Jingyao and Gero, Katy Ilonka and Tran, Nguyen H.},
  booktitle = {Proceedings of the 2026 Conference on Empirical Methods in Natural Language Processing},
  address   = {Budapest, Hungary},
  publisher = {Association for Computational Linguistics},
  month     = oct,
  year      = {2026},
  note      = {To appear}
}
```

Machine-readable citation metadata are available in
[`CITATION.cff`](CITATION.cff).

## License and data rights

The original code in this repository is released under the MIT License; see
[`LICENSE`](LICENSE). MIT applies to the repository's original software, not
to third-party dependencies or corpora. It does not grant rights to the raw
BlogText or Mythos-Reddit corpora, which are not redistributed here. See
[`benchmark/favor_bench_v02_1/DATA_STATEMENT.md`](benchmark/favor_bench_v02_1/DATA_STATEMENT.md)
for the data-release boundary.
