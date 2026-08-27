#!/usr/bin/env python3
"""07_train_classifiers.py — Train authorship and assistant-vs-human classifiers.

(A) Authorship classifier: per-K classifier on real author texts.
(B) Assistant classifier: binary classifier trained once (assistant responses vs blog posts).

Supports two backends:
  - legacy_softmax_classifier
  - arcface

Outputs:
  - runs/exp1/K={K}/author_classifier/  (per K)
  - runs/exp1/assistant_classifier/      (shared)
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import random
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from datasets import Dataset, load_from_disk
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader, Dataset as TorchDataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

from style_asce_runtime import (
    ARC_FACE_BACKEND,
    ASCEEncoderModel,
    ArcMarginProduct,
    classifier_artifact_exists,
    fit_binary_calibrator,
    l2_normalize,
    write_asce_artifacts,
)


def compute_metrics_fn(eval_pred):
    """Compute accuracy and macro-F1 for legacy HF Trainer."""
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average="macro", zero_division=0)
    return {"accuracy": acc, "macro_f1": f1}


def train_legacy_classifier(
    train_texts,
    train_labels,
    val_texts,
    val_labels,
    encoder_name,
    num_labels,
    output_dir,
    max_seq_len=256,
    batch_size=32,
    lr=2e-5,
    epochs=5,
    seed=2026,
):
    """Train the existing AutoModelForSequenceClassification backend."""
    os.makedirs(output_dir, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(encoder_name)

    def tokenize_fn(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=max_seq_len,
            padding=False,
        )

    train_ds = Dataset.from_dict({"text": train_texts, "label": train_labels})
    train_ds = train_ds.map(tokenize_fn, batched=True, remove_columns=["text"])
    train_ds.set_format("torch")

    val_ds = None
    if val_texts and val_labels:
        val_ds = Dataset.from_dict({"text": val_texts, "label": val_labels})
        val_ds = val_ds.map(tokenize_fn, batched=True, remove_columns=["text"])
        val_ds.set_format("torch")

    model = AutoModelForSequenceClassification.from_pretrained(
        encoder_name,
        num_labels=num_labels,
    )
    model.config.problem_type = "single_label_classification"

    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=lr,
        num_train_epochs=epochs,
        eval_strategy="epoch" if val_ds else "no",
        save_strategy="epoch",
        load_best_model_at_end=True if val_ds else False,
        metric_for_best_model="macro_f1" if val_ds else None,
        seed=seed,
        report_to="none",
        logging_steps=50,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=compute_metrics_fn,
    )

    trainer.train()
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    results = {"backend": "legacy_softmax_classifier"}
    if val_ds:
        results.update(trainer.evaluate())
    return results


class TextLabelDataset(TorchDataset):
    """Tiny dataset wrapper for manual ArcFace training."""

    def __init__(self, texts: List[str], labels: List[int]) -> None:
        self.texts = list(texts)
        self.labels = [int(label) for label in labels]

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> dict:
        return {
            "text": self.texts[idx],
            "labels": self.labels[idx],
        }


def build_asce_loader(
    texts: List[str],
    labels: List[int],
    tokenizer,
    max_seq_len: int,
    batch_size: int,
    shuffle: bool,
) -> DataLoader:
    """Build a padded torch DataLoader for ArcFace training."""
    dataset = TextLabelDataset(texts, labels)

    def collate_fn(batch):
        batch_texts = [row["text"] for row in batch]
        batch_labels = torch.tensor([row["labels"] for row in batch], dtype=torch.long)
        encoded = tokenizer(
            batch_texts,
            return_tensors="pt",
            truncation=True,
            max_length=max_seq_len,
            padding=True,
        )
        encoded["labels"] = batch_labels
        return encoded

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn,
    )


def move_batch_to_device(batch: dict, device: torch.device) -> dict:
    """Move all tensor batch entries to the target device."""
    out = {}
    for key, value in batch.items():
        if torch.is_tensor(value):
            out[key] = value.to(device)
        else:
            out[key] = value
    return out


def evaluate_asce_model(
    model: ASCEEncoderModel,
    head: ArcMarginProduct,
    dataloader: Optional[DataLoader],
    device: torch.device,
) -> dict:
    """Evaluate ArcFace classifier with cosine prediction."""
    if dataloader is None:
        return {
            "accuracy": None,
            "macro_f1": None,
        }

    model.eval()
    head.eval()
    all_labels = []
    all_preds = []

    with torch.no_grad():
        for batch in dataloader:
            batch = move_batch_to_device(batch, device)
            labels = batch.pop("labels")
            embeddings = model(**batch)
            cosine_scores = head.cosine_scores(embeddings)
            preds = torch.argmax(cosine_scores, dim=-1)
            all_labels.extend(labels.cpu().numpy().tolist())
            all_preds.extend(preds.cpu().numpy().tolist())

    if not all_labels:
        return {
            "accuracy": None,
            "macro_f1": None,
        }

    return {
        "accuracy": float(accuracy_score(all_labels, all_preds)),
        "macro_f1": float(f1_score(all_labels, all_preds, average="macro", zero_division=0)),
    }


def compute_empirical_prototypes(
    model: ASCEEncoderModel,
    dataloader: DataLoader,
    num_labels: int,
    device: torch.device,
) -> np.ndarray:
    """Compute mean normalized embedding prototypes per class."""
    model.eval()
    prototype_sums = None
    prototype_counts = torch.zeros(num_labels, dtype=torch.float32, device=device)

    with torch.no_grad():
        for batch in dataloader:
            batch = move_batch_to_device(batch, device)
            labels = batch.pop("labels")
            embeddings = model(**batch)

            if prototype_sums is None:
                prototype_sums = torch.zeros(
                    num_labels,
                    embeddings.size(-1),
                    dtype=embeddings.dtype,
                    device=device,
                )

            for label_id in torch.unique(labels):
                mask = labels == label_id
                prototype_sums[label_id] += embeddings[mask].sum(dim=0)
                prototype_counts[label_id] += mask.sum()

    if prototype_sums is None:
        return np.zeros((num_labels, model.embedding_dim), dtype=np.float32)

    prototype_counts = prototype_counts.clamp_min(1.0).unsqueeze(-1)
    prototypes = prototype_sums / prototype_counts
    prototypes = l2_normalize(prototypes)
    return prototypes.cpu().numpy().astype(np.float32)


def collect_binary_raw_scores(
    model: ASCEEncoderModel,
    reference_vectors: np.ndarray,
    dataloader: Optional[DataLoader],
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    """Collect binary margins for calibration."""
    if dataloader is None:
        return np.asarray([], dtype=np.float32), np.asarray([], dtype=np.int64)

    model.eval()
    ref = torch.tensor(reference_vectors, dtype=torch.float32, device=device)
    all_scores = []
    all_labels = []

    with torch.no_grad():
        for batch in dataloader:
            batch = move_batch_to_device(batch, device)
            labels = batch.pop("labels")
            embeddings = model(**batch)
            score_matrix = embeddings @ ref.T
            if score_matrix.shape[1] > 1:
                raw_margin = score_matrix[:, 1] - score_matrix[:, 0]
            else:
                raw_margin = score_matrix[:, 0]
            all_scores.extend(raw_margin.cpu().numpy().tolist())
            all_labels.extend(labels.cpu().numpy().tolist())

    return (
        np.asarray(all_scores, dtype=np.float32),
        np.asarray(all_labels, dtype=np.int64),
    )


def train_arcface_classifier(
    train_texts: List[str],
    train_labels: List[int],
    val_texts: List[str],
    val_labels: List[int],
    encoder_name: str,
    num_labels: int,
    output_dir: str,
    task_name: str,
    label_map: Dict[str, str],
    max_seq_len: int = 256,
    batch_size: int = 32,
    lr: float = 2e-5,
    epochs: int = 3,
    seed: int = 2026,
    embedding_dim: int = 256,
    dropout: float = 0.1,
    use_layer_norm: bool = True,
    normalize_embeddings: bool = True,
    scale_s: float = 30.0,
    margin_m: float = 0.35,
    calibrator_cfg: Optional[dict] = None,
) -> dict:
    """Train an ArcFace classifier and persist runtime artifacts."""
    os.makedirs(output_dir, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(encoder_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader = build_asce_loader(
        train_texts,
        train_labels,
        tokenizer=tokenizer,
        max_seq_len=max_seq_len,
        batch_size=batch_size,
        shuffle=True,
    )
    train_eval_loader = build_asce_loader(
        train_texts,
        train_labels,
        tokenizer=tokenizer,
        max_seq_len=max_seq_len,
        batch_size=batch_size,
        shuffle=False,
    )
    val_loader = None
    if val_texts and val_labels:
        val_loader = build_asce_loader(
            val_texts,
            val_labels,
            tokenizer=tokenizer,
            max_seq_len=max_seq_len,
            batch_size=batch_size,
            shuffle=False,
        )

    model = ASCEEncoderModel(
        encoder_name=encoder_name,
        embedding_dim=embedding_dim,
        dropout=dropout,
        use_layer_norm=use_layer_norm,
        normalize_embeddings=normalize_embeddings,
    ).to(device)
    head = ArcMarginProduct(
        embedding_dim=embedding_dim,
        num_classes=num_labels,
        scale_s=scale_s,
        margin_m=margin_m,
    ).to(device)

    optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(head.parameters()),
        lr=lr,
    )

    best_metric = -float("inf")
    best_payload = None

    print(f"  [ArcFace] task={task_name} encoder={encoder_name} classes={num_labels}")
    for epoch_idx in range(epochs):
        model.train()
        head.train()
        running_loss = 0.0
        seen = 0

        for batch in train_loader:
            batch = move_batch_to_device(batch, device)
            labels = batch.pop("labels")

            optimizer.zero_grad()
            embeddings = model(**batch)
            logits, _ = head(embeddings, labels=labels, apply_margin=True)
            loss = F.cross_entropy(logits, labels)
            loss.backward()
            optimizer.step()

            batch_size_actual = int(labels.size(0))
            running_loss += float(loss.detach().cpu().item()) * batch_size_actual
            seen += batch_size_actual

        train_loss = running_loss / max(seen, 1)
        val_metrics = evaluate_asce_model(model, head, val_loader, device)
        model_selection_metric = (
            val_metrics["macro_f1"]
            if val_metrics["macro_f1"] is not None
            else evaluate_asce_model(model, head, train_eval_loader, device)["macro_f1"]
        )

        print(
            f"    epoch {epoch_idx + 1}/{epochs} "
            f"loss={train_loss:.4f} "
            f"val_acc={val_metrics['accuracy'] if val_metrics['accuracy'] is not None else 'NA'} "
            f"val_f1={val_metrics['macro_f1'] if val_metrics['macro_f1'] is not None else 'NA'}"
        )

        if model_selection_metric is not None and model_selection_metric >= best_metric:
            best_metric = float(model_selection_metric)
            best_payload = {
                "model": copy.deepcopy(model.state_dict()),
                "head": copy.deepcopy(head.state_dict()),
                "train_loss": train_loss,
                "val_metrics": val_metrics,
            }

    if best_payload is not None:
        model.load_state_dict(best_payload["model"])
        head.load_state_dict(best_payload["head"])

    class_prototypes = compute_empirical_prototypes(
        model,
        train_eval_loader,
        num_labels=num_labels,
        device=device,
    )
    class_weight_vectors = head.normalized_weights()

    calibrator = None
    calibrator_method = "none"
    if task_name == "assistant" and num_labels == 2 and val_loader is not None:
        calibrator_cfg = calibrator_cfg or {}
        if calibrator_cfg.get("enabled", True):
            calibrator_method = str(calibrator_cfg.get("method", "temperature"))
            raw_scores, raw_labels = collect_binary_raw_scores(
                model,
                reference_vectors=class_prototypes,
                dataloader=val_loader,
                device=device,
            )
            calibrator = fit_binary_calibrator(
                raw_scores=raw_scores,
                labels=raw_labels,
                method=calibrator_method,
            )
            print(
                "    fitted assistant calibrator "
                f"method={calibrator.method} scale={calibrator.scale:.4f} bias={calibrator.bias:.4f}"
            )

    backend_meta = {
        "backend": ARC_FACE_BACKEND,
        "model_type": "style_asce",
        "task": task_name,
        "encoder_name": encoder_name,
        "embedding_dim": int(embedding_dim),
        "dropout": float(dropout),
        "use_layer_norm": bool(use_layer_norm),
        "normalize_embeddings": bool(normalize_embeddings),
        "margin_m": float(margin_m),
        "scale_s": float(scale_s),
        "num_labels": int(num_labels),
        "max_seq_len": int(max_seq_len),
        "pooling": "mean",
        "score_space": "cosine",
        "use_empirical_prototypes_default": True,
        "prototype_source": "empirical_train_mean",
        "calibration_method": calibrator_method,
        "training_seed": int(seed),
    }

    write_asce_artifacts(
        output_dir=output_dir,
        model=model,
        tokenizer=tokenizer,
        backend_meta=backend_meta,
        label_map=label_map,
        class_weight_vectors=class_weight_vectors,
        class_prototypes=class_prototypes,
        calibrator=calibrator,
    )

    final_val_metrics = evaluate_asce_model(model, head, val_loader, device)
    results = {
        "backend": ARC_FACE_BACKEND,
        "accuracy": final_val_metrics.get("accuracy"),
        "macro_f1": final_val_metrics.get("macro_f1"),
        "class_vector_source": "class_prototypes.npy",
        "calibration_method": calibrator_method,
    }
    return results


def stable_author_seed(seed: int, author_id: str) -> int:
    """Avoid Python's randomized hash for deterministic local sampling."""
    return seed + sum(ord(ch) for ch in str(author_id))


def resolve_backend(cfg: dict) -> str:
    """Resolve classifier backend with ArcFace as the new default."""
    backend = str(cfg.get("classifiers", {}).get("backend", ARC_FACE_BACKEND)).strip().lower()
    if backend in {"legacy", "softmax", "hf"}:
        return "legacy_softmax_classifier"
    return ARC_FACE_BACKEND


def build_asce_task_cfg(shared_cfg: dict, task_cfg: dict, defaults: dict) -> dict:
    """Merge shared ArcFace settings with task-specific overrides."""
    return {
        "encoder": task_cfg.get("encoder", shared_cfg.get("encoder", defaults["encoder"])),
        "embedding_dim": int(task_cfg.get("embedding_dim", shared_cfg.get("embedding_dim", defaults["embedding_dim"]))),
        "dropout": float(task_cfg.get("dropout", shared_cfg.get("dropout", defaults["dropout"]))),
        "use_layer_norm": bool(task_cfg.get("use_layer_norm", shared_cfg.get("use_layer_norm", defaults["use_layer_norm"]))),
        "normalize_embeddings": bool(task_cfg.get("normalize_embeddings", shared_cfg.get("normalize_embeddings", defaults["normalize_embeddings"]))),
        "margin_m": float(task_cfg.get("margin_m", defaults["margin_m"])),
        "scale_s": float(task_cfg.get("scale_s", defaults["scale_s"])),
        "max_seq_len": int(task_cfg.get("max_seq_len", defaults["max_seq_len"])),
        "batch_size": int(task_cfg.get("batch_size", defaults["batch_size"])),
        "lr": float(task_cfg.get("lr", defaults["lr"])),
        "epochs": int(task_cfg.get("epochs", defaults["epochs"])),
    }


def main():
    parser = argparse.ArgumentParser(description="Train classifiers for evaluation")
    parser.add_argument("--config", type=str, default="config/exp1.yaml")
    parser.add_argument(
        "--skip-authorship",
        action="store_true",
        help="Skip authorship classifier training",
    )
    parser.add_argument(
        "--skip-assistant",
        action="store_true",
        help="Skip assistant classifier training",
    )
    parser.add_argument(
        "--gpu",
        type=int,
        default=None,
        help="CUDA device index to use (e.g. 0,1,2,3).",
    )
    args = parser.parse_args()

    if args.gpu is not None and "CUDA_VISIBLE_DEVICES" not in os.environ:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    with open(args.config, "r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)

    data_dir = cfg["paths"]["data_dir"]
    raw_dir = os.path.join(data_dir, "raw")
    runs_dir = cfg["paths"].get("runs_dir") or cfg["paths"].get("runs_dir_controls")
    if not runs_dir:
        raise KeyError("Config paths must contain either 'runs_dir' or 'runs_dir_controls'")

    k_values = cfg.get("sweep", {}).get("k_values", [1, 5, 10, 20, 50])
    seed = cfg.get("seeds", {}).get("training", 2026)
    backend = resolve_backend(cfg)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    classifiers_cfg = cfg.get("classifiers", {})
    shared_cfg = classifiers_cfg.get("shared", {})
    auth_cfg = classifiers_cfg["authorship"]
    asst_cfg = classifiers_cfg["assistant"]

    asce_auth_cfg = build_asce_task_cfg(
        shared_cfg,
        auth_cfg,
        defaults={
            "encoder": "distilroberta-base",
            "embedding_dim": 256,
            "dropout": 0.1,
            "use_layer_norm": True,
            "normalize_embeddings": True,
            "margin_m": 0.35,
            "scale_s": 30.0,
            "max_seq_len": int(auth_cfg.get("max_seq_len", 256)),
            "batch_size": int(auth_cfg.get("batch_size", 32)),
            "lr": float(auth_cfg.get("lr", 2e-5)),
            "epochs": int(auth_cfg.get("epochs", 3)),
        },
    )
    asce_asst_cfg = build_asce_task_cfg(
        shared_cfg,
        asst_cfg,
        defaults={
            "encoder": "distilroberta-base",
            "embedding_dim": 256,
            "dropout": 0.1,
            "use_layer_norm": True,
            "normalize_embeddings": True,
            "margin_m": 0.20,
            "scale_s": 16.0,
            "max_seq_len": int(asst_cfg.get("max_seq_len", 256)),
            "batch_size": int(asst_cfg.get("batch_size", 32)),
            "lr": float(asst_cfg.get("lr", 2e-5)),
            "epochs": int(asst_cfg.get("epochs", 3)),
        },
    )

    print("=" * 60)
    print("Step 07 — Training classifiers")
    print("=" * 60)
    print(f"Backend: {backend}")

    if not args.skip_assistant:
        asst_dir = os.path.join(runs_dir, "assistant_classifier")
        if classifier_artifact_exists(asst_dir):
            print("\nAssistant classifier already exists, skipping.")
        else:
            print("\n--- Training assistant-vs-human classifier ---")
            print("Loading OASST1 ...")
            oasst_ds = load_from_disk(os.path.join(raw_dir, "oasst1"))
            if hasattr(oasst_ds, "keys"):
                oasst_data = oasst_ds[list(oasst_ds.keys())[0]]
            else:
                oasst_data = oasst_ds

            assistant_texts = []
            for row in oasst_data:
                lang = row.get("lang", "en")
                role = row.get("role", "")
                if lang == "en" and role == "assistant":
                    text = row.get("text", "")
                    if text and len(text.strip()) >= 50:
                        assistant_texts.append(text)

            print("Loading blog posts for human class ...")
            blog_ds = load_from_disk(os.path.join(raw_dir, "blog_authorship_corpus"))
            if hasattr(blog_ds, "keys"):
                blog_data = blog_ds[list(blog_ds.keys())[0]]
            else:
                blog_data = blog_ds

            human_texts = []
            for row in blog_data:
                text = row.get("text", "")
                if text and len(text.strip()) >= 50:
                    human_texts.append(text)
                if len(human_texts) >= asst_cfg["max_samples_per_class"] * 2:
                    break

            max_n = min(
                asst_cfg["max_samples_per_class"],
                len(assistant_texts),
                len(human_texts),
            )
            rng = random.Random(seed)
            rng.shuffle(assistant_texts)
            rng.shuffle(human_texts)
            assistant_texts = assistant_texts[:max_n]
            human_texts = human_texts[:max_n]

            print(f"  Assistant samples: {len(assistant_texts)}")
            print(f"  Human samples: {len(human_texts)}")

            all_texts = assistant_texts + human_texts
            all_labels = [1] * len(assistant_texts) + [0] * len(human_texts)
            combined = list(zip(all_texts, all_labels))
            rng.shuffle(combined)
            split_idx = int(0.9 * len(combined))
            train_data = combined[:split_idx]
            val_data = combined[split_idx:]

            train_texts = [text for text, _ in train_data]
            train_labels = [label for _, label in train_data]
            val_texts = [text for text, _ in val_data]
            val_labels = [label for _, label in val_data]

            if backend == ARC_FACE_BACKEND:
                results = train_arcface_classifier(
                    train_texts=train_texts,
                    train_labels=train_labels,
                    val_texts=val_texts,
                    val_labels=val_labels,
                    encoder_name=asce_asst_cfg["encoder"],
                    num_labels=2,
                    output_dir=asst_dir,
                    task_name="assistant",
                    label_map={"0": "human", "1": "assistant"},
                    max_seq_len=asce_asst_cfg["max_seq_len"],
                    batch_size=asce_asst_cfg["batch_size"],
                    lr=asce_asst_cfg["lr"],
                    epochs=asce_asst_cfg["epochs"],
                    seed=seed,
                    embedding_dim=asce_asst_cfg["embedding_dim"],
                    dropout=asce_asst_cfg["dropout"],
                    use_layer_norm=asce_asst_cfg["use_layer_norm"],
                    normalize_embeddings=asce_asst_cfg["normalize_embeddings"],
                    scale_s=asce_asst_cfg["scale_s"],
                    margin_m=asce_asst_cfg["margin_m"],
                    calibrator_cfg=asst_cfg.get("calibrator", {}),
                )
            else:
                results = train_legacy_classifier(
                    train_texts=train_texts,
                    train_labels=train_labels,
                    val_texts=val_texts,
                    val_labels=val_labels,
                    encoder_name=asst_cfg["encoder"],
                    num_labels=2,
                    output_dir=asst_dir,
                    max_seq_len=asst_cfg["max_seq_len"],
                    batch_size=asst_cfg["batch_size"],
                    lr=asst_cfg["lr"],
                    epochs=asst_cfg["epochs"],
                    seed=seed,
                )
            print(f"  Assistant classifier val results: {results}")

    if not args.skip_authorship:
        print("\nLoading blog corpus for authorship classifier ...")
        blog_ds = load_from_disk(os.path.join(raw_dir, "blog_authorship_corpus"))
        if hasattr(blog_ds, "keys"):
            blog_data = blog_ds[list(blog_ds.keys())[0]]
        else:
            blog_data = blog_ds

        for k in k_values:
            print(f"\n--- Authorship classifier for K={k} ---")
            author_file = os.path.join(data_dir, f"authors_{k}.json")
            with open(author_file, "r", encoding="utf-8") as handle:
                authors = json.load(handle)["authors"]

            author_dir = os.path.join(runs_dir, f"K={k}", "author_classifier")
            if classifier_artifact_exists(author_dir):
                print("  classifier already exists, skipping.")
                continue

            author_to_idx = {author_id: idx for idx, author_id in enumerate(sorted(authors))}
            label_map = {str(idx): author_id for author_id, idx in author_to_idx.items()}

            train_texts = []
            train_labels = []
            val_texts = []
            val_labels = []

            for author_id in authors:
                split_path = os.path.join(data_dir, "splits", f"{author_id}.json")
                if not os.path.exists(split_path):
                    continue
                with open(split_path, "r", encoding="utf-8") as handle:
                    split_info = json.load(handle)

                label = author_to_idx[author_id]
                train_indices = list(split_info.get("train", []))
                rng_local = random.Random(stable_author_seed(seed, author_id))
                rng_local.shuffle(train_indices)

                count = 0
                for idx in train_indices:
                    if count >= auth_cfg["max_chunks_per_author"]:
                        break
                    row = blog_data[idx]
                    text = row.get("text", "")
                    if text and len(text.strip()) >= 50:
                        train_texts.append(text[:1500])
                        train_labels.append(label)
                        count += 1

                val_indices = split_info.get("val", [])
                for idx in val_indices[:50]:
                    row = blog_data[idx]
                    text = row.get("text", "")
                    if text and len(text.strip()) >= 50:
                        val_texts.append(text[:1500])
                        val_labels.append(label)

            print(f"  Train: {len(train_texts)}, Val: {len(val_texts)}, Classes: {k}")

            if backend == ARC_FACE_BACKEND:
                results = train_arcface_classifier(
                    train_texts=train_texts,
                    train_labels=train_labels,
                    val_texts=val_texts,
                    val_labels=val_labels,
                    encoder_name=asce_auth_cfg["encoder"],
                    num_labels=k,
                    output_dir=author_dir,
                    task_name="authorship",
                    label_map=label_map,
                    max_seq_len=asce_auth_cfg["max_seq_len"],
                    batch_size=asce_auth_cfg["batch_size"],
                    lr=asce_auth_cfg["lr"],
                    epochs=asce_auth_cfg["epochs"],
                    seed=seed,
                    embedding_dim=asce_auth_cfg["embedding_dim"],
                    dropout=asce_auth_cfg["dropout"],
                    use_layer_norm=asce_auth_cfg["use_layer_norm"],
                    normalize_embeddings=asce_auth_cfg["normalize_embeddings"],
                    scale_s=asce_auth_cfg["scale_s"],
                    margin_m=asce_auth_cfg["margin_m"],
                    calibrator_cfg=None,
                )
            else:
                results = train_legacy_classifier(
                    train_texts=train_texts,
                    train_labels=train_labels,
                    val_texts=val_texts,
                    val_labels=val_labels,
                    encoder_name=auth_cfg["encoder"],
                    num_labels=k,
                    output_dir=author_dir,
                    max_seq_len=auth_cfg["max_seq_len"],
                    batch_size=auth_cfg["batch_size"],
                    lr=auth_cfg["lr"],
                    epochs=auth_cfg["epochs"],
                    seed=seed,
                )
                os.makedirs(author_dir, exist_ok=True)
                with open(os.path.join(author_dir, "label_map.json"), "w", encoding="utf-8") as handle:
                    json.dump(label_map, handle, indent=2)

            print(f"  Authorship classifier K={k} results: {results}")

    print("\n✓ Classifier training complete.")


if __name__ == "__main__":
    main()
