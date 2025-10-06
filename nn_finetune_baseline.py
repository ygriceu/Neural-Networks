#!/usr/bin/env python3
# Compatible with: Python 3.9, transformers==4.36.2, tokenizers==0.15.2, accelerate==0.24.1

import os, json, argparse
from datetime import datetime
import numpy as np
import pandas as pd
import torch

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score, accuracy_score, classification_report, confusion_matrix

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
)

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)


# ---------------------- Data helpers ---------------------- #
def load_data(train_path, test_path):
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    if "news" not in train_df.columns or "label" not in train_df.columns:
        raise ValueError("train.csv must contain columns: 'news' and 'label'")
    if "news" not in test_df.columns:
        raise ValueError("test.csv must contain column: 'news'")

    train_df["news"] = train_df["news"].fillna("").astype(str)
    test_df["news"] = test_df["news"].fillna("").astype(str)
    return train_df, test_df


def compute_class_weights(y_np):
    """Return per-class weights as torch.tensor for CrossEntropyLoss."""
    counts = np.bincount(y_np)
    counts[counts == 0] = 1
    weights = counts.sum() / (len(counts) * counts.astype(np.float32))
    return torch.tensor(weights, dtype=torch.float)


# ---------------------- Dataset ---------------------- #
class TextDataset(torch.utils.data.Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        enc = tokenizer(list(texts), truncation=True, padding=True, max_length=max_len)
        self.enc = {k: torch.tensor(v) for k, v in enc.items()}
        self.labels = None if labels is None else torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return self.enc["input_ids"].shape[0]

    def __getitem__(self, idx):
        item = {k: v[idx] for k, v in self.enc.items()}
        if self.labels is not None:
            item["labels"] = self.labels[idx]
        return item


# ---------------------- Metrics ---------------------- #
def make_metrics():
    def compute(eval_pred):
        logits, labels = eval_pred
        preds = logits.argmax(-1)
        return {
            "accuracy": accuracy_score(labels, preds),
            "f1_weighted": f1_score(labels, preds, average="weighted"),
            "f1_macro": f1_score(labels, preds, average="macro"),
        }
    return compute


# ---------------------- Weighted Trainer ---------------------- #
class WeightedTrainer(Trainer):
    def __init__(self, class_weights=None, **kwargs):
        super(WeightedTrainer, self).__init__(**kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        loss_fct = torch.nn.CrossEntropyLoss(weight=self.class_weights.to(logits.device))
        loss = loss_fct(logits, labels)
        if return_outputs:
            return loss, outputs
        return loss


# ---------------------- Main ---------------------- #
def sanitize(s: str) -> str:
    return s.replace("/", "_").replace(":", "_").replace(" ", "_")


def main(args):
    os.makedirs("artifacts_nn", exist_ok=True)
    os.makedirs("runs_nn", exist_ok=True)

    print("[INFO] Torch:", torch.__version__,
          "| MPS available:", getattr(torch.backends, "mps", None) and torch.backends.mps.is_available())

    # ----- Build a unique run tag -----
    auto_tag = f"{sanitize(args.model)}_lr{args.lr}_len{args.max_len}_bs{args.train_bs}_e{args.epochs}"
    tag = args.tag or auto_tag
    if args.timestamp:
        tag += "_" + datetime.now().strftime("%Y%m%d-%H%M%S")
    print(f"[RUN] tag={tag}")

    # Per-run directories
    runs_dir = os.path.join("runs_nn", tag)
    art_dir  = os.path.join("artifacts_nn", tag)
    os.makedirs(runs_dir, exist_ok=True)
    os.makedirs(art_dir,  exist_ok=True)

    # ----- Load & prep data -----
    train_df, test_df = load_data(args.train, args.test)
    label_encoder = LabelEncoder().fit(train_df["label"].values)
    y_all = label_encoder.transform(train_df["label"].values)
    texts_all = train_df["news"].tolist()

    # ----- Reproducible split (shared by default) -----
    if args.split_scope == "run":
        split_path = os.path.join(art_dir, "split.json")
    else:
        split_path = os.path.join("artifacts_nn", "split.json")

    if os.path.exists(split_path) and args.reuse_split:
        idx = json.load(open(split_path))
        train_idx = np.array(idx["train_idx"], dtype=int)
        valid_idx = np.array(idx["valid_idx"], dtype=int)
        print(f"[SPLIT] Reusing indices from {split_path}")
    else:
        all_idx = np.arange(len(texts_all))
        train_idx, valid_idx = train_test_split(
            all_idx, test_size=0.2, stratify=y_all, random_state=SEED
        )
        json.dump({"train_idx": train_idx.tolist(), "valid_idx": valid_idx.tolist()},
                  open(split_path, "w"))
        print(f"[SPLIT] New indices saved to {split_path}")

    X_tr = [texts_all[i] for i in train_idx]
    y_tr = y_all[train_idx]
    X_va = [texts_all[i] for i in valid_idx]
    y_va = y_all[valid_idx]

    # ----- Tokenizer & model -----
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model, num_labels=len(label_encoder.classes_)
    )

    # ----- Datasets -----
    ds_tr = TextDataset(X_tr, y_tr, tokenizer, args.max_len)
    ds_va = TextDataset(X_va, y_va, tokenizer, args.max_len)
    ds_te = TextDataset(test_df["news"].tolist(), None, tokenizer, args.max_len)

    # ----- Class weights -----
    weights = compute_class_weights(y_all)

    # ----- TrainingArguments -----
    targs = TrainingArguments(
        output_dir=runs_dir,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1_weighted",
        greater_is_better=True,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        per_device_train_batch_size=args.train_bs,
        per_device_eval_batch_size=args.eval_bs,
        weight_decay=0.01,
        warmup_ratio=0.06,
        logging_steps=20,
        seed=SEED,
        report_to="none",
        fp16=False,
        save_total_limit=2,  # keep last 2 checkpoints to avoid clutter
    )

    trainer = WeightedTrainer(
        class_weights=weights,
        model=model,
        args=targs,
        train_dataset=ds_tr,
        eval_dataset=ds_va,
        compute_metrics=make_metrics(),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2, early_stopping_threshold=1e-4)],
    )

    # ----- Train & evaluate -----
    trainer.train(resume_from_checkpoint=args.resume_from)
    dev_metrics = trainer.evaluate()
    print("[DEV] metrics:", dev_metrics)

    # ----- Save artifacts (per run) -----
    save_dir = os.path.join(art_dir, "model")
    trainer.save_model(save_dir)
    tokenizer.save_pretrained(save_dir)
    json.dump({"classes": label_encoder.classes_.tolist()},
              open(os.path.join(art_dir, "label_map.json"), "w"), indent=2)
    json.dump(dev_metrics, open(os.path.join(art_dir, "metrics.json"), "w"), indent=2)
    print("[SAVED] model+tokenizer ->", save_dir)

    # ----- Predict test & unique submission path -----
    te_logits = trainer.predict(ds_te).predictions
    te_preds = te_logits.argmax(-1)
    te_labels = label_encoder.inverse_transform(te_preds)

    # default_out is the plain name; we auto-rename to avoid overwrites
    default_out = "submission_nn.csv"
    if args.out == default_out:
        out_path = os.path.join(art_dir, f"submission_{tag}.csv")
    else:
        # if user provided a custom path, leave it as-is
        out_path = args.out

    if args.sample and os.path.exists(args.sample):
        sub = pd.read_csv(args.sample)
        if len(sub) != len(te_labels):
            sub = pd.DataFrame({"news": test_df["news"]})
        sub["label"] = te_labels
    else:
        sub = pd.DataFrame({"news": test_df["news"], "label": te_labels})

    sub.to_csv(out_path, index=False)
    print("[SAVED] submission ->", out_path, "shape:", sub.shape)

    # ----- Dev report for write-up -----
    yhat = trainer.predict(ds_va).predictions.argmax(-1)
    print("\n[REPORT]\n", classification_report(y_va, yhat, digits=4, zero_division=0))
    print("[CONFUSION]\n", confusion_matrix(y_va, yhat))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default="train.csv")
    ap.add_argument("--test", default="test.csv")
    ap.add_argument("--sample", default="sample_submission.csv")
    ap.add_argument("--out", default="submission_nn.csv",
                    help="Submission CSV path. If left as default, a unique per-run file is created.")
    ap.add_argument("--model", default="distilbert-base-uncased")
    ap.add_argument("--max-len", dest="max_len", type=int, default=160)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--train-bs", dest="train_bs", type=int, default=16)
    ap.add_argument("--eval-bs", dest="eval_bs", type=int, default=32)
    ap.add_argument("--lr", type=float, default=4e-5)
    ap.add_argument("--resume-from", dest="resume_from", default=None)
    ap.add_argument("--reuse-split", action="store_true",
                    help="Reuse saved split indices for reproducible comparisons.")
    ap.add_argument("--split-scope", choices=["global", "run"], default="global",
                    help="'global' shares one split across runs; 'run' stores a split per tag.")
    ap.add_argument("--tag", default=None,
                    help="Short run tag (e.g., lr3e-5_len160). Used to name output dirs/files.")
    ap.add_argument("--timestamp", action="store_true",
                    help="Append YYYYMMDD-HHMMSS to the tag for uniqueness.")
    args = ap.parse_args()
    main(args)
