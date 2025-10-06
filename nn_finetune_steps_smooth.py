#!/usr/bin/env python3
# Compatible with: Python 3.9, transformers==4.36.2, tokenizers==0.15.2, accelerate==0.24.1

import os, json, argparse
from datetime import datetime
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    f1_score, accuracy_score, classification_report, confusion_matrix
)

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
    test_df  = pd.read_csv(test_path)

    if "news" not in train_df.columns or "label" not in train_df.columns:
        raise ValueError("train.csv must contain columns: 'news' and 'label'")
    if "news" not in test_df.columns:
        raise ValueError("test.csv must contain column: 'news'")

    train_df["news"] = train_df["news"].fillna("").astype(str)
    test_df["news"]  = test_df["news"].fillna("").astype(str)
    return train_df, test_df


def oversample_label0_indices(y_enc, label_encoder, factor=1.3):
    """
    Slightly oversample the class whose original label is '0' (int or str).
    If not present, oversample the minority class instead.
    factor=1.3 means +30% extra samples for that class (with replacement).
    Returns a list of indices for the oversampled training set.
    """
    y_enc = np.asarray(y_enc)
    classes = list(label_encoder.classes_)
    target_enc = None

    # Try to find encoded id for raw label 0 (int or "0")
    for raw in (0, "0"):
        if raw in classes:
            target_enc = label_encoder.transform([raw])[0]
            break

    # Fallback: minority class
    if target_enc is None:
        counts = np.bincount(y_enc)
        target_enc = int(np.argmin(counts))

    idx_all = np.arange(len(y_enc))
    idx_target = idx_all[y_enc == target_enc]
    if len(idx_target) == 0 or factor <= 1.0:
        return idx_all.tolist()

    n_extra = max(1, int((factor - 1.0) * len(idx_target)))
    extra = np.random.choice(idx_target, size=n_extra, replace=True)
    mixed = np.concatenate([idx_all, extra])
    np.random.shuffle(mixed)
    return mixed.tolist()


def compute_class_weights(y_np):
    # Kept here for reference; not used when oversampling+label smoothing.
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


# ---------------------- Viz helpers ---------------------- #
def plot_curve(log_history, key, out_png, ylabel=None):
    xs, ys = [], []
    for rec in log_history:
        if key in rec and "step" in rec:
            xs.append(rec["step"])
            ys.append(rec[key])
    if not xs:
        return
    plt.figure(figsize=(7,4))
    plt.plot(xs, ys)
    plt.xlabel("step")
    plt.ylabel(ylabel or key)
    plt.title(key)
    plt.tight_layout()
    plt.savefig(out_png, dpi=160)
    plt.close()


def plot_confmat(cm, class_names, out_png):
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    im = ax.imshow(cm, interpolation='nearest')
    ax.figure.colorbar(im, ax=ax)
    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=class_names,
        yticklabels=class_names,
        ylabel='True label',
        xlabel='Predicted label',
        title='Validation Confusion Matrix'
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    # annotate
    fmt = 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    fig.savefig(out_png, dpi=160)
    plt.close(fig)


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
    if args.ls > 0:
        auto_tag += f"_ls{args.ls}"
    if args.oversample_factor and args.oversample_factor > 1.0:
        auto_tag += f"_os{args.oversample_factor}"
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
    split_path = os.path.join(art_dir if args.split_scope=="run" else "artifacts_nn", "split.json")
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
        json.dump({"train_idx": train_idx.tolist(), "valid_idx": valid_idx.tolist()}, open(split_path, "w"))
        print(f"[SPLIT] New indices saved to {split_path}")

    # ----- Build train/valid texts & labels -----
    X_tr = [texts_all[i] for i in train_idx]
    y_tr = y_all[train_idx]
    X_va = [texts_all[i] for i in valid_idx]
    y_va = y_all[valid_idx]

    # ----- Slight oversampling of label '0' on training only -----
    if args.oversample_factor and args.oversample_factor > 1.0:
        enc_for_oversample = LabelEncoder().fit_transform(y_tr)  # y_tr is already int, but ensure 0..K-1
        mixed_idx_local = oversample_label0_indices(enc_for_oversample, label_encoder, factor=args.oversample_factor)
        X_tr = [X_tr[i] for i in mixed_idx_local]
        y_tr = np.array([y_tr[i] for i in mixed_idx_local], dtype=int)
        print(f"[OS] Oversampled train size: {len(y_tr)} (factor={args.oversample_factor})")

    # ----- Tokenizer & model -----
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model, num_labels=len(label_encoder.classes_)
    )

    # ----- Datasets -----
    ds_tr = TextDataset(X_tr, y_tr, tokenizer, args.max_len)
    ds_va = TextDataset(X_va, y_va, tokenizer, args.max_len)
    ds_te = TextDataset(test_df["news"].tolist(), None, tokenizer, args.max_len)

    # ----- TrainingArguments (step-based eval/save + label smoothing) -----
    targs = TrainingArguments(
        output_dir=runs_dir,
        evaluation_strategy="steps",
        save_strategy="steps",
        eval_steps=args.eval_steps,
        save_steps=args.eval_steps,
        load_best_model_at_end=True,
        metric_for_best_model="f1_weighted",
        greater_is_better=True,

        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        per_device_train_batch_size=args.train_bs,
        per_device_eval_batch_size=args.eval_bs,
        weight_decay=0.01,
        warmup_ratio=0.06,
        logging_steps=max(10, args.eval_steps // 4),
        logging_first_step=True,

        seed=SEED,
        report_to="none",
        fp16=False,
        save_total_limit=3,
        label_smoothing_factor=args.ls,   # <<<<<<<<<< Label smoothing
    )

    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=ds_tr,
        eval_dataset=ds_va,
        compute_metrics=make_metrics(),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=args.es_patience, early_stopping_threshold=1e-4)],
    )

    # ----- Train & evaluate -----
    trainer.train(resume_from_checkpoint=args.resume_from)
    dev_metrics = trainer.evaluate()
    print("[DEV] metrics:", dev_metrics)

    # ----- Save artifacts (per run) -----
    save_dir = os.path.join(art_dir, "model")
    trainer.save_model(save_dir)
    tokenizer.save_pretrained(save_dir)
    json.dump({"classes": label_encoder.classes_.tolist()}, open(os.path.join(art_dir, "label_map.json"), "w"), indent=2)
    json.dump(dev_metrics, open(os.path.join(art_dir, "metrics.json"), "w"), indent=2)
    print("[SAVED] model+tokenizer ->", save_dir)

    # ----- Visualizations -----
    # 1) Curves from log history
    log_hist = trainer.state.log_history
    plot_curve(log_hist, "loss", os.path.join(art_dir, "train_loss_curve.png"), ylabel="train loss")
    plot_curve(log_hist, "eval_f1_macro", os.path.join(art_dir, "eval_f1_macro_curve.png"), ylabel="macro-F1")
    plot_curve(log_hist, "eval_f1_weighted", os.path.join(art_dir, "eval_f1_weighted_curve.png"), ylabel="weighted-F1")

    # 2) Confusion matrix on validation
    yhat_va = trainer.predict(ds_va).predictions.argmax(-1)
    cm = confusion_matrix(y_va, yhat_va)
    plot_confmat(cm, [str(c) for c in label_encoder.classes_], os.path.join(art_dir, "confusion_matrix.png"))
    cr_dict = classification_report(y_va, yhat_va, digits=4, zero_division=0, output_dict=True)
    json.dump(cr_dict, open(os.path.join(art_dir, "classification_report.json"), "w"), indent=2)

    # ----- Predict test & unique submission path -----
    te_logits = trainer.predict(ds_te).predictions
    te_preds  = te_logits.argmax(-1)
    te_labels = label_encoder.inverse_transform(te_preds)

    default_out = "submission_nn.csv"
    out_path = os.path.join(art_dir, f"submission_{sanitize(os.path.basename(runs_dir))}.csv") \
        if args.out == default_out else args.out

    if args.sample and os.path.exists(args.sample):
        sub = pd.read_csv(args.sample)
        if len(sub) != len(te_labels):
            sub = pd.DataFrame({"news": test_df["news"]})
        sub["label"] = te_labels
    else:
        sub = pd.DataFrame({"news": test_df["news"], "label": te_labels})

    sub.to_csv(out_path, index=False)
    print("[SAVED] submission ->", out_path, "shape:", sub.shape)

    # Final dev report to stdout
    print("\n[REPORT]\n", classification_report(y_va, yhat_va, digits=4, zero_division=0))
    print("[CONFUSION]\n", cm)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default="train.csv")
    ap.add_argument("--test", default="test.csv")
    ap.add_argument("--sample", default="sample_submission.csv")
    ap.add_argument("--out", default="submission_nn.csv",
                    help="If left as default, a unique per-run file is created under artifacts_nn/<tag>/")
    ap.add_argument("--model", default="distilbert-base-uncased")
    ap.add_argument("--max-len", dest="max_len", type=int, default=128)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--train-bs", dest="train_bs", type=int, default=32)
    ap.add_argument("--eval-bs", dest="eval_bs", type=int, default=64)
    ap.add_argument("--lr", type=float, default=4e-5)

    # New knobs:
    ap.add_argument("--eval-steps", type=int, default=100, help="Evaluate/save every N steps.")
    ap.add_argument("--ls", type=float, default=0.05, help="Label smoothing factor (0 disables).")
    ap.add_argument("--oversample-factor", type=float, default=1.3,
                    help=">1.0 to slightly oversample label '0' on the training split.")
    ap.add_argument("--es-patience", type=int, default=3, help="Early stopping patience (in eval steps).")

    ap.add_argument("--resume-from", dest="resume_from", default=None)
    ap.add_argument("--reuse-split", action="store_true",
                    help="Reuse saved split indices for reproducible comparisons.")
    ap.add_argument("--split-scope", choices=["global", "run"], default="global",
                    help="'global' shares one split across runs; 'run' stores a split per tag.")
    ap.add_argument("--tag", default=None,
                    help="Short run tag (e.g., lr4e-5_len128_bs32). Used to name output dirs/files.")
    ap.add_argument("--timestamp", action="store_true",
                    help="Append YYYYMMDD-HHMMSS to the tag for uniqueness.")
    args = ap.parse_args()
    main(args)
