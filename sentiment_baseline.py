
# sentiment_baseline.py
# Baseline for tri-class sentiment using provided Paycom files:
#   - train.csv (columns: 'news','label')
#   - test.csv  (column:  'news')
#   - sample_submission.csv (columns: 'news','label')
#
# Usage:
#   python sentiment_baseline.py --train train.csv --test test.csv --sample sample_submission.csv --out submission.csv
#
# Artifacts:
#   - Prints 5-fold CV macro-F1 for TF-IDF + (LogReg, LinearSVC)
#   - Writes 'submission.csv' with columns ['news','label'] (labels predicted as ints)
#   - Saves the selected model pipeline to 'artifacts/baseline_model.joblib'

import argparse, os, re, html, json
import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.preprocessing import LabelEncoder
import joblib

SEED = 42
np.random.seed(SEED)

# --- Minimal, deterministic text cleaner ---
_url = re.compile(r"https?://\\S+|www\\.\\S+")
_html_tag = re.compile(r"<[^>]+>")
_ws = re.compile(r"\\s+")
def clean_text(s: str) -> str:
    if not isinstance(s, str):
        s = "" if pd.isna(s) else str(s)
    s = html.unescape(s)
    s = s.lower()
    s = _html_tag.sub(" ", s)
    s = _url.sub(" URL ", s)
    s = re.sub(r"\\d+", " 0 ", s)      # collapse numbers
    s = re.sub(r"[_\\W]+", " ", s)     # keep only letters/digits -> spaces
    s = _ws.sub(" ", s).strip()
    return s

def identity(x): return x

def load_train(path):
    df = pd.read_csv(path)
    required = {"news","label"}
    if not required.issubset(df.columns):
        raise SystemExit(f"train.csv must contain {required}, found {set(df.columns)}")
    df = df.dropna(subset=["label"]).copy()
    df["news"] = df["news"].fillna("").astype(str)
    print(f"[INFO] train: {len(df)} rows")
    print("[INFO] label distribution:")
    print(df["label"].value_counts(normalize=True).round(3))
    return df

def load_test(path):
    df = pd.read_csv(path)
    if "news" not in df.columns:
        raise SystemExit("test.csv must contain a 'news' column")
    df["news"] = df["news"].fillna("").astype(str)
    print(f"[INFO] test: {len(df)} rows")
    return df

def build_models():
    tfidf = TfidfVectorizer(
        preprocessor=identity,   # <= use top-level function, not a lambda
        tokenizer=None,
        ngram_range=(1, 2),
        min_df=2,
        max_features=120_000,
        sublinear_tf=True,
        lowercase=False          # we already lowercased in clean_text()
    )
    models = {
        "tfidf_logreg": Pipeline([
            ("tfidf", tfidf),
            ("clf", LogisticRegression(
                max_iter=2000, class_weight="balanced",
                solver="liblinear", random_state=SEED
            ))
        ]),
        "tfidf_linear_svm": Pipeline([
            ("tfidf", tfidf),
            ("clf", LinearSVC(C=1.0, class_weight="balanced", random_state=SEED))
        ]),
    }
    return models

def cross_val_report(name, pipe, X, y, k=5):
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=SEED)
    f1s = []
    for fold, (tr, va) in enumerate(skf.split(X, y), 1):
        pipe.fit(X[tr], y[tr])
        yhat = pipe.predict(X[va])
        f1 = f1_score(y[va], yhat, average="macro")
        f1s.append(f1)
        print(f"[{name}] Fold {fold}/{k} macro-F1: {f1:.4f}")
    print(f"[{name}] CV macro-F1: mean={np.mean(f1s):.4f}  std={np.std(f1s):.4f}")
    return float(np.mean(f1s))

def main(args):
    # Load data
    train_df = load_train(args.train)
    test_df = load_test(args.test)

    # Encode labels (supports string labels too)
    le = LabelEncoder().fit(train_df["label"].values)
    y = le.transform(train_df["label"].values)

    # Clean text
    X = train_df["news"].map(clean_text).values
    X_test = test_df["news"].map(clean_text).values

    # Build and evaluate models
    models = build_models()
    cv_scores = {}
    print("\\n========== 5-FOLD STRATIFIED CV ==========")
    for name, model in models.items():
        cv_scores[name] = cross_val_report(name, model, X, y, k=args.folds)

    # Select best, refit on full train
    best_name = max(cv_scores, key=cv_scores.get)
    best_model = models[best_name]
    print(f"\\n[SELECT] Best baseline: {best_name} (macro-F1={cv_scores[best_name]:.4f})")
    best_model.fit(X, y)

    # Quick full-train evaluation report
    yhat_full = best_model.predict(X)
    print("\\n[TRAIN-FIT] classification report (on training data, for sanity only):")
    print(classification_report(y, yhat_full, digits=4, zero_division=0, target_names=le.classes_.astype(str)))

    # Predict test and build submission matching sample_submission structure
    test_pred = best_model.predict(X_test)
    pred_labels = le.inverse_transform(test_pred)  # back to original label space

    if args.sample and os.path.exists(args.sample):
        sub_df = pd.read_csv(args.sample)
        if "label" not in sub_df.columns or "news" not in sub_df.columns:
            # Fall back to simple format
            sub_df = test_df.copy()
            sub_df["label"] = pred_labels
        else:
            # Replace label column, keep news as-is (order must match test.csv)
            sub_df = sub_df.copy()
            if len(sub_df) != len(test_df):
                print("[WARN] sample_submission length != test length; rebuilding submission from test.csv")
                sub_df = test_df.copy()
            sub_df["label"] = pred_labels
    else:
        sub_df = test_df.copy()
        sub_df["label"] = pred_labels

    # Ensure int dtype if labels are 0/1/2
    try:
        sub_df["label"] = sub_df["label"].astype(int)
    except Exception:
        pass

    # Save submission
    out_path = args.out
    sub_df.to_csv(out_path, index=False)
    print(f"\\n[SAVED] Submission written to: {out_path} (shape={sub_df.shape})")
    print(sub_df.head())

    # Save artifacts
    os.makedirs("artifacts", exist_ok=True)
    joblib.dump(best_model, os.path.join("artifacts", "baseline_model.joblib"))
    with open(os.path.join("artifacts", "metadata.json"), "w") as f:
        json.dump({
            "best_model": best_name,
            "cv_macro_f1": cv_scores[best_name],
            "label_classes": le.classes_.tolist(),
            "seed": SEED
        }, f, indent=2)
    print("[SAVED] Model and metadata saved in ./artifacts")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default="train.csv")
    ap.add_argument("--test", default="test.csv")
    ap.add_argument("--sample", default="sample_submission.csv")
    ap.add_argument("--out", default="submission.csv")
    ap.add_argument("--folds", type=int, default=5)
    args = ap.parse_args()
    main(args)
