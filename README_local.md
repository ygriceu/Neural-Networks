# Paycom Q2 – Sentiment Classification

This repo contains my training scripts, EDA notebook, and (optionally) the final `submission.csv` used on HackerRank.

## Final model (dev)
- **Model:** `roberta-base`
- **Max len:** 160
- **Epochs:** 3
- **Batch sizes:** train 16, eval 32
- **LR:** 4e-5
- **Label smoothing:** 0.02
- **Oversample factor (label 0 slight):** 1.3
- **Eval every:** 80 steps
- **Seed:** 42  
- **Dev results:** Accuracy 0.865, F1-weighted 0.865, F1-macro 0.851

## Files
- `nn_finetune_steps_smooth.py` – HuggingFace fine-tune with step-based eval/saving, label smoothing, light oversampling.
- `nn_finetune_baseline.py` – Simpler HF baseline.
- `sentiment_baseline.py` – TF-IDF + Linear models baseline (LogReg / LinearSVC).
- `Questions.ipynb` – EDA + brief report (clean outputs before commit to keep repo small).
- `submission.csv` – (optional) final predictions for reference.

## Environment
```bash
python -m venv .venv && source .venv/bin/activate   # or conda create -n pds python=3.9
pip install -r requirements.txt
# Install torch via pip/conda per your OS (see pytorch.org)
