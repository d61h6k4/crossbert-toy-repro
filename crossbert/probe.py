"""Frozen evaluation (Section 4.1 of the paper): closed-form Ridge linear probe + kNN probe
on frozen encoder features, evaluated on a GLUE-lite subset (CoLA, MRPC, subsampled SST-2)."""

from __future__ import annotations

import json
import os

import torch
from datasets import load_dataset

from crossbert.data import CACHE_DIR, encode_sentences, load_cached_tokenizer
from crossbert.model import CrossBERTModel, FlatBERT

CKPT_DIR = os.path.join(CACHE_DIR, "checkpoints")
MAX_LEN = 64
LAMBDAS = [1.0, 10.0, 100.0, 1000.0, 10000.0]
KNN_KS = [1, 3, 10, 30]
SST2_TRAIN_SUBSAMPLE = 5000


def accuracy(preds: torch.Tensor, labels: torch.Tensor) -> float:
    return (preds == labels).float().mean().item()


def matthews_corrcoef(preds: torch.Tensor, labels: torch.Tensor) -> float:
    tp = ((preds == 1) & (labels == 1)).sum().item()
    tn = ((preds == 0) & (labels == 0)).sum().item()
    fp = ((preds == 1) & (labels == 0)).sum().item()
    fn = ((preds == 0) & (labels == 1)).sum().item()
    denom = ((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) ** 0.5
    return (tp * tn - fp * fn) / denom if denom > 0 else 0.0


def ridge_fit_predict(X_train, Y_train_onehot, X_val, lam: float) -> torch.Tensor:
    """Eq 2-3: closed-form Ridge regression, W* = (X^T X + lam I)^-1 X^T Y."""
    H = X_train.shape[1]
    XtX = X_train.T @ X_train + lam * torch.eye(H, device=X_train.device)
    XtY = X_train.T @ Y_train_onehot
    W = torch.linalg.solve(XtX, XtY)
    return (X_val @ W).argmax(dim=1)


def knn_predict(X_train, y_train, X_val, k: int, metric: str) -> torch.Tensor:
    """Eq 4: k-NN majority vote."""
    if metric == "cosine":
        Xt = torch.nn.functional.normalize(X_train, dim=1)
        Xv = torch.nn.functional.normalize(X_val, dim=1)
        sim = Xv @ Xt.T
        topk = sim.topk(k, dim=1).indices
    else:  # l2
        dist = torch.cdist(X_val, X_train)
        topk = dist.topk(k, dim=1, largest=False).indices
    neighbor_labels = y_train[topk]  # (n_val, k)
    n_classes = int(y_train.max().item()) + 1
    votes = torch.nn.functional.one_hot(neighbor_labels, n_classes).sum(dim=1)
    return votes.argmax(dim=1)


def best_ridge_score(X_train, y_train, X_val, y_val, metric_fn) -> float:
    n_classes = int(y_train.max().item()) + 1
    Y_onehot = torch.nn.functional.one_hot(y_train, n_classes).float()
    best = -1e9
    for lam in LAMBDAS:
        preds = ridge_fit_predict(X_train, Y_onehot, X_val, lam)
        best = max(best, metric_fn(preds, y_val))
    return best


def best_knn_score(X_train, y_train, X_val, y_val, metric_fn) -> float:
    best = -1e9
    for k in KNN_KS:
        for metric in ["l2", "cosine"]:
            preds = knn_predict(X_train, y_train, X_val, k, metric)
            best = max(best, metric_fn(preds, y_val))
    return best


def load_model(name: str, device: str):
    ckpt = torch.load(os.path.join(CKPT_DIR, f"{name}.pt"), map_location=device)
    cfg, vocab_size = ckpt["config"], ckpt["vocab_size"]
    if name == "bert":
        model = FlatBERT(vocab_size, cfg["hidden"], cfg["encoder_layers"], cfg["n_heads"])
    else:
        model = CrossBERTModel(vocab_size, cfg["hidden"], cfg["encoder_layers"], cfg["predictor_layers"], cfg["n_heads"])
    model.load_state_dict(ckpt["model"])
    model.to(device).eval()
    return model, cfg


@torch.no_grad()
def embed_all(model, tokenizer, texts, text_pairs, device, batch_size=64):
    embs = []
    for i in range(0, len(texts), batch_size):
        chunk = texts[i : i + batch_size]
        pairs_chunk = text_pairs[i : i + batch_size] if text_pairs else None
        input_ids, attn_mask = encode_sentences(tokenizer, chunk, MAX_LEN, pairs_chunk)
        input_ids, attn_mask = input_ids.to(device), attn_mask.to(device)
        embs.append(model.encode_frozen(input_ids, attn_mask).cpu())
    return torch.cat(embs, dim=0)


def load_glue_task(name: str):
    if name == "cola":
        ds = load_dataset("nyu-mll/glue", "cola")
        return ds["train"]["sentence"], None, ds["train"]["label"], ds["validation"]["sentence"], None, ds["validation"]["label"]
    if name == "mrpc":
        ds = load_dataset("nyu-mll/glue", "mrpc")
        return (
            ds["train"]["sentence1"], ds["train"]["sentence2"], ds["train"]["label"],
            ds["validation"]["sentence1"], ds["validation"]["sentence2"], ds["validation"]["label"],
        )
    if name == "sst2":
        ds = load_dataset("nyu-mll/glue", "sst2")
        train = ds["train"].shuffle(seed=0).select(range(SST2_TRAIN_SUBSAMPLE))
        return train["sentence"], None, train["label"], ds["validation"]["sentence"], None, ds["validation"]["label"]
    raise ValueError(name)


def run_probe_task(model, tokenizer, task: str, device: str) -> dict:
    tr_x, tr_x2, tr_y, va_x, va_x2, va_y = load_glue_task(task)
    X_train = embed_all(model, tokenizer, tr_x, tr_x2, device).to(device)
    X_val = embed_all(model, tokenizer, va_x, va_x2, device).to(device)
    y_train = torch.tensor(tr_y, device=device)
    y_val = torch.tensor(va_y, device=device)

    metric_fn = matthews_corrcoef if task == "cola" else accuracy
    return {
        "linear": best_ridge_score(X_train, y_train, X_val, y_val, metric_fn),
        "knn": best_knn_score(X_train, y_train, X_val, y_val, metric_fn),
        "metric": "mcc" if task == "cola" else "accuracy",
        "n_train": len(tr_x),
        "n_val": len(va_x),
    }


def main():
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"device: {device}")

    bert, cfg = load_model("bert", device)
    cbert, _ = load_model("crossbert", device)
    tokenizer = load_cached_tokenizer(cfg["vocab_size"])

    results = {}
    for model_name, model in [("bert", bert), ("crossbert", cbert)]:
        results[model_name] = {}
        for task in ["cola", "mrpc", "sst2"]:
            print(f"\n=== {model_name} / {task} ===")
            r = run_probe_task(model, tokenizer, task, device)
            print(r)
            results[model_name][task] = r

    print("\n=== Frozen probing summary ===")
    print(json.dumps(results, indent=2))
    with open(os.path.join(CACHE_DIR, "probe_results.json"), "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
