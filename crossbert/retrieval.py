"""Contrastive fine-tuning + retrieval eval on SciFact (BEIR), a small stand-in for the paper's
MS-MARCO stage. Compares four configurations mirroring Table 3's frozen/unfrozen split:
  - BERT frozen (mean-pool, nothing trainable -> zero-shot frozen features, expected to collapse)
  - BERT unfrozen (full encoder fine-tuned via contrastive loss, mean-pool)
  - CrossBERT frozen (encoder frozen, only the predictor-as-adapter is contrastively trained --
    this is the paper's headline trick: re-using the pretrained predictor as a warm-started pooler)
  - CrossBERT unfrozen (encoder + predictor adapter jointly fine-tuned)

For every trainable config, also compares two training losses:
  - InfoNCE (Eq 5 of the CrossBERT paper): softmax cross-entropy over in-batch negatives.
  - CoSENT (Huang et al., "CoSENT: Consistent Sentence Embedding via Similarity Ranking",
    TASLP 2024): a pairwise ranking loss -- every positive pair's cosine similarity must exceed
    every negative pair's, via a smooth log-sum-exp margin. No threshold/temperature to tune,
    which is the paper's core argument against plain contrastive loss's threshold-sensitivity.
"""

from __future__ import annotations

import json
import math
import os

import torch
import torch.nn.functional as F
from datasets import load_dataset

from crossbert.data import CACHE_DIR, encode_sentences, load_cached_tokenizer
from crossbert.model import CrossBERTModel, FlatBERT

CKPT_DIR = os.path.join(CACHE_DIR, "checkpoints")
MAX_LEN = 128
BATCH_SIZE = 32
EPOCHS = 30
TEMPERATURE = 0.05
COSENT_SCALE = 20.0  # lambda in Eq 8/10 of the CoSENT paper; they use 20 uniformly across datasets
LR_ADAPTER_ONLY = 1e-3
LR_UNFROZEN = 2e-5


def load_scifact():
    corpus = load_dataset("BeIR/scifact", "corpus")["corpus"]
    queries = load_dataset("BeIR/scifact", "queries")["queries"]
    qrels = load_dataset("BeIR/scifact-qrels")

    corpus_text = {row["_id"]: (row["title"] + ". " + row["text"]) for row in corpus}
    query_text = {row["_id"]: row["text"] for row in queries}

    train_pairs = [
        (query_text[str(r["query-id"])], corpus_text[str(r["corpus-id"])])
        for r in qrels["train"]
        if str(r["query-id"]) in query_text and str(r["corpus-id"]) in corpus_text
    ]
    test_qrels: dict[str, set[str]] = {}
    for r in qrels["test"]:
        qid, cid = str(r["query-id"]), str(r["corpus-id"])
        test_qrels.setdefault(qid, set()).add(cid)

    return corpus_text, query_text, train_pairs, test_qrels


def ndcg_at_10(ranked_doc_ids: list[str], relevant: set[str]) -> float:
    dcg = sum(1.0 / math.log2(i + 2) for i, d in enumerate(ranked_doc_ids[:10]) if d in relevant)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(min(10, len(relevant))))
    return dcg / idcg if idcg > 0 else 0.0


def load_model(name: str, device: str):
    ckpt = torch.load(os.path.join(CKPT_DIR, f"{name}.pt"), map_location=device)
    cfg, vocab_size = ckpt["config"], ckpt["vocab_size"]
    if name == "bert":
        model = FlatBERT(vocab_size, cfg["hidden"], cfg["encoder_layers"], cfg["n_heads"])
    else:
        model = CrossBERTModel(vocab_size, cfg["hidden"], cfg["encoder_layers"], cfg["predictor_layers"], cfg["n_heads"])
    model.load_state_dict(ckpt["model"])
    model.to(device)
    return model, cfg


def embed(model, is_crossbert: bool, tokenizer, texts: list[str], device: str, batch_size=64, grad=False):
    out = []
    ctx = torch.enable_grad() if grad else torch.no_grad()
    with ctx:
        for i in range(0, len(texts), batch_size):
            chunk = texts[i : i + batch_size]
            input_ids, attn_mask = encode_sentences(tokenizer, chunk, MAX_LEN)
            input_ids, attn_mask = input_ids.to(device), attn_mask.to(device)
            e = model.pool(input_ids, attn_mask) if is_crossbert else model.encode_frozen(input_ids, attn_mask)
            out.append(e if grad else e.detach().cpu())
    return out if grad else torch.cat(out, dim=0)


def infonce_loss(sims: torch.Tensor) -> torch.Tensor:
    """Eq 5 of the CrossBERT paper: softmax cross-entropy over in-batch negatives.
    sims: (B, B) query-doc cosine similarity matrix; diagonal = matched (positive) pairs."""
    targets = torch.arange(sims.shape[0], device=sims.device)
    return F.cross_entropy(sims / TEMPERATURE, targets)


def cosent_loss(sims: torch.Tensor, scale: float = COSENT_SCALE) -> torch.Tensor:
    """Eq 8 of the CoSENT paper: log(1 + sum_{pos,neg} exp(scale * (cos_neg - cos_pos))).
    Diagonal entries of `sims` are positive pairs, off-diagonal are negative pairs; the loss
    pushes every positive cosine above every negative cosine, ranking-style (no fixed margin)."""
    B = sims.shape[0]
    pos = sims.diag()  # (B,)
    neg = sims[~torch.eye(B, dtype=torch.bool, device=sims.device)]  # (B*(B-1),)
    diffs = scale * (neg.unsqueeze(1) - pos.unsqueeze(0))  # (n_neg, B): every neg vs every pos
    return torch.logsumexp(torch.cat([diffs.reshape(-1), diffs.new_zeros(1)]), dim=0)


def contrastive_finetune(model, is_crossbert: bool, freeze_encoder: bool, tokenizer, train_pairs, device, loss_name: str = "infonce"):
    for p in model.encoder.parameters():
        p.requires_grad_(not freeze_encoder)

    if freeze_encoder and not is_crossbert:
        print("  (BERT frozen + mean-pool: no trainable pooling params -- skipping training)")
        return {"trained": False}

    trainable = [p for p in model.parameters() if p.requires_grad]
    lr = LR_ADAPTER_ONLY if freeze_encoder else LR_UNFROZEN
    opt = torch.optim.AdamW(trainable, lr=lr)

    queries = [q for q, _ in train_pairs]
    docs = [d for _, d in train_pairs]
    n = len(train_pairs)
    losses = []

    model.train()
    for epoch in range(EPOCHS):
        order = torch.randperm(n)
        for i in range(0, n - BATCH_SIZE + 1, BATCH_SIZE):
            idx = order[i : i + BATCH_SIZE].tolist()
            q_batch = [queries[j] for j in idx]
            d_batch = [docs[j] for j in idx]

            q_ids, q_mask = encode_sentences(tokenizer, q_batch, MAX_LEN)
            d_ids, d_mask = encode_sentences(tokenizer, d_batch, MAX_LEN)
            q_ids, q_mask, d_ids, d_mask = q_ids.to(device), q_mask.to(device), d_ids.to(device), d_mask.to(device)

            if is_crossbert:
                q_emb = model.pool(q_ids, q_mask)
                d_emb = model.pool(d_ids, d_mask)
            else:
                q_emb = model.encode_frozen(q_ids, q_mask)
                d_emb = model.encode_frozen(d_ids, d_mask)

            q_emb = F.normalize(q_emb, dim=1)
            d_emb = F.normalize(d_emb, dim=1)
            sims = q_emb @ d_emb.T  # (B, B) in-batch query-doc cosine similarities
            loss = cosent_loss(sims) if loss_name == "cosent" else infonce_loss(sims)

            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(loss.item())
        if epoch % 5 == 0:
            print(f"  epoch {epoch:3d}  loss {sum(losses[-10:]) / len(losses[-10:]):.3f}")
    model.eval()
    for p in model.encoder.parameters():
        p.requires_grad_(True)
    return {"trained": True, "final_loss": sum(losses[-10:]) / len(losses[-10:])}


@torch.no_grad()
def evaluate_retrieval(model, is_crossbert: bool, tokenizer, corpus_text, query_text, test_qrels, device):
    model.eval()
    corpus_ids = list(corpus_text.keys())
    corpus_embs = embed(model, is_crossbert, tokenizer, [corpus_text[c] for c in corpus_ids], device)
    corpus_embs = F.normalize(corpus_embs, dim=1)

    test_qids = [q for q in test_qrels if q in query_text]
    query_embs = embed(model, is_crossbert, tokenizer, [query_text[q] for q in test_qids], device)
    query_embs = F.normalize(query_embs, dim=1)

    sims = query_embs @ corpus_embs.T  # (n_queries, n_corpus)
    top_idx = sims.topk(10, dim=1).indices

    ndcgs = []
    for i, qid in enumerate(test_qids):
        ranked = [corpus_ids[j] for j in top_idx[i].tolist()]
        ndcgs.append(ndcg_at_10(ranked, test_qrels[qid]))
    return sum(ndcgs) / len(ndcgs)


def main():
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"device: {device}")

    print("Loading SciFact...")
    corpus_text, query_text, train_pairs, test_qrels = load_scifact()
    print(f"corpus={len(corpus_text)} queries={len(query_text)} train_pairs={len(train_pairs)} test_queries={len(test_qrels)}")

    bert, cfg = load_model("bert", device)
    cbert, _ = load_model("crossbert", device)
    tokenizer = load_cached_tokenizer(cfg["vocab_size"])

    results = {}
    for model_name, model, is_cb in [("bert", bert, False), ("crossbert", cbert, True)]:
        for freeze in [True, False]:
            trainable = not (freeze and not is_cb)  # BERT-frozen has no trainable pooling params
            loss_names = ["infonce", "cosent"] if trainable else ["infonce"]  # loss choice is moot if nothing trains

            for loss_name in loss_names:
                suffix = f"_{loss_name}" if trainable else ""
                tag = f"{model_name}_{'frozen' if freeze else 'unfrozen'}{suffix}"
                print(f"\n=== {tag}: pre-finetune NDCG@10 ===")
                ndcg_before = evaluate_retrieval(model, is_cb, tokenizer, corpus_text, query_text, test_qrels, device)
                print(f"  {ndcg_before:.4f}")

                print(f"=== {tag}: contrastive fine-tuning ({loss_name}) ===")
                train_info = contrastive_finetune(model, is_cb, freeze, tokenizer, train_pairs, device, loss_name)

                ndcg_after = evaluate_retrieval(model, is_cb, tokenizer, corpus_text, query_text, test_qrels, device)
                print(f"=== {tag}: post-finetune NDCG@10 = {ndcg_after:.4f} ===")

                results[tag] = {"ndcg10_before": ndcg_before, "ndcg10_after": ndcg_after, "loss": loss_name, **train_info}

                # reset weights for next config by reloading checkpoint
                fresh, _ = load_model(model_name, device)
                model.load_state_dict(fresh.state_dict())

    print("\n=== Retrieval summary ===")
    print(json.dumps(results, indent=2))
    with open(os.path.join(CACHE_DIR, "retrieval_results.json"), "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
