"""Pretrain tiny flat-BERT and CrossBERT on a WikiText-103 subset; compare loss curves,
wall-clock throughput, and loss-token coverage per step."""

from __future__ import annotations

import json
import math
import os
import time

import torch
import torch.nn.functional as F

from crossbert.data import (
    CACHE_DIR,
    CMSBatcher,
    MLMBatcher,
    encode_corpus,
    iterate_batches,
    load_or_train_tokenizer,
    load_wikitext_lines,
)
from crossbert.model import CrossBERTModel, FlatBERT

CONFIG = dict(
    max_chars=20_000_000,
    vocab_size=8000,
    seq_len=128,
    hidden=256,
    n_heads=4,
    encoder_layers=4,
    predictor_layers=1,
    batch_size=64,
    steps=3000,
    lr=3e-4,
    warmup_steps=200,
    bert_mask_ratio=0.2,
    log_every=200,
)

CKPT_DIR = os.path.join(CACHE_DIR, "checkpoints")


def lr_schedule(step: int, steps: int, warmup: int, base_lr: float) -> float:
    if step < warmup:
        return base_lr * (step + 1) / warmup
    progress = (step - warmup) / max(1, steps - warmup)
    return base_lr * 0.1 + 0.5 * base_lr * 0.9 * (1 + math.cos(math.pi * progress))


def analytic_flops_per_token(cfg: dict) -> dict:
    """Rough per-token forward FLOPs (matmul only): attn ~ 2*S*hidden (per layer, amortized),
    MLP ~ 8*hidden^2 (per layer). Used to sanity-check *why* wall-clock throughput behaves as it does."""
    S, H = cfg["seq_len"], cfg["hidden"]
    attn_per_layer = 2 * S * H
    mlp_per_layer = 8 * H * H
    per_layer = attn_per_layer + mlp_per_layer
    bert_flops = cfg["encoder_layers"] * per_layer
    # CrossBERT: encoder processes visible half (attn over S/2), predictor cross-attends (Tq=Tk=S/2)
    enc_attn = 2 * (S // 2) * H
    enc_layer = enc_attn + mlp_per_layer
    pred_layer = enc_attn + mlp_per_layer  # cross-attn Tq=Tk=S/2, similar order to self-attn over S/2
    crossbert_flops = cfg["encoder_layers"] * enc_layer + cfg["predictor_layers"] * pred_layer
    return {"bert_flops_per_token": bert_flops, "crossbert_flops_per_token": crossbert_flops}


def train_one_model(model, batcher, seqs, cfg, device, is_cms: bool):
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=0.01)
    vocab_size = model.encoder.embed.weight.shape[0] if hasattr(model, "encoder") else None

    losses = []
    real_tokens_seen = 0
    loss_tokens_seen = 0
    step = 0
    t_start = time.time()

    while step < cfg["steps"]:
        for batch in iterate_batches(seqs, cfg["batch_size"]):
            if step >= cfg["steps"]:
                break
            batch = batch.to(device)
            lr = lr_schedule(step, cfg["steps"], cfg["warmup_steps"], cfg["lr"])
            for g in opt.param_groups:
                g["lr"] = lr

            if is_cms:
                out = batcher(batch)
                logits = model(out["visible_ids"], out["visible_pos"], out["masked_pos"])
                loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), out["labels"].reshape(-1))
                loss_tokens_seen += out["labels"].numel()
            else:
                out = batcher(batch)
                logits = model(out["input_ids"], out["position_ids"])
                loss = F.cross_entropy(
                    logits.reshape(-1, logits.shape[-1]), out["labels"].reshape(-1), ignore_index=-100
                )
                loss_tokens_seen += (out["labels"] != -100).sum().item()

            opt.zero_grad()
            loss.backward()
            opt.step()

            real_tokens_seen += batch.numel()
            losses.append(loss.item())
            if step % cfg["log_every"] == 0:
                print(f"  step {step:5d}  loss {loss.item():.3f}  lr {lr:.2e}")
            step += 1

    if device == "mps":
        torch.mps.synchronize()
    elapsed = time.time() - t_start
    return {
        "losses": losses,
        "final_loss": sum(losses[-20:]) / len(losses[-20:]),
        "elapsed_sec": elapsed,
        "real_tokens_seen": real_tokens_seen,
        "tokens_per_sec": real_tokens_seen / elapsed,
        "loss_tokens_seen": loss_tokens_seen,
        "loss_tokens_per_sec": loss_tokens_seen / elapsed,
    }


def main():
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"device: {device}")
    cfg = CONFIG

    print("Loading WikiText-103 subset...")
    lines = load_wikitext_lines(max_chars=cfg["max_chars"])
    tok = load_or_train_tokenizer(lines, vocab_size=cfg["vocab_size"])
    seqs = encode_corpus(tok, lines, seq_len=cfg["seq_len"])
    n_val = max(1, int(0.02 * seqs.shape[0]))
    train_seqs, val_seqs = seqs[:-n_val], seqs[-n_val:]
    print(f"vocab={tok.get_vocab_size()}  train_seqs={train_seqs.shape}  val_seqs={val_seqs.shape}")

    vocab_size = tok.get_vocab_size()
    bert = FlatBERT(vocab_size, cfg["hidden"], cfg["encoder_layers"], cfg["n_heads"])
    cbert = CrossBERTModel(vocab_size, cfg["hidden"], cfg["encoder_layers"], cfg["predictor_layers"], cfg["n_heads"])
    print(f"BERT params: {sum(p.numel() for p in bert.parameters()):,}")
    print(f"CrossBERT params: {sum(p.numel() for p in cbert.parameters()):,}")

    mlm_batcher = MLMBatcher(tok, mask_ratio=cfg["bert_mask_ratio"])
    cms_batcher = CMSBatcher(tok)

    print("\n=== Training BERT (flat, 20% masking) ===")
    bert_stats = train_one_model(bert, mlm_batcher, train_seqs, cfg, device, is_cms=False)

    print("\n=== Training CrossBERT (bipartite, 50% masking + CMS) ===")
    cbert_stats = train_one_model(cbert, cms_batcher, train_seqs, cfg, device, is_cms=True)

    flops = analytic_flops_per_token(cfg)

    summary = {
        "config": cfg,
        "bert": {k: v for k, v in bert_stats.items() if k != "losses"},
        "crossbert": {k: v for k, v in cbert_stats.items() if k != "losses"},
        "analytic_flops_per_token": flops,
        "throughput_speedup_crossbert_over_bert": cbert_stats["tokens_per_sec"] / bert_stats["tokens_per_sec"],
        "loss_token_coverage_ratio_crossbert_over_bert": (
            cbert_stats["loss_tokens_per_sec"] / bert_stats["loss_tokens_per_sec"]
        ),
    }

    print("\n=== Summary ===")
    print(json.dumps(summary, indent=2))

    os.makedirs(CKPT_DIR, exist_ok=True)
    torch.save({"model": bert.state_dict(), "config": cfg, "vocab_size": vocab_size}, os.path.join(CKPT_DIR, "bert.pt"))
    torch.save(
        {"model": cbert.state_dict(), "config": cfg, "vocab_size": vocab_size}, os.path.join(CKPT_DIR, "crossbert.pt")
    )
    with open(os.path.join(CACHE_DIR, "pretrain_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    with open(os.path.join(CACHE_DIR, "loss_curves.json"), "w") as f:
        json.dump({"bert": bert_stats["losses"], "crossbert": cbert_stats["losses"]}, f)
    print(f"\nSaved checkpoints to {CKPT_DIR}")


if __name__ == "__main__":
    main()
