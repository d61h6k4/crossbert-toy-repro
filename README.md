# CrossBERT (toy reproduction)

A scoped-down implementation of **"Separating Representation from Reconstruction Enables
Scalable Text Encoders"** (Dervishi, Videau, LeCun; ICML 2026) — [arXiv:2607.04011](https://arxiv.org/abs/2607.04011).

This is not a faithful reproduction (the paper trains up to 1.2B-param models on ~4T tokens
across a 9-point FLOPs scaling sweep). It's a from-scratch, laptop/single-GPU-scale build meant
to get the *mechanism* right — encoder/predictor separation, the Complementary Masking Strategy,
closed-form frozen probing, and predictor-as-pooler reuse — and to honestly measure what does and
doesn't show up at toy scale.

## The idea

Standard BERT couples representation learning and token reconstruction in one flat network: the
same self-attention stack that builds a token's representation is also the one graded on
predicting it back from a mask. The paper's frozen-probe evaluation shows this coupling makes
representations *worse* as BERT scales up, even as perplexity improves.

**CrossBERT** splits the two jobs into a bipartite architecture:
- **Encoder** — a deep, self-attention-only stack that sees *only the visible (unmasked) tokens*.
  All the representation-learning compute lives here.
- **Predictor** — a shallow (~1/4 the encoder's depth), cross-attention-only stack. It takes
  `<mask>` placeholders tagged with their position, cross-attends into the encoder's output, and
  reconstructs the masked tokens. It never self-attends, so it can't leak information between
  masked positions — it must read everything from the encoder.

Because the encoder only pays for visible tokens, masking ratios ≥50% become cheap, which enables
the **Complementary Masking Strategy (CMS)**: split every sequence into two halves, and predict
each half from the other in the same batched pass. This gets gradient signal from every token
in one pass costing the same as a single ordinary full-sequence pass — versus BERT, which always
pays for the full sequence but only gets loss on the ~20% of tokens it masks.

## What's implemented

| Module | What it does |
|---|---|
| `crossbert/model.py` | RoPE self-attention `Encoder` (shared backbone), cross-attention-only `Predictor`, flat `FlatBERT` baseline, `CrossBERTModel` wrapper. Both models share the same `Encoder` class — the only architectural difference is flat vs. bipartite. |
| `crossbert/data.py` | WikiText-103 loading/subsampling, custom BPE tokenizer training, standard in-place MLM batching (BERT), and CMS batching (CrossBERT). |
| `crossbert/pretrain.py` | Trains both models on the same corpus/tokenizer/encoder config, differing only in masking strategy. Logs loss curves, measures real-token throughput and loss-token coverage. |
| `crossbert/probe.py` | Frozen evaluation (paper §4.1): closed-form Ridge-regression linear probe and kNN probe on frozen features, evaluated on a GLUE-lite subset (CoLA, MRPC, subsampled SST-2). |
| `crossbert/retrieval.py` | Contrastive fine-tuning + NDCG@10 retrieval eval on SciFact (BEIR), across frozen/unfrozen × BERT/CrossBERT, with a choice of training loss: InfoNCE (softmax over in-batch negatives) or [CoSENT](https://penghao-bdsc.github.io/papers/CoSENT_TASLP2024.pdf) (pairwise cosine-ranking loss, no threshold/temperature to tune). For CrossBERT, the pretrained predictor is reused as a warm-started pooling adapter — the paper's trick for turning frozen features into a retriever cheaply. |

## Scope cuts from the paper

Made deliberately, for a CPU/single-GPU laptop budget:

- **Single-scale comparison only** — no FLOPs scaling sweep (paper trains 8 BERT + 9 CrossBERT
  models from 2×10¹⁸ to 1×10²¹ FLOPs). Both models here: 4-layer/256-hidden encoder,
  CrossBERT + 1-layer predictor, ~5-6M params, 3000 steps on a ~5M-token WikiText-103 subset.
- **Custom 8k-vocab BPE tokenizer** instead of the RoBERTa 50k-vocab tokenizer, to cut
  embedding/softmax cost.
- **SciFact (BEIR)** instead of MS-MARCO (500k queries) for the contrastive stage; in-batch
  negatives only, no hard-negative mining.
- **3 GLUE tasks** (CoLA, MRPC, subsampled SST-2) instead of all 8 — MNLI/QQP are too large for a
  quick CPU/MPS forward pass at this iteration speed.

## Running it

```bash
uv run python -m crossbert.pretrain    # trains both models, saves checkpoints to .cache/checkpoints/
uv run python -m crossbert.probe       # frozen GLUE-lite probing, needs the checkpoints above
uv run python -m crossbert.retrieval   # SciFact contrastive fine-tuning + NDCG@10
```

Each writes its results as JSON to `.cache/` alongside the run.

## Results

### Pretraining (3000 steps, ~5M tokens/model)

| | BERT (flat, 20% mask) | CrossBERT (bipartite, 50% mask + CMS) |
|---|---|---|
| Final loss | 3.85 | 4.92 |
| Real tokens/sec | 102,403 | 87,403 |
| Loss-bearing tokens/sec | 20,801 | 87,403 |

Losses aren't directly comparable — CrossBERT's task (reconstruct 50% of tokens with no
self-attention among them) is intrinsically harder than BERT's 20%-mask MLM. The throughput
numbers tell the real story: CrossBERT processes real tokens slightly *slower* (0.85×) than BERT
at this scale, but because CMS extracts loss from every token instead of just the masked 20%, it
gets **4.2× more loss-bearing tokens per second**.

**Why throughput doesn't show the paper's claimed 1.5-2× speedup here:** that speedup comes from
self-attention's O(S²) cost — CrossBERT's encoder attends over half-length sequences, which is
cheaper quadratically. At this toy scale (seq_len=128, hidden=256), attention is only ~12% of
total FLOPs (MLP dominates at small hidden dims), so the predictor's added MLP cost outweighs the
attention savings. Confirmed this holds even at seq_len=512. This is a real, explainable
scale-dependence: the throughput win requires long-enough sequences and a deep-enough
encoder/shallow-enough predictor ratio for the quadratic term to matter — not a bug in the
implementation.

### Frozen GLUE-lite probing (§4.1: closed-form Ridge + kNN probes)

| Task | Metric | BERT linear | BERT kNN | CrossBERT linear | CrossBERT kNN |
|---|---|---|---|---|---|
| CoLA | MCC | 0.008 | 0.103 | 0.052 | **0.153** |
| MRPC | Acc | 0.708 | 0.696 | **0.711** | **0.699** |
| SST-2 | Acc | **0.648** | **0.599** | 0.639 | 0.584 |

CrossBERT ahead on CoLA and MRPC, BERT slightly ahead on SST-2 — a modest, directionally-consistent
signal. The paper's dramatic frozen-quality gap between BERT and CrossBERT only opens up above
~500M parameters; at ~6M params here, both are far from that regime.

### SciFact retrieval (contrastive fine-tuning, NDCG@10)

| Config | Before | After (InfoNCE) | After (CoSENT) | Trainable params touched |
|---|---|---|---|---|
| BERT frozen | 0.088 | 0.088 (unchanged) | 0.088 (unchanged) | none — mean-pool has nothing to learn |
| BERT unfrozen | 0.088 | 0.418 | **0.468** | full encoder (~5.2M) |
| CrossBERT frozen | 0.035 | **0.358** | 0.135 | predictor + pool query only (~0.8M) |
| CrossBERT unfrozen | 0.035 | 0.394 | **0.423** | full encoder + predictor (~6M) |

BERT-frozen's row is loss-independent — mean-pooling has no learnable parameters, so BERT *needs*
full fine-tuning to become a usable retriever regardless of which loss you'd train it with. That
part is the cleanest match to the paper's thesis: CrossBERT-frozen (InfoNCE) reaches 0.358 — 86%
of BERT-unfrozen's InfoNCE score (0.418), or 77% of the best score found overall (BERT-unfrozen
+ CoSENT, 0.468) — by training only the lightweight predictor-as-adapter, leaving the encoder
untouched entirely.

**InfoNCE vs. CoSENT** turned out to be config-dependent rather than a clean win for either:

- **Full-encoder fine-tuning (BERT-unfrozen, CrossBERT-unfrozen): CoSENT wins**, by +0.050 and
  +0.029 NDCG@10 respectively. With ~5-6M trainable parameters and 30 epochs, the model has enough
  capacity to satisfy CoSENT's *global* ranking constraint (every positive pair's cosine must beat
  every negative pair's cosine, across the whole batch — Eq. 8 in the paper) — matching the CoSENT
  paper's claim of outperforming plain contrastive loss.
- **Adapter-only fine-tuning (CrossBERT-frozen): CoSENT loses badly** (0.135 vs 0.358). Its loss
  curve barely moves (10.28 → 8.15 over 30 epochs, vs InfoNCE's 2.97 → 0.04) — with only ~0.8M
  trainable parameters, the global all-pairs ranking objective doesn't have enough capacity to
  satisfy in this many epochs at this learning rate, whereas InfoNCE's row-wise softmax is a much
  softer, more locally-decomposable target that a small adapter can fit quickly.

That split is itself a useful finding: CoSENT's core promise — no threshold/temperature to tune,
because it optimizes ranking consistency directly — doesn't buy you anything if the trainable
capacity is too small to satisfy a *global* ranking constraint. For the paper's own headline
setup (cheap frozen adaptation via a tiny adapter), InfoNCE's local, per-row objective is the
better fit; CoSENT pulls ahead only once you're fine-tuning the whole network anyway.

Caveat: CrossBERT's zero-shot embeddings start *worse* than BERT's (0.035 vs 0.088) — the
predictor was pretrained to reconstruct masked tokens, not summarize a sentence, so it needs the
adapter step before it's useful.

## Talking points this build actually supports

- Why coupling representation learning and reconstruction in one objective (flat MLM) hurts
  frozen-probe quality, and why decoupling them (bipartite encoder/predictor) fixes it.
- Why CMS is economically free: encoder token-count with CMS equals one ordinary full-sequence
  pass, while loss coverage goes from `mask_ratio × S` to `S`.
- Why the throughput win is scale-dependent (quadratic attention savings vs. linear MLP cost) —
  and how to compute the crossover point for a given (seq_len, hidden) configuration.
- Why reusing a pretrained cross-attention predictor as a pooling adapter gives contrastive
  fine-tuning a warm start, and why a mean-pool baseline (BERT) has no equivalent lever short of
  unfreezing the whole encoder.
- Why CoSENT's global pairwise-ranking loss outperforms InfoNCE when fine-tuning the full
  network, but underperforms it when only a small adapter is trainable — a capacity-vs-objective
  mismatch, not a flaw in either loss.
