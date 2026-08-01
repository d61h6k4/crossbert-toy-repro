"""WikiText-103 loading, BPE tokenizer training, and MLM/CMS batch construction."""

from __future__ import annotations

import os

import torch
from datasets import load_dataset
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

PAD, MASK, UNK, CLS, SEP = "<pad>", "<mask>", "<unk>", "<cls>", "<sep>"
SPECIAL_TOKENS = [PAD, MASK, UNK, CLS, SEP]

CACHE_DIR = os.path.join(os.path.dirname(__file__), "..", ".cache")


def load_wikitext_lines(max_chars: int = 30_000_000, split: str = "train") -> list[str]:
    """Load WikiText-103 raw text, truncated to a char budget, as non-empty lines."""
    ds = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split=split)
    lines, total = [], 0
    for row in ds:
        text = row["text"].strip()
        if not text or text.startswith("="):  # skip section headers / blank lines
            continue
        lines.append(text)
        total += len(text)
        if total >= max_chars:
            break
    return lines


def train_tokenizer(
    lines: list[str], vocab_size: int = 12_000, save_path: str | None = None
) -> Tokenizer:
    tokenizer = Tokenizer(models.BPE(unk_token=UNK))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(vocab_size=vocab_size, special_tokens=SPECIAL_TOKENS)
    tokenizer.train_from_iterator(lines, trainer=trainer)
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        tokenizer.save(save_path)
    return tokenizer


def load_or_train_tokenizer(lines: list[str], vocab_size: int = 12_000) -> Tokenizer:
    path = os.path.join(CACHE_DIR, f"tokenizer_{vocab_size}.json")
    if os.path.exists(path):
        return Tokenizer.from_file(path)
    return train_tokenizer(lines, vocab_size=vocab_size, save_path=path)


def load_cached_tokenizer(vocab_size: int) -> Tokenizer:
    path = os.path.join(CACHE_DIR, f"tokenizer_{vocab_size}.json")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"No cached tokenizer at {path}; run pretrain.py first."
        )
    return Tokenizer.from_file(path)


def encode_sentences(
    tokenizer: Tokenizer,
    texts: list[str],
    max_len: int,
    text_pairs: list[str] | None = None,
):
    """Tokenize (optionally sentence-pair, joined by <sep>) and pad/truncate to max_len.
    Returns (input_ids, attention_mask) bool tensors of shape (N, max_len)."""
    pad_id = tokenizer.token_to_id(PAD)
    sep_id = tokenizer.token_to_id(SEP)
    all_ids = []
    for i, text in enumerate(texts):
        ids = tokenizer.encode(text).ids
        if text_pairs is not None:
            ids = ids + [sep_id] + tokenizer.encode(text_pairs[i]).ids
        ids = ids[:max_len]
        all_ids.append(ids)

    input_ids = torch.full((len(all_ids), max_len), pad_id, dtype=torch.long)
    attention_mask = torch.zeros((len(all_ids), max_len), dtype=torch.bool)
    for i, ids in enumerate(all_ids):
        input_ids[i, : len(ids)] = torch.tensor(ids, dtype=torch.long)
        attention_mask[i, : len(ids)] = True
    return input_ids, attention_mask


def encode_corpus(tokenizer: Tokenizer, lines: list[str], seq_len: int) -> torch.Tensor:
    """Concatenate encoded lines (with <sep> between docs) and chunk into fixed-length sequences."""
    sep_id = tokenizer.token_to_id(SEP)
    ids: list[int] = []
    for enc in tokenizer.encode_batch(lines):
        ids.extend(enc.ids)
        ids.append(sep_id)
    n_seqs = len(ids) // seq_len
    ids = ids[: n_seqs * seq_len]
    return torch.tensor(ids, dtype=torch.long).view(n_seqs, seq_len)


class MLMBatcher:
    """Builds masked-LM batches for the flat BERT baseline (standard in-place masking)."""

    def __init__(self, tokenizer: Tokenizer, mask_ratio: float = 0.2):
        self.mask_id = tokenizer.token_to_id(MASK)
        self.vocab_size = tokenizer.get_vocab_size()
        self.mask_ratio = mask_ratio

    def __call__(self, seqs: torch.Tensor) -> dict:
        """seqs: (B, S) token ids. Returns corrupted input, position ids, labels (-100 = ignore)."""
        B, S = seqs.shape
        mask_count = round(self.mask_ratio * S)
        device = seqs.device

        perm = torch.argsort(torch.rand(B, S, device=device), dim=1)
        masked_pos_idx = perm[
            :, :mask_count
        ]  # (B, mask_count) positions chosen to be masked

        is_masked = torch.zeros(B, S, dtype=torch.bool, device=device)
        is_masked.scatter_(1, masked_pos_idx, True)

        labels = torch.where(is_masked, seqs, torch.full_like(seqs, -100))

        # standard 80/10/10 corruption
        corrupt_roll = torch.rand(B, S, device=device)
        to_mask_token = is_masked & (corrupt_roll < 0.8)
        to_random_token = is_masked & (corrupt_roll >= 0.8) & (corrupt_roll < 0.9)

        input_ids = seqs.clone()
        input_ids[to_mask_token] = self.mask_id
        random_tokens = torch.randint(0, self.vocab_size, (B, S), device=device)
        input_ids[to_random_token] = random_tokens[to_random_token]

        position_ids = torch.arange(S, device=device).unsqueeze(0).expand(B, S)
        return {"input_ids": input_ids, "position_ids": position_ids, "labels": labels}


class CMSBatcher:
    """Builds Complementary-Masking-Strategy batches for CrossBERT (mask_ratio must be 0.5).

    Splits each sequence's positions into two equal halves (set1, set2). Produces two
    complementary "views" stacked along the batch dimension:
      view A: encoder sees set1 (visible), predictor predicts set2 (masked)
      view B: encoder sees set2 (visible), predictor predicts set1 (masked)
    Total encoder tokens processed == B * S (same cost as one ordinary full-sequence pass),
    while loss is collected on every one of the B * S original tokens.
    """

    def __init__(self, tokenizer: Tokenizer):
        self.vocab_size = tokenizer.get_vocab_size()

    def __call__(self, seqs: torch.Tensor) -> dict:
        B, S = seqs.shape
        assert S % 2 == 0, "seq_len must be even for 50% CMS split"
        half = S // 2
        device = seqs.device

        perm = torch.argsort(torch.rand(B, S, device=device), dim=1)
        set1_pos, set2_pos = perm[:, :half], perm[:, half:]  # each (B, half)

        set1_tok = torch.gather(seqs, 1, set1_pos)
        set2_tok = torch.gather(seqs, 1, set2_pos)

        # View A: visible=set1, masked target=set2 ; View B: visible=set2, masked target=set1
        visible_ids = torch.cat([set1_tok, set2_tok], dim=0)  # (2B, half)
        visible_pos = torch.cat([set1_pos, set2_pos], dim=0)  # (2B, half)
        masked_pos = torch.cat([set2_pos, set1_pos], dim=0)  # (2B, half)
        labels = torch.cat([set2_tok, set1_tok], dim=0)  # (2B, half)

        return {
            "visible_ids": visible_ids,
            "visible_pos": visible_pos,
            "masked_pos": masked_pos,
            "labels": labels,
        }


def iterate_batches(seqs: torch.Tensor, batch_size: int, shuffle: bool = True):
    n = seqs.shape[0]
    order = torch.randperm(n) if shuffle else torch.arange(n)
    for i in range(0, n - batch_size + 1, batch_size):
        yield seqs[order[i : i + batch_size]]
