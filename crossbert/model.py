"""RoPE self-attention Encoder (shared backbone), cross-attention-only Predictor (CrossBERT),
and a flat MLM head (BERT baseline). Both models share the same Encoder implementation so the
only architectural difference is flat (self-attn over full sequence) vs bipartite (encoder sees
only visible tokens, predictor cross-attends to reconstruct masked ones).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


def apply_rope(x: torch.Tensor, position_ids: torch.Tensor, inv_freq: torch.Tensor) -> torch.Tensor:
    """x: (B, H, T, Dh); position_ids: (B, T); inv_freq: (Dh/2,)."""
    freqs = position_ids.unsqueeze(-1).float() * inv_freq.view(1, 1, -1)  # (B, T, Dh/2)
    emb = torch.cat([freqs, freqs], dim=-1)  # (B, T, Dh)
    cos = emb.cos().unsqueeze(1)  # (B, 1, T, Dh)
    sin = emb.sin().unsqueeze(1)
    return x * cos + rotate_half(x) * sin


def masked_mean_pool(hidden: torch.Tensor, attention_mask: torch.Tensor | None) -> torch.Tensor:
    if attention_mask is None:
        return hidden.mean(dim=1)
    mask = attention_mask.unsqueeze(-1).float()
    return (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)


def init_weights(module: nn.Module) -> None:
    if isinstance(module, nn.Linear):
        nn.init.normal_(module.weight, mean=0.0, std=0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, mean=0.0, std=0.02)


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return norm * self.weight


class MLP(nn.Module):
    def __init__(self, hidden: int, mult: int = 4):
        super().__init__()
        self.fc1 = nn.Linear(hidden, hidden * mult)
        self.fc2 = nn.Linear(hidden * mult, hidden)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.gelu(self.fc1(x)))


class SelfAttention(nn.Module):
    def __init__(self, hidden: int, n_heads: int, rope_base: float = 10000.0):
        super().__init__()
        assert hidden % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = hidden // n_heads
        self.qkv = nn.Linear(hidden, 3 * hidden)
        self.out = nn.Linear(hidden, hidden)
        inv_freq = 1.0 / (rope_base ** (torch.arange(0, self.head_dim, 2).float() / self.head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, x: torch.Tensor, position_ids: torch.Tensor, key_padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        B, T, H = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        q = apply_rope(q, position_ids, self.inv_freq)
        k = apply_rope(k, position_ids, self.inv_freq)
        attn_mask = None
        if key_padding_mask is not None:
            attn_mask = key_padding_mask.view(B, 1, 1, T)  # True = attend
        attn_out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        attn_out = attn_out.transpose(1, 2).reshape(B, T, H)
        return self.out(attn_out)


class CrossAttention(nn.Module):
    """Query stream cross-attends into a (separate) key/value stream. No query-query self-attention."""

    def __init__(self, hidden: int, n_heads: int, rope_base: float = 10000.0):
        super().__init__()
        assert hidden % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = hidden // n_heads
        self.q_proj = nn.Linear(hidden, hidden)
        self.kv_proj = nn.Linear(hidden, 2 * hidden)
        self.out = nn.Linear(hidden, hidden)
        inv_freq = 1.0 / (rope_base ** (torch.arange(0, self.head_dim, 2).float() / self.head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(
        self,
        query_stream: torch.Tensor,
        query_pos: torch.Tensor,
        kv_stream: torch.Tensor,
        kv_pos: torch.Tensor,
        kv_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        B, Tq, H = query_stream.shape
        Tk = kv_stream.shape[1]
        q = self.q_proj(query_stream).view(B, Tq, self.n_heads, self.head_dim).transpose(1, 2)
        k, v = self.kv_proj(kv_stream).chunk(2, dim=-1)
        k = k.view(B, Tk, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, Tk, self.n_heads, self.head_dim).transpose(1, 2)
        q = apply_rope(q, query_pos, self.inv_freq)
        k = apply_rope(k, kv_pos, self.inv_freq)
        attn_mask = None
        if kv_padding_mask is not None:
            attn_mask = kv_padding_mask.view(B, 1, 1, Tk)  # True = attend
        attn_out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        attn_out = attn_out.transpose(1, 2).reshape(B, Tq, H)
        return self.out(attn_out)


class SelfAttnBlock(nn.Module):
    def __init__(self, hidden: int, n_heads: int):
        super().__init__()
        self.norm1 = RMSNorm(hidden)
        self.attn = SelfAttention(hidden, n_heads)
        self.norm2 = RMSNorm(hidden)
        self.mlp = MLP(hidden)

    def forward(self, x: torch.Tensor, position_ids: torch.Tensor, key_padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), position_ids, key_padding_mask)
        x = x + self.mlp(self.norm2(x))
        return x


class CrossAttnBlock(nn.Module):
    def __init__(self, hidden: int, n_heads: int):
        super().__init__()
        self.norm_q = RMSNorm(hidden)
        self.norm_kv = RMSNorm(hidden)
        self.attn = CrossAttention(hidden, n_heads)
        self.norm2 = RMSNorm(hidden)
        self.mlp = MLP(hidden)

    def forward(
        self,
        query_stream: torch.Tensor,
        query_pos: torch.Tensor,
        kv_stream: torch.Tensor,
        kv_pos: torch.Tensor,
        kv_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        q = query_stream + self.attn(self.norm_q(query_stream), query_pos, self.norm_kv(kv_stream), kv_pos, kv_padding_mask)
        q = q + self.mlp(self.norm2(q))
        return q


class Encoder(nn.Module):
    """Shared backbone: self-attention over whatever tokens it's given (visible-only for
    CrossBERT, full sequence for flat BERT)."""

    def __init__(self, vocab_size: int, hidden: int, n_layers: int, n_heads: int):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden)
        self.blocks = nn.ModuleList([SelfAttnBlock(hidden, n_heads) for _ in range(n_layers)])
        self.final_norm = RMSNorm(hidden)
        self.apply(init_weights)

    def forward(
        self, token_ids: torch.Tensor, position_ids: torch.Tensor, attention_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        x = self.embed(token_ids)
        for block in self.blocks:
            x = block(x, position_ids, attention_mask)
        return self.final_norm(x)


class Predictor(nn.Module):
    """Lightweight cross-attention-only stack. Reconstructs masked tokens from encoder output,
    and (in CrossBERTModel.pool) doubles as a re-usable pooling adapter."""

    def __init__(self, hidden: int, n_layers: int, n_heads: int, vocab_size: int, tie_weights: nn.Embedding):
        super().__init__()
        self.mask_query = nn.Parameter(torch.randn(1, 1, hidden) * 0.02)
        self.blocks = nn.ModuleList([CrossAttnBlock(hidden, n_heads) for _ in range(n_layers)])
        self.final_norm = RMSNorm(hidden)
        self.lm_head = nn.Linear(hidden, vocab_size, bias=False)
        self.apply(init_weights)
        self.lm_head.weight = tie_weights.weight  # tied with encoder token embedding (post-init: shares Encoder's init)

    def run_blocks(
        self,
        query_stream: torch.Tensor,
        query_pos: torch.Tensor,
        kv_stream: torch.Tensor,
        kv_pos: torch.Tensor,
        kv_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        q = query_stream
        for block in self.blocks:
            q = block(q, query_pos, kv_stream, kv_pos, kv_padding_mask)
        return self.final_norm(q)

    def forward(self, masked_pos: torch.Tensor, encoder_hidden: torch.Tensor, encoder_pos: torch.Tensor) -> torch.Tensor:
        B, Tm = masked_pos.shape
        query_stream = self.mask_query.expand(B, Tm, -1)
        q = self.run_blocks(query_stream, masked_pos, encoder_hidden, encoder_pos)
        return self.lm_head(q)


class FlatBERT(nn.Module):
    """Standard flat encoder: self-attention over the full (masked-in-place) sequence, MLM head."""

    def __init__(self, vocab_size: int, hidden: int, n_layers: int, n_heads: int):
        super().__init__()
        self.encoder = Encoder(vocab_size, hidden, n_layers, n_heads)
        self.lm_head = nn.Linear(hidden, vocab_size, bias=False)
        self.lm_head.weight = self.encoder.embed.weight

    def forward(self, input_ids: torch.Tensor, position_ids: torch.Tensor) -> torch.Tensor:
        hidden = self.encoder(input_ids, position_ids)
        return self.lm_head(hidden)

    def encode_frozen(self, token_ids: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        """Full unmasked forward pass -> mean-pooled sentence embedding (padding excluded)."""
        B, S = token_ids.shape
        position_ids = torch.arange(S, device=token_ids.device).unsqueeze(0).expand(B, S)
        hidden = self.encoder(token_ids, position_ids, attention_mask)
        return masked_mean_pool(hidden, attention_mask)


class CrossBERTModel(nn.Module):
    def __init__(self, vocab_size: int, hidden: int, encoder_layers: int, predictor_layers: int, n_heads: int):
        super().__init__()
        self.encoder = Encoder(vocab_size, hidden, encoder_layers, n_heads)
        self.predictor = Predictor(hidden, predictor_layers, n_heads, vocab_size, tie_weights=self.encoder.embed)
        self.pool_query = nn.Parameter(torch.randn(1, 1, hidden) * 0.02)

    def forward(self, visible_ids: torch.Tensor, visible_pos: torch.Tensor, masked_pos: torch.Tensor) -> torch.Tensor:
        encoder_hidden = self.encoder(visible_ids, visible_pos)
        return self.predictor(masked_pos, encoder_hidden, visible_pos)

    def encode_frozen(self, token_ids: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        """Full unmasked forward pass -> mean-pooled sentence embedding (for GLUE-lite probing)."""
        B, S = token_ids.shape
        position_ids = torch.arange(S, device=token_ids.device).unsqueeze(0).expand(B, S)
        hidden = self.encoder(token_ids, position_ids, attention_mask)
        return masked_mean_pool(hidden, attention_mask)

    def pool(self, token_ids: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        """Retrieval pooling: reuse the pretrained predictor's cross-attn blocks (warm start) with
        a single learnable pooling query, instead of the many mask-reconstruction queries."""
        B, S = token_ids.shape
        position_ids = torch.arange(S, device=token_ids.device).unsqueeze(0).expand(B, S)
        encoder_hidden = self.encoder(token_ids, position_ids, attention_mask)
        query = self.pool_query.expand(B, 1, -1)
        query_pos = torch.zeros(B, 1, dtype=torch.long, device=token_ids.device)
        pooled = self.predictor.run_blocks(query, query_pos, encoder_hidden, position_ids, attention_mask)
        return pooled.squeeze(1)
