"""Two-tower retrieval model: a content-aware item tower and a causal sequence user tower.

The user tower is a small pre-norm transformer over a user's recent events (SASRec style):
the output at position t predicts the next event. Attention uses an explicit float mask
and fp32, which is safe on Apple MPS. Both towers output L2-normalized vectors, scored by
dot product divided by a temperature.

Training uses a sampled softmax over the unique target items in the batch plus uniformly
drawn random items (mixed negative sampling), with log-Q correction so popular items are
not over-penalized for appearing often as in-batch negatives.
"""

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from streamrank.models.sequences import MAX_GAP_BUCKET, ItemContent

_MASKED = -1e9  # large finite negative: avoids NaN rows that -inf can produce


@dataclass(frozen=True)
class TwoTowerConfig:
    """Model and loss hyperparameters."""

    dim: int = 128
    max_len: int = 200
    n_layers: int = 2
    n_heads: int = 2
    dropout: float = 0.2
    temperature: float = 0.05
    n_random_negatives: int = 2048
    max_loss_positions: int = 8192
    use_sequence: bool = True  # False: causal mean of input embeddings (ablation)
    use_content: bool = True  # False: item ID only (ablation)
    use_logq: bool = True  # False: no popularity correction (ablation)


@dataclass(frozen=True)
class TensorBatch:
    """A training or query batch on the model's device (see `sequences.Batch`)."""

    tokens: Tensor
    positive: Tensor
    gaps: Tensor
    targets: Tensor
    target_mask: Tensor


class ItemTower(nn.Module):
    """Item vector from ID, genres, release-year bucket, and tags."""

    genres: Tensor
    year: Tensor
    tags: Tensor

    def __init__(self, n_items: int, content: ItemContent, cfg: TwoTowerConfig) -> None:
        super().__init__()
        d = cfg.dim
        self.use_content = cfg.use_content
        self.id_emb = nn.Embedding(n_items + 1, d, padding_idx=0)
        nn.init.normal_(self.id_emb.weight, std=0.02)
        self.register_buffer("genres", torch.from_numpy(content.genres))
        self.register_buffer("year", torch.from_numpy(content.year))
        self.register_buffer("tags", torch.from_numpy(content.tags))
        self.genre_proj = nn.Linear(content.genres.shape[1], d, bias=False)
        self.year_emb = nn.Embedding(content.n_year_buckets, d, padding_idx=0)
        self.tag_emb = nn.Embedding(content.n_tags + 1, d, padding_idx=0)
        self.out = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, d))

    def raw(self, tokens: Tensor) -> Tensor:
        """Unnormalized embedding (also used for the user tower's input tokens)."""
        x: Tensor = self.id_emb(tokens)
        if self.use_content:
            tags = self.tags[tokens]
            tag_mask = (tags > 0).unsqueeze(-1).float()
            tag_mean = (self.tag_emb(tags) * tag_mask).sum(-2) / tag_mask.sum(-2).clamp(min=1.0)
            x = x + self.genre_proj(self.genres[tokens]) + self.year_emb(self.year[tokens])
            x = x + tag_mean
        return x

    def forward(self, tokens: Tensor) -> Tensor:
        x = self.raw(tokens)
        return F.normalize(x + self.out(x), dim=-1)


class Block(nn.Module):
    """Pre-norm transformer block with explicit-mask scaled dot-product attention."""

    def __init__(self, d: int, n_heads: int, dropout: float) -> None:
        super().__init__()
        self.n_heads = n_heads
        self.ln1 = nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d)
        self.proj = nn.Linear(d, d)
        self.ln2 = nn.LayerNorm(d)
        self.ffn = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))
        self.drop = nn.Dropout(dropout)
        self.attn_dropout = dropout

    def forward(self, x: Tensor, mask: Tensor) -> Tensor:
        b, length, d = x.shape
        q, k, v = self.qkv(self.ln1(x)).split(d, dim=-1)
        heads = (b, length, self.n_heads, d // self.n_heads)
        q, k, v = (t.view(heads).transpose(1, 2) for t in (q, k, v))
        att = F.scaled_dot_product_attention(
            q, k, v, attn_mask=mask, dropout_p=self.attn_dropout if self.training else 0.0
        )
        x = x + self.drop(self.proj(att.transpose(1, 2).reshape(b, length, d)))
        out: Tensor = x + self.drop(self.ffn(self.ln2(x)))
        return out


def attention_mask(valid: Tensor) -> Tensor:
    """Float mask (B, 1, L, L): causal, padded keys hidden, padded queries see only themselves."""
    length = valid.shape[1]
    causal = torch.ones(length, length, dtype=torch.bool, device=valid.device).tril()
    allowed = causal & valid[:, None, :]
    allowed = allowed | torch.eye(length, dtype=torch.bool, device=valid.device)
    mask = torch.zeros(allowed.shape, device=valid.device)
    return mask.masked_fill(~allowed, _MASKED).unsqueeze(1)


class UserTower(nn.Module):
    """Per-position user vectors from the events up to and including that position."""

    def __init__(self, items: ItemTower, cfg: TwoTowerConfig) -> None:
        super().__init__()
        d = cfg.dim
        self.items = items
        self.use_sequence = cfg.use_sequence
        self.pos_emb = nn.Embedding(cfg.max_len, d)
        self.rating_emb = nn.Embedding(2, d)
        self.gap_emb = nn.Embedding(MAX_GAP_BUCKET + 1, d, padding_idx=0)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList(
            Block(d, cfg.n_heads, cfg.dropout)
            for _ in range(cfg.n_layers if cfg.use_sequence else 0)
        )
        self.ln = nn.LayerNorm(d)
        self.out = nn.Linear(d, d)

    def forward(self, tokens: Tensor, positive: Tensor, gaps: Tensor) -> Tensor:
        valid = tokens > 0
        length = tokens.shape[1]
        pos = torch.arange(length, device=tokens.device)
        x = self.items.raw(tokens) + self.rating_emb(positive.long()) + self.gap_emb(gaps)
        if self.use_sequence:
            x = self.drop(x + self.pos_emb(pos))
            mask = attention_mask(valid)
            for block in self.blocks:
                x = block(x, mask)
        else:
            # Causal running mean over valid events: no ordering or interaction modeling.
            w = valid.unsqueeze(-1).float()
            x = (x * w).cumsum(1) / w.cumsum(1).clamp(min=1.0)
        return F.normalize(self.out(self.ln(x)), dim=-1)


class TwoTower(nn.Module):
    """Both towers plus the training loss."""

    target_freq: Tensor

    def __init__(
        self, n_items: int, content: ItemContent, item_counts: Tensor, cfg: TwoTowerConfig
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.n_items = n_items
        self.item_tower = ItemTower(n_items, content, cfg)
        self.user_tower = UserTower(self.item_tower, cfg)
        q = item_counts.double() / item_counts.sum()
        self.register_buffer("target_freq", torch.cat([torch.zeros(1), q.float()]))

    def last_position(self, tokens: Tensor, positive: Tensor, gaps: Tensor) -> Tensor:
        """User vector at the final (most recent) position of left-padded inputs."""
        out: Tensor = self.user_tower(tokens, positive, gaps)[:, -1]
        return out

    def all_items(self, batch_size: int = 16384) -> Tensor:
        """Vectors for every catalog item, in catalog order (token 1..n)."""
        device = self.target_freq.device
        parts = [
            self.item_tower(torch.arange(s, min(s + batch_size, self.n_items + 1), device=device))
            for s in range(1, self.n_items + 1, batch_size)
        ]
        return torch.cat(parts)

    def loss(self, batch: TensorBatch, generator: torch.Generator | None = None) -> Tensor:
        """Sampled softmax over in-batch target items plus random items, with log-Q."""
        cfg = self.cfg
        users = self.user_tower(batch.tokens, batch.positive, batch.gaps)[batch.target_mask]
        pos_items = batch.targets[batch.target_mask]  # (P,)
        if pos_items.numel() > cfg.max_loss_positions:
            keep = torch.randperm(pos_items.numel(), device=pos_items.device, generator=generator)
            keep = keep[: cfg.max_loss_positions]
            users, pos_items = users[keep], pos_items[keep]
        n_pos = pos_items.numel()
        in_batch, label = torch.unique(pos_items, return_inverse=True)
        rand = torch.randint(
            1,
            self.n_items + 1,
            (cfg.n_random_negatives,),
            device=pos_items.device,
            generator=generator,
        )
        cands = torch.cat([in_batch, rand])
        logits = users @ self.item_tower(cands).T / cfg.temperature
        if cfg.use_logq:
            expected = n_pos * self.target_freq[cands] + cfg.n_random_negatives / self.n_items
            logits = logits - torch.log(expected.clamp(min=1e-12))
        # A random draw equal to the row's own positive is not a negative.
        hit = rand.unsqueeze(0) == pos_items.unsqueeze(1)
        logits[:, in_batch.numel() :] = logits[:, in_batch.numel() :].masked_fill(hit, _MASKED)
        return F.cross_entropy(logits, label)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def pick_device() -> torch.device:
    """Apple MPS when available, else CPU."""
    return torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def cosine_lr(step: int, total: int, warmup: int) -> float:
    """Linear warmup then cosine decay to 10% of the peak, as a multiplier."""
    if step < warmup:
        return (step + 1) / warmup
    progress = min(1.0, (step - warmup) / max(1, total - warmup))
    return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress))
