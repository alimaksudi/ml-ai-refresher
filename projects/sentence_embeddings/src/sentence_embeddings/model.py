"""Bi-encoder, pooling, normalization, and contrastive loss."""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from transformer_families.models import EncoderOnlyModel, FamilyConfig


def mean_pool(hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Average only valid token states; padding must contribute exactly zero."""
    weights = attention_mask.unsqueeze(-1).to(hidden.dtype)
    return (hidden * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1)


def pool_hidden(hidden: torch.Tensor, attention_mask: torch.Tensor, strategy: str) -> torch.Tensor:
    if strategy == "mean":
        return mean_pool(hidden, attention_mask)
    if strategy == "cls":
        return hidden[:, 0]
    if strategy == "max":
        masked = hidden.masked_fill(~attention_mask.unsqueeze(-1), float("-inf"))
        return masked.max(dim=1).values
    raise ValueError(f"unknown pooling strategy: {strategy}")


class SentenceEncoder(nn.Module):
    """One shared encoder used for both query and document towers."""

    def __init__(self, config: FamilyConfig, pooling: str = "mean"):
        super().__init__()
        self.encoder = EncoderOnlyModel(config)
        self.pooling = pooling

    def forward(self, token_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        hidden = self.encoder.encode(token_ids, attention_mask)
        pooled = pool_hidden(hidden, attention_mask, self.pooling)
        return F.normalize(pooled, p=2, dim=-1)


def multiple_negatives_ranking_loss(
    anchors: torch.Tensor,
    positives: torch.Tensor,
    *,
    temperature: float = 0.08,
    hard_negatives: torch.Tensor | None = None,
) -> torch.Tensor:
    """Cross-entropy where diagonal pairs are correct and other candidates are negatives."""
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    logits = anchors @ positives.T / temperature
    labels = torch.arange(anchors.shape[0], device=anchors.device)
    loss = F.cross_entropy(logits, labels)
    if hard_negatives is not None:
        positive_scores = (anchors * positives).sum(dim=1)
        hard_negative_scores = (anchors * hard_negatives).sum(dim=1)
        loss = loss + F.softplus((hard_negative_scores - positive_scores) / temperature).mean()
    return loss
