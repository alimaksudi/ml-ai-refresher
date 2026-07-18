"""Small inspectable Transformer families; intended for learning, not deployment."""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


def initialize_transformer_weights(module: nn.Module) -> None:
    if isinstance(module, (nn.Linear, nn.Embedding)):
        nn.init.normal_(module.weight, mean=0.0, std=0.02)
        if isinstance(module, nn.Linear) and module.bias is not None:
            nn.init.zeros_(module.bias)


@dataclass(frozen=True)
class FamilyConfig:
    vocab_size: int = 16
    max_length: int = 16
    d_model: int = 32
    n_heads: int = 4
    n_layers: int = 1
    n_classes: int = 2

    def __post_init__(self) -> None:
        if self.d_model % self.n_heads:
            raise ValueError("d_model must be divisible by n_heads")


class MultiHeadAttention(nn.Module):
    """Shared attention supporting causal self-attention and encoder cross-attention."""

    def __init__(self, config: FamilyConfig):
        super().__init__()
        self.n_heads = config.n_heads
        self.head_size = config.d_model // config.n_heads
        self.query_projection = nn.Linear(config.d_model, config.d_model)
        self.key_projection = nn.Linear(config.d_model, config.d_model)
        self.value_projection = nn.Linear(config.d_model, config.d_model)
        self.output_projection = nn.Linear(config.d_model, config.d_model)
        self.last_score_shape: tuple[int, ...] | None = None

    def _split_heads(self, tensor: torch.Tensor) -> torch.Tensor:
        batch, length, channels = tensor.shape
        return tensor.view(batch, length, self.n_heads, self.head_size).transpose(1, 2)

    def forward(
        self,
        query_states: torch.Tensor,
        context_states: torch.Tensor | None = None,
        *,
        key_padding_mask: torch.Tensor | None = None,
        causal: bool = False,
    ) -> torch.Tensor:
        context_states = query_states if context_states is None else context_states
        query = self._split_heads(self.query_projection(query_states))
        key = self._split_heads(self.key_projection(context_states))
        value = self._split_heads(self.value_projection(context_states))
        scores = query @ key.transpose(-2, -1) / self.head_size**0.5
        self.last_score_shape = tuple(scores.shape)

        if key_padding_mask is not None:
            if key_padding_mask.shape != (context_states.shape[0], context_states.shape[1]):
                raise ValueError("key_padding_mask must have shape (batch, key_length)")
            scores = scores.masked_fill(~key_padding_mask[:, None, None, :], float("-inf"))
        if causal:
            query_length, key_length = query.shape[-2], key.shape[-2]
            causal_mask = torch.tril(
                torch.ones(query_length, key_length, dtype=torch.bool, device=scores.device)
            )
            scores = scores.masked_fill(~causal_mask, float("-inf"))

        weights = F.softmax(scores, dim=-1)
        context = weights @ value
        batch, _, query_length, _ = context.shape
        context = context.transpose(1, 2).contiguous().view(batch, query_length, -1)
        return self.output_projection(context)


class FeedForward(nn.Module):
    def __init__(self, config: FamilyConfig):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(config.d_model, 4 * config.d_model),
            nn.GELU(),
            nn.Linear(4 * config.d_model, config.d_model),
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.network(hidden_states)


class EncoderBlock(nn.Module):
    def __init__(self, config: FamilyConfig):
        super().__init__()
        self.attention_norm = nn.LayerNorm(config.d_model)
        self.self_attention = MultiHeadAttention(config)
        self.feed_forward_norm = nn.LayerNorm(config.d_model)
        self.feed_forward = FeedForward(config)

    def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor | None) -> torch.Tensor:
        normalized = self.attention_norm(hidden_states)
        hidden_states = hidden_states + self.self_attention(
            normalized,
            key_padding_mask=attention_mask,
            causal=False,
        )
        return hidden_states + self.feed_forward(self.feed_forward_norm(hidden_states))


class CausalDecoderBlock(nn.Module):
    def __init__(self, config: FamilyConfig):
        super().__init__()
        self.attention_norm = nn.LayerNorm(config.d_model)
        self.self_attention = MultiHeadAttention(config)
        self.feed_forward_norm = nn.LayerNorm(config.d_model)
        self.feed_forward = FeedForward(config)

    def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor | None) -> torch.Tensor:
        normalized = self.attention_norm(hidden_states)
        hidden_states = hidden_states + self.self_attention(
            normalized,
            key_padding_mask=attention_mask,
            causal=True,
        )
        return hidden_states + self.feed_forward(self.feed_forward_norm(hidden_states))


class EncoderDecoderBlock(nn.Module):
    def __init__(self, config: FamilyConfig):
        super().__init__()
        self.self_attention_norm = nn.LayerNorm(config.d_model)
        self.self_attention = MultiHeadAttention(config)
        self.cross_attention_norm = nn.LayerNorm(config.d_model)
        self.cross_attention = MultiHeadAttention(config)
        self.feed_forward_norm = nn.LayerNorm(config.d_model)
        self.feed_forward = FeedForward(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_states: torch.Tensor,
        target_mask: torch.Tensor | None,
        source_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        normalized = self.self_attention_norm(hidden_states)
        hidden_states = hidden_states + self.self_attention(
            normalized,
            key_padding_mask=target_mask,
            causal=True,
        )
        normalized = self.cross_attention_norm(hidden_states)
        hidden_states = hidden_states + self.cross_attention(
            normalized,
            context_states=encoder_states,
            key_padding_mask=source_mask,
            causal=False,
        )
        return hidden_states + self.feed_forward(self.feed_forward_norm(hidden_states))


class TokenAndPositionEmbedding(nn.Module):
    def __init__(self, config: FamilyConfig):
        super().__init__()
        self.token = nn.Embedding(config.vocab_size, config.d_model)
        self.position = nn.Embedding(config.max_length, config.d_model)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        length = token_ids.shape[1]
        if length > self.position.num_embeddings:
            raise ValueError("sequence exceeds max_length")
        positions = torch.arange(length, device=token_ids.device)
        return self.token(token_ids) + self.position(positions)


class DecoderOnlyModel(nn.Module):
    """GPT-style causal decoder trained with next-token prediction."""

    def __init__(self, config: FamilyConfig):
        super().__init__()
        self.config = config
        self.embedding = TokenAndPositionEmbedding(config)
        self.blocks = nn.ModuleList(CausalDecoderBlock(config) for _ in range(config.n_layers))
        self.final_norm = nn.LayerNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.lm_head.weight = self.embedding.token.weight
        self.apply(initialize_transformer_weights)

    def forward(self, token_ids: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        hidden = self.embedding(token_ids)
        for block in self.blocks:
            hidden = block(hidden, attention_mask)
        return self.lm_head(self.final_norm(hidden))


class EncoderOnlyModel(nn.Module):
    """BERT-style bidirectional encoder with MLM and classification heads."""

    def __init__(self, config: FamilyConfig):
        super().__init__()
        self.config = config
        self.embedding = TokenAndPositionEmbedding(config)
        self.blocks = nn.ModuleList(EncoderBlock(config) for _ in range(config.n_layers))
        self.final_norm = nn.LayerNorm(config.d_model)
        self.mlm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.mlm_head.weight = self.embedding.token.weight
        self.classification_head = nn.Linear(config.d_model, config.n_classes)
        self.apply(initialize_transformer_weights)

    def encode(self, token_ids: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        hidden = self.embedding(token_ids)
        for block in self.blocks:
            hidden = block(hidden, attention_mask)
        return self.final_norm(hidden)

    def forward(self, token_ids: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        return self.mlm_head(self.encode(token_ids, attention_mask))

    def classify(self, token_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        hidden = self.encode(token_ids, attention_mask)
        weights = attention_mask.unsqueeze(-1).to(hidden.dtype)
        pooled = (hidden * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1)
        return self.classification_head(pooled)


class EncoderDecoderModel(nn.Module):
    """T5-style encoder-decoder with causal decoder and cross-attention."""

    def __init__(self, config: FamilyConfig):
        super().__init__()
        self.config = config
        self.source_embedding = TokenAndPositionEmbedding(config)
        self.target_embedding = TokenAndPositionEmbedding(config)
        self.encoder_blocks = nn.ModuleList(EncoderBlock(config) for _ in range(config.n_layers))
        self.decoder_blocks = nn.ModuleList(EncoderDecoderBlock(config) for _ in range(config.n_layers))
        self.encoder_norm = nn.LayerNorm(config.d_model)
        self.decoder_norm = nn.LayerNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.lm_head.weight = self.target_embedding.token.weight
        self.apply(initialize_transformer_weights)

    def encode(self, source_ids: torch.Tensor, source_mask: torch.Tensor | None = None) -> torch.Tensor:
        hidden = self.source_embedding(source_ids)
        for block in self.encoder_blocks:
            hidden = block(hidden, source_mask)
        return self.encoder_norm(hidden)

    def forward(
        self,
        source_ids: torch.Tensor,
        target_input_ids: torch.Tensor,
        source_mask: torch.Tensor | None = None,
        target_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        encoder_states = self.encode(source_ids, source_mask)
        hidden = self.target_embedding(target_input_ids)
        for block in self.decoder_blocks:
            hidden = block(hidden, encoder_states, target_mask, source_mask)
        return self.lm_head(self.decoder_norm(hidden))

    @torch.no_grad()
    def generate(
        self,
        source_ids: torch.Tensor,
        source_mask: torch.Tensor,
        *,
        bos_token_id: int,
        eos_token_id: int,
        max_new_tokens: int,
    ) -> torch.Tensor:
        self.eval()
        generated = torch.full(
            (source_ids.shape[0], 1),
            bos_token_id,
            dtype=torch.long,
            device=source_ids.device,
        )
        finished = torch.zeros(source_ids.shape[0], dtype=torch.bool, device=source_ids.device)
        for _ in range(max_new_tokens):
            target_mask = torch.ones_like(generated, dtype=torch.bool)
            logits = self(source_ids, generated, source_mask, target_mask)
            next_token = logits[:, -1].argmax(dim=-1)
            next_token = torch.where(finished, eos_token_id, next_token)
            generated = torch.cat((generated, next_token[:, None]), dim=1)
            finished |= next_token == eos_token_id
            if bool(finished.all()):
                break
        return generated


def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())
