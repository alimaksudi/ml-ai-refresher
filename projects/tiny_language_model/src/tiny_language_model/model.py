"""A transparent decoder-only Transformer for learning, not production serving."""
from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class ModelConfig:
    vocab_size: int
    block_size: int = 48
    d_model: int = 64
    n_heads: int = 4
    n_layers: int = 2
    dropout: float = 0.0

    def __post_init__(self) -> None:
        if self.d_model % self.n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        if min(self.vocab_size, self.block_size, self.d_model, self.n_heads, self.n_layers) < 1:
            raise ValueError("model dimensions must be positive")

    def to_dict(self) -> dict[str, int | float]:
        return asdict(self)


class CharacterTokenizer:
    """Deterministic character tokenizer with an explicit unknown token."""

    unknown_token = "<UNK>"

    def __init__(self, text: str):
        characters = sorted(set(text))
        if not characters:
            raise ValueError("tokenizer text must not be empty")
        self.tokens = [self.unknown_token, *characters]
        self.token_to_id = {token: index for index, token in enumerate(self.tokens)}

    @property
    def vocab_size(self) -> int:
        return len(self.tokens)

    def encode(self, text: str) -> list[int]:
        unknown_id = self.token_to_id[self.unknown_token]
        return [self.token_to_id.get(character, unknown_id) for character in text]

    def decode(self, token_ids: list[int]) -> str:
        pieces = []
        for token_id in token_ids:
            token = self.tokens[token_id]
            pieces.append("?" if token == self.unknown_token else token)
        return "".join(pieces)

    def to_dict(self) -> dict[str, list[str]]:
        return {"tokens": self.tokens}

    @classmethod
    def from_dict(cls, payload: dict[str, list[str]]) -> "CharacterTokenizer":
        tokens = payload["tokens"]
        if not tokens or tokens[0] != cls.unknown_token:
            raise ValueError("invalid tokenizer payload")
        instance = cls("x")
        instance.tokens = list(tokens)
        instance.token_to_id = {token: index for index, token in enumerate(tokens)}
        return instance


class CausalSelfAttention(nn.Module):
    """Multi-head self-attention with an upper-triangular future-token mask."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.n_heads = config.n_heads
        self.head_size = config.d_model // config.n_heads
        self.query_key_value = nn.Linear(config.d_model, 3 * config.d_model)
        self.output_projection = nn.Linear(config.d_model, config.d_model)
        self.attention_dropout = nn.Dropout(config.dropout)
        self.residual_dropout = nn.Dropout(config.dropout)
        causal = torch.tril(torch.ones(config.block_size, config.block_size, dtype=torch.bool))
        self.register_buffer("causal_mask", causal.view(1, 1, config.block_size, config.block_size))

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_length, channels = hidden_states.shape
        if sequence_length > self.causal_mask.shape[-1]:
            raise ValueError("sequence exceeds configured block_size")

        combined = self.query_key_value(hidden_states)
        query, key, value = combined.chunk(3, dim=-1)

        def split_heads(tensor: torch.Tensor) -> torch.Tensor:
            return tensor.view(batch_size, sequence_length, self.n_heads, self.head_size).transpose(1, 2)

        query, key, value = map(split_heads, (query, key, value))
        scores = query @ key.transpose(-2, -1) / self.head_size**0.5
        scores = scores.masked_fill(~self.causal_mask[:, :, :sequence_length, :sequence_length], float("-inf"))
        weights = self.attention_dropout(F.softmax(scores, dim=-1))
        context = weights @ value
        context = context.transpose(1, 2).contiguous().view(batch_size, sequence_length, channels)
        return self.residual_dropout(self.output_projection(context))


class TransformerBlock(nn.Module):
    """Pre-normalization Transformer block: attention, then position-wise MLP."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.attention_norm = nn.LayerNorm(config.d_model)
        self.attention = CausalSelfAttention(config)
        self.feed_forward_norm = nn.LayerNorm(config.d_model)
        self.feed_forward = nn.Sequential(
            nn.Linear(config.d_model, 4 * config.d_model),
            nn.GELU(),
            nn.Linear(4 * config.d_model, config.d_model),
            nn.Dropout(config.dropout),
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = hidden_states + self.attention(self.attention_norm(hidden_states))
        return hidden_states + self.feed_forward(self.feed_forward_norm(hidden_states))


class TinyLanguageModel(nn.Module):
    """Decoder-only language model that predicts the next token at every position."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.position_embedding = nn.Embedding(config.block_size, config.d_model)
        self.blocks = nn.ModuleList(TransformerBlock(config) for _ in range(config.n_layers))
        self.final_norm = nn.LayerNorm(config.d_model)
        self.language_model_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.language_model_head.weight = self.token_embedding.weight
        self.apply(self._initialize_weights)

    @staticmethod
    def _initialize_weights(module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(
        self,
        token_ids: torch.Tensor,
        targets: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if token_ids.ndim != 2:
            raise ValueError("token_ids must have shape (batch, sequence)")
        batch_size, sequence_length = token_ids.shape
        if sequence_length > self.config.block_size:
            raise ValueError("sequence exceeds configured block_size")
        positions = torch.arange(sequence_length, device=token_ids.device)
        hidden_states = self.token_embedding(token_ids) + self.position_embedding(positions)
        for block in self.blocks:
            hidden_states = block(hidden_states)
        logits = self.language_model_head(self.final_norm(hidden_states))
        loss = None
        if targets is not None:
            if targets.shape != (batch_size, sequence_length):
                raise ValueError("targets must match token_ids shape")
            loss = F.cross_entropy(logits.reshape(-1, self.config.vocab_size), targets.reshape(-1))
        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        token_ids: torch.Tensor,
        max_new_tokens: int,
        *,
        temperature: float = 1.0,
        top_k: int | None = None,
        top_p: float | None = None,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """Generate tokens. Set temperature=0 for greedy decoding."""
        self.eval()
        if token_ids.ndim != 2:
            raise ValueError("token_ids must have shape (batch, sequence)")
        if temperature < 0:
            raise ValueError("temperature must be non-negative")
        if top_p is not None and not 0 < top_p <= 1:
            raise ValueError("top_p must be in (0, 1]")

        for _ in range(max_new_tokens):
            context = token_ids[:, -self.config.block_size :]
            logits, _ = self(context)
            next_logits = logits[:, -1, :]
            if temperature == 0:
                next_token = next_logits.argmax(dim=-1, keepdim=True)
            else:
                next_logits = next_logits / temperature
                if top_k is not None:
                    keep = min(top_k, next_logits.shape[-1])
                    threshold = torch.topk(next_logits, keep).values[:, -1, None]
                    next_logits = next_logits.masked_fill(next_logits < threshold, float("-inf"))
                if top_p is not None:
                    sorted_logits, sorted_indices = torch.sort(next_logits, descending=True)
                    cumulative = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    remove = cumulative - F.softmax(sorted_logits, dim=-1) >= top_p
                    sorted_logits = sorted_logits.masked_fill(remove, float("-inf"))
                    filtered = torch.full_like(next_logits, float("-inf"))
                    next_logits = filtered.scatter(1, sorted_indices, sorted_logits)
                probabilities = F.softmax(next_logits, dim=-1)
                next_token = torch.multinomial(probabilities, 1, generator=generator)
            token_ids = torch.cat((token_ids, next_token), dim=1)
        return token_ids


def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())
