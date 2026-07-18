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

    def token_text(self, token_id: int) -> str:
        token = self.tokens[token_id]
        return "?" if token == self.unknown_token else token

    def to_dict(self) -> dict[str, object]:
        return {"type": "character", "tokens": self.tokens}

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> "CharacterTokenizer":
        tokens = payload["tokens"]
        if not isinstance(tokens, list):
            raise ValueError("invalid tokenizer payload")
        if not tokens or tokens[0] != cls.unknown_token:
            raise ValueError("invalid tokenizer payload")
        instance = cls("x")
        instance.tokens = list(tokens)
        instance.token_to_id = {token: index for index, token in enumerate(tokens)}
        return instance


class BPETokenizer:
    """Educational BPE learned by repeatedly merging frequent adjacent symbols.

    The initial symbols are Unicode characters from the training split. Merges operate
    on the raw character stream, including whitespace. This keeps encoding lossless and
    transparent; production byte-level tokenizers add normalization, special tokens,
    and more careful pre-tokenization.
    """

    unknown_token = "<UNK>"

    def __init__(self, tokens: list[str], merges: list[tuple[str, str]]):
        if not tokens or tokens[0] != self.unknown_token:
            raise ValueError("BPE vocabulary must begin with <UNK>")
        self.tokens = list(tokens)
        self.merges = list(merges)
        self.token_to_id = {token: index for index, token in enumerate(self.tokens)}

    @classmethod
    def train(cls, text: str, target_vocab_size: int = 80) -> "BPETokenizer":
        if not text:
            raise ValueError("tokenizer text must not be empty")
        base_tokens = sorted(set(text))
        if target_vocab_size < len(base_tokens) + 1:
            raise ValueError("target_vocab_size cannot be smaller than the character vocabulary")

        vocabulary = [cls.unknown_token, *base_tokens]
        symbols = list(text)
        merges: list[tuple[str, str]] = []

        while len(vocabulary) < target_vocab_size and len(symbols) > 1:
            pair_counts: dict[tuple[str, str], int] = {}
            for pair in zip(symbols, symbols[1:]):
                pair_counts[pair] = pair_counts.get(pair, 0) + 1
            if not pair_counts:
                break
            # Highest frequency wins; lexical tie-breaking makes training reproducible.
            candidates = sorted(pair_counts, key=lambda pair: (-pair_counts[pair], pair))
            selected = next((pair for pair in candidates if "".join(pair) not in vocabulary), None)
            if selected is None:
                break
            merged_token = "".join(selected)
            symbols = cls._merge_pair(symbols, selected, merged_token)
            merges.append(selected)
            vocabulary.append(merged_token)
        return cls(vocabulary, merges)

    @staticmethod
    def _merge_pair(symbols: list[str], pair: tuple[str, str], merged: str) -> list[str]:
        output: list[str] = []
        index = 0
        while index < len(symbols):
            if index + 1 < len(symbols) and (symbols[index], symbols[index + 1]) == pair:
                output.append(merged)
                index += 2
            else:
                output.append(symbols[index])
                index += 1
        return output

    @property
    def vocab_size(self) -> int:
        return len(self.tokens)

    def encode(self, text: str) -> list[int]:
        symbols = [character if character in self.token_to_id else self.unknown_token for character in text]
        for pair in self.merges:
            merged = "".join(pair)
            symbols = self._merge_pair(symbols, pair, merged)
        return [self.token_to_id[symbol] for symbol in symbols]

    def token_text(self, token_id: int) -> str:
        token = self.tokens[token_id]
        return "?" if token == self.unknown_token else token

    def decode(self, token_ids: list[int]) -> str:
        return "".join(self.token_text(token_id) for token_id in token_ids)

    def to_dict(self) -> dict[str, object]:
        return {
            "type": "bpe",
            "tokens": self.tokens,
            "merges": [list(pair) for pair in self.merges],
        }

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> "BPETokenizer":
        tokens = payload.get("tokens")
        raw_merges = payload.get("merges")
        if not isinstance(tokens, list) or not isinstance(raw_merges, list):
            raise ValueError("invalid BPE tokenizer payload")
        merges = []
        for pair in raw_merges:
            if not isinstance(pair, list) or len(pair) != 2:
                raise ValueError("invalid BPE merge")
            merges.append((str(pair[0]), str(pair[1])))
        return cls([str(token) for token in tokens], merges)


Tokenizer = CharacterTokenizer | BPETokenizer


def tokenizer_from_dict(payload: dict[str, object]) -> Tokenizer:
    tokenizer_type = payload.get("type", "character")
    if tokenizer_type == "character":
        return CharacterTokenizer.from_dict(payload)
    if tokenizer_type == "bpe":
        return BPETokenizer.from_dict(payload)
    raise ValueError(f"unknown tokenizer type: {tokenizer_type}")


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

    def forward_with_cache(
        self,
        hidden_states: torch.Tensor,
        past_key_value: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """Attend to new tokens plus cached keys/values and return the enlarged cache."""
        batch_size, query_length, channels = hidden_states.shape
        combined = self.query_key_value(hidden_states)
        query, key, value = combined.chunk(3, dim=-1)

        def split_heads(tensor: torch.Tensor) -> torch.Tensor:
            return tensor.view(batch_size, query_length, self.n_heads, self.head_size).transpose(1, 2)

        query, key, value = map(split_heads, (query, key, value))
        past_length = 0
        if past_key_value is not None:
            past_key, past_value = past_key_value
            if past_key.shape != past_value.shape:
                raise ValueError("cached keys and values must have identical shapes")
            if past_key.shape[:2] != (batch_size, self.n_heads):
                raise ValueError("cache batch or head dimensions do not match the input")
            past_length = past_key.shape[-2]
            key = torch.cat((past_key, key), dim=-2)
            value = torch.cat((past_value, value), dim=-2)

        total_key_length = key.shape[-2]
        if total_key_length > self.causal_mask.shape[-1]:
            raise ValueError("cache exceeds configured block_size")
        query_positions = torch.arange(
            past_length,
            past_length + query_length,
            device=hidden_states.device,
        ).view(query_length, 1)
        key_positions = torch.arange(total_key_length, device=hidden_states.device).view(1, total_key_length)
        allowed = key_positions <= query_positions

        scores = query @ key.transpose(-2, -1) / self.head_size**0.5
        scores = scores.masked_fill(~allowed.view(1, 1, query_length, total_key_length), float("-inf"))
        weights = self.attention_dropout(F.softmax(scores, dim=-1))
        context = weights @ value
        context = context.transpose(1, 2).contiguous().view(batch_size, query_length, channels)
        output = self.residual_dropout(self.output_projection(context))
        return output, (key, value)


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

    def forward_with_cache(
        self,
        hidden_states: torch.Tensor,
        past_key_value: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        attention_output, present_key_value = self.attention.forward_with_cache(
            self.attention_norm(hidden_states),
            past_key_value,
        )
        hidden_states = hidden_states + attention_output
        hidden_states = hidden_states + self.feed_forward(self.feed_forward_norm(hidden_states))
        return hidden_states, present_key_value


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

    def forward_with_cache(
        self,
        token_ids: torch.Tensor,
        past_key_values: list[tuple[torch.Tensor, torch.Tensor]] | None = None,
    ) -> tuple[torch.Tensor, list[tuple[torch.Tensor, torch.Tensor]]]:
        """Forward only new tokens while carrying one key/value cache per block."""
        if token_ids.ndim != 2:
            raise ValueError("token_ids must have shape (batch, sequence)")
        _, new_token_count = token_ids.shape
        if past_key_values is None:
            past_key_values = [None] * len(self.blocks)
            past_length = 0
        else:
            if len(past_key_values) != len(self.blocks):
                raise ValueError("one cache entry is required per Transformer block")
            cache_lengths = {key.shape[-2] for key, _ in past_key_values}
            if len(cache_lengths) != 1:
                raise ValueError("all layer caches must have the same sequence length")
            past_length = cache_lengths.pop()
        if past_length + new_token_count > self.config.block_size:
            raise ValueError("tokens plus cache exceed configured block_size")

        positions = torch.arange(
            past_length,
            past_length + new_token_count,
            device=token_ids.device,
        )
        hidden_states = self.token_embedding(token_ids) + self.position_embedding(positions)
        present_key_values = []
        for block, past_key_value in zip(self.blocks, past_key_values):
            hidden_states, present = block.forward_with_cache(hidden_states, past_key_value)
            present_key_values.append(present)
        logits = self.language_model_head(self.final_norm(hidden_states))
        return logits, present_key_values

    @staticmethod
    def _sample_next_token(
        next_logits: torch.Tensor,
        *,
        temperature: float,
        top_k: int | None,
        top_p: float | None,
        generator: torch.Generator | None,
    ) -> torch.Tensor:
        if temperature == 0:
            return next_logits.argmax(dim=-1, keepdim=True)
        next_logits = next_logits / temperature
        if top_k is not None:
            keep = min(top_k, next_logits.shape[-1])
            threshold = torch.topk(next_logits, keep).values[:, -1, None]
            next_logits = next_logits.masked_fill(next_logits < threshold, float("-inf"))
        if top_p is not None:
            sorted_logits, sorted_indices = torch.sort(next_logits, descending=True)
            sorted_probabilities = F.softmax(sorted_logits, dim=-1)
            cumulative = torch.cumsum(sorted_probabilities, dim=-1)
            remove = cumulative - sorted_probabilities >= top_p
            sorted_logits = sorted_logits.masked_fill(remove, float("-inf"))
            filtered = torch.full_like(next_logits, float("-inf"))
            next_logits = filtered.scatter(1, sorted_indices, sorted_logits)
        probabilities = F.softmax(next_logits, dim=-1)
        return torch.multinomial(probabilities, 1, generator=generator)

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
            next_token = self._sample_next_token(
                next_logits,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                generator=generator,
            )
            token_ids = torch.cat((token_ids, next_token), dim=1)
        return token_ids

    @torch.no_grad()
    def generate_with_cache(
        self,
        token_ids: torch.Tensor,
        max_new_tokens: int,
        *,
        temperature: float = 1.0,
        top_k: int | None = None,
        top_p: float | None = None,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """Generate with prompt prefill and incremental key/value reuse."""
        self.eval()
        if token_ids.ndim != 2:
            raise ValueError("token_ids must have shape (batch, sequence)")
        if temperature < 0:
            raise ValueError("temperature must be non-negative")
        if top_p is not None and not 0 < top_p <= 1:
            raise ValueError("top_p must be in (0, 1]")
        if max_new_tokens == 0:
            return token_ids

        context = token_ids[:, -self.config.block_size :]
        logits, cache = self.forward_with_cache(context)
        for step in range(max_new_tokens):
            next_token = self._sample_next_token(
                logits[:, -1, :],
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                generator=generator,
            )
            token_ids = torch.cat((token_ids, next_token), dim=1)
            if step + 1 == max_new_tokens:
                break

            cached_length = cache[0][0].shape[-2]
            if cached_length >= self.config.block_size:
                # Learned absolute positions change after a sliding-window shift, so
                # recompute the retained context instead of reusing stale positions.
                context = token_ids[:, -self.config.block_size :]
                logits, cache = self.forward_with_cache(context)
            else:
                logits, cache = self.forward_with_cache(next_token, cache)
        return token_ids


def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())
