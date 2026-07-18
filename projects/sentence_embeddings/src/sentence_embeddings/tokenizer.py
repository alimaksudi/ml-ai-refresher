"""A deliberately small word tokenizer so every preprocessing step is visible."""
from __future__ import annotations

import re
from dataclasses import dataclass

import torch

TOKEN_PATTERN = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    return TOKEN_PATTERN.findall(text.casefold())


@dataclass
class WordTokenizer:
    token_to_id: dict[str, int]
    max_length: int = 24

    PAD = 0
    UNK = 1
    CLS = 2

    @classmethod
    def fit(cls, texts: list[str], max_length: int = 24) -> "WordTokenizer":
        vocabulary = sorted({token for text in texts for token in tokenize(text)})
        mapping = {"[PAD]": cls.PAD, "[UNK]": cls.UNK, "[CLS]": cls.CLS}
        mapping.update({token: index + 3 for index, token in enumerate(vocabulary)})
        return cls(mapping, max_length=max_length)

    def encode_batch(self, texts: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
        rows: list[list[int]] = []
        masks: list[list[bool]] = []
        for text in texts:
            ids = [self.CLS, *(self.token_to_id.get(token, self.UNK) for token in tokenize(text))]
            ids = ids[: self.max_length]
            valid_length = len(ids)
            ids += [self.PAD] * (self.max_length - valid_length)
            rows.append(ids)
            masks.append([True] * valid_length + [False] * (self.max_length - valid_length))
        return torch.tensor(rows, dtype=torch.long), torch.tensor(masks, dtype=torch.bool)

    @property
    def vocab_size(self) -> int:
        return len(self.token_to_id)
