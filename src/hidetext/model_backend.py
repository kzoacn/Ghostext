from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import math

from .errors import ModelBackendError


@dataclass(frozen=True)
class TokenProb:
    token: str
    token_id: int
    probability: float


@dataclass(frozen=True)
class BackendMetadata:
    model_id: str
    tokenizer_hash: str
    backend_id: str

    def as_dict(self) -> dict[str, str]:
        return {
            "model_id": self.model_id,
            "tokenizer_hash": self.tokenizer_hash,
            "backend_id": self.backend_id,
        }


def _stable_fraction(*parts: object) -> float:
    data = "||".join(str(part) for part in parts).encode("utf-8")
    value = int.from_bytes(sha256(data).digest()[:8], "big")
    return value / float(2**64 - 1)


def _char_class(ch: str) -> str:
    if ch.isspace():
        return "space"
    if "\u4e00" <= ch <= "\u9fff":
        return "cjk"
    if ch.isalpha():
        return "alpha"
    if ch.isdigit():
        return "digit"
    return "punct"


class ToyCharBackend:
    """A deterministic character-level backend for protocol and E2E tests."""

    def __init__(self) -> None:
        self._streams = {
            "zh": (
                "今天的风很轻，街角的咖啡店还亮着暖黄的灯，路过的人慢慢聊着天，城市显得安静而柔和。"
                "午后的窗边有一阵淡淡的茶香，桌上的书翻到一半，雨声落在树叶上，房间里有种平稳的明亮。"
                "夜色落下时，远处的车灯像细小的星，巷口传来很轻的笑声，整个傍晚都带着温柔的节奏。"
            ),
            "en": (
                "The afternoon feels calm, and the corner cafe keeps a warm light on while people talk softly along the street. "
                "A quiet breeze moves past the window, the pages stay half open on the table, and the room keeps a steady brightness. "
                "When evening arrives, the distant headlights look like small stars and the whole street settles into an easy rhythm. "
            ),
        }
        self._punctuation = " .,!?;:'\"()-"
        self._vocab = {
            language: self._build_vocab(stream)
            for language, stream in self._streams.items()
        }
        self._metadata = BackendMetadata(
            model_id="toy-char-backend-v1",
            tokenizer_hash=sha256(
                "||".join("".join(chars) for chars in self._vocab.values()).encode("utf-8")
            ).hexdigest(),
            backend_id="toy-char",
        )

    @property
    def metadata(self) -> BackendMetadata:
        return self._metadata

    def _build_vocab(self, stream: str) -> tuple[str, ...]:
        chars = sorted(set(stream + self._punctuation))
        return tuple(chars)

    def _language(self, prompt: str) -> str:
        return "zh" if any("\u4e00" <= ch <= "\u9fff" for ch in prompt) else "en"

    def tokenize(self, text: str, prompt: str) -> list[str]:
        language = self._language(prompt)
        vocab = set(self._vocab[language])
        unknown = [ch for ch in text if ch not in vocab]
        if unknown:
            raise ModelBackendError(f"text contains unsupported token(s): {unknown[:5]!r}")
        return list(text)

    def render(self, tokens: list[str]) -> str:
        return "".join(tokens)

    def distribution(self, prompt: str, generated_tokens: list[str], seed: int) -> list[TokenProb]:
        language = self._language(prompt)
        stream = self._streams[language]
        vocab = self._vocab[language]
        position = len(generated_tokens) % len(stream)
        expected = stream[position]
        window = {
            stream[(position + offset) % len(stream)]
            for offset in range(-3, 4)
        }
        context_tail = "".join(generated_tokens[-24:])

        logits: list[float] = []
        for ch in vocab:
            logit = -4.0
            if ch == expected:
                logit += 4.1
            if ch in window:
                logit += 1.1
            if _char_class(ch) == _char_class(expected):
                logit += 0.6
            if ch == " " and expected.isspace():
                logit += 0.8
            logit += (_stable_fraction(prompt, context_tail, seed, ch) - 0.5) * 0.45
            logits.append(logit)

        max_logit = max(logits)
        exps = [math.exp(logit - max_logit) for logit in logits]
        normalizer = sum(exps)
        probs = [value / normalizer for value in exps]
        return [
            TokenProb(token=ch, token_id=token_id, probability=prob)
            for token_id, (ch, prob) in enumerate(zip(vocab, probs, strict=True))
        ]

