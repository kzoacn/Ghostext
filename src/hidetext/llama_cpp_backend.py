from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
import os
from pathlib import Path
from typing import Any

import numpy as np

from .errors import ModelBackendError
from .model_backend import BackendMetadata, RawNextTokenDistribution


QWEN3_USER_PROMPT_TEMPLATE = "<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"


@dataclass(frozen=True)
class LlamaCppBackendConfig:
    model_path: str
    model_id: str = "Qwen/Qwen3-4B-Instruct-2507"
    n_ctx: int = 4096
    n_batch: int = 256
    n_threads: int | None = None
    seed: int = 7
    use_mmap: bool = True
    verbose: bool = False


class QwenLlamaCppBackend:
    """Qwen3 GGUF backend powered by llama.cpp with CPU inference."""

    def __init__(self, config: LlamaCppBackendConfig) -> None:
        try:
            from llama_cpp import Llama
        except ImportError as exc:
            raise ModelBackendError(
                "llama-cpp backend requires llama-cpp-python; install hidetext[llm]"
            ) from exc

        self.config = config
        model_path = Path(config.model_path)
        if not model_path.exists():
            raise ModelBackendError(f"model file not found: {model_path}")

        n_threads = config.n_threads or max(1, os.cpu_count() or 1)
        self._llm = Llama(
            model_path=str(model_path),
            n_gpu_layers=0,
            seed=config.seed,
            n_ctx=config.n_ctx,
            n_batch=config.n_batch,
            n_threads=n_threads,
            logits_all=False,
            use_mmap=config.use_mmap,
            verbose=config.verbose,
        )
        self._n_vocab = int(self._llm.n_vocab())
        self._cached_prompt: str | None = None
        self._cached_prompt_token_ids: list[int] = []
        self._cached_generated_token_ids: list[int] = []
        self._blocked_token_ids = self._build_blocked_token_ids()
        self._metadata = BackendMetadata(
            model_id=config.model_id,
            tokenizer_hash=self._metadata_hash(model_path),
            backend_id="llama-cpp-qwen3",
        )

    @property
    def metadata(self) -> BackendMetadata:
        return self._metadata

    def tokenize(self, text: str, prompt: str) -> list[int]:
        del prompt
        try:
            return self._llm.tokenize(text.encode("utf-8"), add_bos=False, special=False)
        except Exception as exc:
            raise ModelBackendError(f"failed to tokenize stego text: {exc}") from exc

    def render(self, token_ids: list[int]) -> str:
        try:
            rendered = self._llm.detokenize(token_ids, special=False)
        except Exception as exc:
            raise ModelBackendError(f"failed to detokenize generated ids: {exc}") from exc
        try:
            return rendered.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ModelBackendError(
                "generated token sequence is not valid UTF-8; real backend failed closed"
            ) from exc

    def token_text(self, token_id: int) -> str:
        try:
            return self._llm.detokenize([token_id], special=True).decode(
                "utf-8",
                errors="replace",
            )
        except Exception:
            return f"<token:{token_id}>"

    def distribution(
        self,
        prompt: str,
        generated_token_ids: list[int],
        seed: int,
    ) -> RawNextTokenDistribution:
        if seed != self.config.seed:
            raise ModelBackendError(
                f"backend seed mismatch: backend={self.config.seed}, requested={seed}"
            )

        self._ensure_state(prompt, generated_token_ids)
        logits = np.ctypeslib.as_array(
            self._llm._ctx.get_logits(),  # type: ignore[attr-defined]
            shape=(self._n_vocab,),
        ).astype(np.float64, copy=True)
        if self._blocked_token_ids:
            logits[list(self._blocked_token_ids)] = -np.inf
        return RawNextTokenDistribution(
            token_ids=np.arange(self._n_vocab, dtype=np.int32),
            logits=logits,
        )

    def _ensure_state(self, prompt: str, generated_token_ids: list[int]) -> None:
        prompt_token_ids = self._prompt_token_ids(prompt)
        if len(prompt_token_ids) + len(generated_token_ids) >= self.config.n_ctx:
            raise ModelBackendError(
                f"context would exceed n_ctx={self.config.n_ctx}; reduce message or increase ctx"
            )

        prompt_changed = prompt != self._cached_prompt
        prefix_matches = generated_token_ids[: len(self._cached_generated_token_ids)] == (
            self._cached_generated_token_ids
        )

        if prompt_changed or not prefix_matches:
            self._llm.reset()
            self._llm.eval(prompt_token_ids)
            self._cached_prompt = prompt
            self._cached_prompt_token_ids = list(prompt_token_ids)
            self._cached_generated_token_ids = []

        delta = generated_token_ids[len(self._cached_generated_token_ids) :]
        if delta:
            self._llm.eval(delta)
            self._cached_generated_token_ids = list(generated_token_ids)

    def _prompt_token_ids(self, prompt: str) -> list[int]:
        if prompt == self._cached_prompt and self._cached_prompt_token_ids:
            return list(self._cached_prompt_token_ids)
        prompt_text = QWEN3_USER_PROMPT_TEMPLATE.format(prompt=prompt)
        try:
            return self._llm.tokenize(
                prompt_text.encode("utf-8"),
                add_bos=True,
                special=True,
            )
        except Exception as exc:
            raise ModelBackendError(f"failed to tokenize Qwen prompt: {exc}") from exc

    def _build_blocked_token_ids(self) -> set[int]:
        blocked: set[int] = set()
        for token_getter in (self._llm.token_bos, self._llm.token_eos):
            try:
                token_id = int(token_getter())
            except Exception:
                continue
            if token_id >= 0:
                blocked.add(token_id)

        for special_text in ("<|im_start|>", "<|im_end|>", "<|endoftext|>"):
            try:
                token_ids = self._llm.tokenize(
                    special_text.encode("utf-8"),
                    add_bos=False,
                    special=True,
                )
            except Exception:
                continue
            if len(token_ids) == 1:
                blocked.add(int(token_ids[0]))
        return blocked

    def _metadata_hash(self, model_path: Path) -> str:
        payload: dict[str, Any] = {
            "path": str(model_path.resolve()),
            "size": model_path.stat().st_size,
            "llama_metadata": getattr(self._llm, "metadata", {}),
            "prompt_template": "qwen3-user-assistant-v1",
        }
        return sha256(
            json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
        ).hexdigest()
