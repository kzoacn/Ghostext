from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from hashlib import sha256
import json
import os
from pathlib import Path
import re
import sys
import tempfile
from typing import Any

import numpy as np

from .errors import ModelBackendError
from .model_assets import DEFAULT_MODEL_ID
from .model_backend import BackendMetadata, RawNextTokenDistribution


@dataclass(frozen=True)
class PromptTemplateSpec:
    template_id: str
    template_text: str

    def render(self, prompt: str) -> str:
        return self.template_text.format(prompt=prompt)


QWEN3_PROMPT_TEMPLATE = PromptTemplateSpec(
    template_id="qwen3-user-assistant-v1",
    template_text="<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n",
)
QWEN35_PROMPT_TEMPLATE = PromptTemplateSpec(
    template_id="qwen3_5-user-assistant-no-thinking-v1",
    template_text=(
        "<|im_start|>user\n{prompt}<|im_end|>\n"
        "<|im_start|>assistant\n<think>\n\n</think>\n\n"
    ),
)


@dataclass(frozen=True)
class LlamaCppBackendConfig:
    model_path: str
    model_id: str | None = DEFAULT_MODEL_ID
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
                "llama-cpp backend requires llama-cpp-python; install ghostext[llm]"
            ) from exc

        self.config = config
        model_path = Path(config.model_path)
        if not model_path.exists():
            raise ModelBackendError(f"model file not found: {model_path}")

        n_threads = config.n_threads or max(1, os.cpu_count() or 1)
        self._llm = _build_llama_with_filtered_stderr(
            Llama=Llama,
            model_path=str(model_path),
            seed=config.seed,
            n_ctx=config.n_ctx,
            n_batch=config.n_batch,
            n_threads=n_threads,
            use_mmap=config.use_mmap,
            verbose=config.verbose,
        )
        self._n_vocab = int(self._llm.n_vocab())
        self._cached_prompt: str | None = None
        self._cached_prompt_token_ids: list[int] = []
        self._cached_generated_token_ids: list[int] = []
        self._blocked_token_ids = self._build_blocked_token_ids()
        llama_metadata = getattr(self._llm, "metadata", {})
        self._prompt_template = resolve_qwen_prompt_template(
            config.model_id,
            llama_metadata,
        )
        inferred_model_id = config.model_id or _infer_model_id(
            model_path,
            llama_metadata,
        )
        self._metadata = BackendMetadata(
            model_id=inferred_model_id,
            tokenizer_hash=build_llama_cpp_tokenizer_hash(
                model_path,
                llama_metadata,
                prompt_template_id=self._prompt_template.template_id,
            ),
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
        prompt_text = self._prompt_template.render(prompt)
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


def _is_capacity_warning(text: str) -> bool:
    normalized = text.strip().lower()
    return bool(
        re.search(r"llama_context:\s*n_ctx_seq.*<\s*n_ctx_train", normalized)
        or "full capacity of the model will not be utilized" in normalized
    )

@contextmanager
def _redirect_stderr_fd(target_fd: int):
    saved_fd = os.dup(2)
    try:
        os.dup2(target_fd, 2)
        yield
    finally:
        os.dup2(saved_fd, 2)
        os.close(saved_fd)


def _build_llama_with_filtered_stderr(
    *,
    Llama,
    model_path: str,
    seed: int,
    n_ctx: int,
    n_batch: int,
    n_threads: int,
    use_mmap: bool,
    verbose: bool,
):
    if verbose:
        return Llama(
            model_path=model_path,
            n_gpu_layers=0,
            seed=seed,
            n_ctx=n_ctx,
            n_batch=n_batch,
            n_threads=n_threads,
            logits_all=False,
            use_mmap=use_mmap,
            verbose=verbose,
        )

    with tempfile.TemporaryFile(mode="w+b") as stderr_capture:
        with _redirect_stderr_fd(stderr_capture.fileno()):
            llm = Llama(
                model_path=model_path,
                n_gpu_layers=0,
                seed=seed,
                n_ctx=n_ctx,
                n_batch=n_batch,
                n_threads=n_threads,
                logits_all=False,
                use_mmap=use_mmap,
                verbose=verbose,
            )
        stderr_capture.seek(0)
        captured = stderr_capture.read().decode("utf-8", errors="replace")

    for line in captured.splitlines():
        if _is_capacity_warning(line):
            continue
        print(line, file=sys.stderr, flush=True)
    return llm


def build_llama_cpp_tokenizer_hash(
    model_path: Path,
    llama_metadata: dict[str, Any],
    *,
    prompt_template_id: str,
) -> str:
    payload: dict[str, Any] = {
        "size": model_path.stat().st_size,
        "head_sha256": _partial_file_hash(model_path, from_start=True),
        "tail_sha256": _partial_file_hash(model_path, from_start=False),
        "llama_metadata": llama_metadata,
        "prompt_template": prompt_template_id,
    }
    return sha256(
        json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()


def _partial_file_hash(model_path: Path, *, from_start: bool, chunk_size: int = 1 << 20) -> str:
    with model_path.open("rb") as handle:
        if not from_start:
            size = model_path.stat().st_size
            handle.seek(max(0, size - chunk_size))
        data = handle.read(chunk_size)
    return sha256(data).hexdigest()


def _infer_model_id(model_path: Path, llama_metadata: dict[str, Any]) -> str:
    for key in ("general.name", "general.basename"):
        value = llama_metadata.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return model_path.stem


def resolve_qwen_prompt_template(
    model_id: str | None,
    llama_metadata: dict[str, Any],
) -> PromptTemplateSpec:
    model_markers = []
    if model_id:
        model_markers.append(model_id.lower())
    for key in ("general.name", "general.basename"):
        value = llama_metadata.get(key)
        if isinstance(value, str):
            model_markers.append(value.lower())
    if any("qwen3.5" in marker for marker in model_markers):
        return QWEN35_PROMPT_TEMPLATE
    return QWEN3_PROMPT_TEMPLATE
