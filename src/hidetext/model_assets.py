from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path

from .errors import ModelBackendError


DEFAULT_BACKEND = "llama-cpp"
DEFAULT_MODEL_ID = "Qwen/Qwen3.5-2B"
DEFAULT_MODEL_REPO_ID = "bartowski/Qwen_Qwen3.5-2B-GGUF"
DEFAULT_MODEL_FILENAME = "Qwen_Qwen3.5-2B-Q4_K_S.gguf"
DEFAULT_MODEL_CACHE_SUBDIR = "qwen35-2b-q4ks"

DEFAULT_SEED = 7
DEFAULT_CTX_SIZE = 4096
DEFAULT_BATCH_SIZE = 128
DEFAULT_TOP_P = 0.995
DEFAULT_MAX_CANDIDATES = 64
DEFAULT_MIN_ENTROPY_BITS = 0.0
DEFAULT_TOTAL_FREQUENCY = 4096
DEFAULT_HEADER_TOKEN_BUDGET = 2048
DEFAULT_BODY_TOKEN_BUDGET = 4096
DEFAULT_NATURAL_TAIL_MAX_TOKENS = 64
DEFAULT_STALL_PATIENCE_TOKENS = 256
DEFAULT_LOW_ENTROPY_WINDOW_TOKENS = 32
DEFAULT_LOW_ENTROPY_THRESHOLD_BITS = 0.1
DEFAULT_MAX_ENCODE_ATTEMPTS = 3
DEFAULT_PROGRESS_TOKEN_INTERVAL = 50


@dataclass(frozen=True)
class ResolvedModelPath:
    path: Path
    source: str


def default_model_cache_dir() -> Path:
    override = os.environ.get("HIDETEXT_MODEL_DIR")
    if override:
        return Path(override).expanduser()
    return Path.home() / ".cache" / "hidetext" / "models" / DEFAULT_MODEL_CACHE_SUBDIR


def resolve_default_model_path(explicit_path: str | None = None) -> ResolvedModelPath:
    if explicit_path is not None:
        path = Path(explicit_path).expanduser()
        if not path.exists():
            raise ModelBackendError(f"model file not found: {path}")
        return ResolvedModelPath(path=path, source="explicit")

    for env_var in ("HIDETEXT_MODEL_PATH", "HIDETEXT_LLAMA_MODEL_PATH"):
        env_value = os.environ.get(env_var)
        if env_value is None:
            continue
        path = Path(env_value).expanduser()
        if not path.exists():
            raise ModelBackendError(f"{env_var} points to a missing model file: {path}")
        return ResolvedModelPath(path=path, source="env")

    cache_dir = default_model_cache_dir()
    cached_path = cache_dir / DEFAULT_MODEL_FILENAME
    if cached_path.exists():
        return ResolvedModelPath(path=cached_path, source="cache")

    downloaded_path = _download_default_model(cache_dir)
    return ResolvedModelPath(path=downloaded_path, source="downloaded")


def _download_default_model(cache_dir: Path) -> Path:
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise ModelBackendError(
            "automatic model download requires huggingface_hub; install hidetext[llm]"
        ) from exc

    cache_dir.mkdir(parents=True, exist_ok=True)
    try:
        downloaded = hf_hub_download(
            repo_id=DEFAULT_MODEL_REPO_ID,
            filename=DEFAULT_MODEL_FILENAME,
            local_dir=str(cache_dir),
        )
    except Exception as exc:
        raise ModelBackendError(
            "failed to download the default GGUF model "
            f"{DEFAULT_MODEL_REPO_ID}/{DEFAULT_MODEL_FILENAME}: {exc}"
        ) from exc
    return Path(downloaded)
