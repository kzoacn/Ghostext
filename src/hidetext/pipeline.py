from __future__ import annotations

from .candidate_policy import select_candidates
from .config import RuntimeConfig
from .model_backend import TextBackend
from .quantization import QuantizedDistribution, quantize_candidates


def prepare_quantized_distribution(
    backend: TextBackend,
    *,
    prompt: str,
    generated_tokens: list[str],
    config: RuntimeConfig,
) -> QuantizedDistribution:
    distribution = backend.distribution(prompt, generated_tokens, config.seed)
    selection = select_candidates(distribution, config.candidate_policy)
    return quantize_candidates(selection, config.codec.total_frequency)
