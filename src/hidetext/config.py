from __future__ import annotations

from dataclasses import asdict, dataclass, field
from hashlib import sha256
import json


PROTOCOL_VERSION = 1
CODEC_VERSION = "finite-interval-v1"
CANDIDATE_POLICY_VERSION = "top-p-maxk-retoken-stable-v2"


@dataclass(frozen=True)
class CandidatePolicyConfig:
    top_p: float = 0.97
    max_candidates: int = 16
    min_entropy_bits: float = 1.0
    enforce_retokenization_stability: bool = True

    def as_protocol_dict(self) -> dict[str, object]:
        return {
            "version": CANDIDATE_POLICY_VERSION,
            "top_p": self.top_p,
            "max_candidates": self.max_candidates,
            "min_entropy_bits": self.min_entropy_bits,
            "enforce_retokenization_stability": self.enforce_retokenization_stability,
        }


@dataclass(frozen=True)
class CodecConfig:
    total_frequency: int = 65536
    max_header_tokens: int = 2048
    max_body_tokens: int = 4096
    natural_tail_max_tokens: int = 64
    stall_patience_tokens: int = 256
    low_entropy_window_tokens: int = 32
    low_entropy_threshold_bits: float = 0.1
    max_encode_attempts: int = 3

    def as_protocol_dict(self) -> dict[str, object]:
        return {
            "version": CODEC_VERSION,
            "total_frequency": self.total_frequency,
        }


@dataclass(frozen=True)
class CryptoConfig:
    kdf_n: int = 2**14
    kdf_r: int = 8
    kdf_p: int = 1
    salt_len: int = 16
    nonce_len: int = 12

    def as_protocol_dict(self) -> dict[str, object]:
        return {
            "kdf": "scrypt",
            "aead": "chacha20-poly1305",
            "kdf_n": self.kdf_n,
            "kdf_r": self.kdf_r,
            "kdf_p": self.kdf_p,
            "salt_len": self.salt_len,
            "nonce_len": self.nonce_len,
        }


@dataclass(frozen=True)
class RuntimeConfig:
    seed: int = 7
    candidate_policy: CandidatePolicyConfig = field(default_factory=CandidatePolicyConfig)
    codec: CodecConfig = field(default_factory=CodecConfig)
    crypto: CryptoConfig = field(default_factory=CryptoConfig)

    def as_protocol_dict(self) -> dict[str, object]:
        return {
            "protocol_version": PROTOCOL_VERSION,
            "seed": self.seed,
            "candidate_policy": self.candidate_policy.as_protocol_dict(),
            "codec": self.codec.as_protocol_dict(),
            "crypto": self.crypto.as_protocol_dict(),
        }

    def config_fingerprint(
        self,
        *,
        backend_metadata: dict[str, object],
        prompt: str,
    ) -> int:
        payload = {
            "runtime": self.as_protocol_dict(),
            "backend": backend_metadata,
            "prompt_sha256": sha256(prompt.encode("utf-8")).hexdigest(),
        }
        blob = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return int.from_bytes(sha256(blob).digest()[:8], "big")

    def to_json(self) -> str:
        return json.dumps(asdict(self), sort_keys=True, indent=2)
