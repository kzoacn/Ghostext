from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import RuntimeConfig
from .decoder import StegoDecoder
from .encoder import StegoEncoder
from .model_backend import ToyCharBackend


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="hidetext")
    subparsers = parser.add_subparsers(dest="command", required=True)

    for name in ("encode", "decode", "eval"):
        subparser = subparsers.add_parser(name)
        prompt_group = subparser.add_mutually_exclusive_group(required=True)
        prompt_group.add_argument("--prompt")
        prompt_group.add_argument("--prompt-file")

        passphrase_group = subparser.add_mutually_exclusive_group(required=True)
        passphrase_group.add_argument("--passphrase")
        passphrase_group.add_argument("--passphrase-file")

        seed_group = subparser.add_mutually_exclusive_group(required=False)
        seed_group.add_argument("--seed", type=int)
        seed_group.add_argument("--seed-file")
        subparser.add_argument(
            "--backend",
            choices=("toy", "llama-cpp"),
            default="toy",
        )
        subparser.add_argument("--model-path")
        subparser.add_argument("--threads", type=int)
        subparser.add_argument("--ctx-size", type=int, default=4096)
        subparser.add_argument("--batch-size", type=int, default=256)
        subparser.add_argument("--top-p", type=float, default=0.97)
        subparser.add_argument("--max-candidates", type=int, default=16)
        subparser.add_argument("--min-entropy-bits", type=float, default=1.0)
        subparser.add_argument("--totfreq", type=int, default=65536)
        subparser.add_argument("--header-token-budget", type=int, default=1024)
        subparser.add_argument("--body-token-budget", type=int, default=4096)

    encode_parser = subparsers.choices["encode"]
    encode_parser.add_argument("--message", required=True)

    decode_parser = subparsers.choices["decode"]
    decode_parser.add_argument("--text", required=True)

    eval_parser = subparsers.choices["eval"]
    eval_parser.add_argument("--message", required=True)
    return parser


def _read_text_file(path_str: str) -> str:
    try:
        return Path(path_str).read_text(encoding="utf-8")
    except OSError as exc:
        raise ValueError(f"failed to read file {path_str!r}: {exc}") from exc


def _resolve_text_value(
    direct_value: str | None,
    file_value: str | None,
    *,
    label: str,
    strip_trailing_newlines: bool = True,
) -> str:
    if direct_value is not None:
        return direct_value
    if file_value is None:
        raise ValueError(f"missing {label}")
    text = _read_text_file(file_value)
    if strip_trailing_newlines:
        text = text.rstrip("\r\n")
    return text


def _resolve_seed(args: argparse.Namespace) -> int:
    if args.seed is not None:
        return args.seed
    if args.seed_file is None:
        return 7
    raw_text = _read_text_file(args.seed_file).strip()
    try:
        return int(raw_text)
    except ValueError as exc:
        raise ValueError(f"seed file {args.seed_file!r} does not contain a valid integer") from exc


def _config_from_args(args: argparse.Namespace, *, seed: int) -> RuntimeConfig:
    from .config import CandidatePolicyConfig, CodecConfig

    return RuntimeConfig(
        seed=seed,
        candidate_policy=CandidatePolicyConfig(
            top_p=args.top_p,
            max_candidates=args.max_candidates,
            min_entropy_bits=args.min_entropy_bits,
        ),
        codec=CodecConfig(
            total_frequency=args.totfreq,
            max_header_tokens=args.header_token_budget,
            max_body_tokens=args.body_token_budget,
        ),
    )


def _build_backend(args: argparse.Namespace, *, seed: int):
    if args.backend == "toy":
        return ToyCharBackend()
    if args.backend == "llama-cpp":
        if not args.model_path:
            raise ValueError("--model-path is required when --backend llama-cpp is used")
        from .llama_cpp_backend import LlamaCppBackendConfig, QwenLlamaCppBackend

        return QwenLlamaCppBackend(
            LlamaCppBackendConfig(
                model_path=args.model_path,
                n_ctx=args.ctx_size,
                n_batch=args.batch_size,
                n_threads=args.threads,
                seed=seed,
            )
        )
    raise ValueError(f"unsupported backend: {args.backend}")


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    try:
        prompt = _resolve_text_value(args.prompt, args.prompt_file, label="prompt")
        passphrase = _resolve_text_value(
            args.passphrase,
            args.passphrase_file,
            label="passphrase",
        )
        seed = _resolve_seed(args)
        config = _config_from_args(args, seed=seed)
        backend = _build_backend(args, seed=seed)
    except ValueError as exc:
        parser.error(str(exc))

    if args.command == "encode":
        result = StegoEncoder(backend, config).encode(
            args.message,
            passphrase=passphrase,
            prompt=prompt,
        )
        print(
            json.dumps(
                {
                    "text": result.text,
                    "backend": args.backend,
                    "config_fingerprint": f"{result.config_fingerprint:016x}",
                    "packet_len": len(result.packet),
                    "total_tokens": result.total_tokens,
                    "bits_per_token": round(result.bits_per_token, 4),
                    "segments": [
                        {
                            "name": segment.name,
                            "tokens_used": segment.tokens_used,
                            "encoding_steps": segment.encoding_steps,
                            "embedded_bits": round(segment.embedded_bits, 4),
                        }
                        for segment in result.segment_stats
                    ],
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return

    if args.command == "decode":
        result = StegoDecoder(backend, config).decode(
            args.text,
            passphrase=passphrase,
            prompt=prompt,
        )
        print(
            json.dumps(
                {
                    "plaintext": result.plaintext,
                    "backend": args.backend,
                    "consumed_tokens": result.consumed_tokens,
                    "packet_len": len(result.packet),
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return

    if args.command == "eval":
        encoder = StegoEncoder(backend, config)
        decoder = StegoDecoder(backend, config)
        encoded = encoder.encode(args.message, passphrase=passphrase, prompt=prompt)
        decoded = decoder.decode(encoded.text, passphrase=passphrase, prompt=prompt)
        print(
            json.dumps(
                {
                    "roundtrip_ok": decoded.plaintext == args.message,
                    "backend": args.backend,
                    "text": encoded.text,
                    "packet_len": len(encoded.packet),
                    "total_tokens": encoded.total_tokens,
                    "bits_per_token": round(encoded.bits_per_token, 4),
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return

    raise AssertionError(f"unsupported command: {args.command}")


if __name__ == "__main__":
    main()
