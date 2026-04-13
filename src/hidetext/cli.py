from __future__ import annotations

import argparse
import json

from .config import RuntimeConfig
from .decoder import StegoDecoder
from .encoder import StegoEncoder
from .model_backend import ToyCharBackend


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="hidetext")
    subparsers = parser.add_subparsers(dest="command", required=True)

    for name in ("encode", "decode", "eval"):
        subparser = subparsers.add_parser(name)
        subparser.add_argument("--prompt", required=True)
        subparser.add_argument("--passphrase", required=True)
        subparser.add_argument("--seed", type=int, default=7)

    encode_parser = subparsers.choices["encode"]
    encode_parser.add_argument("--message", required=True)

    decode_parser = subparsers.choices["decode"]
    decode_parser.add_argument("--text", required=True)

    eval_parser = subparsers.choices["eval"]
    eval_parser.add_argument("--message", required=True)
    return parser


def _config_from_args(args: argparse.Namespace) -> RuntimeConfig:
    return RuntimeConfig(seed=args.seed)


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    backend = ToyCharBackend()
    config = _config_from_args(args)

    if args.command == "encode":
        result = StegoEncoder(backend, config).encode(
            args.message,
            passphrase=args.passphrase,
            prompt=args.prompt,
        )
        print(
            json.dumps(
                {
                    "text": result.text,
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
            passphrase=args.passphrase,
            prompt=args.prompt,
        )
        print(
            json.dumps(
                {
                    "plaintext": result.plaintext,
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
        encoded = encoder.encode(args.message, passphrase=args.passphrase, prompt=args.prompt)
        decoded = decoder.decode(encoded.text, passphrase=args.passphrase, prompt=args.prompt)
        print(
            json.dumps(
                {
                    "roundtrip_ok": decoded.plaintext == args.message,
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

