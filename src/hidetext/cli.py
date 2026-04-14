from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from .config import RuntimeConfig
from .decoder import StegoDecoder
from .encoder import StegoEncoder
from .errors import ModelBackendError
from .model_backend import ToyCharBackend
from .model_assets import (
    DEFAULT_BACKEND,
    DEFAULT_BATCH_SIZE,
    DEFAULT_BODY_TOKEN_BUDGET,
    DEFAULT_CTX_SIZE,
    DEFAULT_HEADER_TOKEN_BUDGET,
    DEFAULT_LOW_ENTROPY_THRESHOLD_BITS,
    DEFAULT_LOW_ENTROPY_WINDOW_TOKENS,
    DEFAULT_MAX_CANDIDATES,
    DEFAULT_MAX_ENCODE_ATTEMPTS,
    DEFAULT_MIN_ENTROPY_BITS,
    DEFAULT_MODEL_ID,
    DEFAULT_NATURAL_TAIL_MAX_TOKENS,
    DEFAULT_PROGRESS_TOKEN_INTERVAL,
    DEFAULT_SEED,
    DEFAULT_STALL_PATIENCE_TOKENS,
    DEFAULT_TOP_P,
    DEFAULT_TOTAL_FREQUENCY,
    resolve_default_model_path,
)
from .progress import ProgressSnapshot


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="hidetext")
    subparsers = parser.add_subparsers(dest="command", required=True)

    for name in ("encode", "decode"):
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
            choices=("llama-cpp", "toy"),
            default=DEFAULT_BACKEND,
            help="default is the real local llama.cpp backend; use toy only for tests",
        )
        subparser.add_argument(
            "--model-path",
            help="optional GGUF path; otherwise HideText reuses or downloads the default model",
        )
        subparser.add_argument(
            "--model-id",
            help="optional model identifier for protocol metadata; custom model paths infer this by default",
        )
        subparser.add_argument("--threads", type=int)
        subparser.add_argument("--ctx-size", type=int, default=DEFAULT_CTX_SIZE)
        subparser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
        subparser.add_argument("--top-p", type=float, default=DEFAULT_TOP_P)
        subparser.add_argument("--max-candidates", type=int, default=DEFAULT_MAX_CANDIDATES)
        subparser.add_argument(
            "--min-entropy-bits",
            type=float,
            default=DEFAULT_MIN_ENTROPY_BITS,
        )
        subparser.add_argument("--totfreq", type=int, default=DEFAULT_TOTAL_FREQUENCY)
        subparser.add_argument(
            "--header-token-budget",
            type=int,
            default=DEFAULT_HEADER_TOKEN_BUDGET,
        )
        subparser.add_argument(
            "--body-token-budget",
            type=int,
            default=DEFAULT_BODY_TOKEN_BUDGET,
        )
        subparser.add_argument(
            "--natural-tail-max-tokens",
            type=int,
            default=DEFAULT_NATURAL_TAIL_MAX_TOKENS,
        )
        subparser.add_argument(
            "--stall-patience-tokens",
            type=int,
            default=DEFAULT_STALL_PATIENCE_TOKENS,
        )
        subparser.add_argument(
            "--low-entropy-window-tokens",
            type=int,
            default=DEFAULT_LOW_ENTROPY_WINDOW_TOKENS,
        )
        subparser.add_argument(
            "--low-entropy-threshold-bits",
            type=float,
            default=DEFAULT_LOW_ENTROPY_THRESHOLD_BITS,
        )
        subparser.add_argument(
            "--max-encode-attempts",
            type=int,
            default=DEFAULT_MAX_ENCODE_ATTEMPTS,
        )
        subparser.add_argument(
            "--quiet",
            action="store_true",
            help="disable progress logs",
        )
        subparser.add_argument(
            "--progress-token-interval",
            type=int,
            default=DEFAULT_PROGRESS_TOKEN_INTERVAL,
        )

    encode_parser = subparsers.choices["encode"]
    encode_parser.add_argument("--message", required=True)
    encode_parser.add_argument("--json", action="store_true")

    decode_parser = subparsers.choices["decode"]
    decode_text_group = decode_parser.add_mutually_exclusive_group(required=False)
    decode_text_group.add_argument("--text")
    decode_text_group.add_argument("--text-file")
    decode_parser.add_argument("--json", action="store_true")

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
        return DEFAULT_SEED
    raw_text = _read_text_file(args.seed_file).strip()
    try:
        return int(raw_text)
    except ValueError as exc:
        raise ValueError(f"seed file {args.seed_file!r} does not contain a valid integer") from exc


def _resolve_decode_text(args: argparse.Namespace) -> str:
    if getattr(args, "text", None) is not None or getattr(args, "text_file", None) is not None:
        return _resolve_text_value(
            getattr(args, "text", None),
            getattr(args, "text_file", None),
            label="text",
        )
    if sys.stdin.isatty():
        raise ValueError("decode requires --text, --text-file, or stego text on stdin")
    text = sys.stdin.read().rstrip("\r\n")
    if not text:
        raise ValueError("decode requires non-empty stego text on stdin")
    return text


def _write_plain_output(text: str) -> None:
    sys.stdout.write(text)
    sys.stdout.flush()


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
            natural_tail_max_tokens=args.natural_tail_max_tokens,
            stall_patience_tokens=args.stall_patience_tokens,
            low_entropy_window_tokens=args.low_entropy_window_tokens,
            low_entropy_threshold_bits=args.low_entropy_threshold_bits,
            max_encode_attempts=args.max_encode_attempts,
        ),
    )


def _build_backend(args: argparse.Namespace, *, seed: int):
    if args.backend == "toy":
        return ToyCharBackend()
    if args.backend == "llama-cpp":
        try:
            import llama_cpp  # noqa: F401
        except ImportError as exc:
            raise ModelBackendError(
                "llama-cpp backend requires llama-cpp-python; install hidetext[llm]"
            ) from exc
        if not args.model_path:
            resolved_model = resolve_default_model_path()
        else:
            resolved_model = resolve_default_model_path(args.model_path)
        from .llama_cpp_backend import LlamaCppBackendConfig, QwenLlamaCppBackend

        if resolved_model.source == "downloaded":
            print(
                f"[hidetext] downloaded default model to {resolved_model.path}",
                file=sys.stderr,
                flush=True,
            )
        model_id = _resolve_model_id(args, resolved_model_source=resolved_model.source)
        return QwenLlamaCppBackend(
            LlamaCppBackendConfig(
                model_path=str(resolved_model.path),
                model_id=model_id,
                n_ctx=args.ctx_size,
                n_batch=args.batch_size,
                n_threads=args.threads,
                seed=seed,
            )
        )
    raise ValueError(f"unsupported backend: {args.backend}")


def _resolve_model_id(args: argparse.Namespace, *, resolved_model_source: str) -> str | None:
    if getattr(args, "model_id", None):
        return args.model_id
    if resolved_model_source in ("cache", "downloaded"):
        return DEFAULT_MODEL_ID
    return None


def _build_progress_reporter(args: argparse.Namespace):
    if args.quiet:
        return None

    interval = max(1, args.progress_token_interval)
    state = {
        "last_total_tokens": -1,
        "last_segment": None,
        "last_phase": None,
        "active_line": False,
    }

    def _format_clock(seconds: float | None) -> str:
        if seconds is None:
            return "??:??"
        total = max(0, int(seconds))
        minutes, sec = divmod(total, 60)
        hours, minutes = divmod(minutes, 60)
        if hours > 0:
            return f"{hours:02d}:{minutes:02d}:{sec:02d}"
        return f"{minutes:02d}:{sec:02d}"

    def _format_progress_line(snapshot: ProgressSnapshot) -> str:
        if snapshot.segment_bits_total <= 0:
            ratio = 1.0 if snapshot.finished else 0.0
        else:
            ratio = snapshot.segment_bits_done / float(snapshot.segment_bits_total)
        ratio = min(1.0, max(0.0, ratio))
        bar_width = 16
        filled = int(ratio * bar_width)
        bar = ("#" * filled) + ("-" * (bar_width - filled))
        remaining_tokens = max(0, snapshot.token_budget - snapshot.segment_tokens)
        eta_seconds = None
        if snapshot.finished:
            eta_seconds = 0.0
        elif snapshot.tokens_per_second > 0.0:
            eta_seconds = remaining_tokens / snapshot.tokens_per_second
        overall_total = "?" if snapshot.overall_bits_total is None else str(snapshot.overall_bits_total)
        return (
            f"{snapshot.phase}/{snapshot.segment_name} "
            f"{ratio * 100:6.2f}%|{bar}| "
            f"{snapshot.segment_bits_done:.1f}/{snapshot.segment_bits_total}b "
            f"[{_format_clock(snapshot.elapsed_seconds)}<{_format_clock(eta_seconds)}, "
            f"{snapshot.tokens_per_second:.2f} tok/s, "
            f"bpt {snapshot.bits_per_token:.3f}, tok {snapshot.total_tokens}, "
            f"all {snapshot.overall_bits_done:.1f}/{overall_total}b]"
        )

    def report(snapshot: ProgressSnapshot) -> None:
        segment_changed = (
            snapshot.segment_name != state["last_segment"]
            or snapshot.phase != state["last_phase"]
        )
        should_print = snapshot.finished or segment_changed
        if state["last_total_tokens"] < 0:
            should_print = True
        elif snapshot.total_tokens - state["last_total_tokens"] >= interval:
            should_print = True
        if not should_print:
            return

        state["last_total_tokens"] = snapshot.total_tokens
        state["last_segment"] = snapshot.segment_name
        state["last_phase"] = snapshot.phase

        use_carriage = sys.stderr.isatty()
        if use_carriage and segment_changed and state["active_line"]:
            print(file=sys.stderr, flush=True)
            state["active_line"] = False

        line = _format_progress_line(snapshot)
        end = "\r" if use_carriage and not snapshot.finished else "\n"
        print(line, file=sys.stderr, end=end, flush=True)
        state["active_line"] = bool(use_carriage and not snapshot.finished)

    return report


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
    except (ValueError, ModelBackendError) as exc:
        parser.error(str(exc))
    progress_reporter = _build_progress_reporter(args)

    if args.command == "encode":
        retry_notice = None if args.quiet else (lambda line: print(line, file=sys.stderr, flush=True))
        result = StegoEncoder(backend, config).encode(
            args.message,
            passphrase=passphrase,
            prompt=prompt,
            progress_callback=progress_reporter,
            retry_notice_callback=retry_notice,
        )
        if args.json:
            print(
                json.dumps(
                    {
                        "text": result.text,
                        "backend": args.backend,
                        "config_fingerprint": f"{result.config_fingerprint:016x}",
                        "packet_len": len(result.packet),
                        "total_tokens": result.total_tokens,
                        "packet_tokens": result.packet_tokens,
                        "tail_tokens": result.tail_tokens,
                        "attempts_used": result.attempts_used,
                        "elapsed_seconds": round(result.elapsed_seconds, 4),
                        "tokens_per_second": round(result.tokens_per_second, 4),
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
        else:
            _write_plain_output(result.text)
        return

    if args.command == "decode":
        stego_text = _resolve_decode_text(args)
        result = StegoDecoder(backend, config).decode(
            stego_text,
            passphrase=passphrase,
            prompt=prompt,
            progress_callback=progress_reporter,
        )
        if args.json:
            print(
                json.dumps(
                    {
                        "plaintext": result.plaintext,
                        "backend": args.backend,
                        "consumed_tokens": result.consumed_tokens,
                        "trailing_tokens": result.trailing_tokens,
                        "packet_len": len(result.packet),
                        "elapsed_seconds": round(result.elapsed_seconds, 4),
                        "tokens_per_second": round(result.tokens_per_second, 4),
                        "bits_per_token": round(result.bits_per_token, 4),
                    },
                    ensure_ascii=False,
                    indent=2,
                )
            )
        else:
            _write_plain_output(result.plaintext)
        return

    raise AssertionError(f"unsupported command: {args.command}")


if __name__ == "__main__":
    main()
