#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import statistics
import sys
import time
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
GHOSTEXT_ROOT = SCRIPT_DIR.parent
GHOSTEXT_SRC = GHOSTEXT_ROOT / "src"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if str(GHOSTEXT_SRC) not in sys.path:
    sys.path.insert(0, str(GHOSTEXT_SRC))

from eval_artifact_utils import (  # noqa: E402
    FULL_COVER_TEXT_POLICY,
    bootstrap_ci,
    build_runtime_info,
    extract_model_provenance,
    quantile,
    sha256_hex,
    utc_timestamp,
    wilson_interval,
)
from ghostext.config import CandidatePolicyConfig, CodecConfig, RuntimeConfig  # noqa: E402
from ghostext.decoder import StegoDecoder  # noqa: E402
from ghostext.encoder import StegoEncoder  # noqa: E402
from ghostext.errors import (  # noqa: E402
    EncodingExhaustedError,
    GhostextError,
    IntegrityError,
    LowEntropyRetryLimitError,
    ModelBackendError,
    SynchronizationError,
    UnsafeTokenizationError,
)
from ghostext.llama_cpp_backend import LlamaCppBackendConfig, QwenLlamaCppBackend  # noqa: E402
from ghostext.model_assets import DEFAULT_MODEL_ID, resolve_default_model_path  # noqa: E402


GRID_PASSPHRASE_POLICY = (
    "fixed local evaluation passphrase reused across all 100 prompt-message trials "
    "for reproducibility"
)


@dataclass(frozen=True)
class PromptCase:
    prompt_id: str
    language: str
    family: str
    prompt: str


@dataclass(frozen=True)
class MessageCase:
    message_id: str
    language: str
    length_class: str
    message: str


def build_prompts() -> list[PromptCase]:
    return [
        PromptCase(
            prompt_id="p01_en_evening_walk",
            language="en",
            family="quiet-city",
            prompt=(
                "Write one short, natural English paragraph about a quiet evening walk in a city "
                "neighborhood. Keep it fluent, ordinary, and free of dialogue or bullet points."
            ),
        ),
        PromptCase(
            prompt_id="p02_en_rainy_commute",
            language="en",
            family="rainy-commute",
            prompt=(
                "Write one short, natural English paragraph describing a rainy trip home after work. "
                "Use simple everyday details and keep the tone calm."
            ),
        ),
        PromptCase(
            prompt_id="p03_en_corner_cafe",
            language="en",
            family="corner-cafe",
            prompt=(
                "Write one short, natural English paragraph about sitting in a small corner cafe in "
                "the early evening. Avoid dramatic language and keep it realistic."
            ),
        ),
        PromptCase(
            prompt_id="p04_en_market_street",
            language="en",
            family="street-market",
            prompt=(
                "Write one short, natural English paragraph about walking past a neighborhood market "
                "street near closing time. Keep it plain and observational."
            ),
        ),
        PromptCase(
            prompt_id="p05_en_campus_library",
            language="en",
            family="campus-evening",
            prompt=(
                "Write one short, natural English paragraph about leaving a campus library at dusk. "
                "Focus on small concrete details instead of abstract reflection."
            ),
        ),
        PromptCase(
            prompt_id="p06_zh_evening_subway",
            language="zh",
            family="subway-evening",
            prompt=(
                "请写一段自然、连贯、简短的中文段落，描写傍晚下班后乘地铁回家的普通见闻。"
                "不要用列表，不要写成对话，语气平实。"
            ),
        ),
        PromptCase(
            prompt_id="p07_zh_weekend_market",
            language="zh",
            family="weekend-market",
            prompt=(
                "请写一段自然、连贯、简短的中文段落，描写周末傍晚逛社区菜市场时看到的景象。"
                "用日常口吻，不要夸张。"
            ),
        ),
        PromptCase(
            prompt_id="p08_zh_convenience_store",
            language="zh",
            family="convenience-store",
            prompt=(
                "请写一段自然、连贯、简短的中文段落，描写夜里走进社区便利店时的普通观察。"
                "保持平实自然，不要分点。"
            ),
        ),
        PromptCase(
            prompt_id="p09_zh_campus_track",
            language="zh",
            family="campus-track",
            prompt=(
                "请写一段自然、连贯、简短的中文段落，描写傍晚校园操场边的日常场景。"
                "语言简单，像普通人随手写下的观察。"
            ),
        ),
        PromptCase(
            prompt_id="p10_zh_rainy_alley",
            language="zh",
            family="rainy-alley",
            prompt=(
                "请写一段自然、连贯、简短的中文段落，描写雨后穿过居民区小巷时看到的街景。"
                "不要写成诗歌，保持自然叙述。"
            ),
        ),
    ]


def build_messages() -> list[MessageCase]:
    return [
        MessageCase(
            message_id="m01_en_short_bridge",
            language="en",
            length_class="short",
            message="Meet at 7 PM by the old bridge.",
        ),
        MessageCase(
            message_id="m02_en_short_notebook",
            language="en",
            length_class="short",
            message="Bring the blue notebook and wait near the station entrance.",
        ),
        MessageCase(
            message_id="m03_en_mid_locker",
            language="en",
            length_class="medium",
            message="Locker 14 on the east side contains the spare key for tomorrow morning.",
        ),
        MessageCase(
            message_id="m04_en_mid_route",
            language="en",
            length_class="medium",
            message="Use route B tonight and send a confirmation only after you are indoors.",
        ),
        MessageCase(
            message_id="m05_en_long_backup",
            language="en",
            length_class="long",
            message=(
                "If the cafe is crowded, switch to the bookstore across the street, wait near the "
                "back shelf, and leave after ten minutes if nobody arrives."
            ),
        ),
        MessageCase(
            message_id="m06_zh_short_river",
            language="zh",
            length_class="short",
            message="今晚七点在河边老地方见。",
        ),
        MessageCase(
            message_id="m07_zh_short_station",
            language="zh",
            length_class="short",
            message="带上那本蓝色笔记本，在地铁口等我。",
        ),
        MessageCase(
            message_id="m08_zh_mid_package",
            language="zh",
            length_class="medium",
            message="包裹已经转移到东侧十四号柜，请按原路线行动并在结束后回执。",
        ),
        MessageCase(
            message_id="m09_zh_mid_schedule",
            language="zh",
            length_class="medium",
            message="如果现场人多，就改到街对面的书店门口，十分钟后再决定是否继续等待。",
        ),
        MessageCase(
            message_id="m10_zh_long_backup",
            language="zh",
            length_class="long",
            message=(
                "明早的备用钥匙已经放进前台左侧第二个抽屉，拿到后不要停留，沿着上次那条路离开，"
                "到安全位置再发确认。"
            ),
        ),
    ]


def classify_failure(exc: Exception) -> str:
    if isinstance(exc, LowEntropyRetryLimitError):
        return "low_entropy_retry_exhaustion"
    if isinstance(exc, UnsafeTokenizationError):
        return "unstable_tokenization_exhaustion"
    if isinstance(exc, EncodingExhaustedError):
        return "token_budget_exhaustion"
    if isinstance(exc, SynchronizationError):
        return "synchronization_mismatch"
    if isinstance(exc, IntegrityError):
        return "integrity_mismatch"
    if isinstance(exc, ModelBackendError):
        return "backend_error"
    if isinstance(exc, GhostextError):
        return "ghostext_error"
    return "unexpected_error"


def attempts_histogram(rows: list[dict[str, Any]]) -> dict[str, int]:
    histogram: dict[str, int] = {}
    for item in rows:
        attempts = str(int(item["attempts_used"]))
        histogram[attempts] = histogram.get(attempts, 0) + 1
    return histogram


def failure_histogram(rows: list[dict[str, Any]]) -> dict[str, int]:
    histogram: dict[str, int] = {}
    for item in rows:
        key = item.get("failure_class") or "unknown"
        histogram[key] = histogram.get(key, 0) + 1
    return histogram


def summarize_float(values: list[float]) -> dict[str, float]:
    if not values:
        return {
            "mean": 0.0,
            "median": 0.0,
            "min": 0.0,
            "max": 0.0,
            "p90": 0.0,
            "p99": 0.0,
            "bootstrap_ci_95": [0.0, 0.0],
        }
    return {
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "min": min(values),
        "max": max(values),
        "p90": quantile(values, 0.9),
        "p99": quantile(values, 0.99),
        "bootstrap_ci_95": list(bootstrap_ci(values, statistics.fmean)),
    }


def summarize_subset(rows: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(rows)
    success = [item for item in rows if item.get("status") == "success" and item.get("decode_match")]
    failures = [item for item in rows if item not in success]
    encode_lat = [item["encode_wall_seconds"] for item in success]
    decode_lat = [item["decode_wall_seconds"] for item in success]
    e2e_lat = [item["e2e_elapsed_seconds"] for item in success]
    bits_per_token = [item["bits_per_token_encode"] for item in success]
    net_bits_per_token = [8 * item["plaintext_bytes"] / item["total_tokens"] for item in success]
    packet_bytes = [item["packet_len"] for item in success]
    total_tokens = [item["total_tokens"] for item in success]
    encode_tps = [item["encode_tokens_per_second"] for item in success]
    decode_tps = [item["decode_tokens_per_second"] for item in success]
    attempts = [item["attempts_used"] for item in success]
    wilson = wilson_interval(len(success), total)
    return {
        "trials": total,
        "success_count": len(success),
        "failure_count": len(failures),
        "decode_success_rate": (len(success) / total) if total else 0.0,
        "decode_success_rate_wilson_95": [wilson[0], wilson[1]],
        "failure_histogram": failure_histogram(failures),
        "encode_latency_seconds": summarize_float(encode_lat),
        "decode_latency_seconds": summarize_float(decode_lat),
        "end_to_end_latency_seconds": summarize_float(e2e_lat),
        "bits_per_token_encode": summarize_float(bits_per_token),
        "net_plaintext_bits_per_token": summarize_float(net_bits_per_token),
        "packet_len_bytes": summarize_float(packet_bytes),
        "total_tokens": summarize_float(total_tokens),
        "encode_tokens_per_second": summarize_float(encode_tps),
        "decode_tokens_per_second": summarize_float(decode_tps),
        "attempts_used": {
            "mean": statistics.fmean(attempts) if attempts else 0.0,
            "median": statistics.median(attempts) if attempts else 0.0,
            "max": max(attempts) if attempts else 0,
            "histogram": attempts_histogram(success),
        },
    }


def load_existing_runs(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def render_markdown(summary: dict[str, Any]) -> str:
    metrics = summary["metrics"]
    model = summary["model"]
    runtime = summary["runtime"]
    backend_metadata = model.get("backend_metadata", {})
    lines = [
        "# Prompt-Message Grid Summary",
        "",
        f"Generated at `{summary['generated_at_utc']}`.",
        "",
        "## Overall",
        "",
        "| Metric | Value |",
        "|---|---|",
        (
            f"| Decode success rate | "
            f"{metrics['success_count']}/{summary['dataset']['completed_trials']}="
            f"{metrics['decode_success_rate']:.3f} "
            f"(Wilson 95% CI: [{metrics['decode_success_rate_wilson_95'][0]:.3f}, "
            f"{metrics['decode_success_rate_wilson_95'][1]:.3f}]) |"
        ),
        f"| Failure histogram | `{json.dumps(metrics['failure_histogram'], ensure_ascii=False)}` |",
        (
            f"| Encode latency (s) | mean {metrics['encode_latency_seconds']['mean']:.2f}, "
            f"median {metrics['encode_latency_seconds']['median']:.2f}, "
            f"p90 {metrics['encode_latency_seconds']['p90']:.2f}, "
            f"p99 {metrics['encode_latency_seconds']['p99']:.2f} |"
        ),
        (
            f"| Decode latency (s) | mean {metrics['decode_latency_seconds']['mean']:.2f}, "
            f"median {metrics['decode_latency_seconds']['median']:.2f}, "
            f"p90 {metrics['decode_latency_seconds']['p90']:.2f}, "
            f"p99 {metrics['decode_latency_seconds']['p99']:.2f} |"
        ),
        (
            f"| End-to-end latency (s) | mean {metrics['end_to_end_latency_seconds']['mean']:.2f}, "
            f"median {metrics['end_to_end_latency_seconds']['median']:.2f} |"
        ),
        (
            f"| Encode bits/token | mean {metrics['bits_per_token_encode']['mean']:.3f}, "
            f"median {metrics['bits_per_token_encode']['median']:.3f}, "
            f"min {metrics['bits_per_token_encode']['min']:.3f}, "
            f"max {metrics['bits_per_token_encode']['max']:.3f} |"
        ),
        (
            f"| Packet length (bytes) | mean {metrics['packet_len_bytes']['mean']:.1f}, "
            f"median {metrics['packet_len_bytes']['median']:.1f}, "
            f"min {metrics['packet_len_bytes']['min']:.1f}, "
            f"max {metrics['packet_len_bytes']['max']:.1f} |"
        ),
        (
            f"| Attempts used | mean {metrics['attempts_used']['mean']:.2f}, "
            f"median {metrics['attempts_used']['median']:.2f}, "
            f"max {metrics['attempts_used']['max']} |"
        ),
        "",
        "## By Prompt Language",
        "",
        "| Language | Trials | Success | Encode median (s) | Decode median (s) | Mean bits/token | Attempts mean |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for language, item in summary["breakdowns"]["by_prompt_language"].items():
        lines.append(
            f"| {language} | {item['trials']} | {item['success_count']}/{item['trials']} | "
            f"{item['encode_latency_seconds']['median']:.2f} | "
            f"{item['decode_latency_seconds']['median']:.2f} | "
            f"{item['bits_per_token_encode']['mean']:.3f} | "
            f"{item['attempts_used']['mean']:.2f} |"
        )

    lines.extend(
        [
            "",
            "## By Message Length",
            "",
            "| Length | Trials | Success | Encode median (s) | Decode median (s) | Mean bits/token | Mean packet bytes |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for length_class, item in summary["breakdowns"]["by_message_length"].items():
        lines.append(
            f"| {length_class} | {item['trials']} | {item['success_count']}/{item['trials']} | "
            f"{item['encode_latency_seconds']['median']:.2f} | "
            f"{item['decode_latency_seconds']['median']:.2f} | "
            f"{item['bits_per_token_encode']['mean']:.3f} | "
            f"{item['packet_len_bytes']['mean']:.1f} |"
        )

    lines.extend(
        [
            "",
            "## Runtime and Provenance",
            "",
            "| Field | Value |",
            "|---|---|",
            f"| Ghostext git head | `{runtime['ghostext_git_head']}` |",
            f"| Paper git head | `{runtime['paper_git_head']}` |",
            f"| Host | `{runtime['host']}` |",
            f"| Platform | `{runtime['platform']}` |",
            f"| Python | `{runtime['python']}` |",
            f"| CPU model | `{runtime['cpu_model']}` |",
            f"| RAM (GiB) | `{runtime['ram_total_gib']}` |",
            f"| llama-cpp-python version | `{runtime['llama_cpp_python_version']}` |",
            f"| llama.cpp build commit | `{runtime['llama_cpp_commit_sha']}` |",
            f"| ggml build commit | `{runtime['ggml_build_commit']}` |",
            f"| Model id | `{backend_metadata.get('model_id', 'unknown')}` |",
            f"| Canonical GGUF name | `{model['gguf_metadata']['general.name']}` |",
            f"| Backend id | `{backend_metadata.get('backend_id', 'unknown')}` |",
            f"| Tokenizer hash | `{backend_metadata.get('tokenizer_hash', 'unknown')}` |",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the 10x10 Ghostext prompt-message grid")
    parser.add_argument(
        "--out-dir",
        default=str(GHOSTEXT_ROOT / "results" / "prompt-message-grid"),
    )
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--passphrase", default="grid-eval-pass")
    parser.add_argument("--ctx-size", type=int, default=4096)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--top-p", type=float, default=0.995)
    parser.add_argument("--max-candidates", type=int, default=64)
    parser.add_argument("--min-entropy-bits", type=float, default=0.0)
    parser.add_argument("--total-frequency", type=int, default=4096)
    parser.add_argument("--guard-bits", type=int, default=128)
    parser.add_argument("--header-token-budget", type=int, default=2048)
    parser.add_argument("--body-token-budget", type=int, default=4096)
    parser.add_argument("--natural-tail-max-tokens", type=int, default=64)
    parser.add_argument("--stall-patience-tokens", type=int, default=256)
    parser.add_argument("--low-entropy-window-tokens", type=int, default=32)
    parser.add_argument("--low-entropy-threshold-bits", type=float, default=0.1)
    parser.add_argument("--max-encode-attempts", type=int, default=10)
    parser.add_argument(
        "--max-trials",
        type=int,
        default=None,
        help="optional limit for smoke tests; when omitted, run the full 10x10 grid",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    runs_path = out_dir / "prompt_message_grid_runs.jsonl"
    summary_path = out_dir / "prompt_message_grid_summary.json"
    markdown_path = out_dir / "prompt_message_grid_summary.md"
    dataset_path = out_dir / "prompt_message_grid_dataset.json"

    prompts = build_prompts()
    messages = build_messages()
    dataset_manifest = {
        "name": "prompt-message-grid-v1",
        "prompt_count": len(prompts),
        "message_count": len(messages),
        "expected_trials": len(prompts) * len(messages),
        "prompts": [asdict(item) for item in prompts],
        "messages": [asdict(item) for item in messages],
    }
    dataset_path.write_text(json.dumps(dataset_manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    model = resolve_default_model_path(args.model_path)
    backend = QwenLlamaCppBackend(
        LlamaCppBackendConfig(
            model_path=str(model.path),
            model_id=args.model_id or DEFAULT_MODEL_ID,
            n_ctx=args.ctx_size,
            n_batch=args.batch_size,
            n_threads=args.threads,
            seed=args.seed,
        )
    )
    config = RuntimeConfig(
        seed=args.seed,
        candidate_policy=CandidatePolicyConfig(
            top_p=args.top_p,
            max_candidates=args.max_candidates,
            min_entropy_bits=args.min_entropy_bits,
            enforce_retokenization_stability=True,
        ),
        codec=CodecConfig(
            total_frequency=args.total_frequency,
            guard_bits=args.guard_bits,
            max_header_tokens=args.header_token_budget,
            max_body_tokens=args.body_token_budget,
            natural_tail_max_tokens=args.natural_tail_max_tokens,
            stall_patience_tokens=args.stall_patience_tokens,
            low_entropy_window_tokens=args.low_entropy_window_tokens,
            low_entropy_threshold_bits=args.low_entropy_threshold_bits,
            max_encode_attempts=args.max_encode_attempts,
        ),
    )
    encoder = StegoEncoder(backend, config)
    decoder = StegoDecoder(backend, config)

    existing_runs = load_existing_runs(runs_path)
    run_manifest_path = out_dir / "run_manifest.json"
    run_manifest = {
        "runtime_config": json.loads(config.to_json()),
        "backend_metadata": backend.metadata.as_dict(),
        "source_sha256": {
            p.name: sha256_hex(p.read_bytes())
            for p in sorted((GHOSTEXT_ROOT / "src/ghostext").glob("*.py"))
        },
        "threads": args.threads, "ctx_size": args.ctx_size, "batch_size": args.batch_size,
    }
    if run_manifest_path.exists():
        if json.loads(run_manifest_path.read_text()) != run_manifest:
            raise ValueError("Source or configuration changed; use a new output directory")
    elif existing_runs:
        raise ValueError("Historical results have no resume manifest; use a new output directory")
    else:
        run_manifest_path.write_text(json.dumps(run_manifest, indent=2) + "\n")
    completed_pair_ids = {item["pair_id"] for item in existing_runs}
    all_runs = list(existing_runs)
    trial_pairs = [(prompt, message) for prompt in prompts for message in messages]
    if args.max_trials is not None:
        trial_pairs = trial_pairs[: args.max_trials]

    with runs_path.open("a", encoding="utf-8") as handle:
        for pair_index, (prompt_case, message_case) in enumerate(trial_pairs):
            pair_id = f"{prompt_case.prompt_id}__{message_case.message_id}"
            if pair_id in completed_pair_ids:
                print(f"[skip] {pair_id}", flush=True)
                continue

            item: dict[str, Any] = {
                "pair_index": pair_index,
                "pair_id": pair_id,
                "prompt": asdict(prompt_case),
                "message": {
                    **asdict(message_case),
                    "message_bytes": len(message_case.message.encode("utf-8")),
                },
                "status": "unknown",
            }
            t0 = time.perf_counter()
            try:
                enc = encoder.encode(
                    message_case.message,
                    passphrase=args.passphrase,
                    prompt=prompt_case.prompt,
                )
                t1 = time.perf_counter()
                dec = decoder.decode(
                    enc.text,
                    passphrase=args.passphrase,
                    prompt=prompt_case.prompt,
                )
                t2 = time.perf_counter()
                item.update(
                    {
                        "status": "success",
                        "decode_match": dec.plaintext == message_case.message,
                        "failure_class": None,
                        "plaintext_bytes": len(message_case.message.encode("utf-8")),
                        "packet_len": len(enc.packet),
                        "encode_elapsed_seconds": enc.elapsed_seconds,
                        "decode_elapsed_seconds": dec.elapsed_seconds,
                        "e2e_elapsed_seconds": t2 - t0,
                        "encode_wall_seconds": t1 - t0,
                        "decode_wall_seconds": t2 - t1,
                        "attempts_used": enc.attempts_used,
                        "total_tokens": enc.total_tokens,
                        "packet_tokens": enc.packet_tokens,
                        "tail_tokens": enc.tail_tokens,
                        "decode_consumed_tokens": dec.consumed_tokens,
                        "decode_trailing_tokens": dec.trailing_tokens,
                        "bits_per_token_encode": enc.bits_per_token,
                        "net_plaintext_bits_per_token": (
                            8 * len(message_case.message.encode("utf-8")) / enc.total_tokens
                        ),
                        "bits_per_token_decode": dec.bits_per_token,
                        "encode_tokens_per_second": enc.tokens_per_second,
                        "decode_tokens_per_second": dec.tokens_per_second,
                        "cover_text": enc.text,
                        "cover_text_sha256": sha256_hex(enc.text),
                        "packet_sha256": sha256_hex(enc.packet),
                        "config_fingerprint_hex": f"{enc.config_fingerprint:016x}",
                    }
                )
                if not item["decode_match"]:
                    item["status"] = "failure"
                    item["failure_class"] = "plaintext_mismatch"
                print(
                    (
                        f"[{pair_index + 1:03d}/{len(trial_pairs):03d}] {pair_id} "
                        f"{item['status']} encode={item.get('encode_wall_seconds', 0.0):.2f}s "
                        f"decode={item.get('decode_wall_seconds', 0.0):.2f}s "
                        f"bpt={item.get('bits_per_token_encode', 0.0):.3f} "
                        f"attempts={item.get('attempts_used', 0)}"
                    ),
                    flush=True,
                )
            except Exception as exc:  # noqa: BLE001
                item.update(
                    {
                        "status": "failure",
                        "decode_match": False,
                        "failure_class": classify_failure(exc),
                        "error": f"{type(exc).__name__}: {exc}",
                        "e2e_elapsed_seconds": time.perf_counter() - t0,
                    }
                )
                print(
                    f"[{pair_index + 1:03d}/{len(trial_pairs):03d}] {pair_id} failure "
                    f"{item['failure_class']}: {item['error']}",
                    flush=True,
                )

            handle.write(json.dumps(item, ensure_ascii=False) + "\n")
            handle.flush()
            all_runs.append(item)
            completed_pair_ids.add(pair_id)

    metrics = summarize_subset(all_runs)
    by_prompt_language: dict[str, list[dict[str, Any]]] = {}
    by_message_length: dict[str, list[dict[str, Any]]] = {}
    for row in all_runs:
        by_prompt_language.setdefault(row["prompt"]["language"], []).append(row)
        by_message_length.setdefault(row["message"]["length_class"], []).append(row)

    summary = {
        "generated_at_utc": utc_timestamp(),
        "runtime": build_runtime_info(),
        "model": {
            **extract_model_provenance(
                model_path=model.path,
                resolved_model_source=model.source,
                backend_metadata=backend.metadata.as_dict(),
                llama_metadata=getattr(getattr(backend, "_llm", None), "metadata", {}),
            ),
            "ctx_size": args.ctx_size,
            "batch_size": args.batch_size,
            "threads": args.threads,
        },
        "config": {
            "seed": args.seed,
            "passphrase_policy": GRID_PASSPHRASE_POLICY,
            "cover_text_policy": FULL_COVER_TEXT_POLICY,
            "candidate": {
                "top_p": args.top_p,
                "max_candidates": args.max_candidates,
                "min_entropy_bits": args.min_entropy_bits,
                "retokenization_stability": True,
            },
            "codec": {
                "total_frequency": args.total_frequency,
                "guard_bits": args.guard_bits,
                "header_token_budget": args.header_token_budget,
                "body_token_budget": args.body_token_budget,
                "natural_tail_max_tokens": args.natural_tail_max_tokens,
                "stall_patience_tokens": args.stall_patience_tokens,
                "low_entropy_window_tokens": args.low_entropy_window_tokens,
                "low_entropy_threshold_bits": args.low_entropy_threshold_bits,
                "max_encode_attempts": args.max_encode_attempts,
            },
        },
        "dataset": {
            **dataset_manifest,
            "completed_trials": len(all_runs),
            "artifacts": {
                "dataset_json": str(dataset_path),
                "runs_jsonl": str(runs_path),
                "summary_json": str(summary_path),
                "summary_md": str(markdown_path),
            },
        },
        "metrics": metrics,
        "breakdowns": {
            "by_prompt_language": {
                key: summarize_subset(value) for key, value in sorted(by_prompt_language.items())
            },
            "by_message_length": {
                key: summarize_subset(value) for key, value in sorted(by_message_length.items())
            },
        },
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    markdown_path.write_text(render_markdown(summary) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "dataset_file": str(dataset_path),
                "runs_file": str(runs_path),
                "summary_file": str(summary_path),
                "summary_md": str(markdown_path),
                "completed_trials": len(all_runs),
                "success_rate": metrics["decode_success_rate"],
                "failure_histogram": metrics["failure_histogram"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
