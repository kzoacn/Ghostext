import contextlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from hidetext import cli
from hidetext.model_assets import (
    DEFAULT_BACKEND,
    DEFAULT_BATCH_SIZE,
    DEFAULT_CTX_SIZE,
    DEFAULT_MAX_CANDIDATES,
    DEFAULT_MODEL_ID,
    DEFAULT_SEED,
    DEFAULT_TOP_P,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


class CliTests(unittest.TestCase):
    def _run_completed(
        self,
        *args: str,
        input_text: str | None = None,
    ) -> subprocess.CompletedProcess[str]:
        env = os.environ.copy()
        env["PYTHONPATH"] = str(REPO_ROOT / "src")
        return subprocess.run(
            [sys.executable, "-m", "hidetext.cli", *args],
            cwd=REPO_ROOT,
            env=env,
            check=True,
            capture_output=True,
            text=True,
            input=input_text,
        )

    def _run(self, *args: str) -> dict[str, object]:
        completed = self._run_completed(*args)
        return json.loads(completed.stdout)

    def test_encode_then_decode(self) -> None:
        prompt = "Write a calm and readable English paragraph."
        passphrase = "cli-pass"
        message = "CLI roundtrip works."

        encoded = self._run_completed(
            "encode",
            "--backend",
            "toy",
            "--prompt",
            prompt,
            "--passphrase",
            passphrase,
            "--message",
            message,
        )
        self.assertTrue(encoded.stdout)
        self.assertFalse(encoded.stdout.startswith("{"))
        decoded = self._run_completed(
            "decode",
            "--backend",
            "toy",
            "--prompt",
            prompt,
            "--passphrase",
            passphrase,
            input_text=encoded.stdout,
        )
        self.assertEqual(decoded.stdout, message)

    def test_encode_then_decode_with_files(self) -> None:
        prompt = "请写一段温柔自然的中文短文。"
        passphrase = "cli-file-pass"
        message = "文件参数也能正确往返。"

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            prompt_path = tmp_path / "prompt.txt"
            passphrase_path = tmp_path / "passphrase.txt"
            seed_path = tmp_path / "seed.txt"

            prompt_path.write_text(prompt + "\n", encoding="utf-8")
            passphrase_path.write_text(passphrase + "\n", encoding="utf-8")
            seed_path.write_text("19\n", encoding="utf-8")

            encoded = self._run(
                "encode",
                "--backend",
                "toy",
                "--json",
                "--prompt-file",
                str(prompt_path),
                "--passphrase-file",
                str(passphrase_path),
                "--message",
                message,
                "--seed-file",
                str(seed_path),
            )
            decoded = self._run(
                "decode",
                "--backend",
                "toy",
                "--json",
                "--prompt-file",
                str(prompt_path),
                "--passphrase-file",
                str(passphrase_path),
                "--text",
                str(encoded["text"]),
                "--seed-file",
                str(seed_path),
            )
            self.assertEqual(decoded["plaintext"], message)
            self.assertIn("tail_tokens", encoded)
            self.assertIn("trailing_tokens", decoded)

    def test_encode_and_decode_json_flag_preserves_detailed_output(self) -> None:
        prompt = "Write a calm and readable English paragraph."
        passphrase = "cli-json-pass"
        message = "CLI json mode works."

        encoded = self._run(
            "encode",
            "--backend",
            "toy",
            "--json",
            "--prompt",
            prompt,
            "--passphrase",
            passphrase,
            "--message",
            message,
        )
        decoded = self._run(
            "decode",
            "--backend",
            "toy",
            "--json",
            "--prompt",
            prompt,
            "--passphrase",
            passphrase,
            "--text",
            str(encoded["text"]),
        )

        self.assertEqual(decoded["plaintext"], message)
        self.assertIn("packet_tokens", encoded)
        self.assertIn("tail_tokens", encoded)
        self.assertIn("trailing_tokens", decoded)

    def test_progress_default_on_for_encode(self) -> None:
        completed = self._run_completed(
            "encode",
            "--backend",
            "toy",
            "--prompt",
            "请写一段温柔自然的中文短文。",
            "--passphrase",
            "progress-pass",
            "--message",
            "进度日志可见",
            "--progress-token-interval",
            "400",
        )
        self.assertTrue(completed.stdout)
        self.assertIn("encode/header", completed.stderr)
        self.assertIn("|", completed.stderr)
        self.assertIn("tok/s", completed.stderr)
        self.assertIn("bpt ", completed.stderr)

    def test_quiet_disables_progress(self) -> None:
        completed = self._run_completed(
            "encode",
            "--backend",
            "toy",
            "--prompt",
            "Write a calm and readable English paragraph.",
            "--passphrase",
            "quiet-pass",
            "--message",
            "quiet mode works",
            "--quiet",
            "--progress-token-interval",
            "1",
        )
        self.assertTrue(completed.stdout)
        self.assertEqual(completed.stderr, "")

    def test_stall_patience_argument_is_accepted(self) -> None:
        payload = self._run(
            "encode",
            "--backend",
            "toy",
            "--json",
            "--prompt",
            "Write a calm and readable English paragraph.",
            "--passphrase",
            "cli-pass",
            "--message",
            "CLI roundtrip works.",
            "--natural-tail-max-tokens",
            "8",
            "--stall-patience-tokens",
            "64",
            "--low-entropy-window-tokens",
            "16",
            "--low-entropy-threshold-bits",
            "0.1",
            "--max-encode-attempts",
            "3",
        )
        self.assertIn("text", payload)
        self.assertIn("attempts_used", payload)
        self.assertIn("tail_tokens", payload)

    def test_parser_defaults_match_real_model_profile(self) -> None:
        parser = cli._build_parser()
        args = parser.parse_args(
            [
                "encode",
                "--prompt",
                "Write a calm and readable English paragraph.",
                "--passphrase",
                "cli-pass",
                "--message",
                "CLI roundtrip works.",
            ]
        )

        self.assertEqual(args.backend, DEFAULT_BACKEND)
        self.assertEqual(args.ctx_size, DEFAULT_CTX_SIZE)
        self.assertEqual(args.batch_size, DEFAULT_BATCH_SIZE)
        self.assertEqual(args.top_p, DEFAULT_TOP_P)
        self.assertEqual(args.max_candidates, DEFAULT_MAX_CANDIDATES)
        self.assertFalse(args.quiet)
        self.assertEqual(cli._resolve_seed(args), DEFAULT_SEED)

    def test_default_model_id_is_used_for_cached_default_model(self) -> None:
        parser = cli._build_parser()
        args = parser.parse_args(
            [
                "encode",
                "--prompt",
                "Write a calm and readable English paragraph.",
                "--passphrase",
                "cli-pass",
                "--message",
                "CLI roundtrip works.",
            ]
        )

        self.assertEqual(
            cli._resolve_model_id(args, resolved_model_source="cache"),
            DEFAULT_MODEL_ID,
        )

    def test_custom_model_path_defaults_to_inferred_model_id(self) -> None:
        parser = cli._build_parser()
        args = parser.parse_args(
            [
                "encode",
                "--model-path",
                "/tmp/demo.gguf",
                "--prompt",
                "Write a calm and readable English paragraph.",
                "--passphrase",
                "cli-pass",
                "--message",
                "CLI roundtrip works.",
            ]
        )

        self.assertIsNone(
            cli._resolve_model_id(args, resolved_model_source="explicit"),
        )

    def test_eval_command_removed(self) -> None:
        parser = cli._build_parser()
        with self.assertRaises(SystemExit):
            with contextlib.redirect_stderr(io.StringIO()):
                parser.parse_args(
                    [
                        "eval",
                        "--prompt",
                        "Write a calm and readable English paragraph.",
                        "--passphrase",
                        "cli-pass",
                        "--message",
                        "CLI roundtrip works.",
                    ]
                )


if __name__ == "__main__":
    unittest.main()
