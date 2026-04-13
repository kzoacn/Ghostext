import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


REPO_ROOT = Path(__file__).resolve().parents[1]


class CliTests(unittest.TestCase):
    def _run_completed(self, *args: str) -> subprocess.CompletedProcess[str]:
        env = os.environ.copy()
        env["PYTHONPATH"] = str(REPO_ROOT / "src")
        return subprocess.run(
            [sys.executable, "-m", "hidetext.cli", *args],
            cwd=REPO_ROOT,
            env=env,
            check=True,
            capture_output=True,
            text=True,
        )

    def _run(self, *args: str) -> dict[str, object]:
        completed = self._run_completed(*args)
        return json.loads(completed.stdout)

    def test_encode_then_decode(self) -> None:
        prompt = "Write a calm and readable English paragraph."
        passphrase = "cli-pass"
        message = "CLI roundtrip works."

        encoded = self._run(
            "encode",
            "--prompt",
            prompt,
            "--passphrase",
            passphrase,
            "--message",
            message,
            "--seed",
            "11",
        )
        decoded = self._run(
            "decode",
            "--prompt",
            prompt,
            "--passphrase",
            passphrase,
            "--text",
            str(encoded["text"]),
            "--seed",
            "11",
        )
        self.assertEqual(decoded["plaintext"], message)

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

    def test_eval_progress_goes_to_stderr(self) -> None:
        completed = self._run_completed(
            "eval",
            "--prompt",
            "请写一段温柔自然的中文短文。",
            "--passphrase",
            "progress-pass",
            "--message",
            "进度日志可见",
            "--seed",
            "13",
            "--show-progress",
            "--progress-token-interval",
            "400",
        )
        payload = json.loads(completed.stdout)
        self.assertTrue(payload["roundtrip_ok"])
        self.assertIn("[encode][header]", completed.stderr)
        self.assertIn("bits=", completed.stderr)
        self.assertIn("tps=", completed.stderr)
        self.assertIn("bpt=", completed.stderr)

    def test_stall_patience_argument_is_accepted(self) -> None:
        payload = self._run(
            "encode",
            "--prompt",
            "Write a calm and readable English paragraph.",
            "--passphrase",
            "cli-pass",
            "--message",
            "CLI roundtrip works.",
            "--seed",
            "11",
            "--stall-patience-tokens",
            "64",
        )
        self.assertIn("text", payload)


if __name__ == "__main__":
    unittest.main()
