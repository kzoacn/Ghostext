import json
import os
from pathlib import Path
import subprocess
import sys
import unittest


REPO_ROOT = Path(__file__).resolve().parents[1]


class CliTests(unittest.TestCase):
    def _run(self, *args: str) -> dict[str, object]:
        env = os.environ.copy()
        env["PYTHONPATH"] = str(REPO_ROOT / "src")
        completed = subprocess.run(
            [sys.executable, "-m", "hidetext.cli", *args],
            cwd=REPO_ROOT,
            env=env,
            check=True,
            capture_output=True,
            text=True,
        )
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


if __name__ == "__main__":
    unittest.main()
