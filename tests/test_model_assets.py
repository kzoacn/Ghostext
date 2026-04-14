import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from ghostext.llama_cpp_backend import (
    QWEN35_PROMPT_TEMPLATE,
    QWEN3_PROMPT_TEMPLATE,
    _is_capacity_warning,
    _infer_model_id,
    build_llama_cpp_tokenizer_hash,
    resolve_qwen_prompt_template,
)
from ghostext.model_assets import (
    DEFAULT_MODEL_FILENAME,
    ResolvedModelPath,
    resolve_default_model_path,
)


class ModelAssetTests(unittest.TestCase):
    def test_capacity_warning_matcher_targets_n_ctx_seq_message(self) -> None:
        self.assertTrue(
            _is_capacity_warning(
                "llama_context: n_ctx_seq (4096) < n_ctx_train (262144) -- "
                "the full capacity of the model will not be utilized"
            )
        )
        self.assertFalse(_is_capacity_warning("llama model loaded successfully"))

    def test_explicit_model_path_wins(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            model_path = Path(tmp_dir) / "explicit.gguf"
            model_path.write_bytes(b"gguf-explicit")

            resolved = resolve_default_model_path(str(model_path))

            self.assertEqual(
                resolved,
                ResolvedModelPath(path=model_path, source="explicit"),
            )

    def test_cached_default_model_is_used_without_download(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            cache_dir = Path(tmp_dir)
            cached_model = cache_dir / DEFAULT_MODEL_FILENAME
            cached_model.write_bytes(b"cached-gguf")
            with mock.patch.dict(os.environ, {"GHOSTEXT_MODEL_DIR": str(cache_dir)}, clear=True):
                resolved = resolve_default_model_path()

            self.assertEqual(resolved.path, cached_model)
            self.assertEqual(resolved.source, "cache")

    def test_missing_default_model_triggers_download(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            cache_dir = Path(tmp_dir)
            downloaded_path = cache_dir / DEFAULT_MODEL_FILENAME
            with mock.patch.dict(os.environ, {"GHOSTEXT_MODEL_DIR": str(cache_dir)}, clear=True):
                with mock.patch(
                    "ghostext.model_assets._download_default_model",
                    return_value=downloaded_path,
                ) as download_mock:
                    resolved = resolve_default_model_path()

            self.assertEqual(resolved.path, downloaded_path)
            self.assertEqual(resolved.source, "downloaded")
            download_mock.assert_called_once_with(cache_dir)

    def test_llama_cpp_tokenizer_hash_ignores_local_path(self) -> None:
        llama_metadata = {"general.name": "Qwen3-4B", "tokenizer.ggml.model": "gpt2"}
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            first_path = tmp_path / "first.gguf"
            second_path = tmp_path / "nested" / "second.gguf"
            second_path.parent.mkdir(parents=True, exist_ok=True)
            payload = (b"gguf" * 1024) + b"demo"
            first_path.write_bytes(payload)
            second_path.write_bytes(payload)

            first_hash = build_llama_cpp_tokenizer_hash(
                first_path,
                llama_metadata,
                prompt_template_id=QWEN3_PROMPT_TEMPLATE.template_id,
            )
            second_hash = build_llama_cpp_tokenizer_hash(
                second_path,
                llama_metadata,
                prompt_template_id=QWEN3_PROMPT_TEMPLATE.template_id,
            )

            self.assertEqual(first_hash, second_hash)

    def test_infer_model_id_prefers_llama_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            model_path = Path(tmp_dir) / "demo.gguf"
            model_path.write_bytes(b"gguf-demo")

            inferred = _infer_model_id(
                model_path,
                {"general.name": "Qwen3.5-2B"},
            )

            self.assertEqual(inferred, "Qwen3.5-2B")

    def test_resolve_qwen_prompt_template_uses_qwen35_no_think_template(self) -> None:
        template = resolve_qwen_prompt_template(
            "Qwen/Qwen3.5-2B",
            {},
        )

        self.assertEqual(template.template_id, QWEN35_PROMPT_TEMPLATE.template_id)

    def test_resolve_qwen_prompt_template_defaults_to_qwen3_template(self) -> None:
        template = resolve_qwen_prompt_template(
            "Qwen/Qwen3-4B-Instruct-2507",
            {},
        )

        self.assertEqual(template.template_id, QWEN3_PROMPT_TEMPLATE.template_id)


if __name__ == "__main__":
    unittest.main()
