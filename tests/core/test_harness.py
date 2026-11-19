import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from dataclasses import dataclass
from unittest.mock import patch

import torch

CORE = Path(__file__).resolve().parents[2] / "cyoai" / "core"
sys.path.insert(0, str(CORE))
from attention import flash_attn_func
from checkpoint import save_checkpoint
from device import get_device


@dataclass
class Config:
    n_layer: int = 2


class HarnessTests(unittest.TestCase):
    def test_attention_masks_and_grouped_heads(self):
        q = torch.zeros(1, 4, 2, 2)
        k = torch.zeros(1, 4, 1, 2)
        v = torch.arange(4.0).view(1, 4, 1, 1).expand(1, 4, 1, 2)
        for window, expected in [((-1, -1), [0, .5, 1, 1.5]), ((1, 0), [0, .5, 1.5, 2.5])]:
            out = flash_attn_func(q, k, v, causal=True, window_size=window)
            torch.testing.assert_close(out[0, :, 0, 0], torch.tensor(expected))
        out = flash_attn_func(q, k, v)
        torch.testing.assert_close(out, torch.full_like(out, 1.5))

    def test_attention_empty_sequence(self):
        x = torch.empty(1, 0, 1, 2)
        self.assertEqual(flash_attn_func(x, x, x).shape, x.shape)

    def test_device_override_and_invalid_value(self):
        with patch.dict(os.environ, CYOAI_DEVICE="cpu"):
            self.assertEqual(get_device(), torch.device("cpu"))
        for value in ["", "invalid"]:
            with patch.dict(os.environ, CYOAI_DEVICE=value):
                with self.assertRaises(ValueError):
                    get_device()

    def test_checkpoint_artifacts_and_unwrap(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "tokenizer.pkl").write_bytes(b"tokenizer")
            (root / "source.py").write_text("architecture")
            model = torch.nn.Linear(2, 2)
            model.config = Config()
            wrapper = type("Compiled", (), {"_orig_mod": model})()
            metrics = {"val_bpb": 1.25, "num_steps": 3}
            with patch.dict(os.environ, CYOAI_OUT_DIR=str(root / "out")):
                save_checkpoint(wrapper, metrics, root, root / "source.py")
            out = root / "out"
            state = torch.load(out / "model.pt", weights_only=True)
            self.assertEqual(state["config"], {"n_layer": 2})
            self.assertEqual(state["val_bpb"], 1.25)
            self.assertEqual(set(state["state_dict"]), {"weight", "bias"})
            self.assertEqual(json.loads((out / "metrics.json").read_text()), metrics)
            self.assertEqual((out / "tokenizer.pkl").read_bytes(), b"tokenizer")
            self.assertEqual((out / "train.py").read_text(), "architecture")

    def test_checkpoint_missing_tokenizer_errors(self):
        with tempfile.TemporaryDirectory() as tmp:
            model = torch.nn.Linear(1, 1)
            model.config = Config()
            with patch.dict(os.environ, CYOAI_OUT_DIR=str(Path(tmp) / "out")):
                with self.assertRaises(FileNotFoundError):
                    save_checkpoint(model, {}, tmp, __file__)


if __name__ == "__main__":
    unittest.main()
