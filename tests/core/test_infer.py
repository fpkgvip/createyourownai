"""Offline smoke coverage; run with python -m pytest tests/core."""

import importlib.util
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

for dependency in ("torch", "rustbpe", "pyarrow", "tiktoken"):
    if importlib.util.find_spec(dependency) is None:
        pytest.skip(f"Offline inference test requires importable {dependency}", allow_module_level=True)

import pyarrow as pa
import pyarrow.parquet as pq
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from cyoai.core.infer import _snapshot_module, generate, load_model, repl

CORE = ROOT / "cyoai" / "core"


@pytest.fixture(scope="module")
def smoke_checkpoint(tmp_path_factory):
    root = tmp_path_factory.mktemp("offline-infer")
    cache = root / "cache"
    data = cache / "data"
    data.mkdir(parents=True)
    texts = [
        f"Once upon a time the curious researcher {i} explored language and learning. "
        "The meaning of life is to discover and share ideas with friends. " * 8
        for i in range(2500)
    ]
    table = pa.table({"text": texts})
    for shard in ("shard_00000.parquet", "shard_06542.parquet"):
        pq.write_table(table, data / shard)
    out = root / "out"
    env = dict(os.environ, CYOAI_CACHE_DIR=str(cache), CYOAI_OUT_DIR=str(out),
               CYOAI_SMOKE="1", CYOAI_TIME_BUDGET="3", CYOAI_DEVICE="cpu")
    for script, args in [("prepare.py", ["--num-shards", "1"]), ("train.py", [])]:
        result = subprocess.run([sys.executable, str(CORE / script), *args], cwd=CORE,
                                env=env, capture_output=True, text=True, timeout=45)
        assert result.returncode == 0, result.stdout + result.stderr
    assert {"model.pt", "metrics.json", "tokenizer.pkl", "train.py"} <= {p.name for p in out.iterdir()}
    return out


def test_offline_inference_and_cli(smoke_checkpoint):
    torch.set_num_threads(2)
    model, tokenizer, info = load_model(smoke_checkpoint, "cpu")
    assert info["snapshot_error"] is None
    chunks = []
    completion = generate(model, tokenizer, "Once " * 300, max_new_tokens=5,
                          temperature=0, stream=chunks.append)
    assert completion
    assert "".join(chunks) == completion
    assert generate(model, tokenizer, "", max_new_tokens=0) == ""
    for kwargs in ({"temperature": -1}, {"top_k": -1}, {"max_new_tokens": -1}):
        with pytest.raises(ValueError):
            generate(model, tokenizer, "Once", **kwargs)
    result = subprocess.run([sys.executable, str(CORE / "infer.py"), str(smoke_checkpoint),
                             "--prompt", "Once", "--max-new-tokens", "5", "--seed", "42"],
                            capture_output=True, text=True, timeout=15)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip()


def test_agent_modified_architecture_and_fallback(smoke_checkpoint, tmp_path):
    out = tmp_path / "modified"
    shutil.copytree(smoke_checkpoint, out)
    source = (out / "train.py").read_text().replace("lm_head", "agent_head")
    (out / "train.py").write_text(source + '\nraise RuntimeError("training must not execute")\n')
    checkpoint = torch.load(out / "model.pt", weights_only=True)
    checkpoint["state_dict"] = {k.replace("lm_head", "agent_head"): v for k, v in checkpoint["state_dict"].items()}
    torch.save(checkpoint, out / "model.pt")
    model, tokenizer, info = load_model(out / "model.pt", "cpu")
    assert hasattr(model, "agent_head")
    assert info["snapshot_error"] is None
    assert generate(model, tokenizer, "Once", max_new_tokens=2, top_k=1)
    shutil.copy2(smoke_checkpoint / "model.pt", out / "model.pt")
    (out / "train.py").write_text("not valid python !!!")
    with pytest.warns(RuntimeWarning, match="Snapshot failed"):
        _, _, info = load_model(out, "cpu")
    assert info["source"] == str(CORE / "train.py")
    checkpoint = torch.load(out / "model.pt", weights_only=True)
    checkpoint["state_dict"]["unexpected"] = torch.zeros(1)
    torch.save(checkpoint, out / "model.pt")
    with pytest.warns(RuntimeWarning), pytest.raises(RuntimeError, match="Could not load architecture"):
        load_model(out, "cpu")


def test_ast_helpers_and_repl_exit(tmp_path, monkeypatch):
    snapshot = tmp_path / "train.py"
    snapshot.write_text('import math\nOFFSET = 3\ndef helper(x):\n    return math.sqrt(x) + OFFSET\nraise RuntimeError("no")\n')
    module = _snapshot_module(snapshot, torch.device("cpu"))
    assert module.helper(4) == 5
    for value in ("", "exit"):
        monkeypatch.setattr("builtins.input", lambda _: value)
        repl(None, None)
    def eof(_):
        raise EOFError
    monkeypatch.setattr("builtins.input", eof)
    repl(None, None)


def test_snapshot_computed_constant_in_forward(tmp_path):
    snapshot = tmp_path / "train.py"
    snapshot.write_text('''import math
import torch
BASE = 2
SCALE = 1 / math.sqrt(BASE)
SHIFT: float = SCALE * 2
class GPT(torch.nn.Module):
    def forward(self, x):
        return x * SCALE + SHIFT
model = GPT()
loss = model(torch.ones(1))
raise RuntimeError("training must not execute")
''')
    module = _snapshot_module(snapshot, torch.device("cpu"))
    torch.testing.assert_close(module.GPT()(torch.ones(1)), torch.tensor([3 / 2**0.5]))
    assert not hasattr(module, "model")
    assert not hasattr(module, "loss")
