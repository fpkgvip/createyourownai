"""Offline, real CPU training through the full keep/revert cycle (opt in)."""

import csv
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import sys

import pytest

from cyoai.agent.llm import make_llm
from cyoai.agent.loop import ResearchLoop
from cyoai.agent.runners import LocalRunner
from cyoai.agent.workspace import create_workspace, git
from cyoai.config import load_config


@pytest.mark.skipif(os.environ.get("CYOAI_RUN_CPU_SMOKE") != "1", reason="opt in to real CPU training")
def test_scripted_cpu_keep_revert(tmp_path, monkeypatch):
    import pyarrow as pa
    import pyarrow.parquet as pq
    import responses
    import torch

    core = Path(__file__).resolve().parents[2] / "cyoai/core"
    cache = tmp_path / "cache"
    monkeypatch.setenv("CYOAI_SMOKE", "1")
    monkeypatch.setenv("CYOAI_DEVICE", "cpu")
    monkeypatch.setenv("CYOAI_CACHE_DIR", str(cache))
    monkeypatch.setenv("CYOAI_LLM_PROVIDER", "fake")
    monkeypatch.setenv("CYOAI_CONFIG_DIR", str(tmp_path / "config"))
    monkeypatch.syspath_prepend(str(core))
    spec = importlib.util.spec_from_file_location("loop_cpu_prepare", core / "prepare.py")
    prepare = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(prepare)

    corpus = pa.Table.from_pydict({"text": [
        f"Once upon a time, a little robot counted {i} stars. The sky was blue and the sun was warm. " * 4
        for i in range(2000)
    ]})
    buffer = pa.BufferOutputStream()
    pq.write_table(corpus, buffer)
    (cache / "data").mkdir(parents=True)
    with responses.RequestsMock() as http:
        for index in (0, prepare.VAL_SHARD):
            http.get(f"{prepare.BASE_URL}/shard_{index:05d}.parquet", body=buffer.getvalue().to_pybytes())
            assert prepare.download_single_shard(index)
        assert len(http.calls) == 2
    prepare.train_tokenizer()

    workspace = create_workspace("Improve a real tiny CPU model offline", tmp_path / "runs")
    trained = (workspace / "train.py").read_text().replace(
        "if step > 10 and total_training_time >= TIME_BUDGET:", "if step >= 11:")
    trained = trained.replace("device = get_device()", "TIME_BUDGET = 1000\ndevice = get_device()")
    untrained = trained.replace("    optimizer.step()", "    pass  # no parameter updates")
    (workspace / "train.py").write_text(untrained)
    git(workspace, "add", "train.py")
    git(workspace, "commit", "-m", "Deterministic smoke baseline")
    baseline = git(workspace, "rev-parse", "HEAD")

    def reply(code, description):
        return f"DESCRIPTION: {description}\n```python\n{code.rstrip()}\n```"

    replies = [
        reply(trained, "improve with optimizer updates"),
        reply(untrained, "regress without updates"),
        reply("raise RuntimeError('scripted crash')", "runtime crash"),
        reply(trained + "\nfrom pathlib import Path\nPath('new.py').write_text('unexpected')\n", "other file"),
        "not a parseable proposal",
        reply("def broken(:\n", "syntax error"),
        reply(untrained, "final regression"),
    ]
    script = tmp_path / "replies.json"
    script.write_text(json.dumps(replies))
    monkeypatch.setenv("CYOAI_FAKE_LLM_RESPONSES", str(script))

    snapshots = []
    checkpoints = []

    def digest_best():
        best = workspace / "best"
        return {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in best.iterdir()}

    class Runner(LocalRunner):
        def run(self):
            snapshots.append((git(workspace, "rev-parse", "HEAD"), (workspace / "train.py").read_text()))
            return super().run()

        def keep(self):
            super().keep()
            checkpoints.append(digest_best())

    logs = []
    runner = Runner(workspace, budget_secs=1, cmd=f"{sys.executable} train.py", log=logs.append)
    best = ResearchLoop(workspace, make_llm(load_config()), runner, len(replies)).run()
    with (workspace / "results.tsv").open() as stream:
        rows = list(csv.DictReader(stream, delimiter="\t"))
    assert [row["status"] for row in rows] == [
        "keep", "keep", "discard", "crash", "crash", "crash", "crash", "discard",
    ], "\n".join(logs)
    assert float(rows[1]["val_bpb"]) < float(rows[0]["val_bpb"])
    assert rows[0]["val_bpb"] == rows[2]["val_bpb"] == rows[7]["val_bpb"]
    assert all(math.isinf(float(row["val_bpb"])) for row in rows[3:7])
    assert all(float(row["memory_gb"]) == 0 for row in rows)
    assert best.commit == rows[1]["commit"] == git(workspace, "rev-parse", "HEAD")
    assert best.val_bpb == float(rows[1]["val_bpb"])
    assert (workspace / "train.py").read_text() == trained
    assert git(workspace, "status", "--porcelain") == ""
    assert not (workspace / "new.py").exists()
    assert git(workspace, "diff", "--name-only", baseline, best.commit) == "train.py"
    for row in rows[1:]:
        if row["commit"] != best.commit:
            assert git(workspace, "diff-tree", "--no-commit-id", "--name-only", "-r", row["commit"]) == "train.py"
    assert len(checkpoints) == 2
    assert checkpoints[0] != checkpoints[1] == digest_best()
    metrics = json.loads((workspace / "best/metrics.json").read_text())
    state = torch.load(workspace / "best/model.pt", weights_only=True)
    assert metrics["val_bpb"] == state["val_bpb"] == best.val_bpb
    assert metrics["num_steps"] == 11
    assert (workspace / "best/train.py").read_text() == trained
    assert snapshots[0] == (baseline, untrained)
    assert snapshots[1][1] == trained
    print("CPU loop results:", [(row["status"], row["val_bpb"]) for row in rows])
    print("Best checkpoint verified:", best.commit, best.val_bpb)
