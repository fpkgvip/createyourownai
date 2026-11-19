import csv
import hashlib
import json
import shutil
import subprocess
import sys

import pytest

from cyoai.agent.llm import FakeLLM
from cyoai.agent.loop import ResearchLoop
from cyoai.agent.runners import LocalRunner, RemoteRunner
from cyoai.agent.workspace import create_workspace, git


def reply(code, description="experiment"):
    return f"DESCRIPTION: {description}\n```python\n{code}\n```"


@pytest.fixture
def workspace(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "train.py").write_text("print('val_bpb: 2')\n")
    (source / "prepare.py").write_text("# protected\n")
    (source / "program.md").write_text("Minimize validation bits per byte")
    return create_workspace("test integrity", tmp_path / "runs", source)


@pytest.mark.parametrize("effect", [
    "Path('new.py').write_text('unexpected')",
    "Path('program.md').write_text('unexpected')",
    "Path('train.py').write_text('unexpected')",
    "Path('new.py').write_text('unexpected'); subprocess.run(['git', 'add', 'new.py'], check=True)",
])
@pytest.mark.parametrize("exit_code", [0, 1])
def test_training_side_effects_rejected_and_restored(workspace, effect, exit_code):
    original = git(workspace, "rev-parse", "HEAD")
    (workspace / "notes.txt").write_text("existing user notes")
    code = ("from pathlib import Path\nimport subprocess\n" + effect +
            f"\nprint('val_bpb: 1')\nraise SystemExit({exit_code})\n")
    class Runner(LocalRunner):
        def keep(self):
            pass

    runner = Runner(workspace, cmd=f"{sys.executable} train.py", log=lambda _: None)
    best = ResearchLoop(workspace, FakeLLM([reply(code)]), runner, 1).run()
    assert best.commit == original
    assert best.val_bpb == 2
    assert git(workspace, "rev-parse", "HEAD") == original
    assert not (workspace / "new.py").exists()
    assert (workspace / "program.md").read_text().startswith("Minimize")
    assert (workspace / "train.py").read_text() == "print('val_bpb: 2')\n"
    assert (workspace / "notes.txt").read_text() == "existing user notes"
    assert git(workspace, "status", "--porcelain") == "?? notes.txt"
    with (workspace / "results.tsv").open() as stream:
        rows = list(csv.DictReader(stream, delimiter="\t"))
    assert [row["status"] for row in rows] == ["keep", "crash"]


def test_local_runner_does_not_promote_stale_checkpoint(workspace):
    out = workspace / "out"
    out.mkdir()
    for name in ("model.pt", "tokenizer.pkl", "train.py", "metrics.json"):
        (out / name).write_text("stale")
    runner = LocalRunner(workspace, cmd=f"{sys.executable} train.py", log=lambda _: None)
    assert ResearchLoop(workspace, FakeLLM(), runner, 0).run() is None
    assert not (workspace / "best").exists()


def test_remote_runner_does_not_promote_stale_checkpoint(workspace, tmp_path):
    remote = tmp_path / "remote"
    remote.mkdir()
    (remote / "out").mkdir()
    for name in ("model.pt", "tokenizer.pkl", "train.py", "metrics.json"):
        (remote / "out" / name).write_text("stale")

    class Provider:
        def sync_up(self, instance, source, destination):
            shutil.copy2(source, destination)

        def sync_down(self, instance, source, destination):
            shutil.copy2(source, destination)

        def exec_stream(self, instance, command):
            command = command.replace("uv run train.py", f"{sys.executable} train.py")
            result = subprocess.run(command, shell=True, capture_output=True, text=True, check=True)
            yield result.stdout

    runner = RemoteRunner(Provider(), "offline", str(remote), workspace, log=lambda _: None)
    assert ResearchLoop(workspace, FakeLLM(), runner, 0).run() is None
    assert not (workspace / "best").exists()


def test_scripted_loop_preserves_best_checkpoint(workspace):
    def code(value):
        return ("from pathlib import Path\nimport json\nout = Path('out')\nout.mkdir(exist_ok=True)\n"
                f"value = {value}\n"
                "(out / 'model.pt').write_text(str(value))\n"
                "(out / 'tokenizer.pkl').write_text('tokenizer')\n"
                "(out / 'train.py').write_text(Path('train.py').read_text())\n"
                "(out / 'metrics.json').write_text(json.dumps({'val_bpb': value}))\n"
                "print(f'val_bpb: {value}')\n")

    baseline = code(2)
    improved = code(1)
    (workspace / "train.py").write_text(baseline)
    git(workspace, "add", "train.py")
    git(workspace, "commit", "-m", "Test checkpoint baseline")
    replies = [reply(improved), reply(code(3)), reply("raise RuntimeError('crash')"),
               reply(code(.5) + "Path('new.py').write_text('unexpected')\n"),
               "malformed", reply("def broken(:"), reply(code(1))]
    states = []
    kept = []

    def digest():
        return {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                for p in (workspace / "best").iterdir()}

    class Runner(LocalRunner):
        def keep(self):
            super().keep()
            kept.append(digest())

    def log(message):
        if message.startswith("Iteration") and " crashed:" not in message:
            states.append((git(workspace, "rev-parse", "HEAD"),
                           git(workspace, "status", "--porcelain"), digest()))

    runner = Runner(workspace, cmd=f"{sys.executable} train.py", log=lambda _: None)
    best = ResearchLoop(workspace, FakeLLM(replies), runner, len(replies), log=log).run()
    with (workspace / "results.tsv").open() as stream:
        rows = list(csv.DictReader(stream, delimiter="\t"))
    assert [row["status"] for row in rows] == [
        "keep", "keep", "discard", "crash", "crash", "crash", "crash", "discard"]
    assert [float(row["val_bpb"]) for row in rows] == [2, 1, 3, *([float("inf")] * 4), 1]
    assert len(kept) == 2
    assert all(commit == best.commit and status == "" and artifacts == kept[1]
               for commit, status, artifacts in states[1:])
    assert git(workspace, "show", f"{best.commit}:train.py") == improved.strip()
    for row in rows[1:]:
        changed = git(workspace, "diff-tree", "--no-commit-id", "--name-only", "-r", row["commit"])
        assert set(changed.splitlines()) <= {"train.py"}
    assert (workspace / "best/train.py").read_text().strip() == improved.strip()
    assert json.loads((workspace / "best/metrics.json").read_text())["val_bpb"] == 1
    assert (workspace / "best/model.pt").read_text() == "1"
