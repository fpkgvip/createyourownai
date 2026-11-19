import json
import os
from pathlib import Path
import sys
import threading
import time

import httpx
import pytest

from cyoai.agent.llm import AnthropicClient, FakeLLM, OpenAIClient, make_llm
from cyoai.agent.loop import ResearchLoop, parse_metrics, parse_proposal
from cyoai.agent.runners import LocalRunner, RemoteRunner
from cyoai.agent import runners
from cyoai.agent.workspace import create_workspace, git
from cyoai.config import Config


@pytest.fixture
def workspace(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "train.py").write_text("x = 0\n")
    (source / "prepare.py").write_text("# immutable\n")
    (source / "program.md").write_text("Minimize val_bpb\n")
    for name in ("attention.py", "checkpoint.py", "device.py"):
        (source / name).write_text("# helper\n")
    return create_workspace("test goal", tmp_path / "runs", source)


def proposal(x):
    return f"DESCRIPTION: candidate {x}\n```python\nx = {x}\n```"


def test_loop_keep_discard_crash(workspace):
    class Runner:
        outputs = iter([(0, "val_bpb: 2\npeak_vram_mb: 2048"),
                        (0, "val_bpb: 1"), (0, "val_bpb: 3"), (1, "crashed")])
        kept = 0

        def run(self):
            return next(self.outputs)

        def keep(self):
            self.kept += 1

    runner = Runner()
    best = ResearchLoop(workspace, FakeLLM([proposal(1), proposal(2), proposal(3)]), runner, 3).run()
    assert best.val_bpb == 1
    assert runner.kept == 2
    assert (workspace / "train.py").read_text() == "x = 1\n"
    rows = (workspace / "results.tsv").read_text().splitlines()
    assert rows[0] == "commit\tval_bpb\tmemory_gb\tstatus\tdescription"
    assert [row.split("\t")[3] for row in rows[1:]] == ["keep", "keep", "discard", "crash"]
    assert git(workspace, "rev-parse", "HEAD") == best.commit
    assert (workspace / "prepare.py").read_text() == "# immutable\n"
    assert "## User goal\n\ntest goal" in (workspace / "program.md").read_text()
    for name in ("attention.py", "checkpoint.py", "device.py"):
        assert (workspace / name).read_text() == "# helper\n"
        assert name in git(workspace, "ls-files").splitlines()


def test_loop_interrupt_restores(workspace):
    class Runner:
        count = 0

        def run(self):
            self.count += 1
            if self.count == 2:
                raise KeyboardInterrupt
            return 0, "val_bpb: 1"

    with pytest.raises(KeyboardInterrupt):
        ResearchLoop(workspace, FakeLLM([proposal(1)]), Runner(), 1).run()
    assert (workspace / "train.py").read_text() == "x = 0\n"


def test_loop_all_crashes(workspace):
    class Runner:
        def run(self):
            return 0, "no metric"
    assert ResearchLoop(workspace, FakeLLM(), Runner(), 0).run() is None
    assert "crash" in (workspace / "results.tsv").read_text()


@pytest.mark.parametrize("delete", [False, True])
def test_prepare_is_protected(workspace, delete):
    class Runner:
        def run(self):
            if delete:
                (workspace / "prepare.py").unlink()
            else:
                (workspace / "prepare.py").write_text("changed")
            return 0, "val_bpb: 1"
    assert ResearchLoop(workspace, FakeLLM(), Runner(), 0).run() is None
    assert (workspace / "prepare.py").read_text() == "# immutable\n"


@pytest.mark.parametrize("response", ["", "```python\nx=1\n```", "DESCRIPTION: no\n```python\nx=\n```",
    "DESCRIPTION: bad\n```python\nx=1\n```\n```python\nx=2\n```",
    "DESCRIPTION: bad\nFILE: prepare.py\n```python\nx=1\n```"])
def test_invalid_proposal(response):
    with pytest.raises((ValueError, SyntaxError)):
        parse_proposal(response)


def test_metrics():
    assert parse_metrics("val_bpb:          0.997900\npeak_vram_mb: 1024") == (.9979, 1)
    assert parse_metrics("val_bpb: 2\nval_bpb: 1e-1") == (.1, 0)
    with pytest.raises(ValueError):
        parse_metrics("")
    with pytest.raises(ValueError):
        parse_metrics("val_bpb: 1e999")


def test_fake_llm(tmp_path, monkeypatch):
    response_file = tmp_path / "responses.json"
    response_file.write_text(json.dumps([proposal(1), proposal(2)]))
    monkeypatch.setenv("CYOAI_FAKE_LLM_RESPONSES", str(response_file))
    fake = make_llm(Config(llm_provider="fake"))
    assert fake.complete("", "") == proposal(1)
    assert fake.complete("", "") == proposal(2)
    monkeypatch.delenv("CYOAI_FAKE_LLM_RESPONSES")
    assert "x=0" in FakeLLM().complete("", "<train.py>\nx=0\n</train.py>")
    with pytest.raises(ValueError):
        make_llm(Config())


@pytest.mark.parametrize("kind", ["anthropic", "openai"])
def test_http_clients(kind):
    def handle(request):
        payload = json.loads(request.content)
        assert payload["model"] == "test-model"
        if kind == "anthropic":
            assert request.headers["x-api-key"] == "test-key"
            assert payload["system"] == "system"
            body = {"content": [{"type": "text", "text": "answer"}]}
        else:
            assert str(request.url) == "https://example.test/v1/chat/completions"
            assert request.headers["Authorization"] == "Bearer test-key"
            body = {"choices": [{"message": {"content": "answer"}}]}
        return httpx.Response(200, json=body)
    transport = httpx.MockTransport(handle)
    client = (AnthropicClient("test-key", "test-model", transport=transport) if kind == "anthropic"
              else OpenAIClient("test-key", "test-model", "https://example.test/v1/", transport=transport))
    assert client.complete("system", "prompt") == "answer"
    client.transport = httpx.MockTransport(lambda request: httpx.Response(401, json={"error": "no"}))
    with pytest.raises(httpx.HTTPStatusError):
        client.complete("system", "prompt")


def test_local_runner_scrubs_secrets_and_keeps_flags(workspace, monkeypatch):
    secret_names = ["ANTHROPIC_API_KEY", "OPENAI_API_KEY", "RUNPOD_API_KEY", "VAST_API_KEY",
                    "CYOAI_LLM_API_KEY", "OTHER_API_KEY", "GITHUB_TOKEN", "MY_SECRET_VALUE", "DB_PASSWORD"]
    for name in secret_names:
        monkeypatch.setenv(name, "never-forward-this")
    monkeypatch.setenv("CYOAI_SMOKE", "1")
    monkeypatch.setenv("CYOAI_DEVICE", "cpu")
    (workspace / "envcheck.py").write_text("import os, json\nprint(json.dumps(dict(os.environ)))\n")
    runner = LocalRunner(workspace, 17, cmd=f"{sys.executable} envcheck.py",
                         extra_env={"EXTRA_TOKEN": "also-secret", "SAFE_EXTRA": "ok"}, log=lambda _: None)
    code, output = runner.run()
    env = json.loads(output)
    assert code == 0
    assert not any(name in env for name in secret_names + ["EXTRA_TOKEN"])
    assert env["SAFE_EXTRA"] == "ok"
    assert env["CYOAI_TIME_BUDGET"] == "17"
    assert env["CYOAI_SMOKE"] == "1"
    assert env["CYOAI_DEVICE"] == "cpu"
    assert env["CYOAI_OUT_DIR"] == str(workspace / "out")


def test_best_artifacts(workspace):
    (workspace / "out").mkdir()
    for name in ("model.pt", "tokenizer.pkl", "train.py", "metrics.json"):
        (workspace / "out" / name).write_text("best")
    LocalRunner(workspace).keep()
    for name in ("model.pt", "tokenizer.pkl", "train.py", "metrics.json"):
        assert (workspace / "best" / name).read_text() == "best"
    git(workspace, "reset", "--hard", "HEAD")
    assert (workspace / "out/model.pt").exists()
    assert (workspace / "best/model.pt").exists()


def test_remote_runner(workspace, monkeypatch):
    calls = []
    class Provider:
        def sync_up(self, *args): calls.append(("up", args))
        def sync_down(self, *args): calls.append(("down", args))
        def exec_stream(self, instance, command):
            calls.append(("exec", command))
            yield "val_bpb: 1"
    monkeypatch.setenv("CYOAI_SMOKE", "1")
    runner = RemoteRunner(Provider(), "id", "/workspace/test", workspace, extra_env={"CYOAI_DEVICE": "cpu"})
    assert runner.run() == (0, "val_bpb: 1\n")
    runner.keep()
    assert "CYOAI_SMOKE=1" in calls[1][1]
    assert "CYOAI_DEVICE=cpu" in calls[1][1]
    assert "CYOAI_OUT_DIR=/workspace/test/out" in calls[1][1]
    assert "uv run --extra cpu train.py" in calls[1][1]
    assert {call[1][1].split("/")[-1] for call in calls if call[0] == "down"} == {"model.pt", "tokenizer.pkl", "train.py", "metrics.json"}


@pytest.mark.parametrize("body", ["time.sleep(30)", "\nwhile True:\n print('working', flush=True)\n time.sleep(.01)",
                                  "os.close(1); os.close(2); time.sleep(30)"])
def test_local_hard_timeout(workspace, monkeypatch, body):
    monkeypatch.setattr(runners, "hard_timeout_secs", lambda budget: .2)
    (workspace / "hang.py").write_text(
        "import os, time\nfrom pathlib import Path\nPath('pid').write_text(str(os.getpid()))\n" + body + "\n")
    start = time.monotonic()
    code, output = LocalRunner(workspace, cmd=f"{sys.executable} hang.py", log=lambda _: None).run()
    assert code != 0 and "hard timeout" in output
    assert time.monotonic() - start < 3
    with pytest.raises(ProcessLookupError):
        os.kill(int((workspace / "pid").read_text()), 0)


def test_remote_hard_timeout(workspace, monkeypatch):
    release = threading.Event()
    commands = []
    class Provider:
        def sync_up(self, *args): pass
        def exec_stream(self, instance, command):
            commands.append(command)
            if "pkill" in command:
                release.set()
                return
            yield "started"
            release.wait(5)
    monkeypatch.setattr(runners, "hard_timeout_secs", lambda budget: .1)
    start = time.monotonic()
    try:
        code, output = RemoteRunner(Provider(), "id", "/work", workspace).run()
        assert code != 0 and "hard timeout" in output
        assert time.monotonic() - start < 2
        assert "timeout --signal=TERM --kill-after=5s 0.1s env" in commands[0]
    finally:
        release.set()


@pytest.mark.parametrize("cleanup_fails", [False, True])
def test_remote_timeout_drains_cleanup_before_return(workspace, monkeypatch, cleanup_fails):
    release = threading.Event()
    events = []
    class Provider:
        def sync_up(self, *args): pass
        def exec_stream(self, instance, command):
            if "pkill" in command:
                events.append("cleanup started")
                assert "-KILL" in command and "'[t]rain.py'" in command
                release.set()
                yield "stopping training"
                if cleanup_fails:
                    events.append("cleanup failed")
                    raise RuntimeError("cleanup transport failed")
                events.append("cleanup finished")
                return
            yield "started"
            release.wait(5)
    monkeypatch.setattr(runners, "hard_timeout_secs", lambda budget: .05)
    try:
        code, output = RemoteRunner(Provider(), "id", "/work", workspace).run()
        events.append("returned")
        assert code != 0 and "hard timeout" in output
        assert events == ["cleanup started", "cleanup failed" if cleanup_fails else "cleanup finished", "returned"]
    finally:
        release.set()


@pytest.mark.parametrize("hang_in_factory", [False, True])
def test_remote_hanging_cleanup_has_deadline(workspace, monkeypatch, hang_in_factory):
    release = threading.Event()
    commands, logs = [], []

    def hanging_stream():
        yield "cleanup started"
        release.wait(5)

    class Provider:
        def sync_up(self, *args): pass
        def exec_stream(self, instance, command):
            if "pkill" not in command:
                raise RuntimeError("training failed")
            commands.append(command)
            if hang_in_factory:
                release.wait(5)
                return iter(())
            return hanging_stream()

    monkeypatch.setattr(RemoteRunner, "cleanup_timeout_secs", .05)
    start = time.monotonic()
    try:
        code, output = RemoteRunner(Provider(), "id", "/work", workspace, log=logs.append).run()
        assert code != 0 and "training failed" in output
        assert time.monotonic() - start < 2
        assert commands == ["timeout --signal=KILL 0.05s pkill -KILL -f '[t]rain.py' || true"]
        assert "Remote training cleanup failed or timed out" in logs
    finally:
        release.set()


@pytest.mark.parametrize("remote", [False, True])
@pytest.mark.parametrize("existing", [False, True])
def test_artifact_copy_failure_preserves_best(workspace, monkeypatch, remote, existing):
    best = workspace / "best"
    if existing:
        best.mkdir()
        (best / "model.pt").write_text("original")
        (best / "extra").write_text("also preserved")
    calls = []
    def copy(source, destination):
        calls.append(source)
        Path(destination).write_text("new")
        if len(calls) == 2:
            raise OSError("copy failed")
    if remote:
        class Provider:
            def sync_down(self, instance, source, destination): copy(source, destination)
        runner = RemoteRunner(Provider(), "id", "/work", workspace)
    else:
        monkeypatch.setattr(runners.shutil, "copy2", copy)
        runner = LocalRunner(workspace)
    with pytest.raises(OSError, match="copy failed"):
        runner.keep()
    if existing:
        assert {p.name: p.read_text() for p in best.iterdir()} == {
            "model.pt": "original", "extra": "also preserved"}
    else:
        assert not best.exists()
    assert not list(workspace.glob(".best-*"))


def test_artifact_swap_failure_rolls_back(workspace, monkeypatch):
    best = workspace / "best"
    best.mkdir()
    (best / "model.pt").write_text("original")
    rename = Path.rename
    def fail_publish(source, target):
        if source.name == "next":
            raise OSError("rename failed")
        return rename(source, target)
    monkeypatch.setattr(Path, "rename", fail_publish)
    with pytest.raises(OSError, match="rename failed"):
        with runners.staged_best(workspace) as staging:
            (staging / "model.pt").write_text("new")
    assert (best / "model.pt").read_text() == "original"
    assert not list(workspace.glob(".best-*"))


def test_artifact_swap_replaces_complete_directory(workspace):
    (workspace / "best").mkdir()
    (workspace / "best/old").write_text("old")
    with runners.staged_best(workspace) as staging:
        (staging / "new").write_text("new")
    assert [p.name for p in (workspace / "best").iterdir()] == ["new"]
    assert not list(workspace.glob(".best-*"))
