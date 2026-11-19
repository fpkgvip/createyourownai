import json
import sys

import httpx
import pytest

from cyoai.agent.llm import AnthropicClient, FakeLLM, OpenAIClient, make_llm
from cyoai.agent.loop import ResearchLoop, parse_metrics, parse_proposal
from cyoai.agent.runners import LocalRunner, RemoteRunner
from cyoai.agent.workspace import create_workspace, git
from cyoai.config import Config


@pytest.fixture
def workspace(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "train.py").write_text("x = 0\n")
    (source / "prepare.py").write_text("# immutable\n")
    (source / "program.md").write_text("Minimize val_bpb\n")
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


def test_best_artifacts(workspace):
    for name in ("model.pt", "tokenizer.pkl"):
        (workspace / name).write_text("best")
    LocalRunner(workspace).keep()
    for name in ("model.pt", "tokenizer.pkl"):
        assert (workspace / "best" / name).read_text() == "best"


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
    assert {call[1][1].split("/")[-1] for call in calls if call[0] == "down"} == {"model.pt", "tokenizer.pkl"}
