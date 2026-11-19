import json
from pathlib import Path
import stat
import types

import pytest

from cyoai import cli
from cyoai.config import Config, load_config, masked_config, save_config


@pytest.fixture(autouse=True)
def isolated_config(tmp_path, monkeypatch):
    monkeypatch.setenv("CYOAI_CONFIG_DIR", str(tmp_path / "config"))
    for key in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY", "RUNPOD_API_KEY", "VAST_API_KEY",
                "CYOAI_LLM_PROVIDER", "CYOAI_LLM_MODEL", "CYOAI_LLM_BASE_URL", "CYOAI_FAKE_LLM_RESPONSES"):
        monkeypatch.delenv(key, raising=False)


def test_config_permissions_override(tmp_path, monkeypatch):
    save_config(Config(llm_provider="openai", llm_api_key="saved", runpod_api_key="runpod"))
    assert stat.S_IMODE((tmp_path / "config").stat().st_mode) == 0o700
    assert stat.S_IMODE((tmp_path / "config/config.json").stat().st_mode) == 0o600
    monkeypatch.setenv("OPENAI_API_KEY", "environment")
    monkeypatch.setenv("CYOAI_LLM_MODEL", "custom-model")
    monkeypatch.setenv("CYOAI_LLM_BASE_URL", "https://example.test/v1")
    config = load_config()
    assert config.llm_api_key == "environment"
    assert config.llm_model == "custom-model"
    assert config.llm_base_url == "https://example.test/v1"
    assert "environment" not in json.dumps(masked_config(config))
    assert "runpod\"" not in json.dumps(masked_config(config))


def test_init_noninteractive(tmp_path, capsys, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "private-key")
    monkeypatch.setenv("CYOAI_LLM_PROVIDER", "openai")
    assert cli.main(["init", "--non-interactive"]) == 0
    assert load_config().llm_api_key == "private-key"
    assert load_config().llm_model == "gpt-4o"
    assert "private-key" not in capsys.readouterr().out


def test_train_dry_run(tmp_path):
    dest = tmp_path / "runs"
    assert cli.main(["train", "learn simple prose", "--dry-run", "--workdir", str(dest), "--max-iters", "2"]) == 0
    workspace, = dest.iterdir()
    assert len((workspace / "results.tsv").read_text().splitlines()) == 4
    assert (workspace / "best/model.pt").exists()
    assert (workspace / "best/tokenizer.pkl").exists()


def test_init_provider_flag_with_env_key(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "private-key")
    assert cli.main(["init", "--non-interactive", "--llm-provider", "openai"]) == 0
    config = load_config()
    assert config.llm_provider == "openai"
    assert config.llm_model == "gpt-4o"
    assert config.llm_api_key == "private-key"


@pytest.mark.parametrize("exception,expected", [(RuntimeError("bootstrap failed"), 1), (KeyboardInterrupt(), 130)])
def test_remote_destroy_on_error(tmp_path, monkeypatch, exception, expected):
    destroyed = []
    (tmp_path / "train.py").write_text("x=0\n")
    class Provider:
        def provision(self, **kwargs): return types.SimpleNamespace(id="instance-id")
        def wait_ready(self, instance): return instance
        def exec_stream(self, *args): return iter(())
        def sync_up(self, *args): raise exception
        def destroy(self, instance): destroyed.append(instance)
    save_config(Config(llm_provider="fake", runpod_api_key="test-key"))
    monkeypatch.setattr(cli, "get_provider", lambda name, key: Provider())
    monkeypatch.setattr(cli, "create_workspace", lambda *args: tmp_path)
    assert cli.main(["train", "goal", "--provider", "runpod"]) == expected
    assert destroyed == ["instance-id"]


def test_remote_smoke_and_destroy_success(tmp_path, monkeypatch):
    commands, destroyed = [], []
    class Provider:
        def provision(self, **kwargs): return types.SimpleNamespace(id="id")
        def wait_ready(self, instance): return instance
        def sync_up(self, *args): pass
        def sync_down(self, instance, remote, local): Path(local).write_text("artifact")
        def destroy(self, instance): destroyed.append(instance)
        def exec_stream(self, instance, command):
            commands.append(command)
            yield "val_bpb: 1"
    source = tmp_path / "source"
    source.mkdir()
    (source / "train.py").write_text("x=0\n")
    (source / "program.md").write_text("goal\n")
    original = cli.create_workspace
    monkeypatch.setattr(cli, "create_workspace", lambda goal, dest: original(goal, dest, source))
    monkeypatch.setattr(cli, "get_provider", lambda *args: Provider())
    save_config(Config(llm_provider="fake", runpod_api_key="test-key"))
    assert cli.main(["train", "goal", "--provider", "runpod", "--smoke", "--device", "cpu",
                     "--workdir", str(tmp_path / "runs"), "--max-iters", "0"]) == 0
    assert "uv run prepare.py --num-shards 1" in commands[1]
    assert "CYOAI_SMOKE=1" in commands[2] and "CYOAI_DEVICE=cpu" in commands[2]
    assert destroyed == ["id"]


def test_chat_one_shot(tmp_path, monkeypatch, capsys):
    (tmp_path / "model.pt").write_text("model")
    fake = types.SimpleNamespace(load_model=lambda path: path,
                                 generate=lambda model, prompt, max_tokens: "generated:" + prompt)
    monkeypatch.setattr(cli.importlib, "import_module", lambda name: fake)
    assert cli.main(["run", str(tmp_path), "--prompt", "hello"]) == 0
    assert "generated:hello" in capsys.readouterr().out


def test_cli_errors(tmp_path):
    assert cli.main(["train", "goal", "--max-iters", "-1"]) == 1
    assert cli.main(["run", str(tmp_path / "absent")]) == 1
    assert cli.main(["destroy"]) == 1
