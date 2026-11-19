import json
import os
from pathlib import Path
import stat
import types
from contextlib import contextmanager

import pytest

from cyoai import cli
from cyoai.config import Config, load_config, masked_config, save_config


@pytest.fixture(autouse=True)
def isolated_config(tmp_path, monkeypatch):
    monkeypatch.setenv("CYOAI_CONFIG_DIR", str(tmp_path / "config"))
    for key in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY", "RUNPOD_API_KEY", "VAST_API_KEY",
                "CYOAI_LLM_PROVIDER", "CYOAI_LLM_MODEL", "CYOAI_LLM_BASE_URL", "CYOAI_FAKE_LLM_RESPONSES"):
        monkeypatch.delenv(key, raising=False)
    @contextmanager
    def manage(provider, **kwargs):
        instance = provider.provision(**kwargs)
        try:
            yield provider.wait_ready(instance)
        finally:
            provider.destroy(instance.id)
    monkeypatch.setattr(cli, "managed_instance", manage)


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


def test_train_dry_run(tmp_path, capsys):
    dest = tmp_path / "runs"
    assert cli.main(["train", "learn simple prose", "--dry-run", "--workdir", str(dest), "--max-iters", "2"]) == 0
    workspace, = dest.iterdir()
    assert len((workspace / "results.tsv").read_text().splitlines()) == 4
    assert (workspace / "best/model.pt").exists()
    assert (workspace / "best/tokenizer.pkl").exists()
    assert "no trained model" in capsys.readouterr().out


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
    assert "uv run --extra cpu prepare.py --num-shards 1" in commands[1]
    assert "CYOAI_SMOKE=1" in commands[2] and "CYOAI_DEVICE=cpu" in commands[2]
    assert destroyed == ["id"]


def test_chat_one_shot(tmp_path, monkeypatch, capsys):
    (tmp_path / "model.pt").write_text("model")
    fake = types.SimpleNamespace(load_model=lambda path, device: (path, "tokenizer", {}),
                                 generate=lambda model, tokenizer, prompt, **kwargs: "generated:" + prompt)
    monkeypatch.setattr(cli.importlib, "import_module", lambda name: fake)
    assert cli.main(["run", str(tmp_path), "--prompt", "hello"]) == 0
    assert "generated:hello" in capsys.readouterr().out


@pytest.mark.parametrize("command", [["run", "--latest"], ["chat"]])
def test_inference_selects_latest_trained_checkpoint(tmp_path, monkeypatch, capsys, command):
    monkeypatch.chdir(tmp_path)
    for index in (1, 2):
        checkpoint = tmp_path / f"runs/20000101-000000-00000{index}/best"
        checkpoint.mkdir(parents=True)
        (checkpoint / "model.pt").write_bytes(b"model")
        (checkpoint / "tokenizer.pkl").write_bytes(b"tokenizer")
    assert cli.main(["train", "later dry run", "--dry-run", "--max-iters", "0"]) == 0
    fake = types.SimpleNamespace(load_model=lambda path, device: (path, None, {}),
                                 generate=lambda model, tokenizer, prompt, **kwargs: str(model))
    monkeypatch.setattr(cli.importlib, "import_module", lambda name: fake)
    assert cli.main([*command, "--prompt", "hello"]) == 0
    assert capsys.readouterr().out.splitlines()[-1] == str(checkpoint.relative_to(tmp_path))


def test_inference_rejects_dry_run_artifacts_without_loading_torch(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    assert cli.main(["train", "simulate", "--dry-run", "--max-iters", "0"]) == 0
    (best,) = Path("runs").glob("*/best")
    monkeypatch.setattr(cli.importlib, "import_module", lambda name: pytest.fail("loaded inference"))
    assert cli.main(["run", "--latest", "--prompt", "hello"]) == 1
    assert "No trained checkpoint found under runs/" in capsys.readouterr().err
    for path in (best, best / "model.pt"):
        assert cli.main(["run", str(path), "--prompt", "hello"]) == 1
        assert "is a dry-run artifact" in capsys.readouterr().err


def test_cli_errors(tmp_path):
    assert cli.main(["train", "goal", "--max-iters", "-1"]) == 1
    assert cli.main(["run", str(tmp_path / "absent")]) == 1
    assert cli.main(["destroy"]) == 1


@pytest.mark.parametrize("arguments", [["status"], ["destroy", "--all"]])
def test_all_providers_skips_missing_credentials(arguments, capsys):
    assert cli.main(arguments) == 0
    assert "Skipping runpod" in capsys.readouterr().err
    assert cli.main(arguments + ["--provider", "runpod"]) == 1


def test_inference_uv_fallback(tmp_path, monkeypatch):
    (tmp_path / "model.pt").write_text("model")
    calls = []
    def missing(name):
        raise ModuleNotFoundError("No module named torch", name="torch")
    monkeypatch.setattr(cli.importlib, "import_module", missing)
    monkeypatch.setattr(cli.subprocess, "run", lambda command, **kwargs:
                        calls.append((command, kwargs)) or types.SimpleNamespace(returncode=0))
    assert cli.main(["run", str(tmp_path), "--prompt", "hi"]) == 0
    command = calls[0][0]
    assert command[:5] == ["uv", "run", "--extra", "cpu", "python"]
    assert "--prompt" in command
    assert "--max-new-tokens" in command


@pytest.mark.parametrize("command", ["run", "chat"])
def test_default_checkpoint_and_empty_runs(command, tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "runs").mkdir()
    assert cli.main([command, "--prompt", "hi"]) == 1
    assert "No trained checkpoint found" in capsys.readouterr().err
    for run in ("20260101", "20260102"):
        best = tmp_path / "runs" / run / "best"
        best.mkdir(parents=True)
        (best / "model.pt").touch()
        (best / "tokenizer.pkl").touch()
    paths = []
    fake = types.SimpleNamespace(
        load_model=lambda path, device: paths.append(path) or (None, None, {}),
        generate=lambda *args, **kwargs: "completion",
    )
    monkeypatch.setattr(cli.importlib, "import_module", lambda name: fake)
    assert cli.main([command, "--prompt", "hi"]) == 0
    assert paths == [Path("runs/20260102/best")]
    assert "completion" in capsys.readouterr().out


@pytest.mark.parametrize("value", ["exit", "quit", "/exit", " EXIT ", ""])
def test_chat_exit_commands(tmp_path, monkeypatch, value):
    (tmp_path / "model.pt").write_text("model")
    def unexpected(*args, **kwargs):
        pytest.fail("Exit commands must not generate text")
    fake = types.SimpleNamespace(load_model=lambda path, device: (None, None, {}), generate=unexpected)
    monkeypatch.setattr("builtins.input", lambda _: value)
    monkeypatch.setattr(cli.importlib, "import_module", lambda name: fake)
    assert cli.main(["chat", str(tmp_path)]) == 0


@pytest.mark.parametrize("option", ["--max-tokens", "--temperature", "--top-k"])
def test_generation_options_rejected_before_loading(option, tmp_path, monkeypatch, capsys):
    def unexpected(name):
        pytest.fail("Invalid options must not load inference dependencies")
    monkeypatch.setattr(cli.importlib, "import_module", unexpected)
    assert cli.main(["run", str(tmp_path), option, "-1"]) == 1
    assert "must be non-negative" in capsys.readouterr().err


def test_status_does_not_print_raw(monkeypatch, capsys):
    instance = types.SimpleNamespace(id="safe-id", status="running", gpu="safe-gpu",
                                     raw={"env": {"API_KEY": "must-not-print"}})
    provider = types.SimpleNamespace(list_instances=lambda: [instance])
    save_config(Config(runpod_api_key="test-key"))
    monkeypatch.setattr(cli, "get_provider", lambda *args: provider)
    assert cli.main(["status", "--provider", "runpod"]) == 0
    output = capsys.readouterr().out
    assert output == "runpod\tsafe-id\trunning\tsafe-gpu\n"
    assert "must-not-print" not in output and "API_KEY" not in output


@pytest.mark.parametrize("saved,selected,model,key_env", [
    ("anthropic", "openai", "gpt-4o", "OPENAI_API_KEY"),
    ("openai", "anthropic", "claude-sonnet-4-5", "ANTHROPIC_API_KEY"),
    ("openai", "fake", "fake", None),
])
@pytest.mark.parametrize("have_key", [False, True])
def test_provider_env_switch_clears_saved_llm_settings(monkeypatch, saved, selected, model, key_env, have_key):
    save_config(Config(llm_provider=saved, llm_model="saved-custom-model", llm_api_key="saved-key",
                       llm_base_url="https://old-provider.test", runpod_api_key="compute-key"))
    monkeypatch.setenv("CYOAI_LLM_PROVIDER", selected)
    if have_key:
        monkeypatch.setenv("OPENAI_API_KEY", "new-openai-key")
        monkeypatch.setenv("ANTHROPIC_API_KEY", "new-anthropic-key")
    config = load_config()
    assert config.llm_provider == selected
    assert config.llm_model == model
    assert config.llm_api_key == (os.environ[key_env] if have_key and key_env else "")
    assert config.llm_base_url is None
    assert config.runpod_api_key == "compute-key"
    monkeypatch.setenv("CYOAI_LLM_MODEL", "env-model")
    monkeypatch.setenv("CYOAI_LLM_BASE_URL", "https://new-provider.test")
    assert load_config().llm_model == "env-model"
    assert load_config().llm_base_url == "https://new-provider.test"


def test_same_provider_env_preserves_saved_llm_settings(monkeypatch):
    config = Config(llm_provider="openai", llm_model="custom-model", llm_api_key="saved-key",
                    llm_base_url="https://provider.test")
    save_config(config)
    monkeypatch.setenv("CYOAI_LLM_PROVIDER", "openai")
    assert load_config() == config
