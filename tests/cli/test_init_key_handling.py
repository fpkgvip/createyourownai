import builtins
import getpass
import stat

import pytest

from cyoai import cli
from cyoai.config import Config, save_config


@pytest.fixture(autouse=True)
def isolated_config(tmp_path, monkeypatch):
    monkeypatch.setenv("CYOAI_CONFIG_DIR", str(tmp_path / "config"))
    for key in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY", "RUNPOD_API_KEY", "VAST_API_KEY",
                "CYOAI_LLM_PROVIDER", "CYOAI_LLM_MODEL", "CYOAI_LLM_BASE_URL"):
        monkeypatch.delenv(key, raising=False)


def config_file(tmp_path):
    return tmp_path / "config/config.json"


def write_raw(tmp_path, text):
    config_file(tmp_path).parent.mkdir(parents=True, exist_ok=True)
    config_file(tmp_path).write_text(text)
    config_file(tmp_path).chmod(0o600)


@pytest.mark.parametrize("text, message", [
    ("{bad", "is not valid JSON (line 1, column 2)"),
    ("[1]", "must contain a JSON object"),
    ('{"llm_provider": 5}', "llm_provider must be a string"),
])
def test_broken_config_gives_clear_message(tmp_path, capsys, text, message):
    write_raw(tmp_path, text)
    assert cli.main(["init", "--non-interactive"]) == 1
    err = capsys.readouterr().err
    assert message in err and "Traceback" not in err


def test_group_readable_config_is_tightened(tmp_path, capsys):
    save_config(Config(llm_provider="fake", llm_api_key="sk-secret"))
    config_file(tmp_path).chmod(0o644)
    assert cli.main(["init", "--non-interactive"]) == 0
    out = capsys.readouterr()
    assert "changed it to 600" in out.err
    assert "sk-secret" not in out.out + out.err
    assert stat.S_IMODE(config_file(tmp_path).stat().st_mode) == 0o600


def test_unreadable_config_gives_clear_message(tmp_path, capsys):
    save_config(Config(llm_provider="fake"))
    config_file(tmp_path).chmod(0o000)
    try:
        code = cli.main(["init", "--non-interactive"])
    finally:
        config_file(tmp_path).chmod(0o600)
    if code == 0:
        pytest.skip("running as root")
    assert "Cannot read config file" in capsys.readouterr().err


def test_wrong_provider_name(capsys):
    assert cli.main(["init", "--non-interactive", "--llm-provider", "bogus"]) == 1
    assert "Unknown LLM provider 'bogus'" in capsys.readouterr().err


def test_provider_name_is_case_insensitive(tmp_path):
    assert cli.main(["init", "--non-interactive", "--llm-provider", "OpenAI", "--llm-api-key", "k"]) == 0
    assert '"llm_provider": "openai"' in config_file(tmp_path).read_text()


def test_unknown_env_provider_at_train(monkeypatch, tmp_path, capsys):
    monkeypatch.setenv("CYOAI_LLM_PROVIDER", "bogus")
    assert cli.main(["train", "goal", "--max-iters", "0", "--workdir", str(tmp_path / "runs")]) == 1
    assert "Unknown LLM provider 'bogus'" in capsys.readouterr().err


def test_interactive_eof_saves_nothing(monkeypatch, tmp_path, capsys):
    def eof(*args):
        raise EOFError
    monkeypatch.setattr(builtins, "input", eof)
    assert cli.main(["init"]) == 1
    assert "Input ended before setup finished" in capsys.readouterr().err
    assert not config_file(tmp_path).exists()


def test_interactive_rerun_keeps_existing_values(monkeypatch, tmp_path, capsys):
    save_config(Config(llm_provider="openai", llm_model="my-model", llm_api_key="sk-keep",
                       llm_base_url="https://x.test", runpod_api_key="rp"))
    monkeypatch.setattr(builtins, "input", lambda prompt="": "")
    monkeypatch.setattr(getpass, "getpass", lambda prompt="": "")
    assert cli.main(["init"]) == 0
    text = config_file(tmp_path).read_text()
    for value in ("my-model", "sk-keep", "https://x.test", '"rp"'):
        assert value in text
    assert "sk-keep" not in capsys.readouterr().out
    assert stat.S_IMODE(config_file(tmp_path).stat().st_mode) == 0o600
