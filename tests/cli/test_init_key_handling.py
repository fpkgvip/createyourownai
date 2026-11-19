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


@pytest.mark.parametrize("field", ["llm_provider", "llm_base_url", "llm_api_key", "runpod_api_key"])
def test_wrong_type_credential_field_is_an_error(tmp_path, capsys, field):
    write_raw(tmp_path, '{"%s": 5}' % field)
    assert cli.main(["destroy", "--all"]) == 1
    assert f"{field} must be a string" in capsys.readouterr().err


def test_wrong_type_model_is_ignored_with_warning(tmp_path, capsys):
    write_raw(tmp_path, '{"llm_provider": "fake", "llm_model": 5, "runpod_api_key": "rp"}')
    assert cli.main(["init", "--non-interactive"]) == 0
    assert "llm_model must be a string; using the provider default" in capsys.readouterr().err
    assert '"llm_model": "fake"' in config_file(tmp_path).read_text()


def test_symlinked_config_target_is_not_chmodded(tmp_path, capsys):
    target = tmp_path / "other.json"
    target.write_text('{"llm_provider": "fake"}')
    target.chmod(0o644)
    config_file(tmp_path).parent.mkdir(parents=True)
    config_file(tmp_path).symlink_to(target)
    from cyoai.config import load_config
    assert load_config().llm_provider == "fake"
    assert "run `chmod 600" in capsys.readouterr().err
    assert stat.S_IMODE(target.stat().st_mode) == 0o644


def test_interactive_invalid_saved_provider_uses_flag(monkeypatch, tmp_path):
    write_raw(tmp_path, '{"llm_provider": "bogus"}')
    monkeypatch.setattr(builtins, "input", lambda prompt="": "")
    monkeypatch.setattr(getpass, "getpass", lambda prompt="": "")
    assert cli.main(["init", "--llm-provider", "fake"]) == 0
    assert '"llm_provider": "fake"' in config_file(tmp_path).read_text()


def test_interactive_base_url_can_be_cleared(monkeypatch, tmp_path):
    save_config(Config(llm_provider="openai", llm_api_key="k", llm_base_url="https://x.test"))
    answers = iter(["", "", "-"])
    monkeypatch.setattr(builtins, "input", lambda prompt="": next(answers))
    monkeypatch.setattr(getpass, "getpass", lambda prompt="": "")
    assert cli.main(["init"]) == 0
    assert '"llm_base_url": null' in config_file(tmp_path).read_text()
