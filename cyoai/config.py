"""User configuration, kept outside research workspaces."""
from dataclasses import asdict, dataclass, fields
import json
import os
from pathlib import Path
import stat
import sys
import tempfile

PROVIDERS = ("anthropic", "openai", "fake")
DEFAULT_MODELS = {"openai": "gpt-4o", "anthropic": "claude-sonnet-4-5", "fake": "fake"}


class ConfigError(ValueError):
    pass


@dataclass
class Config:
    llm_provider: str = "anthropic"
    llm_model: str = "claude-sonnet-4-5"
    llm_api_key: str = ""
    llm_base_url: str | None = None
    runpod_api_key: str = ""
    vast_api_key: str = ""


def config_dir() -> Path:
    return Path(os.environ.get("CYOAI_CONFIG_DIR", "~/.config/cyoai")).expanduser()


def _read_config_file(path: Path) -> dict:
    if not path.exists():
        return {}
    hint = f"Fix it or delete it and run `cyoai init` again."
    try:
        if not path.is_file():
            raise ConfigError(f"Config path {path} is not a file. {hint}")
        mode = stat.S_IMODE(path.stat().st_mode)
        if mode & 0o077:
            try:
                path.chmod(0o600)
                print(f"cyoai: warning: {path} was readable by other users (mode {mode:o}); "
                      "changed it to 600", file=sys.stderr)
            except OSError:
                print(f"cyoai: warning: {path} is readable by other users (mode {mode:o}); "
                      f"run `chmod 600 {path}`", file=sys.stderr)
        text = path.read_text()
    except OSError as exc:
        raise ConfigError(f"Cannot read config file {path}: {exc.strerror}. "
                          f"Run `chmod 600 {path}` or delete it and run `cyoai init`.") from None
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ConfigError(f"Config file {path} is not valid JSON "
                          f"(line {exc.lineno}, column {exc.colno}). {hint}") from None
    if not isinstance(data, dict):
        raise ConfigError(f"Config file {path} must contain a JSON object. {hint}")
    for f in fields(Config):
        value = data.get(f.name)
        if f.name in data and not (isinstance(value, str) or (value is None and f.name == "llm_base_url")):
            raise ConfigError(f"Config file {path}: {f.name} must be a string. {hint}")
    return data


def load_config() -> Config:
    data = _read_config_file(config_dir() / "config.json")
    data = {f.name: data[f.name] for f in fields(Config) if f.name in data}
    provider_override = os.environ.get("CYOAI_LLM_PROVIDER")
    if provider_override is not None and provider_override != data.get("llm_provider", "anthropic"):
        for field in ("llm_model", "llm_api_key", "llm_base_url"):
            data.pop(field, None)
    for field, env in {
        "llm_provider": "CYOAI_LLM_PROVIDER", "llm_model": "CYOAI_LLM_MODEL",
        "llm_base_url": "CYOAI_LLM_BASE_URL", "runpod_api_key": "RUNPOD_API_KEY",
        "vast_api_key": "VAST_API_KEY",
    }.items():
        if env in os.environ:
            data[field] = os.environ[env]
    provider = data.get("llm_provider", "anthropic")
    key_env = {"openai": "OPENAI_API_KEY", "anthropic": "ANTHROPIC_API_KEY"}.get(provider)
    if key_env is not None and key_env in os.environ:
        data["llm_api_key"] = os.environ[key_env]
    if "llm_model" not in data:
        data["llm_model"] = DEFAULT_MODELS.get(provider, "claude-sonnet-4-5")
    return Config(**data)


def save_config(config: Config) -> None:
    directory = config_dir()
    try:
        directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        directory.chmod(0o700)
        fd, temporary = tempfile.mkstemp(dir=directory)
    except OSError as exc:
        raise ConfigError(f"Cannot write config directory {directory}: {exc.strerror}") from None
    try:
        with os.fdopen(fd, "w") as handle:
            json.dump(asdict(config), handle, indent=2)
            handle.write("\n")
        os.chmod(temporary, 0o600)
        os.replace(temporary, directory / "config.json")
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def masked_config(config: Config) -> dict:
    return {key: ("********" if value else "") if "key" in key else value
            for key, value in asdict(config).items()}
