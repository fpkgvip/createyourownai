"""User configuration, kept outside research workspaces."""
from dataclasses import asdict, dataclass, fields
import json
import os
from pathlib import Path
import tempfile


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


def load_config() -> Config:
    path = config_dir() / "config.json"
    data = json.loads(path.read_text()) if path.exists() else {}
    data = {f.name: data[f.name] for f in fields(Config) if f.name in data}
    for field, env in {
        "llm_provider": "CYOAI_LLM_PROVIDER", "llm_model": "CYOAI_LLM_MODEL",
        "llm_base_url": "CYOAI_LLM_BASE_URL", "runpod_api_key": "RUNPOD_API_KEY",
        "vast_api_key": "VAST_API_KEY",
    }.items():
        if env in os.environ:
            data[field] = os.environ[env]
    provider = data.get("llm_provider", "anthropic")
    key_env = "OPENAI_API_KEY" if provider == "openai" else "ANTHROPIC_API_KEY"
    if key_env in os.environ:
        data["llm_api_key"] = os.environ[key_env]
    if "llm_model" not in data and provider == "openai":
        data["llm_model"] = "gpt-4o"
    return Config(**data)


def save_config(config: Config) -> None:
    directory = config_dir()
    directory.mkdir(parents=True, exist_ok=True, mode=0o700)
    directory.chmod(0o700)
    fd, temporary = tempfile.mkstemp(dir=directory)
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
