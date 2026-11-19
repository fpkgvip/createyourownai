"""Exercise the installed console command without sending keys or buying GPU time."""

import csv
import json
import os
from pathlib import Path
import shutil
import stat
import subprocess

import pytest


@pytest.fixture
def cli(tmp_path):
    executable = shutil.which("cyoai")
    assert executable, "Install cyoai and run these tests from its environment (uv run pytest)."

    home = tmp_path / "home"
    home.mkdir()
    env = {
        "PATH": os.environ["PATH"],
        "HOME": str(home),
        "XDG_CONFIG_HOME": str(home / ".config"),
        "CYOAI_CONFIG_DIR": str(home / ".config" / "cyoai"),
        "CYOAI_LLM_PROVIDER": "fake",
        "UV_CACHE_DIR": os.environ.get("UV_CACHE_DIR", str(Path.home() / ".cache" / "uv")),
        "PYTHONUTF8": "1",
        "TERM": "dumb",
    }

    def invoke(*args, timeout=30, overrides=None, check=True):
        result = subprocess.run(
            [executable, *args],
            cwd=tmp_path,
            env=env | (overrides or {}),
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        if check:
            assert result.returncode == 0, result.stdout + result.stderr
        return result

    return invoke


def test_init_and_local_dry_run(cli, tmp_path):
    dummy_key = "offline-test-credential"
    result = cli("init", "--non-interactive", overrides={
        "CYOAI_LLM_PROVIDER": "openai",
        "OPENAI_API_KEY": dummy_key,
    })
    config = tmp_path / "home" / ".config" / "cyoai" / "config.json"
    assert json.loads(config.read_text(encoding="utf-8"))["llm_api_key"] == dummy_key
    assert dummy_key not in result.stdout + result.stderr
    if os.name == "posix":
        assert stat.S_IMODE(config.stat().st_mode) == 0o600

    goal = "Try a smaller learning rate"
    cli("train", goal, "--provider", "local", "--max-iters", "2", "--dry-run")

    (results,) = (tmp_path / "runs").glob("*/results.tsv")
    assert goal in results.with_name("program.md").read_text(encoding="utf-8")
    with results.open(encoding="utf-8", newline="") as stream:
        records = list(csv.DictReader(stream, delimiter="\t"))
    assert len(records) == 3  # one baseline plus two experiments
    assert records[0]["description"] == "baseline"
    assert all(record["status"] == "keep" for record in records)
    scores = [float(record["val_bpb"]) for record in records]
    assert scores[0] > scores[1] > scores[2] > 0


def test_run_rejects_missing_checkpoint(cli):
    result = cli("run", "missing.pt", "--prompt", "Once upon a time", check=False)
    assert result.returncode != 0
    assert "missing.pt" in result.stdout + result.stderr


@pytest.mark.skipif(
    os.environ.get("CYOAI_RUN_CPU_SMOKE") != "1",
    reason="opt in to downloading training data and running PyTorch on CPU",
)
def test_tiny_local_training_and_inference(cli, tmp_path):
    data_cache = os.environ.get("CYOAI_CACHE_DIR")
    cli(
        "train",
        "Train a tiny baseline that can complete a prompt",
        "--provider", "local",
        "--smoke",
        "--device", "cpu",
        "--max-iters", "0",
        "--budget-mins", "0.05",
        timeout=900,
        overrides={"CYOAI_CACHE_DIR": str(Path(data_cache).resolve())} if data_cache else None,
    )

    (best,) = (tmp_path / "runs").glob("*/best")
    assert (best / "model.pt").stat().st_size > 0
    assert (best / "tokenizer.pkl").stat().st_size > 0
    assert (best / "train.py").is_file()
    metrics = json.loads((best / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["num_steps"] > 0
    assert 0 < metrics["val_bpb"] < float("inf")

    prompt = "Once upon a time"
    completion = cli("run", str(best), "--prompt", prompt, "--max-tokens", "16", timeout=120)
    assert completion.stdout.strip() not in {"", prompt}, completion.stderr


@pytest.mark.parametrize(
    "provider,key_env,list_url,list_body,instance_url",
    [
        (
            "runpod", "RUNPOD_API_KEY", "https://api.runpod.io/v2/pods",
            {"pods": [{"id": "qa-instance", "name": "cyoai", "status": "RUNNING",
                       "env": {"PRIVATE_VALUE": "provider-only-metadata"}}],
             "pagination": {"hasNextPage": False}},
            "https://api.runpod.io/v2/pods/qa-instance",
        ),
        (
            "vast", "VAST_API_KEY", "https://console.vast.ai/api/v0/instances/",
            {"instances": [{"id": 12345, "label": "cyoai", "actual_status": "running",
                            "env": {"PRIVATE_VALUE": "provider-only-metadata"}}]},
            "https://console.vast.ai/api/v0/instances/12345/",
        ),
    ],
)
def test_cloud_status_and_destroy(provider, key_env, list_url, list_body, instance_url,
                                  monkeypatch, tmp_path, capsys):
    import responses
    from cyoai.cli import main

    monkeypatch.setenv("CYOAI_CONFIG_DIR", str(tmp_path / "config"))
    monkeypatch.setenv(key_env, "offline-test-credential")
    instance_id = instance_url.rstrip("/").rsplit("/", 1)[-1]
    auth = responses.matchers.header_matcher({"Authorization": "Bearer offline-test-credential"})
    with responses.RequestsMock() as api:
        api.get(list_url, json=list_body, match=[auth])
        api.delete(instance_url, status=204, match=[auth])

        assert main(["status", "--provider", provider]) == 0
        output = capsys.readouterr().out
        assert instance_id in output
        assert "provider-only-metadata" not in output
        assert main(["destroy", instance_id, "--provider", provider]) == 0
