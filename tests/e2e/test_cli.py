"""Exercise the installed console command without sending keys or buying GPU time."""

import csv
import os
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
        "UV_CACHE_DIR": os.environ.get("UV_CACHE_DIR", str(home / ".cache" / "uv")),
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
    cli("init", "--non-interactive")
    config = tmp_path / "home" / ".config" / "cyoai" / "config.json"
    assert config.is_file()
    if os.name == "posix":
        assert stat.S_IMODE(config.stat().st_mode) == 0o600

    goal = "Try a smaller learning rate"
    cli("train", goal, "--provider", "local", "--max-iters", "2", "--dry-run")

    (results,) = (tmp_path / "runs").glob("*/results.tsv")
    assert goal in results.with_name("program.md").read_text(encoding="utf-8")
    with results.open(encoding="utf-8", newline="") as stream:
        records = list(csv.DictReader(stream, delimiter="\t"))
    assert records
    assert any(float(record["val_bpb"]) > 0 for record in records)


def test_run_rejects_missing_checkpoint(cli):
    result = cli("run", "missing.pt", "--prompt", "Once upon a time", check=False)
    assert result.returncode != 0
    assert "missing.pt" in result.stdout + result.stderr


@pytest.mark.skipif(
    os.environ.get("CYOAI_RUN_CPU_SMOKE") != "1",
    reason="opt in to downloading training data and running PyTorch on CPU",
)
def test_tiny_local_training_and_inference(cli, tmp_path):
    cli(
        "train",
        "Train a tiny baseline that can complete a prompt",
        "--provider", "local",
        "--smoke",
        "--device", "cpu",
        "--max-iters", "1",
        "--budget-mins", "1",
        timeout=900,
    )

    (best,) = (tmp_path / "runs").glob("*/best")
    assert (best / "model.pt").stat().st_size > 0
    assert (best / "tokenizer.pkl").stat().st_size > 0
    assert "val_bpb" in (best.parent / "results.tsv").read_text(encoding="utf-8")

    prompt = "Once upon a time"
    completion = cli("run", str(best), "--prompt", prompt, timeout=120)
    assert completion.stdout.strip().removeprefix(prompt).strip(), completion.stderr
