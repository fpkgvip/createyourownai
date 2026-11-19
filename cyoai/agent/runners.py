import fnmatch
import os
from pathlib import Path
import shlex
import shutil
import signal
import subprocess


def scrub_env(env):
    patterns = ("*_API_KEY", "*_TOKEN", "*_SECRET*", "*PASSWORD*")
    return {key: value for key, value in env.items()
            if not any(fnmatch.fnmatch(key.upper(), pattern) for pattern in patterns)}


class LocalRunner:
    def __init__(self, workspace, budget_secs=300, cmd="uv run train.py", extra_env=None, log=print):
        self.workspace, self.budget_secs = Path(workspace), budget_secs
        self.cmd, self.extra_env, self.log = cmd, extra_env or {}, log

    def run(self):
        env = scrub_env({**os.environ, **self.extra_env})
        env["CYOAI_TIME_BUDGET"] = str(int(self.budget_secs))
        process = subprocess.Popen(shlex.split(self.cmd), cwd=self.workspace, env=env,
                                   stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                                   start_new_session=True)
        output = []
        try:
            for line in process.stdout:
                self.log(line.rstrip())
                output.append(line)
            return process.wait(), "".join(output)
        finally:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()

    def keep(self):
        best = self.workspace / "best"
        for name in ("model.pt", "tokenizer.pkl"):
            source = self.workspace / name
            if source.exists():
                best.mkdir(exist_ok=True)
                shutil.copy2(source, best / name)


class RemoteRunner:
    def __init__(self, provider, instance, remote_dir, workspace, budget_secs=300, log=print, extra_env=None):
        self.provider, self.instance = provider, instance
        self.remote_dir, self.workspace = remote_dir, Path(workspace)
        self.budget_secs, self.log = budget_secs, log
        self.extra_env = extra_env or {}

    def run(self):
        self.provider.sync_up(self.instance, self.workspace / "train.py", self.remote_dir + "/train.py")
        lines = []
        env = {key: value for key, value in {**os.environ, **self.extra_env}.items()
               if key in {"CYOAI_SMOKE", "CYOAI_DEVICE"}}
        env["CYOAI_TIME_BUDGET"] = str(int(self.budget_secs))
        prefix = " ".join(f"{key}={shlex.quote(value)}" for key, value in env.items())
        command = f"cd {shlex.quote(self.remote_dir)} && {prefix} uv run train.py"
        try:
            for line in self.provider.exec_stream(self.instance, command):
                self.log(line.rstrip())
                lines.append(line.rstrip() + "\n")
        except Exception as exc:
            return 1, "".join(lines) + str(exc)
        return 0, "".join(lines)

    def keep(self):
        best = self.workspace / "best"
        best.mkdir(exist_ok=True)
        for name in ("model.pt", "tokenizer.pkl"):
            self.provider.sync_down(self.instance, self.remote_dir + "/" + name, best / name)


class FakeRunner:
    def __init__(self, workspace, log=print):
        self.workspace, self.log, self.iteration = Path(workspace), log, 0

    def run(self):
        self.iteration += 1
        output = f"val_bpb: {1 / self.iteration:.6f}\npeak_vram_mb: 0\n"
        self.log(output.rstrip())
        (self.workspace / "checkpoint").write_text("Synthetic dry-run checkpoint; not a trained model.\n")
        return 0, output

    def keep(self):
        best = self.workspace / "best"
        best.mkdir(exist_ok=True)
        for name in ("model.pt", "tokenizer.pkl"):
            (best / name).write_text("Synthetic dry-run artifact, not a trained model.\n")
