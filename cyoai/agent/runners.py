import fnmatch
import os
from pathlib import Path
import queue
import shlex
import shutil
import signal
import subprocess
import threading
import time


def hard_timeout_secs(budget_secs):
    return 2 * budget_secs + 120


def stream_until(stream, deadline):
    messages = queue.Queue()

    def read():
        try:
            for line in stream:
                messages.put(("line", line))
        except BaseException as exc:
            messages.put(("error", exc))
        finally:
            messages.put(("end", None))

    reader = threading.Thread(target=read, daemon=True)
    reader.start()
    try:
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("Training exceeded hard timeout")
            try:
                kind, value = messages.get(timeout=remaining)
            except queue.Empty:
                raise TimeoutError("Training exceeded hard timeout") from None
            if kind == "end":
                break
            if kind == "error":
                raise value
            yield value
    finally:
        close = getattr(stream, "close", None)
        if close is not None and not reader.is_alive():
            try:
                close()
            except (ValueError, RuntimeError):
                pass


def scrub_env(env):
    patterns = ("*_API_KEY", "*_TOKEN", "*_SECRET*", "*PASSWORD*")
    return {key: value for key, value in env.items()
            if not any(fnmatch.fnmatch(key.upper(), pattern) for pattern in patterns)}


class LocalRunner:
    def __init__(self, workspace, budget_secs=300, cmd=None, extra_env=None, log=print):
        self.workspace, self.budget_secs = Path(workspace), budget_secs
        self.extra_env, self.log = extra_env or {}, log
        cpu = self.extra_env.get("CYOAI_DEVICE", os.environ.get("CYOAI_DEVICE")) == "cpu"
        self.cmd = cmd or ("uv run --extra cpu train.py" if cpu else "uv run train.py")

    def run(self):
        env = scrub_env({**os.environ, **self.extra_env})
        env["CYOAI_TIME_BUDGET"] = str(int(self.budget_secs))
        env["CYOAI_OUT_DIR"] = str((self.workspace / "out").resolve())
        process = subprocess.Popen(shlex.split(self.cmd), cwd=self.workspace, env=env,
                                   stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                                   start_new_session=True)
        output = []
        deadline = time.monotonic() + hard_timeout_secs(self.budget_secs)
        try:
            for line in stream_until(process.stdout, deadline):
                self.log(line.rstrip())
                output.append(line)
            return process.wait(timeout=max(0, deadline - time.monotonic())), "".join(output)
        except (TimeoutError, subprocess.TimeoutExpired):
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            self.log("Training exceeded hard timeout")
            return 124, "".join(output) + "\nTraining exceeded hard timeout\n"
        finally:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
            process.stdout.close()

    def keep(self):
        best = self.workspace / "best"
        best.mkdir(exist_ok=True)
        for name in ("model.pt", "tokenizer.pkl", "train.py", "metrics.json"):
            shutil.copy2(self.workspace / "out" / name, best / name)


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
        env["CYOAI_OUT_DIR"] = self.remote_dir + "/out"
        prefix = " ".join(f"{key}={shlex.quote(value)}" for key, value in env.items())
        run = "uv run --extra cpu" if env.get("CYOAI_DEVICE") == "cpu" else "uv run"
        timeout = hard_timeout_secs(self.budget_secs)
        command = (f"cd {shlex.quote(self.remote_dir)} && timeout --signal=TERM --kill-after=5s "
                   f"{timeout}s env {prefix} {run} train.py")
        try:
            stream = self.provider.exec_stream(self.instance, command)
            for line in stream_until(stream, time.monotonic() + timeout):
                self.log(line.rstrip())
                lines.append(line.rstrip() + "\n")
        except Exception as exc:
            return 1, "".join(lines) + str(exc)
        return 0, "".join(lines)

    def keep(self):
        best = self.workspace / "best"
        best.mkdir(exist_ok=True)
        for name in ("model.pt", "tokenizer.pkl", "train.py", "metrics.json"):
            self.provider.sync_down(self.instance, self.remote_dir + "/out/" + name, best / name)


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
