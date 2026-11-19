"""Real CLI processes with local HTTP and SSH stand-ins; no cloud access."""
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest


API_KEY = "cloud-api-secret-sentinel"
PRIVATE_KEY = "private-ssh-secret-sentinel"


@pytest.fixture
def cloud_cli(tmp_path):
    state = {"live": False, "calls": [], "create_error": 0, "delete_error": 0,
             "phase": "", "created": threading.Event(), "release": threading.Event()}

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_request(self):
            self.rfile.read(int(self.headers.get("Content-Length", 0)))
            path = self.path.split("?")[0]
            method = self.command
            state["calls"].append((method, path))
            code, data = 200, {}
            create = (method == "POST" and path == "/pods") or (method == "PUT")
            if create:
                code = state["create_error"] or 200
                if code == 200:
                    state["live"] = True
                    state["created"].set()
                    if state["phase"] == "provision":
                        state["release"].wait(10)
                    data = {"id": "42", "status": "pending"} if path == "/pods" else {"new_contract": 42}
            elif method == "DELETE":
                code = state["delete_error"] or 204
                if code == 204:
                    state["live"] = False
            elif path == "/bundles/":
                data = {"offers": [{"id": 7, "dph_total": 0.2}]}
            elif method == "GET":
                pod = {"id": "42", "name": "cyoai", "status": "running",
                       "ssh": {"direct": {"host": "fake-host", "port": 22}}}
                vast = {"id": 42, "label": "cyoai", "actual_status": "running",
                        "ssh_host": "fake-host", "ssh_port": 22}
                live = [pod if path.startswith("/pods") else vast] if state["live"] else []
                if path == "/pods":
                    data = {"pods": live, "pagination": {"hasNextPage": False}}
                elif path == "/instances/":
                    data = {"instances": live}
                else:
                    data = pod if path.startswith("/pods") else {"instances": vast}
            if code >= 400:
                data = {"title": "Request failed", "detail": "Authorization: Bearer " + API_KEY}
            payload = json.dumps(data).encode()
            try:
                self.send_response(code)
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)
            except (BrokenPipeError, ConnectionResetError):
                pass

        do_GET = do_POST = do_PUT = do_DELETE = do_request

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    shim = tmp_path / "shim"
    shim.mkdir()
    (shim / "sitecustomize.py").write_text(
        "from cyoai.providers import RunPodProvider, VastProvider\n"
        f"RunPodProvider.api_url = VastProvider.api_url = 'http://127.0.0.1:{server.server_port}'\n")
    tool = shim / "ssh"
    tool.write_text(f"#!{sys.executable}\n" + '''import os, pathlib, sys, time
name = pathlib.Path(sys.argv[0]).name
command = sys.argv[-1]
phase = "sync" if name in {"rsync", "scp"} else ("train" if "timeout --signal" in command else "")
if phase and phase == os.environ.get("BLOCK_PHASE"):
    pathlib.Path(os.environ["PHASE_MARKER"] + ".pid").write_text(str(os.getpid()))
    pathlib.Path(os.environ["PHASE_MARKER"]).write_text(phase)
    time.sleep(15)
if name == "ssh" and phase == "train":
    print("val_bpb: 1.25", flush=True)
if name in {"rsync", "scp"} and "fake-host:" in sys.argv[-2]:
    pathlib.Path(command).write_text("artifact")
''')
    tool.chmod(0o755)
    (shim / "rsync").symlink_to(tool)
    (shim / "scp").symlink_to(tool)
    key = tmp_path / "key"
    key.write_text(PRIVATE_KEY)
    Path(str(key) + ".pub").write_text("ssh-ed25519 public-placeholder")
    env = {k: os.environ[k] for k in ("PATH", "LANG", "SYSTEMROOT") if k in os.environ}
    env.update(HOME=str(tmp_path), CYOAI_CONFIG_DIR=str(tmp_path / "config"),
               CYOAI_LLM_PROVIDER="fake", RUNPOD_API_KEY=API_KEY, VAST_API_KEY=API_KEY,
               CYOAI_SSH_KEY=str(key), PYTHONPATH=str(shim),
               PATH=str(shim) + os.pathsep + os.environ["PATH"],
               PHASE_MARKER=str(tmp_path / "phase"))

    def run(provider, *arguments, interrupt=None, phase=None):
        state["phase"] = phase or ""
        env["BLOCK_PHASE"] = phase or ""
        command = [str(Path(sys.executable).parent / "cyoai"), *arguments, "--provider", provider]
        if arguments[0] == "train":
            command += ["--max-iters", "0", "--workdir", str(tmp_path / "runs")]
        process = subprocess.Popen(command, env=env, cwd=tmp_path, stdout=subprocess.PIPE,
                                   stderr=subprocess.PIPE, text=True)
        if interrupt:
            deadline = time.monotonic() + 10
            while time.monotonic() < deadline:
                ready = state["created"].is_set() if phase == "provision" else (tmp_path / "phase").exists()
                if ready or process.poll() is not None:
                    break
                time.sleep(0.01)
            assert ready, process.communicate(timeout=20)
            process.send_signal(interrupt)
            state["release"].set()
        out, err = process.communicate(timeout=25)
        if phase in {"sync", "train"}:
            pid = int((tmp_path / "phase.pid").read_text())
            deadline = time.monotonic() + 2
            while Path(f"/proc/{pid}").exists() and time.monotonic() < deadline:
                time.sleep(0.01)
            assert not Path(f"/proc/{pid}").exists(), "SSH/transfer child survived CLI exit"
        assert API_KEY not in out + err
        assert PRIVATE_KEY not in out + err
        assert "Authorization:" not in out + err
        for root in (tmp_path / "runs", tmp_path / ".cache", tmp_path / "config"):
            for path in root.rglob("*"):
                if path.is_file() and ".git" not in path.parts:
                    content = path.read_bytes()
                    assert API_KEY.encode() not in content
                    assert PRIVATE_KEY.encode() not in content
        return process.returncode, out + err

    yield state, env, run
    state["release"].set()
    server.shutdown()
    server.server_close()
    thread.join()


@pytest.mark.parametrize("provider", ["runpod", "vast"])
def test_cloud_happy_lifecycle(cloud_cli, provider):
    state, env, run = cloud_cli
    code, output = run(provider, "train", "tiny goal")
    assert code == 0, output
    assert "Best val_bpb: 1.250000" in output
    assert not state["live"]
    assert any(method == "DELETE" for method, _ in state["calls"])
    state["live"] = True
    code, output = run(provider, "status")
    assert code == 0 and f"{provider}\t42\trunning" in output
    assert run(provider, "destroy", "42")[0] == 0
    assert not state["live"]
    assert run(provider, "status") == (0, "")
    assert run(provider, "destroy", "42")[0] == 0


@pytest.mark.parametrize("provider", ["runpod", "vast"])
@pytest.mark.parametrize("phase", ["provision", "sync", "train"])
@pytest.mark.parametrize("signum", [signal.SIGINT, signal.SIGTERM])
def test_cloud_signals(cloud_cli, provider, phase, signum):
    state, env, run = cloud_cli
    code, output = run(provider, "train", "tiny goal", interrupt=signum, phase=phase)
    assert code == 128 + signum, output
    assert not state["live"], output
    assert any(method == "DELETE" for method, _ in state["calls"])


@pytest.mark.parametrize("provider", ["runpod", "vast"])
@pytest.mark.parametrize("status", [401, 500])
@pytest.mark.parametrize("operation", ["create", "delete"])
def test_cloud_api_errors(cloud_cli, provider, status, operation):
    state, env, run = cloud_cli
    state[operation + "_error"] = status
    code, output = run(provider, "train", "tiny goal")
    assert code == 1, output
    assert f"HTTP {status}" in output
    if operation == "delete":
        assert state["live"]
        assert f"cyoai destroy 42 --provider {provider}" in output
        code, output = run(provider, "destroy", "42")
        assert code == 1 and f"HTTP {status}" in output
        state["delete_error"] = 0
        assert run(provider, "destroy", "42")[0] == 0
    assert not state["live"]


@pytest.mark.parametrize("provider", ["runpod", "vast"])
@pytest.mark.parametrize("missing", ["api", "ssh", "private"])
def test_cloud_missing_credentials(cloud_cli, provider, missing):
    state, env, run = cloud_cli
    if missing == "api":
        env.pop(provider.upper() + "_API_KEY")
    elif missing == "ssh":
        env["CYOAI_SSH_KEY"] += "-absent"
    else:
        Path(env["CYOAI_SSH_KEY"]).unlink()
    code, output = run(provider, "train", "tiny goal")
    assert code == 1, output
    assert "key" in output.lower()
    assert not state["live"]
    assert not state["calls"]
    if missing == "api":
        assert run(provider, "status")[0] == 1
        assert run(provider, "destroy", "42")[0] == 1
