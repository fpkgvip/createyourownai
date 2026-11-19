import io
import json
import os
from pathlib import Path
import shlex
import signal
from unittest.mock import Mock, patch

import pytest
import requests
import responses

from cyoai.providers import (
    Instance, LocalProvider, MissingCredentialsError, PROVIDERS, ProviderError,
    RunPodProvider, VastProvider, get_provider, managed_instance,
)
from cyoai.providers import ssh


@pytest.fixture(autouse=True)
def isolated_home(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))


@pytest.fixture
def key(tmp_path):
    path = tmp_path / "id_test"
    path.write_text("private-test-placeholder")
    path.with_suffix(".pub").write_text("ssh-ed25519 test-public-key\n")
    return str(path)


@pytest.fixture(params=[RunPodProvider, VastProvider])
def cloud(request, key):
    return request.param(api_key="test-key", ssh_key_path=key)


def test_factory():
    assert set(PROVIDERS) == {"local", "runpod", "vast"}
    assert isinstance(get_provider("local"), LocalProvider)
    assert isinstance(get_provider("runpod", api_key="test"), RunPodProvider)
    assert isinstance(get_provider("vast", api_key="test"), VastProvider)
    with pytest.raises(ProviderError, match="Unknown provider"):
        get_provider("other")


@pytest.mark.parametrize("cls,env", [(RunPodProvider, "RUNPOD_API_KEY"),
                                     (VastProvider, "VAST_API_KEY")])
def test_credentials(cls, env, monkeypatch):
    monkeypatch.delenv(env, raising=False)
    with pytest.raises(MissingCredentialsError, match=env):
        cls()
    monkeypatch.setenv(env, "from-env")
    assert cls().api_key == "from-env"
    assert cls(api_key="explicit").api_key == "explicit"


@responses.activate
@pytest.mark.parametrize("custom", [False, True])
def test_runpod_provision(key, custom):
    provider = RunPodProvider(api_key="test-key", ssh_key_path=key)
    assert provider.api_url == "https://api.runpod.io/v2"
    responses.post(provider.api_url + "/pods", json={"id": "p1", "status": "PROVISIONING", "ssh": {"direct": None, "proxy": None}})
    kwargs = {"gpu": "A100", "image": "custom-image", "disk_gb": 100, "name": "cyoai-test"} if custom else {}
    instance = provider.provision(**kwargs)
    body = json.loads(responses.calls[0].request.body)
    assert body == {
        "name": kwargs.get("name", "cyoai"), "image": kwargs.get("image", provider.default_image),
        "gpu": {"id": kwargs.get("gpu", "NVIDIA GeForce RTX 4090"), "count": 1},
        "disk": kwargs.get("disk_gb", 50), "ports": ["22/tcp"],
        "startSsh": True, "env": {"PUBLIC_KEY": "ssh-ed25519 test-public-key"},
    }
    assert responses.calls[0].request.headers["Authorization"] == "Bearer test-key"
    assert instance.id == "p1"
    assert instance.status == "provisioning"
    assert instance.ssh_host is None


@responses.activate
def test_vast_provision_cheapest(key):
    provider = VastProvider(api_key="test-key", ssh_key_path=key)
    responses.post(provider.api_url + "/bundles/", json={"offers": [
        {"id": 1, "dph_total": 0.9}, {"id": 2, "dph_total": 0.3, "gpu_name": "RTX_4090"},
    ]})
    responses.put(provider.api_url + "/asks/2/", json={"success": True, "new_contract": 42})
    responses.post(provider.api_url + "/instances/42/ssh/", json={"success": True})
    instance = provider.provision()
    assert instance.id == "42"
    assert instance.cost_per_hr == 0.3
    assert json.loads(responses.calls[0].request.body) == {
        "num_gpus": {"eq": 1}, "gpu_name": {"eq": "RTX_4090"},
        "rentable": {"eq": True}, "verified": {"eq": True},
        "disk_space": {"gte": 50}, "order": [["dph_total", "asc"]], "type": "on-demand",
    }
    assert json.loads(responses.calls[1].request.body) == {
        "client_id": "me", "image": provider.default_image, "disk": 50,
        "runtype": "ssh", "label": "cyoai",
    }
    assert json.loads(responses.calls[2].request.body) == {"ssh_key": "ssh-ed25519 test-public-key"}
    assert all(call.request.headers["Authorization"] == "Bearer test-key" for call in responses.calls)


@responses.activate
def test_vast_custom_and_attach_failure_cleanup(key):
    provider = VastProvider(api_key="test", ssh_key_path=key)
    responses.post(provider.api_url + "/bundles/", json={"offers": [{"id": 1, "dph_total": 1}]})
    responses.put(provider.api_url + "/asks/1/", json={"new_contract": 2})
    responses.post(provider.api_url + "/instances/2/ssh/", status=500)
    responses.delete(provider.api_url + "/instances/2/", status=204)
    with pytest.raises(ProviderError, match="HTTP 500"):
        provider.provision(gpu="A100", image="custom", disk_gb=90, name="cyoai-test")
    assert json.loads(responses.calls[0].request.body)["gpu_name"] == {"eq": "A100"}
    assert json.loads(responses.calls[1].request.body) == {
        "client_id": "me", "image": "custom", "disk": 90, "runtype": "ssh", "label": "cyoai-test",
    }
    assert responses.calls[-1].request.method == "DELETE"


@responses.activate
def test_no_vast_offers(key):
    provider = VastProvider(api_key="test", ssh_key_path=key)
    responses.post(provider.api_url + "/bundles/", json={"offers": []})
    with pytest.raises(ProviderError, match="No matching"):
        provider.provision()
    assert len(responses.calls) == 1


@responses.activate
def test_runpod_get_list():
    provider = RunPodProvider(api_key="test")
    data = {"id": "p1", "name": "cyoai", "status": "RUNNING",
            "ssh": {"direct": {"host": "1.2.3.4", "port": 2201, "username": "trainer",
                               "command": "ssh trainer@1.2.3.4 -p 2201"},
                    "proxy": {"host": "ssh.runpod.io", "port": 22, "username": "opaque",
                              "command": "ssh opaque@ssh.runpod.io"}},
            "cost": 0.5, "gpu": {"id": "4090", "count": 1}}
    responses.get(provider.api_url + "/pods/p1", json=data)
    responses.get(provider.api_url + "/pods", json={
        "pods": [data, {**data, "name": "unrelated"}, {**data, "name": None},
                 {**data, "id": "p2", "name": "cyoai-test"}],
        "pagination": {"nextCursor": None, "hasNextPage": False}})
    instance = provider.get("p1")
    assert (instance.ssh_host, instance.ssh_port, instance.gpu, instance.cost_per_hr) == ("1.2.3.4", 2201, "4090", 0.5)
    assert instance.raw == data
    assert instance.ssh_user == "trainer"
    assert [i.id for i in provider.list_instances()] == ["p1", "p2"]


@responses.activate
def test_runpod_paginated_list():
    provider = RunPodProvider(api_key="test")
    url = "https://api.runpod.io/v2/pods"
    responses.get(url, json={"pods": [{"id": "other", "name": "other", "status": "RUNNING"}],
                            "pagination": {"nextCursor": "opaque+/=", "hasNextPage": True}})
    responses.get(url + "?cursor=opaque%2B%2F%3D", json={
        "pods": [{"id": "p1", "name": "cyoai", "status": "RUNNING", "ssh": {"direct": None}}],
        "pagination": {"nextCursor": "last", "hasNextPage": True}})
    responses.get(url + "?cursor=last", json={
        "pods": [{"id": "p2", "name": "cyoai-test", "status": "ERROR"}],
        "pagination": {"nextCursor": None, "hasNextPage": False}})
    instances = provider.list_instances()
    assert [item.id for item in instances] == ["p1", "p2"]
    assert instances[0].ssh_host is None
    assert len(responses.calls) == 3


@responses.activate
@pytest.mark.parametrize("cursor", [None, "repeated"])
def test_runpod_invalid_pagination(cursor):
    provider = RunPodProvider(api_key="test")
    responses.get(provider.api_url + "/pods", json={
        "pods": [], "pagination": {"nextCursor": cursor, "hasNextPage": True}})
    with pytest.raises(ProviderError, match="pagination cursor"):
        provider.list_instances()


@responses.activate
@pytest.mark.parametrize("status", ["PROVISIONING", "STARTING", "RUNNING"])
def test_runpod_waits_for_direct_ssh(status):
    provider = RunPodProvider(api_key="test")
    responses.get(provider.api_url + "/pods/p1", json={
        "id": "p1", "status": status, "ssh": {"direct": None,
        "proxy": {"host": "ssh.runpod.io", "port": 22, "username": "opaque"}}})
    with pytest.raises(ProviderError, match="Timed out"):
        provider.wait_ready(Instance("p1", "runpod", "pending"), timeout=0)


@responses.activate
def test_runpod_error_status_fails_immediately():
    provider = RunPodProvider(api_key="test")
    responses.get(provider.api_url + "/pods/p1", json={"id": "p1", "status": "ERROR"})
    with patch("cyoai.providers.base.time.sleep") as sleep:
        with pytest.raises(ProviderError, match="p1: error"):
            provider.wait_ready(Instance("p1", "runpod", "pending"))
    sleep.assert_not_called()


@responses.activate
@pytest.mark.parametrize("problem,expected", [
    ({"title": "Bad Request", "status": 422, "detail": "GPU type is unavailable."},
     "Bad Request: GPU type is unavailable."),
    ({"title": "Bad Request"}, "Bad Request"),
    ({"detail": "Invalid disk"}, "Invalid disk"),
    ({"detail": "Invalid secret-api-key"}, "Invalid [REDACTED]"),
    (["unexpected"], "HTTP 422"),
    ({"title": 5, "detail": None}, "HTTP 422"),
])
def test_runpod_problem_errors(problem, expected):
    provider = RunPodProvider(api_key="secret-api-key")
    responses.get(provider.api_url + "/pods", status=422, json=problem,
                  content_type="application/problem+json")
    with pytest.raises(ProviderError) as error:
        provider.list_instances()
    assert expected in str(error.value)
    assert "secret-api-key" not in str(error.value)


@responses.activate
def test_vast_get_list():
    provider = VastProvider(api_key="test")
    data = {"id": 42, "label": "cyoai", "actual_status": "running", "ssh_host": "ssh.vast.ai",
            "ssh_port": 22042, "gpu_name": "RTX_4090", "dph_total": 0.3}
    responses.get(provider.api_url + "/instances/42/", json={"instances": data})
    responses.get(provider.api_url + "/instances/", json={"instances": [data,
        {**data, "label": "other"}, {**data, "label": None}, {**data, "id": 43, "label": "cyoai-test"}]})
    instance = provider.get("42")
    assert (instance.ssh_host, instance.ssh_port, instance.gpu, instance.cost_per_hr) == ("ssh.vast.ai", 22042, "RTX_4090", 0.3)
    assert [i.id for i in provider.list_instances()] == ["42", "43"]


def instance_url(provider, suffix=""):
    return provider.api_url + ("/pods" if provider.name == "runpod" else "/instances/") + suffix


@responses.activate
def test_empty_list(cloud):
    responses.get(instance_url(cloud), json={"pods": [], "pagination": {"nextCursor": None, "hasNextPage": False}}
                  if cloud.name == "runpod" else {"instances": []})
    assert cloud.list_instances() == []


@responses.activate
@pytest.mark.parametrize("status", [204, 404])
def test_destroy(cloud, status):
    path = "/pods/42" if cloud.name == "runpod" else "/instances/42/"
    responses.delete(cloud.api_url + path, status=status)
    assert cloud.destroy("42") is None


@responses.activate
@pytest.mark.parametrize("failure", [401, 500, "invalid", "network", "rejected"])
def test_http_errors(cloud, failure):
    url = instance_url(cloud)
    if isinstance(failure, int):
        responses.get(url, status=failure)
    elif failure == "invalid":
        responses.get(url, body="not-json")
    elif failure == "network":
        responses.get(url, body=requests.ConnectionError("offline"))
    else:
        responses.get(url, json={"success": False})
    with pytest.raises(ProviderError):
        cloud.list_instances()


def test_wait_ready_polling(cloud):
    pending = Instance("42", cloud.name, "pending")
    no_ssh = Instance("42", cloud.name, "running")
    ready = Instance("42", cloud.name, "running", ssh_host="host", ssh_port=22)
    cloud.get = Mock(side_effect=[pending, no_ssh, ready])
    with patch("cyoai.providers.base.time.sleep") as sleep:
        assert cloud.wait_ready(pending, poll=1) is ready
    assert sleep.call_count == 2


@pytest.mark.parametrize("status,match", [("pending", "Timed out"), ("failed", "failed")])
def test_wait_ready_errors(cloud, status, match):
    instance = Instance("42", cloud.name, status)
    cloud.get = Mock(return_value=instance)
    with pytest.raises(ProviderError, match=match):
        cloud.wait_ready(instance, timeout=0)


@responses.activate
def test_wait_ready_http(cloud):
    if cloud.name == "runpod":
        url = cloud.api_url + "/pods/42"
        pending = {"id": "42", "status": "STARTING", "ssh": {"direct": None, "proxy": None}}
        ready = {**pending, "status": "RUNNING", "ssh": {"direct": {
            "host": "host", "port": 2200, "username": "root", "command": "ssh root@host -p 2200"}}}
    else:
        url = cloud.api_url + "/instances/42/"
        pending = {"instances": {"id": 42, "actual_status": "loading"}}
        ready = {"instances": {"id": 42, "actual_status": "running", "ssh_host": "host", "ssh_port": 2200}}
    responses.get(url, json=pending)
    responses.get(url, json=ready)
    with patch("cyoai.providers.base.time.sleep"):
        instance = cloud.wait_ready(Instance("42", cloud.name, "pending"))
    assert instance.ssh_host == "host"
    assert instance.ssh_port == 2200
    assert len(responses.calls) == 2


@pytest.mark.parametrize("exception", [None, ValueError, KeyboardInterrupt])
def test_managed_cleanup(exception):
    provider = Mock()
    provider.provision.return_value = Instance("42", "test", "pending")
    provider.wait_ready.return_value = Instance("42", "test", "running")
    def use():
        with managed_instance(provider, gpu="test") as instance:
            assert instance.status == "running"
            if exception:
                raise exception()
    if exception:
        with pytest.raises(exception):
            use()
    else:
        use()
    provider.provision.assert_called_once_with(gpu="test")
    provider.destroy.assert_called_once_with("42")


def test_managed_readiness_failure():
    provider = Mock()
    provider.provision.return_value = Instance("42", "test", "pending")
    provider.wait_ready.side_effect = ProviderError("timeout")
    with pytest.raises(ProviderError):
        with managed_instance(provider):
            pytest.fail("must not yield")
    provider.destroy.assert_called_once_with("42")


def test_managed_provision_failure():
    provider = Mock()
    provider.provision.side_effect = ProviderError("cannot provision")
    with pytest.raises(ProviderError):
        with managed_instance(provider):
            pytest.fail("must not yield")
    provider.destroy.assert_not_called()


@pytest.mark.parametrize("signum", [signal.SIGINT, signal.SIGTERM])
def test_signals_deferred_during_delete(signum):
    provider = Mock()
    provider.provision.return_value = Instance("42", "test", "pending")
    provider.destroy.side_effect = lambda instance_id: os.kill(os.getpid(), signum)
    previous = signal.getsignal(signum)
    expected = KeyboardInterrupt if signum == signal.SIGINT else SystemExit
    with pytest.raises(expected):
        with managed_instance(provider):
            pass
    provider.destroy.assert_called_once_with("42")
    assert signal.getsignal(signum) == previous


@responses.activate
@pytest.mark.parametrize("delete_status", [204, 500])
def test_runpod_invalid_create_details_cleanup(key, delete_status):
    provider = RunPodProvider(api_key="test-key", ssh_key_path=key)
    responses.post(provider.api_url + "/pods", json={
        "id": "42", "ssh": {"direct": {"port": "invalid"}}})
    responses.delete(provider.api_url + "/pods/42", status=delete_status)
    with pytest.raises(ProviderError) as error:
        provider.provision()
    assert responses.calls[-1].request.method == "DELETE"
    if delete_status == 500:
        assert "cyoai destroy 42 --provider runpod" in str(error.value)
    else:
        assert "instance was deleted" in str(error.value)


@responses.activate
def test_vast_attach_and_delete_failure_guidance(key):
    provider = VastProvider(api_key="test-key", ssh_key_path=key)
    responses.post(provider.api_url + "/bundles/", json={"offers": [{"id": 1, "dph_total": 1}]})
    responses.put(provider.api_url + "/asks/1/", json={"new_contract": 42})
    responses.post(provider.api_url + "/instances/42/ssh/", status=500)
    responses.delete(provider.api_url + "/instances/42/", status=401)
    with pytest.raises(ProviderError, match="cyoai destroy 42 --provider vast"):
        provider.provision()


@responses.activate
@pytest.mark.parametrize("detail", ["Authorization: Basic encoded-secret", "Bearer encoded-secret",
                                  "-----BEGIN OPENSSH PRIVATE KEY-----\\nprivate-material"])
def test_runpod_error_drops_sensitive_details(detail):
    provider = RunPodProvider(api_key="test-key")
    responses.get(provider.api_url + "/pods", status=500, json={"detail": detail})
    with pytest.raises(ProviderError) as error:
        provider.list_instances()
    assert detail not in str(error.value)
    assert "HTTP 500" in str(error.value)


def test_managed_sigterm():
    previous = signal.getsignal(signal.SIGTERM)
    provider = Mock()
    provider.provision.return_value = Instance("42", "test", "pending")
    with pytest.raises(SystemExit) as error:
        with managed_instance(provider):
            os.kill(os.getpid(), signal.SIGTERM)
    assert error.value.code == 143
    provider.destroy.assert_called_once_with("42")
    assert signal.getsignal(signal.SIGTERM) == previous


@pytest.fixture
def remote():
    return Instance("42", "test", "running", ssh_host="host", ssh_port=2222)


def test_ssh_exec(key, remote):
    process = Mock()
    process.stdout = io.StringIO("hello\nworld\n")
    process.wait.return_value = 0
    process.poll.return_value = 0
    with patch("cyoai.providers.ssh.subprocess.Popen", return_value=process) as popen:
        assert list(ssh.exec_stream(remote, "echo hello", key)) == ["hello\n", "world\n"]
    args = popen.call_args.args[0]
    assert args == ["ssh", "-i", key, "-o", "StrictHostKeyChecking=accept-new", "-o",
                    f'UserKnownHostsFile="{ssh.known_hosts_path(remote)}"', "-o", "BatchMode=yes", "-o",
                    "ConnectTimeout=30", "-o", "IdentitiesOnly=yes", "-p", "2222", "root@host", "echo hello"]


@pytest.mark.parametrize("rsync", [True, False])
@pytest.mark.parametrize("remote_path", ["/workspace/", "/workspace with spaces/"])
def test_ssh_sync_commands(key, remote, rsync, remote_path):
    with patch("cyoai.providers.ssh.shutil.which", return_value="/bin/rsync" if rsync else None), \
            patch("cyoai.providers.ssh.stream_process", return_value=iter(())) as run:
        ssh.sync_up(remote, "workspace/", remote_path, key)
        ssh.sync_down(remote, remote_path + "results.tsv", "results.tsv", key)
    upload, download = [call.args[0] for call in run.call_args_list]
    assert upload[-2:] == ["workspace/", f"root@host:{remote_path}"]
    assert download[-2:] == [f"root@host:{remote_path}results.tsv", "results.tsv"]
    assert upload[0] == ("rsync" if rsync else "scp")
    if rsync:
        assert "--protect-args" in upload
        assert "--protect-args" in download
        assert "-p 2222" in upload[upload.index("-e") + 1]
    else:
        assert upload[upload.index("-P") + 1] == "2222"
    for command in (upload, download):
        transport = shlex.split(command[command.index("-e") + 1]) if rsync else command
        assert "StrictHostKeyChecking=accept-new" in transport
        assert "IdentitiesOnly=yes" in transport
        assert f'UserKnownHostsFile="{ssh.known_hosts_path(remote)}"' in transport
    assert all(call.args[1] is remote for call in run.call_args_list)


def test_known_hosts_isolation_and_permissions(key, remote):
    expected = Path.home() / ".cache" / "cyoai" / "known_hosts" / "test-42"
    assert ssh.known_hosts_path(remote) == expected
    ssh.options(remote, key)
    assert expected.parent.stat().st_mode & 0o777 == 0o700
    expected.write_text("trusted host key\n")
    expected.parent.chmod(0o755)
    ssh.options(remote, key)
    assert expected.parent.stat().st_mode & 0o777 == 0o700
    assert expected.read_text() == "trusted host key\n"
    reused_ip = Instance("43", "test", "running", ssh_host="host", ssh_port=2222)
    other_provider = Instance("42", "other", "running", ssh_host="host", ssh_port=2222)
    assert len({ssh.known_hosts_path(item) for item in (remote, reused_ip, other_provider)}) == 3
    assert not ssh.known_hosts_path(reused_ip).exists()
    unsafe = Instance("../../escape", "../provider", "running")
    assert ssh.known_hosts_path(unsafe).parent == expected.parent


@pytest.mark.parametrize("failure", [None, ValueError, KeyboardInterrupt])
def test_managed_known_hosts_cleanup(key, remote, failure):
    provider = Mock()
    provider.provision.return_value = remote
    provider.wait_ready.return_value = remote
    ssh.options(remote, key)
    hosts = ssh.known_hosts_path(remote)
    hosts.write_text("trusted host key")

    def use():
        with managed_instance(provider):
            assert hosts.exists()
            if failure:
                raise failure()

    if failure:
        with pytest.raises(failure):
            use()
    else:
        use()
    assert not hosts.exists()
    ssh.cleanup_known_hosts(remote)


def test_known_hosts_kept_on_destroy_failure(key, remote):
    provider = Mock()
    provider.provision.return_value = remote
    provider.wait_ready.return_value = remote
    provider.destroy.side_effect = ProviderError("destroy failed")
    ssh.options(remote, key)
    hosts = ssh.known_hosts_path(remote)
    hosts.write_text("trusted host key")
    with pytest.raises(ProviderError, match="destroy failed"):
        with managed_instance(provider):
            pass
    assert hosts.exists()


def test_known_hosts_directory_failure(key, remote):
    with patch("pathlib.Path.mkdir", side_effect=PermissionError):
        with pytest.raises(ProviderError, match="known-hosts directory"):
            ssh.options(remote, key)


def test_ssh_key_selection(key, monkeypatch, tmp_path):
    monkeypatch.setenv("CYOAI_SSH_KEY", key)
    assert ssh.key_path() == __import__("pathlib").Path(key)
    assert str(ssh.key_path("/explicit")) == "/explicit"
    monkeypatch.delenv("CYOAI_SSH_KEY")
    monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)
    with pytest.raises(ProviderError, match="No SSH key"):
        ssh.key_path()
    directory = tmp_path / ".ssh"
    directory.mkdir()
    (directory / "id_rsa").touch()
    assert ssh.key_path().name == "id_rsa"
    (directory / "id_ed25519").touch()
    assert ssh.key_path().name == "id_ed25519"
    with pytest.raises(ProviderError, match="public key"):
        ssh.public_key()


def test_missing_ssh_endpoint(key):
    with pytest.raises(ProviderError, match="no SSH endpoint"):
        list(ssh.exec_stream(Instance("42", "test", "pending"), "true", key))


def test_stream_process_unavailable():
    with patch("cyoai.providers.ssh.subprocess.Popen", side_effect=OSError("not installed")):
        with pytest.raises(ProviderError, match="Cannot execute"):
            list(ssh.stream_process(["missing"]))


def test_stream_process_early_close():
    process = Mock()
    process.stdout = io.StringIO("one\ntwo\n")
    process.poll.return_value = None
    with patch("cyoai.providers.ssh.subprocess.Popen", return_value=process):
        stream = ssh.stream_process(["command"])
        assert next(stream) == "one\n"
        stream.close()
    process.terminate.assert_called_once()
    assert process.stdout.closed


def test_transfer_error(key, remote):
    with patch("cyoai.providers.ssh.stream_process", side_effect=ProviderError("missing")):
        with pytest.raises(ProviderError, match="transfer failed"):
            ssh.sync_up(remote, "local", "remote", key)


def test_local_exec():
    provider = LocalProvider()
    instance = provider.provision()
    assert instance.provider == "local"
    assert provider.wait_ready(instance) is instance
    assert provider.get("local").status == "running"
    assert len(provider.list_instances()) == 1
    assert provider.destroy("anything") is None
    assert list(provider.exec_stream(instance, "printf 'one\\ntwo\\n'; printf 'err\\n' >&2")) == ["one\n", "two\n", "err\n"]
    assert list(provider.exec_stream(instance, "true")) == []
    with pytest.raises(ProviderError, match="status 7"):
        list(provider.exec_stream(instance, "exit 7"))


def test_local_sync(tmp_path):
    provider = LocalProvider()
    instance = provider.provision()
    source = tmp_path / "source"
    source.mkdir()
    (source / "results.tsv").write_text("result\n")
    remote = tmp_path / "remote"
    provider.sync_up(instance, source, remote)
    assert (remote / "results.tsv").read_text() == "result\n"
    target = tmp_path / "nested" / "download.tsv"
    provider.sync_down(instance, remote / "results.tsv", target)
    assert target.read_text() == "result\n"
    provider.sync_up(instance, source, source)
    empty = tmp_path / "empty"
    empty.mkdir()
    provider.sync_up(instance, empty, tmp_path / "empty-copy")
    assert list((tmp_path / "empty-copy").iterdir()) == []
    with pytest.raises(FileNotFoundError):
        provider.sync_up(instance, tmp_path / "missing", tmp_path / "unused")
