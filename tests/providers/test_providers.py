import io
import json
import os
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
    responses.post(provider.api_url + "/pods", json={"id": "p1", "desiredStatus": "RUNNING"})
    kwargs = {"gpu": "A100", "image": "custom-image", "disk_gb": 100, "name": "cyoai-test"} if custom else {}
    instance = provider.provision(**kwargs)
    body = json.loads(responses.calls[0].request.body)
    assert body == {
        "name": kwargs.get("name", "cyoai"), "imageName": kwargs.get("image", provider.default_image),
        "gpuTypeIds": [kwargs.get("gpu", "NVIDIA GeForce RTX 4090")], "gpuCount": 1,
        "containerDiskInGb": kwargs.get("disk_gb", 50), "ports": ["22/tcp"],
        "supportPublicIp": True, "env": {"PUBLIC_KEY": "ssh-ed25519 test-public-key"},
    }
    assert responses.calls[0].request.headers["Authorization"] == "Bearer test-key"
    assert instance.id == "p1"
    assert instance.status == "running"


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
    data = {"id": "p1", "name": "cyoai", "desiredStatus": "RUNNING", "publicIp": "1.2.3.4",
            "portMappings": {"22": 2201}, "costPerHr": 0.5, "gpu": {"displayName": "4090"}}
    responses.get(provider.api_url + "/pods/p1", json=data)
    responses.get(provider.api_url + "/pods", json=[data, {**data, "name": "unrelated"}, {**data, "name": None},
                                                       {**data, "id": "p2", "name": "cyoai-test"}])
    instance = provider.get("p1")
    assert (instance.ssh_host, instance.ssh_port, instance.gpu, instance.cost_per_hr) == ("1.2.3.4", 2201, "4090", 0.5)
    assert instance.raw == data
    assert [i.id for i in provider.list_instances()] == ["p1", "p2"]


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
    responses.get(instance_url(cloud), json=[] if cloud.name == "runpod" else {"instances": []})
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
        pending = {"id": "42", "desiredStatus": "RUNNING"}
        ready = {**pending, "publicIp": "host", "portMappings": {"22": "2200"}}
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
    assert args == ["ssh", "-i", key, "-o", "StrictHostKeyChecking=no", "-o",
                    "UserKnownHostsFile=/dev/null", "-o", "BatchMode=yes", "-o",
                    "ConnectTimeout=30", "-p", "2222", "root@host", "echo hello"]


@pytest.mark.parametrize("rsync", [True, False])
def test_ssh_sync_commands(key, remote, rsync):
    with patch("cyoai.providers.ssh.shutil.which", return_value="/bin/rsync" if rsync else None), \
            patch("cyoai.providers.ssh.subprocess.run") as run:
        ssh.sync_up(remote, "workspace/", "/workspace/", key)
        ssh.sync_down(remote, "/workspace/results.tsv", "results.tsv", key)
    upload, download = [call.args[0] for call in run.call_args_list]
    assert upload[-2:] == ["workspace/", "root@host:/workspace/"]
    assert download[-2:] == ["root@host:/workspace/results.tsv", "results.tsv"]
    assert upload[0] == ("rsync" if rsync else "scp")
    if rsync:
        assert "-p 2222" in upload[upload.index("-e") + 1]
    else:
        assert upload[upload.index("-P") + 1] == "2222"
    assert all(call.kwargs["check"] for call in run.call_args_list)


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
    with patch("cyoai.providers.ssh.subprocess.run", side_effect=OSError("missing")):
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
