from abc import ABC, abstractmethod
from contextlib import contextmanager
from dataclasses import dataclass, field
import signal
import threading
import time
from typing import Iterator

from . import ssh
from .errors import MissingCredentialsError, ProviderError


@dataclass
class Instance:
    id: str
    provider: str
    status: str
    gpu: str | None = None
    ssh_host: str | None = None
    ssh_port: int | None = None
    ssh_user: str = "root"
    cost_per_hr: float | None = None
    raw: dict = field(default_factory=dict)


class ComputeProvider(ABC):
    name: str

    def __init__(self, ssh_key_path=None):
        self.ssh_key_path = ssh_key_path

    @abstractmethod
    def provision(self, gpu=None, image=None, disk_gb=50, name="cyoai") -> Instance:
        raise NotImplementedError

    def wait_ready(self, instance, timeout=900, poll=10) -> Instance:
        deadline = time.monotonic() + timeout
        while True:
            current = self.get(instance.id)
            if current.status.lower() == "running" and current.ssh_host and current.ssh_port:
                return current
            if current.status.lower() in {"error", "failed", "terminated", "exited", "deleted", "offline"}:
                raise ProviderError(f"{self.name} instance {instance.id}: {current.status}")
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise ProviderError(f"Timed out waiting for {self.name} instance {instance.id}")
            time.sleep(min(poll, remaining))

    @abstractmethod
    def get(self, instance_id) -> Instance:
        raise NotImplementedError

    @abstractmethod
    def list_instances(self) -> list[Instance]:
        raise NotImplementedError

    @abstractmethod
    def destroy(self, instance_id) -> None:
        raise NotImplementedError

    def exec_stream(self, instance, command) -> Iterator[str]:
        return ssh.exec_stream(instance, command, self.ssh_key_path)

    def sync_up(self, instance, local_path, remote_path):
        ssh.sync_up(instance, local_path, remote_path, self.ssh_key_path)

    def sync_down(self, instance, remote_path, local_path):
        ssh.sync_down(instance, remote_path, local_path, self.ssh_key_path)


@contextmanager
def managed_instance(provider, **provision_kwargs):
    instance = None
    previous = None
    install_signal = threading.current_thread() is threading.main_thread()

    def terminate(signum, frame):
        raise SystemExit(128 + signum)

    if install_signal:
        previous = signal.signal(signal.SIGTERM, terminate)
    try:
        instance = provider.provision(**provision_kwargs)
        ready = provider.wait_ready(instance)
        yield ready
    finally:
        try:
            if instance is not None:
                provider.destroy(instance.id)
                ssh.cleanup_known_hosts(instance)
        finally:
            if install_signal:
                signal.signal(signal.SIGTERM, previous)
