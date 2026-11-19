from pathlib import Path
import shutil

from .base import ComputeProvider, Instance
from .ssh import stream_process


class LocalProvider(ComputeProvider):
    name = "local"

    def provision(self, gpu=None, image=None, disk_gb=50, name="cyoai"):
        return Instance("local", self.name, "running", gpu=gpu)

    def wait_ready(self, instance, timeout=900, poll=10):
        return instance

    def get(self, instance_id):
        return self.provision()

    def list_instances(self):
        return [self.provision()]

    def destroy(self, instance_id):
        pass

    def exec_stream(self, instance, command):
        return stream_process(["/bin/sh", "-c", command])

    @staticmethod
    def _copy(source, destination):
        source, destination = Path(source).expanduser(), Path(destination).expanduser()
        if source.resolve() == destination.resolve():
            return
        if source.is_dir():
            shutil.copytree(source, destination, dirs_exist_ok=True)
        else:
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)

    def sync_up(self, instance, local_path, remote_path):
        self._copy(local_path, remote_path)

    def sync_down(self, instance, remote_path, local_path):
        self._copy(remote_path, local_path)
