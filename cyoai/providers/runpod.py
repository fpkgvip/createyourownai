from .base import Instance, ProviderError
from .http import HTTPProvider
from .ssh import public_key


class RunPodProvider(HTTPProvider):
    name = "runpod"
    api_url = "https://rest.runpod.io/v1"
    credential_env = "RUNPOD_API_KEY"
    default_image = "runpod/pytorch:2.8.0-py3.11-cuda12.8.1-cudnn-devel-ubuntu22.04"

    def _instance(self, data):
        port = (data.get("portMappings") or {}).get("22")
        return Instance(
            id=str(data["id"]), provider=self.name,
            status=str(data.get("desiredStatus") or data.get("status") or "pending").lower(),
            gpu=data.get("gpu", {}).get("displayName") if isinstance(data.get("gpu"), dict)
            else data.get("gpuTypeId"),
            ssh_host=data.get("publicIp"), ssh_port=int(port) if port else None,
            cost_per_hr=data.get("costPerHr"), raw=data,
        )

    def provision(self, gpu=None, image=None, disk_gb=50, name="cyoai"):
        data = self._request("POST", "/pods", json={
            "name": name, "imageName": image or self.default_image,
            "gpuTypeIds": [gpu or "NVIDIA GeForce RTX 4090"], "gpuCount": 1,
            "containerDiskInGb": disk_gb, "ports": ["22/tcp"], "supportPublicIp": True,
            "env": {"PUBLIC_KEY": public_key(self.ssh_key_path)},
        })
        if not isinstance(data, dict) or not data.get("id"):
            raise ProviderError("runpod provision response has no instance id")
        return self._instance(data)

    def get(self, instance_id):
        return self._instance(self._request("GET", f"/pods/{instance_id}"))

    def list_instances(self):
        data = self._request("GET", "/pods")
        return [self._instance(item) for item in data
                if item.get("name") == "cyoai" or (item.get("name") or "").startswith("cyoai-")]

    def destroy(self, instance_id):
        self._request("DELETE", f"/pods/{instance_id}", allow_missing=True)
