from .base import Instance, ProviderError
from .http import HTTPProvider
from .ssh import public_key


class RunPodProvider(HTTPProvider):
    name = "runpod"
    api_url = "https://api.runpod.io/v2"
    credential_env = "RUNPOD_API_KEY"
    default_image = "runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404"

    def _error_detail(self, response):
        try:
            problem = response.json()
        except ValueError:
            return ""
        if not isinstance(problem, dict):
            return ""
        return ": ".join(problem[key] for key in ("title", "detail")
                         if isinstance(problem.get(key), str)).replace(self.api_key, "[REDACTED]")

    def _instance(self, data):
        endpoint = (data.get("ssh") or {}).get("direct") or {}
        port = endpoint.get("port")
        return Instance(
            id=str(data["id"]), provider=self.name,
            status=str(data.get("status") or "pending").lower(),
            gpu=(data.get("gpu") or {}).get("id"),
            ssh_host=endpoint.get("host"), ssh_port=int(port) if port else None,
            ssh_user=endpoint.get("username") or "root",
            cost_per_hr=data.get("cost"), raw=data,
        )

    def provision(self, gpu=None, image=None, disk_gb=50, name="cyoai"):
        data = self._request("POST", "/pods", json={
            "name": name, "image": image or self.default_image,
            "gpu": {"id": gpu or "NVIDIA GeForce RTX 4090", "count": 1},
            "disk": disk_gb, "ports": ["22/tcp"], "startSsh": True,
            "env": {"PUBLIC_KEY": public_key(self.ssh_key_path)},
        })
        if not isinstance(data, dict) or not data.get("id"):
            raise ProviderError("runpod provision response has no instance id")
        return self._instance(data)

    def get(self, instance_id):
        return self._instance(self._request("GET", f"/pods/{instance_id}"))

    def list_instances(self):
        instances = []
        cursor = None
        seen = set()
        while True:
            data = self._request("GET", "/pods", params={"cursor": cursor} if cursor else {})
            instances.extend(self._instance(item) for item in data["pods"]
                             if item.get("name") == "cyoai"
                             or (item.get("name") or "").startswith("cyoai-"))
            pagination = data["pagination"]
            if not pagination["hasNextPage"]:
                return instances
            cursor = pagination["nextCursor"]
            if not cursor or cursor in seen:
                raise ProviderError("runpod list response has an invalid pagination cursor")
            seen.add(cursor)

    def destroy(self, instance_id):
        self._request("DELETE", f"/pods/{instance_id}", allow_missing=True)
