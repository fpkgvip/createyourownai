from .base import Instance, ProviderError
from .http import HTTPProvider
from .ssh import public_key


class VastProvider(HTTPProvider):
    name = "vast"
    api_url = "https://console.vast.ai/api/v0"
    credential_env = "VAST_API_KEY"
    default_image = "pytorch/pytorch:2.8.0-cuda12.8-cudnn9-devel"

    def _instance(self, data):
        port = data.get("ssh_port")
        return Instance(
            id=str(data["id"]), provider=self.name,
            status=str(data.get("actual_status") or "pending").lower(),
            gpu=data.get("gpu_name"), ssh_host=data.get("ssh_host"),
            ssh_port=int(port) if port else None,
            cost_per_hr=data.get("dph_total"), raw=data,
        )

    def provision(self, gpu=None, image=None, disk_gb=50, name="cyoai"):
        key = public_key(self.ssh_key_path)
        query = {"num_gpus": {"eq": 1}, "gpu_name": {"eq": gpu or "RTX_4090"},
                 "rentable": {"eq": True}, "verified": {"eq": True},
                 "disk_space": {"gte": disk_gb}, "order": [["dph_total", "asc"]],
                 "type": "on-demand"}
        data = self._request("POST", "/bundles/", json=query)
        offers = data.get("offers", [])
        if not offers:
            raise ProviderError("No matching Vast GPU offers available")
        offer = min(offers, key=lambda item: float(item["dph_total"]))
        result = self._request("PUT", f"/asks/{offer['id']}/", json={
            "client_id": "me", "image": image or self.default_image, "disk": disk_gb,
            "runtype": "ssh", "label": name,
        })
        instance_id = result.get("new_contract")
        if instance_id is None:
            raise ProviderError("vast provision response has no instance id")
        try:
            self._request("POST", f"/instances/{instance_id}/ssh/", json={"ssh_key": key})
        except BaseException:
            try:
                self.destroy(str(instance_id))
            except Exception as exc:
                raise ProviderError(
                    f"{exc}. Instance {instance_id} may still be running; "
                    f"clean up with: cyoai destroy {instance_id} --provider vast"
                ) from None
            raise
        return Instance(str(instance_id), self.name, "pending", gpu=offer.get("gpu_name"),
                        cost_per_hr=offer.get("dph_total"), raw=result)

    def get(self, instance_id):
        data = self._request("GET", f"/instances/{instance_id}/")
        return self._instance(data.get("instances", data))

    def list_instances(self):
        data = self._request("GET", "/instances/")
        return [self._instance(item) for item in data.get("instances", [])
                if item.get("label") == "cyoai" or (item.get("label") or "").startswith("cyoai-")]

    def destroy(self, instance_id):
        self._request("DELETE", f"/instances/{instance_id}/", allow_missing=True)
