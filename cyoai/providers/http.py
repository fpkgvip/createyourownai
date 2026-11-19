import os

import requests

from .base import ComputeProvider, MissingCredentialsError, ProviderError


class HTTPProvider(ComputeProvider):
    api_url: str
    credential_env: str

    def __init__(self, api_key=None, ssh_key_path=None, session=None):
        super().__init__(ssh_key_path)
        self.api_key = api_key or os.environ.get(self.credential_env)
        if not self.api_key:
            raise MissingCredentialsError(f"Set {self.credential_env} or pass api_key")
        self.session = session if session is not None else requests.Session()

    def _request(self, method, path, *, allow_missing=False, **kwargs):
        try:
            response = self.session.request(
                method, self.api_url + path,
                headers={"Authorization": f"Bearer {self.api_key}"}, timeout=60, **kwargs
            )
            if allow_missing and response.status_code == 404:
                return None
            response.raise_for_status()
            if response.status_code == 204 or not response.content:
                return None
            data = response.json()
        except (requests.RequestException, ValueError) as exc:
            # Do not include API response bodies or headers, which may contain credentials.
            status = getattr(locals().get("response"), "status_code", None)
            raise ProviderError(f"{self.name} {method} {path} failed"
                                + (f" (HTTP {status})" if status else "")) from exc
        if isinstance(data, dict) and data.get("success") is False:
            raise ProviderError(f"{self.name} {method} {path} was rejected")
        return data
