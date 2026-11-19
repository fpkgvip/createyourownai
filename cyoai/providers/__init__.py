from .base import ComputeProvider, Instance, MissingCredentialsError, ProviderError, managed_instance
from .local import LocalProvider
from .runpod import RunPodProvider
from .vast import VastProvider

PROVIDERS = {"local": LocalProvider, "runpod": RunPodProvider, "vast": VastProvider}


def get_provider(name: str, **kwargs) -> ComputeProvider:
    try:
        provider = PROVIDERS[name]
    except KeyError:
        raise ProviderError(f"Unknown provider {name!r}; choose from {', '.join(PROVIDERS)}") from None
    return provider(**kwargs)


__all__ = ["ComputeProvider", "Instance", "MissingCredentialsError", "ProviderError",
           "managed_instance", "LocalProvider", "RunPodProvider", "VastProvider",
           "PROVIDERS", "get_provider"]
