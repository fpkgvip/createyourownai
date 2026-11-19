# Compute providers

`get_provider("local" | "runpod" | "vast", **kwargs)` returns a `ComputeProvider`.
Cloud constructors accept `api_key=None`, `ssh_key_path=None`, and an injectable
`requests.Session` (`session=None`). Keys default to `RUNPOD_API_KEY` and
`VAST_API_KEY`; missing credentials raise `MissingCredentialsError`.
`LocalProvider()` requires no credentials.

SSH uses the explicit key, `CYOAI_SSH_KEY`, `~/.ssh/id_ed25519`, then `id_rsa`.
The matching `.pub` file must exist before provisioning. Install OpenSSH and,
preferably, rsync; otherwise transfers use recursive scp. Host-key verification
is disabled for disposable hosts. Custom images must run sshd on port 22 and
support the provider's SSH-key setup. CUDA/PyTorch images are the defaults.

```python
from cyoai.providers import get_provider, managed_instance

provider = get_provider("runpod")
with managed_instance(provider, gpu="NVIDIA GeForce RTX 4090") as instance:
    provider.sync_up(instance, "workspace/", "/workspace/")
    for line in provider.exec_stream(instance, "cd /workspace && uv sync && uv run prepare.py"):
        print(line, end="")
    provider.sync_down(instance, "/workspace/results.tsv", "results.tsv")
```

`provision(gpu=None, image=None, disk_gb=50, name="cyoai")` returns an `Instance`.
`wait_ready(instance, timeout=900, poll=10)` polls for running + SSH endpoint.
`get(id)`, `list_instances()`, and idempotent `destroy(id)` support management.
Lists include names/labels `cyoai` and `cyoai-*`; use that prefix for custom names.
`exec_stream` yields newline-preserving stdout/stderr and raises `ProviderError`
on nonzero exit. Sync methods copy directories recursively; use trailing slashes
with rsync to copy directory contents. Local sync merges directory contents.

`managed_instance` tears down after success, errors, readiness failure,
KeyboardInterrupt, and SIGTERM (main thread only). It temporarily replaces and
then restores the SIGTERM handler. SIGKILL, process/machine crashes, or a lost
provisioning response cannot guarantee cleanup; inspect your cloud dashboard.
Transfer desired checkpoints/results before leaving the context; teardown
deletes the instance and its ephemeral storage. Local destroy is a no-op.

The Runpod adapter intentionally implements the requested REST v1 contract at
`rest.runpod.io/v1`. Current Runpod documentation says v1 is no longer available;
live use requires a v2 migration. Tests mock HTTP, not live cloud availability.
