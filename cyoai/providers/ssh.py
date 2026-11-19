import os
from pathlib import Path
import shlex
import shutil
import subprocess
from urllib.parse import quote

from .errors import ProviderError


def key_path(path=None):
    selected = path or os.environ.get("CYOAI_SSH_KEY")
    if selected:
        return Path(selected).expanduser()
    for name in ("id_ed25519", "id_rsa"):
        candidate = Path.home() / ".ssh" / name
        if candidate.is_file():
            return candidate
    raise ProviderError("No SSH key found; set CYOAI_SSH_KEY or pass ssh_key_path")


def public_key(path=None):
    try:
        return Path(str(key_path(path)) + ".pub").read_text().strip()
    except OSError as exc:
        raise ProviderError("Cannot read SSH public key (.pub)") from exc


def known_hosts_path(instance):
    filename = f"{quote(instance.provider, safe='')}-{quote(str(instance.id), safe='')}"
    return Path.home() / ".cache" / "cyoai" / "known_hosts" / filename


def cleanup_known_hosts(instance):
    try:
        known_hosts_path(instance).unlink(missing_ok=True)
    except OSError:
        pass


def options(instance, path=None):
    hosts = known_hosts_path(instance)
    try:
        hosts.parent.mkdir(parents=True, mode=0o700, exist_ok=True)
        hosts.parent.chmod(0o700)
    except OSError as exc:
        raise ProviderError("Cannot create SSH known-hosts directory") from exc
    return ["-i", str(key_path(path)), "-o", "StrictHostKeyChecking=accept-new",
            "-o", f'UserKnownHostsFile="{hosts}"', "-o", "BatchMode=yes",
            "-o", "ConnectTimeout=30"]


def target(instance):
    if not instance.ssh_host or not instance.ssh_port:
        raise ProviderError(f"Instance {instance.id} has no SSH endpoint")
    return f"{instance.ssh_user}@{instance.ssh_host}"


def stream_process(args):
    try:
        process = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                   text=True, bufsize=1)
    except OSError as exc:
        raise ProviderError(f"Cannot execute {args[0]}: {exc}") from exc
    try:
        yield from process.stdout
        code = process.wait()
        if code:
            raise ProviderError(f"{args[0]} exited with status {code}")
    finally:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        process.stdout.close()


def exec_stream(instance, command, ssh_key_path=None):
    host = target(instance)
    return stream_process(["ssh", *options(instance, ssh_key_path), "-p", str(instance.ssh_port),
                           host, command])


def _sync(instance, source, destination, path, upload):
    host = target(instance)
    opts = options(instance, path)
    if shutil.which("rsync"):
        transport = shlex.join(["ssh", *opts, "-p", str(instance.ssh_port)])
        remote = str(destination if upload else source)
        src = str(source) if upload else f"{host}:{remote}"
        dst = f"{host}:{remote}" if upload else str(destination)
        args = ["rsync", "-az", "--protect-args", "-e", transport, "--", src, dst]
    else:
        src = str(source) if upload else f"{host}:{source}"
        dst = f"{host}:{destination}" if upload else str(destination)
        args = ["scp", *opts, "-P", str(instance.ssh_port), "-r", "--", src, dst]
    try:
        subprocess.run(args, check=True)
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ProviderError(f"{args[0]} transfer failed") from exc


def sync_up(instance, local_path, remote_path, ssh_key_path=None):
    _sync(instance, local_path, remote_path, ssh_key_path, True)


def sync_down(instance, remote_path, local_path, ssh_key_path=None):
    _sync(instance, remote_path, local_path, ssh_key_path, False)
