"""Small, scriptable command line interface."""
import argparse
from dataclasses import asdict
import getpass
import importlib
import json
import os
from pathlib import Path
import shlex
import sys
import subprocess
import tempfile
import warnings

from cyoai.config import Config, load_config, masked_config, save_config
from cyoai.agent.llm import FakeLLM, make_llm
from cyoai.agent.loop import ResearchLoop
from cyoai.agent.runners import DRY_RUN_ARTIFACT, FakeRunner, LocalRunner, RemoteRunner, scrub_env
from cyoai.agent.workspace import create_workspace


def get_provider(name, api_key):
    try:
        from cyoai.providers import get_provider as factory
    except ImportError as exc:
        raise RuntimeError("Compute providers are not installed") from exc
    return factory(name, api_key=api_key)


def provider_for(name, config):
    key = config.runpod_api_key if name == "runpod" else config.vast_api_key
    if not key:
        raise ValueError(f"No {name} API key configured")
    return get_provider(name, key)


def managed_instance(provider, **kwargs):
    from cyoai.providers import managed_instance as manage
    return manage(provider, **kwargs)


def init_command(args):
    config = load_config()
    if not args.non_interactive:
        config.llm_provider = input(f"LLM provider [{config.llm_provider}]: ").strip() or config.llm_provider
        default = "gpt-4o" if config.llm_provider == "openai" else "claude-sonnet-4-5"
        config.llm_model = input(f"Model [{default}]: ").strip() or default
        config.llm_api_key = getpass.getpass("LLM API key (blank to keep): ") or config.llm_api_key
        config.llm_base_url = input("OpenAI-compatible base URL (optional): ").strip() or None
        config.runpod_api_key = getpass.getpass("RunPod API key (optional): ") or config.runpod_api_key
        config.vast_api_key = getpass.getpass("Vast API key (optional): ") or config.vast_api_key
    if args.llm_provider and not args.llm_model and args.llm_provider != config.llm_provider:
        config.llm_model = "gpt-4o" if args.llm_provider == "openai" else "claude-sonnet-4-5"
    if args.llm_provider and not args.llm_api_key:
        key_env = "OPENAI_API_KEY" if args.llm_provider == "openai" else "ANTHROPIC_API_KEY"
        if key_env in os.environ:
            config.llm_api_key = os.environ[key_env]
    for key in asdict(config):
        value = getattr(args, key, None)
        if value is not None:
            setattr(config, key, value)
    if config.llm_provider not in {"anthropic", "openai", "fake"}:
        raise ValueError("LLM provider must be anthropic, openai or fake")
    save_config(config)
    print(json.dumps(masked_config(config), indent=2))
    return 0


def train_command(args):
    if args.budget_mins <= 0 or args.max_iters < 0:
        raise ValueError("Budget must be positive and max-iters nonnegative")
    config = load_config()
    llm = FakeLLM() if args.dry_run else make_llm(config)
    extra_env = {}
    if args.smoke:
        extra_env["CYOAI_SMOKE"] = "1"
    if args.device:
        extra_env["CYOAI_DEVICE"] = args.device
    # Dry runs do not depend on the optional GPU training package.
    if args.dry_run:
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory)
            (source / "train.py").write_text("print('dry run')\n")
            (source / "program.md").write_text("Synthetic research run.\n")
            workspace = create_workspace(args.goal, Path(args.workdir), source)
    else:
        workspace = create_workspace(args.goal, Path(args.workdir))
    print(f"Workspace: {workspace}")
    if args.dry_run:
        best = ResearchLoop(workspace, llm, FakeRunner(workspace), args.max_iters).run()
    elif args.provider == "local":
        runner = LocalRunner(workspace, args.budget_mins * 60, extra_env=extra_env)
        smoke = extra_env.get("CYOAI_SMOKE", os.environ.get("CYOAI_SMOKE")) == "1"
        cpu = extra_env.get("CYOAI_DEVICE", os.environ.get("CYOAI_DEVICE")) == "cpu"
        run = "uv run --extra cpu" if cpu else "uv run"
        bootstrap = LocalRunner(workspace, cmd=run + " prepare.py" + (" --num-shards 1" if smoke else ""), extra_env=extra_env)
        code, _ = bootstrap.run()
        if code:
            raise RuntimeError("Dataset preparation failed")
        best = ResearchLoop(workspace, llm, runner, args.max_iters).run()
    else:
        provider = provider_for(args.provider, config)
        with managed_instance(provider, gpu=args.gpu) as instance:
            remote_dir = "/workspace/cyoai"
            for line in provider.exec_stream(instance, f"mkdir -p {shlex.quote(remote_dir)}"):
                print(line.rstrip())
            for source in workspace.iterdir():
                if source.is_file() and source.suffix in {".py", ".md", ".toml"}:
                    provider.sync_up(instance, source, remote_dir + "/" + source.name)
            smoke = extra_env.get("CYOAI_SMOKE", os.environ.get("CYOAI_SMOKE")) == "1"
            cpu = extra_env.get("CYOAI_DEVICE", os.environ.get("CYOAI_DEVICE")) == "cpu"
            run = "uv run --extra cpu" if cpu else "uv run"
            sync = "uv sync --extra cpu" if cpu else "uv sync"
            command = (f"cd {shlex.quote(remote_dir)} && "
                       "(command -v uv >/dev/null 2>&1 || python -m pip install uv) && "
                       f"{sync} && CYOAI_SMOKE={'1' if smoke else '0'} {run} prepare.py")
            if smoke:
                command += " --num-shards 1"
            for line in provider.exec_stream(instance, command):
                print(line.rstrip())
            runner = RemoteRunner(provider, instance, remote_dir, workspace,
                                  args.budget_mins * 60, extra_env=extra_env)
            best = ResearchLoop(workspace, llm, runner, args.max_iters).run()
    if best is None:
        print("No successful training run", file=sys.stderr)
        return 1
    if args.dry_run:
        print(f"Dry run complete. Simulated results: {workspace / 'results.tsv'} (no trained model)")
    else:
        print(f"Best val_bpb: {best.val_bpb:.6f}; artifacts: {workspace / 'best'}")
    return 0


def is_dry_run_model(path):
    with path.open("rb") as model:
        return model.read(len(DRY_RUN_ARTIFACT) + 1).replace(b"\r\n", b"\n") == DRY_RUN_ARTIFACT


def chat_command(args):
    if args.max_tokens < 0 or args.temperature < 0 or args.top_k < 0:
        raise ValueError("max-tokens, temperature, and top-k must be non-negative")
    if args.path is None:
        for model in sorted(Path("runs").glob("*/best/model.pt"), reverse=True):
            if (model.is_file() and (model.parent / "tokenizer.pkl").is_file()
                    and not is_dry_run_model(model)):
                path = model.parent
                break
        else:
            raise ValueError("No trained checkpoint found under runs/. Pass a checkpoint path or train without --dry-run.")
    else:
        path = Path(args.path).expanduser()
        if path.is_dir() and not (path / "model.pt").is_file() and (path / "best").is_dir():
            path = path / "best"
        model_path = path / "model.pt" if path.is_dir() else path
        if not model_path.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {model_path}")
        if is_dry_run_model(model_path):
            raise ValueError(f"{path} is a dry-run artifact. Train without --dry-run to generate text.")
    try:
        infer = importlib.import_module("cyoai.core.infer")
    except ModuleNotFoundError as exc:
        if exc.name != "torch":
            raise RuntimeError("Local inference is unavailable; install the complete core package") from exc
        import cyoai
        core_dir = Path(cyoai.__file__).parent / "core"
        if not (core_dir / "infer.py").exists():
            core_dir = core_dir / "autoresearch"
        device = args.device or os.environ.get("CYOAI_DEVICE", "cpu")
        command = ["uv", "run"] + (["--extra", "cpu"] if device == "cpu" else [])
        command += ["python", "infer.py", str(path.resolve()), "--device", device,
                    "--max-new-tokens", str(args.max_tokens), "--temperature", str(args.temperature),
                    "--top-k", str(args.top_k)]
        if args.prompt is not None:
            command += ["--prompt", args.prompt]
        return subprocess.run(command, cwd=core_dir, env=scrub_env(os.environ)).returncode
    except ImportError as exc:
        raise RuntimeError("Local inference is unavailable; install cyoai[train] with core inference support") from exc
    if not hasattr(infer, "load_model") or not hasattr(infer, "generate"):
        raise RuntimeError("Core inference must expose load_model and generate")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", RuntimeWarning)
        model, tokenizer, _ = infer.load_model(path, device=args.device)
    for warning in caught:
        print(f"cyoai: warning: {warning.message}", file=sys.stderr)
    def generate(prompt):
        print(infer.generate(model, tokenizer, prompt, max_new_tokens=args.max_tokens,
                             temperature=args.temperature, top_k=args.top_k))
    if args.prompt is not None:
        generate(args.prompt)
    else:
        while True:
            try:
                prompt = input("You> ")
            except EOFError:
                break
            if not prompt.strip() or prompt.strip().lower() in {"exit", "quit", "/exit"}:
                break
            generate(prompt)
    return 0


def instances_command(args):
    config = load_config()
    names = ["runpod", "vast"] if args.provider == "all" else [args.provider]
    if args.command == "destroy" and not args.all and not args.id:
        raise ValueError("Specify an instance ID or --all")
    for name in names:
        key = config.runpod_api_key if name == "runpod" else config.vast_api_key
        if args.provider == "all" and not key:
            print(f"Skipping {name}: no API key configured", file=sys.stderr)
            continue
        provider = provider_for(name, config)
        if args.command == "destroy" and not args.all:
            provider.destroy(args.id)
        else:
            for instance in provider.list_instances():
                if args.command == "destroy":
                    provider.destroy(instance.id)
                else:
                    print(f"{name}\t{instance.id}\t{instance.status}\t{instance.gpu or ''}")
    return 0


def parser():
    root = argparse.ArgumentParser(prog="cyoai")
    subs = root.add_subparsers(dest="command", required=True)
    init = subs.add_parser("init", help="Configure BYOK credentials")
    init.add_argument("--non-interactive", action="store_true")
    for field in asdict(Config()):
        init.add_argument("--" + field.replace("_", "-"))
    init.set_defaults(func=init_command)
    train = subs.add_parser("train", help="Research a model for a goal")
    train.add_argument("goal", help="Describe what to train or improve")
    train.add_argument("--provider", choices=["local", "runpod", "vast"], default="local",
                       help="Where to train (default: local)")
    train.add_argument("--gpu", help="GPU type to request from a cloud provider")
    train.add_argument("--budget-mins", type=float, default=5,
                       help="Training minutes per attempt, excluding setup (default: 5)")
    train.add_argument("--max-iters", type=int, default=5,
                       help="Agent edits to try after the baseline (default: 5; 0 for baseline only)")
    train.add_argument("--workdir", default="runs", help="Directory for training workspaces (default: runs)")
    train.add_argument("--dry-run", action="store_true", help="Simulate training; no usable checkpoint is produced")
    train.add_argument("--smoke", action="store_true", help="Train a tiny model to check the pipeline")
    train.add_argument("--device", choices=["cpu", "cuda", "mps"], help="Override the training device")
    train.set_defaults(func=train_command)
    for name in ("run", "chat"):
        chat = subs.add_parser(name, help="Generate text from a trusted checkpoint")
        checkpoint = chat.add_mutually_exclusive_group()
        checkpoint.add_argument("path", nargs="?", help="Trusted run directory, checkpoint directory, or model.pt (default: latest trained checkpoint)")
        checkpoint.add_argument("--latest", action="store_true",
                                help="Use the newest trained checkpoint under the local runs/ directory")
        chat.add_argument("--prompt", help="Generate once; omit for an interactive prompt session")
        chat.add_argument("--max-tokens", type=int, default=128, help="Maximum new tokens (default: 128)")
        chat.add_argument("--device", choices=["cpu", "cuda", "mps"], help="Override the inference device")
        chat.add_argument("--temperature", type=float, default=0.8, help="Sampling temperature (default: 0.8)")
        chat.add_argument("--top-k", type=int, default=50, help="Sample from the top K tokens (default: 50)")
        chat.set_defaults(func=chat_command)
    for name in ("status", "destroy"):
        command = subs.add_parser(name)
        command.add_argument("--provider", choices=["runpod", "vast", "all"], default="all")
        if name == "destroy":
            command.add_argument("id", nargs="?")
            command.add_argument("--all", action="store_true")
        command.set_defaults(func=instances_command)
    return root


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        return args.func(args)
    except KeyboardInterrupt:
        print("Interrupted", file=sys.stderr)
        return 130
    except Exception as exc:
        print(f"cyoai: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
