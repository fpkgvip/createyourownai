"""Inference for trusted local autoresearch checkpoints and source snapshots.

Snapshots and tokenizer pickles are executable Python: load trusted artifacts only.
"""

import argparse
import ast
import os
from pathlib import Path
import pickle
import sys
import types
import uuid
import warnings

import torch

if __package__:
    from . import attention
    from .device import get_device
else:
    import attention
    from device import get_device


def _constant_expression(node, names):
    if isinstance(node, ast.Name):
        return node.id in names
    if isinstance(node, ast.Attribute):
        return isinstance(node.value, ast.Name) and node.value.id == "math" and not node.attr.startswith("_")
    if isinstance(node, ast.Call):
        return (isinstance(node.func, ast.Attribute) and _constant_expression(node.func, names)
                and all(_constant_expression(arg, names) for arg in node.args)
                and all(_constant_expression(kw.value, names) for kw in node.keywords))
    if isinstance(node, (ast.Constant, ast.operator, ast.unaryop, ast.boolop, ast.cmpop, ast.Load)):
        return True
    if isinstance(node, (ast.BinOp, ast.UnaryOp, ast.BoolOp, ast.Compare, ast.IfExp,
                         ast.Tuple, ast.List, ast.Set, ast.Dict)):
        return all(_constant_expression(child, names) for child in ast.iter_child_nodes(node))
    return False


def _snapshot_module(path, device):
    tree = ast.parse(path.read_text(), filename=str(path))
    excluded = {"kernels", "prepare", "device", "checkpoint"}
    injected = {"fa3", "USE_CUDA", "SMOKE", "device", "compile_cuda"}
    body = []
    constants = {"USE_CUDA", "SMOKE"}
    for node in tree.body:
        if isinstance(node, ast.Import):
            names = [alias for alias in node.names if alias.name.split(".")[0] not in excluded]
            if names:
                body.append(ast.Import(names=names))
        elif isinstance(node, ast.ImportFrom):
            if node.module and node.module.split(".")[0] not in excluded:
                body.append(node)
        elif isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.name not in injected:
                body.append(node)
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            if not all(isinstance(t, ast.Name) and t.id not in injected for t in targets):
                continue
            try:
                ast.literal_eval(node.value)
            except (ValueError, TypeError):
                if not _constant_expression(node.value, constants):
                    continue
            body.append(node)
            constants.update(t.id for t in targets)
    module = types.ModuleType(f"_cyoai_snapshot_{uuid.uuid4().hex}")
    module.__dict__.update(
        __file__=str(path), fa3=attention, USE_CUDA=False, SMOKE=False,
        device=device, compile_cuda=lambda fn: fn,
    )
    sys.modules[module.__name__] = module
    try:
        code = ast.fix_missing_locations(ast.Module(body=body, type_ignores=[]))
        exec(compile(code, str(path), "exec"), module.__dict__)
    except Exception:
        sys.modules.pop(module.__name__, None)
        raise
    return module


def load_model(path, device=None):
    """Load trusted artifacts, preferring their architecture snapshot.

    Returns the model, tiktoken Encoding, and checkpoint metadata including
    the selected architecture source and any snapshot loading error.
    """
    path = Path(path).expanduser().resolve()
    checkpoint_path = path / "model.pt" if path.is_dir() else path
    out = checkpoint_path.parent
    device = torch.device(device) if device is not None else get_device()
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    snapshot = out / "train.py"
    fallback = Path(__file__).with_name("train.py")
    snapshot_error = None
    for source in dict.fromkeys([snapshot, fallback]):
        try:
            module = _snapshot_module(source, device)
            config = module.GPTConfig(**checkpoint["config"])
            model = module.GPT(config)
            model.load_state_dict(checkpoint["state_dict"], strict=True)
            model.to(device).eval()
            break
        except Exception as exc:
            if source == fallback:
                raise RuntimeError(f"Could not load architecture from {snapshot} or {fallback}") from exc
            snapshot_error = f"{type(exc).__name__}: {exc}"
            warnings.warn(f"Snapshot failed ({snapshot_error}); trying vendored train.py", RuntimeWarning)
    with (out / "tokenizer.pkl").open("rb") as handle:
        tokenizer = pickle.load(handle)
    info = {k: v for k, v in checkpoint.items() if k != "state_dict"}
    info.update(source=str(source), snapshot_error=snapshot_error, device=str(device))
    return model, tokenizer, info


@torch.inference_mode()
def generate(model, tokenizer, prompt, max_new_tokens=200, temperature=0.8, top_k=50, stream=None):
    """Return only generated text; optionally send decoded chunks to stream."""
    if max_new_tokens < 0 or temperature < 0 or top_k < 0:
        raise ValueError("max_new_tokens, temperature, and top_k must be non-negative")
    device = next(model.parameters()).device
    bos = tokenizer.encode_single_token("<|reserved_0|>")
    tokens = [bos] + tokenizer.encode_ordinary(prompt)
    generated = []
    emitted = ""
    context = model.config.sequence_len
    if context <= 0:
        raise ValueError("sequence_len must be positive")
    for _ in range(max_new_tokens):
        x = torch.tensor([tokens[-context:]], dtype=torch.long, device=device)
        logits = model(x)[0, -1].float()
        if temperature == 0:
            token = logits.argmax().item()
        else:
            logits = logits / temperature
            if top_k:
                threshold = torch.topk(logits, min(top_k, logits.numel())).values[-1]
                logits = logits.masked_fill(logits < threshold, -float("inf"))
            token = torch.multinomial(logits.softmax(dim=-1), 1).item()
        tokens.append(token)
        generated.append(token)
        if stream is not None:
            text = tokenizer.decode(generated)
            if not text.endswith("\ufffd"):
                stream(text[len(emitted):])
                emitted = text
    completion = tokenizer.decode(generated)
    if stream is not None and completion != emitted:
        stream(completion[len(emitted):])
    return completion


def repl(model, tokenizer, **generation_kwargs):
    while True:
        try:
            prompt = input("> ")
        except (EOFError, KeyboardInterrupt):
            break
        if not prompt.strip() or prompt.strip() == "exit":
            break
        generate(model, tokenizer, prompt, stream=lambda text: print(text, end="", flush=True), **generation_kwargs)
        print()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", help="Trusted output directory or model.pt")
    parser.add_argument("--prompt")
    parser.add_argument("--max-new-tokens", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--device", choices=["cuda", "mps", "cpu"])
    parser.add_argument("--seed", type=int)
    args = parser.parse_args()
    if args.seed is not None:
        torch.manual_seed(args.seed)
    model, tokenizer, _ = load_model(args.path, args.device)
    if next(model.parameters()).device.type == "cpu":
        torch.set_num_threads(min(4, os.cpu_count() or 1))
    kwargs = dict(max_new_tokens=args.max_new_tokens, temperature=args.temperature, top_k=args.top_k)
    if args.prompt is not None:
        print(generate(model, tokenizer, args.prompt, **kwargs))
    else:
        repl(model, tokenizer, **kwargs)


if __name__ == "__main__":
    main()
