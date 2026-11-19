"""Persist a self-contained result without expanding the research script."""

import json
import os
import shutil
from dataclasses import asdict
from pathlib import Path

import torch


def save_checkpoint(model, metrics, tokenizer_dir, source_path):
    out = Path(os.environ.get("CYOAI_OUT_DIR", "./out"))
    out.mkdir(parents=True, exist_ok=True)
    model = getattr(model, "_orig_mod", model)
    config = asdict(model.config)
    torch.save({"state_dict": model.state_dict(), "config": config, **metrics}, out / "model.pt")
    shutil.copy2(Path(tokenizer_dir) / "tokenizer.pkl", out / "tokenizer.pkl")
    shutil.copy2(source_path, out / "train.py")
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")


def save_training_result(namespace):
    from prepare import TOKENIZER_DIR

    g = namespace
    metrics = {
        "val_bpb": round(g["val_bpb"], 6),
        "training_seconds": round(g["total_training_time"], 1),
        "total_seconds": round(g["t_end"] - g["t_start"], 1),
        "peak_vram_mb": round(g["peak_vram_mb"], 1),
        "mfu_percent": round(g["steady_state_mfu"], 2),
        "total_tokens_M": round(g["total_tokens"] / 1e6, 1),
        "num_steps": g["step"],
        "num_params_M": round(g["num_params"] / 1e6, 1),
        "depth": g["DEPTH"],
    }
    save_checkpoint(g["model"], metrics, TOKENIZER_DIR, g["__file__"])
