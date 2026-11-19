# Upstream

Source: https://github.com/karpathy/autoresearch
Commit: `228791fb` (MIT license; copyright Andrej Karpathy).

`prepare.py`, `train.py`, `program.md`, and `pyproject.toml` originate
from this revision. The upstream Python files have no MIT header comments;
their original module docstrings are retained. `program.md` is unchanged.

## Local modifications

- Environment-controlled time budget, cache directory, smoke configuration,
  and CUDA/MPS/CPU device selection.
- Small smoke tokenizer/model and shared PyTorch attention fallback; CUDA
  retains upstream FlashAttention, compilation, and bf16 behavior.
- Device-aware data buffers/evaluation, without changing BPB arithmetic.
- A two-line training hook saves model/config, source, tokenizer, and metrics.
- Optional `cpu` uv extra selects CPU PyTorch on Linux; the default keeps
  upstream cu128 PyTorch. Run CPU workspaces with `uv run --extra cpu ...`
  (or export `UV_EXTRA=cpu`). Removed the reference to the absent README.

## Environment

- `CYOAI_TIME_BUDGET`: training seconds, default `300`.
- `CYOAI_CACHE_DIR`: data/tokenizer cache, default `~/.cache/autoresearch`.
- `CYOAI_SMOKE=1`: 256-token sequences, 4096 evaluation tokens, 2048-token
  vocabulary, 1M tokenizer-training characters, depth 2, full attention,
  device batch 2, total batch 512. Default cache becomes
  `~/.cache/autoresearch-smoke` (explicit cache overrides take precedence).
- `CYOAI_DEVICE`: `cuda`, `mps`, or `cpu`; default auto-detect in that order.
- `CYOAI_OUT_DIR`: artifact directory, default `./out` relative to cwd.

Smoke metrics are not comparable to full upstream runs.

## MIT license (upstream)

Copyright (c) 2026 Andrej Karpathy

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
