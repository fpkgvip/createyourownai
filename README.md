# Create Your Own AI

`cyoai` wraps [Karpathy's autoresearch](https://github.com/karpathy/autoresearch) in a CLI. Describe a training goal, bring your own LLM API key, and let a research agent try edits to the training script. Then run your best checkpoint on your own machine.

## Five-minute quickstart

Install [uv](https://docs.astral.sh/uv/getting-started/installation/) and Git, then:

```bash
git clone https://github.com/fpkgvip/createyourownai.git
cd createyourownai
uv sync
uv run cyoai train "Try a smaller learning rate" --provider local --max-iters 2 --dry-run
```

The dry run exercises the loop with simulated scores. It does not need an API
key, a GPU, or training data, and it does not produce a usable checkpoint.

To try real training on CPU without a paid LLM, use the built-in fake research
agent. The first run downloads training data and model dependencies:

```bash
CYOAI_LLM_PROVIDER=fake uv run cyoai train "Train a tiny baseline" --provider local --smoke --device cpu --budget-mins 1 --max-iters 1
uv run cyoai run runs/RUN_ID/best
```

Use the actual run ID printed by `cyoai train`. The best model and tokenizer stay
together in `best/`, and `results.tsv` records validation bits per byte
(`val_bpb`, lower is better). Pass `--prompt "Once upon a time"` to sample once
instead of starting an interactive session.

For LLM-driven experiments, run `uv run cyoai init` to choose OpenAI or
Anthropic and configure your key. Use `--provider local` for local training or
choose a GPU provider below.

This is a small language model trained from scratch, not an instruction-tuned
chatbot. It continues prompts; a tiny smoke run checks the pipeline, not model
quality.

## Train on a cloud GPU

Choose `runpod` or `vast`, provide its API key, and have an SSH keypair and the
`ssh` client available. Set `CYOAI_SSH_KEY` if you do not use a default key from
`~/.ssh`. Cloud GPU time and LLM calls can cost money.

```bash
uv run cyoai train "Improve validation loss" --provider runpod --budget-mins 5 --max-iters 5
uv run cyoai status --provider runpod
uv run cyoai destroy INSTANCE_ID --provider runpod
```

Use `--gpu` to request a specific GPU and substitute `vast` for Vast.ai. The
budget is per training attempt, not a spending limit; provisioning, data
preparation, and transfers take extra time. The CLI attempts to tear down the
instance when training exits. After a crash, forced kill, or network outage,
check the provider dashboard and destroy any instance still accruing charges.

## Credentials and trust

- `cyoai init` stores credentials in `~/.config/cyoai/config.json` (mode `0600`).
  On shared systems, prefer environment variables and restrict access to your account.
- If you use [.env.example](.env.example) to create a local `.env`, export the
  populated values in your shell (for example, `set -a; . ./.env; set +a`).
  Do not add real keys, private SSH keys, or config files to Git.
- The research agent sends the training goal and workspace context to the LLM
  provider, and its generated Python runs as code. It is not sandboxed by the CLI.
  Run it in an isolated environment without unrelated credentials or sensitive files.
  Only load checkpoints you trust.

## Tests

```bash
uv run pytest
CYOAI_RUN_CPU_SMOKE=1 uv run pytest tests/e2e -k tiny_local
```

The default suite uses offline or mocked providers. The opt-in test downloads
training data, trains a tiny model locally, and generates text with `cyoai run`.

## Attribution

The training core is vendored from [karpathy/autoresearch](https://github.com/karpathy/autoresearch)
by Andrej Karpathy and adapted for checkpointing and local inference.
See [LICENSE](LICENSE) for the MIT license and upstream attribution.
