# Create Your Own AI

`cyoai` wraps [Karpathy's autoresearch](https://github.com/karpathy/autoresearch) in a CLI. Describe a training goal, bring your own LLM API key, and let a research agent try edits to the training script. Then run your best checkpoint on your own machine.

## Quickstart: generate text on CPU

Use a Bash-compatible shell on Linux or macOS, with Python 3.10+,
[uv](https://docs.astral.sh/uv/getting-started/installation/) and Git on your
PATH. You'll need internet access and a few GB of free disk space for the CPU
run. No API key or GPU is needed.

```bash
git clone https://github.com/fpkgvip/createyourownai.git
cd createyourownai
uv sync
uv run cyoai train "Try a smaller learning rate" --provider local --max-iters 2 --dry-run
```

The dry run exercises the loop with simulated scores. It doesn't download
training data or produce a model you can run.

Next, train a real CPU baseline with no LLM calls or agent edits. The first
run downloads PyTorch and two public Hugging Face data shards (about 180 MiB
for the data alone). The three-second training budget does not include setup,
evaluation, or saving the checkpoint.

```bash
CYOAI_LLM_PROVIDER=fake uv run cyoai train "Train a tiny baseline" \
  --provider local --smoke --device cpu --budget-mins 0.05 --max-iters 0
```

After training succeeds, generate a short continuation:

```bash
uv run cyoai run --latest --prompt "Once upon a time" --max-tokens 32
```

`--max-iters 0` trains only the baseline. `--latest` loads the newest trained
checkpoint under `runs/`, skipping dry runs. To select a specific run or use
a custom `--workdir`, pass the `best/` path printed after `artifacts:` instead
of `--latest`. The model, tokenizer, and architecture snapshot stay together
in `best/`; `results.tsv` records validation bits per byte (`val_bpb`, lower
is better). Omit `--prompt` for an interactive session; type `exit` to quit.

For LLM-driven experiments, run `uv run cyoai init` to choose OpenAI or
Anthropic and configure your key. Use `--provider local` for local training or
choose a GPU provider below.

This is a small language model trained from scratch, not an instruction-tuned
chatbot. It continues prompts; a tiny smoke run checks the pipeline, not model
quality.

## Train on a cloud GPU

Choose `runpod` or `vast`, export `RUNPOD_API_KEY` or `VAST_API_KEY` respectively
(or save the key with `uv run cyoai init`), and install OpenSSH. Have an SSH
keypair with the matching `.pub` file. Set `CYOAI_SSH_KEY` to the private key
path if you do not use a default key from `~/.ssh`. File transfers use rsync
when available or scp otherwise. Cloud GPU time and LLM calls can cost money.

```bash
uv run cyoai train "Improve validation loss" --provider runpod --budget-mins 5 --max-iters 5
uv run cyoai status --provider runpod
uv run cyoai destroy INSTANCE_ID --provider runpod
```

Use `--gpu` to request a specific GPU and substitute `vast` for Vast.ai. The
budget is per training attempt, not a spending limit; provisioning, data
preparation, and transfers take extra time. The CLI attempts to tear down the
instance on exit, including Ctrl+C. After a crash, forced kill, or network
outage, check the provider dashboard and destroy any instance still accruing charges.
Cloud provisioning and teardown have mock coverage; live GPU rentals are unverified.

## Credentials and trust

- `cyoai init` stores credentials in `~/.config/cyoai/config.json` (mode `0600`).
  On shared systems, prefer environment variables and restrict access to your account.
- If you use [.env.example](.env.example) to create a local `.env`, export the
  populated values in your shell (for example, `set -a; . ./.env; set +a`).
  Do not add real keys, private SSH keys, or config files to Git.
- The research agent sends the training goal and workspace context to the LLM
  provider. It runs generated Python without sandboxing.
  Run it in an isolated environment without unrelated credentials or sensitive files.
  Only load checkpoints you trust, including the contents of `runs/` when
  using `--latest`.

## Tests

```bash
uv run --extra dev pytest tests/agent tests/cli tests/e2e tests/providers
uv run --project cyoai/core --extra cpu --with pytest pytest tests/core
CYOAI_RUN_CPU_SMOKE=1 uv run --extra dev pytest tests/e2e -k tiny_local
```

The default suite uses offline or mocked providers. The opt-in test downloads
training data, trains a tiny model locally, and generates text with `cyoai run`.

## Attribution

The training core is vendored from [karpathy/autoresearch](https://github.com/karpathy/autoresearch)
by Andrej Karpathy and adapted for checkpointing and local inference.
See [LICENSE](LICENSE) for the MIT license and upstream attribution.
