from dataclasses import dataclass
import math
from pathlib import Path
import re

from .workspace import git, initialize_git


@dataclass
class Result:
    commit: str
    val_bpb: float
    memory_gb: float
    status: str
    description: str


def parse_metrics(output: str) -> tuple[float, float]:
    values = re.findall(r"(?m)^\s*val_bpb:\s*([\d.eE+-]+)", output)
    if not values:
        raise ValueError("Training output has no val_bpb metric")
    value = float(values[-1])
    if not math.isfinite(value):
        raise ValueError("Non-finite val_bpb")
    memory = re.findall(r"(?m)^\s*peak_vram_mb:\s*([\d.eE+-]+)", output)
    return value, float(memory[-1]) / 1024 if memory else 0.0


def parse_proposal(response: str) -> tuple[str, str]:
    blocks = re.findall(r"```([^\n]*)\n(.*?)```", response, re.S)
    if len(blocks) != 1 or blocks[0][0].strip() not in {"python", "py", ""}:
        raise ValueError("Expected exactly one Python block replacing only train.py")
    if re.search(r"(?:^|\n)\s*(?:#{1,6}\s*)?(?:FILE:|filename:)\s*prepare.py", response, re.I):
        raise ValueError("Only train.py may be edited")
    description = re.search(r"(?m)^DESCRIPTION:\s*(.+)$", response)
    if not description:
        raise ValueError("Missing DESCRIPTION")
    code = blocks[0][1]
    compile(code, "train.py", "exec")
    return code, description[1].strip()


class ResearchLoop:
    def __init__(self, workspace: Path, llm, runner, max_iters=5, log=print):
        self.workspace = Path(workspace)
        self.llm, self.runner, self.max_iters, self.log = llm, runner, max_iters, log

    def run(self) -> Result | None:
        workspace = self.workspace
        if not (workspace / ".git").exists():
            initialize_git(workspace)
        best_commit = git(workspace, "rev-parse", "HEAD")
        best = None

        def untracked():
            return set(filter(None, git(workspace, "ls-files", "--others", "--exclude-standard", "-z").split("\0")))

        existing_untracked = untracked()
        results = workspace / "results.tsv"
        if not results.exists():
            results.write_text("commit\tval_bpb\tmemory_gb\tstatus\tdescription\n")
        try:
            for iteration in range(self.max_iters + 1):
                existing_files = set(filter(None, git(workspace, "ls-files", "--cached", "--others", "-z").split("\0")))
                description = "baseline"
                candidate = best_commit
                try:
                    if iteration:
                        prompt = (f"Iteration {iteration}\n<train.py>\n" +
                                  (workspace / "train.py").read_text() + "\n</train.py>\n" +
                                  "Results:\n" + results.read_text())
                        system = ((workspace / "program.md").read_text() +
                                  "\nReplace ONLY train.py. Never change prepare.py. Return one full "
                                  "```python block and one DESCRIPTION: line.")
                        code, description = parse_proposal(self.llm.complete(system, prompt))
                        (workspace / "train.py").write_text(code)
                        git(workspace, "add", "train.py")
                        git(workspace, "commit", "--only", "--allow-empty", "-m", description, "--", "train.py")
                        candidate = git(workspace, "rev-parse", "HEAD")
                    exit_code, output = self.runner.run()
                    if exit_code:
                        raise RuntimeError(f"Training exited with code {exit_code}")
                    if (git(workspace, "rev-parse", "HEAD") != candidate or
                            git(workspace, "diff", candidate, "--name-only") or
                            untracked() - existing_untracked):
                        raise ValueError("Training modified workspace source files")
                    value, memory = parse_metrics(output)
                    status = "keep" if best is None or value < best.val_bpb else "discard"
                    result = Result(candidate, value, memory, status, description)
                    if status == "keep":
                        if hasattr(self.runner, "keep"):
                            self.runner.keep()
                        best, best_commit = result, candidate
                except Exception as exc:
                    self.log(f"Iteration {iteration} crashed: {exc}")
                    result = Result(candidate, float("inf"), 0.0, "crash", description)
                finally:
                    new_files = untracked() - existing_files
                    git(workspace, "reset", "--hard", best_commit)
                    for name in new_files:
                        path = workspace / name
                        if path.is_file() or path.is_symlink():
                            path.unlink()
                with results.open("a") as handle:
                    description = result.description.replace("\t", " ").replace("\n", " ")
                    handle.write(f"{result.commit}\t{result.val_bpb}\t{result.memory_gb}\t{result.status}\t{description}\n")
                self.log(f"Iteration {iteration}: {result.status} val_bpb={result.val_bpb}")
        except KeyboardInterrupt:
            self.log("Research interrupted; best train.py restored.")
            raise
        return best
