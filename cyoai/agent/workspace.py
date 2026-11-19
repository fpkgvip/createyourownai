from datetime import datetime, timezone
from pathlib import Path
import shutil
import subprocess


def git(workspace: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", "-c", "user.name=cyoai", "-c", "user.email=cyoai@localhost", *args],
        cwd=workspace, text=True, stderr=subprocess.STDOUT).strip()


def initialize_git(workspace: Path) -> None:
    if not (workspace / ".git").exists():
        git(workspace, "init")
    sources = [path.name for path in workspace.iterdir() if path.is_file() and
               (path.suffix in {".py", ".md", ".toml"} or path.name == ".gitignore")]
    git(workspace, "add", *sources)
    git(workspace, "commit", "--allow-empty", "-m", "Research baseline")


def create_workspace(goal: str, dest: Path, source_dir: Path | None = None) -> Path:
    if source_dir is None:
        import cyoai
        source_dir = Path(cyoai.__file__).parent / "core"
        if (source_dir / "autoresearch").is_dir():
            source_dir = source_dir / "autoresearch"
    source_dir = Path(source_dir)
    if not (source_dir / "train.py").is_file():
        raise FileNotFoundError("Core training files are unavailable; install the complete cyoai package.")
    workspace = Path(dest) / datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S-%f")
    workspace.mkdir(parents=True)
    for source in source_dir.iterdir():
        if source.is_file() and source.suffix in {".py", ".md", ".toml"}:
            shutil.copy2(source, workspace / source.name)
    program = workspace / "program.md"
    program.write_text((program.read_text() if program.exists() else "") + "\n## User goal\n\n" + goal + "\n")
    (workspace / ".gitignore").write_text("out/\nbest/\n.venv/\n__pycache__/\nresults.tsv\n")
    initialize_git(workspace)
    return workspace
