"""
Commit Branch worktree changes

NOTE: Run via `uv run python ...`.
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

BRANCH = "badges"
REMOTE = "origin"
BOT_NAME = "github-actions[bot]"
BOT_EMAIL = "github-actions[bot]@users.noreply.github.com"


def git(repo_dir: Path, *args) -> subprocess.CompletedProcess[str]:
    cmd = ["git", "-C", str(repo_dir), *args]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0 and "diff" not in args:
        print(f"❌ Git Error: {' '.join(cmd)}\n{result.stderr}")
        sys.exit(result.returncode)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Commit & push badge changes if any.")
    parser.add_argument("file_path", help="File or directory to stage (relative to repo-dir).")
    parser.add_argument("commit_message", help="Commit message to use.")
    parser.add_argument(
        "--repo-dir",
        default=".",
        help="Path to the git worktree/repo to operate on (default: current directory).",
    )
    args = parser.parse_args()

    repo_dir = Path(args.repo_dir).resolve()
    file_path = (repo_dir / args.file_path).resolve()
    commit_msg = args.commit_message

    git(repo_dir, "config", "user.name", BOT_NAME)
    git(repo_dir, "config", "user.email", BOT_EMAIL)

    if not file_path.exists():
        print(f"❌ File not found: {file_path}")
        sys.exit(1)

    git(repo_dir, "add", str(file_path))

    diff_check = git(repo_dir, "diff", "--staged", "--quiet")
    if diff_check.returncode == 0:
        print(f"✅ No changes detected in {file_path}. Skipping commit.")
        sys.exit(0)

    if file_path.suffix == ".json":
        try:
            data = json.loads(file_path.read_text())
            extra = f"{data.get('message', '')} {data.get('color', '')}".strip()
            if extra:
                commit_msg = f"{commit_msg} {extra}"
        except Exception as exc:  # noqa: BLE001
            print(f"⚠️ Could not parse JSON for commit message: {exc}")

    print(f"📝 Committing to {BRANCH}: '{commit_msg}'")
    git(repo_dir, "commit", "-m", commit_msg)
    git(repo_dir, "push", REMOTE, BRANCH)


if __name__ == "__main__":
    main()
