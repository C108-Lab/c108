#!/usr/bin/env python3

import sys
import json
import subprocess
from pathlib import Path

# Configuration
BRANCH = "badges"
REMOTE = "origin"
BOT_NAME = "github-actions[bot]"
BOT_EMAIL = "github-actions[bot]@users.noreply.github.com"


def git(*args):
    """Run a git command and return (returncode, stdout)."""
    result = subprocess.run(["git"] + list(args), capture_output=True, text=True)
    if result.returncode != 0 and "diff" not in args:
        print(f"❌ Git Error: {' '.join(args)}\n{result.stderr}")
        sys.exit(result.returncode)
    return result


def main():
    if len(sys.argv) < 3:
        print(f"Usage: {sys.argv[0]} <file-path> <commit-message>")
        sys.exit(1)

    file_path = Path(sys.argv[1])
    commit_msg = sys.argv[2]

    # 1. Configure Bot Identity
    git("config", "user.name", BOT_NAME)
    git("config", "user.email", BOT_EMAIL)

    # 2. Stage the file
    if not file_path.exists():
        print(f"❌ File not found: {file_path}")
        sys.exit(1)

    git("add", str(file_path))

    # 3. Check for changes (git diff --staged --quiet returns 1 if changed, 0 if clean)
    diff_check = git("diff", "--staged", "--quiet")
    if diff_check.returncode == 0:
        print(f"✅ No changes detected in {file_path}. Skipping commit.")
        sys.exit(0)

    # 4. Enrich Commit Message (if JSON)
    if file_path.suffix == ".json":
        try:
            data = json.loads(file_path.read_text())
            # Extract badge details: "84% green"
            extra = f"{data.get('message', '')} {data.get('color', '')}".strip()
            if extra:
                commit_msg = f"{commit_msg} {extra}"
        except Exception as e:
            print(f"⚠️ Could not parse JSON for commit message: {e}")

    # 5. Commit and Push
    print(f"📝 Committing to {BRANCH}: '{commit_msg}'")
    git("commit", "-m", commit_msg)
    git("push", REMOTE, BRANCH)


if __name__ == "__main__":
    main()
