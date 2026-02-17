#!/usr/bin/env bash
set -euo pipefail

BRANCH="badges"
REMOTE="origin"

echo "🔄 Syncing with ${REMOTE}/${BRANCH}..."

git fetch "${REMOTE}" "${BRANCH}"

CURRENT_BRANCH=$(git branch --show-current || echo "detached")
if [[ "$CURRENT_BRANCH" != "$BRANCH" ]]; then
  echo "Switching to ${BRANCH}..."
  # Force-create or reset local branch to match remote
  git switch -C "${BRANCH}" "${REMOTE}/${BRANCH}"
fi

git reset --hard "${REMOTE}/${BRANCH}"
echo "✅ Workspace is now at $(git rev-parse --short HEAD) on ${BRANCH}"