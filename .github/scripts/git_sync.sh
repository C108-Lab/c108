#!/usr/bin/env bash
#
# Crete and publish branch worktree
#

set -euo pipefail

BRANCH=${1:-"badges"}
REMOTE="origin"
DEFAULT_WORKTREE_ROOT="${RUNNER_TEMP:-/tmp}"
WORKTREE_DIR="${BADGES_WORKTREE:-${DEFAULT_WORKTREE_ROOT}/${BRANCH}-worktree}"

echo "🔄 Preparing worktree ${WORKTREE_DIR} for ${REMOTE}/${BRANCH}..."

git fetch "${REMOTE}" "${BRANCH}"

# Remove any stale worktree at the same path (helpful if the runner reuses a workspace)
if [[ -d "${WORKTREE_DIR}" ]]; then
  echo "🧹 Removing existing worktree at ${WORKTREE_DIR}"
  git worktree remove --force "${WORKTREE_DIR}"
fi

git worktree add "${WORKTREE_DIR}" "${REMOTE}/${BRANCH}"

git -C "${WORKTREE_DIR}" reset --hard "${REMOTE}/${BRANCH}"
REV=$(git -C "${WORKTREE_DIR}" rev-parse --short HEAD)
echo "✅ ${WORKTREE_DIR} is now at ${REV}"

# Make the path available to later workflow steps
{
  echo "BADGES_WORKTREE=${WORKTREE_DIR}"
} >> "${GITHUB_ENV}"

echo "📦 Exported BADGES_WORKTREE=${WORKTREE_DIR}"