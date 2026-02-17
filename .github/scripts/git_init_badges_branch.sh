#!/usr/bin/env bash
set -e  # Stop on error

# 1. Setup temporary directory
REPO_URL="git@github.com:C108-Lab/c108.git"
TEMP_DIR=${1:-"/tmp/c108-badges-setup"}

rm -rf "$TEMP_DIR"
git clone "$REPO_URL" "$TEMP_DIR"
cd "$TEMP_DIR"

# 2. Create orphan branch (empty, no history)
git checkout --orphan badges
git reset --hard

# 3. Create directory structure & placeholders
mkdir -p docs/badges

# Create valid initial JSON so shields.io doesn't error out
echo '{
  "schemaVersion": 1,
  "label": "Coverage",
  "message": "init",
  "color": "lightgrey"
}' > docs/badges/coverage-badge.json

# Create empty TOML
touch docs/badges/badges.toml

# 4. Commit and Push
git add docs/badges/
git commit -m "setup: init badges branch"
git push -u origin badges

# 5. Cleanup
rm -rf "$TEMP_DIR"

echo "✅ 'badges' branch initialized and pushed!"
