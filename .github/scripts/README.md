
# Workflow Scripts


## Initialize badges Branch

Run this locally once to set up the empty `badges` branch structure

```shell
# Checkout
cd /tmp/c108

# Create orphan branch (no history)
git checkout --orphan badges
git reset --hard

# Recreate directory structure
mkdir -p docs/badges

# Create placeholders (so scripts don't fail on first run)
echo '["placeholder"]' > docs/badges/coverage-badge.json
touch docs/badges/badges.toml

git add docs/badges/
git commit -m "setup: init badges branch"
git push -u origin badges

# Switch back to main
git checkout main
```

## Badge Update Scripts

These scripts are used by GitHub Actions to update status badges (like coverage) in the repository.

### Environment and Dependencies

Run `*.py` scripts via `uv run python ...` after `uv sync --extra badges`.

This 2 out of all Python scripts depend on extras from the pyproject.toml badges group

### ⚠️ Critical Requirement

**The coverage report file (e.g., `coverage-unit.xml`) MUST be ignored by Git.**

Ensure your root `.gitignore` contains:
```gitignore
coverage*.xml
```