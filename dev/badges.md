# Badge Generation Commands

Developer reference for generating project badges.

## Create badges

Run commands from project root:
```shell
uv sync --extra dev
uv run anybadge --label=Coverage --value=70 --suffix='%' --file=docs/badges/coverage.svg \
                --overwrite 50=red 70=gold 80=green 
uv run anybadge --label=Docs --value=readthedocs.io --file=docs/badges/docs.svg --overwrite --color=royalblue
uv run anybadge --label=Python --value='3.10–3.14' --file=docs/badges/python.svg --overwrite --color=green
```

## Reference badges

URI should be set to the main branch:

```markdown
[![Docs](https://raw.githubusercontent.com/C108-Lab/c108/main/docs/badges/docs.svg)](https://c108.readthedocs.io/)
![Python Versions](https://raw.githubusercontent.com/C108-Lab/c108/main/docs/badges/python.svg)
[![Coverage](https://raw.githubusercontent.com/C108-Lab/c108/main/docs/badges/coverage.svg)](https://app.codecov.io/gh/C108-Lab/c108)
```

## Coverage Badge Update

Update Coverage badge with stats in CI workflow.
