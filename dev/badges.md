# Badge Generation Commands

Developer reference for generating project badges.

## Creating badges for the first time

Run commands from project root:
```shell
uv sync --extra dev
uv run anybadge --label=Coverage --value=70 --suffix='%' --file=docs/badges/coverage.svg \
                --overwrite 50=red 70=gold 80=green 
uv run anybadge --label=Docs --value=readthedocs.io --file=docs/badges/docs.svg --overwrite --color=royalblue
uv run anybadge --label=Python --value='3.10–3.14' --file=docs/badges/python.svg --overwrite --color=green
```

## Coverage Badge Update

The Coverage badge with stats should be updated in CI workflow.
