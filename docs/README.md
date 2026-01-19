# Documentation workflow

We use MkDocs with the config stored at `docs/mkdocs.yaml`.

```bash
# Install dev extras first
pip install -e .["dev"]

# Serve with live reload (default port 8000)
mkdocs serve --config-file docs/mkdocs.yaml

# Build static site into ./build
mkdocs build --config-file docs/mkdocs.yaml --site-dir build
```

You can also use Invoke shortcuts: `invoke build_docs` or `invoke serve_docs`.
