# API Reference

This repository ships a typed Python package in `src/platformx`. We recommend generating an API reference using one of the following tools:

- pdoc (quick, single-command):

```bash
pip install pdoc3
pdoc --output-dir docs/api platformx
```

- Sphinx + autodoc (more flexible, best for larger projects):

1. Create a `docs/sphinx` directory and run `sphinx-quickstart`.
2. Enable `sphinx.ext.autodoc` and `sphinx.ext.napoleon` in `conf.py`.
3. Add `.. automodule:: platformx.core` style directives to the RST files or use `sphinx-apidoc`.

Keep API docs in `docs/api/` so MkDocs can include/generated pages if you prefer HTML generation first.
