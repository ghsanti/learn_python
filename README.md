Mini-project to learn some modern python.

It uses:

- Project Management: [astral/uv](https://github.com/astral-sh/uv)
  One needs to activate the `.venv` as well.
- Lint: [astral/ruff](https://github.com/astral-sh/ruff)
  Own separate config file.
- Type-Check: [pyright](https://github.com/microsoft/pyright)
  Own separate config file. Install using _[nodejs]_ extra.
- [pre-commit](https://pre-commit.com/)
  Run `precommit install` after installing.
- testing: [pytest](https://docs.pytest.org/)
- pyproject: python project metadata.
- Github Actions: integration for quality checks, PyPI publishing, and documentation publishing.
- Docker for Codespaces integration (_.devcontainer_)

The project itself is a simple neural network in Pytorch, with Hyperparameter tuning using Ray Tune. It uses code from Pytorch examples.

## Dev

```bash
git clone ...
uv sync
source .venv/bin/activate
```

Select the `.venv` python-interpreter in VSCode.

## Run

You can run the project using:

```python
python -m src.train
```
