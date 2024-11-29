Mini-project to learn Ray, PyTorch, and Python packaging

## Packaging

This project uses Python>=3.11

It uses:

- Project Management: [astral/uv](https://github.com/astral-sh/uv)
  One needs to activate the _.venv_ as well.

  After activating, it's good to update _pip_ and _build_

```python
python3 -m pip install --upgrade pip
python3 -m pip install --upgrade build
```

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

For distributing packages one needs to:

1. Build it (uses _hatch_ in this repo.) Setuptools, Flitch, PDM are other build backends.

One can then use any build frontend (like _build_), they will search for the build backend and create the distribution files.

In this case, the file has:

```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"
```

To show which backend to use. The frontend installs _hatch_ if needed.

2. Have the right project structure and metadata (_pyproject.toml_) 3. Publish it on PyPI (publicly or privately.)

Key is to have _src/project_name_ and the name of the package in the _pyproject.toml_ file.

Also have \_\_\_init.py\_\_\_ files in each subdirectory. **This allows the user to import the directory as if it were a module.**

The _tests_ folder is outside the source folder, and won't be distributed. Other files outside source will be incl. in the source distribution (_tar.gz_ compression.)

Lastly one needs _Readme.md_ and _LICENSE_

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
python -m path.to.file
```

## Build

```python
uv pip install --upgrade build
uv build
```

## PyTorch settings

`autocast` isn't worth even trying, it makes things slower. One can check even unresolved issues. Ofc, this is for CPU, for other devices it probably changes.
