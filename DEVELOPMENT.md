# Developer Readme

This document contains documentation intended for developers of voronoi_generator.

## Developer environment setup

To work on the voronoi_generator as a developer, you'll need to configure your local development environment. You can do this by simply running:
```bash
make project-setup
```
This will install Python `3.10` using PyEnv, create a virtual environment using Poetry, and install the pre-commit hooks.

> Note: The `project-setup` process will check whether `pre-commits`, `pyenv`, and `poetry` are installed. If not, it will ask to install them on your behalf as they're required to use this template.
> The first time the project is set up, you should follow the instructions in [First time setup](#first-time-setup).


A Makefile is just a usual text file to define a set of rules or instructions to run which can be run using the `make` command. To see the available make commands:
```bash
make help
```

## Testing

With the poetry shell active (see above), you can run all the tests using:

```bash
make test
```

Or specific tests:

```bash
python -m pytest tests/test_dummy.py
