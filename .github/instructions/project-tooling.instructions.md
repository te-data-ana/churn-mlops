---
name: Project Tooling and Dependency Standards
description: "Use when changing project dependencies, pyproject.toml, uv.lock, pre-commit configuration, tests, notebooks, or repository tooling. Covers uv workflows, lockfile management, validation, and project conventions."
applyTo: ["pyproject.toml", "uv.lock", ".pre-commit-config.yaml", "**/*.py", "**/*.ipynb", "**/*.md"]
---
# Project Tooling and Dependency Standards

## uv and dependencies

- Use `uv` for environment, dependency, and command management.
- Run project commands with `uv run` so they use the locked project environment.
- Add runtime dependencies with `uv add <package>` and development dependencies with `uv add --dev <package>`.
- Update dependencies through `pyproject.toml` and regenerate `uv.lock` with `uv lock`; do not edit `uv.lock` manually.
- Keep the declared Python requirement (`>=3.14`) and the lockfile consistent after dependency changes.
- Do not commit virtual environments, generated caches, coverage output, or local experiment artifacts unless the repository explicitly requires them.

## Project configuration

- Treat `pyproject.toml` as the source of truth for package metadata, dependencies, pytest settings, and Ruff configuration.
- Preserve the existing `src/` package layout and `tests/` test discovery conventions.
- Keep dependency and tooling changes focused; avoid unrelated formatting or configuration churn.

## Validation and hooks

- Run `uv run pytest` for the full test suite. Use focused pytest targets while iterating, then run the full suite before completing a change.
- Run `uv run pre-commit run --all-files` when changing Python, configuration, documentation, or notebook files.
- Respect pre-commit autofixes from Ruff, formatting, whitespace, notebook stripping, and secret detection; review the resulting diff before finishing.
- Do not bypass secret detection or commit credentials, tokens, personal data, or other sensitive values.

## Notebooks

- Keep notebooks reproducible and minimize committed cell output.
- Run the configured notebook cleanup hook before committing notebook changes.
