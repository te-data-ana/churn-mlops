---
name: Python Coding and Testing Standards
description: "Use when writing or modifying Python source code or pytest tests. Covers type hints, PEP 8, naming, Google-style docstrings, and function-level test coverage."
applyTo: "**/*.py"
---
# Python Coding and Testing Standards

## Coding style

- Add type hints to every function, including parameter types and the return type.
- Follow PEP 8 and keep formatting consistent with the surrounding code.
- Use `snake_case` for function names.
- Write Google-style docstrings for public modules, classes, and functions. Document arguments, returns, and raises when applicable.

## Comments

- Prefer clear names and small functions over comments that restate the code.
- Add brief comments only to explain non-obvious intent, constraints, workarounds, or trade-offs.
- Keep comments accurate and update or remove them when the implementation changes.
- Use docstrings for public API documentation and comments for local implementation context.

## Logging

- Use the `logging` module instead of `print()` for application diagnostics.
- Create a module-level logger with `logging.getLogger(__name__)` and use the appropriate log level for each message.
- Use lazy logging interpolation, such as `logger.info("Loaded %s records", record_count)`, rather than eagerly formatted strings.
- Do not log passwords, tokens, personal data, or other sensitive values.
- Log exceptions with `logger.exception()` when handling an error and preserve the original exception context when re-raising.
- Keep logging configuration centralized; application modules should emit log records rather than configure global handlers.

## Testing

- Use `pytest` for automated tests.
- Add at least one focused test for every production function.
- Keep test modules named `test_*.py` in the `tests/` directory and mirror the source package structure where practical.
- Use parametrization when multiple cases exercise the same function without duplicating test logic.

## Documentation
- Scan README.md and adjust if necessary, when adding or changing public modules, classes, or functions.
