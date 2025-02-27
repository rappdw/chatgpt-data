# ChatGPT Data Project Guidelines

## Commands
- Install: `uv pip install -e .` or `pip install -e .`
- Install dev dependencies: `uv pip install -e ".[dev]"`
- Run tests: `pytest` or `pytest tests/test_specific_file.py::TestClass::test_method`
- Lint: `black .` `isort .` `ruff check .`
- Type check: `mypy .`
- Build: `python -m build`

## CLI Usage
- All trends: `all_trends --data-dir ./rawdata --output-dir ./data`
- User trends: `user_trends --data-dir ./rawdata --output-dir ./data`
- GPT trends: `gpt_trends --data-dir ./rawdata --output-dir ./data`

## Code Style
- Use type hints for all functions and methods
- Follow black/isort formatting (88 char line length)
- Organized imports: stdlib first, then third-party, then local
- Class names: PascalCase, functions/methods: snake_case
- Include docstrings with Args and Returns for all functions/methods
- Use pathlib.Path for file operations, not os.path
- Handle file loading errors with appropriate exception handling
- Use pandas/matplotlib for data analysis and visualization