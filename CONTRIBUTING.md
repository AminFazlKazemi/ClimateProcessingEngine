# Contributing to Climatology Engine

We welcome contributions! Please follow these guidelines.

## Getting Started

1. Fork the repository
2. Clone your fork: `git clone https://github.com/yourusername/ClimateProcessingEngine.git`
3. Create a virtual environment: `python -m venv venv`
4. Activate it: `source venv/bin/activate` (Linux/macOS) or `venv\Scripts\activate` (Windows)
5. Install development dependencies: `pip install -e .[dev]`
6. Install pre-commit hooks: `pre-commit install`

## Code Style

- Follow **Black** code style: `black .`
- Sort imports: `isort .`
- Check types: `mypy .`
- Lint: `flake8 .`

## Testing

- Run tests: `pytest tests/ -v`
- Check coverage: `pytest tests/ --cov=. --cov-report=html`

## Pull Request Process

1. Create a feature branch: `git checkout -b feature/your-feature`
2. Make your changes with tests
3. Commit: `git commit -m "Description of changes"`
4. Push: `git push origin feature/your-feature`
5. Open a Pull Request

## Adding a New Distribution

1. Create a file in `plugins/distributions/`
2. Inherit from `DistributionPlugin`
3. Implement the `fit()` method
4. Add tests in `tests/test_distributions.py`
5. Update documentation

## Questions?

Open an issue or start a discussion on GitHub.