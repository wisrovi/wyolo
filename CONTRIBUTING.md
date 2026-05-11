# Contributing to wyolo

First off, thank you for considering contributing to `wyolo`! It's people like you who make it a great tool for the community.

## 🛠 Development Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/wisrovi/wyoloservice2_worker.git
   cd wyoloservice2_worker
   ```

2. **Install dependencies:**
   We recommend using a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/macOS
   pip install -e ".[dev]"
   ```

3. **Install pre-commit hooks:**
   ```bash
   pre-commit install
   ```

## 🧪 Running Tests

We use `pytest` for testing. To run the tests with coverage:
```bash
pytest --cov=src/wyolo
```

## 📝 Coding Standards

- Follow [PEP 8](https://www.python.org/dev/peps/pep-0008/).
- Use `black` for formatting.
- Ensure all new features are covered by tests.
- Update documentation if you change any public APIs.

## 🚀 Pull Request Process

1. Create a new branch for your feature or bugfix.
2. Commit your changes with clear, descriptive messages.
3. Push to your fork and submit a pull request.
4. Ensure CI passes.

## ⚖️ Code of Conduct

Please be respectful and professional in all interactions. See [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md) for details.
