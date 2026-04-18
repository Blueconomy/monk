# Contributing to monk

Thanks for wanting to improve monk. Contributions are welcome — from new detectors to parser support to docs.

## Quick start

```bash
git clone https://github.com/blueconomy-ai/monk
cd monk
pip install -e ".[dev]"
pytest tests/ -v
```

## Adding a new detector

1. Create `monk/detectors/your_detector.py` extending `BaseDetector`:

```python
from monk.detectors.base import BaseDetector, Finding
from monk.parsers.auto import TraceCall

class YourDetector(BaseDetector):
    name = "your_detector"

    def run(self, calls: list[TraceCall]) -> list[Finding]:
        findings = []
        # your logic here
        return findings
```

2. Register it in `monk/detectors/__init__.py`:

```python
from .your_detector import YourDetector
ALL_DETECTORS = [..., YourDetector()]
```

3. Add tests in `tests/test_detectors.py`.

4. Open a PR with a description of what pattern it detects and why it matters.

## Adding a trace format parser

Add a parser in `monk/parsers/auto.py` following the existing patterns (`_is_X_format` + `_parse_X`).

## Updating model pricing

Edit `monk/pricing.py`. Prices are in USD per 1M tokens.

## Code style

- Python 3.9+, type hints throughout
- No external dependencies beyond `click` and `rich`
- Keep detectors stateless and fast

## Reporting issues

Open a GitHub issue with:
- The trace format you're using
- What monk did (or didn't) detect
- A minimal anonymised example if possible
