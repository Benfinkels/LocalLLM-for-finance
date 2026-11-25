# Contributing to LocalLLM-for-finance

Contributions are welcome from humans and AI-assisted workflows alike. Look for issues labeled `good first issue` or `help wanted`, or open a short issue if you have a new finance or ops benchmark track in mind.

Note on naming: the repository is `LocalLLM-for-finance`, and the installable Python package and CLI inside `src/` are named `when_not_to_llm` (`when-not-to-llm`).

## Local setup and offline tests

The core package has zero runtime dependencies and the test suite runs completely offline with no GPU or API keys.

```bash
pip install -e ".[dev]"
PYTHONPATH=src pytest -v
```

## What makes a PR easy to merge

- Keep each PR focused on one issue with a small, reviewable diff.
- New benchmark tasks in `src/when_not_to_llm/datasets.py` must be 100% synthetic, seeded for determinism, and paired with an exact ground-truth verifier in `verifiers.py` plus offline tests in `tests/test_suite.py`.
- Keep the core package zero-dependency (standard library only outside optional `charts`/`dev` extras).
- Never commit real financial data, company names, private IPs, or API keys.
- Include a short summary of what changed and your `pytest` output in the PR description.

## Using AI coding agents

AI-assisted PRs are fine as long as the diff stays tight, matches the issue scope, and passes the offline test suite without rewriting unrelated files.
