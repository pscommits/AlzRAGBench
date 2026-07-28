# Working on this repo

## Ground rules

1. **No magic numbers in `src/`.** Every path, model id, threshold, and `k` lives in `configs/`. If you're editing source to change behaviour, add a config key instead.
2. **`results/metrics.json` is generated, never hand-written.** If the README table and that file disagree, the README is wrong — regenerate.
3. **No data in git.** Everything under `data/raw/` and `data/processed/` is ignored. Document how to obtain it in `data/README.md`.
4. **`scripts/` is the reproducibility contract.** If a reviewer can't reproduce the results table with `prepare_data → build_index → build_kg → run_eval`, the repo isn't shippable.
5. **New retrieval or fusion strategy → new ablation row.** A change that isn't measured didn't happen.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m pytest tests/ -q
```

## Adding a modality

1. Write a `prepare_<name>()` in `scripts/prepare_data.py` emitting the unified schema.
2. Register it in `PREPARERS`.
3. Add the dataset block to `configs/default.yaml`.
4. Document access + licence in `data/README.md`.
5. Rebuild index and KG; rerun eval; update the README table.

## Tests

`tests/` covers metrics, config merging, and fusion — the three places where a silent bug would corrupt every number in the paper. Anything that touches those needs a test.
