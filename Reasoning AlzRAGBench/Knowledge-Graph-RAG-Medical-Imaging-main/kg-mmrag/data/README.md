# Datasets

**No data is committed to this repository.** Everything in `data/raw/` and `data/processed/` is gitignored. This document is how you or a reviewer reconstructs the data from scratch.

Expected layout after running `scripts/prepare_data.py`:

```
data/
├── raw/
│   ├── slake/
│   ├── brats/
│   └── rocov2/
└── processed/
    ├── slake/
    │   ├── pairs.jsonl        # {image_path, caption, question, answer, entities}
    │   └── manifest.json
    ├── brats/
    └── rocov2/
```

---

## SLAKE

Bilingual radiology VQA dataset with semantic annotations and an accompanying knowledge base — the KB is why it's useful here, not just the VQA pairs.

- **Access:** _(TODO: add download URL / HF dataset ID)_
- **Licence:** _(TODO: confirm — check terms before redistributing anything derived from it)_
- **Used for:** dense index, KG seed entities, VQA evaluation
- **Splits:** train / val / test
- **Size after processing:** _(TODO: N image–text pairs)_

## BraTS

Multimodal brain tumour MRI segmentation dataset (T1, T1ce, T2, FLAIR).

- **Access:** _(TODO: registration is required — note the year/edition you used, e.g. BraTS 2021)_
- **Licence:** _(TODO: research use — confirm terms)_
- **Used for:** MRI modality coverage, anatomical KG grounding
- **Preprocessing:** volumes are sliced to 2D; see `scripts/prepare_data.py`
- **Size after processing:** _(TODO: N slices / N volumes)_

## ROCOv2

Radiology Objects in COntext, version 2 — large-scale radiology image–caption pairs with concept annotations.

- **Access:** _(TODO: add URL / HF dataset ID)_
- **Licence:** _(TODO)_
- **Used for:** bulk dense-index coverage across modalities
- **Size after processing:** _(TODO: N image–caption pairs)_

---

## Knowledge Graph Source

The KG is built by `scripts/build_kg.py` from:

- _(TODO: e.g. SLAKE's bundled knowledge base)_
- _(TODO: e.g. UMLS / RadLex concept mappings, if used — note that UMLS requires a licence)_

Resulting graph statistics:

| Property | Count |
|---|---|
| Nodes (entities) | — |
| Edges (relations) | — |
| Relation types | — |
| Avg. degree | — |

> **TODO(Nathan):** fill these in from `artifacts/graphs/stats.json` once the KG builds.

---

## Reproducing

```bash
# Download and place raw data per the instructions above, then:
python scripts/prepare_data.py --config configs/default.yaml
```

This writes `processed/*/pairs.jsonl` in a uniform schema so downstream code never has to know which dataset a record came from:

```json
{
  "id": "slake_00123",
  "image_path": "data/raw/slake/imgs/xmlab1/source.jpg",
  "modality": "xray",
  "caption": "Chest X-ray showing ...",
  "question": "What organ is abnormal?",
  "answer": "Lung",
  "entities": ["lung", "consolidation"],
  "split": "train"
}
```

## Ethical & Licensing Notes

All datasets are de-identified public research datasets. No patient-identifiable information is present or handled. Dataset licences are independent of this repository's MIT licence — check each dataset's terms before redistribution or commercial use.
