# 🧠 ExplainBench: Benchmarking XAI Methods on Time-Series Data

ExplainBench is a **modular and extensible benchmarking framework** for evaluating explainability methods on **time series models** across diverse datasets and metrics.

## 🚀 Features

- **Extensible**: Add new datasets, models, explainers, metrics via a simple registry.
- **Reproducible**: YAML-configured runs, fixed seeds, deterministic loaders where possible.
- **Visual**: Attribution plots and tabular reports.
- **Batteries included**: Synthetic dataset, LSTM & Transformer baselines, Grad/IG/LIME explainers, Faithfulness/Consistency/Stability/AUROC metrics.

## 🗂️ Structure

```
explain-bench/
├── src/
│   ├── configs/          # YAML configs
│   ├── datasets/         # loaders: synthetic, UCR (stub)
│   ├── models/           # LSTM, Transformer
│   ├── explanations/     # Grad, IG, LIME (simple)
│   ├── metrics/          # Faithfulness, Consistency, Stability
│   ├── visualization/    # plotting & reporting
│   ├── utils/            # registry, config, logging
│   ├── benchmark.py      # core benchmark loop
│   ├── runner.py         # orchestrator
├── scripts/              # CLI scripts
├── tests/                # basic sanity tests
├── README.md
├── requirements.txt
├── pyproject.toml
└── setup.py
```

## ⚙️ Install

```bash
pip install -e .
# or
pip install -r requirements.txt && python setup.py develop
```

## 🧠 Quickstart

```bash
python scripts/run_benchmark.py --config configs/default.yaml
#python scripts/visualize_results.py --input results/benchmark_results.csv
```

## 🧱 Add New Components

See docstrings in `src/utils/registry.py` and base classes in each submodule.

## 📜 License

MIT


# Appendix

## Datasets Overview

This page summarizes the synthetic and real‑world datasets used in **Explain‑Bench**, and tracks their implementation status and references.

---

### 📊 Datasets at a Glance

> (SeqComb‑UV = univariate; SeqComb‑MV = multivariate.)

| Dataset           | # of Samples | Length | Dimension | Classes | Task                       |
|-------------------|:------------:|:------:|:---------:|:-------:|----------------------------|
| **State**         | 1000         | 200    | 3         | 2       | Binary classification      |
| **Switch-Feature**| 1000         | 100    | 3         | 2       | Binary classification      |
| **FreqShapes**    | 6,100        | 50     | 1         | 4       | Multi‑classification       |
| **SeqComb‑UV**    | 6,100        | 200    | 1         | 4       | Multi‑classification       |
| **SeqComb‑MV**    | 6,100        | 200    | 4         | 4       | Multi‑classification       |
| **LowVar**        | 6,100        | 200    | 2         | 4       | Multi‑classification       |
| **ECG**           | 92,511       | 360    | 1         | 5       | ECG classification         |
| **PAM**           | 5,333        | 600    | 17        | 8       | Action recognition         |
| **Epilepsy**      | 11,500       | 178    | 1         | 2       | EEG classification         |
| **Boiler**        | 160,719      | 36     | 20        | 2       | Mechanical fault detection |
| **Wafer**         | 7,164        | 152    | 1         | 2       | Sensor classification      |
| **FreezerRegular**| 3,000        | 301    | 1         | 2       | Sensor classification      |
| **Water**         | 573          | 168    | 13        | 2       | Binary classification      |

---

### ✅ Dataset Tracking

> Update **Status**, **Web Source**, and **Reference** as you wire things up.  
> Use ✅ for implemented, 🧪 for in‑progress, and ⏳ for planned.

| Dataset | Status | Web Source | Reference |
|--------|:------:|------------|-----------|
| State |  🧪 In progress | [link][State-src] | [ref][State-ref] |
| Switch-Feature | 🧪 In progress | [link][Switch-src] | [ref][Switch-ref] |
| FreqShapes | ✅ Implemented | [link][freqshape-src] | [ref][freqshape-ref] |
| SeqComb‑UV | 🧪 In progress | [link][seqcombuv-src] | [ref][seqcombuv-ref] |
| SeqComb‑MV | 🧪 In progress | [link][seqcombmv-src] | [ref][seqcombmv-ref] |
| LowVar     | 🧪 In progress | [link][lowvar-src]    | [ref][lowvar-ref]    |
| ECG        | 🧪 In progress | [link][ecg-src]       | [ref][ecg-ref]       |
| PAM        | 🧪 In progress | [link][pam-src]       | [ref][pam-ref]       |
| Epilepsy   | 🧪 In progress | [link][epilepsy-src]  | [ref][epilepsy-ref]  |
| Boiler     | 🧪 In progress | [link][boiler-src]    | [ref][boiler-ref]    |
| Wafer      | ⏳ Planned     | [link][wafer-src]     | [ref][wafer-ref]     |
| FreezerRegular | ⏳ Planned | [link][freezer-src]   | [ref][freezer-ref]   |
| Water      | ⏳ Planned     | [link][water-src]     | [ref][water-ref]     |

<!-- Replace the placeholders below with real URLs/citations -->
[State-src]: https://github.com/visriv/explain-bench/blob/main/src/datasets/datagen/state_data.py#
[Switch-src]: https://github.com/visriv/explain-bench/blob/main/src/datasets/datagen/switch_data.py#
[freqshape-src]: https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/B0DEQJ# 
[seqcombuv-src]: https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/B0DEQJ#
[seqcombmv-src]: https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/B0DEQJ#
[lowvar-src]: https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/B0DEQJ#
[ecg-src]: https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/B0DEQJ#
[pam-src]: https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/B0DEQJ#
[epilepsy-src]: https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/B0DEQJ#
[boiler-src]: https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/B0DEQJ#
[wafer-src]: #
[freezer-src]: #
[water-src]: #

[State-ref]: #
[Switch-ref]: #
[freqshape-ref]: #
[seqcombuv-ref]: #
[seqcombmv-ref]: #
[lowvar-ref]: #
[ecg-ref]: #
[pam-ref]: #
[epilepsy-ref]: #
[boiler-ref]: #
[wafer-ref]: #
[freezer-ref]: #
[water-ref]: #

