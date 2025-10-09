# 🧠 ExplainBench: Benchmarking XAI Methods on Time-Series Data

ExplainBench is a **modular and extensible benchmarking framework** for evaluating explainability methods on **time series models** across diverse datasets and metrics.

## 🚀 Features

- **Extensible**: Add new datasets, models, explainers, metrics via a simple registry.
- **Reproducible**: YAML-configured runs, fixed seeds, deterministic loaders where possible.
- **Visual**: Attribution plots and tabular reports.
- **Batteries included**: Synthetic dataset, LSTM & Transformer baselines, Grad/IG/LIME explainers, Faithfulness/Consistency/Stability metrics.

## 🗂️ Structure

```
explain-bench/
├── explainbench/
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
python scripts/run_benchmark.py --config explainbench/configs/default.yaml
python scripts/visualize_results.py --input results/benchmark_results.csv
```

## 🧱 Add New Components

See docstrings in `explainbench/utils/registry.py` and base classes in each submodule.

## 📜 License

MIT
