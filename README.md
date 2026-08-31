# Jet classification with graph neural networks

This repository studies quark/gluon jet classification using CMS Open Data.
Each jet is represented as a graph: particle-flow candidates are graph nodes,
and the jet flavour is the graph-level target.

## Project status

The project is currently in **Stage 5: information-source analysis**. The
controlled full-feature paired benchmark found no consistent Dynamic EdgeConv
advantage over Fixed EdgeConv across five training seeds. Current work therefore
focuses on a traditional physics baseline and controlled feature ablations,
rather than assuming dynamic rewiring is beneficial.

See [`docs/PROJECT_PROGRESS.md`](docs/PROJECT_PROGRESS.md) for the canonical
project timeline, evidence, scientific decisions, and next steps. It must be
updated after every material experiment or research-direction change.

## Repository layout

```text
data/           Dataset documentation and local raw/processed data directories
docs/papers/    Project papers and background reading
docs/PROJECT_PROGRESS.md
                Canonical project timeline, results, decisions, and next steps
notebooks/      Exploratory analysis notebooks
scripts/        Reusable command-line utilities
QCDJetsMachineLearning/
                Upstream CERN example repository (Git submodule)
```

## Recommended workflow

1. Put one ROOT file in `data/raw/` as described in `data/README.md`.
2. Run `notebooks/01_explore_data.ipynb` for the fixed geometric k-NN baseline.
3. Run `notebooks/02_dynamicgraph.ipynb` for the dynamic EdgeConv model.

The controlled fixed and dynamic models are implemented in `jet_gnn.py` with
native PyTorch k-NN construction plus ordinary PyG `EdgeConv`. Both start from
the same `(dEta, dPhi)` geometric graph. Only the dynamic model rebuilds the
second-layer graph in latent space.

Run its unit tests from the repository root with:

```powershell
python -m unittest discover -s tests -v
```

## Reproducible command-line experiments

The experiment runner supports matched fixed and dynamic EdgeConv models and
the three controlled particle feature sets. For example:

```powershell
python scripts/train.py --model dynamic --features full --seed 1
```

The data split uses a separate, fixed split seed (default `42`) so changing the
training seed does not silently change the train/validation/test events. Every
run writes `config.json`, `metrics.json`, `model.pt`, and
`training_history.csv` below `results/runs/<experiment_id>/`, and appends its
summary to `results/metrics.csv`. Use `python scripts/train.py --help` for all
available controls. Training defaults to `--max-epochs 50 --patience 8`, uses
validation-loss early stopping, and restores the best validation checkpoint.


The ROOT dataset is intentionally not tracked by Git. See
[`data/README.md`](data/README.md) for its source and loading instructions.

## Data model

The main tree is `AK4jets/jetTree`. One entry represents one jet. Jet-level
branches contain scalar values, while `PF_*` branches contain variable-length
arrays with one value per particle-flow candidate.

## Setup

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements-lock.txt
```

`requirements.txt` lists the direct project dependencies;
`requirements-lock.txt` records the complete tested environment.

Inspect the ROOT file from the repository root with:

```powershell
python scripts/inspect_root.py
```

## Python environment

The project is pinned to Python 3.12 with modern TensorFlow/Keras and a
CUDA 13.0-enabled PyTorch stack. In VS Code, use `.venv\Scripts\python.exe` as
the notebook kernel.

```powershell
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

The PyTorch wheel includes its CUDA runtime, so a separate CUDA Toolkit is not
required for normal notebook use. A compatible NVIDIA driver is still required.
