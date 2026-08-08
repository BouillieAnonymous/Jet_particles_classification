# Jet classification with graph neural networks

This repository studies quark/gluon jet classification using CMS Open Data.
Each jet is represented as a graph: particle-flow candidates are graph nodes,
and the jet flavour is the graph-level target.

## Repository layout

```text
data/           Dataset documentation and local raw/processed data directories
docs/papers/    Project papers and background reading
notebooks/      Exploratory analysis notebooks
scripts/        Reusable command-line utilities
QCDJetsMachineLearning/
                Upstream CERN example repository (Git submodule)
```

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

The project is pinned to Python 3.12 with modern TensorFlow/Keras and PyTorch
stacks. In VS Code, use `.venv\Scripts\python.exe` as the notebook kernel.

```powershell
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```
