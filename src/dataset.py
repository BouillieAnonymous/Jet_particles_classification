"""ROOT loading and graph construction for jet experiments."""

from __future__ import annotations

from pathlib import Path

import awkward as ak
import numpy as np
import torch
import uproot
from torch_geometric.data import Data

from src.features import feature_names

DATA_FILENAME = "JetNtuple_RunIISummer16_13TeV_MC_1.root"
REQUIRED_BRANCHES = (
    "run",
    "lumi",
    "event",
    "physFlav",
    "isPhysUDS",
    "isPhysG",
    "jetPt",
    "PF_pT",
    "PF_dEta",
    "PF_dPhi",
    "PF_fromAK4Jet",
)


def find_root_path(repository_root: Path, requested: Path | None = None) -> Path:
    """Resolve an explicit dataset path or find the project's current ROOT file."""
    if requested is not None:
        path = requested.expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"ROOT file does not exist: {path}")
        return path

    candidates = (
        repository_root / "data" / "raw" / DATA_FILENAME,
        repository_root / "QCDJetsMachineLearning" / DATA_FILENAME,
    )
    for path in candidates:
        if path.is_file():
            return path.resolve()
    searched = "\n".join(f"  - {path}" for path in candidates)
    raise FileNotFoundError(f"Could not find {DATA_FILENAME}. Searched:\n{searched}")


def _latest_tree(root_file: uproot.ReadOnlyDirectory):
    candidates = [
        key
        for key, class_name in root_file.classnames(recursive=True).items()
        if "Tree" in class_name or "RNTuple" in class_name
    ]
    if not candidates:
        raise ValueError("The ROOT file contains no TTree or RNTuple")

    def cycle_number(key: str) -> int:
        return int(key.rsplit(";", 1)[1]) if ";" in key else 0

    return root_file[max(candidates, key=cycle_number)]


def load_jet_graphs(
    root_path: Path,
    feature_set: str,
    max_entries: int | None = None,
) -> list[Data]:
    """Load selected UDS/gluon jets using one common validity selection.

    The validity cuts deliberately use all features in the current full feature
    set. This keeps the selected jets identical across feature ablations.
    """
    names = feature_names(feature_set)
    with uproot.open(root_path) as root_file:
        tree = _latest_tree(root_file)
        missing = sorted(set(REQUIRED_BRANCHES) - set(tree.keys()))
        if missing:
            raise KeyError(f"Missing required ROOT branches: {missing}")
        raw_data = tree.arrays(
            list(REQUIRED_BRANCHES), entry_stop=max_entries, library="ak"
        )

    selected = raw_data[
        (raw_data["isPhysUDS"] == 1) ^ (raw_data["isPhysG"] == 1)
    ]
    graphs: list[Data] = []
    for jet in selected:
        pt_all = ak.to_numpy(jet["PF_pT"]).astype(np.float32, copy=False)
        d_eta_all = ak.to_numpy(jet["PF_dEta"]).astype(np.float32, copy=False)
        d_phi_all = ak.to_numpy(jet["PF_dPhi"]).astype(np.float32, copy=False)
        in_jet = ak.to_numpy(jet["PF_fromAK4Jet"]) == 1
        jet_pt = float(jet["jetPt"])

        valid = in_jet & np.isfinite(pt_all) & (pt_all > 0)
        valid &= np.isfinite(d_eta_all) & np.isfinite(d_phi_all)
        if not np.isfinite(jet_pt) or jet_pt <= 0:
            continue

        pt = pt_all[valid]
        d_eta = d_eta_all[valid]
        d_phi = d_phi_all[valid]
        if pt.size < 2:
            continue

        columns = {
            "dEta": d_eta,
            "dPhi": d_phi,
            "log_pT": np.log(pt),
            "log_pT_over_jetPt": np.log(pt / jet_pt),
        }
        matrix = np.column_stack([columns[name] for name in names]).astype(
            np.float32, copy=False
        )
        finite_rows = np.isfinite(matrix).all(axis=1)
        matrix = matrix[finite_rows]
        positions = np.column_stack((d_eta, d_phi))[finite_rows].astype(
            np.float32, copy=False
        )
        if matrix.shape[0] < 2:
            continue

        graphs.append(
            Data(
                x=torch.from_numpy(matrix),
                pos=torch.from_numpy(positions),
                y=torch.tensor([int(jet["isPhysG"])], dtype=torch.long),
                original_flavor=torch.tensor(
                    [int(jet["physFlav"])], dtype=torch.long
                ),
                run=torch.tensor([int(jet["run"])], dtype=torch.long),
                lumi=torch.tensor([int(jet["lumi"])], dtype=torch.long),
                event=torch.tensor([int(jet["event"])], dtype=torch.long),
                event_id=torch.tensor([int(jet["event"])], dtype=torch.long),
            )
        )
    if not graphs:
        raise ValueError("No valid UDS/gluon jet graphs were constructed")
    return graphs
