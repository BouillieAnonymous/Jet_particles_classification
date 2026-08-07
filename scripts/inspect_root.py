from pathlib import Path

import uproot

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
PREFERRED_ROOT_PATH = REPOSITORY_ROOT / "data" / "raw" / "JetNtuple_RunIISummer16_13TeV_MC_1.root"
LEGACY_ROOT_PATH = REPOSITORY_ROOT / "JetNtuple_RunIISummer16_13TeV_MC_1.root"
ROOT_PATH = PREFERRED_ROOT_PATH if PREFERRED_ROOT_PATH.exists() else LEGACY_ROOT_PATH

f = uproot.open(ROOT_PATH)
classnames = f.classnames(recursive=True)   # {name: ROOT classname} — digs into every subdirectory, not just the top level

# takes objects of a certain type (Tree or RNTuple)
tree_candidates = [k for k, cls in classnames.items() if "Tree" in cls or "RNTuple" in cls]

# function to extract the cycle number (version)
def cycle_num(key):
    return int(key.rsplit(";", 1)[1]) if ";" in key else 0

# the goal is to take the highest cycle (newest version)
tree_candidates.sort(key=cycle_num, reverse=True)   
tree = f[tree_candidates[0]]

print(f"Using: {tree_candidates[0]}")
print(tree.keys())   # branch/features names
