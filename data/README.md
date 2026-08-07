# Loading the jet dataset

## 1. Download the dataset

Dataset: [CMS Open Data, record 12100](https://opendata.cern.ch/record/12100)
("Jet flavour content" — quark/gluon jets from simulated 13 TeV collisions).

The record page lists several `JetNtuple_RunIISummer16_13TeV_MC_*.root` files.
You should download exactly one (e.g. `JetNtuple_RunIISummer16_13TeV_MC_1.root`,
~85 MB)! Each file already has thousands of jets, which is more than enough for 
the purposes of this project.

To give a bit of context: `.root` is a binary format developed at CERN
because storing petabytes of collision data needed something standard
formats like CSV/JSON weren't built for. Unlike those, it lets you skip
columns and store a different number of values per row without padding.

To load data from `.root` file, you'll need to install uproot and/or awkward:

```bash
pip install uproot awkward
```


## 2. Opening the file

A ROOT file is a generic container, closer to a mini filesystem than a single
table. Here's what to know about how the data's laid out, so you can find
and load the right table:

1. **The event table is usually nested in a subdirectory.** In this file it's
   at `AK4jets/jetTree`, not at the top level.
2. **ROOT files can hold multiple versions ("cycles") of the same object**,
   shown as `jetTree;2`, `jetTree;3`, etc. You want the newest version.
3. **Not everything you find inside is the table you want.** `AK4jets` itself
   is a directory, not data. Check what an object actually is before trying
   to read it as a table.

The safest way to handle all three: look at each object's actual classname
(rather than guessing from its name) and just keep the ones that are tables.
`TTree` is the classic ROOT table format; `RNTuple` is a newer, faster
replacement some recent files use instead — checking for both means the
same code works either way, without you having to know in advance which one
a given file uses:

```python
import uproot

ROOT_PATH = "./data/raw/JetNtuple_RunIISummer16_13TeV_MC_1.root"

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
```

In other words: instead of writing `f["AK4jets/jetTree;3"]` directly (which
only works for this exact file), the code above figures out the tree's name,
location, and cycle number on its own. So if you ever open a different
`.root` file where the tree is named differently, sits in a different
subdirectory, or is on a different cycle, this same code still finds it.

## 3. Reading the branches — figure out which ones you need

The tree has ~65 branches, a mix of jet-level (one value per jet) and
per-particle ones (prefixed `PF_*`, one value *per particle in the jet*).
Each branch is a candidate feature, and figuring out which ones you actually
need is on you — cross-reference `tree.keys()` (Section 2) against the
[record's own field documentation](https://opendata.cern.ch/record/12100),
which describes what each branch means, and decide which ones belong in
your feature vector to build the point-cloud representation. Once you've
picked your features:

```python
import awkward as ak

N_JETS = 5000   # cap the read — don't pull the whole file at once

arrays = tree.arrays(
    [...],   # your chosen fetures: list of strings
    entry_stop=N_JETS,
    library="ak",
)
```

Per-particle branches vary in length jet to jet, so `uproot` hands them back
as an `awkward` array, not numpy. You should work with that instead of forcing a fixed
shape yet. There are couple of things to have in mind:

- Not every saved particle actually belongs to the jet — there's a flag for
  that, and skipping it leaks outside particles into your point cloud.
- The flavor label isn't a clean binary split — decide how to handle jets
  that are neither quark nor gluon before building `labels`.

End goal: a plain Python list of `(n_particles_i, n_features)` numpy arrays,
one per jet —> a variable-length point cloud, ready for padding/masking.
