# BWSI-Group1-SepsisDetection

Early sepsis prediction on the [PhysioNet/Computing in Cardiology Challenge 2019](https://physionet.org/content/challenge-2019/1.0.0/)
dataset, plus a small Streamlit app for demoing the trained models.

Each row of the dataset is one hour of one ICU patient's stay, and `SepsisLabel` marks
whether that patient-hour is labeled sepsis. The data ships as two hospitals
(`training_setA`, `training_setB`), which lets us ask whether a model trained at one
hospital still works at the other.

| Source | Model | Result |
| --- | --- | --- |
| `sepsis_random_forest_cross_hospital.ipynb` | Random Forest | 0.79 ROC AUC within hospital, 0.74 across hospitals |
| `GRU-D.ipynb` | GRU-D | 0.799 within hospital, 0.653 across (not deployed) |
| `temptransformersepsis-2.py` | Causal Temporal Transformer | 0.734 within hospital, 0.686 across |
| `CATBOOST_MEDLYTICS.ipynb` | CatBoost | **0.829 within hospital, 0.751 across** |

CatBoost is the strongest cross-hospital model of the four, and by a clear margin. It is
also the only pipeline where missingness genuinely carries signal: it reads the raw export
with gaps intact and learns from *which labs a clinician chose to order*, not just their
values.

The app serves the Random Forest, the Temporal Transformer and CatBoost. GRU-D and BRITS
appear in the dropdown as "coming soon", for different reasons:

- **GRU-D** is fully implemented and tested -- `load_grud` and `predict_grud` are wired in
  `model_utils.py`. It is greyed out only because no trained artifact ships here. Run the
  export cell at the end of `GRU-D.ipynb`, then delete the `coming_soon=True` line from its
  registry entry.
- **BRITS** has weights but is missing the metadata needed to serve them. See
  [BRITS](#brits-not-yet-servable) below.

## Getting the data

**The CSVs are not in this repo** -- they total roughly 750 MB, which is past what GitHub
will accept comfortably. `data/` is gitignored. You need to rebuild it locally.

We chose gitignore over Git LFS deliberately:

- GitHub's free LFS tier is 1 GB of storage and 1 GB/month of bandwidth. At ~750 MB per
  clone, two teammates cloning in a month exhausts the quota and pushes fail until someone pays.
- These files are derived artifacts, not source. They are reproducible from PhysioNet with
  the script below, so version-controlling them buys nothing that a documented recipe doesn't.
- PhysioNet asks users to get the data from PhysioNet under its own terms. Redistributing a
  repackaged copy through our repo sidesteps that, which we'd rather not do.

### 1. Download from PhysioNet

Grab `training_setA.zip` and `training_setB.zip` from the
[Challenge 2019 files page](https://physionet.org/content/challenge-2019/1.0.0/) and unzip
them. Each contains one pipe-separated `.psv` file per patient (`p000001.psv`, ...).

### 2. Build the combined CSVs

The notebooks read four files from `data/`:

| File | Columns | Used by |
| --- | --- | --- |
| `combined_training_setA.csv` | 41 | Random Forest |
| `combined_training_setB.csv` | 41 | Random Forest |
| `combinedA_patient_id_cleaned.csv` | 32 | GRU-D |
| `combinedB_patient_id_cleaned.csv` | 32 | GRU-D |

The `combined_*` files are every patient's `.psv` concatenated top to bottom, in filename
order, with a single header row. Patient order matters: the Random Forest notebook recovers
patient boundaries by watching for `ICULOS` resetting to a lower value, which only works if
each patient's hours stay contiguous and in sequence.

```python
from pathlib import Path
import pandas as pd

for letter in ("A", "B"):
    files = sorted(Path(f"training_set{letter}").glob("*.psv"))
    frames = [pd.read_csv(f, sep="|") for f in files]
    pd.concat(frames, ignore_index=True).to_csv(
        f"data/combined_training_set{letter}.csv", index=False
    )
    print(letter, len(files), "patients")
```

The `*_patient_id_cleaned.csv` files are the same data with three changes: an explicit
`patient_id` column instead of inferred boundaries, ten very sparse columns dropped
(`EtCO2`, `SaO2`, `AST`, `Alkalinephos`, `Bilirubin_direct`, `Magnesium`,
`Bilirubin_total`, `TroponinI`, `PTT`, `Fibrinogen`), and the remaining gaps filled with a
constant per column. GRU-D recovers the missingness mask at load time by detecting those
constants, so keep the fill values constant per column rather than interpolating.

### 3. Place them

```
data/
  combined_training_setA.csv
  combined_training_setB.csv
  combinedA_patient_id_cleaned.csv
  combinedB_patient_id_cleaned.csv
```

## Running the demo app

```bash
pip install -r requirements-dev.txt
```

`requirements.txt` holds only what the deployed app needs; `requirements-dev.txt` adds
matplotlib, seaborn and Jupyter for the notebooks and training scripts.

The app serves models from `models/`, which **is** committed so cloud deploys have
something to load. Rebuild any artifact by running its notebook through to the export cell
at the bottom:

- `sepsis_random_forest_cross_hospital.ipynb` -> `models/random_forest_sepsis.joblib` (6.9 MB)
- `temptransformersepsis-2.py` -> `models/temporal_transformer_sepsis.pt` (section 8b, 2.3 MB)
- `CATBOOST_MEDLYTICS.ipynb` -> `models/catboost_sepsis.cbm` (Block 16, 0.5 MB)
- `GRU-D.ipynb` -> `models/grud_sepsis.pt` (not currently built)

The transformer script is a Colab export containing two separate pipelines. Only the
first one (the Causal Temporal Transformer, with focal loss and engineered features) is
wired into the app; the second, 4-quadrant pipeline still runs but exports nothing.
Running the whole file trains both, which takes a while -- the feature engineering pass
alone is slow because it does per-patient rolling and elapsed-time computations over
~790k rows.

Then:

```bash
streamlit run app.py
```

The app starts fine with an empty `models/` folder -- it just tells you which export cell to
run. Fill in vitals and labs, pick a model, and it returns a positive/negative call plus the
probability. The decision threshold is adjustable in the sidebar, since at a 0.5 cutoff both
models miss most sepsis hours.

**GRU-D and the single-snapshot form.** GRU-D was trained on hourly sequences, not one
moment in time. The form collects a single set of values and repeats it across every
timestep, so the model sees a patient whose vitals never move. That is a demo convenience,
not a realistic input, and its output should not be read as comparable to the notebook's
reported ROC AUC.

The Temporal Transformer carries the same caveat, and one more: it was trained on
engineered inputs (3-hour deltas, 6-hour rolling min/max/mean, hours-since-measurement,
missingness flags). The app derives all of those from the single snapshot by assuming
nothing changed and every value was measured this hour, which is the most favourable
possible reading of the input. Its sequence length is also capped at 48 hours, because
that is the positional-encoding window it was built with.

**CatBoost and the lab-draw control.** CatBoost learned from 13 `*_tested` flags marking
which labs were drawn each hour. Labs are ordered rarely, so in training those flags are
usually 0 -- and when they are 1, it generally means a clinician was already concerned.
Ticking every box would therefore push the score up for reasons that have nothing to do
with the values you typed. The sidebar defaults to none drawn, which reads your entries as
carried forward from an earlier draw. That is the most common pattern in the training data,
so it is the most in-distribution default rather than merely the most conservative one.

Two notes on the CatBoost pipeline worth knowing. Its input is rebuilt locally rather than
read from a committed file: the Colab original read `combinedA_patient_id.csv`, which is the
raw hourly export plus a `patient_id` column, and the load block reconstructs exactly that
by joining `patient_id` from the `_cleaned` CSV onto the raw one (same rows, same order,
asserted at load). And its per-patient fill step was rewritten -- the original
`groupby('patient_id').apply(...)` silently drops the grouping column on pandas 2.2+, which
made the next block fail with `KeyError: 'patient_id'`. The `groupby.ffill()/bfill()` form
now used does the same fill, keeps the column, and is faster.

This is coursework. None of these models is validated for clinical use.

## Deploying to Streamlit Community Cloud

Point Streamlit Cloud at this repo with `app.py` as the entry point. Three things about
this project make a naive deploy fail, and all three are already handled:

**A note on the Random Forest size.** The deployed forest uses `min_samples_leaf=200`
rather than the `3` used in the notebook's experiments. This was originally a size fix --
`leaf=3` grows 3.4M nodes and a 75 MB artifact -- but it turned out to improve the result
as well:

| `min_samples_leaf` | Artifact | Cross-hospital AUC |
| --- | --- | --- |
| 3 | 74.7 MB | 0.7194 |
| 50 | 18.9 MB | 0.7365 |
| 200 | 6.9 MB | 0.7373 |

Tiny leaves let each tree memorize Hospital A. Constraining them costs nothing within
hospital and transfers noticeably better across hospitals, which is the effect this whole
project is about. The export cell reports both numbers so the comparison stays visible.

**Model files must be committed.** Streamlit Cloud builds from GitHub and has no access
to your local `models/` folder. That directory is deliberately *not* gitignored -- if the
artifacts aren't in the repo, the deployed app has nothing to serve and every model shows
a "no saved model in this deployment" warning. Keep the artifacts small; anything near
50 MB makes GitHub complain and slows every redeploy.

**PyTorch must be the CPU build.** On Linux, a bare `torch` requirement resolves to the
CUDA build (~2.5 GB) and exceeds the resource limit. `requirements.txt` pins
`torch==2.13.0+cpu` against PyTorch's own index, which is the fix. If the build ever
fails to resolve that exact version, drop the `==...` but keep the `--extra-index-url`
line. This is the most common cause of a failed Streamlit deploy for this kind of project.

**The data does not go up, and does not need to.** The app only loads model artifacts;
nothing in `app.py` or `model_utils.py` reads a CSV. The 750 MB of PhysioNet data is a
training-time dependency only.

Cold start pulls PyTorch, so the first load after a deploy takes a few minutes. Redeploys
are faster. If you are demoing live, open the app once beforehand to warm it up.

## BRITS: not yet servable

`brits_best.pt` is a bare `state_dict`. The architecture is fully recoverable from it and
is reconstructed in `brits_model.py`, which loads the checkpoint with `strict=True` and
no missing or unexpected keys: input size 33, hidden size 64, two RITS directions, and a
64 -> 64 -> 1 classifier head.

What the file does not contain, and what has to come from the original training code
before the app can serve it:

1. **Which 33 features, in what order.** Without the exact list there is no way to build
   an input row. (The 34 PhysioNet physiological variables minus EtCO2 also happens to
   equal 33, but that is arithmetic, not evidence.)
2. **Whether inputs were standardized, and with what mean/std.** Every other sequence
   model here was, so it probably was too, but the statistics are gone.
3. **How the forward and backward hidden states combine.** The classifier takes 64
   features, not 128, so it is not a concatenation -- it is a mean, a sum, or
   forward-only, and the three give different answers.

Guessing any of these produces a model that outputs confident, plausible-looking
probabilities that mean nothing. BRITS therefore stays greyed out in the dropdown until
the training script turns up. Once it does, filling in `brits_model.py` and adding a
loader plus predictor to `model_utils.py` is the same three-step recipe as any other model.

## Adding a model

`app.py` contains no model-specific code. It renders a form from `model_utils.FEATURE_SPEC`
and dispatches through `model_utils.MODEL_REGISTRY`, so a new model is three edits, all in
`model_utils.py`:

1. **A loader** -- takes a `Path`, returns whatever object you want to keep around.
2. **A predictor** -- takes `(bundle, values, options)` where `values` is
   `{feature_name: float}`, and returns a `Prediction`.
3. **A registry entry** -- flip `coming_soon` off on the existing CatBoost or Temporal
   Transformer stub and point it at your artifact:

```python
"catboost": ModelEntry(
    key="catboost",
    display_name="CatBoost",
    blurb="Gradient-boosted trees over a single snapshot.",
    artifact=MODELS_DIR / "catboost_sepsis.cbm",
    loader=load_catboost,
    predictor=predict_catboost,
    export_hint="Run the export cell at the end of the CatBoost notebook.",
),
```

Set `sequence_model=True` if it consumes hourly sequences -- that turns on the sequence
length control and the simplification caption automatically.

If your model needs an input the form doesn't collect yet, add a `FeatureSpec` to
`FEATURE_SPEC` and the form grows a field for it. Have your export cell save the feature
list and any scaling stats *inside* the artifact rather than hardcoding them in the app;
both existing models do this, which is why the app never needs to know a column order.
