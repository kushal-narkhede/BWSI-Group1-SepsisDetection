"""Pluggable model registry for the sepsis prediction demo.

The app never imports a model library directly. It asks this module for
``MODEL_REGISTRY``, calls ``load_model(key)`` to get a bundle, and calls
``predict(key, bundle, values)`` to get a ``Prediction``. Everything a model needs
to know about its own preprocessing lives in the artifact it exports, not in the app.

Adding a model is three things:
  1. a loader   -- read the artifact off disk, return whatever object you like
  2. a predictor -- take that object plus a {feature_name: value} dict, return a Prediction
  3. one ``ModelEntry`` in MODEL_REGISTRY below

If the new model needs an input the form does not collect yet, add a ``FeatureSpec``
to FEATURE_SPEC and the form grows a field for it automatically.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Sequence

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent
MODELS_DIR = PROJECT_ROOT / "models"


# --------------------------------------------------------------------------------------
# Form / feature definitions
# --------------------------------------------------------------------------------------
# One entry per input any model might need. This is the union across all models -- each
# model picks the subset it was trained on. Defaults are the Hospital A (PhysioNet
# training_setA) medians, so an untouched form represents a roughly typical ICU hour.

VITALS = "Vital signs"
CHEMISTRY = "Blood gas & chemistry"
HEMATOLOGY = "Hematology"
CONTEXT = "Demographics & admission"

GROUP_ORDER = [VITALS, CHEMISTRY, HEMATOLOGY, CONTEXT]


@dataclass(frozen=True)
class FeatureSpec:
    name: str
    label: str
    unit: str
    default: float
    min_value: float
    max_value: float
    step: float
    group: str
    help: str = ""
    binary: bool = False


FEATURE_SPEC: list[FeatureSpec] = [
    # --- vitals ---
    FeatureSpec("HR", "Heart rate", "bpm", 84.0, 20.0, 250.0, 1.0, VITALS),
    FeatureSpec("O2Sat", "Pulse oximetry (SpO2)", "%", 98.0, 50.0, 100.0, 1.0, VITALS),
    FeatureSpec("Temp", "Temperature", "deg C", 37.06, 25.0, 43.0, 0.1, VITALS),
    FeatureSpec("SBP", "Systolic BP", "mmHg", 118.5, 40.0, 300.0, 1.0, VITALS),
    FeatureSpec("MAP", "Mean arterial pressure", "mmHg", 77.0, 20.0, 200.0, 1.0, VITALS),
    FeatureSpec("DBP", "Diastolic BP", "mmHg", 58.5, 20.0, 200.0, 1.0, VITALS),
    FeatureSpec("Resp", "Respiration rate", "breaths/min", 18.0, 4.0, 60.0, 1.0, VITALS),
    # --- blood gas & chemistry ---
    FeatureSpec("BaseExcess", "Base excess", "mmol/L", 0.0, -30.0, 30.0, 0.1, CHEMISTRY),
    FeatureSpec("HCO3", "Bicarbonate", "mmol/L", 24.0, 5.0, 50.0, 0.1, CHEMISTRY),
    FeatureSpec("FiO2", "Fraction of inspired O2", "fraction", 0.5, 0.21, 1.0, 0.01, CHEMISTRY),
    FeatureSpec("pH", "pH", "", 7.39, 6.8, 7.8, 0.01, CHEMISTRY),
    FeatureSpec("PaCO2", "Arterial CO2 partial pressure", "mmHg", 40.0, 10.0, 120.0, 1.0, CHEMISTRY),
    FeatureSpec("BUN", "Blood urea nitrogen", "mg/dL", 18.0, 1.0, 200.0, 1.0, CHEMISTRY),
    FeatureSpec("Calcium", "Calcium", "mg/dL", 8.3, 1.0, 20.0, 0.1, CHEMISTRY),
    FeatureSpec("Chloride", "Chloride", "mmol/L", 106.0, 70.0, 140.0, 1.0, CHEMISTRY),
    FeatureSpec("Creatinine", "Creatinine", "mg/dL", 0.9, 0.1, 20.0, 0.1, CHEMISTRY),
    FeatureSpec("Glucose", "Glucose", "mg/dL", 124.0, 20.0, 600.0, 1.0, CHEMISTRY),
    FeatureSpec("Lactate", "Lactate", "mmol/L", 1.8, 0.1, 30.0, 0.1, CHEMISTRY),
    FeatureSpec("Phosphate", "Phosphate", "mg/dL", 3.4, 0.5, 15.0, 0.1, CHEMISTRY),
    FeatureSpec("Potassium", "Potassium", "mmol/L", 4.1, 1.5, 9.0, 0.1, CHEMISTRY),
    # --- hematology ---
    FeatureSpec("Hct", "Hematocrit", "%", 30.2, 10.0, 60.0, 0.1, HEMATOLOGY),
    FeatureSpec("Hgb", "Hemoglobin", "g/dL", 10.4, 3.0, 20.0, 0.1, HEMATOLOGY),
    FeatureSpec("WBC", "White blood cell count", "10^3/uL", 10.8, 0.1, 100.0, 0.1, HEMATOLOGY),
    FeatureSpec("Platelets", "Platelets", "10^3/uL", 181.0, 5.0, 1000.0, 1.0, HEMATOLOGY),
    # --- demographics & admission context ---
    FeatureSpec("Age", "Age", "years", 65.25, 0.0, 110.0, 1.0, CONTEXT),
    FeatureSpec("Gender", "Gender", "0 = female, 1 = male", 1.0, 0.0, 1.0, 1.0, CONTEXT, binary=True),
    FeatureSpec("Unit1", "In MICU", "0 / 1", 1.0, 0.0, 1.0, 1.0, CONTEXT, binary=True),
    FeatureSpec("Unit2", "In SICU", "0 / 1", 0.0, 0.0, 1.0, 1.0, CONTEXT, binary=True),
    FeatureSpec(
        "HospAdmTime",
        "Hours between hospital and ICU admission",
        "hours (negative)",
        -2.6, -400.0, 24.0, 0.1, CONTEXT,
        help="Negative means the patient was admitted to the hospital before the ICU.",
    ),
    FeatureSpec(
        "ICULOS",
        "Hours since ICU admission",
        "hours",
        21.0, 1.0, 500.0, 1.0, CONTEXT,
        help="ICU length of stay so far. Both notebooks found this to be a strong -- and "
             "somewhat time-leaky -- predictor, so it moves the score a lot.",
    ),
]

FEATURES_BY_NAME = {spec.name: spec for spec in FEATURE_SPEC}


def default_values() -> dict[str, float]:
    """Population-median starting point for the form."""
    return {spec.name: spec.default for spec in FEATURE_SPEC}


# --------------------------------------------------------------------------------------
# Results / errors
# --------------------------------------------------------------------------------------


class ModelUnavailableError(RuntimeError):
    """Raised when a model's artifact has not been exported yet."""


class MissingFeatureError(RuntimeError):
    """Raised when a model needs an input the form does not collect."""


@dataclass
class Prediction:
    probability: float
    positive: bool
    threshold: float
    features_used: Sequence[str]
    notes: list[str] = field(default_factory=list)

    @property
    def label(self) -> str:
        return "SEPSIS RISK" if self.positive else "NO SEPSIS"


def _vector(feature_names: Sequence[str], values: dict[str, float]) -> np.ndarray:
    """Pull the model's features out of the form dict, in the model's own order."""
    missing = [name for name in feature_names if name not in values]
    if missing:
        raise MissingFeatureError(
            "The form does not collect these inputs: "
            + ", ".join(missing)
            + ". Add a FeatureSpec for each one in model_utils.FEATURE_SPEC."
        )
    return np.asarray([float(values[name]) for name in feature_names], dtype=np.float32)


# --------------------------------------------------------------------------------------
# Random Forest adapter
# --------------------------------------------------------------------------------------


def load_random_forest(path: Path):
    import joblib

    bundle = joblib.load(path)
    for key in ("model", "feature_columns"):
        if key not in bundle:
            raise ModelUnavailableError(
                f"{path.name} is missing '{key}'. Re-run the export cell at the end of "
                "sepsis_random_forest_cross_hospital.ipynb."
            )
    return bundle


def predict_random_forest(bundle, values: dict[str, float], options: dict | None = None) -> Prediction:
    import pandas as pd

    options = options or {}
    columns = list(bundle["feature_columns"])
    row = _vector(columns, values)
    X = pd.DataFrame([row], columns=columns)

    # The forest was trained on median-filled data; the form always supplies a number,
    # so this only matters if a value somehow arrives as NaN.
    fill = bundle.get("fill_values") or {}
    if fill:
        X = X.fillna(pd.Series(fill))
    X = X.fillna(0)

    threshold = float(options.get("threshold", bundle.get("threshold", 0.5)))
    prob = float(bundle["model"].predict_proba(X)[0, 1])

    notes = []
    if bundle.get("trained_on"):
        notes.append(f"Forest trained on {bundle['trained_on']}.")
    return Prediction(prob, prob >= threshold, threshold, columns, notes)


# --------------------------------------------------------------------------------------
# GRU-D adapter
# --------------------------------------------------------------------------------------


def load_grud(path: Path):
    import torch

    from grud_model import GRUDModel

    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    features = list(ckpt["features"])

    model = GRUDModel(
        input_size=len(features),
        hidden_size=int(ckpt.get("hidden_size", 256)),
        dropout=float(ckpt.get("dropout", 0.2)),
    )
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    return {
        "model": model,
        "features": features,
        "mean": np.asarray(ckpt["mean"], dtype=np.float32),
        "std": np.asarray(ckpt["std"], dtype=np.float32),
        "mean_values": np.asarray(
            ckpt.get("mean_values", np.zeros(len(features))), dtype=np.float32
        ),
        "default_seq_len": int(ckpt.get("default_seq_len", 24)),
        "threshold": float(ckpt.get("threshold", 0.5)),
        "trained_on": ckpt.get("trained_on"),
    }


def predict_grud(bundle, values: dict[str, float], options: dict | None = None) -> Prediction:
    import torch

    options = options or {}
    features = bundle["features"]
    seq_len = int(options.get("sequence_length", bundle["default_seq_len"]))
    threshold = float(options.get("threshold", bundle["threshold"]))

    x = _vector(features, values)
    x = (x - bundle["mean"]) / bundle["std"]

    # The demo collects one snapshot, so we hold it constant for `seq_len` hours.
    # Every value counts as observed (mask = 1) and nothing is stale (decay = 0).
    X = np.tile(x, (seq_len, 1))[None, :, :]
    M = np.ones_like(X, dtype=np.float32)
    D = np.zeros_like(X, dtype=np.float32)

    with torch.no_grad():
        logits = bundle["model"](
            torch.from_numpy(X),
            torch.from_numpy(M),
            torch.from_numpy(D),
            torch.from_numpy(bundle["mean_values"]),
        ).squeeze(-1)
        # Patient-level score = highest hourly risk in the stay, matching how the
        # notebook scores patients when it computes ROC AUC.
        prob = float(torch.sigmoid(logits)[0].max())

    notes = [
        f"Your single set of values was repeated across {seq_len} hourly timesteps.",
    ]
    if bundle.get("trained_on"):
        notes.append(f"Network trained on {bundle['trained_on']}.")
    return Prediction(prob, prob >= threshold, threshold, features, notes)


# --------------------------------------------------------------------------------------
# Causal Temporal Transformer adapter
# --------------------------------------------------------------------------------------
# This pipeline trains on ~93 engineered columns (3h deltas, 6h rolling stats,
# elapsed-time-since-measurement, missingness flags) rather than raw vitals. The form
# collects only the base values, so the engineered ones are derived below under the same
# "one snapshot held constant" assumption the sequence itself uses.

_ROLLING_SUFFIXES = ("_min_6h", "_max_6h", "_mean_6h")


def _base_name(col: str, suffix: str) -> str:
    return col[: -len(suffix)]


def _transformer_row(feature_cols: Sequence[str], values: dict[str, float]) -> np.ndarray:
    """Derive the engineered feature vector from one snapshot of raw values."""
    row = []
    needed = []

    for col in feature_cols:
        if col.endswith("_delta3h"):
            # Values are held constant across the sequence, so the 3-hour change is zero.
            row.append(0.0)
        elif col.endswith("_elapsed_hrs"):
            # Everything the user typed counts as measured this hour.
            row.append(0.0)
        elif col.endswith("_is_missing"):
            row.append(0.0)
        elif col.endswith(_ROLLING_SUFFIXES):
            suffix = next(s for s in _ROLLING_SUFFIXES if col.endswith(s))
            base = _base_name(col, suffix)
            needed.append(base)
            # Min, max and mean of a constant window all equal that constant.
            row.append(float(values.get(base, 0.0)))
        else:
            needed.append(col)
            row.append(float(values.get(col, 0.0)))

    missing = sorted({n for n in needed if n not in values})
    if missing:
        raise MissingFeatureError(
            "The form does not collect these inputs: "
            + ", ".join(missing)
            + ". Add a FeatureSpec for each one in model_utils.FEATURE_SPEC."
        )
    return np.asarray(row, dtype=np.float32)


def load_temporal_transformer(path: Path):
    import torch

    from temporal_transformer_model import CausalTemporalTransformer

    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model = CausalTemporalTransformer(
        num_features=int(ckpt["num_features"]),
        d_model=int(ckpt.get("d_model", 128)),
        nhead=int(ckpt.get("nhead", 8)),
        num_layers=int(ckpt.get("num_layers", 4)),
        dim_feedforward=int(ckpt.get("dim_feedforward", 256)),
        dropout=float(ckpt.get("dropout", 0.3)),
        max_len=int(ckpt.get("max_seq_len", 48)),
    )
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    return {
        "model": model,
        "feature_cols": list(ckpt["feature_cols"]),
        "mean": np.asarray(ckpt["scaler_mean"], dtype=np.float32),
        "scale": np.asarray(ckpt["scaler_scale"], dtype=np.float32),
        "max_seq_len": int(ckpt.get("max_seq_len", 48)),
        "threshold": float(ckpt.get("threshold", 0.5)),
        "trained_on": ckpt.get("trained_on"),
        "metrics": ckpt.get("metrics", {}),
    }


def predict_temporal_transformer(bundle, values: dict[str, float], options: dict | None = None) -> Prediction:
    import torch

    options = options or {}
    max_len = bundle["max_seq_len"]
    # The positional encoding buffer is only max_len long, so longer requests must clamp.
    requested = int(options.get("sequence_length", max_len))
    seq_len = min(requested, max_len)
    threshold = float(options.get("threshold", bundle["threshold"]))

    x = _transformer_row(bundle["feature_cols"], values)
    x = (x - bundle["mean"]) / bundle["scale"]

    X = torch.from_numpy(np.tile(x, (seq_len, 1))[None, :, :])
    padding = torch.zeros(1, seq_len, dtype=torch.bool)   # nothing is padding

    with torch.no_grad():
        logits = bundle["model"](X, src_key_padding_mask=padding)
        # Attention is causally masked, so the last hour is the one that has seen the
        # whole history. That is the "risk right now" reading the demo wants.
        prob = float(torch.sigmoid(logits)[0, -1])

    notes = [
        f"Your single set of values was repeated across {seq_len} hourly timesteps; "
        "the score is the model's estimate for the final hour.",
        "Engineered inputs (3h deltas, 6h rolling stats, elapsed-time and missingness "
        "flags) were derived assuming nothing changed and everything was measured.",
    ]
    if requested > max_len:
        notes.append(f"Sequence length clamped from {requested} to the model's maximum of {max_len}.")
    if bundle.get("trained_on"):
        notes.append(f"Transformer trained on {bundle['trained_on']}.")
    return Prediction(prob, prob >= threshold, threshold, bundle["feature_cols"], notes)


# --------------------------------------------------------------------------------------
# The registry
# --------------------------------------------------------------------------------------


@dataclass
class ModelEntry:
    key: str
    display_name: str
    blurb: str
    artifact: Path | None = None
    loader: Callable[[Path], object] | None = None
    predictor: Callable[..., Prediction] | None = None
    coming_soon: bool = False
    # Models that consume a sequence rather than one snapshot get the seq-length control
    # and the "this is a simplification" caption in the UI.
    sequence_model: bool = False
    # Upper bound for the sequence-length control. The transformer's positional encoding
    # is built for a fixed window, so asking for more hours than that cannot work.
    max_sequence_length: int = 72
    export_hint: str = ""

    @property
    def artifact_exists(self) -> bool:
        return self.artifact is not None and self.artifact.exists()

    @property
    def selectable(self) -> bool:
        return not self.coming_soon

    def option_label(self) -> str:
        return f"{self.display_name} (coming soon)" if self.coming_soon else self.display_name


MODEL_REGISTRY: dict[str, ModelEntry] = {
    "random_forest": ModelEntry(
        key="random_forest",
        display_name="Random Forest",
        blurb="Per-hour classifier over a single snapshot of vitals and labs. "
              "Trained cross-hospital on PhysioNet training_setA.",
        artifact=MODELS_DIR / "random_forest_sepsis.joblib",
        loader=load_random_forest,
        predictor=predict_random_forest,
        export_hint="Run the last cell of sepsis_random_forest_cross_hospital.ipynb.",
    ),
    "temporal_transformer": ModelEntry(
        key="temporal_transformer",
        display_name="Temporal Transformer",
        blurb="Causal attention over the hourly sequence, trained with focal loss on "
              "engineered features (deltas, rolling stats, missingness).",
        artifact=MODELS_DIR / "temporal_transformer_sepsis.pt",
        loader=load_temporal_transformer,
        predictor=predict_temporal_transformer,
        sequence_model=True,
        max_sequence_length=48,
        export_hint="Run temptransformersepsis-2.py; its section 8b writes the artifact.",
    ),
    # --- not in the current deployment ------------------------------------------------
    # GRU-D is fully implemented: load_grud and predict_grud below are wired and tested.
    # It is greyed out only because no trained artifact ships with this deployment. To
    # bring it back, run the export cell at the end of GRU-D.ipynb to produce
    # models/grud_sepsis.pt, then delete the `coming_soon=True` line here.
    "grud": ModelEntry(
        key="grud",
        display_name="GRU-D",
        blurb="Recurrent network with learned decay for irregular sampling. Implemented "
              "and tested, but not trained for this deployment.",
        artifact=MODELS_DIR / "grud_sepsis.pt",
        loader=load_grud,
        predictor=predict_grud,
        sequence_model=True,
        coming_soon=True,
        export_hint="Run the last cell of GRU-D.ipynb (it trains on Hospital A, then saves).",
    ),
    "catboost": ModelEntry(
        key="catboost",
        display_name="CatBoost",
        blurb="Gradient-boosted trees. Not trained yet.",
        coming_soon=True,
    ),
    "brits": ModelEntry(
        key="brits",
        display_name="BRITS",
        blurb="Bidirectional recurrent imputation. Weights exist (brits_best.pt) but the "
              "feature list and scaling were never saved with them, so it cannot be "
              "served yet -- see brits_model.py for what is still needed.",
        coming_soon=True,
        sequence_model=True,
    ),
}


def load_model(key: str):
    """Load a model's artifact. Raises ModelUnavailableError with an actionable message."""
    entry = MODEL_REGISTRY[key]
    if entry.coming_soon or entry.loader is None:
        raise ModelUnavailableError(f"{entry.display_name} has not been implemented yet.")
    if not entry.artifact_exists:
        raise ModelUnavailableError(
            f"No artifact at {entry.artifact}. {entry.export_hint}"
        )
    return entry.loader(entry.artifact)


def predict(key: str, bundle, values: dict[str, float], options: dict | None = None) -> Prediction:
    entry = MODEL_REGISTRY[key]
    if entry.predictor is None:
        raise ModelUnavailableError(f"{entry.display_name} has no predictor registered.")
    return entry.predictor(bundle, values, options)
