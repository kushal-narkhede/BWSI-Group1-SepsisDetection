"""Streamlit demo for the BWSI Group 1 sepsis detection models.

Run with:  streamlit run app.py

The UI knows nothing about any specific model. It renders a form from
``model_utils.FEATURE_SPEC`` and dispatches through ``model_utils.MODEL_REGISTRY``,
so adding CatBoost or the Temporal Transformer means editing model_utils.py, not this file.
"""

import streamlit as st

import model_utils
from model_utils import (
    FEATURE_SPEC,
    GROUP_ORDER,
    MODEL_REGISTRY,
    ModelUnavailableError,
)

st.set_page_config(
    page_title="Sepsis Prediction Demo",
    page_icon="+",
    layout="wide",
    # The model picker lives in the sidebar, so it must not start collapsed.
    initial_sidebar_state="expanded",
)

# Streamlit's selectbox cannot disable individual options, so unfinished models stay in
# the list with a "(coming soon)" suffix and selecting one disables the form instead.
BINARY_LABELS = {
    "Gender": {0.0: "Female", 1.0: "Male"},
    "Unit1": {0.0: "No", 1.0: "Yes"},
    "Unit2": {0.0: "No", 1.0: "Yes"},
}


@st.cache_resource(show_spinner="Loading model...")
def get_model(model_key: str):
    """Cached so the artifact is read from disk once per session, not once per click."""
    return model_utils.load_model(model_key)


def render_inputs() -> dict[str, float]:
    """Draw every FeatureSpec, grouped, and return {feature_name: value}."""
    values = {}
    for group in GROUP_ORDER:
        specs = [s for s in FEATURE_SPEC if s.group == group]
        if not specs:
            continue
        st.subheader(group)
        columns = st.columns(4)
        for i, spec in enumerate(specs):
            with columns[i % 4]:
                label = f"{spec.label} ({spec.unit})" if spec.unit else spec.label
                if spec.binary:
                    choices = BINARY_LABELS.get(spec.name, {0.0: "No", 1.0: "Yes"})
                    values[spec.name] = st.selectbox(
                        label,
                        options=[0.0, 1.0],
                        index=int(spec.default),
                        format_func=lambda v, c=choices: c[v],
                        help=spec.help or None,
                        key=f"input_{spec.name}",
                    )
                else:
                    values[spec.name] = st.number_input(
                        label,
                        min_value=spec.min_value,
                        max_value=spec.max_value,
                        value=spec.default,
                        step=spec.step,
                        help=spec.help or None,
                        key=f"input_{spec.name}",
                    )
    return values


def render_result(entry, prediction) -> None:
    left, right = st.columns([1, 2])
    with left:
        st.metric("Sepsis probability", f"{prediction.probability:.1%}")
        st.caption(f"Decision threshold: {prediction.threshold:.2f}")
    with right:
        if prediction.positive:
            st.error(f"**{prediction.label}** - {entry.display_name} flags this patient-hour.")
        else:
            st.success(f"**{prediction.label}** - {entry.display_name} does not flag this patient-hour.")
        st.progress(min(max(prediction.probability, 0.0), 1.0))

    for note in prediction.notes:
        st.caption(note)
    with st.expander(f"Features this model used ({len(prediction.features_used)})"):
        st.write(", ".join(prediction.features_used))


# ---------------------------------------------------------------------------------
# Sidebar: model choice and decision settings
# ---------------------------------------------------------------------------------
st.sidebar.title("Model")

model_key = st.sidebar.selectbox(
    "Prediction model",
    options=list(MODEL_REGISTRY),
    format_func=lambda k: MODEL_REGISTRY[k].option_label(),
)
entry = MODEL_REGISTRY[model_key]
st.sidebar.caption(entry.blurb)

# Load before drawing the threshold slider so it can start at the model's own operating
# point. These models are not calibrated to each other -- the transformer's tuned cutoff
# is 0.11, and forcing 0.5 on it would make it flag nothing at all.
bundle = None
load_error = None
if entry.selectable and entry.artifact_exists:
    try:
        bundle = get_model(model_key)
    except Exception as exc:
        load_error = exc

tuned = float(bundle.get("threshold", 0.5)) if isinstance(bundle, dict) else 0.5

threshold = st.sidebar.slider(
    "Decision threshold",
    min_value=0.01,
    max_value=0.95,
    value=round(tuned, 2),
    step=0.01,
    # A per-model key so switching models resets to that model's own default rather
    # than carrying the previous model's cutoff across.
    key=f"threshold_{model_key}",
    help="Anything at or above this probability is called positive. Sepsis hours are rare, "
         "so the useful cutoffs sit well below 0.5 -- a missed case costs far more than a "
         "false alarm.",
)
if abs(threshold - tuned) < 1e-9 and tuned != 0.5:
    st.sidebar.caption(f"Starting at {tuned:.2f}, this model's tuned operating point.")

options = {"threshold": threshold}
if entry.sequence_model:
    st.sidebar.markdown("---")
    seq_max = entry.max_sequence_length
    options["sequence_length"] = st.sidebar.slider(
        "Sequence length (hours)",
        min_value=4,
        max_value=seq_max,
        value=min(24, seq_max),
        step=4,
        help="How many hourly timesteps to build from your single set of values. "
             f"{entry.display_name} accepts at most {seq_max}.",
    )

# ---------------------------------------------------------------------------------
# Main panel
# ---------------------------------------------------------------------------------
st.title("Sepsis Prediction Demo")
st.caption(
    "BWSI Group 1 - PhysioNet 2019 Sepsis Challenge. Educational demo only: "
    "these models are not validated for clinical use and must not inform patient care."
)

blocked = None
if entry.coming_soon:
    ready = [e.display_name for e in MODEL_REGISTRY.values() if e.selectable and e.artifact_exists]
    suggestion = f" Pick {' or '.join(ready)}." if ready else ""
    blocked = f"{entry.display_name} is not available yet.{suggestion}"
elif not entry.artifact_exists:
    # Show the repo-relative path -- an absolute container path reads like a crash when
    # the app is deployed rather than run locally.
    try:
        where = entry.artifact.relative_to(model_utils.PROJECT_ROOT)
    except ValueError:
        where = entry.artifact
    blocked = (
        f"{entry.display_name} is wired up but has no saved model in this deployment "
        f"(expected `{where}`).\n\n{entry.export_hint}"
    )

if blocked:
    st.warning(blocked)
elif load_error is not None:
    st.error(f"Could not load {entry.display_name}: {load_error}")

if entry.sequence_model and not entry.coming_soon:
    st.info(
        f"**{entry.display_name} expects an hourly sequence, not a single snapshot.** "
        "This form collects one set of values and repeats it across every timestep, so the "
        "model sees a patient whose vitals never change. That is a demo simplification, not "
        "a scientifically accurate multi-hour input, and the resulting probability should "
        "not be compared directly against the notebook's reported ROC AUC."
    )

with st.form("patient_form"):
    values = render_inputs()
    submitted = st.form_submit_button(
        f"Predict with {entry.display_name}",
        type="primary",
        disabled=bool(blocked) or load_error is not None,
    )

if submitted and not blocked and load_error is None:
    try:
        prediction = model_utils.predict(model_key, bundle, values, options)
    except ModelUnavailableError as exc:
        st.error(str(exc))
    except Exception as exc:  # surface the real error instead of a blank page
        st.exception(exc)
    else:
        st.markdown("---")
        render_result(entry, prediction)
