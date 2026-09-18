"""
Breast Cancer Ferroptosis Prognostic Model
29-gene Cox Proportional Hazards Model | SHAP Explanation of Log Partial Hazard

Data directory should contain:
- cox_model.pkl: Model saved using lifelines.CoxPHFitter
- model_info.csv: At least columns: gene, coef, hr, median_expression
- background_data.csv: Training expression data with model genes as columns
- risk_threshold.txt: Median partial hazard from training set
- training_risk_scores_29genes.csv: Training set partial hazards, single column without header

Important: Input expression values must exactly match model training preprocessing,
e.g., log2(TPM + 1).
"""

from pathlib import Path
import warnings

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import shap
import streamlit as st

warnings.filterwarnings("ignore")

# ============================================================
# Path and Page Configuration
# ============================================================
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

# Fallback directory for local development. Remove this branch for deployment,
# keeping only relative path data/.
DATA_DIR = Path(__file__).resolve().parent / "data"

st.set_page_config(
    page_title="Ferroptosis Predictor | Breast Cancer Prognostic Model",
    page_icon="⚕️",
    layout="wide",
)


# ============================================================
# Model and Data Loading
# ============================================================
@st.cache_resource
def load_models():
    """Load Cox model, model info, background data, and risk distribution."""

    required_files = [
        "cox_model.pkl",
        "model_info.csv",
        "background_data.csv",
        "risk_threshold.txt",
        "training_risk_scores_29genes.csv",
    ]

    missing_files = [
        name for name in required_files
        if not (DATA_DIR / name).exists()
    ]

    if missing_files:
        raise FileNotFoundError(
            f"Data directory missing files: {', '.join(missing_files)}\n"
            f"Current directory: {DATA_DIR.resolve()}"
        )

    # 1. Cox model
    cph = joblib.load(DATA_DIR / "cox_model.pkl")

    # 2. Model gene information
    model_info = pd.read_csv(DATA_DIR / "model_info.csv")

    required_columns = {"gene", "coef", "hr", "median_expression"}
    missing_columns = required_columns - set(model_info.columns)
    if missing_columns:
        raise ValueError(
            "model_info.csv missing columns: " + ", ".join(sorted(missing_columns))
        )

    if model_info["gene"].duplicated().any():
        raise ValueError("model_info.csv contains duplicate gene names.")

    feature_names = model_info["gene"].astype(str).tolist()

    # 3. SHAP background data
    background_raw = pd.read_csv(DATA_DIR / "background_data.csv", index_col=0)
    missing_genes = set(feature_names) - set(background_raw.columns)
    if missing_genes:
        raise ValueError(
            "background_data.csv missing model genes: " + ", ".join(sorted(missing_genes))
        )

    background_df = background_raw.loc[:, feature_names].copy()
    background_df = background_df.apply(pd.to_numeric, errors="coerce")
    background_df = background_df.replace([np.inf, -np.inf], np.nan).dropna()

    if background_df.empty:
        raise ValueError("background_data.csv contains no valid numeric background samples.")

    # 4. Risk threshold
    with open(DATA_DIR / "risk_threshold.txt", "r", encoding="utf-8") as f:
        risk_threshold = float(f.read().strip())

    if not np.isfinite(risk_threshold) or risk_threshold <= 0:
        raise ValueError("risk_threshold.txt threshold must be a positive number.")

    # 5. Training set risk distribution
    risk_file = DATA_DIR / "training_risk_scores_29genes.csv"
    risk_raw = pd.read_csv(risk_file, header=None)

    risk_distribution = pd.to_numeric(
        risk_raw.iloc[:, 0],
        errors="coerce"
    ).dropna().to_numpy(dtype=float)

    risk_distribution = risk_distribution[np.isfinite(risk_distribution)]

    if risk_distribution.size == 0:
        raise ValueError(
            "training_risk_scores_29genes.csv contains no valid numeric risk scores."
        )

    # 6. Verify cox_model.pkl and model_info.csv coefficients match
    model_coef = cph.params_.reindex(feature_names)
    if model_coef.isna().any():
        missing_in_model = model_coef[model_coef.isna()].index.tolist()
        raise ValueError(
            "cox_model.pkl missing genes: " + ", ".join(missing_in_model)
        )

    info_coef = model_info.set_index("gene").loc[feature_names, "coef"].astype(float)
    if not np.allclose(
            model_coef.to_numpy(),
            info_coef.to_numpy(),
            rtol=1e-6,
            atol=1e-8,
    ):
        raise ValueError(
            "cox_model.pkl and model_info.csv coefficients do not match."
        )

    # 7. SHAP: Explain log partial hazard
    background_sample = background_df.sample(
        n=min(100, len(background_df)),
        random_state=42,
    )

    def predict_log_partial_hazard(x_values):
        x_df = pd.DataFrame(x_values, columns=feature_names)
        return cph.predict_log_partial_hazard(x_df).to_numpy()

    explainer = shap.KernelExplainer(
        predict_log_partial_hazard,
        background_sample.to_numpy(),
    )

    return (
        cph,
        explainer,
        model_info,
        feature_names,
        risk_threshold,
        risk_distribution,
    )


try:
    with st.spinner("Loading models..."):
        (
            cph,
            explainer,
            model_info,
            feature_names,
            risk_threshold,
            risk_distribution,
        ) = load_models()
except Exception as error:
    st.error(f"Model loading failed: {error}")
    st.stop()


# ============================================================
# Calculation Functions
# ============================================================
def get_shap_values_safe(explainer, x_values):
    """Handle different SHAP version return formats."""
    result = explainer.shap_values(x_values)

    if isinstance(result, list):
        result = result[0] if result else np.zeros_like(x_values, dtype=float)

    values = np.asarray(result, dtype=float)
    if values.ndim == 1:
        values = values.reshape(1, -1)

    expected_shape = x_values.shape
    if values.shape != expected_shape:
        raise ValueError(
            f"SHAP output dimension mismatch: got {values.shape}, expected {expected_shape}."
        )
    return values


def validate_input(input_df):
    if input_df.isna().any().any():
        raise ValueError("Input contains missing values.")
    if not np.isfinite(input_df.to_numpy(dtype=float)).all():
        raise ValueError("Input contains infinite values or non-numeric values.")


def calculate_risk_and_shap(cph, explainer, feature_names, user_input):
    input_df = pd.DataFrame([user_input], columns=feature_names)
    input_df = input_df.apply(pd.to_numeric, errors="raise")
    validate_input(input_df)

    log_risk = float(cph.predict_log_partial_hazard(input_df).iloc[0])
    partial_hazard = float(cph.predict_partial_hazard(input_df).iloc[0])

    shap_values = get_shap_values_safe(explainer, input_df.to_numpy())[0]
    expected_value = float(np.asarray(explainer.expected_value).reshape(-1)[0])
    shap_absolute_error = abs(log_risk - (expected_value + shap_values.sum()))

    return input_df, log_risk, partial_hazard, shap_values, expected_value, shap_absolute_error


def calculate_survival_probabilities(cph, input_df):
    """Predict individual survival probabilities using the model's baseline survival function.

    These time points assume the Cox model was trained with os_time in "days".
    If the model uses months or years, modify target_times accordingly.
    """
    target_months = [6, 12, 18]
    target_times = [182.5, 365.0, 547.5]

    try:
        survival = cph.predict_survival_function(input_df, times=target_times)
        return {
            month: float(survival.iloc[row_index, 0])
            for row_index, month in enumerate(target_months)
        }
    except Exception:
        return {month: np.nan for month in target_months}


def calculate_percentile(partial_hazard, distribution):
    return float(np.mean(distribution <= partial_hazard) * 100)


def create_preset_values(scenario_type, gene_list, median_values, risk_genes, protective_genes):
    """Generate demonstration input values.

    With log2(TPM + 1) input, a ±1 change in log2 scale approximates doubling/halving
    of raw expression. These presets are for demonstration only and do not represent
    actual patient expression profiles.
    """
    delta = 1.0

    if scenario_type == "High Risk Demo":
        return {
            gene: max(0.0, median_values[gene] + delta)
            if gene in risk_genes
            else max(0.0, median_values[gene] - delta)
            for gene in gene_list
        }

    if scenario_type == "Low Risk Demo":
        return {
            gene: max(0.0, median_values[gene] - delta)
            if gene in risk_genes
            else max(0.0, median_values[gene] + delta)
            for gene in gene_list
        }

    return {gene: median_values[gene] for gene in gene_list}


# ============================================================
# Plotting Functions
# ============================================================
def plot_gauge_chart(partial_hazard, threshold):
    max_value = max(3.0, threshold * 1.8, partial_hazard * 1.15)

    figure = go.Figure(
        go.Indicator(
            mode="gauge+number+delta",
            value=partial_hazard,
            title={"text": "Relative Hazard (Partial Hazard)", "font": {"size": 15}},
            delta={"reference": threshold, "increasing": {"color": "#d73027"}},
            gauge={
                "axis": {"range": [0, max_value]},
                "bar": {"color": "#d73027" if partial_hazard >= threshold else "#4575b4"},
                "steps": [
                    {"range": [0, threshold], "color": "#dceef4"},
                    {"range": [threshold, max_value], "color": "#f8dddd"},
                ],
                "threshold": {
                    "line": {"color": "#d73027", "width": 4},
                    "thickness": 0.75,
                    "value": threshold,
                },
            },
        )
    )
    figure.update_layout(height=260, margin=dict(l=20, r=20, t=45, b=20))
    return figure


def plot_survival_curve(survival_probs):
    months = list(survival_probs.keys())
    probabilities = list(survival_probs.values())

    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=months,
            y=probabilities,
            mode="lines+markers",
            connectgaps=False,
            line=dict(color="#4575b4", width=3),
            marker=dict(size=9, color="#4575b4"),
            fill="tozeroy",
            fillcolor="rgba(69, 117, 180, 0.18)",
            name="Predicted survival",
        )
    )
    figure.update_layout(
        title="Predicted Overall Survival",
        xaxis_title="Time (months)",
        yaxis_title="Survival probability",
        xaxis=dict(tickvals=months),
        yaxis=dict(range=[0, 1], tickformat=".0%"),
        height=260,
        margin=dict(l=20, r=20, t=50, b=20),
        plot_bgcolor="white",
    )
    figure.update_xaxes(gridcolor="#e0e0e0")
    figure.update_yaxes(gridcolor="#e0e0e0")
    return figure


def plot_shap_bar_chart(shap_values, feature_names, expected_value, log_risk, top_n=15):
    shap_table = pd.DataFrame({"Gene": feature_names, "SHAP": shap_values})
    shap_table = shap_table.reindex(shap_table["SHAP"].abs().sort_values(ascending=False).index)
    shap_table = shap_table.head(min(top_n, len(shap_table))).sort_values("SHAP")

    colors = np.where(shap_table["SHAP"] >= 0, "#d73027", "#4575b4")
    figure = go.Figure(
        go.Bar(
            x=shap_table["SHAP"],
            y=shap_table["Gene"],
            orientation="h",
            marker_color=colors,
            marker_line_color="black",
            marker_line_width=0.5,
            text=shap_table["SHAP"].round(4),
            textposition="outside",
            hovertemplate="<b>%{y}</b><br>SHAP: %{x:.4f}<extra></extra>",
        )
    )
    figure.add_vline(x=0, line_width=1.5, line_color="black")
    figure.update_layout(
        title=(
            f"Top {min(top_n, len(feature_names))} SHAP Contributions<br>"
            f"<sup>Expected value {expected_value:.4f} + sum(SHAP) "
            f"= {expected_value + np.sum(shap_values):.4f}; "
            f"model log-risk = {log_risk:.4f}</sup>"
        ),
        xaxis_title="SHAP value (log partial hazard scale)",
        yaxis_title="Gene",
        height=520,
        margin=dict(l=10, r=30, t=85, b=40),
        plot_bgcolor="white",
    )
    figure.update_xaxes(gridcolor="#e0e0e0")
    return figure


# ============================================================
# Initial Load
# ============================================================
try:
    with st.spinner("Loading models..."):
        cph, explainer, model_info, feature_names, risk_threshold, risk_distribution = load_models()
except Exception as error:
    st.error(f"Model loading failed: {error}")
    st.stop()

model_info = model_info.set_index("gene").loc[feature_names].reset_index()
coef_dict = model_info.set_index("gene")["coef"].astype(float).to_dict()
hr_dict = model_info.set_index("gene")["hr"].astype(float).to_dict()
median_values = model_info.set_index("gene")["median_expression"].astype(float).to_dict()

risk_genes = [gene for gene in feature_names if coef_dict[gene] > 0]
protective_genes = [gene for gene in feature_names if coef_dict[gene] < 0]
neutral_genes = [gene for gene in feature_names if np.isclose(coef_dict[gene], 0.0)]

for gene in feature_names:
    st.session_state.setdefault(f"expression_{gene}", median_values[gene])
st.session_state.setdefault("predicted", False)

# ============================================================
# Main Page
# ============================================================
st.title("Breast Cancer Ferroptosis Prognostic Prediction Model")
st.caption(
    f"Cox proportional hazards model based on {len(feature_names)} ferroptosis-related genes. "
    "Input expression values must use the same preprocessing as training data."
)

with st.sidebar:
    st.header("Model Information")
    st.markdown(
        f"- Model: Cox Proportional Hazards Regression  \n"
        f"- Number of genes: {len(feature_names)}  \n"
        f"- Risk threshold: training set partial hazard median = `{risk_threshold:.4f}`  \n"
        f"- SHAP explanation scale: log partial hazard"
    )

    st.divider()
    st.header("Input Expression Values")
    st.caption("Expression format must match model training, e.g., log2(TPM + 1).")

    preset = st.selectbox("Demo Preset", ["Median Patient", "High Risk Demo", "Low Risk Demo"])
    if st.button("Apply Preset", use_container_width=True):
        preset_values = create_preset_values(
            preset, gene_list=feature_names, median_values=median_values,
            risk_genes=risk_genes, protective_genes=protective_genes,
        )
        for gene, value in preset_values.items():
            st.session_state[f"expression_{gene}"] = float(value)
        st.session_state["predicted"] = False
        st.rerun()

    left, right = st.columns(2)
    with left:
        if st.button("Reset to Median", use_container_width=True):
            for gene in feature_names:
                st.session_state[f"expression_{gene}"] = float(median_values[gene])
            st.session_state["predicted"] = False
            st.rerun()
    with right:
        if st.button("Set All to Zero", use_container_width=True):
            for gene in feature_names:
                st.session_state[f"expression_{gene}"] = 0.0
            st.session_state["predicted"] = False
            st.rerun()

    st.divider()
    st.caption("All model genes must be entered. Genes are classified by final Cox coefficient direction.")

    with st.expander(f"Risk Genes: {len(risk_genes)}", expanded=True):
        for gene in risk_genes:
            st.number_input(
                f"{gene} (HR={hr_dict[gene]:.3f})",
                value=float(st.session_state[f"expression_{gene}"]),
                format="%.4f",
                key=f"expression_{gene}",
            )

    with st.expander(f"Protective Genes: {len(protective_genes)}", expanded=False):
        for gene in protective_genes:
            st.number_input(
                f"{gene} (HR={hr_dict[gene]:.3f})",
                value=float(st.session_state[f"expression_{gene}"]),
                format="%.4f",
                key=f"expression_{gene}",
            )

    if neutral_genes:
        with st.expander(f"Neutral Genes: {len(neutral_genes)}", expanded=False):
            for gene in neutral_genes:
                st.number_input(
                    gene,
                    value=float(st.session_state[f"expression_{gene}"]),
                    format="%.4f",
                    key=f"expression_{gene}",
                )

current_input = {
    gene: float(st.session_state[f"expression_{gene}"])
    for gene in feature_names
}

_, center, _ = st.columns([1, 2, 1])
with center:
    predict_clicked = st.button("Predict", type="primary", use_container_width=True)

if predict_clicked:
    try:
        with st.spinner("Calculating risk, SHAP contributions, and survival probabilities..."):
            (
                input_df,
                log_risk,
                partial_hazard,
                shap_values,
                expected_value,
                shap_error,
            ) = calculate_risk_and_shap(cph, explainer, feature_names, current_input)

            survival_probs = calculate_survival_probabilities(cph, input_df)
            percentile = calculate_percentile(partial_hazard, risk_distribution)
            risk_group = "High Risk" if partial_hazard >= risk_threshold else "Low Risk"

            st.session_state.update(
                predicted=True,
                log_risk=log_risk,
                partial_hazard=partial_hazard,
                shap_values=shap_values,
                expected_value=expected_value,
                shap_error=shap_error,
                survival_probs=survival_probs,
                percentile=percentile,
                risk_group=risk_group,
                input_df=input_df,
            )
    except Exception as error:
        st.error(f"Prediction failed: {error}")
        st.stop()

if not st.session_state["predicted"]:
    st.info("Please enter all gene expression values in the sidebar, then click 'Predict'.")
    st.stop()

log_risk = st.session_state["log_risk"]
partial_hazard = st.session_state["partial_hazard"]
risk_group = st.session_state["risk_group"]
shap_values = st.session_state["shap_values"]
expected_value = st.session_state["expected_value"]
shap_error = st.session_state["shap_error"]
survival_probs = st.session_state["survival_probs"]
percentile = st.session_state["percentile"]

metric1, metric2, metric3, metric4 = st.columns(4)
metric1.metric("Log partial hazard", f"{log_risk:.4f}")
metric2.metric("Relative hazard", f"{partial_hazard:.4f}")
metric3.metric("Risk Group", risk_group)
metric4.metric("Training Set Risk Percentile", f"{percentile:.1f}%")

if shap_error > 1e-4:
    st.warning(
        f"SHAP sum absolute error: {shap_error:.6g}. Kernel SHAP is approximate; consider increasing background samples or iterations.")

left_column, right_column = st.columns(2)
with left_column:
    st.plotly_chart(plot_gauge_chart(partial_hazard, risk_threshold), use_container_width=True)
with right_column:
    if any(np.isnan(value) for value in survival_probs.values()):
        st.warning(
            "Cannot compute survival probabilities at specified time points. Please confirm model includes baseline survival function and os_time units match prediction times.")
    else:
        st.plotly_chart(plot_survival_curve(survival_probs), use_container_width=True)

st.divider()
st.subheader("SHAP Explainability Analysis")
st.caption("Positive SHAP values increase log partial hazard; negative values decrease log partial hazard.")
st.plotly_chart(
    plot_shap_bar_chart(shap_values, feature_names, expected_value, log_risk),
    use_container_width=True,
)

st.subheader("Gene Contribution Details")
result_df = pd.DataFrame(
    {
        "Gene": feature_names,
        "Input Expression": [current_input[gene] for gene in feature_names],
        "Training Set Median": [median_values[gene] for gene in feature_names],
        "Final Cox Coefficient": [coef_dict[gene] for gene in feature_names],
        "Final Cox HR": [hr_dict[gene] for gene in feature_names],
        "SHAP Value": shap_values,
    }
)
result_df["Expression Difference (vs Median)"] = result_df["Input Expression"] - result_df["Training Set Median"]
result_df["Contribution Direction"] = np.where(result_df["SHAP Value"] >= 0, "Increases log-risk", "Decreases log-risk")
result_df = result_df.reindex(result_df["SHAP Value"].abs().sort_values(ascending=False).index)
st.dataframe(result_df.round(4), use_container_width=True, height=500)

with st.expander("Model Description", expanded=False):
    st.markdown(
        """
- `Relative hazard` is the Cox model's partial hazard, not the absolute mortality risk for a clinical benchmark.
- The high/low risk threshold is the median partial hazard of the training set; should be recalibrated for new cohorts.
- Individual survival probabilities are computed from the Cox baseline survival function; only interpretable when input features, expression preprocessing, and survival time units match the training model.
- This tool is for research and risk stratification exploration only and should not replace clinical decision-making.
        """
    )
    st.dataframe(model_info.round(6), use_container_width=True)
