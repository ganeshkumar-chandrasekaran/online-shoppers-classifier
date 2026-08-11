"""
Streamlit app — Online Shoppers Purchasing Intention classifiers.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
import streamlit.components.v1 as components
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "model"
DEFAULT_TEST = ROOT / "test_data.csv"

MODEL_FILES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "kNN": "knn.joblib",
    "Naive Bayes": "naive_bayes.joblib",
    "Random Forest (Ensemble)": "random_forest_ensemble.joblib",
}

COLUMN_HELP = {
    "Administrative": "Count of administrative pages visited in the session (account, profile, etc.).",
    "Administrative_Duration": "Total time (seconds) spent on administrative pages.",
    "Informational": "Count of informational pages visited (about us, FAQ, etc.).",
    "Informational_Duration": "Total time (seconds) spent on informational pages.",
    "ProductRelated": "Count of product-related pages visited (catalog / product details).",
    "ProductRelated_Duration": "Total time (seconds) spent on product-related pages.",
    "BounceRates": "Average bounce rate of the pages visited (left after one page).",
    "ExitRates": "Average exit rate of the pages visited (share of exits from those pages).",
    "PageValues": "Average page value — how much a page contributes toward a purchase.",
    "SpecialDay": "Closeness to a special day (e.g. Mother’s Day, Valentine’s), from 0 to 1.",
    "Month": "Month of the session visit (e.g. Feb, May, Nov).",
    "OperatingSystems": "Encoded operating system used by the visitor.",
    "Browser": "Encoded browser used by the visitor.",
    "Region": "Encoded geographic region of the visitor.",
    "TrafficType": "Encoded traffic source type (direct, ads, referral, etc.).",
    "VisitorType": "Returning_Visitor, New_Visitor, or Other.",
    "Weekend": "Whether the session happened on a weekend (True/False).",
    "Revenue": "Target label — True if the session ended in a purchase, else False.",
}

ACCENT_PALETTES = {
    "Orange": {
        "primary": "#e85d04",
        "primary_dark": "#c2410c",
        "soft": "#fff4eb",
        "border": "#fdba74",
        "heading": "#9a3412",
        "cmap": "Oranges",
    },
    "Blue": {
        "primary": "#2563eb",
        "primary_dark": "#1d4ed8",
        "soft": "#eff6ff",
        "border": "#93c5fd",
        "heading": "#1e3a8a",
        "cmap": "Blues",
    },
    "Teal": {
        "primary": "#0d9488",
        "primary_dark": "#0f766e",
        "soft": "#f0fdfa",
        "border": "#5eead4",
        "heading": "#115e59",
        "cmap": "BuGn",
    },
    "Green": {
        "primary": "#16a34a",
        "primary_dark": "#15803d",
        "soft": "#f0fdf4",
        "border": "#86efac",
        "heading": "#14532d",
        "cmap": "Greens",
    },
    "Purple": {
        "primary": "#7c3aed",
        "primary_dark": "#6d28d9",
        "soft": "#f5f3ff",
        "border": "#c4b5fd",
        "heading": "#4c1d95",
        "cmap": "Purples",
    },
    "Rose": {
        "primary": "#e11d48",
        "primary_dark": "#be123c",
        "soft": "#fff1f2",
        "border": "#fda4af",
        "heading": "#9f1239",
        "cmap": "RdPu",
    },
}


@st.cache_resource
def load_artifacts():
    models = {}
    for name, filename in MODEL_FILES.items():
        path = MODEL_DIR / filename
        if not path.exists():
            raise FileNotFoundError(
                f"Missing model file: {path}. "
                "Run model/online_shoppers_classification.py first to train and save models."
            )
        models[name] = joblib.load(path)

    label_encoder = joblib.load(MODEL_DIR / "label_encoder.joblib")
    with open(MODEL_DIR / "metrics.json", encoding="utf-8") as f:
        meta = json.load(f)
    return models, label_encoder, meta


@st.cache_data
def load_default_test() -> pd.DataFrame:
    return pd.read_csv(DEFAULT_TEST)


def make_sample_csv(full_test: pd.DataFrame, n_rows: int, seed: int) -> bytes:
    """Build a smaller sample from the reserved test set for download/upload practice."""
    n = min(n_rows, len(full_test))
    sample = full_test.sample(n=n, random_state=seed).reset_index(drop=True)
    return sample.to_csv(index=False).encode("utf-8")


def prepare_xy(df: pd.DataFrame, meta: dict, label_encoder):
    feature_order = meta["feature_order"]
    target = meta["target"]

    missing = [c for c in feature_order if c not in df.columns]
    if missing:
        raise ValueError(f"CSV is missing columns: {missing}")

    X = df[feature_order].copy()
    y = None
    if target in df.columns:
        y_raw = df[target]
        if set(pd.Series(y_raw).astype(str).unique()) <= {"0", "1", "0.0", "1.0"}:
            y = y_raw.astype(int).to_numpy()
        else:
            y = label_encoder.transform(y_raw.astype(str).to_numpy())
    return X, y


def compute_metrics(y_true, y_pred, y_proba):
    return {
        "Accuracy": float(accuracy_score(y_true, y_pred)),
        "AUC": float(roc_auc_score(y_true, y_proba)),
        "Precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "Recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "F1": float(f1_score(y_true, y_pred, zero_division=0)),
        "MCC": float(matthews_corrcoef(y_true, y_pred)),
    }


def render_colored_table(
    df: pd.DataFrame,
    accent: dict,
    float_format: str | None = None,
    max_rows: int | None = None,
):
    """
    Render an HTML table with clearly colored headers.
    Streamlit's interactive dataframe often ignores CSS header colors, so HTML is used.
    """
    view = df.copy()
    if max_rows is not None:
        view = view.head(max_rows)

    if float_format:
        formatters = {
            c: (lambda x, fmt=float_format: fmt.format(x) if pd.notna(x) and isinstance(x, (int, float)) else x)
            for c in view.select_dtypes(include="number").columns
        }
        view = view.copy()
        for col, fn in formatters.items():
            view[col] = view[col].map(lambda v, f=fn: f(v) if pd.notna(v) else "")

    header_cells = "".join(
        f'<th style="background:linear-gradient(180deg,{accent["primary"]} 0%,'
        f'{accent["primary_dark"]} 100%);color:#fff;font-weight:800;'
        f'letter-spacing:0.03em;padding:0.65rem 0.75rem;border:1px solid '
        f'{accent["primary_dark"]};text-align:center;">{col}</th>'
        for col in view.columns
    )

    body_rows = []
    for _, row in view.iterrows():
        tds = "".join(
            f'<td style="padding:0.45rem 0.65rem;border:1px solid {accent["border"]};'
            f'background:rgba(255,255,255,0.02);">{row[col]}</td>'
            for col in view.columns
        )
        body_rows.append(f"<tr>{tds}</tr>")

    html = f"""
<div style="overflow-x:auto;border:2px solid {accent['primary']};border-radius:10px;margin:0.25rem 0 0.75rem 0;">
  <table style="border-collapse:collapse;width:100%;min-width:640px;">
    <thead><tr>{header_cells}</tr></thead>
    <tbody>
      {''.join(body_rows)}
    </tbody>
  </table>
</div>
"""
    st.markdown(html, unsafe_allow_html=True)


@st.dialog("Full table view", width="large")
def show_full_table_dialog(df: pd.DataFrame, accent: dict, title: str):
    st.markdown(f"**{title}** — {len(df)} rows")
    render_colored_table(df, accent)
    st.caption("Close this dialog to return to the main page.")


def make_confusion_figure(y, y_pred, label_encoder, model_name, accent, figsize=(6, 5)):
    fig, ax = plt.subplots(figsize=figsize)
    ConfusionMatrixDisplay.from_predictions(
        y,
        y_pred,
        display_labels=label_encoder.classes_,
        cmap=accent["cmap"],
        ax=ax,
        colorbar=True,
    )
    ax.set_title(f"{model_name} — evaluation / test data")
    fig.tight_layout()
    return fig


@st.dialog("Confusion matrix — full view", width="large")
def show_confusion_matrix_dialog(y, y_pred, label_encoder, model_name, accent, source_label):
    st.markdown(
        f"**Model:** {model_name}  \n"
        f"**Data:** {source_label}  \n"
        "Scope: test / evaluation rows only — not training data"
    )
    fig = make_confusion_figure(
        y, y_pred, label_encoder, model_name, accent, figsize=(9, 7)
    )
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)
    st.caption("Close this dialog to return to the main page.")



def apply_theme(mode: str, accent_name: str) -> dict:
    accent = ACCENT_PALETTES[accent_name]
    if mode == "Dark":
        app_bg = "linear-gradient(180deg, #141210 0%, #1c1815 50%, #12100e 100%)"
        ink = "#f8fafc"
        muted = "#e2e8f0"
        sidebar_bg = "#1a1613"
        sidebar_border = accent["primary"]
        sidebar_ink = "#f8fafc"
        sidebar_muted = "#cbd5e1"
        card_bg = "#2f2721"
        metric_bg = "#322a23"
        heading = accent["border"]
        card_border = accent["primary"]
        card_text = accent["border"]
        alert_info_bg = "#1e3a5f"
        alert_info_border = "#60a5fa"
        alert_info_text = "#dbeafe"
        alert_ok_bg = "#14532d"
        alert_ok_border = "#4ade80"
        alert_ok_text = "#dcfce7"
        metric_label = "#fdba74" if accent_name == "Orange" else accent["border"]
        metric_value = "#ffffff"
        table_header_bg = accent["primary"]
        table_header_fg = "#ffffff"
        input_bg = "#2a231e"
        input_border = accent["border"]
    else:
        app_bg = (
            f"linear-gradient(180deg, {accent['soft']} 0%, #ffffff 42%, "
            f"{accent['soft']} 100%)"
        )
        ink = "#1f1724"
        muted = "#334155"
        sidebar_bg = accent["soft"]
        sidebar_border = accent["border"]
        sidebar_ink = "#1f1724"
        sidebar_muted = "#334155"
        card_bg = "#ffffff"
        metric_bg = accent["soft"]
        heading = accent["heading"]
        card_border = accent["border"]
        card_text = accent["heading"]
        alert_info_bg = "#eff6ff"
        alert_info_border = "#93c5fd"
        alert_info_text = "#1e3a8a"
        alert_ok_bg = "#ecfdf5"
        alert_ok_border = "#6ee7b7"
        alert_ok_text = "#065f46"
        metric_label = accent["heading"]
        metric_value = accent["heading"]
        table_header_bg = accent["primary"]
        table_header_fg = "#ffffff"
        input_bg = "#ffffff"
        input_border = accent["border"]

    st.markdown(
        f"""
<style>
    .stApp {{
        background: {app_bg};
        color: {ink};
    }}
    .stApp p, .stApp li {{
        color: {ink} !important;
    }}
    /* Do not force color on all spans — breaks toolbar / fullscreen icons */
    .stApp label {{
        color: {ink} !important;
    }}
    /* Hide Streamlit chrome clutter, but NEVER remove the sidebar expand control.
       (On mobile that control lives in the header/toolbar area.) */
    #MainMenu {{
        visibility: hidden !important;
    }}
    header[data-testid="stHeader"] {{
        background: transparent !important;
        background-color: transparent !important;
        height: auto !important;
        min-height: 2.5rem !important;
        display: block !important;
        visibility: visible !important;
        z-index: 999990 !important;
    }}
    /* Hide clutter only — do not zero-out the whole toolbar (expand button can live there) */
    div[data-testid="stDecoration"],
    div[data-testid="stStatusWidget"],
    div[data-testid="stToolbarActions"],
    .stAppDeployButton,
    .stDeployButton {{
        display: none !important;
        visibility: hidden !important;
    }}
    /* Keep sidebar open/close controls visible (esp. mobile when sidebar is collapsed) */
    [data-testid="stExpandSidebarButton"],
    [data-testid="stSidebarCollapsedControl"],
    [data-testid="collapsedControl"],
    [data-testid="stSidebarCollapseButton"],
    [data-testid="stExpandSidebarButton"] button,
    [data-testid="stSidebarCollapsedControl"] button,
    [data-testid="collapsedControl"] button {{
        display: flex !important;
        visibility: visible !important;
        opacity: 1 !important;
        pointer-events: auto !important;
        z-index: 1000000 !important;
        width: auto !important;
        height: auto !important;
        min-width: 2.5rem !important;
        min-height: 2.5rem !important;
        color: {ink} !important;
        background: {card_bg} !important;
        background-color: {card_bg} !important;
        border: 1px solid {card_border} !important;
        border-radius: 0.5rem !important;
        box-shadow: 0 1px 4px rgba(15, 23, 42, 0.18) !important;
    }}
    [data-testid="stExpandSidebarButton"] svg,
    [data-testid="stSidebarCollapsedControl"] svg,
    [data-testid="collapsedControl"] svg,
    [data-testid="stExpandSidebarButton"] span,
    [data-testid="stSidebarCollapsedControl"] span,
    [data-testid="collapsedControl"] span {{
        color: {ink} !important;
        fill: {ink} !important;
        opacity: 1 !important;
    }}
    footer {{
        visibility: hidden !important;
        display: none !important;
    }}
    [data-testid="stElementToolbar"] {{
        display: none !important; /* unclear icon-only fullscreen control */
    }}
    .stCaption, div[data-testid="stCaptionContainer"],
    div[data-testid="stCaptionContainer"] p {{
        color: {muted} !important;
        opacity: 1 !important;
    }}
    h1, h2, h3 {{
        color: {heading} !important;
    }}

    /* Sidebar readability — force dark panel in Dark mode */
    section[data-testid="stSidebar"],
    div[data-testid="stSidebar"],
    section[data-testid="stSidebar"] > div:first-child,
    div[data-testid="stSidebar"] > div:first-child,
    [data-testid="stSidebarContent"],
    [data-testid="stSidebarUserContent"] {{
        background: {sidebar_bg} !important;
        background-color: {sidebar_bg} !important;
        background-image: none !important;
    }}
    section[data-testid="stSidebar"],
    div[data-testid="stSidebar"] {{
        border-right: 1px solid {sidebar_border} !important;
    }}
    section[data-testid="stSidebar"] [data-testid="stVerticalBlock"],
    section[data-testid="stSidebar"] [data-testid="stVerticalBlockBorderWrapper"],
    section[data-testid="stSidebar"] [data-testid="element-container"],
    section[data-testid="stSidebar"] [data-testid="stExpander"],
    section[data-testid="stSidebar"] details,
    section[data-testid="stSidebar"] summary {{
        background: transparent !important;
        background-color: transparent !important;
    }}
    section[data-testid="stSidebar"] p,
    section[data-testid="stSidebar"] span,
    section[data-testid="stSidebar"] label,
    section[data-testid="stSidebar"] li,
    section[data-testid="stSidebar"] small,
    div[data-testid="stSidebar"] p,
    div[data-testid="stSidebar"] span,
    div[data-testid="stSidebar"] label,
    div[data-testid="stSidebar"] li {{
        color: {sidebar_ink} !important;
        opacity: 1 !important;
    }}
    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3,
    section[data-testid="stSidebar"] h4,
    section[data-testid="stSidebar"] strong,
    div[data-testid="stSidebar"] h1,
    div[data-testid="stSidebar"] h2,
    div[data-testid="stSidebar"] h3,
    div[data-testid="stSidebar"] h4,
    div[data-testid="stSidebar"] strong {{
        color: {heading} !important;
    }}
    section[data-testid="stSidebar"] .stCaption,
    section[data-testid="stSidebar"] [data-testid="stCaptionContainer"],
    section[data-testid="stSidebar"] [data-testid="stCaptionContainer"] p,
    div[data-testid="stSidebar"] .stCaption,
    div[data-testid="stSidebar"] [data-testid="stCaptionContainer"],
    div[data-testid="stSidebar"] [data-testid="stCaptionContainer"] p {{
        color: {sidebar_muted} !important;
        opacity: 1 !important;
    }}
    section[data-testid="stSidebar"] .theme-card,
    div[data-testid="stSidebar"] .theme-card {{
        background: {card_bg} !important;
        color: {sidebar_ink} !important;
        border: 1px solid {card_border} !important;
        border-left: 5px solid {accent['primary']} !important;
    }}
    section[data-testid="stSidebar"] .theme-card strong,
    div[data-testid="stSidebar"] .theme-card strong {{
        color: {card_text} !important;
    }}
    section[data-testid="stSidebar"] input,
    section[data-testid="stSidebar"] textarea,
    section[data-testid="stSidebar"] [data-baseweb="select"] > div,
    section[data-testid="stSidebar"] [data-baseweb="input"],
    section[data-testid="stSidebar"] [data-baseweb="base-input"],
    div[data-testid="stSidebar"] input,
    div[data-testid="stSidebar"] [data-baseweb="select"] > div {{
        background-color: {input_bg} !important;
        color: {sidebar_ink} !important;
        border-color: {input_border} !important;
        caret-color: {sidebar_ink} !important;
    }}
    section[data-testid="stSidebar"] [data-baseweb="select"] span,
    section[data-testid="stSidebar"] [data-baseweb="select"] div {{
        color: {sidebar_ink} !important;
    }}
    section[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"],
    section[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] * {{
        background-color: {input_bg} !important;
        color: {sidebar_ink} !important;
    }}
    section[data-testid="stSidebar"] hr,
    div[data-testid="stSidebar"] hr {{
        border-color: {sidebar_border} !important;
        opacity: 0.7;
    }}
    /* Keep download buttons readable in sidebar */
    section[data-testid="stSidebar"] .stDownloadButton > button,
    section[data-testid="stSidebar"] .stButton > button {{
        color: #ffffff !important;
    }}

    .stButton > button,
    .stDownloadButton > button,
    button[data-testid="baseButton-primary"],
    button[data-testid="baseButton-secondary"],
    button[data-testid="baseButton-tertiary"],
    div[data-testid="stButton"] button,
    div[data-testid="stDownloadButton"] button {{
        background-color: {accent['primary']} !important;
        background-image: none !important;
        color: #ffffff !important;
        border: 1px solid {accent['primary_dark']} !important;
        border-radius: 8px !important;
        font-weight: 700 !important;
        box-shadow: none !important;
    }}
    .stButton > button:hover,
    .stDownloadButton > button:hover,
    button[data-testid="baseButton-primary"]:hover,
    button[data-testid="baseButton-secondary"]:hover,
    div[data-testid="stButton"] button:hover,
    div[data-testid="stDownloadButton"] button:hover {{
        background-color: {accent['primary_dark']} !important;
        color: #ffffff !important;
        border-color: {accent['primary_dark']} !important;
    }}
    .stButton > button p,
    .stDownloadButton > button p,
    div[data-testid="stButton"] button p,
    div[data-testid="stDownloadButton"] button p {{
        color: #ffffff !important;
    }}
    div[data-testid="stMetric"] {{
        background: {metric_bg};
        border: 1px solid {card_border};
        border-radius: 10px;
        padding: 0.55rem 0.7rem;
    }}
    div[data-testid="stMetric"] label {{
        color: {metric_label} !important;
        font-weight: 600 !important;
    }}
    div[data-testid="stMetric"] [data-testid="stMetricValue"] {{
        color: {metric_value} !important;
        font-weight: 700 !important;
    }}

    /* Colored / sharper table headers */
    div[data-testid="stDataFrame"] {{
        border: 2px solid {table_header_bg} !important;
        border-radius: 10px !important;
        overflow: hidden !important;
    }}
    div[data-testid="stDataFrame"] thead tr th,
    div[data-testid="stDataFrame"] [role="columnheader"],
    .stDataFrame th,
    div[data-testid="stDataFrame"] .col-header,
    div[data-testid="stDataFrame"] [class*="header"] {{
        background-color: {accent['primary_dark']} !important;
        background: linear-gradient(180deg, {accent['primary']} 0%, {accent['primary_dark']} 100%) !important;
        color: #ffffff !important;
        font-weight: 800 !important;
        letter-spacing: 0.03em !important;
        text-transform: none !important;
        border-bottom: 2px solid {accent['primary_dark']} !important;
    }}
    div[data-testid="stDataFrame"] [role="columnheader"] *,
    div[data-testid="stDataFrame"] [class*="header"] * {{
        color: #ffffff !important;
        fill: #ffffff !important;
    }}
    /* HTML tables from Styler */
    table.dataframe thead th,
    .dataframe thead th,
    table thead th {{
        background: linear-gradient(180deg, {accent['primary']} 0%, {accent['primary_dark']} 100%) !important;
        color: #ffffff !important;
        font-weight: 800 !important;
        letter-spacing: 0.03em !important;
        border: 1px solid {accent['primary_dark']} !important;
        text-shadow: 0 1px 0 rgba(0,0,0,0.25);
    }}
    table.dataframe tbody th {{
        background-color: {accent['primary_dark']} !important;
        color: #ffffff !important;
        font-weight: 700 !important;
    }}
    table.dataframe tbody td {{
        border: 1px solid {accent['border']} !important;
    }}
    .model-banner {{
        background: linear-gradient(90deg, {accent['primary']} 0%, {accent['primary_dark']} 100%);
        color: #ffffff;
        border-radius: 12px;
        padding: 0.95rem 1.1rem;
        margin: 0.25rem 0 0.9rem 0;
        font-size: 1.05rem;
        font-weight: 700;
        letter-spacing: 0.01em;
        box-shadow: 0 6px 18px rgba(0,0,0,0.18);
    }}
    .model-banner span {{
        font-weight: 500;
        opacity: 0.95;
    }}

    .eval-banner {{
        background: {alert_ok_bg};
        border: 1px solid {alert_ok_border};
        color: {alert_ok_text};
        border-radius: 10px;
        padding: 0.8rem 1rem;
        margin: 0.35rem 0 0.7rem 0;
        font-weight: 600;
    }}
    .source-banner {{
        background: {alert_info_bg};
        border: 1px solid {alert_info_border};
        color: {alert_info_text};
        border-radius: 10px;
        padding: 0.85rem 1rem;
        margin: 0.2rem 0 0.9rem 0;
        line-height: 1.45;
    }}
    .source-banner strong {{
        color: {alert_info_text};
    }}
    .theme-card {{
        background: {card_bg};
        border: 1px solid {card_border};
        border-left: 5px solid {accent['primary']};
        border-radius: 10px;
        padding: 0.85rem 1rem;
        margin: 0.4rem 0 0.9rem 0;
        color: {ink};
    }}
    .theme-card strong {{ color: {card_text}; }}
    .theme-panel {{
        background: {card_bg};
        border: 1px solid {card_border};
        border-radius: 12px;
        padding: 0.75rem 0.9rem 0.35rem 0.9rem;
        margin-bottom: 0.6rem;
    }}
</style>
""",
        unsafe_allow_html=True,
    )
    return accent


def render_open_sidebar_button(mode: str, accent: dict) -> None:
    """Themeable Streamlit-style >> control to open the sidebar (esp. on mobile)."""
    if mode == "Dark":
        ink = "#f8fafc"
        btn_bg = "#2f2721"
        btn_border = accent["primary"]
        hover_bg = "#3a322b"
    else:
        ink = "#1f1724"
        btn_bg = "#ffffff"
        btn_border = accent["border"]
        hover_bg = accent["soft"]

    # Same visual language as Streamlit's keyboard_double_arrow_right expand control.
    components.html(
        f"""
<div style="display:flex;align-items:center;gap:0.65rem;padding:0.15rem 0;">
  <button id="open-sidebar-btn" aria-label="Open sidebar" title="Open sidebar" style="
    display:inline-flex;
    align-items:center;
    justify-content:center;
    width:2.5rem;
    height:2.5rem;
    min-width:2.5rem;
    min-height:2.5rem;
    margin:0;
    padding:0;
    border:1px solid {btn_border};
    border-radius:0.5rem;
    background:{btn_bg};
    color:{ink};
    box-shadow:0 1px 4px rgba(15,23,42,0.16);
    cursor:pointer;
  ">
    <svg width="22" height="22" viewBox="0 0 24 24" aria-hidden="true" focusable="false">
      <path fill="{ink}" d="M6.49 17.21 11.71 12 6.49 6.79 7.91 5.38 14.53 12l-6.62 6.62-1.42-1.41z"/>
      <path fill="{ink}" d="M12.49 17.21 17.71 12l-5.22-5.21 1.42-1.41L20.53 12l-6.62 6.62-1.42-1.41z"/>
    </svg>
  </button>
  <span style="
    color:{ink};
    font-family:Source Sans Pro,Segoe UI,sans-serif;
    font-size:0.95rem;
    font-weight:600;
  ">Open sidebar</span>
</div>
<script>
(function() {{
  function findAndClick() {{
    const doc = window.parent.document;
    const selectors = [
      '[data-testid="stExpandSidebarButton"]',
      '[data-testid="stExpandSidebarButton"] button',
      '[data-testid="stSidebarCollapsedControl"]',
      '[data-testid="stSidebarCollapsedControl"] button',
      '[data-testid="collapsedControl"]',
      '[data-testid="collapsedControl"] button',
      'button[kind="header"]',
    ];
    for (const sel of selectors) {{
      const el = doc.querySelector(sel);
      if (el) {{ el.click(); return true; }}
    }}
    const buttons = Array.from(doc.querySelectorAll('button'));
    for (const b of buttons) {{
      const label = ((b.getAttribute('aria-label') || '') + ' ' + (b.innerText || '')).toLowerCase();
      if (label.includes('sidebar') || label.includes('expand') || label.includes('keyboard_double_arrow_right')) {{
        b.click();
        return true;
      }}
    }}
    return false;
  }}
  const btn = document.getElementById('open-sidebar-btn');
  btn.addEventListener('mouseenter', function() {{ btn.style.background = '{hover_bg}'; }});
  btn.addEventListener('mouseleave', function() {{ btn.style.background = '{btn_bg}'; }});
  btn.addEventListener('click', function() {{ findAndClick(); }});
}})();
</script>
""",
        height=48,
    )


def main():
    st.set_page_config(
        page_title="Online Shoppers Classifier",
        page_icon="🛒",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    try:
        models, label_encoder, meta = load_artifacts()
    except FileNotFoundError as exc:
        st.error(str(exc))
        st.stop()

    if not DEFAULT_TEST.exists():
        st.error("Missing test_data.csv in the project folder.")
        st.stop()

    full_test = load_default_test()

    # Theme defaults before controls so the mobile sidebar opener can match accent.
    theme_mode_seed = st.session_state.get("theme_mode", "Light")
    accent_name_seed = st.session_state.get("accent_color", "Orange")
    accent_seed = apply_theme(theme_mode_seed, accent_name_seed)
    render_open_sidebar_button(theme_mode_seed, accent_seed)

    # Theme + model on the right; title on the left
    title_col, control_col = st.columns([2.2, 1.1], gap="large")
    with control_col:
        st.markdown("##### Theme")
        theme_mode = st.radio(
            "Appearance",
            ["Light", "Dark"],
            index=0,
            horizontal=True,
            key="theme_mode",
        )
        accent_name = st.selectbox(
            "Accent color",
            list(ACCENT_PALETTES.keys()),
            index=0,  # Orange default
            key="accent_color",
        )
        accent = apply_theme(theme_mode, accent_name)
        st.markdown("##### Model")
        model_name = st.selectbox(
            "Evaluate with this model",
            list(MODEL_FILES.keys()),
            key="model_name_top",
            help="All live metrics, predictions, and the confusion matrix below use this model.",
        )

    with title_col:
        st.title("Online Shoppers Purchasing Intention")
        st.markdown(
            """
Predict whether a browsing session ends in a purchase (`Revenue = True / False`)
using five classical classifiers trained on the UCI Online Shoppers dataset.
"""
        )

    st.markdown(
        f"""
<div class="model-banner">
Currently evaluating with: {model_name}<br/>
<span>Live predictions, metrics, and confusion matrix below are for this model only.</span>
</div>
""",
        unsafe_allow_html=True,
    )

    with st.sidebar:
        st.header("Controls")
        uploaded = st.file_uploader(
            "Upload evaluation CSV (test data)",
            type=["csv"],
            help="Use unseen test rows (kept aside during training) with the same columns.",
        )
        st.caption(f"Active model: **{model_name}** (change it at the top-right)")

        st.markdown("---")
        st.markdown("### Get a test CSV")
        st.markdown(
            """
<div class="theme-card">
<strong>What to do</strong><br/>
1. Download a CSV below.<br/>
2. (Optional) open it and keep the same column names.<br/>
3. Upload it with <em>Upload evaluation CSV</em> above — or skip upload and the app already uses the full test file.
</div>
""",
            unsafe_allow_html=True,
        )

        st.markdown("**Option 1 — full test file**")
        st.caption(
            f"Complete test set kept aside during training ({len(full_test)} rows). "
            "Best when you want the same evaluation used during model comparison."
        )
        st.download_button(
            label="Download full test_data.csv",
            data=full_test.to_csv(index=False).encode("utf-8"),
            file_name="test_data.csv",
            mime="text/csv",
        )

        st.markdown("**Option 2 — smaller random sample**")
        st.caption(
            "Creates a new smaller CSV from the full test set. Useful for a quick check "
            "or as a short template before uploading."
        )
        sample_n = st.slider(
            "How many rows in the sample?",
            min_value=20,
            max_value=300,
            value=50,
            step=10,
            help="Number of rows to include in the downloaded sample CSV.",
        )
        sample_seed = st.number_input(
            "Random seed (controls which rows are picked)",
            min_value=0,
            max_value=10_000,
            value=7,
            step=1,
            help=(
                "Same seed + same sample size = same rows every time. "
                "Change the seed to draw a different random subset."
            ),
        )
        st.markdown(
            f"""
<div class="theme-card">
<strong>Sample settings right now</strong><br/>
• Rows: <strong>{int(sample_n)}</strong><br/>
• Seed: <strong>{int(sample_seed)}</strong><br/>
• Same seed keeps the sample reproducible. Change seed to get different rows.
</div>
""",
            unsafe_allow_html=True,
        )
        sample_bytes = make_sample_csv(full_test, int(sample_n), int(sample_seed))
        st.download_button(
            label=f"Download sample CSV ({sample_n} rows)",
            data=sample_bytes,
            file_name=f"sample_test_data_{sample_n}_seed{int(sample_seed)}.csv",
            mime="text/csv",
        )

        st.markdown("---")
        st.markdown("### Train vs test")
        st.markdown(
            f"""
- **Training:** {meta['n_train']} rows (80%) — used only to fit models  
- **Default evaluation:** {meta['n_test']} test rows kept aside during training (`test_data.csv`)  
- **Upload:** scores the CSV provided in the uploader  

Live metrics and the confusion matrix always use the **evaluation file**, never the training set.
"""
        )
        st.caption(f"Classes: {', '.join(meta['classes'])}")


    if uploaded is not None:
        df = pd.read_csv(uploaded)
        source_label = "your uploaded CSV"
        eval_scope = "CUSTOM EVALUATION FILE"
        data_note = (
            "<strong>Evaluation source:</strong> your uploaded CSV.<br/>"
            "Live metrics and the confusion matrix use <strong>these rows only</strong> "
            "(unseen / evaluation data — <strong>not</strong> the training set)."
        )
        bundle_note = ""
    else:
        df = full_test.copy()
        source_label = "default test file (test_data.csv)"
        eval_scope = "RESERVED TEST SET"
        data_note = (
            "<strong>Evaluation source:</strong> default project file "
            "<code>test_data.csv</code>.<br/>"
            "These rows were <strong>kept aside during training</strong> "
            f"(20% test split, {meta['n_test']} rows, "
            f"<code>random_state={meta['random_state']}</code>).<br/>"
            "Live metrics and the confusion matrix are on this <strong>test</strong> data — "
            "<strong>not</strong> on training data."
        )
        bundle_note = f"""
<div class="theme-card">
<strong>What “default / bundled test_data.csv” means</strong><br/>
• It is a CSV file saved inside this project (next to <code>app.py</code>).<br/>
• “Bundled” just means it ships with the app — no upload needed to start.<br/>
• It contains the <strong>test rows kept aside while training</strong>
  ({meta['n_test']} sessions), so the models are checked on data they did not learn from.<br/>
• You can still download it from the sidebar, edit it, or replace it by uploading another CSV.
</div>
"""

    st.markdown(
        f'<div class="eval-banner">Current evaluation scope: {eval_scope}</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        f'<div class="source-banner">{data_note}</div>',
        unsafe_allow_html=True,
    )
    if bundle_note:
        st.markdown(bundle_note, unsafe_allow_html=True)

    with st.expander("Dataset columns — what each field means", expanded=False):
        st.markdown(
            """
Each row is one browsing **session**. Features describe what the visitor did;
`Revenue` says whether that session ended in a purchase.
"""
        )
        help_rows = [
            {"Column": col, "Meaning": COLUMN_HELP.get(col, "Feature used by the model.")}
            for col in list(meta["feature_order"]) + [meta["target"]]
        ]
        help_df = pd.DataFrame(help_rows)
        render_colored_table(help_df, accent)

    with st.expander("How to prepare a test CSV", expanded=False):
        st.markdown(
            f"""
### Goal
Build a CSV the app can score. For a fair check, use rows the model did **not** train on.

### Step A — start from a template
1. Download **full test_data.csv** or a **dynamic sample** from the sidebar.  
2. Open it in Excel / Sheets / pandas.  
3. Keep the header row exactly as-is.

### Step B — put your own rows (optional)
Replace data rows with your sessions, but keep the same column names:

`{'`, `'.join(meta['feature_order'])}`

and ideally:

`Revenue` (True/False) so metrics and the confusion matrix can be computed.

### Step C — upload
Use **Upload evaluation CSV** in the sidebar. The app will switch from the default
project file (`test_data.csv`) to your file.

### What happens after upload

| What is in the CSV | What the app shows |
|--------------------|--------------------|
| Feature columns only | Predictions + purchase probabilities for each row |
| Features + `Revenue` | Everything above **plus** Accuracy / AUC / Precision / Recall / F1 / MCC, confusion matrix, and classification report on **your file** |
| Wrong / missing feature columns | Error message listing the missing columns |

### How this differs from the “Saved model comparison” table
- **Live metrics** = scored right now on whatever file is loaded (upload or default test set).  
- **Saved comparison table** = fixed scores from the original training run on the full reserved `test_data.csv`.  
  If the upload is a small sample or different rows, live numbers can differ from that table — that is expected.
"""
        )

    st.markdown("## 1) Evaluation data")
    st.caption(f"Source: **{source_label}** · {len(df)} rows")
    render_colored_table(df, accent, max_rows=10)
    st.caption("First 10 rows of the file being scored.")
    if st.button(
        "Open full screen table — evaluation data",
        key="full_eval_preview",
        type="primary",
    ):
        show_full_table_dialog(df, accent, f"Evaluation data — {source_label}")

    try:
        X, y = prepare_xy(df, meta, label_encoder)
    except Exception as exc:
        st.error(f"Could not prepare data: {exc}")
        st.stop()

    model = models[model_name]
    y_pred = model.predict(X)
    y_proba = model.predict_proba(X)[:, 1]
    pred_labels = label_encoder.inverse_transform(y_pred)

    st.markdown(f"## 2) Live results for **{model_name}**")
    st.markdown(
        f"""
<div class="theme-card">
<strong>What this section does</strong><br/>
The selected model (<strong>{model_name}</strong>) scores each row in the evaluation file above.<br/>
Everything here (predictions, metrics, confusion matrix, classification report) updates when you
change the model or upload a different CSV.<br/>
This is <strong>not</strong> the saved comparison table at the bottom.
</div>
""",
        unsafe_allow_html=True,
    )

    st.markdown(f"### Predictions from {model_name}")
    st.markdown(
        f"""
<div class="theme-card">
<strong>How to read this table</strong><br/>
• Each row = one browsing session from your evaluation file.<br/>
• <strong>S.No</strong> = serial number for that row in the evaluation file.<br/>
• <strong>Predicted_Revenue</strong> = what <em>{model_name}</em> decides
  (<code>True</code> = purchase, <code>False</code> = no purchase).<br/>
• <strong>Purchase_Probability</strong> = model’s estimated chance the session is a purchase
  (closer to 1.0 → more confident it is a purchase).<br/>
• Other columns are the original session features (and <code>Revenue</code> if present = true label).<br/>
• This table is <strong>row-level output</strong> for the selected model — it is not a scoreboard of all models.
</div>
""",
        unsafe_allow_html=True,
    )
    preview = df.copy()
    preview.insert(0, "S.No", range(1, len(preview) + 1))
    preview["Predicted_Revenue"] = pred_labels
    preview["Purchase_Probability"] = np.round(y_proba, 4)
    render_colored_table(preview, accent, max_rows=20)
    st.caption(f"Showing first 20 of {len(preview)} predicted rows.")
    if st.button(
        "Open full screen table — predictions",
        key="full_predictions",
        type="primary",
    ):
        show_full_table_dialog(preview, accent, f"Predictions — {model_name}")

    if y is None:
        st.warning(
            "No `Revenue` column in this CSV, so only predictions are shown. "
            "Add true labels to see evaluation metrics and a confusion matrix."
        )
        st.stop()

    metrics = compute_metrics(y, y_pred, y_proba)

    st.markdown(f"### Scores on the current evaluation file — {model_name}")
    st.caption(
        f"Summary metrics for **{model_name}** on `{source_label}` "
        "(same rows as the prediction table)."
    )
    cols = st.columns(6)
    for col, (name, value) in zip(cols, metrics.items()):
        col.metric(name, f"{value:.4f}")

    left, right = st.columns(2)
    with left:
        st.markdown("### Confusion matrix")
        st.caption(
            f"{model_name} on `{source_label}` · rows = actual Revenue, columns = predicted Revenue"
        )
        fig = make_confusion_figure(
            y, y_pred, label_encoder, model_name, accent, figsize=(5, 4)
        )
        st.pyplot(fig)
        plt.close(fig)
        if st.button(
            "Open full screen — confusion matrix",
            key="full_confusion_matrix",
            type="primary",
            help="Opens a larger view of the confusion matrix image",
        ):
            show_confusion_matrix_dialog(
                y, y_pred, label_encoder, model_name, accent, source_label
            )

    with right:
        st.markdown("### Classification report")
        st.caption(
            f"Per-class precision / recall / F1 / support for **{model_name}** "
            f"on the same evaluation file ({len(df)} rows)."
        )
        with st.expander("What each classification-report row means"):
            st.markdown(
                """
| Row | Meaning |
|-----|---------|
| `False` / `True` | Per-class scores for no-purchase / purchase |
| `precision` | Of rows predicted as that class, how many were correct |
| `recall` | Of actual rows of that class, how many were found |
| `f1-score` | Balance of precision and recall |
| `support` | How many evaluation rows belong to that class |
| `accuracy` | Overall share correct on this evaluation file |
| `macro avg` | Unweighted average across classes |
| `weighted avg` | Average weighted by `support` |
"""
            )
        report = classification_report(
            y,
            y_pred,
            target_names=[str(c) for c in label_encoder.classes_],
            output_dict=True,
            zero_division=0,
        )
        report_df = pd.DataFrame(report).T.reset_index().rename(columns={"index": "Label"})
        render_colored_table(report_df, accent, float_format="{:.4f}")
        if st.button(
            "Open full screen table — classification report",
            key="full_report",
            type="primary",
        ):
            show_full_table_dialog(
                report_df, accent, f"Classification report — {model_name}"
            )

    st.markdown("### Purchase probability distribution")
    st.caption(
        f"How often **{model_name}** assigns low vs high purchase probability "
        "across the current evaluation file."
    )
    fig2, ax2 = plt.subplots(figsize=(7, 3.5))
    sns.histplot(y_proba, bins=30, ax=ax2, color=accent["primary"])
    ax2.set_xlabel("P(Revenue = True)")
    ax2.set_ylabel("Count")
    st.pyplot(fig2)
    plt.close(fig2)

    st.markdown("---")
    st.markdown("## 3) Reference — compare all models (from training)")
    st.markdown(
        f"""
<div class="theme-card">
<strong>Keep this separate from Section 2.</strong><br/><br/>
<strong>What this table is for</strong><br/>
A side-by-side scoreboard of all five models, saved when training finished.
Each row is one model’s score on the full reserved <code>test_data.csv</code>
({meta['n_test']} rows).<br/><br/>
<strong>Use it to</strong> pick an overall winner / compare models.<br/>
<strong>Unlike Section 2:</strong> this table does not change when you switch the model or upload a new CSV.
Section 2 always shows live results for the one model chosen at the top, on the file currently loaded.
</div>
""",
        unsafe_allow_html=True,
    )
    stored = pd.DataFrame(meta["metrics"]).T.reset_index().rename(columns={"index": "Model"})
    metric_cols = ["Model", "Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]
    stored = stored[[c for c in metric_cols if c in stored.columns]]
    render_colored_table(stored, accent, float_format="{:.4f}")
    st.caption(
        "Tip: if you uploaded a custom/sample CSV, live metrics in Section 2 can differ from this reference table."
    )


if __name__ == "__main__":
    main()
