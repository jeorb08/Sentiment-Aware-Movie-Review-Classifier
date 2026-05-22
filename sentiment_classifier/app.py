"""
Sentiment-Aware Movie Review Classifier — Streamlit Dashboard
=============================================================
Run with:  streamlit run app.py

Requires (run notebooks 01-04 first):
  models/distilbert-imdb/          — fine-tuned DistilBERT
  models/tfidf_logreg_baseline.joblib
  data/test_predictions.csv        — from notebook 03
  data/model_comparison.csv        — from notebook 03
  data/shap_comparison.png         — from notebook 04
  data/training_curves.png         — from notebook 03
  shap_helper.py                   — from notebook 04
"""

# ── Imports ───────────────────────────────────────────────────────────────────
import os
import sys
import json
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import torch
import shap
import joblib
import streamlit as st
from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification,
    pipeline,
)
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from sklearn.metrics import (
    confusion_matrix, roc_curve, roc_auc_score
)

# ── Must be first Streamlit call ──────────────────────────────────────────────
st.set_page_config(
    page_title = "Movie Sentiment Classifier",
    page_icon  = "🎬",
    layout     = "wide",
    initial_sidebar_state = "expanded",
)

# ── Paths ─────────────────────────────────────────────────────────────────────
MODEL_DIR      = "models/distilbert-imdb"
BASELINE_PATH  = "models/tfidf_logreg_baseline.joblib"
PREDICTIONS    = "data/test_predictions.csv"
COMPARISON     = "data/model_comparison.csv"
SHAP_IMG       = "data/shap_comparison.png"
TRAINING_IMG   = "data/training_curves.png"
CONF_IMG       = "data/confidence_deep_dive.png"

# ── Colour palette ────────────────────────────────────────────────────────────
COLOURS = {
    "positive": "#2E9E6B",
    "negative": "#E74C3C",
    "neutral":  "#3B8BD0",
    "bg":       "#0F1117",
    "card":     "#1A1D2E",
    "warning":  "#E67E22",
}

# ── Label map ─────────────────────────────────────────────────────────────────
LABEL_MAP = {"LABEL_0": "NEGATIVE", "LABEL_1": "POSITIVE"}

# ── Example reviews ───────────────────────────────────────────────────────────
EXAMPLE_REVIEWS = {
    "🟢 Clear positive": (
        "An absolute masterpiece. The performances were breathtaking, "
        "the cinematography stunning, and the story deeply moving. "
        "One of the finest films I have ever had the privilege of watching. "
        "I left the cinema with tears in my eyes and hope in my heart."
    ),
    "🔴 Clear negative": (
        "A complete waste of two hours. The plot made no sense, "
        "the acting was wooden, and the dialogue cringe-worthy. "
        "I cannot believe this was given a theatrical release. "
        "Avoid at all costs."
    ),
    "🟡 Mixed sentiment": (
        "Stunning visuals and an incredible score, but the screenplay "
        "lets everything down badly. The first act is gripping, then it "
        "completely falls apart. A real disappointment given the talent involved."
    ),
    "🟠 Sarcasm (tricky)": (
        "Oh brilliant, just what cinema needed — another generic superhero "
        "sequel with zero originality. The director has truly outdone himself "
        "in producing the most predictable film of the decade."
    ),
    "🔵 Negation (tricky)": (
        "Not since The Godfather have I seen such a commanding performance. "
        "This film is not to be missed under any circumstances. "
        "I have never been so moved by a piece of cinema in my life."
    ),
}


# ══════════════════════════════════════════════════════════════════════════════
# CACHED LOADERS  (run once per session)
# ══════════════════════════════════════════════════════════════════════════════
@st.cache_resource
def load_pipeline():
    """Load fine-tuned DistilBERT as a HuggingFace pipeline."""
    if not os.path.exists(MODEL_DIR):
        return None
    clf = pipeline(
        "text-classification",
        model      = MODEL_DIR,
        tokenizer  = MODEL_DIR,
        device     = 0 if torch.cuda.is_available() else -1,
        truncation = True,
        max_length = 512,
        top_k      = None,
    )
    return clf


@st.cache_resource
def load_shap_explainer(_clf):          # ← underscore added
    """Build SHAP explainer (cached — slow to build)."""
    if _clf is None:                    # ← underscore added
        return None
    from transformers import DistilBertTokenizerFast
    tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_DIR)
    masker    = shap.maskers.Text(tokenizer)
    explainer = shap.Explainer(
        _clf,                           # ← underscore added
        masker       = masker,
        output_names = ["NEGATIVE", "POSITIVE"],
    )
    return explainer

@st.cache_resource
def load_baseline():
    """Load TF-IDF + LogReg baseline model."""
    if not os.path.exists(BASELINE_PATH):
        return None
    return joblib.load(BASELINE_PATH)


@st.cache_data
def load_test_predictions():
    """Load test set predictions from notebook 03."""
    if not os.path.exists(PREDICTIONS):
        return None
    return pd.read_csv(PREDICTIONS)


@st.cache_data
def load_model_comparison():
    """Load model comparison table from notebook 03."""
    if not os.path.exists(COMPARISON):
        return None
    return pd.read_csv(COMPARISON)


@st.cache_data
def load_training_meta():
    """Load training metadata JSON saved by notebook 03."""
    meta_path = os.path.join(MODEL_DIR, "training_meta.json")
    if not os.path.exists(meta_path):
        return {}
    with open(meta_path) as f:
        return json.load(f)


# ══════════════════════════════════════════════════════════════════════════════
# PREDICTION HELPERS
# ══════════════════════════════════════════════════════════════════════════════
def predict_sentiment(clf, text: str) -> dict:
    """
    Run DistilBERT inference on one review.
    Returns dict with label, confidence, pos_score, neg_score.
    """
    results = clf([text])[0]
    scores  = {LABEL_MAP[item["label"]]: item["score"] for item in results}
    label   = max(scores, key=scores.get)
    return {
        "label":      label,
        "confidence": scores[label],
        "pos_score":  scores.get("POSITIVE", 0.0),
        "neg_score":  scores.get("NEGATIVE", 0.0),
    }


def predict_baseline(baseline, text: str) -> dict:
    """Run TF-IDF + LogReg baseline on one review."""
    import re
    clean = re.sub(r"<[^>]+>", " ", text)
    clean = re.sub(r"[^a-zA-Z\s']", " ", clean).lower()
    clean = " ".join(clean.split())

    pred  = baseline.predict([clean])[0]
    proba = baseline.predict_proba([clean])[0]
    label = "POSITIVE" if pred == 1 else "NEGATIVE"
    return {
        "label":      label,
        "confidence": proba[pred],
        "pos_score":  proba[1],
        "neg_score":  proba[0],
    }


def compute_shap(explainer, text: str):
    """Compute SHAP values for one review. Returns shap_values object."""
    return explainer([text], fixed_context=1)


def shap_to_html(shap_values, review_idx: int = 0,
                 class_idx: int = 1) -> str:
    """Render SHAP token heatmap as HTML string."""
    tokens  = shap_values.data[review_idx]
    values  = shap_values.values[review_idx, :, class_idx]
    max_abs = max(abs(values.max()), abs(values.min()), 1e-6)
    norm    = values / max_abs

    parts = []
    for token, val, norm_val in zip(tokens, values, norm):
        if token in {"[CLS]", "[SEP]", "[PAD]"}:
            continue
        display_token = token.replace("##", "")
        opacity = min(abs(norm_val) * 0.85 + 0.1, 0.90)
        if norm_val > 0:
            r, g, b = 46, 158, 107
        else:
            r, g, b = 231, 76, 60
        bg = f"rgba({r},{g},{b},{opacity:.2f})"
        parts.append(
            f'<span title="SHAP: {val:+.4f}" style="'
            f'background:{bg};padding:3px 5px;margin:2px;'
            f'border-radius:4px;font-family:monospace;'
            f'font-size:14px;display:inline-block;">'
            f'{display_token}</span>'
        )

    legend = (
        '<div style="margin-top:12px;font-size:12px;color:#888;">'
        '<span style="background:rgba(46,158,107,0.6);'
        'padding:2px 8px;border-radius:3px;">positive signal</span>'
        '&nbsp;&nbsp;'
        '<span style="background:rgba(231,76,60,0.6);'
        'padding:2px 8px;border-radius:3px;">negative signal</span>'
        '&nbsp;&nbsp;Hover each token for exact SHAP value'
        '</div>'
    )
    body = (
        '<div style="line-height:2.6;padding:8px;">'
        + " ".join(parts) + '</div>'
    )
    return f'<div style="font-family:sans-serif;">{body}{legend}</div>'


# ══════════════════════════════════════════════════════════════════════════════
# PLOTLY CHARTS
# ══════════════════════════════════════════════════════════════════════════════
def gauge_chart(score: float, label: str, colour: str) -> go.Figure:
    """Circular gauge showing prediction confidence."""
    fig = go.Figure(go.Indicator(
        mode  = "gauge+number+delta",
        value = score * 100,
        title = {"text": f"{label}<br><span style='font-size:12px'>confidence</span>"},
        number= {"suffix": "%", "font": {"size": 28}},
        gauge = {
            "axis":      {"range": [0, 100], "tickwidth": 1},
            "bar":       {"color": colour, "thickness": 0.25},
            "bgcolor":   "white",
            "borderwidth": 1,
            "steps": [
                {"range": [0,  50], "color": "#f0f0f0"},
                {"range": [50, 70], "color": "#e8e8e8"},
                {"range": [70, 90], "color": "#d8d8d8"},
                {"range": [90,100], "color": "#c8c8c8"},
            ],
            "threshold": {
                "line":  {"color": colour, "width": 3},
                "thickness": 0.8,
                "value": score * 100,
            },
        },
    ))
    fig.update_layout(
        height          = 220,
        margin          = dict(l=20, r=20, t=40, b=10),
        paper_bgcolor   = "rgba(0,0,0,0)",
        plot_bgcolor    = "rgba(0,0,0,0)",
        font            = {"color": "#E0E0E0"},
    )
    return fig


def prob_bar_chart(pos_score: float, neg_score: float) -> go.Figure:
    """Horizontal bar chart showing positive vs negative probability."""
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x            = [pos_score * 100],
        y            = [""],
        orientation  = "h",
        name         = "Positive",
        marker_color = COLOURS["positive"],
        text         = [f"Positive: {pos_score*100:.1f}%"],
        textposition = "inside",
        insidetextanchor = "middle",
    ))
    fig.add_trace(go.Bar(
        x            = [neg_score * 100],
        y            = [""],
        orientation  = "h",
        name         = "Negative",
        marker_color = COLOURS["negative"],
        text         = [f"Negative: {neg_score*100:.1f}%"],
        textposition = "inside",
        insidetextanchor = "middle",
    ))
    fig.update_layout(
        barmode         = "stack",
        height          = 80,
        margin          = dict(l=10, r=10, t=10, b=10),
        showlegend      = False,
        paper_bgcolor   = "rgba(0,0,0,0)",
        plot_bgcolor    = "rgba(0,0,0,0)",
        xaxis           = dict(range=[0, 100], showticklabels=False,
                               showgrid=False, zeroline=False),
        yaxis           = dict(showticklabels=False, showgrid=False),
    )
    return fig


def confusion_matrix_chart(test_df: pd.DataFrame) -> go.Figure:
    """Plotly confusion matrix heatmap."""
    cm = confusion_matrix(test_df["label"], test_df["pred_label"])
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    annotations = []
    for i in range(2):
        for j in range(2):
            annotations.append(dict(
                x=j, y=i,
                text=f"<b>{cm[i,j]:,}</b><br>({cm_norm[i,j]*100:.1f}%)",
                showarrow=False,
                font=dict(
                    color="white" if cm_norm[i,j] > 0.55 else "#333",
                    size=13,
                ),
            ))

    fig = go.Figure(go.Heatmap(
        z          = cm_norm,
        x          = ["Negative", "Positive"],
        y          = ["Negative", "Positive"],
        colorscale = "Blues",
        zmin=0, zmax=1,
        showscale  = True,
        colorbar   = dict(title="Rate", tickformat=".0%"),
    ))
    fig.update_layout(
        annotations   = annotations,
        xaxis_title   = "Predicted",
        yaxis_title   = "Actual",
        height        = 320,
        margin        = dict(l=10, r=10, t=30, b=10),
        paper_bgcolor = "rgba(0,0,0,0)",
        plot_bgcolor  = "rgba(0,0,0,0)",
        font          = dict(color="#E0E0E0"),
    )
    return fig


def roc_chart(test_df: pd.DataFrame, baseline=None) -> go.Figure:
    """ROC curve comparing DistilBERT and TF-IDF baseline."""
    import re

    fig = go.Figure()

    # DistilBERT ROC
    fpr, tpr, _ = roc_curve(test_df["label"], test_df["prob_pos"])
    auc         = roc_auc_score(test_df["label"], test_df["prob_pos"])
    fig.add_trace(go.Scatter(
        x=fpr, y=tpr, mode="lines",
        name=f"DistilBERT (AUC={auc:.3f})",
        line=dict(color=COLOURS["neutral"], width=2.5),
    ))

    # TF-IDF baseline ROC
    if baseline is not None and "clean_text" not in test_df.columns:
        def clean(text):
            text = re.sub(r"<[^>]+>", " ", text)
            text = re.sub(r"[^a-zA-Z\s']", " ", text).lower()
            return " ".join(text.split())
        test_df["clean_text"] = test_df["text"].apply(clean)

    if baseline is not None and "clean_text" in test_df.columns:
        try:
            base_probs  = baseline.predict_proba(
                test_df["clean_text"]
            )[:, 1]
            fpr_b, tpr_b, _ = roc_curve(test_df["label"], base_probs)
            auc_b           = roc_auc_score(test_df["label"], base_probs)
            fig.add_trace(go.Scatter(
                x=fpr_b, y=tpr_b, mode="lines",
                name=f"TF-IDF baseline (AUC={auc_b:.3f})",
                line=dict(color=COLOURS["warning"], width=1.8,
                          dash="dash"),
            ))
        except Exception:
            pass

    # Diagonal
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1], mode="lines",
        name="Random (AUC=0.500)",
        line=dict(color="#666", width=1, dash="dot"),
    ))

    fig.update_layout(
        xaxis_title   = "False positive rate",
        yaxis_title   = "True positive rate",
        height        = 320,
        margin        = dict(l=10, r=10, t=30, b=10),
        legend        = dict(x=0.6, y=0.1, font=dict(size=10)),
        paper_bgcolor = "rgba(0,0,0,0)",
        plot_bgcolor  = "rgba(0,0,0,0)",
        font          = dict(color="#E0E0E0"),
        xaxis         = dict(range=[-0.01, 1.01], gridcolor="#333"),
        yaxis         = dict(range=[-0.01, 1.01], gridcolor="#333"),
    )
    return fig


def confidence_hist(test_df: pd.DataFrame) -> go.Figure:
    """Histogram of confidence scores by correct / incorrect."""
    fig = go.Figure()
    for is_correct, name, colour in [
        (True,  "Correct",   COLOURS["positive"]),
        (False, "Incorrect", COLOURS["negative"]),
    ]:
        subset = test_df[test_df["correct"] == is_correct]["confidence"]
        fig.add_trace(go.Histogram(
            x=subset, name=name,
            marker_color=colour, opacity=0.65,
            nbinsx=40, bingroup=1,
        ))
    fig.update_layout(
        barmode       = "overlay",
        xaxis_title   = "Prediction confidence",
        yaxis_title   = "Count",
        height        = 280,
        margin        = dict(l=10, r=10, t=20, b=10),
        legend        = dict(font=dict(size=10)),
        paper_bgcolor = "rgba(0,0,0,0)",
        plot_bgcolor  = "rgba(0,0,0,0)",
        font          = dict(color="#E0E0E0"),
        xaxis         = dict(gridcolor="#333"),
        yaxis         = dict(gridcolor="#333"),
    )
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# CSS
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
  .stApp { background-color: #0F1117; }
  section[data-testid="stSidebar"] { background-color: #1A1D2E; }

  div[data-testid="metric-container"] {
    background-color: #1A1D2E;
    border: 1px solid #2A2D3E;
    border-radius: 10px;
    padding: 14px 18px;
  }

  .result-positive {
    background: rgba(46,158,107,0.12);
    border-left: 4px solid #2E9E6B;
    border-radius: 0 8px 8px 0;
    padding: 14px 18px;
    margin: 10px 0;
  }
  .result-negative {
    background: rgba(231,76,60,0.12);
    border-left: 4px solid #E74C3C;
    border-radius: 0 8px 8px 0;
    padding: 14px 18px;
    margin: 10px 0;
  }

  .shap-box {
    background: #1A1D2E;
    border: 1px solid #2A2D3E;
    border-radius: 10px;
    padding: 16px;
    margin: 10px 0;
  }

  .section-header {
    font-size: 17px;
    font-weight: 600;
    color: #E0E0E0;
    margin: 22px 0 10px;
    padding-bottom: 6px;
    border-bottom: 1px solid #2A2D3E;
  }

  .stTabs [data-baseweb="tab"] { color: #aaa; }
  .stTabs [aria-selected="true"] { color: #3B8BD0 !important; }
  .stTextArea textarea { font-size: 14px !important; }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# LOAD ALL RESOURCES
# ══════════════════════════════════════════════════════════════════════════════
clf        = load_pipeline()
baseline   = load_baseline()
test_df    = load_test_predictions()
comparison = load_model_comparison()
meta       = load_training_meta()

# Build SHAP explainer (only if model loaded)
explainer  = load_shap_explainer(clf) if clf else None   # call stays the same ✓
# Check what is available
model_ready    = clf is not None
baseline_ready = baseline is not None
data_ready     = test_df is not None


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 🎬 Movie Sentiment")
    st.markdown("**DistilBERT fine-tuned on IMDb**")
    st.divider()

    # Status indicators
    st.markdown("**System status:**")
    st.markdown(
        f"{'✅' if model_ready    else '❌'} DistilBERT model"
    )
    st.markdown(
        f"{'✅' if baseline_ready else '❌'} TF-IDF baseline"
    )
    st.markdown(
        f"{'✅' if data_ready     else '❌'} Test predictions"
    )
    st.markdown(
        f"{'✅' if explainer      else '❌'} SHAP explainer"
    )

    st.divider()

    # Training metadata
    if meta:
        st.markdown("**Model card:**")
        st.markdown(f"Base model: `{meta.get('model_name','—')}`")
        st.markdown(f"Max length: `{meta.get('max_length','—')}` tokens")
        st.markdown(f"Epochs: `{meta.get('num_epochs','—')}`")
        st.markdown(f"LR: `{meta.get('learning_rate','—')}`")
        st.markdown(
            f"Best val F1: `{meta.get('best_val_f1',0)*100:.2f}%`"
        )
        st.markdown(
            f"Best val acc: `{meta.get('best_val_acc',0)*100:.2f}%`"
        )

    st.divider()

    # SHAP toggle
    run_shap = st.toggle(
        "Enable SHAP explanations",
        value = True,
        help  = "SHAP adds ~15–30s per prediction on CPU. "
                "Disable for instant results.",
    )

    st.divider()
    st.markdown("""
    **Dataset:** IMDb (50k reviews)
    **Labels:** 0=negative, 1=positive
    **Baseline:** TF-IDF + LogReg

    **Alert thresholds:**
    - Confidence < 65% → uncertain
    - Confidence > 90% → high confidence

    **Links:**
    - [HuggingFace Hub](#)
    - [GitHub repo](#)
    """)


# ══════════════════════════════════════════════════════════════════════════════
# HEADER
# ══════════════════════════════════════════════════════════════════════════════
st.title("🎬 Sentiment-Aware Movie Review Classifier")
st.caption(
    "Fine-tuned DistilBERT · IMDb 50k · SHAP token-level explanations · "
    "TF-IDF baseline comparison"
)

if not model_ready:
    st.error(
        "❌ Model not found at `models/distilbert-imdb/`. "
        "Run `notebook_03_finetune.ipynb` first."
    )
    st.stop()


# ══════════════════════════════════════════════════════════════════════════════
# TABS
# ══════════════════════════════════════════════════════════════════════════════
tab1, tab2, tab3, tab4 = st.tabs([
    "🔍 Live Classifier",
    "📊 Model Performance",
    "🧠 SHAP Analysis",
    "📋 Test Predictions",
])


# ─────────────────────────────────────────────────────────────────────────────
# TAB 1 — Live Classifier
# ─────────────────────────────────────────────────────────────────────────────
with tab1:

    # ── Example buttons ───────────────────────────────────────────────────────
    st.markdown(
        '<div class="section-header">Quick examples</div>',
        unsafe_allow_html=True,
    )

    cols = st.columns(len(EXAMPLE_REVIEWS))
    selected_example = None
    for col, (label, text) in zip(cols, EXAMPLE_REVIEWS.items()):
        with col:
            if st.button(label, use_container_width=True):
                selected_example = text

    # ── Text input ────────────────────────────────────────────────────────────
    st.markdown(
        '<div class="section-header">Enter a movie review</div>',
        unsafe_allow_html=True,
    )

    review_text = st.text_area(
        label       = "Review text",
        value       = selected_example or "",
        height      = 150,
        placeholder = "Paste or type a movie review here...",
        label_visibility = "collapsed",
    )

    col_btn, col_clear = st.columns([1, 5])
    with col_btn:
        classify_btn = st.button(
            "Classify →", type="primary", use_container_width=True
        )

    # ── Run classification ────────────────────────────────────────────────────
    if classify_btn and review_text.strip():

        with st.spinner("Running DistilBERT inference..."):
            distilbert_result = predict_sentiment(clf, review_text)

        baseline_result = None
        if baseline_ready:
            baseline_result = predict_baseline(baseline, review_text)

        label      = distilbert_result["label"]
        confidence = distilbert_result["confidence"]
        pos_score  = distilbert_result["pos_score"]
        neg_score  = distilbert_result["neg_score"]

        # ── Result banner ─────────────────────────────────────────────────────
        result_class = (
            "result-positive" if label == "POSITIVE" else "result-negative"
        )
        emoji = "🟢" if label == "POSITIVE" else "🔴"
        conf_warning = (
            " ⚠️ Low confidence — review may be ambiguous"
            if confidence < 0.65 else ""
        )

        st.markdown(
            f'<div class="{result_class}">'
            f'<span style="font-size:22px;font-weight:600;">'
            f'{emoji} {label}</span>'
            f'<span style="font-size:15px;color:#aaa;margin-left:14px;">'
            f'{confidence*100:.1f}% confidence{conf_warning}</span>'
            f'</div>',
            unsafe_allow_html=True,
        )

        # ── Metric cards ──────────────────────────────────────────────────────
        st.markdown(
            '<div class="section-header">Prediction scores</div>',
            unsafe_allow_html=True,
        )

        col_gauge, col_probs, col_base = st.columns([2, 3, 2])

        with col_gauge:
            colour = (
                COLOURS["positive"] if label == "POSITIVE"
                else COLOURS["negative"]
            )
            st.plotly_chart(
                gauge_chart(confidence, label, colour),
                use_container_width=True,
            )

        with col_probs:
            st.markdown(
                "**Probability split** — positive vs negative"
            )
            st.plotly_chart(
                prob_bar_chart(pos_score, neg_score),
                use_container_width=True,
            )
            st.metric("Positive score", f"{pos_score*100:.1f}%")
            st.metric("Negative score", f"{neg_score*100:.1f}%")
            st.metric(
                "Word count",
                f"{len(review_text.split()):,}",
                help="DistilBERT truncates at ~380 words (512 tokens)",
            )

        with col_base:
            if baseline_result:
                st.markdown("**TF-IDF baseline**")
                b_label  = baseline_result["label"]
                b_conf   = baseline_result["confidence"]
                b_emoji  = "🟢" if b_label == "POSITIVE" else "🔴"
                b_match  = "✅ Agrees" if b_label == label else "⚠️ Disagrees"
                st.markdown(
                    f"{b_emoji} **{b_label}** ({b_conf*100:.1f}%)"
                )
                st.markdown(f"{b_match} with DistilBERT")
                st.caption(
                    "Disagreement = potentially ambiguous review. "
                    "DistilBERT handles context; TF-IDF uses word frequency only."
                )

        # ── SHAP explanation ──────────────────────────────────────────────────
        if run_shap and explainer is not None:
            st.markdown(
                '<div class="section-header">'
                '🧠 SHAP token explanation — which words drove this prediction?'
                '</div>',
                unsafe_allow_html=True,
            )

            with st.spinner(
                "Computing SHAP values (~15–30s on CPU)..."
            ):
                shap_vals = compute_shap(explainer, review_text)

            html = shap_to_html(shap_vals, review_idx=0, class_idx=1)
            st.markdown(
                f'<div class="shap-box">{html}</div>',
                unsafe_allow_html=True,
            )

            # Top tokens table
            tokens = shap_vals.data[0]
            values = shap_vals.values[0, :, 1]
            skip   = {"[CLS]", "[SEP]", "[PAD]"}
            pairs  = [
                (t.replace("##", ""), v)
                for t, v in zip(tokens, values)
                if t not in skip and not t.startswith("##")
                and len(t) > 1
            ]
            pairs_sorted = sorted(pairs, key=lambda x: abs(x[1]),
                                  reverse=True)[:10]

            shap_table = pd.DataFrame(pairs_sorted,
                                      columns=["Token", "SHAP value"])
            shap_table["Direction"] = shap_table["SHAP value"].apply(
                lambda v: "↑ Positive" if v > 0 else "↓ Negative"
            )
            shap_table["SHAP value"] = shap_table["SHAP value"].round(4)

            st.markdown("**Top 10 most impactful tokens:**")
            st.dataframe(
                shap_table,
                use_container_width=True,
                hide_index=True,
            )

        elif not run_shap:
            st.info(
                "SHAP explanations disabled. "
                "Toggle on in the sidebar to see token-level explanations."
            )

    elif classify_btn:
        st.warning("Please enter a review before clicking Classify.")


# ─────────────────────────────────────────────────────────────────────────────
# TAB 2 — Model Performance
# ─────────────────────────────────────────────────────────────────────────────
with tab2:

    if not data_ready:
        st.warning(
            "Test predictions not found. "
            "Run `notebook_03_finetune.ipynb` Cell 13 first."
        )
    else:

        # ── Summary metrics ───────────────────────────────────────────────────
        st.markdown(
            '<div class="section-header">Test set results (25,000 reviews)</div>',
            unsafe_allow_html=True,
        )

        acc   = test_df["correct"].mean()
        f1    = 2 * acc / (1 + acc)   # approximation until loaded from CSV
        auc   = roc_auc_score(test_df["label"], test_df["prob_pos"])
        n_err = (~test_df["correct"]).sum()

        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Accuracy",  f"{acc*100:.2f}%")
        m2.metric("ROC-AUC",   f"{auc:.4f}")
        m3.metric("Errors",    f"{n_err:,}")
        m4.metric("Test size", f"{len(test_df):,}")
        m5.metric(
            "vs baseline",
            f"+{(acc - meta.get('baseline_acc', 0))*100:.2f}%"
            if meta.get("baseline_acc") else "—",
        )

        # ── Comparison table ──────────────────────────────────────────────────
        if comparison is not None:
            st.markdown(
                '<div class="section-header">Model comparison</div>',
                unsafe_allow_html=True,
            )
            st.dataframe(
                comparison.set_index("Model") if "Model" in comparison.columns
                else comparison,
                use_container_width=True,
            )

        # ── Charts ────────────────────────────────────────────────────────────
        col_cm, col_roc = st.columns(2)

        with col_cm:
            st.markdown("**Confusion matrix**")
            st.plotly_chart(
                confusion_matrix_chart(test_df),
                use_container_width=True,
            )

        with col_roc:
            st.markdown("**ROC curves**")
            st.plotly_chart(
                roc_chart(test_df, baseline),
                use_container_width=True,
            )

        # ── Confidence histogram ──────────────────────────────────────────────
        st.markdown(
            '<div class="section-header">Confidence distribution</div>',
            unsafe_allow_html=True,
        )
        st.plotly_chart(
            confidence_hist(test_df),
            use_container_width=True,
        )

        # ── Training curves ───────────────────────────────────────────────────
        if os.path.exists(TRAINING_IMG):
            st.markdown(
                '<div class="section-header">Training curves</div>',
                unsafe_allow_html=True,
            )
            st.image(
                TRAINING_IMG,
                caption="Loss, accuracy, and F1 per epoch — "
                        "DistilBERT fine-tuning",
                use_container_width=True,
            )


# ─────────────────────────────────────────────────────────────────────────────
# TAB 3 — SHAP Analysis
# ─────────────────────────────────────────────────────────────────────────────
with tab3:

    st.markdown(
        '<div class="section-header">SHAP explained</div>',
        unsafe_allow_html=True,
    )
    st.info(
        "**SHAP (SHapley Additive exPlanations)** assigns each token a value "
        "representing its contribution to the model's prediction. "
        "Green tokens push toward POSITIVE · Red tokens push toward NEGATIVE · "
        "Opacity encodes magnitude."
    )

    # ── Saved SHAP comparison from notebook 04 ────────────────────────────────
    if os.path.exists(SHAP_IMG):
        st.markdown(
            '<div class="section-header">'
            'Token importance — positive vs negative review'
            '</div>',
            unsafe_allow_html=True,
        )
        st.image(
            SHAP_IMG,
            caption="SHAP token explanations — which words drove each prediction",
            use_container_width=True,
        )

    # ── Confidence deep dive ──────────────────────────────────────────────────
    if os.path.exists(CONF_IMG):
        st.markdown(
            '<div class="section-header">'
            'Where does the model struggle?'
            '</div>',
            unsafe_allow_html=True,
        )
        st.image(
            CONF_IMG,
            caption="Confidence and error analysis from notebook 04",
            use_container_width=True,
        )

    # ── Model limitations ─────────────────────────────────────────────────────
    st.markdown(
        '<div class="section-header">Known model limitations</div>',
        unsafe_allow_html=True,
    )

    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("**What the model handles well:**")
        st.markdown("""
        - Clear positive vocabulary: *masterpiece, brilliant, stunning*
        - Clear negative vocabulary: *awful, waste, boring, terrible*
        - Simple negation: *not good, never boring*
        - Long reviews (uses first ~380 words effectively)
        - Contextual understanding of ambiguous words
        """)

    with col_b:
        st.markdown("**Known failure modes:**")
        st.markdown("""
        - **Sarcasm** — *"Oh brilliant, another terrible sequel"*
        - **Mixed sentiment** — *"Amazing visuals, terrible plot"*
        - **Complex negation** — *"Not without some merit, but largely fails"*
        - **Contrast structures** — review starts negative, ends positive
        - **Very short reviews** — too little context (<20 words)
        """)

    # ── Live SHAP on custom input ─────────────────────────────────────────────
    st.markdown(
        '<div class="section-header">'
        'Try SHAP on your own review'
        '</div>',
        unsafe_allow_html=True,
    )

    shap_input = st.text_area(
        "Enter a review for SHAP analysis:",
        height      = 100,
        placeholder = "Type a review to see token-level explanations...",
        key         = "shap_custom_input",
    )

    if st.button("Explain →", key="shap_btn"):
        if shap_input.strip() and explainer is not None:
            with st.spinner("Computing SHAP values..."):
                sv = compute_shap(explainer, shap_input)

            result = predict_sentiment(clf, shap_input)
            label  = result["label"]
            conf   = result["confidence"]
            colour = (
                COLOURS["positive"] if label == "POSITIVE"
                else COLOURS["negative"]
            )
            emoji  = "🟢" if label == "POSITIVE" else "🔴"

            st.markdown(
                f'{emoji} **{label}** — {conf*100:.1f}% confidence',
            )
            html = shap_to_html(sv, class_idx=1)
            st.markdown(
                f'<div class="shap-box">{html}</div>',
                unsafe_allow_html=True,
            )

        elif explainer is None:
            st.error("SHAP explainer not loaded. Check model path.")
        else:
            st.warning("Please enter a review.")


# ─────────────────────────────────────────────────────────────────────────────
# TAB 4 — Test Predictions
# ─────────────────────────────────────────────────────────────────────────────
with tab4:

    if not data_ready:
        st.warning(
            "No test predictions found. Run notebook_03 Cell 13 first."
        )
    else:

        st.markdown(
            '<div class="section-header">Browse test predictions</div>',
            unsafe_allow_html=True,
        )

        # ── Filters ───────────────────────────────────────────────────────────
        col_f1, col_f2, col_f3, col_f4 = st.columns(4)

        with col_f1:
            filter_sentiment = st.selectbox(
                "Actual sentiment",
                ["All", "Positive", "Negative"],
            )
        with col_f2:
            filter_correct = st.selectbox(
                "Prediction",
                ["All", "Correct only", "Incorrect only"],
            )
        with col_f3:
            filter_conf = st.slider(
                "Min confidence",
                min_value=0.50, max_value=1.00,
                value=0.50, step=0.05,
            )
        with col_f4:
            n_display = st.selectbox(
                "Rows to show",
                [25, 50, 100, 250],
                index=0,
            )

        # ── Apply filters ─────────────────────────────────────────────────────
        filtered = test_df.copy()

        if filter_sentiment != "All":
            filtered = filtered[
                filtered["sentiment"] == filter_sentiment.lower()
            ]
        if filter_correct == "Correct only":
            filtered = filtered[filtered["correct"] == True]
        elif filter_correct == "Incorrect only":
            filtered = filtered[filtered["correct"] == False]

        filtered = filtered[filtered["confidence"] >= filter_conf]

        st.caption(
            f"Showing {min(n_display, len(filtered)):,} of "
            f"{len(filtered):,} filtered reviews "
            f"(from {len(test_df):,} total)"
        )

        # ── Display table ─────────────────────────────────────────────────────
        display_cols = [
            "sentiment", "pred_sentiment",
            "confidence", "correct", "word_count",
        ]
        if "text" in filtered.columns:
            display_cols = ["text"] + display_cols

        available = [c for c in display_cols if c in filtered.columns]

        show_df = filtered[available].head(n_display).copy()
        if "text" in show_df.columns:
            show_df["text"] = show_df["text"].str[:120] + "..."

        show_df["confidence"] = show_df["confidence"].round(3)

        st.dataframe(show_df, use_container_width=True,
                     hide_index=True, height=420)

        # ── Download ──────────────────────────────────────────────────────────
        csv = filtered.to_csv(index=False).encode("utf-8")
        st.download_button(
            label     = "⬇ Download filtered predictions (CSV)",
            data      = csv,
            file_name = "filtered_predictions.csv",
            mime      = "text/csv",
        )

        # ── Error analysis summary ────────────────────────────────────────────
        st.markdown(
            '<div class="section-header">Error summary</div>',
            unsafe_allow_html=True,
        )

        col_e1, col_e2 = st.columns(2)

        with col_e1:
            fp = test_df[
                (test_df["label"] == 0) &
                (test_df["pred_label"] == 1)
            ]
            fn = test_df[
                (test_df["label"] == 1) &
                (test_df["pred_label"] == 0)
            ]
            st.metric("False positives", f"{len(fp):,}",
                      help="Predicted POSITIVE, actually NEGATIVE")
            st.metric("False negatives", f"{len(fn):,}",
                      help="Predicted NEGATIVE, actually POSITIVE")

        with col_e2:
            low_conf_err = test_df[
                (~test_df["correct"]) &
                (test_df["confidence"] < 0.70)
            ]
            high_conf_err = test_df[
                (~test_df["correct"]) &
                (test_df["confidence"] > 0.90)
            ]
            st.metric(
                "Low-conf errors (<70%)",
                f"{len(low_conf_err):,}",
                help="Model was uncertain AND wrong — genuinely hard reviews",
            )
            st.metric(
                "High-conf errors (>90%)",
                f"{len(high_conf_err):,}",
                help="Model was confident AND wrong — likely sarcasm or mixed sentiment",
            )


# ── Footer ─────────────────────────────────────────────────────────────────────
st.divider()
st.caption(
    "DistilBERT fine-tuned on IMDb 50k · "
    "SHAP token explanations · "
    "TF-IDF + LogReg baseline · "
    "Built with HuggingFace Transformers · Streamlit · Plotly"
)