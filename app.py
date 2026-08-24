"""
Emotion AI — Text Emotion Classifier
Premium light-mode deployment app for the Bidirectional GRU emotion
classification model trained on the dair-ai/emotion dataset.

Run with:
    streamlit run app.py
"""

import os
import pickle

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# --------------------------------------------------------------------------
# CONFIG
# --------------------------------------------------------------------------
# All files (app.py, model, tokenizer) live together in the same
# "emotion-classification" folder, so we resolve paths relative to this
# script's own location — this works no matter where streamlit is launched from.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "BiGRU_Model.keras")
TOKENIZER_PATH = os.path.join(BASE_DIR, "tokenizer.pkl")
MAX_LENGTH = 50

LABEL_NAMES = ["sadness", "joy", "love", "anger", "fear", "surprise"]

# Pastel-on-light palette: soft fill for backgrounds, richer tone for accents/text
EMOTION_META = {
    "sadness":  {"emoji": "😢", "color": "#5B8DEF", "soft": "#EAF0FF", "desc": "A feeling of sorrow, loss, or low mood."},
    "joy":      {"emoji": "😄", "color": "#F5A524", "soft": "#FFF4E0", "desc": "A feeling of happiness, delight, or pleasure."},
    "love":     {"emoji": "❤️", "color": "#EF5C8C", "soft": "#FFEAF1", "desc": "A feeling of deep affection or attachment."},
    "anger":    {"emoji": "😠", "color": "#EF4444", "soft": "#FEECEC", "desc": "A feeling of strong displeasure or hostility."},
    "fear":     {"emoji": "😨", "color": "#8B5CF6", "soft": "#F2ECFF", "desc": "A feeling of anxiety, worry, or being threatened."},
    "surprise": {"emoji": "😲", "color": "#10B981", "soft": "#E5F9F1", "desc": "A feeling of astonishment at something unexpected."},
}

SAMPLE_SENTENCES = [
    "I can't believe how happy I am right now, this is amazing!",
    "I feel so alone and hopeless today.",
    "I am furious that they cancelled the trip at the last minute.",
    "I feel terrified when walking down dark alleyways alone.",
    "I was shocked and completely surprised by the unexpected gift!",
    "I love spending quiet mornings with my family.",
]

# --------------------------------------------------------------------------
# PAGE CONFIG + PREMIUM LIGHT-MODE STYLING
# --------------------------------------------------------------------------
st.set_page_config(
    page_title="Emotion AI | Text Emotion Classifier",
    page_icon="🎭",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=Poppins:wght@600;700;800&display=swap');

        /* Force light theme via CSS variables — no .streamlit/config.toml needed */
        :root, .stApp {
            --background-color: #FBFAFF;
            --secondary-background-color: #FFFFFF;
            --text-color: #1F2033;
            --primary-color: #7C5CFC;
        }

        html, body, [class*="css"] {
            font-family: 'Plus Jakarta Sans', sans-serif;
            color: #1F2033;
        }

        .stMarkdown, .stMarkdown p, .stCaption, label, .stText,
        h1, h2, h3, h4, h5, h6 {
            color: #1F2033 !important;
        }

        /* Catch-all: force every generic text/label element to the dark ink color,
           since we removed .streamlit/config.toml and the app no longer inherits
           Streamlit's light theme text-color defaults. */
        p, span, div, li {
            color: inherit;
        }
        [data-testid="stMarkdownContainer"] * {
            color: #1F2033 !important;
        }
        [data-testid="stWidgetLabel"] p {
            color: #1F2033 !important;
        }

        /* ---------- Inline code chips (e.g. `dair-ai/emotion`, `sklearn`) ---------- */
        [data-testid="stMarkdownContainer"] code {
            background: #F2ECFF !important;
            color: #6D3FF0 !important;
            padding: 2px 7px !important;
            border-radius: 6px !important;
            font-size: 0.9em !important;
            font-weight: 600 !important;
        }
        [data-testid="stMarkdownContainer"] pre code {
            background: transparent !important;
            color: #1F2033 !important;
            padding: 0 !important;
        }

        /* ---------- Top spacing so the header doesn't sit under Streamlit's dev toolbar ---------- */
        .main .block-container {
            padding-top: 3.5rem !important;
        }

        /* ---------- Remove the stray colored focus ring left on buttons after a click ---------- */
        .stButton>button:focus,
        .stButton>button:focus:not(:active),
        .stButton>button:active {
            outline: none !important;
            box-shadow: 0 0 0 3px rgba(109, 63, 240, 0.18) !important;
            border-color: #6D3FF0 !important;
            color: #6D3FF0 !important;
        }
        .stButton>button[kind="primary"]:focus,
        .stButton>button[kind="primary"]:focus:not(:active),
        .stButton>button[kind="primary"]:active {
            box-shadow: 0 8px 20px rgba(109, 63, 240, 0.28) !important;
            color: #FFFFFF !important;
        }

        /* ---------- App background ---------- */
        .stApp {
            background: radial-gradient(circle at 10% 0%, #F4F1FF 0%, #FBFAFF 35%, #FFFFFF 100%);
        }

        section[data-testid="stSidebar"] {
            background: #FFFFFF;
            border-right: 1px solid #EDEAFB;
        }
        section[data-testid="stSidebar"] * {
            color: #1F2033;
        }

        /* ---------- Header ---------- */
        .hero-badge {
            display: inline-block;
            padding: 5px 14px;
            border-radius: 999px;
            background: linear-gradient(90deg, #F2ECFF, #FFEAF1);
            color: #6D3FF0 !important;
            font-size: 0.78rem;
            font-weight: 700;
            letter-spacing: 0.06em;
            text-transform: uppercase;
            margin-bottom: 0.9rem;
        }
        .main-header {
            font-family: 'Poppins', sans-serif;
            font-size: 2.7rem;
            font-weight: 800;
            background: linear-gradient(90deg, #6D3FF0, #EF5C8C);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 0.2rem;
            line-height: 1.15;
        }
        .sub-header {
            font-size: 1.06rem;
            color: #6B6B80 !important;
            margin-top: 0rem;
            margin-bottom: 1.6rem;
            max-width: 640px;
        }

        /* ---------- Generic premium card ---------- */
        .pcard {
            background: #FFFFFF;
            border: 1px solid #EEECF7;
            border-radius: 20px;
            padding: 1.6rem 1.7rem;
            box-shadow: 0 4px 24px rgba(109, 63, 240, 0.06);
        }

        /* ---------- Result card ---------- */
        .result-card {
            padding: 2.2rem 1.8rem;
            border-radius: 22px;
            text-align: center;
            color: white;
            margin-bottom: 1rem;
            box-shadow: 0 14px 36px -12px rgba(0,0,0,0.28);
        }
        /* These sit inside [data-testid="stMarkdownContainer"], so they need an
           explicit !important white to survive the global text-color catch-all above. */
        .result-card, .result-card * {
            color: #FFFFFF !important;
        }
        .result-emoji { font-size: 3.6rem; line-height: 1; }
        .result-label {
            font-family: 'Poppins', sans-serif;
            font-size: 1.9rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 1px;
            margin-top: 0.3rem;
        }
        .result-conf { font-size: 0.98rem; opacity: 0.92; margin-top: 0.2rem; }

        /* ---------- Metric / stat boxes ---------- */
        /* Make the row of st.columns stretch so every card in the row shares
           the height of the tallest one (Streamlit's flex row does not do
           this by default, which is why a plain min-height wasn't enough). */
        [data-testid="stHorizontalBlock"] {
            align-items: stretch !important;
        }
        [data-testid="stHorizontalBlock"] > [data-testid="stColumn"] {
            display: flex !important;
        }
        [data-testid="stHorizontalBlock"] > [data-testid="stColumn"] > div {
            width: 100%;
            display: flex;
            flex-direction: column;
        }
        [data-testid="stHorizontalBlock"] [data-testid="stMarkdownContainer"] {
            width: 100%;
            display: flex;
            flex-direction: column;
            flex: 1;
        }
        .metric-box {
            padding: 1.3rem 1rem;
            border-radius: 18px;
            background: #FFFFFF;
            border: 1px solid #EEECF7;
            box-shadow: 0 4px 20px rgba(109, 63, 240, 0.05);
            text-align: center;
            transition: transform 0.15s ease, box-shadow 0.15s ease;
            height: 128px;               /* FIX: fixed size so box never grows with its text content */
            width: 100%;
            box-sizing: border-box;
            overflow: hidden;
            display: flex;
            flex-direction: column;
            justify-content: center;
            align-items: center;
        }
        .metric-box:hover {
            transform: translateY(-2px);
            box-shadow: 0 10px 26px rgba(109, 63, 240, 0.12);
        }
        .metric-box h2, .metric-box h3 {
            font-family: 'Poppins', sans-serif;
            background: linear-gradient(90deg, #6D3FF0, #EF5C8C);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin: 0 0 0.15rem 0;
        }
        .metric-box b { color: #1F2033; }

        /* ---------- Sidebar emotion chips ---------- */
        .emo-chip {
            display: flex;
            align-items: center;
            gap: 0.5rem;
            padding: 0.4rem 0.65rem;
            border-radius: 12px;
            margin-bottom: 0.35rem;
            font-weight: 600;
            font-size: 0.92rem;
        }

        /* ---------- Section headers ---------- */
        .section-title {
            font-family: 'Poppins', sans-serif;
            font-weight: 700;
            color: #1F2033;
            margin-bottom: 0.4rem;
        }

        /* ---------- Tabs ---------- */
        .stTabs [data-baseweb="tab-list"] {
            gap: 8px !important;
            background: #F6F4FE !important;
            padding: 8px !important;
            border-radius: 16px !important;
            border: none !important;
            width: fit-content !important;
        }
        .stTabs [data-baseweb="tab"] {
            height: auto !important;
            border-radius: 11px !important;
            padding: 10px 22px !important;
            margin: 0 !important;
            font-weight: 600 !important;
            font-size: 0.96rem !important;
            color: #6B6B80 !important;
            background: transparent !important;
            transition: all 0.15s ease;
        }
        .stTabs [data-baseweb="tab"]:hover {
            color: #6D3FF0 !important;
            background: rgba(109, 63, 240, 0.06) !important;
        }
        .stTabs [data-baseweb="tab"] p {
            font-size: 0.96rem !important;
            font-weight: 600 !important;
            color: inherit !important;
        }
        .stTabs [aria-selected="true"] {
            background: #FFFFFF !important;
            color: #6D3FF0 !important;
            box-shadow: 0 3px 12px rgba(109, 63, 240, 0.16);
        }
        .stTabs [aria-selected="true"] p {
            color: #6D3FF0 !important;
        }
        /* Remove Streamlit's default underline indicator — we use pill fill instead */
        .stTabs [data-baseweb="tab-highlight"] {
            display: none !important;
            background: transparent !important;
        }
        .stTabs [data-baseweb="tab-border"] {
            display: none !important;
        }
        .stTabs [data-testid="stTabsContentArea"],
        .stTabs > div > div:last-child {
            border: none !important;
        }

        /* ---------- Buttons ---------- */
        .stButton>button {
            background: #FFFFFF !important;
            color: #1F2033 !important;
            border-radius: 12px !important;
            font-weight: 600 !important;
            border: 1.5px solid #EEECF7 !important;
            transition: all 0.15s ease;
        }
        .stButton>button p {
            color: inherit !important;
        }
        .stButton>button:hover {
            border-color: #6D3FF0 !important;
            color: #6D3FF0 !important;
        }
        .stButton>button[kind="primary"] {
            background: linear-gradient(90deg, #6D3FF0, #EF5C8C) !important;
            color: #FFFFFF !important;
            border: none !important;
            box-shadow: 0 8px 20px rgba(109, 63, 240, 0.28);
        }
        .stButton>button[kind="primary"] p {
            color: #FFFFFF !important;
        }
        .stButton>button[kind="primary"]:hover {
            filter: brightness(1.06);
            color: white !important;
        }

        /* ---------- Text area ---------- */
        .stTextArea textarea {
            border-radius: 14px !important;
            border: 1.5px solid #EEECF7 !important;
            background: #FFFFFF !important;
            color: #1F2033 !important;
            caret-color: #6D3FF0;
        }
        .stTextArea textarea::placeholder {
            color: #A9A7BE !important;
            opacity: 1 !important;
        }
        .stTextArea textarea:focus {
            border-color: #6D3FF0 !important;
            box-shadow: 0 0 0 3px rgba(109, 63, 240, 0.12) !important;
        }

        /* ---------- Tables (custom HTML — see render_table()) ---------- */
        .table-wrap {
            border-radius: 14px;
            overflow: hidden;
            border: 1px solid #EEECF7;
            box-shadow: 0 4px 20px rgba(109, 63, 240, 0.05);
            overflow-x: auto;
        }
        .pretty-table {
            width: 100%;
            border-collapse: collapse;
            background: #FFFFFF;
            table-layout: auto;
        }
        .pretty-table thead th {
            background: #F6F4FE !important;
            color: #1F2033 !important;
            font-weight: 700 !important;
            text-align: center !important;
            padding: 12px 18px !important;
            border-bottom: 1px solid #EEECF7 !important;
            white-space: nowrap;
        }
        .pretty-table tbody td {
            background: #FFFFFF !important;
            color: #1F2033 !important;
            padding: 12px 18px !important;
            border-bottom: 1px solid #F3F1FA !important;
            white-space: normal !important;   /* FIX: let long cell text wrap instead of being clipped */
            word-break: break-word;
            line-height: 1.5;
        }
        .pretty-table tbody tr:last-child td {
            border-bottom: none !important;
        }
        .pretty-table tbody tr:hover td {
            background: #FAF9FF !important;
        }
        .pretty-table td.cell-center {
            text-align: center !important;
        }
        .pretty-table td.cell-left {
            text-align: left !important;
        }

        footer {visibility: hidden;}
        #MainMenu {visibility: hidden;}
    </style>
    """,
    unsafe_allow_html=True,
)

# --------------------------------------------------------------------------
# MODEL LOADING (cached)
# --------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading model & tokenizer...")
def load_artifacts():
    if not os.path.exists(MODEL_PATH) or not os.path.exists(TOKENIZER_PATH):
        return None, None
    model = load_model(MODEL_PATH)
    with open(TOKENIZER_PATH, "rb") as f:
        tokenizer = pickle.load(f)
    return model, tokenizer


def predict_emotion(text, model, tokenizer):
    seq = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(seq, maxlen=MAX_LENGTH, padding="post", truncating="post")
    probs = model.predict(padded, verbose=0)[0]
    pred_idx = int(np.argmax(probs))
    return LABEL_NAMES[pred_idx], probs


def render_table(df, center_cols=None):
    """Render a DataFrame as a hand-built HTML table styled to match the
    app's light theme. Used instead of st.table (had a hidden-index column
    that misaligned headers/data) and st.dataframe (its canvas-based grid
    ignores our CSS theme entirely and clips long cell text)."""
    center_cols = set(center_cols or [])

    header_html = "".join(f"<th>{col}</th>" for col in df.columns)

    rows_html = ""
    for _, row in df.iterrows():
        cells = ""
        for col in df.columns:
            css_class = "cell-center" if col in center_cols else "cell-left"
            cells += f'<td class="{css_class}">{row[col]}</td>'
        rows_html += f"<tr>{cells}</tr>"

    st.markdown(
        f"""
        <div class="table-wrap">
            <table class="pretty-table">
                <thead><tr>{header_html}</tr></thead>
                <tbody>{rows_html}</tbody>
            </table>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_probability_chart(probs):
    df = pd.DataFrame({
        "Emotion": [e.capitalize() for e in LABEL_NAMES],
        "Confidence": probs,
        "Color": [EMOTION_META[e]["color"] for e in LABEL_NAMES],
    }).sort_values("Confidence", ascending=True)

    fig = go.Figure(go.Bar(
        x=df["Confidence"],
        y=df["Emotion"],
        orientation="h",
        marker=dict(color=df["Color"], line=dict(width=0)),
        text=[f"{v*100:.1f}%" for v in df["Confidence"]],
        textposition="outside",
        cliponaxis=False,  # FIX: don't clip "outside" labels at the plot boundary
        textfont=dict(color="#1F2033", size=13),
    ))
    fig.update_layout(
        xaxis=dict(
            title="",
            range=[0, 1.12],  # FIX: headroom past 100% so near-max labels have room
            tickformat=".0%",
            tick0=0,
            dtick=0.2,
            gridcolor="#F0EEFA",
            zeroline=False,
            tickfont=dict(color="#6B6B80", size=12),  # FIX: was unset -> defaulted to near-invisible gray
        ),
        yaxis=dict(title="", tickfont=dict(color="#1F2033", size=13)),
        height=320,
        margin=dict(l=10, r=50, t=10, b=10),  # FIX: extra right margin for labels
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Plus Jakarta Sans, sans-serif"),
    )
    st.plotly_chart(fig, use_container_width=True)


# --------------------------------------------------------------------------
# SIDEBAR
# --------------------------------------------------------------------------
with st.sidebar:
    st.markdown(
        """
        <div style="text-align:center; padding: 0.6rem 0 1rem 0;">
            <div style="font-size:2.6rem;">🎭</div>
            <div style="font-family:'Poppins',sans-serif; font-weight:800; font-size:1.3rem;
                        background: linear-gradient(90deg, #6D3FF0, #EF5C8C);
                        -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
                Emotion AI
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        "A deep learning NLP app that classifies text into one of "
        "**six emotions** using a Bidirectional GRU network."
    )
    st.divider()
    st.markdown("**Emotions detected**")
    for label, meta in EMOTION_META.items():
        st.markdown(
            f"""<div class="emo-chip" style="background:{meta['soft']}; color:{meta['color']} !important;">
                    <span>{meta['emoji']}</span><span>{label.capitalize()}</span>
                </div>""",
            unsafe_allow_html=True,
        )
    st.divider()
    st.markdown("**Model** &nbsp;·&nbsp; Bidirectional GRU (Keras)")
    st.markdown("**Dataset** &nbsp;·&nbsp; dair-ai/emotion")
    st.markdown("**Test Accuracy** &nbsp;·&nbsp; 91.55%")
    st.caption("Built with TensorFlow/Keras + Streamlit")

model, tokenizer = load_artifacts()

# --------------------------------------------------------------------------
# HEADER
# --------------------------------------------------------------------------
st.markdown('<div class="hero-badge">Deep Learning · NLP</div>', unsafe_allow_html=True)
st.markdown('<div class="main-header">Understand the emotion behind any text</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">Powered by a Bidirectional GRU network trained on 20,000 '
    'labeled sentences — classifying text into sadness, joy, love, anger, fear, or surprise.</div>',
    unsafe_allow_html=True,
)

if model is None:
    st.error(
        f"⚠️ Model artifacts not found. Expected `{MODEL_PATH}` and `{TOKENIZER_PATH}`. "
        "Make sure `BiGRU_Model.keras` and `tokenizer.pkl` are placed directly inside "
        "the `emotion-classification` folder, next to `app.py`."
    )

# --------------------------------------------------------------------------
# TABS
# --------------------------------------------------------------------------
st.write("")
tab_predict, tab_about, tab_arch, tab_perf = st.tabs(
    ["🔮  Predict", "📖  About the Model", "🏗️  Architecture", "📊  Performance"]
)

# ============================== PREDICT TAB ===============================
with tab_predict:
    st.write("")
    col_input, col_result = st.columns([1.15, 1])

    with col_input:
        st.markdown('<div class="section-title" style="text-align:center;">Enter text to analyze</div>', unsafe_allow_html=True)
        user_text = st.text_area(
            label="Text input",
            placeholder="e.g. I can't believe how happy I am right now, this is amazing!",
            height=150,
            label_visibility="collapsed",
            key="user_text",
        )

        st.caption("Or try a sample sentence")
        sample_cols = st.columns(3)

        def _set_sample_text(sample_sentence):
            # Runs BEFORE the script reruns and BEFORE the text_area below is
            # re-instantiated, so updating session_state here is safe.
            st.session_state["user_text"] = sample_sentence

        for i, sample in enumerate(SAMPLE_SENTENCES):
            sample_cols[i % 3].button(
                f"Sample {i + 1}",
                use_container_width=True,
                key=f"sample_{i}",
                on_click=_set_sample_text,
                args=(sample,),
            )

        st.write("")
        predict_clicked = st.button("✨  Predict Emotion", type="primary", use_container_width=True)

    with col_result:
        st.markdown('<div class="section-title" style="text-align:center;">Result</div>', unsafe_allow_html=True)
        if predict_clicked:
            if model is None:
                st.warning("Model not loaded — cannot run prediction.")
            elif not user_text or not user_text.strip():
                st.warning("Please enter some text first.")
            else:
                emotion, probs = predict_emotion(user_text, model, tokenizer)
                meta = EMOTION_META[emotion]
                confidence = float(np.max(probs))

                st.markdown(
                    f"""
                    <div class="result-card" style="background: linear-gradient(135deg, {meta['color']}, {meta['color']}CC);">
                        <div class="result-emoji">{meta['emoji']}</div>
                        <div class="result-label">{emotion}</div>
                        <div class="result-conf">Confidence: {confidence*100:.1f}%</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                st.caption(meta["desc"])
                st.markdown("**Confidence across all emotions**")
                render_probability_chart(probs)
        else:
            st.markdown(
                """
                <div class="pcard" style="text-align:center; color:#6B6B80 !important; padding: 3rem 1.5rem;">
                    <div style="font-size:2.4rem; margin-bottom:0.5rem;">🎭</div>
                    Enter text on the left and click <b>Predict Emotion</b><br/>to see results here.
                </div>
                """,
                unsafe_allow_html=True,
            )

# ============================== ABOUT TAB ==================================
with tab_about:
    st.write("")
    st.markdown('<div class="section-title" style="font-size:1.3rem;">About This Project</div>', unsafe_allow_html=True)
    st.markdown(
        """
        This app classifies free-form text into one of **six basic emotions** —
        sadness, joy, love, anger, fear, and surprise — using a deep learning
        NLP model trained end-to-end in TensorFlow/Keras.
        """
    )

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown('<div class="metric-box"><h3>20,000</h3>Labeled sentences</div>', unsafe_allow_html=True)
    with c2:
        st.markdown('<div class="metric-box"><h3>6</h3>Emotion classes</div>', unsafe_allow_html=True)
    with c3:
        st.markdown('<div class="metric-box"><h3>91.55%</h3>Test accuracy</div>', unsafe_allow_html=True)

    st.write("")
    st.markdown('<div class="section-title">Dataset</div>', unsafe_allow_html=True)
    st.markdown(
        """
        - **Source:** [`dair-ai/emotion`](https://huggingface.co/datasets/dair-ai/emotion) on Hugging Face
        - **Splits:** official train / validation / test splits (test set held out until final evaluation)
        - **Classes:** sadness, joy, love, anger, fear, surprise
        - **Class imbalance:** handled using `sklearn`'s balanced class weights during training
        """
    )

    st.markdown('<div class="section-title">Preprocessing Pipeline</div>', unsafe_allow_html=True)
    st.markdown(
        """
        1. **Tokenization** — Keras `Tokenizer`, vocabulary capped at **10,000** words, fit only on training text
        2. **Out-of-vocabulary handling** — unseen words mapped to an `<OOV>` token
        3. **Sequencing** — text converted to integer sequences
        4. **Padding/Truncation** — every sequence normalized to a fixed length of **50** tokens (`post` padding & truncation)
        """
    )

    st.markdown('<div class="section-title">Methodology</div>', unsafe_allow_html=True)
    st.markdown(
        """
        The project was built in two phases:

        **Phase 1 — Foundational models.** Plain (non-bidirectional) SimpleRNN, LSTM,
        and GRU networks were trained and compared on validation accuracy to establish
        a baseline and identify the strongest recurrent unit type.

        **Phase 2 — Advanced bidirectional models.** The two strongest architectures
        (LSTM and GRU) were rebuilt as stacked **Bidirectional** networks with larger
        (300-dim) embeddings. Their validation performance was compared, and the
        **Bidirectional GRU** was selected as the final model — evaluated **once** on
        the untouched test set to avoid data leakage.
        """
    )

    st.markdown('<div class="section-title">Training Setup</div>', unsafe_allow_html=True)
    st.markdown(
        """
        - **Loss:** sparse categorical cross-entropy
        - **Optimizer:** Adam
        - **Class weighting:** balanced (to counter class imbalance, e.g. rare `surprise` samples)
        - **Early stopping:** monitored `val_loss`, patience of 3 epochs, best weights restored
        - **Epochs:** up to 20 (with early stopping)
        - **Batch size:** 32
        """
    )

# ============================== ARCHITECTURE TAB ============================
with tab_arch:
    st.write("")
    st.markdown('<div class="section-title" style="font-size:1.3rem;">Model Architecture — Bidirectional GRU</div>', unsafe_allow_html=True)
    st.markdown(
        "The final deployed model is a **stacked Bidirectional GRU** network that "
        "reads each sentence in both directions to capture context from the "
        "words before *and* after each token."
    )

    st.markdown('<div class="section-title">Layer-by-Layer Breakdown</div>', unsafe_allow_html=True)
    arch_df = pd.DataFrame(
        [
            ["1", "Embedding", "input_dim=10,000, output_dim=300, input_length=50", "Maps each token to a 300-dim dense vector"],
            ["2", "Bidirectional GRU", "128 units, return_sequences=True", "Forward + backward pass, outputs full sequence (256-dim per step)"],
            ["3", "Dropout", "rate=0.5", "Regularization to reduce overfitting"],
            ["4", "Bidirectional GRU", "64 units", "Forward + backward pass, outputs final context vector (128-dim)"],
            ["5", "Dropout", "rate=0.5", "Regularization to reduce overfitting"],
            ["6", "Dense (Output)", "6 units, softmax activation", "Produces probability distribution over 6 emotion classes"],
        ],
        columns=["Step", "Layer", "Configuration", "Purpose"],
    )
    render_table(arch_df, center_cols=["Step"])

    st.write("")
    st.markdown('<div class="section-title">Compilation Settings</div>', unsafe_allow_html=True)
    cc1, cc2, cc3 = st.columns(3)
    cc1.markdown('<div class="metric-box"><b>Loss</b><br>Sparse Categorical<br>Cross-Entropy</div>', unsafe_allow_html=True)
    cc2.markdown('<div class="metric-box"><b>Optimizer</b><br>Adam</div>', unsafe_allow_html=True)
    cc3.markdown('<div class="metric-box"><b>Metric</b><br>Accuracy</div>', unsafe_allow_html=True)

    st.write("")
    st.markdown('<div class="section-title">Why Bidirectional GRU?</div>', unsafe_allow_html=True)
    st.markdown(
        """
        - **GRU vs LSTM:** GRUs use a simpler gating mechanism (2 gates vs LSTM's 3),
          training faster with fewer parameters while still capturing long-range dependencies well.
        - **Bidirectional processing:** reading a sentence both left-to-right and
          right-to-left lets the model use context from the *whole* sentence when
          interpreting each word — important for emotion words whose meaning depends
          on what surrounds them.
        - **Stacked layers:** the first BiGRU layer returns full sequences so the
          second layer can learn higher-level temporal patterns before compressing
          to a single context vector for classification.
        """
    )

    st.markdown('<div class="section-title">Architecture Comparison (Model Selection Journey)</div>', unsafe_allow_html=True)
    st.caption("All models compared on the same validation split before final selection.")

    phase1_df = pd.DataFrame({
        "Model": ["LSTM", "GRU", "SimpleRNN"],
        "Validation Accuracy": ["89.85%", "35.10%", "19.95%"],
        "Validation Loss": [0.3291, 1.7695, 1.7841],
    })
    st.markdown("**Phase 1 — Plain recurrent models**")
    render_table(phase1_df, center_cols=["Validation Accuracy", "Validation Loss"])

    phase2_df = pd.DataFrame({
        "Model": ["Bidirectional GRU ✅ (selected)", "Bidirectional LSTM"],
        "Validation Accuracy": ["93.10%", "91.90%"],
        "Validation Loss": [0.1960, 0.2540],
    })
    st.markdown("**Phase 2 — Bidirectional models**")
    render_table(phase2_df, center_cols=["Validation Accuracy", "Validation Loss"])

# ============================== PERFORMANCE TAB =============================
with tab_perf:
    st.write("")
    st.markdown('<div class="section-title" style="font-size:1.3rem;">Final Test Set Performance</div>', unsafe_allow_html=True)
    st.caption("Evaluated once, on the held-out test set, after model selection was finalized.")

    p1, p2, p3 = st.columns(3)
    p1.markdown('<div class="metric-box"><h2>91.55%</h2>Test Accuracy</div>', unsafe_allow_html=True)
    p2.markdown('<div class="metric-box"><h2>0.2253</h2>Test Loss</div>', unsafe_allow_html=True)
    p3.markdown('<div class="metric-box"><h2>2,000</h2>Test Samples</div>', unsafe_allow_html=True)

    st.write("")
    st.markdown('<div class="section-title">Per-Class Classification Report</div>', unsafe_allow_html=True)
    report_df = pd.DataFrame(
        [
            ["Sadness", 0.9818, 0.9294, 0.9549, 581],
            ["Joy", 0.9722, 0.9050, 0.9374, 695],
            ["Love", 0.7451, 0.9560, 0.8375, 159],
            ["Anger", 0.9094, 0.9491, 0.9288, 275],
            ["Fear", 0.8767, 0.8571, 0.8668, 224],
            ["Surprise", 0.6129, 0.8636, 0.7170, 66],
        ],
        columns=["Emotion", "Precision", "Recall", "F1-Score", "Support"],
    )
    render_table(report_df, center_cols=["Precision", "Recall", "F1-Score", "Support"])

    st.markdown(
        """
        - **Macro avg F1:** 0.8737 &nbsp;|&nbsp; **Weighted avg F1:** 0.9182
        - Strongest classes: **sadness** and **joy** (largest support, highest precision)
        - Weakest class: **surprise** — smallest class (only 66 test samples), so lower
          precision is expected; recall is still strong at 86.4%
        - **Love** shows high recall but lower precision — the model sometimes
          confuses affectionate language with joy
        """
    )

    st.write("")
    st.markdown('<div class="section-title">Training Curves</div>', unsafe_allow_html=True)
    st.caption(
        "During training, the Bi-GRU's training and validation accuracy tracked closely "
        "with early stopping (patience=3 on val_loss), indicating limited overfitting "
        "before convergence."
    )

    st.markdown('<div class="section-title">Confusion Matrix</div>', unsafe_allow_html=True)
    st.markdown(
        "The confusion matrix (see training notebook) shows the majority of "
        "predictions fall on the diagonal, with the main confusion occurring "
        "between semantically related emotion pairs — e.g. **love ↔ joy** and "
        "**fear ↔ sadness**."
    )