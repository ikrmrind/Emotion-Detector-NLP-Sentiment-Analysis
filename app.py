import streamlit as st
import joblib
import re
import string
import numpy as np
from pathlib import Path

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="Emotion Detector",
    page_icon="🎭",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ============================================================
# CUSTOM CSS
# ============================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700;800&display=swap');
html, body, [class*="css"] { font-family: 'Poppins', sans-serif; }

.stApp {
    background:
        radial-gradient(circle at 15% 20%, rgba(139,92,246,0.25), transparent 45%),
        radial-gradient(circle at 85% 80%, rgba(236,72,153,0.22), transparent 45%),
        radial-gradient(circle at 50% 50%, rgba(59,130,246,0.15), transparent 55%),
        linear-gradient(135deg, #0f0c29 0%, #1a1a3e 50%, #24243e 100%);
    background-attachment: fixed;
    min-height: 100vh;
}
.block-container { padding-top: 2rem; padding-bottom: 3rem; max-width: 1100px; }

.main-header {
    text-align: center; padding: 2.6rem 1.5rem;
    background: rgba(255,255,255,0.05);
    border-radius: 30px;
    backdrop-filter: blur(24px);
    border: 1px solid rgba(255,255,255,0.12);
    margin-bottom: 2rem;
    box-shadow: 0 10px 50px rgba(0,0,0,0.45), inset 0 1px 0 rgba(255,255,255,0.08);
    position: relative; overflow: hidden;
}
.main-header::before {
    content: ''; position: absolute; top: -50%; left: -50%;
    width: 200%; height: 200%;
    background: conic-gradient(from 0deg, transparent, rgba(167,139,250,0.08), transparent, rgba(236,72,153,0.08), transparent);
    animation: spin 18s linear infinite; z-index: 0;
}
@keyframes spin { to { transform: rotate(360deg); } }
.main-header > * { position: relative; z-index: 1; }

.main-header h1 {
    background: linear-gradient(135deg, #f472b6 0%, #a78bfa 45%, #60a5fa 100%);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    background-clip: text; font-size: 3.2rem; font-weight: 800;
    margin: 0; letter-spacing: -1.5px;
}
.main-header p { color:#cbd5e1; font-size:1.08rem; margin-top:0.7rem; font-weight:300; letter-spacing:0.3px; }

.stTextArea textarea {
    background: rgba(255,255,255,0.05) !important;
    color: #fff !important;
    border: 2px solid rgba(167,139,250,0.35) !important;
    border-radius: 20px !important;
    font-size: 1.05rem !important; padding: 1.3rem !important;
    font-family: 'Poppins', sans-serif !important;
    transition: all 0.35s cubic-bezier(0.4,0,0.2,1) !important;
    resize: none !important; line-height: 1.6 !important;
}
.stTextArea textarea::placeholder { color:#94a3b8 !important; font-weight:300; }
.stTextArea textarea:focus {
    border-color: #a78bfa !important;
    box-shadow: 0 0 30px rgba(167,139,250,0.45) !important;
    outline: none !important;
    background: rgba(255,255,255,0.07) !important;
}
.stTextArea label { color:#cbd5e1 !important; font-weight:500 !important; }

.stButton > button {
    background: linear-gradient(135deg, #8b5cf6 0%, #ec4899 100%) !important;
    color:#fff !important; border:none !important;
    border-radius: 18px !important;
    padding: 0.95rem 2rem !important;
    font-size: 1.08rem !important; font-weight:600 !important;
    width: 100% !important; letter-spacing: 0.6px !important;
    transition: all 0.35s cubic-bezier(0.4,0,0.2,1) !important;
    box-shadow: 0 8px 28px rgba(139,92,246,0.5) !important;
}
.stButton > button:hover {
    transform: translateY(-3px) scale(1.01) !important;
    box-shadow: 0 14px 40px rgba(236,72,153,0.65) !important;
    filter: brightness(1.12) !important;
}

.result-card {
    padding: 2.4rem 2rem; border-radius: 28px; text-align: center;
    border: 1px solid rgba(255,255,255,0.22);
    box-shadow: 0 16px 50px rgba(0,0,0,0.4), inset 0 1px 0 rgba(255,255,255,0.2);
    animation: fadeIn 0.7s cubic-bezier(0.4,0,0.2,1);
    margin-top: 1.8rem; position: relative; overflow: hidden;
}
@keyframes fadeIn {
    from { opacity:0; transform: translateY(24px) scale(0.97); }
    to   { opacity:1; transform: translateY(0) scale(1); }
}

.section-title {
    color:#e0e7ff; font-size:1.18rem; font-weight:600;
    margin: 1.8rem 0 1rem; padding: 0.45rem 0.9rem;
    border-left: 4px solid #a78bfa;
    background: linear-gradient(90deg, rgba(167,139,250,0.15), transparent);
    border-radius: 0 8px 8px 0; letter-spacing: 0.3px;
}

.bar-container {
    background: rgba(255,255,255,0.045);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 16px; padding: 1rem 1.2rem; margin-bottom: 0.75rem;
    backdrop-filter: blur(10px); transition: all 0.3s ease;
}
.bar-container:hover {
    background: rgba(255,255,255,0.08);
    transform: translateX(6px);
    border-color: rgba(167,139,250,0.3);
}
.bar-header {
    display:flex; justify-content: space-between;
    color:#e2e8f0; font-size:0.99rem; margin-bottom:0.55rem; align-items:center;
}
.bar-track {
    height: 11px; background: rgba(255,255,255,0.08);
    border-radius: 6px; overflow: hidden;
    box-shadow: inset 0 1px 3px rgba(0,0,0,0.3);
}
.bar-fill {
    height: 100%; border-radius: 6px;
    animation: growBar 1.1s cubic-bezier(0.4,0,0.2,1);
}
@keyframes growBar { from { width: 0 !important; } }

#MainMenu, footer, header { visibility: hidden; }
.stDeployButton { display: none; }
.footer-note {
    text-align:center; color:#64748b; font-size:0.85rem; margin-top:2.5rem;
    padding-top:1.5rem; border-top:1px solid rgba(255,255,255,0.06);
}
</style>
""", unsafe_allow_html=True)

# ============================================================
# EMOTION STYLES
# ============================================================
EMO = {
    'joy':      ('😄', '#FFD93D', 'linear-gradient(135deg,#FFD93D 0%,#F59E0B 100%)'),
    'sadness':  ('😢', '#60A5FA', 'linear-gradient(135deg,#60A5FA 0%,#3B82F6 100%)'),
    'anger':    ('😠', '#F87171', 'linear-gradient(135deg,#F87171 0%,#DC2626 100%)'),
    'fear':     ('😨', '#A78BFA', 'linear-gradient(135deg,#A78BFA 0%,#7C3AED 100%)'),
    'love':     ('❤️', '#F472B6', 'linear-gradient(135deg,#F472B6 0%,#EC4899 100%)'),
    'surprise': ('😲', '#FB923C', 'linear-gradient(135deg,#FB923C 0%,#EA580C 100%)'),
}

# ============================================================
# LOAD ARTIFACTS (joblib)
# ============================================================
@st.cache_resource(show_spinner=False)
def load_artifacts():
    base = Path(__file__).parent
    model      = joblib.load(base / "emotion_model.pkl")
    vectorizer = joblib.load(base / "tfidf_vectorizer.pkl")
    label_map  = joblib.load(base / "label_map.pkl")        # {int: 'emotion'}
    columns    = joblib.load(base / "feature_columns.pkl")  # ['Text','Emotion','Numeric_Emotions']
    try:
        stop_words = set(joblib.load(base / "stopwords.pkl"))
    except FileNotFoundError:
        stop_words = set()
    return model, vectorizer, label_map, columns, stop_words

# ============================================================
# PREPROCESS (must match training)
# ============================================================
def preprocess(txt, stop_words):
    txt = str(txt).lower()
    txt = txt.translate(str.maketrans('', '', string.punctuation))
    txt = re.sub(r'\d+', '', txt)
    txt = re.sub(r'https?\S*', '', txt)
    txt = ''.join(c for c in txt if c.isascii())
    txt = re.sub(r'\s+', ' ', txt).strip()
    words = [w for w in txt.split() if w not in stop_words and len(w) > 1]
    return ' '.join(words)

# ============================================================
# LOAD
# ============================================================
model_loaded = False
try:
    with st.spinner("🧠  Loading AI model..."):
        model, vectorizer, label_map, feature_columns, stop_words = load_artifacts()
        model_loaded = True
except FileNotFoundError as e:
    st.error(f"❌  Missing artifact file: `{e.filename}`\n\n"
             "Make sure these files exist next to `app.py`:\n"
             "- `emotion_model.pkl`\n- `tfidf_vectorizer.pkl`\n"
             "- `label_map.pkl`\n- `feature_columns.pkl`\n- `stopwords.pkl` (optional)")
except Exception as e:
    st.error(f"❌  Error loading model: `{e}`")

# ============================================================
# HEADER
# ============================================================
st.markdown("""
<div class="main-header">
    <h1>🎭 Emotion Detector</h1>
    <p>Discover the emotion hidden behind your words with AI</p>
</div>
""", unsafe_allow_html=True)

# ============================================================
# UI
# ============================================================
if model_loaded:
    c1, c2, c3 = st.columns([1, 4, 1])
    with c2:
        user_text = st.text_area(
            "✍️  Enter your text",
            placeholder="Type or paste your text here…  e.g.  \"I feel so happy and grateful today\"",
            height=160,
            key="input_text"
        )
        detect = st.button("🔍   Detect Emotion")

        if detect:
            if not user_text.strip():
                st.warning("⚠️  Please enter some text first.")
            else:
                with st.spinner("Analyzing emotion…"):
                    clean = preprocess(user_text, stop_words)
                    if not clean.strip():
                        st.warning("⚠️  After cleaning, the text is empty. Try a different sentence.")
                    else:
                        vec   = vectorizer.transform([clean])
                        proba = model.predict_proba(vec)[0]
                        top_idx   = int(np.argmax(proba))
                        top_label = int(model.classes_[top_idx])
                        top_emotion = label_map[top_label]

                        emoji, color, grad = EMO.get(top_emotion, ('🙂','#94a3b8','linear-gradient(135deg,#94a3b8,#64748b)'))
                        conf = proba[top_idx] * 100

                        st.markdown(f"""
                        <div class="result-card" style="background:{grad};">
                            <div style="font-size:5.5rem; line-height:1;
                                        filter: drop-shadow(0 6px 24px rgba(0,0,0,0.35));">
                                {emoji}
                            </div>
                            <h2 style="color:#fff; margin:0.7rem 0 0.35rem; font-size:2.8rem;
                                       text-transform:capitalize; letter-spacing:-0.8px;
                                       text-shadow:0 3px 16px rgba(0,0,0,0.35);">
                                {top_emotion}
                            </h2>
                            <p style="color:rgba(255,255,255,0.95); font-size:1.15rem;
                                      margin:0; font-weight:500; letter-spacing:0.5px;">
                                Confidence &nbsp;·&nbsp; {conf:.1f}%
                            </p>
                        </div>
                        """, unsafe_allow_html=True)

                        st.markdown('<div class="section-title">📊  Emotion Breakdown</div>',
                                    unsafe_allow_html=True)

                        pairs = [(label_map[int(c)], p) for c, p in zip(model.classes_, proba)]
                        pairs.sort(key=lambda x: x[1], reverse=True)

                        for emo, p in pairs:
                            em, c, g = EMO.get(emo, ('🙂','#94a3b8','#94a3b8'))
                            pct = p * 100
                            st.markdown(f"""
                            <div class="bar-container">
                                <div class="bar-header">
                                    <span>{em} &nbsp;
                                        <b style="text-transform:capitalize; color:#f1f5f9;">{emo}</b>
                                    </span>
                                    <span style="color:{c}; font-weight:700;">{pct:.1f}%</span>
                                </div>
                                <div class="bar-track">
                                    <div class="bar-fill"
                                         style="width:{pct}%; background:{g};
                                                box-shadow:0 0 14px {c}66;">
                                    </div>
                                </div>
                            </div>
                            """, unsafe_allow_html=True)

    # ---- Examples ----
    st.markdown('<div class="section-title">💡  Try these examples</div>',
                unsafe_allow_html=True)
    examples = [
        "I feel so happy and grateful today!",
        "I miss my old friends and feel empty.",
        "This is absolutely frustrating and unfair!",
        "I'm terrified about what might happen next.",
        "You mean the world to me, I love you.",
        "Wow, I did not see that coming at all!",
    ]
    cols = st.columns(3)
    for i, ex in enumerate(examples):
        with cols[i % 3]:
            st.code(ex, language=None)

    st.markdown("""
    <div class="footer-note">
        Powered by <b>TF-IDF + Logistic Regression</b> · Loaded from joblib artifacts
    </div>
    """, unsafe_allow_html=True)