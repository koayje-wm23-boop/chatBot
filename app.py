import streamlit as st
import os, json, datetime, uuid
import joblib
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from keras_preprocessing.sequence import pad_sequences
import plotly.express as px

# ---------------- Basic setup ----------------
st.set_page_config(page_title="🎓 UniHelp", page_icon="🎓", layout="centered")

MODEL_DIR   = "models"
DL_MODEL_DIR= "models_dl"
DATA_PATH   = "data/intents_university.json"
REPORTS_DIR = "reports"
LOG_DIR     = "logs"

os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)

# ---------------- Load ML Artifacts ----------------
@st.cache_resource
def load_ml_artifacts():
    model = joblib.load(os.path.join(MODEL_DIR, "intent_model.joblib"))
    label_to_responses = joblib.load(os.path.join(MODEL_DIR, "label_to_responses.joblib"))
    vectorizer = model.named_steps["tfidf"]
    return model, vectorizer, label_to_responses

# ---------------- Load DL Artifacts (Safe) ----------------
@st.cache_resource
def try_load_dl():
    try:
        model = load_model(os.path.join(DL_MODEL_DIR, "dl_intent_model.h5"))
        tokenizer = joblib.load(os.path.join(DL_MODEL_DIR, "tokenizer.joblib"))
        le = joblib.load(os.path.join(DL_MODEL_DIR, "label_encoder.joblib"))
        label_to_responses = joblib.load(os.path.join(DL_MODEL_DIR, "label_to_responses.joblib"))
        return model, tokenizer, le, label_to_responses
    except Exception as e:
        st.warning(f"⚠️ DL model not available. Fallback only to ML. ({e})")
        return None, None, None, None

# ---------------- Predictors ----------------
def predict_ml(text, model, vectorizer):
    probs = model.predict_proba([text])[0]
    labels = model.classes_
    j = probs.argmax()
    return labels[j], float(probs[j])

def predict_dl(text, model, tokenizer, le, maxlen=20):
    seq = tokenizer.texts_to_sequences([text])
    X = pad_sequences(seq, maxlen=maxlen, padding="post")
    probs = model.predict(X, verbose=0)[0]
    idx = np.argmax(probs)
    return le.classes_[idx], float(probs[idx])

# ---------------- UI State ----------------
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role":"assistant","content":"Hi! You can ask about TAR UMT admissions, programs, fees, scholarships, hostel, or library."}
    ]

if "page" not in st.session_state:
    st.session_state.page = "chat"

# ---------------- Sidebar ----------------
with st.sidebar:
    st.header("UniHelp")
    if st.button("➕ New Chat"):
        st.session_state.messages = [
            {"role":"assistant","content":"Hi! You can ask about TAR UMT admissions, programs, fees, scholarships, hostel, or library."}
        ]
        st.rerun()

    if st.button("💬 Chat"): st.session_state.page = "chat"; st.rerun()
    if st.button("📊 Evaluation"): st.session_state.page = "evaluation"; st.rerun()

    st.markdown("---")
    threshold = st.slider("Confidence threshold", 0.0, 1.0, 0.3, 0.01)

# ---------------- Styling ----------------
st.markdown("""
<style>
section.main > div { max-width: 850px; margin: auto; }
.bubble { border-radius: 14px; padding: 10px 14px; margin: 6px 0; }
.user { background: #0e1117; border: 1px solid #2b2b2b; }
.bot  { background: #161a23; border: 1px solid #2b2b2b; }
</style>
""", unsafe_allow_html=True)

# ---------------- Chat Page ----------------
if st.session_state.page == "chat":
    st.markdown("<h1 style='text-align:center'>🎓 UniHelp</h1>", unsafe_allow_html=True)

    # Quick Buttons
    faq_map = {
        "Programs": "What programs are offered?",
        "Fees": "How much is the tuition fee?",
        "Scholarships": "What scholarships are available?",
        "Hostel": "How do I apply for housing?",
        "Library": "What are the library hours?",
        "Contact": "How do I contact TAR UMT?"
    }
    cols1, cols2, cols3 = st.columns(3)
    for i, label in enumerate(faq_map.keys()):
        if (i < 3 and cols1.button(label)) or (i >= 3 and (cols2 if i<6 else cols3).button(label)):
            st.session_state.messages.append({"role":"user","content":faq_map[label]})
            st.rerun()

    # Display history
    for m in st.session_state.messages:
        with st.chat_message(m["role"]):
            cls = "bot" if m["role"]=="assistant" else "user"
            st.markdown(f"<div class='bubble {cls}'>{m['content']}</div>", unsafe_allow_html=True)

    # Input
    user_text = st.chat_input("Type your message…")
    if user_text:
        st.session_state.messages.append({"role":"user","content":user_text})

        # Load models
        ml_model, vectorizer, ml_responses = load_ml_artifacts()
        dl_model, tokenizer, le, dl_responses = try_load_dl()

        # ML prediction
        ml_intent, ml_score = predict_ml(user_text, ml_model, vectorizer)
        ml_answer = ml_responses.get(ml_intent, ["I'm not sure."])[0] if ml_score >= threshold else "Sorry (ML) not confident."

        # DL prediction
        if dl_model:
            dl_intent, dl_score = predict_dl(user_text, dl_model, tokenizer, le)
            dl_answer = dl_responses.get(dl_intent, ["I'm not sure."])[0] if dl_score >= threshold else "Sorry (DL) not confident."
        else:
            dl_answer = "⚠️ DL model unavailable."

        # Show both
        st.session_state.messages.append({"role":"assistant","content":f"🤖 **ML:** {ml_answer}"})
        st.session_state.messages.append({"role":"assistant","content":f"🧠 **DL:** {dl_answer}"})
        st.rerun()

# ---------------- Evaluation Page ----------------
elif st.session_state.page == "evaluation":
    st.title("📊 Model Evaluation")

    # ML evaluation
    eval_path = os.path.join(REPORTS_DIR, "eval.txt")
    if os.path.exists(eval_path):
        st.subheader("ML (TF-IDF) Evaluation")
        st.text(open(eval_path).read())

    # DL evaluation
    eval_dl_path = os.path.join(REPORTS_DIR, "eval_dl.txt")
    if os.path.exists(eval_dl_path):
        st.subheader("DL (Neural Net) Evaluation")
        st.text(open(eval_dl_path).read())
