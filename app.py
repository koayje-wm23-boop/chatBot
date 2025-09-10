import streamlit as st
import os, json, datetime, uuid
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
import plotly.express as px

# ---------------- Basic setup ----------------
st.set_page_config(page_title="🎓 UniHelp", page_icon="🎓", layout="centered")

MODEL_DIR   = "models"
DL_MODEL_DIR= "models_dl"
DATA_PATH   = "data/intents_university.json"
REPORTS_DIR = "reports"
LOG_DIR     = "logs"
CHAT_CSV    = os.path.join(LOG_DIR, "chat_logs.csv")

os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)

# ---------------- Logging ----------------
def ensure_chat_csv():
    if not os.path.exists(CHAT_CSV):
        pd.DataFrame(columns=["timestamp","chat_id","user","bot","model","intent","score"]).to_csv(CHAT_CSV, index=False)

def log_row(chat_id, user, bot, model, intent=None, score=None):
    ensure_chat_csv()
    ts = datetime.datetime.now().isoformat(timespec="seconds")
    pd.DataFrame([{
        "timestamp": ts, "chat_id": chat_id, "user": user,
        "bot": bot, "model": model, "intent": intent, "score": score
    }]).to_csv(CHAT_CSV, mode="a", header=False, index=False)

# ---------------- Load JSON ----------------
def load_intents_json(path=DATA_PATH):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def norm(s: str) -> str:
    return " ".join(s.lower().strip().split())

# ---------------- Load ML Artifacts ----------------
@st.cache_resource
def load_ml_artifacts():
    model = joblib.load(os.path.join(MODEL_DIR, "intent_model.joblib"))
    label_to_responses = joblib.load(os.path.join(MODEL_DIR, "label_to_responses.joblib"))
    vectorizer = model.named_steps["tfidf"]
    return model, vectorizer, label_to_responses

# ---------------- Load DL Artifacts ----------------
@st.cache_resource
def load_dl_artifacts():
    model = load_model(os.path.join(DL_MODEL_DIR, "dl_intent_model.h5"))
    tokenizer = joblib.load(os.path.join(DL_MODEL_DIR, "tokenizer.joblib"))
    le = joblib.load(os.path.join(DL_MODEL_DIR, "label_encoder.joblib"))
    label_to_responses = joblib.load(os.path.join(DL_MODEL_DIR, "label_to_responses.joblib"))
    return model, tokenizer, le, label_to_responses

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

# ---------------- Sidebar ----------------
if "chat_id" not in st.session_state:
    st.session_state.chat_id = str(uuid.uuid4())[:8]
    st.session_state.messages = [
        {"role":"assistant","content":"Hi! You can ask about TAR UMT admissions, programs, fees, scholarships, hostel, or library."}
    ]

if "page" not in st.session_state:
    st.session_state.page = "chat"

with st.sidebar:
    st.header("UniHelp")
    if st.button("➕ New Chat"):
        st.session_state.chat_id = str(uuid.uuid4())[:8]
        st.session_state.messages = [
            {"role":"assistant","content":"Hi! You can ask about TAR UMT admissions, programs, fees, scholarships, hostel, or library."}
        ]
        st.rerun()

    # Model selection
    model_type = st.radio("Select Model", ["ML (TF-IDF)", "DL (Neural Net)"])

    # Retrain buttons
    if st.button("🔁 Retrain ML"):
        from train_evaluate import main as train_ml
        train_ml(DATA_PATH, MODEL_DIR, REPORTS_DIR)
        st.cache_resource.clear()
        st.success("ML Model retrained.")

    if st.button("🔁 Retrain DL"):
        from train_evaluate_dl import main as train_dl
        train_dl(DATA_PATH, DL_MODEL_DIR, REPORTS_DIR)
        st.cache_resource.clear()
        st.success("DL Model retrained.")

    st.markdown("---")
    if st.button("💬 Chat"): st.session_state.page = "chat"; st.rerun()
    if st.button("📊 Evaluation"): st.session_state.page = "evaluation"; st.rerun()

    st.markdown("---")
    threshold = st.slider("Confidence threshold", 0.0, 1.0, 0.45, 0.01)
    rating = st.select_slider("⭐ Rate this chat", options=[1,2,3,4,5], value=3)
    if st.button("Save rating"): st.toast("Thanks for your rating!")

# ---------------- Styling ----------------
st.markdown("""
<style>
section.main > div { max-width: 850px; margin: auto; }
.bubble { border-radius: 14px; padding: 10px 14px; margin: 6px 0; }
.user { background: #0e1117; border: 1px solid #2b2b2b; }
.bot  { background: #161a23; border: 1px solid #2b2b2b; }
button[kind="secondary"] { border-radius: 20px !important; }
</style>
""", unsafe_allow_html=True)

# ---------------- Chat Page ----------------
if st.session_state.page == "chat":
    st.markdown("<h1 style='text-align:center'>🎓 UniHelp</h1>", unsafe_allow_html=True)

    # Quick Access Buttons
    faq_map = {
        "🎓 Programs": "What programs are offered?",
        "💰 Fees": "How much is the tuition fee?",
        "🎓 Scholarships": "What scholarships are available?",
        "🏠 Hostel": "How do I apply for housing?",
        "📚 Library": "What are the library hours?",
        "☎️ Contact": "How do I contact TAR UMT?"
    }
    labels, queries = list(faq_map.keys()), list(faq_map.values())
    cols1, cols2 = st.columns(3), st.columns(3)
    for i, col in enumerate(cols1):
        if col.button(labels[i], use_container_width=True):
            st.session_state.messages.append({"role":"user","content":queries[i]}); st.rerun()
    for i, col in enumerate(cols2):
        if col.button(labels[i+3], use_container_width=True):
            st.session_state.messages.append({"role":"user","content":queries[i+3]}); st.rerun()

    # Chat History
    for m in st.session_state.messages:
        with st.chat_message(m["role"]):
            cls = "bot" if m["role"]=="assistant" else "user"
            st.markdown(f"<div class='bubble {cls}'>{m['content']}</div>", unsafe_allow_html=True)

    # User Input
    user_text = st.chat_input("Type your message…")
    if user_text:
        st.session_state.messages.append({"role":"user","content":user_text})

        if model_type.startswith("ML"):
            model, vectorizer, label_to_responses = load_ml_artifacts()
            intent, score = predict_ml(user_text, model, vectorizer)
        else:
            dl_model, tokenizer, le, label_to_responses = load_dl_artifacts()
            intent, score = predict_dl(user_text, dl_model, tokenizer, le)

        if score < threshold:
            response = "Sorry, I’m not confident about that. Please try rephrasing."
        else:
            response = label_to_responses.get(intent, ["I'm not sure."])[0]

        st.session_state.messages.append({"role":"assistant","content":response})
        log_row(st.session_state.chat_id, user_text, response, model_type, intent, score)
        st.rerun()

# ---------------- Evaluation Page ----------------
elif st.session_state.page == "evaluation":
    st.title("📊 Model Evaluation")

    # ML evaluation
    eval_path = os.path.join(REPORTS_DIR, "eval.txt")
    if os.path.exists(eval_path):
        with open(eval_path, "r") as f: lines = f.readlines()
        data = []
        for line in lines:
            parts = line.strip().split()
            if len(parts) >= 4 and not parts[0].startswith("="):
                try:
                    intent = parts[0]
                    prec, rec, f1 = float(parts[1]), float(parts[2]), float(parts[3])
                    support = int(parts[4]) if len(parts) > 4 and parts[4].isdigit() else "-"
                    data.append([intent, prec, rec, f1, support])
                except: continue
        if data:
            df = pd.DataFrame(data, columns=["Intent","Precision","Recall","F1-score","Support"])
            st.subheader("ML (TF-IDF) Evaluation")
            st.dataframe(df, use_container_width=True)

            st.markdown("### 📊 Intent-wise F1 Score")
            fig_f1 = px.bar(df[df["Intent"]!="weighted_avg"], x="Intent", y="F1-score", color="F1-score", text="F1-score", range_y=[0,1])
            st.plotly_chart(fig_f1, use_container_width=True)

            col1, col2 = st.columns(2)
            with col1:
                fig_prec = px.bar(df[df["Intent"]!="weighted_avg"], x="Intent", y="Precision", color="Precision", text="Precision", range_y=[0,1])
                st.plotly_chart(fig_prec, use_container_width=True)
            with col2:
                fig_rec = px.bar(df[df["Intent"]!="weighted_avg"], x="Intent", y="Recall", color="Recall", text="Recall", range_y=[0,1])
                st.plotly_chart(fig_rec, use_container_width=True)

            if "Support" in df.columns:
                st.markdown("### 📊 Training Data Distribution")
                fig_pie = px.pie(df[df["Intent"]!="weighted_avg"], names="Intent", values="Support", title="Samples per Intent")
                st.plotly_chart(fig_pie, use_container_width=True)

    # DL evaluation
    eval_dl_path = os.path.join(REPORTS_DIR, "eval_dl.txt")
    if os.path.exists(eval_dl_path):
        st.subheader("DL (Neural Net) Evaluation")
        st.text(open(eval_dl_path).read())
