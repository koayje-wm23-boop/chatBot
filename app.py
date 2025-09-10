import streamlit as st
import os, json, datetime, uuid
import joblib
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from keras_preprocessing.sequence import pad_sequences

# ---------------- Basic setup ----------------
st.set_page_config(page_title="🎓 UniHelp", page_icon="🎓", layout="centered")

MODEL_DIR    = "models"
DL_MODEL_DIR = "models_dl"
DATA_PATH    = "data/intents_university.json"
REPORTS_DIR  = "reports"
LOG_DIR      = "logs"
os.makedirs(LOG_DIR, exist_ok=True)

CHAT_CSV = os.path.join(LOG_DIR, "chat_logs.csv")

# ---------------- Logging ----------------
def ensure_chat_csv():
    if not os.path.exists(CHAT_CSV):
        pd.DataFrame(columns=[
            "timestamp","chat_id","user_text",
            "engine","intent","score","response"
        ]).to_csv(CHAT_CSV, index=False)

def log_row(chat_id, user_text, engine, intent, score, response):
    ensure_chat_csv()
    ts = datetime.datetime.now().isoformat(timespec="seconds")
    pd.DataFrame([{
        "timestamp": ts,
        "chat_id": chat_id,
        "user_text": user_text,
        "engine": engine,           # "ML" or "DL"
        "intent": intent,
        "score": score,
        "response": response
    }]).to_csv(CHAT_CSV, mode="a", header=False, index=False)

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
        meta = joblib.load(os.path.join(DL_MODEL_DIR, "dl_meta.joblib"))
        maxlen = meta.get("maxlen", 20)
        return model, tokenizer, le, label_to_responses, maxlen
    except Exception as e:
        st.warning(f"⚠️ DL model not available. Using ML only. ({e})")
        return None, None, None, None, None

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
    idx = int(np.argmax(probs))
    return le.classes_[idx], float(probs[idx])

# ---------------- UI State ----------------
if "chat_id" not in st.session_state:
    st.session_state.chat_id = str(uuid.uuid4())[:8]
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role":"assistant","content":"Hello! 👋 I’m UniHelp. Ask me about TAR UMT admissions, programmes, fees, scholarships, housing, library or campus life."}
    ]
if "page" not in st.session_state:
    st.session_state.page = "chat"

# ---------------- Sidebar ----------------
with st.sidebar:
    st.header("UniHelp")
    if st.button("➕ New Chat"):
        st.session_state.chat_id = str(uuid.uuid4())[:8]
        st.session_state.messages = [
            {"role":"assistant","content":"Hello! 👋 I’m UniHelp. Ask me about TAR UMT admissions, programmes, fees, scholarships, housing, library or campus life."}
        ]
        st.rerun()

    if st.button("💬 Chat"): 
        st.session_state.page = "chat"; st.rerun()
    if st.button("📊 Evaluation"): 
        st.session_state.page = "evaluation"; st.rerun()

    st.markdown("---")
    threshold = st.slider("Confidence threshold", 0.0, 1.0, 0.45, 0.01,
                          help="If model confidence < this value, show a 'not confident' message.")
    st.caption("Tip: Lower threshold = more answers; higher = stricter & safer.")

# ---------------- Styling ----------------
st.markdown("""
<style>
section.main > div { max-width: 850px; margin: auto; }
.bubble { border-radius: 14px; padding: 10px 14px; margin: 6px 0; }
.user { background: #0e1117; border: 1px solid #2b2b2b; }
.bot  { background: #161a23; border: 1px solid #2b2b2b; }
.btnrow { margin: 10px 0 16px 0; }
</style>
""", unsafe_allow_html=True)

# ---------------- Chat Page ----------------
if st.session_state.page == "chat":
    st.markdown("<h1 style='text-align:center'>🎓 UniHelp</h1>", unsafe_allow_html=True)

    # Quick Buttons (2 rows)
    faq_map = {
        "🎓 Programs": "What programs are offered?",
        "💰 Fees": "How much is the tuition fee?",
        "🎓 Scholarships": "What scholarships are available?",
        "🏠 Hostel": "How do I apply for housing?",
        "📚 Library": "What are the library hours?",
        "☎️ Contact": "How do I contact TAR UMT?"
    }
    labels = list(faq_map.keys()); queries = list(faq_map.values())

    c1, c2, c3 = st.columns(3)
    if c1.button(labels[0], use_container_width=True): st.session_state.messages.append({"role":"user","content":queries[0]}); st.rerun()
    if c2.button(labels[1], use_container_width=True): st.session_state.messages.append({"role":"user","content":queries[1]}); st.rerun()
    if c3.button(labels[2], use_container_width=True): st.session_state.messages.append({"role":"user","content":queries[2]}); st.rerun()
    c4, c5, c6 = st.columns(3)
    if c4.button(labels[3], use_container_width=True): st.session_state.messages.append({"role":"user","content":queries[3]}); st.rerun()
    if c5.button(labels[4], use_container_width=True): st.session_state.messages.append({"role":"user","content":queries[4]}); st.rerun()
    if c6.button(labels[5], use_container_width=True): st.session_state.messages.append({"role":"user","content":queries[5]}); st.rerun()

    # Display chat history
    for m in st.session_state.messages:
        with st.chat_message(m["role"]):
            cls = "bot" if m["role"]=="assistant" else "user"
            st.markdown(f"<div class='bubble {cls}'>{m['content']}</div>", unsafe_allow_html=True)

    # Load artifacts once (outside of submit for speed)
    ml_model, ml_vectorizer, ml_responses = load_ml_artifacts()
    dl_model, dl_tokenizer, dl_le, dl_responses, dl_maxlen = try_load_dl()

    # User input
    user_text = st.chat_input("Type your message…")
    if user_text:
        st.session_state.messages.append({"role":"user","content":user_text})

        # ---- ML inference ----
        ml_intent, ml_score = predict_ml(user_text, ml_model, ml_vectorizer)
        if ml_score >= threshold:
            ml_answer = ml_responses.get(ml_intent, ["I'm not sure."])[0]
        else:
            ml_answer = "Sorry, I’m not confident about that. Please try rephrasing."

        # show ML
        with st.chat_message("assistant"):
            st.markdown(f"<div class='bubble bot'>🤖 <b>ML</b>: {ml_answer}</div>", unsafe_allow_html=True)
            st.caption(f"DEBUG → intent={ml_intent}, score={ml_score:.2f}, threshold={threshold:.2f}")
        st.session_state.messages.append({"role":"assistant","content":f"🤖 ML: {ml_answer}"})
        log_row(st.session_state.chat_id, user_text, "ML", ml_intent, ml_score, ml_answer)

        # ---- DL inference (if available) ----
        if dl_model and dl_tokenizer and dl_le:
            dl_intent, dl_score = predict_dl(user_text, dl_model, dl_tokenizer, dl_le, maxlen=dl_maxlen or 20)
            if dl_score >= threshold:
                dl_answer = dl_responses.get(dl_intent, ["I'm not sure."])[0]
            else:
                dl_answer = "Sorry, I’m not confident about that. Please try rephrasing."

            with st.chat_message("assistant"):
                st.markdown(f"<div class='bubble bot'>🧠 <b>DL</b>: {dl_answer}</div>", unsafe_allow_html=True)
                st.caption(f"DEBUG → intent={dl_intent}, score={dl_score:.2f}, threshold={threshold:.2f}")

            st.session_state.messages.append({"role":"assistant","content":f"🧠 DL: {dl_answer}"})
            log_row(st.session_state.chat_id, user_text, "DL", dl_intent, dl_score, dl_answer)
        else:
            with st.chat_message("assistant"):
                st.markdown("<div class='bubble bot'>🧠 DL: ⚠️ Not available on this deployment.</div>", unsafe_allow_html=True)
            st.session_state.messages.append({"role":"assistant","content":"🧠 DL: ⚠️ Not available on this deployment."})

# ---------------- Evaluation Page ----------------
elif st.session_state.page == "evaluation":
    st.title("📊 Model Evaluation")
    st.write("This page shows the latest saved evaluation reports produced during training.")

    ml_eval = os.path.join(REPORTS_DIR, "eval.txt")
    if os.path.exists(ml_eval):
        st.subheader("ML (TF-IDF + Logistic Regression)")
        st.text(open(ml_eval, "r", encoding="utf-8").read())
    else:
        st.info("No ML evaluation file found (reports/eval.txt). Train ML first.")

    dl_eval = os.path.join(REPORTS_DIR, "eval_dl.txt")
    if os.path.exists(dl_eval):
        st.subheader("DL (Neural Net, Keras)")
        st.text(open(dl_eval, "r", encoding="utf-8").read())
    else:
        st.info("No DL evaluation file found (reports/eval_dl.txt). Train DL first.")
