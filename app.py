# app.py
import streamlit as st
import json, os, random, re, datetime
import joblib
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="🎓 UniHelp — Streamlit Chatbot", page_icon="🎓", layout="centered")

MODEL_DIR = "models"
DATA_PATH = "data/intents_university.json"
LOG_PATH = "logs/chat_logs.csv"

# ------------ Styling ------------
st.markdown("""
<style>
section.main > div { max-width: 900px; margin: auto; }
.chat { border-radius: 14px; padding: 10px 14px; margin: 6px 0; }
.user { background: #0e1117; border: 1px solid #2b2b2b; }
.bot  { background: #161a23; border: 1px solid #2b2b2b; }
.small { color:#9aa0a6;font-size:0.85rem }
</style>
""", unsafe_allow_html=True)

# ------------ Cached artifacts ------------
@st.cache_resource
def load_artifacts():
    model = joblib.load(os.path.join(MODEL_DIR, "intent_model.joblib"))
    label_to_responses = joblib.load(os.path.join(MODEL_DIR, "label_to_responses.joblib"))
    vectorizer = model.named_steps["tfidf"]
    return model, vectorizer, label_to_responses

def artifacts_exist() -> bool:
    need = [
        os.path.join(MODEL_DIR, "intent_model.joblib"),
        os.path.join(MODEL_DIR, "label_to_responses.joblib"),
    ]
    return all(os.path.exists(p) for p in need)

def train_now(data_path=DATA_PATH):
    from train_evaluate import main as train_main
    with st.spinner("Training model…"):
        os.makedirs(MODEL_DIR, exist_ok=True)
        os.makedirs("reports", exist_ok=True)
        train_main(data_path, MODEL_DIR, "reports")
    st.cache_resource.clear()
    st.success("Training complete.")

# ------------ Sidebar ------------
with st.sidebar:
    st.header("Setup")
    retrain = st.button("Train / Retrain Model")
    threshold = st.slider("Confidence threshold", 0.0, 1.0, 0.55, 0.01)
    st.markdown("Upload a custom intents JSON to fine-tune:")
    uploaded = st.file_uploader("intents.json", type=["json"], accept_multiple_files=False)

if retrain:
    if uploaded is not None:
        with open(DATA_PATH, "wb") as f: f.write(uploaded.getvalue())
        st.success("Uploaded new intents.json")
    train_now(DATA_PATH)
    st.stop()

# Auto-train first time
if not artifacts_exist():
    train_now(DATA_PATH)

# ------------ Load model & data ------------
model, vectorizer, label_to_responses = load_artifacts()

with open(DATA_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)
patterns, pattern_to_label = [], {}
for it in data["intents"]:
    for p in it.get("patterns", []):
        patterns.append(p)
        pattern_to_label[p] = it["tag"]

def predict_intent(text):
    probs = model.predict_proba([text])[0]
    labels = model.classes_
    j = probs.argmax()
    return labels[j], float(probs[j])

def retrieval_fallback(text, min_sim=0.22):
    try:
        X = vectorizer.transform([text])
        P = vectorizer.transform(patterns)
        sims = cosine_similarity(X, P)[0]
        idx = sims.argmax()
        if sims[idx] >= min_sim:
            return pattern_to_label[patterns[idx]], float(sims[idx])
    except Exception:
        pass
    return "fallback", 0.0

# ------------ Entities (simple regex) ------------
def extract_entities(t):
    ents = {}
    m = re.search(r"(cs|it|business|engineering|ai|data|law|medicine)", t, re.I)
    if m: ents["program_name"] = m.group(0).title()
    m = re.search(r"(undergrad|postgrad|master|phd)", t, re.I)
    if m: ents["level"] = m.group(0).title()
    m = re.search(r"\b(\d{2}/\d{2}/\d{2,4}|\d{1,2}-\d{1,2}-\d{2,4}|today|tomorrow|next (mon|tue|wed|thu|fri|sat|sun))\b", t, re.I)
    if m: ents["date"] = m.group(0)
    m = re.search(r"\b\d{2}:\d{2}\s?(am|pm)?\b", t, re.I)
    if m: ents["time"] = m.group(0)
    m = re.search(r"\b[a-z0-9._%+-]+@[a-z0-9.-]+\.[a-z]{2,}\b", t, re.I)
    if m: ents["email"] = m.group(0)
    m = re.search(r"\b[AU]\d{7}\b", t) # e.g., A1234567 student id
    if m: ents["student_id"] = m.group(0)
    return ents

# ------------ External API stub ------------
def next_bus_to_campus():
    # Simulated data (replace with real API call if you have one)
    timetable = [
        ("North Gate", "10 min"),
        ("Central Station", "18 min"),
        ("South Park", "27 min"),
    ]
    return "Next buses to campus:\n" + "\n".join([f"• {stop}: {eta}" for stop, eta in timetable])

# ------------ Logging & satisfaction ------------
def ensure_logs():
    os.makedirs("logs", exist_ok=True)
    if not os.path.exists(LOG_PATH):
        pd.DataFrame(columns=["timestamp","user","bot","intent","score","feedback","rating"]).to_csv(LOG_PATH, index=False)

def log_row(user, bot, intent, score, feedback=None, rating=None):
    ensure_logs()
    ts = datetime.datetime.now().isoformat(timespec="seconds")
    pd.DataFrame([{
        "timestamp": ts, "user": user, "bot": bot,
        "intent": intent, "score": score,
        "feedback": feedback, "rating": rating
    }]).to_csv(LOG_PATH, mode="a", header=False, index=False)

# ------------ Header ------------
st.title("🎓 UniHelp — Streamlit Chatbot")
st.caption("Intent classification + templated response generation, with entities & evaluation.")

# ------------ Chat loop ------------
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role":"assistant","content":"Hi! Ask me about admissions, programs, tuition, scholarships, library, or transport."}
    ]

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        cls = "bot" if m["role"]=="assistant" else "user"
        st.markdown(f"<div class='chat {cls}'>{m['content']}</div>", unsafe_allow_html=True)

user_text = st.chat_input("Type your message…")
if user_text:
    st.session_state.messages.append({"role":"user","content":user_text})
    with st.chat_message("user"):
        st.markdown(f"<div class='chat user'>{user_text}</div>", unsafe_allow_html=True)

    ents = extract_entities(user_text)
    intent, score = predict_intent(user_text)
    if score < st.session_state.get("thr", 0.0) or score < threshold:
        intent, score = retrieval_fallback(user_text)

    # Dynamic responses for some intents
    if intent == "transport" or "bus" in user_text.lower():
        bot = next_bus_to_campus()
    else:
        bot = random.choice(label_to_responses.get(intent, label_to_responses.get("fallback", ["I'm not sure."])))
        # lightly personalize if we captured a program/level
        if intent in {"admission_requirements","course_advice","programs_offered"} and ("program_name" in ents or "level" in ents):
            add = []
            if "program_name" in ents: add.append(f"program: **{ents['program_name']}**")
            if "level" in ents: add.append(f"level: **{ents['level']}**")
            bot += "\n\nDetected " + ", ".join(add) + "."
        if intent == "reset_password":
            bot = ("Reset steps:\n"
                   "1) Open Student Portal → **Forgot Password**\n"
                   "2) Enter your student email\n"
                   "3) Click reset link sent to your inbox\n"
                   "4) If no email, contact **help@uni.edu**")

    with st.chat_message("assistant"):
        st.markdown(f"<div class='chat bot'>{bot}</div>", unsafe_allow_html=True)
        c1, c2, c3 = st.columns([1,1,2])
        if c1.button("👍 Helpful", key=f"up_{len(st.session_state.messages)}"):
            log_row(user_text, bot, intent, score, feedback="up")
            st.toast("Thanks!")
        if c2.button("👎 Not helpful", key=f"down_{len(st.session_state.messages)}"):
            log_row(user_text, bot, intent, score, feedback="down")
            st.toast("Recorded.")
        with c3:
            rating = st.select_slider("Rate this reply", [1,2,3,4,5], key=f"rt_{len(st.session_state.messages)}")
            if st.button("Submit rating", key=f"rs_{len(st.session_state.messages)}"):
                log_row(user_text, bot, intent, score, rating=int(rating))
                st.toast("Rating saved.")

    st.session_state.messages.append({"role":"assistant","content":bot})
    log_row(user_text, bot, intent, score)

# ------------ Evaluation panel ------------
st.divider(); st.subheader("📊 Evaluation")
eval_path = os.path.join("reports","eval.txt")
if os.path.exists(eval_path):
    with open(eval_path,"r",encoding="utf-8") as f: st.text(f.read())
else:
    st.info("No evaluation report yet. Use **Train / Retrain Model** in the sidebar.")

st.divider(); st.subheader("🧪 Download chat logs")
ensure_logs()
if os.path.exists(LOG_PATH):
    with open(LOG_PATH, "rb") as f:
        st.download_button("Download chat_logs.csv", f, file_name="chat_logs.csv", mime="text/csv")
