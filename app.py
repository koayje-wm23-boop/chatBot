import streamlit as st
import os, json, random, datetime, uuid
import joblib
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# ---------- Basic page setup ----------
st.set_page_config(page_title="🎓 UniHelp", page_icon="🎓", layout="centered")

MODEL_DIR = "models"
DATA_PATH = "data/intents_university.json"
REPORTS_DIR = "reports"
LOG_DIR = "logs"
CHAT_CSV = os.path.join(LOG_DIR, "chat_logs.csv")

os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)

# ---------- Tiny helpers ----------
def ensure_chat_csv():
    if not os.path.exists(CHAT_CSV):
        pd.DataFrame(columns=["timestamp","chat_id","user","bot","intent","score"]).to_csv(CHAT_CSV, index=False)

def log_row(chat_id, user, bot, intent, score):
    ensure_chat_csv()
    ts = datetime.datetime.now().isoformat(timespec="seconds")
    pd.DataFrame([{
        "timestamp": ts, "chat_id": chat_id, "user": user,
        "bot": bot, "intent": intent, "score": score
    }]).to_csv(CHAT_CSV, mode="a", header=False, index=False)

@st.cache_resource
def load_artifacts():
    model = joblib.load(os.path.join(MODEL_DIR, "intent_model.joblib"))
    label_to_responses = joblib.load(os.path.join(MODEL_DIR, "label_to_responses.joblib"))
    vectorizer = model.named_steps["tfidf"]
    return model, vectorizer, label_to_responses

def artifacts_exist():
    need = [os.path.join(MODEL_DIR, "intent_model.joblib"),
            os.path.join(MODEL_DIR, "label_to_responses.joblib")]
    return all(os.path.exists(p) for p in need)

def train_now(data_path=DATA_PATH):
    from train_evaluate import main as train_main
    with st.spinner("Training model…"):
        os.makedirs(MODEL_DIR, exist_ok=True)
        train_main(data_path, MODEL_DIR, REPORTS_DIR)
    st.cache_resource.clear()
    st.success("Training complete.")

# Auto-train on first run
if not artifacts_exist():
    train_now(DATA_PATH)

# Load model + responses
model, vectorizer, label_to_responses = load_artifacts()

# Load patterns for retrieval fallback
with open(DATA_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)
patterns, pattern_to_label = [], {}
for it in data["intents"]:
    tag = it["tag"]
    for p in it.get("patterns", []):
        patterns.append(p)
        pattern_to_label[p] = tag

def predict_intent(text):
    probs = model.predict_proba([text])[0]
    labels = model.classes_
    j = probs.argmax()
    return labels[j], float(probs[j])

def retrieval_fallback(text, min_sim=0.12):  # relaxed to be friendly
    try:
        X = vectorizer.transform([text])
        P = vectorizer.transform(patterns)
        sims = cosine_similarity(X, P)[0]
        j = sims.argmax()
        if sims[j] >= min_sim:
            return pattern_to_label[patterns[j]], float(sims[j])
    except Exception:
        pass
    return "fallback", 0.0

def get_response(intent):
    return random.choice(label_to_responses.get(intent, label_to_responses.get("fallback", ["I'm not sure."])))

# ---------- Sidebar: simple controls ----------
def new_chat():
    st.session_state.chat_id = str(uuid.uuid4())[:8]
    st.session_state.messages = [
        {"role":"assistant","content":"Hi! Ask me about admissions, programs, tuition, scholarships, library, housing, or contacts."}
    ]
if "chat_id" not in st.session_state:
    new_chat()

with st.sidebar:
    st.header("UniHelp")
    if st.button("➕ New Chat"):
        new_chat()
        st.experimental_rerun()
    retrain = st.button("🔁 Retrain model")
    threshold = st.slider("Confidence threshold", 0.0, 1.0, 0.45, 0.01)
    st.markdown("---")
    rating = st.select_slider("Rate this chat", options=[1,2,3,4,5], value=3)
    if st.button("Save rating"):
        st.toast("Thanks for your rating! ⭐")

if retrain:
    train_now(DATA_PATH)
    st.experimental_rerun()

# ---------- Minimal styling ----------
st.markdown("""
<style>
section.main > div { max-width: 850px; margin: auto; }
.bubble { border-radius: 14px; padding: 10px 14px; margin: 6px 0; }
.user { background: #0e1117; border: 1px solid #2b2b2b; }
.bot  { background: #161a23; border: 1px solid #2b2b2b; }
</style>
""", unsafe_allow_html=True)

st.title("🎓 UniHelp")

# render history
for m in st.session_state.get("messages", []):
    with st.chat_message(m["role"]):
        cls = "bot" if m["role"]=="assistant" else "user"
        st.markdown(f"<div class='bubble {cls}'>{m['content']}</div>", unsafe_allow_html=True)

# input
user_text = st.chat_input("Type your message…")
if user_text:
    st.session_state.messages.append({"role":"user","content":user_text})
    with st.chat_message("user"):
        st.markdown(f"<div class='bubble user'>{user_text}</div>", unsafe_allow_html=True)

    intent, score = predict_intent(user_text)
    if score < threshold:
        intent, score = retrieval_fallback(user_text)

    bot_text = get_response(intent)

    with st.chat_message("assistant"):
        st.markdown(f"<div class='bubble bot'>{bot_text}</div>", unsafe_allow_html=True)

    st.session_state.messages.append({"role":"assistant","content":bot_text})
    log_row(st.session_state.chat_id, user_text, bot_text, intent, score)

# evaluation
st.divider()
st.subheader("📊 Evaluation")
eval_path = os.path.join(REPORTS_DIR, "eval.txt")
if os.path.exists(eval_path):
    with open(eval_path, "r", encoding="utf-8") as f:
        st.text(f.read())
else:
    st.info("No evaluation report yet — click Retrain to generate it.")
