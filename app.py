import streamlit as st
import json, os, random, datetime
import joblib
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# ---------- Page setup ----------
st.set_page_config(page_title="UniHelp — Streamlit Chatbot", page_icon="🎓", layout="centered")

# ---------- Paths (match your repo) ----------
MODEL_DIR = "models"
DATA_PATH = "data/intents_university.json"
LOG_PATH = "logs/chat_logs.csv"

# ---------- Small CSS for a clean look ----------
st.markdown("""
<style>
/* make chat a bit wider and cleaner */
section.main > div { max-width: 900px; margin: auto; }
.chat-bubble { border-radius: 14px; padding: 10px 14px; margin: 6px 0; }
.user { background: #0e1117; border: 1px solid #2b2b2b; }
.bot  { background: #161a23; border: 1px solid #2b2b2b; }
.small { color:#9aa0a6;font-size:0.85rem }
</style>
""", unsafe_allow_html=True)

# ---------- Cached artifacts ----------
@st.cache_resource
def load_artifacts():
    model = joblib.load(os.path.join(MODEL_DIR, "intent_model.joblib"))
    label_to_responses = joblib.load(os.path.join(MODEL_DIR, "label_to_responses.joblib"))
    vectorizer = model.named_steps["tfidf"]
    return model, vectorizer, label_to_responses

def predict_intent(model, text):
    probs = model.predict_proba([text])[0]
    labels = model.classes_
    best_idx = probs.argmax()
    return labels[best_idx], float(probs[best_idx])

def get_response(intent, label_to_responses):
    return random.choice(label_to_responses.get(intent, label_to_responses.get("fallback", ["I'm not sure."])))

def ensure_logs():
    os.makedirs("logs", exist_ok=True)
    if not os.path.exists(LOG_PATH):
        pd.DataFrame(columns=["timestamp","user","bot","intent","score","feedback"]).to_csv(LOG_PATH, index=False)

def log_interaction(user_text, bot_text, intent, score, feedback=None):
    ensure_logs()
    ts = datetime.datetime.now().isoformat(timespec="seconds")
    row = {"timestamp": ts, "user": user_text, "bot": bot_text, "intent": intent, "score": score, "feedback": feedback}
    pd.DataFrame([row]).to_csv(LOG_PATH, mode="a", header=False, index=False)

# ---------- Header ----------
st.title("🎓 UniHelp — Streamlit Chatbot")
st.caption("TF-IDF + Logistic Regression intent classifier with templated responses.")

# ---------- Sidebar ----------
with st.sidebar:
    st.header("Setup")
    retrain = st.button("Train / Retrain Model")
    threshold = st.slider("Confidence threshold", 0.0, 1.0, 0.55, 0.01)
    st.markdown("Upload a custom intents JSON to fine-tune:")
    uploaded = st.file_uploader("intents.json", type=["json"], accept_multiple_files=False)

# ---------- Training entry point ----------
if retrain:
    from train_evaluate import main as train_main
    if uploaded is not None:
        with open(DATA_PATH, "wb") as f:
            f.write(uploaded.getvalue())
        st.success("Uploaded custom intents.json saved. Training on new data.")
    train_main(DATA_PATH, MODEL_DIR, "reports")
    st.success("Training complete. Refresh the page if model does not update.")
    st.stop()

# ---------- Load model or ask to train ----------
try:
    model, vectorizer, label_to_responses = load_artifacts()
except Exception:
    st.error("Model artifacts not found. Click **'Train / Retrain Model'** in the sidebar to build the model.")
    st.stop()

# ---------- Patterns for retrieval fallback ----------
with open(DATA_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)
patterns, pattern_to_label = [], {}
for it in data["intents"]:
    tag = it["tag"]
    for p in it.get("patterns", []):
        patterns.append(p)
        pattern_to_label[p] = tag

def retrieval_fallback(vectorizer, patterns, text, pattern_to_label, min_sim=0.22):
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

# ---------- Chat history ----------
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hi! Ask me about admissions, tuition, scholarships, courses, or library hours."}
    ]

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        klass = "bot" if m["role"]=="assistant" else "user"
        st.markdown(f"<div class='chat-bubble {klass}'>{m['content']}</div>", unsafe_allow_html=True)

# ---------- Input ----------
user_input = st.chat_input("Type your message...")
if user_input:
    st.session_state.messages.append({"role":"user","content":user_input})
    with st.chat_message("user"):
        st.markdown(f"<div class='chat-bubble user'>{user_input}</div>", unsafe_allow_html=True)

    intent, score = predict_intent(model, user_input)
    if score < threshold:
        intent, score = retrieval_fallback(vectorizer, patterns, user_input, pattern_to_label)

    bot_text = get_response(intent, label_to_responses)

    with st.chat_message("assistant"):
        st.markdown(f"<div class='chat-bubble bot'>{bot_text}</div>", unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        if c1.button("👍 Helpful", key=f"up_{len(st.session_state.messages)}"):
            log_interaction(user_input, bot_text, intent, score, "up")
            st.toast("Thanks for the feedback!")
        if c2.button("👎 Not helpful", key=f"down_{len(st.session_state.messages)}"):
            log_interaction(user_input, bot_text, intent, score, "down")
            st.toast("Feedback recorded.")

    st.session_state.messages.append({"role":"assistant","content":bot_text})
    log_interaction(user_input, bot_text, intent, score)

# ---------- Evaluation panel ----------
st.divider()
st.subheader("📊 Evaluation (from last training)")
eval_path = os.path.join("reports", "eval.txt")
if os.path.exists(eval_path):
    with open(eval_path, "r", encoding="utf-8") as f:
        st.text(f.read())
else:
    st.info("No evaluation report yet. Use **Train / Retrain Model** in the sidebar.")
