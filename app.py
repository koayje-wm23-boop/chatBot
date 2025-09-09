import os, json, random, datetime
import streamlit as st
import pandas as pd
import joblib
from sklearn.metrics.pairwise import cosine_similarity
from train_evaluate import main as train_main

st.set_page_config(page_title="UniHelp Chatbot", page_icon="🎓", layout="centered")

MODEL_DIR = "models"
DATA_PATH = "data/intents_university.json"
LOG_PATH = "logs/chat_logs.csv"

@st.cache_resource
def load_artifacts():
    model = joblib.load(os.path.join(MODEL_DIR, "intent_model.joblib"))
    label_to_responses = joblib.load(os.path.join(MODEL_DIR, "label_to_responses.joblib"))
    vectorizer = model.named_steps["tfidf"]
    return model, vectorizer, label_to_responses

def predict_intent(model, text: str):
    probs = model.predict_proba([text])[0]
    labels = model.classes_
    idx = probs.argmax()
    return labels[idx], float(probs[idx])

def get_response(intent, label_to_responses):
    responses = label_to_responses.get(intent) or label_to_responses.get("fallback", ["I'm not sure."])
    return random.choice(responses)

def retrieval_fallback(vectorizer, patterns, text, pattern_to_label):
    X = vectorizer.transform([text])
    pat_mat = vectorizer.transform(patterns)
    sims = cosine_similarity(X, pat_mat)[0]
    idx = sims.argmax()
    if sims[idx] > 0.2:
        return pattern_to_label[patterns[idx]], float(sims[idx])
    return "fallback", float(sims[idx])

def ensure_logs():
    os.makedirs("logs", exist_ok=True)
    if not os.path.exists(LOG_PATH):
        pd.DataFrame(columns=["timestamp","user","bot","intent","score","feedback"]).to_csv(LOG_PATH, index=False)

def log_interaction(user_text, bot_text, intent, score, feedback=None):
    ensure_logs()
    ts = datetime.datetime.now().isoformat()
    row = {"timestamp": ts, "user": user_text, "bot": bot_text, "intent": intent, "score": score, "feedback": feedback}
    df = pd.DataFrame([row])
    header = not os.path.exists(LOG_PATH) or os.path.getsize(LOG_PATH) == 0
    df.to_csv(LOG_PATH, mode="a", header=header, index=False)

st.title("🎓 UniHelp — Streamlit Chatbot")
st.caption("TF‑IDF + Logistic Regression intent classifier with templated responses.")

with st.sidebar:
    st.header("Setup")
    retrain = st.button("Train / Retrain Model")
    threshold = st.slider("Confidence threshold", 0.0, 1.0, 0.55, 0.01)
    st.markdown("Upload a custom intents JSON to fine-tune:")
    uploaded = st.file_uploader("intents.json", type=["json"], accept_multiple_files=False)

if retrain:
    # If user uploaded a dataset, save it first
    if uploaded is not None:
        os.makedirs("data", exist_ok=True)
        with open(DATA_PATH, "wb") as f:
            f.write(uploaded.getvalue())
        st.success("Uploaded custom intents.json saved. Training on new data.")
    with st.spinner("Training model..."):
        train_main(DATA_PATH, MODEL_DIR, "reports")
    st.success("Training complete.")

# Try to load model after potential training
try:
    model, vectorizer, label_to_responses = load_artifacts()
except Exception:
    st.error("Model artifacts not found. Click 'Train / Retrain Model' in the sidebar to build the model.")
    st.stop()

# Build retrieval pattern list for fallback matching
with open(DATA_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)
patterns, pattern_to_label = [], {}
for it in data["intents"]:
    tag = it["tag"]
    for p in it.get("patterns", []):
        patterns.append(p)
        pattern_to_label[p] = tag

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hi! I'm UniHelp bot. Ask me about admissions, fees, scholarships, courses, or campus info."}
    ]

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

user_input = st.chat_input("Type your message...")
if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Predict
    intent, score = predict_intent(model, user_input)
    if score < threshold:
        intent, score = retrieval_fallback(vectorizer, patterns, user_input, pattern_to_label)

    bot_text = get_response(intent, label_to_responses)

    with st.chat_message("assistant"):
        st.markdown(bot_text)
        cols = st.columns(2)
        with cols[0]:
            if st.button("👍", key=f"up_{len(st.session_state.messages)}"):
                log_interaction(user_input, bot_text, intent, score, feedback="up")
                st.toast("Thanks for the feedback!")
        with cols[1]:
            if st.button("👎", key=f"down_{len(st.session_state.messages)}"):
                log_interaction(user_input, bot_text, intent, score, feedback="down")
                st.toast("Feedback recorded.")

    st.session_state.messages.append({"role": "assistant", "content": bot_text})
    log_interaction(user_input, bot_text, intent, score)

st.divider()
st.subheader("📊 Evaluation (from last training)")
eval_path = os.path.join("reports", "eval.txt")
if os.path.exists(eval_path):
    with open(eval_path, "r", encoding="utf-8") as f:
        st.text(f.read())
else:
    st.info("No evaluation report yet. Use **Train / Retrain Model** in the sidebar.")

st.divider()
st.subheader("🧪 Export chat logs")
ensure_logs()
if os.path.exists(LOG_PATH):
    with open(LOG_PATH, "rb") as f:
        st.download_button(label="Download chat_logs.csv", data=f, file_name="chat_logs.csv", mime="text/csv")
