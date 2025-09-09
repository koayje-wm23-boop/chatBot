
import streamlit as st
import json, os, random, datetime
import joblib
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

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

def predict_intent(model, text):
    # logistic regression supports predict_proba
    probs = model.predict_proba([text])[0]
    labels = model.classes_
    best_idx = probs.argmax()
    return labels[best_idx], float(probs[best_idx])

def get_response(intent, label_to_responses):
    responses = label_to_responses.get(intent) or label_to_responses.get("fallback", ["I'm not sure."])
    return random.choice(responses)

def retrieval_fallback(vectorizer, patterns, text, label_to_responses, pattern_to_label):
    # cosine similarity to nearest known pattern; if close, use its intent response
    X = vectorizer.transform([text])
    pat_mat = vectorizer.transform(patterns)
    sims = cosine_similarity(X, pat_mat)[0]
    idx = sims.argmax()
    if sims[idx] > 0.2:
        intent = pattern_to_label[patterns[idx]]
        return intent, sims[idx]
    return "fallback", sims[idx]

def ensure_logs():
    os.makedirs("logs", exist_ok=True)
    if not os.path.exists(LOG_PATH):
        pd.DataFrame(columns=["timestamp","user","bot","intent","score","feedback"]).to_csv(LOG_PATH, index=False)

def log_interaction(user_text, bot_text, intent, score, feedback=None):
    ensure_logs()
    ts = datetime.datetime.now().isoformat()
    row = {"timestamp": ts, "user": user_text, "bot": bot_text, "intent": intent, "score": score, "feedback": feedback}
    df = pd.DataFrame([row])
    df.to_csv(LOG_PATH, mode="a", header=not os.path.exists(LOG_PATH), index=False)

st.title("🎓 UniHelp — Streamlit Chatbot")
st.caption("TF‑IDF + Logistic Regression intent classifier with templated responses.")

# Sidebar controls
with st.sidebar:
    st.header("Setup")
    retrain = st.button("Train / Retrain Model")
    threshold = st.slider("Confidence threshold", 0.0, 1.0, 0.55, 0.01)
    st.markdown("Upload a custom intents JSON to fine-tune:")
    uploaded = st.file_uploader("intents.json", type=["json"], accept_multiple_files=False)

# Option to replace dataset and retrain
if retrain:
    from train_evaluate import main as train_main  # local import
    if uploaded is not None:
        # Save uploaded JSON to data path
        bytes_data = uploaded.getvalue()
        with open(DATA_PATH, "wb") as f:
            f.write(bytes_data)
        st.success("Uploaded custom intents.json saved. Training on new data.")
    train_main(DATA_PATH, MODEL_DIR, "reports")
    st.success("Training complete. Refresh the page if model does not update.")
    st.stop()

# load model + responses
try:
    model, vectorizer, label_to_responses = load_artifacts()
except Exception as e:
    st.error("Model artifacts not found. Click 'Train / Retrain Model' in the sidebar to build the model.")
    st.stop()

# Load raw patterns for retrieval fallback
with open(DATA_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)
patterns, pattern_to_label = [], {}
for it in data["intents"]:
    tag = it["tag"]
    for p in it.get("patterns", []):
        patterns.append(p)
        pattern_to_label[p] = tag

# Chat UI
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

    # predict
    intent, score = predict_intent(model, user_input)
    if score < threshold:
        intent, score = retrieval_fallback(vectorizer, patterns, user_input, label_to_responses, pattern_to_label)

    bot_text = get_response(intent, label_to_responses)

    with st.chat_message("assistant"):
        st.markdown(bot_text)
        col1, col2 = st.columns(2)
        with col1:
            if st.button("👍 Helpful", key=f"up_{len(st.session_state.messages)}"):
                log_interaction(user_input, bot_text, intent, score, feedback="up")
                st.toast("Thanks for the feedback!")
        with col2:
            if st.button("👎 Not helpful", key=f"down_{len(st.session_state.messages)}"):
                log_interaction(user_input, bot_text, intent, score, feedback="down")
                st.toast("Feedback recorded.")

    st.session_state.messages.append({"role": "assistant", "content": bot_text})
    log_interaction(user_input, bot_text, intent, score)

# Metrics viewer
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
