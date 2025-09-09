# app.py — Final with auto-train, chat history, ratings, debug, stronger fallback
import streamlit as st
import os, json, random, re, datetime, uuid
import joblib
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# --------------------- App Setup ---------------------
st.set_page_config(page_title="🎓 UniHelp — University Chatbot", page_icon="🎓", layout="centered")

MODEL_DIR = "models"
DATA_PATH = "data/intents_university.json"
REPORTS_DIR = "reports"
LOG_DIR = "logs"
CHAT_DIR = os.path.join(LOG_DIR, "chats")
LOG_PATH = os.path.join(LOG_DIR, "chat_logs.csv")

os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(CHAT_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)

# --------------------- Utilities ---------------------
def ensure_chat_csv():
    if not os.path.exists(LOG_PATH):
        pd.DataFrame(columns=["timestamp","chat_id","user","bot","intent","score","feedback"]).to_csv(LOG_PATH, index=False)

def log_row(chat_id, user, bot, intent, score, feedback=None):
    ensure_chat_csv()
    ts = datetime.datetime.now().isoformat(timespec="seconds")
    pd.DataFrame([{
        "timestamp": ts, "chat_id": chat_id, "user": user,
        "bot": bot, "intent": intent, "score": score, "feedback": feedback
    }]).to_csv(LOG_PATH, mode="a", header=False, index=False)

def save_chat_to_disk(session):
    payload = {
        "id": session["id"],
        "title": session["title"],
        "created_at": session["created_at"],
        "updated_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "rating": session.get("rating"),
        "messages": session["messages"],
    }
    path = os.path.join(CHAT_DIR, f"{session['id']}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

def list_saved_chats():
    chats = []
    for fname in sorted(os.listdir(CHAT_DIR)):
        if fname.endswith(".json"):
            try:
                with open(os.path.join(CHAT_DIR, fname), "r", encoding="utf-8") as f:
                    data = json.load(f)
                chats.append({"id": data["id"], "title": data.get("title","(untitled)"), "updated_at": data.get("updated_at","")})
            except Exception:
                pass
    chats.sort(key=lambda x: x.get("updated_at",""), reverse=True)
    return chats

def load_chat_by_id(cid):
    path = os.path.join(CHAT_DIR, f"{cid}.json")
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return {
            "id": data["id"],
            "title": data.get("title","Chat"),
            "created_at": data.get("created_at", datetime.datetime.now().isoformat(timespec="seconds")),
            "rating": data.get("rating"),
            "messages": data.get("messages", [])
        }
    return None

# --------------------- Training Helpers ---------------------
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

# Auto-train if missing
if not artifacts_exist():
    train_now(DATA_PATH)

# Load artifacts
model, vectorizer, label_to_responses = load_artifacts()

# Load intents for retrieval patterns
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

# **Lowered similarity threshold** so single-word like "scholarship" maps correctly
def retrieval_fallback(text, min_sim=0.12):
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

def get_response(intent):
    return random.choice(label_to_responses.get(intent, label_to_responses.get("fallback", ["I'm not sure."])))

# --------------------- Sidebar (New Chat, History, Rating) ---------------------
def new_chat(title=None):
    cid = str(uuid.uuid4())[:8]
    title = title or "New Chat"
    sess = {
        "id": cid,
        "title": title,
        "created_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "rating": None,
        "messages": [{"role":"assistant","content":"Hi! Ask me about admissions, programs, tuition, scholarships, library, or campus services."}]
    }
    st.session_state.current_chat = sess
    save_chat_to_disk(sess)

if "current_chat" not in st.session_state:
    new_chat(title="Welcome")

with st.sidebar:
    st.header("Chat")
    cols = st.columns([1,1.2])
    with cols[0]:
        if st.button("➕ New Chat"):
            new_chat()
    with cols[1]:
        new_title = st.text_input("Title", value=st.session_state.current_chat["title"])
        if new_title != st.session_state.current_chat["title"]:
            st.session_state.current_chat["title"] = new_title
            save_chat_to_disk(st.session_state.current_chat)

    st.markdown("**History**")
    saved = list_saved_chats()
    labels = [f"{c['title']}  ·  {c['id']}" for c in saved] if saved else []
    selected = st.selectbox("Open chat", options=["(current)"] + labels, index=0)
    if selected != "(current)":
        picked_id = selected.split()[-1]
        loaded = load_chat_by_id(picked_id)
        if loaded:
            st.session_state.current_chat = loaded

    st.divider()
    st.header("Settings")
    # Default lower threshold to avoid unnecessary fallback
    threshold = st.slider("Confidence threshold", 0.0, 1.0, 0.45, 0.01)

    st.divider()
    st.header("Conversation Rating")
    rating = st.select_slider("How helpful was this chat?", options=[1,2,3,4,5],
                              value=st.session_state.current_chat.get("rating") or 3)
    if st.button("Save Rating"):
        st.session_state.current_chat["rating"] = int(rating)
        save_chat_to_disk(st.session_state.current_chat)
        st.success("Rating saved.")

# --------------------- Main Chat UI ---------------------
# Style
st.markdown("""
<style>
section.main > div { max-width: 900px; margin: auto; }
.chat { border-radius: 14px; padding: 10px 14px; margin: 6px 0; }
.user { background: #0e1117; border: 1px solid #2b2b2b; }
.bot  { background: #161a23; border: 1px solid #2b2b2b; }
</style>
""", unsafe_allow_html=True)

st.title("🎓 UniHelp — University Chatbot")
st.caption("Intent classification + templated responses with retrieval fallback and evaluation.")

# Render existing messages
for m in st.session_state.current_chat["messages"]:
    with st.chat_message(m["role"]):
        cls = "bot" if m["role"] == "assistant" else "user"
        st.markdown(f"<div class='chat {cls}'>{m['content']}</div>", unsafe_allow_html=True)

# Input
user_text = st.chat_input("Type your message…")
if user_text:
    st.session_state.current_chat["messages"].append({"role":"user","content":user_text})
    with st.chat_message("user"):
        st.markdown(f"<div class='chat user'>{user_text}</div>", unsafe_allow_html=True)

    # Predict intent
    intent, score = predict_intent(user_text)
    if score < threshold:
        intent, score = retrieval_fallback(user_text)

    bot_text = get_response(intent)

    with st.chat_message("assistant"):
        st.markdown(f"<div class='chat bot'>{bot_text}</div>", unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        if c1.button("👍 Helpful", key=f"up_{st.session_state.current_chat['id']}_{len(st.session_state.current_chat['messages'])}"):
            log_row(st.session_state.current_chat["id"], user_text, bot_text, intent, score, feedback="up")
            st.toast("Thanks for the feedback!")
        if c2.button("👎 Not helpful", key=f"down_{st.session_state.current_chat['id']}_{len(st.session_state.current_chat['messages'])}"):
            log_row(st.session_state.current_chat["id"], user_text, bot_text, intent, score, feedback="down")
            st.toast("Feedback recorded.")

    st.session_state.current_chat["messages"].append({"role":"assistant","content":bot_text})
    log_row(st.session_state.current_chat["id"], user_text, bot_text, intent, score)
    save_chat_to_disk(st.session_state.current_chat)

# --------------------- Debug + Evaluation ---------------------
with st.expander("🔍 Debug (intent)"):
    st.write("Last predicted intent and confidence are shown after you send a message.")
    if st.session_state.current_chat["messages"] and st.session_state.current_chat["messages"][-1]["role"] == "assistant":
        # Peek at last log row for score/intent if needed (best-effort)
        try:
            df = pd.read_csv(LOG_PATH)
            if len(df):
                row = df.iloc[-1]
                st.write(f"Predicted: **{row['intent']}**  • confidence: **{row['score']:.2f}**")
        except Exception:
            pass

st.divider()
st.subheader("📊 Evaluation (from last training)")
eval_path = os.path.join(REPORTS_DIR, "eval.txt")
if os.path.exists(eval_path):
    with open(eval_path, "r", encoding="utf-8") as f:
        st.text(f.read())
else:
    st.info("No evaluation report yet. Retrain locally (python train_evaluate.py) or delete 'models/' to trigger auto-training.")
