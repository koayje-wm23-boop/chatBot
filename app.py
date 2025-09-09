import streamlit as st
import os, json, datetime, uuid
import joblib
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# ---------------- Basic setup ----------------
st.set_page_config(page_title="🎓 UniHelp", page_icon="🎓", layout="centered")

MODEL_DIR   = "models"
DATA_PATH   = "data/intents_university.json"
REPORTS_DIR = "reports"
LOG_DIR     = "logs"
CHAT_CSV    = os.path.join(LOG_DIR, "chat_logs.csv")

os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)

# ---------------- Logging ----------------
def ensure_chat_csv():
    if not os.path.exists(CHAT_CSV):
        pd.DataFrame(columns=["timestamp","chat_id","user","bot","route","intent","score"]).to_csv(CHAT_CSV, index=False)

def log_row(chat_id, user, bot, route, intent=None, score=None):
    ensure_chat_csv()
    ts = datetime.datetime.now().isoformat(timespec="seconds")
    pd.DataFrame([{
        "timestamp": ts, "chat_id": chat_id, "user": user,
        "bot": bot, "route": route, "intent": intent, "score": score
    }]).to_csv(CHAT_CSV, mode="a", header=False, index=False)

# ---------------- Load & train ----------------
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

# ---------------- Load JSON for pattern routing ----------------
def load_intents_json(path=DATA_PATH):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def norm(s: str) -> str:
    return " ".join(s.lower().strip().split())

def build_pattern_index(intents_json):
    """
    Returns:
      - exact_map: {normalized_pattern: (tag, response_text)}
      - contains_list: [(normalized_pattern, tag, response_text)] for substring matching
      - patterns, pattern_to_label for retrieval fallback
    """
    exact_map = {}
    contains_list = []
    patterns = []
    p2l = {}

    for it in intents_json["intents"]:
        tag = it["tag"]
        responses = it.get("responses", [])
        if not responses:
            continue
        resp = responses[0]  # deterministic: ALWAYS first response
        for p in it.get("patterns", []):
            np = norm(p)
            exact_map[np] = (tag, resp)
            contains_list.append((np, tag, resp))
            patterns.append(p)
            p2l[p] = tag
    return exact_map, contains_list, patterns, p2l

# ---------- Start-up: ensure model exists, then load everything ----------
if not artifacts_exist():
    train_now(DATA_PATH)

model, vectorizer, label_to_responses = load_artifacts()
intents_json = load_intents_json()
exact_map, contains_list, patterns, pattern_to_label = build_pattern_index(intents_json)

# ---------------- Inference utils ----------------
def predict_intent(text):
    probs = model.predict_proba([text])[0]
    labels = model.classes_
    j = probs.argmax()
    return labels[j], float(probs[j])

def retrieval_fallback(text, min_sim=0.12):
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

def deterministic_response(intent: str) -> str:
    # Always return the FIRST response from JSON for determinism.
    for it in intents_json["intents"]:
        if it["tag"] == intent and it.get("responses"):
            return it["responses"][0]
    # fallback if somehow missing
    return label_to_responses.get("fallback", ["I'm not sure."])[0]

# ---------------- Sidebar (simple) ----------------
def new_chat():
    st.session_state.chat_id = str(uuid.uuid4())[:8]
    st.session_state.messages = [
        {"role":"assistant","content":"Hi! Ask about admissions, programs, tuition, scholarships, library, housing, or contacts."}
    ]
if "chat_id" not in st.session_state:
    new_chat()

with st.sidebar:
    st.header("UniHelp")
    if st.button("➕ New Chat"):
        new_chat()
        st.experimental_rerun()
    retrain = st.button("🔁 Retrain model")
    threshold = st.slider("Confidence threshold (used only if no pattern match)", 0.0, 1.0, 0.45, 0.01)

if retrain:
    train_now(DATA_PATH)
    # Rebuild pattern index after retrain (in case JSON changed)
    intents_json = load_intents_json()
    exact_map, contains_list, patterns, pattern_to_label = build_pattern_index(intents_json)
    st.experimental_rerun()

# ---------------- Styling ----------------
st.markdown("""
<style>
section.main > div { max-width: 850px; margin: auto; }
.bubble { border-radius: 14px; padding: 10px 14px; margin: 6px 0; }
.user { background: #0e1117; border: 1px solid #2b2b2b; }
.bot  { background: #161a23; border: 1px solid #2b2b2b; }
</style>
""", unsafe_allow_html=True)

st.title("🎓 UniHelp")

# ---------------- Render history ----------------
for m in st.session_state.get("messages", []):
    with st.chat_message(m["role"]):
        cls = "bot" if m["role"]=="assistant" else "user"
        st.markdown(f"<div class='bubble {cls}'>{m['content']}</div>", unsafe_allow_html=True)

# ---------------- Chat input ----------------
user_text = st.chat_input("Type your message…")
if user_text:
    st.session_state.messages.append({"role":"user","content":user_text})
    with st.chat_message("user"):
        st.markdown(f"<div class='bubble user'>{user_text}</div>", unsafe_allow_html=True)

    # 1) PATTERN-FIRST: exact match (case-insensitive, trimmed)
    nuser = norm(user_text)
    if nuser in exact_map:
        tag, resp = exact_map[nuser]
        route = "pattern_exact"
        bot_text = resp

    # 2) PATTERN-CONTAINS: relaxed substring match (safe, deterministic)
    else:
        found = None
        for npat, tag, resp in contains_list:
            if npat and npat in nuser:
                found = (tag, resp); break
        if found:
            tag, resp = found
            route = "pattern_contains"
            bot_text = resp
        else:
            # 3) ML classifier (only if no pattern matched)
            intent, score = predict_intent(user_text)
            if score < threshold:
                intent, score = retrieval_fallback(user_text)
            bot_text = deterministic_response(intent)
            route = "ml" if intent != "fallback" else "fallback"
            # log ML route with score/intent
            log_row(st.session_state.chat_id, user_text, bot_text, route, intent=intent, score=score)

    with st.chat_message("assistant"):
        st.markdown(f"<div class='bubble bot'>{bot_text}</div>", unsafe_allow_html=True)

    # Log deterministic routes too
    if 'route' in locals() and (route.startswith("pattern")):
        log_row(st.session_state.chat_id, user_text, bot_text, route, intent=tag, score=1.0)

    st.session_state.messages.append({"role":"assistant","content":bot_text})

# ---------------- Evaluation (optional) ----------------
st.divider()
st.subheader("📊 Evaluation")
eval_path = os.path.join(REPORTS_DIR, "eval.txt")
if os.path.exists(eval_path):
    with open(eval_path, "r", encoding="utf-8") as f:
        st.text(f.read())
else:
    st.info("No evaluation report yet — click Retrain to generate it.")
