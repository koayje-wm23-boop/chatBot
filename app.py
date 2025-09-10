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

# ---------------- Load JSON ----------------
def load_intents_json(path=DATA_PATH):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def norm(s: str) -> str:
    return " ".join(s.lower().strip().split())

def build_pattern_index(intents_json):
    exact_map = {}
    contains_list = []
    patterns = []
    p2l = {}
    for it in intents_json["intents"]:
        tag = it["tag"]
        responses = it.get("responses", [])
        if not responses:
            continue
        resp = responses[0]  # deterministic
        for p in it.get("patterns", []):
            np = norm(p)
            exact_map[np] = (tag, resp)
            contains_list.append((np, tag, resp))
            patterns.append(p)
            p2l[p] = tag
    return exact_map, contains_list, patterns, p2l

# ---------- Start-up ----------
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
    for it in intents_json["intents"]:
        if it["tag"] == intent and it.get("responses"):
            return it["responses"][0]
    return label_to_responses.get("fallback", ["I'm not sure."])[0]

# ---------------- Sidebar ----------------
def new_chat():
    st.session_state.chat_id = str(uuid.uuid4())[:8]
    st.session_state.messages = [
        {"role":"assistant","content":"Hi! Ask about TAR UMT admissions, programs, tuition, scholarships, library, housing, or contacts."}
    ]
if "chat_id" not in st.session_state:
    new_chat()

if "page" not in st.session_state:
    st.session_state.page = "chat"

with st.sidebar:
    st.header("UniHelp")
    if st.button("➕ New Chat"):
        new_chat()
        st.rerun()
    retrain = st.button("🔁 Retrain model")

    st.markdown("---")
    if st.button("💬 Chat"):
        st.session_state.page = "chat"
        st.rerun()
    if st.button("📊 Evaluation"):
        st.session_state.page = "evaluation"
        st.rerun()

    # --- Bottom section ---
    st.markdown("### ")
    st.markdown("### ")
    st.markdown("---")

    threshold = st.slider(
        "Confidence threshold (used if no pattern match)", 
        0.0, 1.0, 0.45, 0.01,
        help="Lower = more answers but less accurate. Higher = fewer answers but more accurate."
    )

    rating = st.select_slider(
        "⭐ Rate this chat", 
        options=[1,2,3,4,5], 
        value=3
    )
    if st.button("Save rating"):
        st.toast("Thanks for your rating!")

if retrain:
    train_now(DATA_PATH)
    intents_json = load_intents_json()
    exact_map, contains_list, patterns, pattern_to_label = build_pattern_index(intents_json)
    st.rerun()

# ---------------- Styling ----------------
st.markdown("""
<style>
section.main > div { max-width: 850px; margin: auto; }
.bubble { border-radius: 14px; padding: 10px 14px; margin: 6px 0; }
.user { background: #0e1117; border: 1px solid #2b2b2b; }
.bot  { background: #161a23; border: 1px solid #2b2b2b; }
</style>
""", unsafe_allow_html=True)

# ---------------- Page content ----------------
if st.session_state.page == "chat":
    # Header
    st.markdown("""
    <div style="text-align:center; padding: 10px;">
      <h1 style="color:#f5f5f5; font-size: 38px;">🎓 UniHelp</h1>
      <p style="color: #bbbbbb; font-size:18px;">TAR UMT Virtual Assistant — Ask me about admissions, tuition, scholarships, library, housing, or student life.</p>
    </div>
    """, unsafe_allow_html=True)

    # --- Simple Quick Buttons ---
    cols = st.columns(6)
    faq_map = {
        "Programs": "What programs are offered?",
        "Fees": "How much is the tuition fee?",
        "Scholarships": "What scholarships are available?",
        "Hostel": "How do I apply for housing?",
        "Library": "What are the library hours?",
        "Contact": "How do I contact TAR UMT?"
    }

    for i, (label, q) in enumerate(faq_map.items()):
        if cols[i].button(label):
            st.session_state.messages.append({"role": "user", "content": q})
            nuser = norm(q)

            if nuser in exact_map:
                tag, resp = exact_map[nuser]
                bot_text = resp
            else:
                found = None
                for npat, tag, resp in contains_list:
                    if npat and npat in nuser:
                        found = (tag, resp); break
                if found:
                    tag, resp = found
                    bot_text = resp
                else:
                    intent, score = predict_intent(q)
                    if score < threshold:
                        intent, score = retrieval_fallback(q)
                    bot_text = deterministic_response(intent)

            st.session_state.messages.append({"role": "assistant", "content": bot_text})
            st.rerun()

    # --- Chat Messages ---
    for m in st.session_state.get("messages", []):
        with st.chat_message(m["role"]):
            cls = "bot" if m["role"] == "assistant" else "user"
            st.markdown(f"<div class='bubble {cls}'>{m['content']}</div>", unsafe_allow_html=True)

    # --- User input ---
    user_text = st.chat_input("Type your message…")
    if user_text:
        st.session_state.messages.append({"role": "user", "content": user_text})
        with st.chat_message("user"):
            st.markdown(f"<div class='bubble user'>{user_text}</div>", unsafe_allow_html=True)

        nuser = norm(user_text)
        if nuser in exact_map:
            tag, resp = exact_map[nuser]
            bot_text = resp
        else:
            found = None
            for npat, tag, resp in contains_list:
                if npat and npat in nuser:
                    found = (tag, resp); break
            if found:
                tag, resp = found
                bot_text = resp
            else:
                intent, score = predict_intent(user_text)
                if score < threshold:
                    intent, score = retrieval_fallback(user_text)
                bot_text = deterministic_response(intent)
                log_row(st.session_state.chat_id, user_text, bot_text, "ml", intent=intent, score=score)

        with st.chat_message("assistant"):
            st.markdown(f"<div class='bubble bot'>{bot_text}</div>", unsafe_allow_html=True)

        st.session_state.messages.append({"role": "assistant", "content": bot_text})

elif st.session_state.page == "evaluation":
    st.title("📊 Model Evaluation")

    eval_path = os.path.join(REPORTS_DIR, "eval.txt")
    if os.path.exists(eval_path):
        with open(eval_path, "r", encoding="utf-8") as f:
            lines = f.readlines()

        data = []
        for line in lines:
            parts = line.strip().split()
            if not parts or parts[0].startswith("="):
                continue
            if len(parts) >= 4:
                try:
                    intent = parts[0]
                    prec, rec, f1 = float(parts[1]), float(parts[2]), float(parts[3])
                    support = int(parts[4]) if len(parts) > 4 and parts[4].isdigit() else "-"
                    data.append([intent, prec, rec, f1, support])
                except Exception:
                    continue

        if data:
            df = pd.DataFrame(data, columns=["Intent", "Precision", "Recall", "F1-score", "Support"])
            st.dataframe(df, use_container_width=True)

            weighted = df[df["Intent"]=="weighted_avg"]
            if not weighted.empty:
                p, r, f1 = weighted.iloc[0]["Precision"], weighted.iloc[0]["Recall"], weighted.iloc[0]["F1-score"]
                st.markdown(f"""
                <div style="padding:15px; background:#161a23; border-radius:12px; margin-top:15px;">
                  <h3 style="margin:0; color:#f5f5f5;">📊 Overall Performance</h3>
                  <p style="margin:0; color:#bbb;">Weighted Precision: <b>{p:.2f}</b> • Recall: <b>{r:.2f}</b> • F1 Score: <b>{f1:.2f}</b></p>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.warning("Could not parse evaluation file. Try retraining the model first.")
    else:
        st.info("No evaluation report yet — click Retrain model first.")
