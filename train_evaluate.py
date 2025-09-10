import json
import os
import re
import numpy as np
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Dense, GlobalAveragePooling1D
from tensorflow.keras.utils import to_categorical
import joblib

# ---------------- Preprocessing setup ----------------
nltk.download("stopwords")
nltk.download("wordnet")

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

def preprocess_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z\s]", "", text)
    tokens = text.split()
    processed = []
    for w in tokens:
        if w not in stop_words:
            lemma = lemmatizer.lemmatize(w)
            processed.append(lemma)
            if lemma != w:   # keep plural too
                processed.append(w)
    # ✅ Sort & deduplicate
    return " ".join(sorted(set(processed)))

# ---------------- Main training ----------------
def main(data_path="data/intents_university.json", model_dir="models", reports_dir="reports"):
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(reports_dir, exist_ok=True)

    # --- Load data ---
    with open(data_path, "r", encoding="utf-8") as f:
        intents = json.load(f)["intents"]

    texts, labels = [], []
    label_to_responses = {}
    for intent in intents:
        tag = intent["tag"]
        responses = intent.get("responses", [])
        if responses:
            label_to_responses[tag] = responses
        for pattern in intent.get("patterns", []):
            cleaned = preprocess_text(pattern)  # <--- preprocess here
            texts.append(cleaned)
            labels.append(tag)

    # --- Encode labels ---
    le = LabelEncoder()
    labels_enc = le.fit_transform(labels)
    num_classes = len(le.classes_)

    # --- Tokenize text ---
    tokenizer = Tokenizer(oov_token="<OOV>")
    tokenizer.fit_on_texts(texts)
    seqs = tokenizer.texts_to_sequences(texts)
    maxlen = max(len(s) for s in seqs)
    X = pad_sequences(seqs, maxlen=maxlen, padding="post")
    y = to_categorical(labels_enc, num_classes=num_classes)

    # --- Train/test split ---
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # --- Build model ---
    vocab_size = len(tokenizer.word_index) + 1
    model = Sequential([
        Embedding(vocab_size, 64, input_length=maxlen),
        GlobalAveragePooling1D(),
        Dense(64, activation="relu"),
        Dense(num_classes, activation="softmax")
    ])
    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

    # --- Train ---
    history = model.fit(X_train, y_train, epochs=30, batch_size=8, validation_data=(X_test, y_test), verbose=0)

    # --- Evaluate ---
    loss, acc = model.evaluate(X_test, y_test, verbose=0)
    with open(os.path.join(reports_dir, "eval.txt"), "w") as f:
        f.write(f"Deep Learning Model Accuracy: {acc:.3f}, Loss: {loss:.3f}\n")

    # --- Save artifacts ---
    model.save(os.path.join(model_dir, "dl_intent_model.h5"))
    joblib.dump(tokenizer, os.path.join(model_dir, "tokenizer.joblib"))
    joblib.dump(le, os.path.join(model_dir, "label_encoder.joblib"))
    joblib.dump(label_to_responses, os.path.join(model_dir, "label_to_responses.joblib"))

    print("✅ Deep Learning model trained and saved!")

if __name__ == "__main__":
    main()
