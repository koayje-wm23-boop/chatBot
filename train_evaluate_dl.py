# train_evaluate_dl.py
import json, os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras_preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Dense, GlobalAveragePooling1D
from tensorflow.keras.utils import to_categorical
import joblib

from utils_text import clean_text

def main(data_path="data/intents_university.json", model_dir="models_dl", reports_dir="reports", random_state=42):
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(reports_dir, exist_ok=True)

    # --- Load data ---
    with open(data_path, "r", encoding="utf-8") as f:
        intents = json.load(f)["intents"]

    texts_raw, labels = [], []
    label_to_responses = {}
    for it in intents:
        tag = it["tag"]
        resps = it.get("responses", [])
        if resps:
            label_to_responses[tag] = resps
        for p in it.get("patterns", []):
            texts_raw.append(p)
            labels.append(tag)

    # --- Clean text (same as ML) ---
    texts = [clean_text(t) for t in texts_raw]

    # --- Encode labels ---
    le = LabelEncoder()
    y_enc = le.fit_transform(labels)
    num_classes = len(le.classes_)

    # --- Tokenize ---
    tokenizer = Tokenizer(oov_token="<OOV>")
    tokenizer.fit_on_texts(texts)
    seqs = tokenizer.texts_to_sequences(texts)

    # Cap maxlen to a reasonable value (handles short datasets)
    maxlen = max(5, min(25, max(len(s) for s in seqs)))  # 5..25
    X = pad_sequences(seqs, maxlen=maxlen, padding="post")
    y = to_categorical(y_enc, num_classes=num_classes)

    # --- Split ---
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state, stratify=y_enc
    )

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
    model.fit(X_train, y_train, epochs=30, batch_size=8, validation_data=(X_test, y_test), verbose=0)

    # --- Evaluate ---
    loss, acc = model.evaluate(X_test, y_test, verbose=0)
    with open(os.path.join(reports_dir, "eval_dl.txt"), "w", encoding="utf-8") as f:
        f.write(f"Deep Learning Model Accuracy: {acc:.3f}, Loss: {loss:.3f}\n")

    # --- Save artifacts ---
    model.save(os.path.join(model_dir, "dl_intent_model.h5"))
    joblib.dump(tokenizer, os.path.join(model_dir, "tokenizer.joblib"))
    joblib.dump(le, os.path.join(model_dir, "label_encoder.joblib"))
    joblib.dump(label_to_responses, os.path.join(model_dir, "label_to_responses.joblib"))
    joblib.dump({"maxlen": maxlen}, os.path.join(model_dir, "dl_meta.joblib"))

    print("✅ DL model trained. Reports saved to reports/eval_dl.txt")

if __name__ == "__main__":
    main()
