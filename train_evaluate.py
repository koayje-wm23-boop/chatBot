# train_evaluate.py
import json, os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import joblib

from utils_text import clean_text

def main(data_path="data/intents_university.json", model_dir="models", reports_dir="reports", random_state=42):
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(reports_dir, exist_ok=True)

    # --- Load JSON dataset ---
    with open(data_path, "r", encoding="utf-8") as f:
        intents = json.load(f)["intents"]

    texts, labels = [], []
    label_to_responses = {}
    for it in intents:
        tag = it["tag"]
        resps = it.get("responses", [])
        if resps:
            label_to_responses[tag] = resps
        for p in it.get("patterns", []):
            texts.append(p)
            labels.append(tag)

    if not texts:
        raise ValueError("No training patterns found in the JSON.")

    # --- Encode labels ---
    le = LabelEncoder()
    y = le.fit_transform(labels)

    # --- Split (stratify to keep class balance) ---
    X_train, X_test, y_train, y_test = train_test_split(
        texts, y, test_size=0.2, random_state=random_state, stratify=y
    )

    # --- Build pipeline (cleaning inside TF-IDF) ---
    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(
            preprocessor=clean_text,      # <<— runs our cleaner
            ngram_range=(1,2),
            stop_words="english",         # basic stopword removal
            min_df=1,
            max_df=1.0
        )),
        ("clf", LogisticRegression(
            max_iter=2000,
            C=3.0,
            class_weight="balanced",
            random_state=random_state
        ))
    ])

    # --- Train ---
    pipe.fit(X_train, y_train)

    # --- Evaluate ---
    y_pred = pipe.predict(X_test)
    target_names = list(le.classes_)
    report = classification_report(y_test, y_pred, target_names=target_names, digits=3)
    with open(os.path.join(reports_dir, "eval.txt"), "w", encoding="utf-8") as f:
        f.write(report)

    # --- Save artifacts ---
    joblib.dump(pipe, os.path.join(model_dir, "intent_model.joblib"))
    joblib.dump(label_to_responses, os.path.join(model_dir, "label_to_responses.joblib"))

    print("✅ ML model trained. Reports saved to reports/eval.txt")

if __name__ == "__main__":
    main()
