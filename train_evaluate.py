import os, json, argparse, random
from typing import Dict, List
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, precision_recall_fscore_support
import joblib

def load_intents(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def build_dataset(intents: Dict):
    X, y = [], []
    label_to_responses = {}
    for it in intents["intents"]:
        tag = it["tag"]
        label_to_responses[tag] = it.get("responses", [])
        for p in it.get("patterns", []):
            X.append(p)
            y.append(tag)
    return X, y, label_to_responses

def main(data_path: str, model_dir: str = "models", reports_dir: str = "reports"):
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(reports_dir, exist_ok=True)

    intents = load_intents(data_path)
    X, y, label_to_responses = build_dataset(intents)

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y if len(set(y))>1 else None)

    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1,2), min_df=1)),
        ("clf", LogisticRegression(max_iter=1000, n_jobs=None))
    ])
    pipe.fit(X_tr, y_tr)
    y_pred = pipe.predict(X_te)

    report = classification_report(y_te, y_pred, zero_division=0)
    pr, rc, f1, _ = precision_recall_fscore_support(y_te, y_pred, average="weighted", zero_division=0)

    joblib.dump(pipe, os.path.join(model_dir, "intent_model.joblib"))
    joblib.dump(label_to_responses, os.path.join(model_dir, "label_to_responses.joblib"))

    with open(os.path.join(reports_dir, "eval.txt"), "w", encoding="utf-8") as f:
        f.write(report + "\n")
        f.write(f"Weighted Precision: {pr:.3f} Recall: {rc:.3f} F1: {f1:.3f}\n")

    print("Training complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/intents_university.json")
    parser.add_argument("--model_dir", default="models")
    parser.add_argument("--reports_dir", default="reports")
    args = parser.parse_args()
    main(args.data, args.model_dir, args.reports_dir)
