
import json, os, random, argparse
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, precision_recall_fscore_support
from sklearn.pipeline import Pipeline
import joblib
import nltk

def load_intents(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def build_xy(intents: Dict) -> Tuple[List[str], List[str]]:
    X, y = [], []
    for intent in intents["intents"]:
        tag = intent["tag"]
        for p in intent.get("patterns", []):
            X.append(p)
            y.append(tag)
    return X, y

def main(data_path: str, model_dir: str, reports_dir: str, test_size: float=0.2, seed: int=42):
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(reports_dir, exist_ok=True)

    data = load_intents(data_path)
    X, y = build_xy(data)

    # keep a label -> responses map
    label_to_responses = {i["tag"]: i["responses"] for i in data["intents"] if i["responses"]}

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=seed, stratify=y)

    # simple TF-IDF + Logistic Regression classifier
    clf = Pipeline([
        ("tfidf", TfidfVectorizer(lowercase=True, stop_words="english", ngram_range=(1,2))),
        ("logreg", LogisticRegression(max_iter=1000, n_jobs=None))
    ])
    clf.fit(X_train, y_train)

    # predictions + report
    y_pred = clf.predict(X_test)
    report = classification_report(y_test, y_pred, digits=3)
    pr, rc, f1, _ = precision_recall_fscore_support(y_test, y_pred, average="weighted", zero_division=0)

    # simple BLEU using model responses vs a default reference (first response per tag)
    # this is illustrative because responses are templated.
    try:
        from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
        smoothie = SmoothingFunction().method1
        refs = [[label_to_responses[t][0].split()] for t in y_test]  # one reference per ground-truth intent
        hyps = [random.choice(label_to_responses.get(p, [""])) .split() for p in y_pred]
        bleu_scores = [sentence_bleu(refs[i], hyps[i], smoothing_function=smoothie) for i in range(len(y_test))]
        bleu = float(np.mean(bleu_scores))
    except Exception as e:
        bleu = float("nan")

    # save artifacts
    joblib.dump(clf, os.path.join(model_dir, "intent_model.joblib"))
    joblib.dump(label_to_responses, os.path.join(model_dir, "label_to_responses.joblib"))

    # write evaluation report
    with open(os.path.join(reports_dir, "eval.txt"), "w", encoding="utf-8") as f:
        f.write("Classification Report\n")
        f.write(report + "\n")
        f.write(f"Weighted Precision: {pr:.3f} Recall: {rc:.3f} F1: {f1:.3f}\n")
        f.write(f"Approx BLEU (templated): {bleu:.3f}\n")
    print("Training complete.")
    print(f"Saved model to: {model_dir}")
    print(f"Eval report: {os.path.join(reports_dir, 'eval.txt')}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/intents_university.json")
    parser.add_argument("--model_dir", default="models")
    parser.add_argument("--reports_dir", default="reports")
    args = parser.parse_args()
    main(args.data, args.model_dir, args.reports_dir)
