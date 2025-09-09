# train_evaluate.py
import json, os, argparse, random
from typing import Dict, List, Tuple
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, precision_recall_fscore_support
import joblib

# BLEU & ROUGE
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer

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

def main(data_path="data/intents_university.json", model_dir="models", reports_dir="reports", test_size=0.25, seed=42):
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(reports_dir, exist_ok=True)

    data = load_intents(data_path)
    X, y = build_xy(data)
    label_to_responses = {i["tag"]: i["responses"] for i in data["intents"] if i.get("responses")}

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y
    )

    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(lowercase=True, stop_words="english", ngram_range=(1,2))),
        ("clf", LogisticRegression(max_iter=1000))
    ])
    pipe.fit(X_train, y_train)

    y_pred = pipe.predict(X_test)

    # Intent classification metrics
    report = classification_report(y_test, y_pred, digits=3)
    pr, rc, f1, _ = precision_recall_fscore_support(y_test, y_pred, average="weighted", zero_division=0)

    # Response generation metrics (templated baseline)
    # reference = first response of the TRUE intent, hypothesis = first response of PRED intent
    smoothie = SmoothingFunction().method1
    rouge = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)

    bleu_scores, rougeL_scores = [], []
    for yt, yp in zip(y_test, y_pred):
        ref = (label_to_responses.get(yt, [""]))[0].split()
        hyp = (label_to_responses.get(yp, [""]))[0].split()
        bleu_scores.append(sentence_bleu([ref], hyp, smoothing_function=smoothie))
        rougeL_scores.append(rouge.score(" ".join(ref), " ".join(hyp))["rougeL"].fmeasure)

    bleu = float(np.mean(bleu_scores)) if bleu_scores else 0.0
    rougeL = float(np.mean(rougeL_scores)) if rougeL_scores else 0.0

    # Save artifacts
    joblib.dump(pipe, os.path.join(model_dir, "intent_model.joblib"))
    joblib.dump(label_to_responses, os.path.join(model_dir, "label_to_responses.joblib"))

    with open(os.path.join(reports_dir, "eval.txt"), "w", encoding="utf-8") as f:
        f.write("=== Intent Classification ===\n")
        f.write(report + "\n")
        f.write(f"Weighted Precision: {pr:.3f} | Recall: {rc:.3f} | F1: {f1:.3f}\n\n")
        f.write("=== Response Generation (templated baseline) ===\n")
        f.write(f"BLEU: {bleu:.3f} | ROUGE-L: {rougeL:.3f}\n")

    print("Training complete.")
    print(f"Saved model to: {model_dir}")
    print(f"Eval report written to: {os.path.join(reports_dir, 'eval.txt')}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/intents_university.json")
    parser.add_argument("--model_dir", default="models")
    parser.add_argument("--reports_dir", default="reports")
    args = parser.parse_args()
    main(args.data, args.model_dir, args.reports_dir)
