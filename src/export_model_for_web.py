"""
Export trained Scikit-learn Pipeline (TfidfVectorizer + MultinomialNB)
to a compact, standalone JSON for instant browser-side and serverless inference.
"""

import json
import os
import joblib
import numpy as np

def export_pipeline():
    model_path = os.path.join("models", "spam_classifier.joblib")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}")

    model = joblib.load(model_path)
    tfidf = model.named_steps["tfidf"]
    clf = model.named_steps["clf"]

    feature_names = tfidf.get_feature_names_out()
    idf_diag = tfidf.idf_
    ham_log_prob = clf.feature_log_prob_[0]
    spam_log_prob = clf.feature_log_prob_[1]
    log_prob_diff = spam_log_prob - ham_log_prob

    # Load metrics if available
    metrics_path = os.path.join("outputs", "metrics.json")
    metrics_data = {}
    if os.path.exists(metrics_path):
        with open(metrics_path, "r", encoding="utf-8") as f:
            metrics_data = json.load(f)

    # Build dictionary for vocabulary features:
    # { term: [idf_rounded, log_prob_diff_rounded, ham_log_prob_rounded, spam_log_prob_rounded] }
    vocab_export = {}
    for idx, term in enumerate(feature_names):
        vocab_export[term] = [
            round(float(idf_diag[idx]), 5),
            round(float(log_prob_diff[idx]), 5),
            round(float(ham_log_prob[idx]), 5),
            round(float(spam_log_prob[idx]), 5)
        ]

    export_obj = {
        "model_name": "Spam SMS/Email Classifier (TF-IDF + MultinomialNB)",
        "version": "1.0.0",
        "classes": ["Ham", "Spam"],
        "class_log_prior": [round(float(clf.class_log_prior_[0]), 6), round(float(clf.class_log_prior_[1]), 6)],
        "ngram_range": [1, 2],
        "norm": "l2",
        "token_pattern": r"\b\w+\b|[!?$£]",
        "vocab_size": len(feature_names),
        "vocabulary": vocab_export,
        "metrics": metrics_data.get("primary_model", {}).get("metrics", {}),
        "confusion_matrix": metrics_data.get("primary_model", {}).get("confusion_matrix", {}),
        "test_distribution": metrics_data.get("split_summary", {}).get("test_class_distribution", {})
    }

    os.makedirs("public", exist_ok=True)
    out_path = os.path.join("public", "model_export.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(export_obj, f, separators=(',', ':'))

    print(f"Exported model to {out_path} ({os.path.getsize(out_path) / 1024:.1f} KB)")

if __name__ == "__main__":
    export_pipeline()
