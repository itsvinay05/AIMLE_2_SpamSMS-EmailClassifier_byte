"""
Spam SMS / Email Classifier - Training and Evaluation Pipeline
--------------------------------------------------------------
Author: Advanced AI Assistant & Pair Programmer
Dataset: UCI SMS Spam Collection
Pipeline: TF-IDF N-gram Vectorizer + Multinomial Naive Bayes (with Model Comparisons)
"""

import json
import os
import sys
import platform
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import sklearn

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB, ComplementNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score
)

# Set random seed for reproducibility
RANDOM_STATE = 42
DATA_PATH = os.path.join("data", "SMSSpamCollection")
MODELS_DIR = "models"
OUTPUTS_DIR = "outputs"


def load_and_preprocess_data(data_path: str):
    """
    Loads raw SMS Spam Collection data, maps labels, inspects duplicates,
    and returns cleaned DataFrame and audit metadata.
    """
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset file not found at: {data_path}")

    # Load tab-separated file: columns are label and message text
    df = pd.read_csv(
        data_path,
        sep="\t",
        names=["raw_label", "text"],
        quoting=3,
        encoding="utf-8"
    )

    raw_total_rows = len(df)
    missing_counts = df.isnull().sum().to_dict()

    # Drop any nulls (if present)
    df = df.dropna(subset=["raw_label", "text"]).copy()

    # Label mapping: spam -> 1, ham -> 0
    df["label"] = df["raw_label"].str.strip().str.lower().map({"spam": 1, "ham": 0})
    if df["label"].isnull().any():
        raise ValueError("Encountered unexpected labels outside 'spam' and 'ham'")

    # Duplicate analysis
    raw_duplicates = int(df.duplicated(subset=["text"]).sum())
    duplicate_spam = int(df[df.duplicated(subset=["text"])]["label"].sum())
    duplicate_ham = raw_duplicates - duplicate_spam

    # Documented decision: Drop exact text duplicates to prevent train-test data leakage
    # Identical promotional spam broadcasts and generic phrases ("Ok", "Sorry, I'll call later")
    # if split between train and test would artificially inflate test metrics.
    df_clean = df.drop_duplicates(subset=["text"]).copy()
    cleaned_total_rows = len(df_clean)

    audit_summary = {
        "raw_total_rows": raw_total_rows,
        "clean_total_rows": cleaned_total_rows,
        "missing_values": missing_counts,
        "duplicates_removed": raw_duplicates,
        "duplicates_breakdown": {
            "spam_duplicates": duplicate_spam,
            "ham_duplicates": duplicate_ham
        },
        "raw_distribution": df["raw_label"].value_counts().to_dict(),
        "clean_distribution": {
            "ham": int((df_clean["label"] == 0).sum()),
            "spam": int((df_clean["label"] == 1).sum())
        }
    }

    return df_clean, audit_summary


def evaluate_model(name: str, pipeline, X_test, y_test):
    """
    Evaluates a fitted pipeline on held-out test data.
    """
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]

    acc = float(accuracy_score(y_test, y_pred))
    prec = float(precision_score(y_test, y_pred, pos_label=1, zero_division=0))
    rec = float(recall_score(y_test, y_pred, pos_label=1, zero_division=0))
    f1 = float(f1_score(y_test, y_pred, pos_label=1, zero_division=0))
    cm = confusion_matrix(y_test, y_pred).tolist()
    roc_auc = float(auc(*roc_curve(y_test, y_proba)[:2]))

    return {
        "model_name": name,
        "accuracy": acc,
        "precision_spam": prec,
        "recall_spam": rec,
        "f1_spam": f1,
        "roc_auc": roc_auc,
        "confusion_matrix": cm,
        "y_pred": y_pred,
        "y_proba": y_proba
    }


def main():
    print("=" * 60)
    print(" Spam SMS / Email Classifier - Training & Benchmark Run")
    print("=" * 60)

    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(OUTPUTS_DIR, exist_ok=True)

    # 1. Load & Preprocess
    print("[1/6] Loading and auditing dataset...")
    df, audit_info = load_and_preprocess_data(DATA_PATH)
    print(f"      Total unique messages: {len(df)}")
    print(f"      Clean Class distribution: Ham = {audit_info['clean_distribution']['ham']} "
          f"({audit_info['clean_distribution']['ham']/len(df):.1%}), "
          f"Spam = {audit_info['clean_distribution']['spam']} "
          f"({audit_info['clean_distribution']['spam']/len(df):.1%})")
    print(f"      Duplicates removed to prevent data leakage: {audit_info['duplicates_removed']}")

    # 2. Stratified Train/Test Split (80/20)
    print("\n[2/6] Performing stratified 80/20 train/test split (random_state=42)...")
    X_raw = np.array(df["text"].tolist(), dtype=object)
    y_raw = np.array(df["label"].tolist(), dtype=int)

    X_train, X_test, y_train, y_test = train_test_split(
        X_raw,
        y_raw,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=y_raw
    )

    split_info = {
        "train_size": len(X_train),
        "test_size": len(X_test),
        "train_class_distribution": {
            "ham": int((y_train == 0).sum()),
            "spam": int((y_train == 1).sum())
        },
        "test_class_distribution": {
            "ham": int((y_test == 0).sum()),
            "spam": int((y_test == 1).sum())
        }
    }
    print(f"      Train set: {len(X_train)} messages (Ham: {split_info['train_class_distribution']['ham']}, Spam: {split_info['train_class_distribution']['spam']})")
    print(f"      Test set:  {len(X_test)} messages (Ham: {split_info['test_class_distribution']['ham']}, Spam: {split_info['test_class_distribution']['spam']})")

    # 3. Model Pipelines
    print("\n[3/6] Defining candidate model architectures...")
    # Baseline: TF-IDF (unigrams + bigrams) + Multinomial Naive Bayes
    models = {
        "Multinomial Naive Bayes (Baseline)": Pipeline([
            ("tfidf", TfidfVectorizer(
                lowercase=True,
                ngram_range=(1, 2),
                min_df=2,
                token_pattern=r"(?u)\b\w+\b|[!?$£€]"  # Preserve key punctuation signals (!, ?, currency)
            )),
            ("clf", MultinomialNB(alpha=0.1))  # Smoothing parameter tuned for text
        ]),
        "Logistic Regression (Extension)": Pipeline([
            ("tfidf", TfidfVectorizer(
                lowercase=True,
                ngram_range=(1, 2),
                min_df=2,
                token_pattern=r"(?u)\b\w+\b|[!?$£€]"
            )),
            ("clf", LogisticRegression(
                C=2.0,
                max_iter=1000,
                random_state=RANDOM_STATE,
                class_weight="balanced"
            ))
        ]),
        "Complement Naive Bayes (Imbalance-Tuned)": Pipeline([
            ("tfidf", TfidfVectorizer(
                lowercase=True,
                ngram_range=(1, 2),
                min_df=2,
                token_pattern=r"(?u)\b\w+\b|[!?$£€]"
            )),
            ("clf", ComplementNB(alpha=0.2))
        ]),
        "Linear SVM (Calibrated)": Pipeline([
            ("tfidf", TfidfVectorizer(
                lowercase=True,
                ngram_range=(1, 2),
                min_df=2,
                token_pattern=r"(?u)\b\w+\b|[!?$£€]"
            )),
            ("clf", CalibratedClassifierCV(
                LinearSVC(random_state=RANDOM_STATE, dual="auto"),
                cv=3
            ))
        ])
    }

    results = {}
    print("\n[4/6] Training and evaluating candidate pipelines...")
    for model_name, pipeline in models.items():
        print(f"   -> Training: {model_name}...")
        pipeline.fit(X_train, y_train)
        eval_res = evaluate_model(model_name, pipeline, X_test, y_test)
        results[model_name] = eval_res
        print(f"      Accuracy: {eval_res['accuracy']:.4f} | Precision: {eval_res['precision_spam']:.4f} | "
              f"Recall: {eval_res['recall_spam']:.4f} | F1: {eval_res['f1_spam']:.4f} | ROC-AUC: {eval_res['roc_auc']:.4f}")

    # Primary baseline model for delivery
    primary_name = "Multinomial Naive Bayes (Baseline)"
    primary_pipeline = models[primary_name]
    primary_res = results[primary_name]

    # Save primary model
    model_path = os.path.join(MODELS_DIR, "spam_classifier.joblib")
    joblib.dump(primary_pipeline, model_path)
    print(f"\n[5/6] Serialized primary model saved to: {model_path}")

    # Also save Logistic Regression for comparison app
    joblib.dump(models["Logistic Regression (Extension)"], os.path.join(MODELS_DIR, "logistic_regression.joblib"))

    # 4. Plots and Visualizations
    print("\n[6/6] Generating evaluation visual artifacts...")

    # A) Confusion Matrix Plot
    plt.figure(figsize=(6, 5))
    cm = np.array(primary_res["confusion_matrix"])
    labels = ["Ham (0)", "Spam (1)"]
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
        cbar=False,
        annot_kws={"size": 14}
    )
    plt.title(f"Confusion Matrix: {primary_name}\n(Held-out Test Set, N={len(X_test)})", fontsize=12, pad=12)
    plt.xlabel("Predicted Label", fontsize=11, fontweight="bold")
    plt.ylabel("Actual Label", fontsize=11, fontweight="bold")
    plt.tight_layout()
    cm_path = os.path.join(OUTPUTS_DIR, "confusion_matrix.png")
    plt.savefig(cm_path, dpi=300)
    plt.close()
    print(f"      Saved: {cm_path}")

    # B) ROC and Precision-Recall Curves Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    for model_name, eval_res in results.items():
        fpr, tpr, _ = roc_curve(y_test, eval_res["y_proba"])
        roc_score = auc(fpr, tpr)
        ax1.plot(fpr, tpr, lw=2, label=f"{model_name[:18]} (AUC={roc_score:.3f})")

        prec, rec, _ = precision_recall_curve(y_test, eval_res["y_proba"])
        pr_score = average_precision_score(y_test, eval_res["y_proba"])
        ax2.plot(rec, prec, lw=2, label=f"{model_name[:18]} (AP={pr_score:.3f})")

    ax1.plot([0, 1], [0, 1], "k--", lw=1.5, alpha=0.7)
    ax1.set_title("Receiver Operating Characteristic (ROC)", fontsize=12)
    ax1.set_xlabel("False Positive Rate (FPR)")
    ax1.set_ylabel("True Positive Rate (TPR / Recall)")
    ax1.legend(loc="lower right", fontsize=9)
    ax1.grid(True, alpha=0.3)

    ax2.set_title("Precision-Recall Curve (Spam = Positive)", fontsize=12)
    ax2.set_xlabel("Recall (Spam Caught)")
    ax2.set_ylabel("Precision (Accuracy of Spam Flags)")
    ax2.legend(loc="lower left", fontsize=9)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    roc_pr_path = os.path.join(OUTPUTS_DIR, "roc_pr_curve.png")
    plt.savefig(roc_pr_path, dpi=300)
    plt.close()
    print(f"      Saved: {roc_pr_path}")

    # C) Model Comparison Bar Chart
    fig, ax = plt.subplots(figsize=(10, 5))
    comp_df = pd.DataFrame([
        {
            "Model": k.replace(" (Baseline)", "").replace(" (Extension)", "").replace(" (Imbalance-Tuned)", "").replace(" (Calibrated)", ""),
            "Accuracy": v["accuracy"],
            "Precision (Spam)": v["precision_spam"],
            "Recall (Spam)": v["recall_spam"],
            "F1-Score (Spam)": v["f1_spam"]
        }
        for k, v in results.items()
    ])
    comp_df_melted = comp_df.melt(id_vars="Model", var_name="Metric", value_name="Score")
    sns.barplot(data=comp_df_melted, x="Model", y="Score", hue="Metric", ax=ax, palette="Set2")
    ax.set_ylim(0.85, 1.01)
    ax.set_title("Model Architecture Performance Comparison on Test Set", fontsize=12, fontweight="bold")
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()
    comp_path = os.path.join(OUTPUTS_DIR, "model_comparison.png")
    plt.savefig(comp_path, dpi=300)
    plt.close()
    print(f"      Saved: {comp_path}")

    # 5. Required 10+ Example Predictions
    example_messages = [
        "Congratulations! You have won a prize. Claim now.",
        "Are we still meeting at 5 pm?",
        "URGENT: claim your free reward today!",
        "Please send me the notes from class.",
        "Your account has been selected for a bonus.",
        "I will call you when I reach home.",
        "Limited offer! Click the link to get a discount.",
        "Can you share the assignment file?",
        "You have been selected for a cash reward.",
        "Thanks, see you tomorrow.",
        "Hey, can you review the pull request on GitHub before tonight?",
        "IMPORTANT: Your parcel delivery failed. Update your card details at http://scam-delivery.link",
        "Your Google verification code is 482910. Do not share this with anyone."
    ]

    example_preds = []
    classes = list(primary_pipeline.classes_)  # [0, 1]
    for idx, msg in enumerate(example_messages, 1):
        pred_label_int = int(primary_pipeline.predict([msg])[0])
        pred_probas = primary_pipeline.predict_proba([msg])[0]
        confidence = float(pred_probas[classes.index(pred_label_int)])
        example_preds.append({
            "id": idx,
            "input_message": msg,
            "predicted_label": "Spam" if pred_label_int == 1 else "Ham",
            "confidence": round(confidence, 4),
            "ham_probability": round(float(pred_probas[0]), 4),
            "spam_probability": round(float(pred_probas[1]), 4)
        })

    example_df = pd.DataFrame(example_preds)
    example_csv_path = os.path.join(OUTPUTS_DIR, "example_predictions.csv")
    example_df.to_csv(example_csv_path, index=False)
    print(f"      Saved: {example_csv_path} (13 test cases)")

    # 6. Save comprehensive metrics JSON
    metrics_data = {
        "environment": {
            "python_version": platform.python_version(),
            "scikit_learn_version": sklearn.__version__,
            "pandas_version": pd.__version__,
            "random_state": RANDOM_STATE
        },
        "dataset_audit": audit_info,
        "split_summary": split_info,
        "primary_model": {
            "name": primary_name,
            "architecture": "TfidfVectorizer(ngram_range=(1,2), min_df=2) + MultinomialNB(alpha=0.1)",
            "metrics": {
                "accuracy": primary_res["accuracy"],
                "precision_spam": primary_res["precision_spam"],
                "recall_spam": primary_res["recall_spam"],
                "f1_spam": primary_res["f1_spam"],
                "roc_auc": primary_res["roc_auc"]
            },
            "confusion_matrix": {
                "true_negative_ham": int(cm[0][0]),
                "false_positive_spam_flagged": int(cm[0][1]),
                "false_negative_spam_missed": int(cm[1][0]),
                "true_positive_spam_caught": int(cm[1][1])
            }
        },
        "model_comparison": {
            m: {
                "accuracy": r["accuracy"],
                "precision_spam": r["precision_spam"],
                "recall_spam": r["recall_spam"],
                "f1_spam": r["f1_spam"],
                "roc_auc": r["roc_auc"]
            }
            for m, r in results.items()
        }
    }

    metrics_json_path = os.path.join(OUTPUTS_DIR, "metrics.json")
    with open(metrics_json_path, "w", encoding="utf-8") as f:
        json.dump(metrics_data, f, indent=4)
    print(f"      Saved: {metrics_json_path}")

    # Plain text summary for quick terminal review
    metrics_txt_path = os.path.join(OUTPUTS_DIR, "metrics.txt")
    with open(metrics_txt_path, "w", encoding="utf-8") as f:
        f.write("Spam SMS / Email Classifier - Performance Metrics Report\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Test Set Size: {len(X_test)} messages\n")
        f.write(f"Test Class Balance: Ham={split_info['test_class_distribution']['ham']}, Spam={split_info['test_class_distribution']['spam']}\n\n")
        f.write(f"Primary Model: {primary_name}\n")
        f.write(f"  Accuracy:         {primary_res['accuracy']:.4%}\n")
        f.write(f"  Precision (Spam): {primary_res['precision_spam']:.4%}\n")
        f.write(f"  Recall (Spam):    {primary_res['recall_spam']:.4%}\n")
        f.write(f"  F1-Score (Spam):  {primary_res['f1_spam']:.4%}\n")
        f.write(f"  ROC-AUC:          {primary_res['roc_auc']:.4%}\n\n")
        f.write("Confusion Matrix:\n")
        f.write(f"  True Negative (Ham correctly caught):  {cm[0][0]}\n")
        f.write(f"  False Positive (Ham misflagged):       {cm[0][1]}\n")
        f.write(f"  False Negative (Spam missed):          {cm[1][0]}\n")
        f.write(f"  True Positive (Spam caught):           {cm[1][1]}\n\n")
        f.write("Model Comparison Benchmark:\n")
        f.write(comp_df.to_string(index=False))
        f.write("\n")
    print(f"      Saved: {metrics_txt_path}")

    print("\nTraining and benchmarking completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
