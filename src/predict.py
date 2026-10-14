"""
Spam SMS / Email Classifier - Command Line Inference Interface
--------------------------------------------------------------
Loads serialized scikit-learn pipeline and predicts Spam vs Ham with confidence scores.
"""

import argparse
import os
import sys
import joblib
import numpy as np

# Ensure project root is in python path
sys.path.insert(0, os.path.abspath("."))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

try:
    from src.email_parser import parse_email_message
except ImportError:
    try:
        from email_parser import parse_email_message
    except ImportError:
        parse_email_message = None

DEFAULT_MODEL_PATH = os.path.join("models", "spam_classifier.joblib")


def load_model(model_path: str = DEFAULT_MODEL_PATH):
    """Loads serialized pipeline from disk."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model file not found at: {model_path}. "
            f"Please run 'python src/train.py' first to train and serialize the model."
        )
    return joblib.load(model_path)


def predict_message(model, message_text: str, explain: bool = False):
    """
    Runs inference on a message and returns label, confidence, class probabilities,
    and optional token importance explanation.
    """
    message = [message_text]
    
    # 1. Predict label and probability distribution
    label = model.predict(message)[0]
    probabilities = model.predict_proba(message)[0]
    classes = list(model.classes_)
    confidence = probabilities[classes.index(label)]
    
    ham_prob = probabilities[classes.index(0)]
    spam_prob = probabilities[classes.index(1)]
    
    result = {
        "text": message_text,
        "label_int": int(label),
        "label_str": "Spam" if label == 1 else "Ham",
        "confidence": float(confidence),
        "ham_probability": float(ham_prob),
        "spam_probability": float(spam_prob)
    }
    
    # Optional explainability using model log probabilities/coefficients
    if explain:
        try:
            tfidf = model.named_steps["tfidf"]
            clf = model.named_steps["clf"]
            feature_names = tfidf.get_feature_names_out()
            
            # Vectorize input
            vec = tfidf.transform([message_text]).toarray()[0]
            non_zero_indices = np.where(vec > 0)[0]
            
            contributions = []
            if hasattr(clf, "feature_log_prob_"):
                # Naive Bayes log probability difference (Spam vs Ham)
                diff = clf.feature_log_prob_[1] - clf.feature_log_prob_[0]
                for idx in non_zero_indices:
                    term = feature_names[idx]
                    weight = vec[idx] * diff[idx]
                    contributions.append((term, weight))
            elif hasattr(clf, "coef_"):
                # Linear model coefficients
                coef = clf.coef_[0]
                for idx in non_zero_indices:
                    term = feature_names[idx]
                    weight = vec[idx] * coef[idx]
                    contributions.append((term, weight))
            
            # Sort: positive weight pushes to Spam, negative pushes to Ham
            contributions.sort(key=lambda x: x[1], reverse=True)
            result["explanation"] = contributions
        except Exception:
            result["explanation"] = []
            
    return result


def display_result(res: dict, show_explanation: bool = False):
    """Prints formatted prediction output to console."""
    badge = "[ SPAM ]" if res["label_str"] == "Spam" else "[ HAM / NOT SPAM ]"
    print("\n" + "-" * 55)
    print(f"Input Message : \"{res['text']}\"")
    print(f"Prediction    : {badge} -> {res['label_str']}")
    print(f"Confidence    : {res['confidence']:.2%}")
    print(f"Probabilities : Ham={res['ham_probability']:.2%} | Spam={res['spam_probability']:.2%}")
    
    if show_explanation and "explanation" in res and res["explanation"]:
        print("\nFeature Impact (Word Signals):")
        spam_signals = [t for t, w in res["explanation"] if w > 0][:5]
        ham_signals = [t for t, w in reversed(res["explanation"]) if w < 0][:5]
        if spam_signals:
            print(f"  • Pushing towards SPAM : {', '.join(spam_signals)}")
        if ham_signals:
            print(f"  • Pushing towards HAM  : {', '.join(ham_signals)}")
    print("-" * 55)


def main():
    parser = argparse.ArgumentParser(
        description="Spam SMS / Email Classifier - Text Inference Utility"
    )
    parser.add_argument(
        "-m", "--message",
        type=str,
        help="Input text message to classify."
    )
    parser.add_argument(
        "-f", "--file",
        type=str,
        help="Path to a text file containing message or raw email to classify."
    )
    parser.add_argument(
        "--batch-input",
        type=str,
        help="Path to input CSV file containing multiple messages for bulk scoring."
    )
    parser.add_argument(
        "--text-col",
        type=str,
        default="text",
        help="Column name containing message text in batch CSV (default: 'text')."
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        help="Path to save scored batch CSV output."
    )
    parser.add_argument(
        "--email",
        action="store_true",
        help="Enable email parsing mode (strips HTML, extracts Subject and headers)."
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL_PATH,
        help=f"Path to serialized joblib model (default: {DEFAULT_MODEL_PATH})."
    )
    parser.add_argument(
        "-e", "--explain",
        action="store_true",
        help="Display top words pushing prediction toward Spam or Ham."
    )
    
    args = parser.parse_args()
    
    try:
        model = load_model(args.model)
    except Exception as e:
        print(f"Error loading model: {e}", file=sys.stderr)
        sys.exit(1)
        
    if args.batch_input:
        import pandas as pd
        if not os.path.exists(args.batch_input):
            print(f"Error: Batch file not found at {args.batch_input}", file=sys.stderr)
            sys.exit(1)
        print(f"Loading batch dataset from: {args.batch_input}...")
        batch_df = pd.read_csv(args.batch_input)
        if args.text_col not in batch_df.columns:
            print(f"Error: Column '{args.text_col}' not found in CSV. Available columns: {list(batch_df.columns)}", file=sys.stderr)
            sys.exit(1)
            
        messages = batch_df[args.text_col].fillna("").astype(str).tolist()
        preds = model.predict(messages)
        probas = model.predict_proba(messages)
        classes = list(model.classes_)
        
        batch_df["predicted_label"] = ["Spam" if p == 1 else "Ham" for p in preds]
        batch_df["spam_probability"] = [round(float(pr[classes.index(1)]), 4) for pr in probas]
        batch_df["ham_probability"] = [round(float(pr[classes.index(0)]), 4) for pr in probas]
        batch_df["confidence"] = [
            round(float(pr[classes.index(p)]), 4) for p, pr in zip(preds, probas)
        ]
        
        out_path = args.output_csv or "outputs/batch_scored_results.csv"
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        batch_df.to_csv(out_path, index=False)
        print(f"[+] Successfully scored {len(batch_df)} messages.")
        print(f"    Spam Count: {(batch_df['predicted_label'] == 'Spam').sum()} ({(batch_df['predicted_label'] == 'Spam').mean():.1%})")
        print(f"    Ham Count:  {(batch_df['predicted_label'] == 'Ham').sum()} ({(batch_df['predicted_label'] == 'Ham').mean():.1%})")
        print(f"    Results saved to: {out_path}")
    elif args.message:
        text_to_process = args.message
        if args.email and parse_email_message:
            parsed = parse_email_message(args.message)
            text_to_process = parsed["combined_text"]
            if parsed["subject"]:
                print(f"Parsed Email Subject: {parsed['subject']}")
        res = predict_message(model, text_to_process, explain=args.explain)
        display_result(res, show_explanation=args.explain)
    elif args.file:
        if not os.path.exists(args.file):
            print(f"Error: File not found at {args.file}", file=sys.stderr)
            sys.exit(1)
        with open(args.file, "r", encoding="utf-8", errors="replace") as f:
            text = f.read().strip()
        text_to_process = text
        if (args.email or args.file.endswith(".eml")) and parse_email_message:
            parsed = parse_email_message(text)
            text_to_process = parsed["combined_text"]
            if parsed["subject"]:
                print(f"Parsed Email Subject: {parsed['subject']}")
        res = predict_message(model, text_to_process, explain=args.explain)
        display_result(res, show_explanation=args.explain)
    else:
        # Interactive mode
        print("=" * 55)
        print(" Spam SMS / Email Classifier - Interactive Mode")
        print(" Model: " + args.model)
        print(" Type 'quit' or 'exit' to terminate.")
        print("=" * 55)
        while True:
            try:
                user_input = input("\nEnter message: ").strip()
                if not user_input:
                    continue
                if user_input.lower() in ("exit", "quit", "q"):
                    print("Exiting. Goodbye!")
                    break
                res = predict_message(model, user_input, explain=True)
                display_result(res, show_explanation=True)
            except (KeyboardInterrupt, EOFError):
                print("\nExiting. Goodbye!")
                break


if __name__ == "__main__":
    main()
