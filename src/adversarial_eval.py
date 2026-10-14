"""
Adversarial Evasion & Stress-Testing Suite for Spam Classifier
--------------------------------------------------------------
Simulates real-world evasion attacks used by spammers:
1. Leetspeak obfuscation (e.g. 'fr33', 'pr!ze', 'cl@im')
2. Character spacing / token fragmentation ('F R E E', 'C L A I M')
3. Invisible Zero-Width Unicode insertion ('p\u200br\u200bi\u200bz\u200be')
4. Good-word dictionary stuffing (dilution attack)
5. Homoglyph substitution (Cyrillic lookalikes)

Measures attack bypass rates and exports outputs/adversarial_report.json and .png.
"""

import json
import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

LEET_MAP = {
    "a": ["@", "4"],
    "e": ["3"],
    "i": ["1", "!"],
    "o": ["0"],
    "s": ["$", "5"],
    "l": ["1", "|"],
    "t": ["7"]
}

CYRILLIC_HOMOGLYPHS = {
    "a": "\u0430",
    "e": "\u0435",
    "o": "\u043e",
    "p": "\u0440",
    "c": "\u0441",
    "x": "\u0445",
    "y": "\u0443"
}

GOOD_WORDS_CORPUS = [
    "meeting", "schedule", "attached", "report", "sincerely", "thanks", "regards",
    "project", "team", "review", "tomorrow", "class", "notes", "family", "dinner",
    "discussion", "update", "document", "contract", "client", "office", "colleague"
]


def attack_leetspeak(text: str, intensity: float = 0.5) -> str:
    """Substitutes characters with numbers and symbols."""
    chars = []
    for c in text:
        lower = c.lower()
        if lower in LEET_MAP and random.random() < intensity:
            chars.append(random.choice(LEET_MAP[lower]))
        else:
            chars.append(c)
    return "".join(chars)


def attack_char_spacing(text: str) -> str:
    """Splits key uppercase or urgent words with spaces (e.g. U R G E N T)."""
    words = text.split()
    new_words = []
    for w in words:
        if len(w) > 2 and (w.isupper() or w.lower() in ("free", "prize", "claim", "urgent", "reward", "cash", "bonus")):
            new_words.append(" ".join(list(w)))
        else:
            new_words.append(w)
    return " ".join(new_words)


def attack_zero_width_chars(text: str) -> str:
    """Inserts zero-width spaces (\u200b) to break tokenization while preserving visual appearance."""
    chars = []
    for c in text:
        chars.append(c)
        if random.random() < 0.35 and c.isalnum():
            chars.append("\u200b")
    return "".join(chars)


def attack_homoglyphs(text: str, intensity: float = 0.6) -> str:
    """Substitutes Latin characters with visually identical Cyrillic homoglyphs."""
    chars = []
    for c in text:
        lower = c.lower()
        if lower in CYRILLIC_HOMOGLYPHS and random.random() < intensity:
            chars.append(CYRILLIC_HOMOGLYPHS[lower])
        else:
            chars.append(c)
    return "".join(chars)


def attack_good_word_stuffing(text: str, num_words: int = 15) -> str:
    """Appends innocent corporate/casual vocabulary to dilute spam keyword frequency."""
    selected = random.sample(GOOD_WORDS_CORPUS, min(num_words, len(GOOD_WORDS_CORPUS)))
    return text + " " + " ".join(selected)


def main():
    print("=" * 60)
    print(" Adversarial Evasion & Robustness Stress-Testing Suite")
    print("=" * 60)

    model_path = os.path.join("models", "spam_classifier.joblib")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}. Run 'python src/train.py' first.")

    model = joblib.load(model_path)
    random.seed(42)
    np.random.seed(42)

    # Core spam test set (unseen high-confidence spam messages)
    spam_samples = [
        "Congratulations! You have won a prize. Claim now.",
        "URGENT: claim your free reward today!",
        "Your account has been selected for a bonus.",
        "Limited offer! Click the link to get a discount.",
        "You have been selected for a cash reward.",
        "WINNER!! As a valued customer you have been selected for a £1000 prize reward! Call 09061701461.",
        "Free entry in 2 a weekly competition to win FA Cup final tickets. Text FA to 87121 now!",
        "Had your mobile 11 months or more? You are entitled to update to the latest color mobile for Free!",
        "IMPORTANT: Your parcel delivery failed. Update your card details at http://scam-delivery.link",
        "Exclusive deal: get 50% discount on credit card debt relief today only. Call now!"
    ]

    attacks = {
        "Clean Baseline (No Attack)": lambda t: t,
        "Leetspeak Obfuscation (e.g. pr!z3)": attack_leetspeak,
        "Token Fragmentation ('F R E E')": attack_char_spacing,
        "Zero-Width Unicode Insertion": attack_zero_width_chars,
        "Homoglyph Substitution (Cyrillic)": attack_homoglyphs,
        "Good-Word Dilution Attack": attack_good_word_stuffing
    }

    report = {}
    attack_summary = []

    print(f"\nEvaluating {len(spam_samples)} spam samples across {len(attacks)} evasion attack strategies...")

    for attack_name, attack_fn in attacks.items():
        perturbed_messages = [attack_fn(m) for m in spam_samples]
        preds = model.predict(perturbed_messages)
        probas = model.predict_proba(perturbed_messages)[:, 1]

        caught_count = int(np.sum(preds == 1))
        evaded_count = int(np.sum(preds == 0))
        detection_rate = caught_count / len(spam_samples)
        bypass_rate = evaded_count / len(spam_samples)
        avg_confidence = float(np.mean(probas))

        report[attack_name] = {
            "detection_rate": round(detection_rate, 4),
            "bypass_rate": round(bypass_rate, 4),
            "caught_count": caught_count,
            "evaded_count": evaded_count,
            "avg_spam_probability": round(avg_confidence, 4),
            "sample_perturbed": perturbed_messages[0]
        }

        attack_summary.append({
            "Attack Strategy": attack_name,
            "Detection Rate": detection_rate,
            "Bypass Rate": bypass_rate,
            "Avg Spam Confidence": avg_confidence
        })

        print(f"\n• {attack_name}:")
        print(f"  Detection: {detection_rate:.1%} | Evaded: {evaded_count}/{len(spam_samples)} | Avg Spam Conf: {avg_confidence:.2%}")
        safe_example = perturbed_messages[0].encode("ascii", "backslashreplace").decode("ascii")
        print(f"  Example: \"{safe_example}\"")

    # Save JSON report
    report_json_path = os.path.join("outputs", "adversarial_report.json")
    with open(report_json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=4)
    print(f"\n[+] Saved adversarial audit report to: {report_json_path}")

    # Generate Chart
    summary_df = pd.DataFrame(attack_summary)
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(
        data=summary_df,
        y="Attack Strategy",
        x="Detection Rate",
        palette="Blues_r",
        ax=ax
    )
    ax.set_xlim(0, 1.05)
    ax.set_xlabel("Spam Detection Rate (Higher = More Resilient)", fontsize=11, fontweight="bold")
    ax.set_title("Classifier Resilience Under Adversarial Evasion Attacks", fontsize=12, fontweight="bold", pad=12)
    for p in ax.patches:
        width = p.get_width()
        ax.annotate(f"{width:.1%}", (width + 0.02, p.get_y() + p.get_height() / 2.),
                    va="center", fontsize=10, fontweight="bold")
    plt.tight_layout()
    chart_path = os.path.join("outputs", "adversarial_report.png")
    plt.savefig(chart_path, dpi=300)
    plt.close()
    print(f"[+] Saved adversarial robustness plot to: {chart_path}")

    print("\nAdversarial evaluation complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
