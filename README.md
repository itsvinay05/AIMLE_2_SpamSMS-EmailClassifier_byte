# 🛡️ Spam SMS / Email Classifier

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.9.1-orange.svg)](https://scikit-learn.org/)
[![License: CC BY 4.0](https://img.shields.io/badge/License-CC_BY_4.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)
[![Tests Passing](https://img.shields.io/badge/tests-6%2F6%20passed-brightgreen.svg)]()

A complete, production-grade Natural Language Processing (NLP) text classification system that detects whether incoming messages are **Spam** or **Ham** (legitimate). Built with a leak-free preprocessing pipeline, TF-IDF n-gram vectorization, Multinomial Naive Bayes baseline, multiple candidate model benchmarks, an explainability engine, and an interactive Streamlit Web UI.

---

## 📌 Executive Project Summary

> **"This project classifies SMS/email text as Spam or Ham using a Multinomial Naive Bayes pipeline and the UCI SMS Spam Collection dataset. Text is transformed using sub-linear TF-IDF word unigram and bigram tokenization preserving currency and urgency punctuation. On a held-out test set of 1,035 messages, the model achieved 98.74% accuracy, 98.36% spam precision, 91.60% spam recall, and 94.86% spam F1-score. The confusion matrix, ROC/PR curves, and 13 confidence-scored predictions are included. Limitations include short-text dataset bias, evolving adversarial spam patterns, and uncalibrated Naive Bayes probability overconfidence."**

---

## 1. Project Description & Intended Use

### Objective
Unsolicited spam messages and phishing scams cause billions of dollars in losses, security breaches, and user frustration annually. The goal of this project is to build an automated, low-latency, and interpretable classification system that:
1. Accurately flags Spam messages while prioritizing **very high precision** (minimizing False Positives) so critical personal or transactional communications are never mistakenly discarded.
2. Generates well-calibrated probabilistic **confidence scores** for downstream threshold tuning.
3. Provides explainable predictions by isolating the exact token signals pushing a message toward Spam or Ham.

### Intended Use Cases
- Inbound SMS gateway filtering for mobile network operators or IoT devices.
- Email spam and phishing filtering pre-processors.
- Customer support messaging inbox triage to block promotional bot spam.

---

## 2. Dataset Source, License, & Preprocessing

- **Dataset Name**: UCI SMS Spam Collection
- **Source URL**: [https://archive.ics.uci.edu/dataset/228/sms+spam+collection](https://archive.ics.uci.edu/dataset/228/sms+spam+collection)
- **Direct Zip URL**: `https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip`
- **Citation**: Almeida, T.A., Gómez Hidalgo, J.M., Yamakami, A. (2011). *Contributions to the Study of SMS Spam Filtering: New Collection and Results*. ACM DocEng'11.
- **License**: Creative Commons Attribution 4.0 International (CC BY 4.0).
- **Format**: Tab-separated text (`<label>\t<text>`).
- **Raw Rows**: 5,574 messages (4,827 Ham, 747 Spam).
- **Label Mapping**: `ham` $\rightarrow$ `0` (negative class), `spam` $\rightarrow$ `1` (positive class).

### Preprocessing & Data Leakage Prevention
1. **Missing Values**: Checked for null values in both label and text columns (0 missing values found).
2. **Duplicate Removal Decision**:
   - 403 exact duplicate messages were detected (309 Ham duplicates such as `"Ok"`, `"Sorry, I'll call later"`, and 94 Spam duplicates from repeated commercial broadcast campaigns).
   - **Documented Decision**: All 403 duplicate rows were removed prior to train/test splitting, leaving **5,171 unique messages**. Allowing duplicates to be split across train and test sets causes **data leakage** and artificially inflates test performance on memorized templates.
3. **Train/Test Split**:
   - 80% Training set: **4,136 messages** (3,614 Ham, 522 Spam).
   - 20% Held-out Test set: **1,035 messages** (904 Ham, 131 Spam).
   - Stratified by class label to preserve the native 87.4% / 12.6% distribution.
   - Fixed random seed: `random_state=42`.
4. **TF-IDF Vectorization**:
   - Fit strictly on the training set (`X_train`) only; test set (`X_test`) is transformed via the fitted pipeline to prevent data snooping.
   - Unigrams and bigrams: `ngram_range=(1, 2)`.
   - Minimum document frequency: `min_df=2`.
   - Punctuation Preservation: Instead of aggressively removing all punctuation, key spam signals (currency symbols `$`, `£`, `€` and urgency indicators `!`, `?`) are preserved in the token pattern `(?u)\b\w+\b|[!?$£€]`.

---

## 3. Repository Structure

```
spam-sms-email-classifier/
├── data/
│   ├── SMSSpamCollection          # Cleaned raw dataset
│   ├── readme                     # Original dataset documentation
│   └── sms_spam_collection.zip    # Source archive
├── notebooks/
│   └── spam_classifier.ipynb      # Complete walkthrough notebook with EDA and analysis
├── src/
│   ├── train.py                   # Training pipeline and model benchmark script
│   ├── predict.py                 # CLI inference tool with confidence and explanation
│   └── app.py                     # Streamlit web application
├── models/
│   ├── spam_classifier.joblib     # Serialized primary pipeline (TF-IDF + MultinomialNB)
│   └── logistic_regression.joblib # Serialized comparative model (TF-IDF + Logistic Regression)
├── outputs/
│   ├── confusion_matrix.png       # Test set confusion matrix heatmap
│   ├── roc_pr_curve.png           # ROC and Precision-Recall comparison curves
│   ├── model_comparison.png       # Multi-model metric comparison chart
│   ├── metrics.json               # Detailed machine-readable metrics and audit data
│   ├── metrics.txt                # Human-readable metrics summary
│   └── example_predictions.csv    # Benchmark test cases with confidence scores
├── tests/
│   └── test_classifier.py         # Automated unit test suite (6/6 passing)
├── requirements.txt               # Locked dependencies
└── README.md                      # Project documentation and user guide
```

---

## 4. Environment Setup & Installation

### 1. Clone or Navigate to the Repository
```bash
cd spam-sms-email-classifier
```

### 2. Create and Activate a Virtual Environment
```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# Linux / macOS
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

---

## 5. How to Train and Evaluate the Model

To execute the complete training, benchmarking, and artifact generation pipeline:

```bash
python src/train.py
```

### What `train.py` Does:
1. Loads `data/SMSSpamCollection` and audits missing/duplicate rows.
2. Splits data into stratified 80/20 train/test sets with `random_state=42`.
3. Trains four candidate models:
   - **Multinomial Naive Bayes (Baseline)**
   - **Logistic Regression (Extension with balanced class weights)**
   - **Complement Naive Bayes (Imbalance-tuned)**
   - **Linear Support Vector Classifier (Calibrated probabilities)**
4. Serializes the primary pipeline into `models/spam_classifier.joblib`.
5. Outputs evaluation plots to `outputs/confusion_matrix.png`, `outputs/roc_pr_curve.png`, and `outputs/model_comparison.png`.
6. Generates `outputs/metrics.json` and evaluates the 10+ benchmark predictions into `outputs/example_predictions.csv`.

---

## 6. Actual Measured Test Metrics

All results measured on the held-out test set ($N = 1,035$ messages; 904 Ham, 131 Spam):

| Metric | Primary Model: MultinomialNB | Extension: Logistic Regression | Extension: ComplementNB | Extension: Calibrated LinearSVC |
| :--- | :---: | :---: | :---: | :---: |
| **Accuracy** | **98.74%** | 97.78% | 98.16% | 98.26% |
| **Precision (Spam)** | **98.36%** | 92.19% | 91.18% | 95.93% |
| **Recall (Spam)** | 91.60% | 90.08% | **94.66%** | 90.08% |
| **F1-Score (Spam)** | **94.86%** | 91.12% | 92.88% | 92.91% |
| **ROC-AUC** | **99.60%** | 99.51% | 99.53% | **99.67%** |

### Confusion Matrix Breakdown (Primary Model)

```
                     Predicted Ham (0)    Predicted Spam (1)
Actual Ham (0)             902                     2   (False Positives: 0.22%)
Actual Spam (1)             11                   120   (True Positives: 91.60%)
```

- **True Negatives (Ham correctly identified)**: 902 / 904 ($99.78\%$)
- **False Positives (Ham incorrectly flagged as Spam)**: **Only 2 out of 904** ($0.22\%$)
- **True Positives (Spam successfully caught)**: 120 / 131 ($91.60\%$)
- **False Negatives (Spam missed)**: 11 / 131 ($8.40\%$)

The high spam precision ($98.36\%$) ensures that legitimate users almost never experience having valid messages sent to the spam folder.

---

## 7. How to Run Inference

### A. Command Line Interface (`src/predict.py`)

Run single-message inference with confidence score and explainability:

```bash
python src/predict.py --message "URGENT: claim your free reward today!" --explain
```

**Output:**
```text
-------------------------------------------------------
Input Message : "URGENT: claim your free reward today!"
Prediction    : [ SPAM ] -> Spam
Confidence    : 99.77%
Probabilities : Ham=0.23% | Spam=99.77%

Feature Impact (Word Signals):
  • Pushing towards SPAM : claim, urgent, free, reward, today
-------------------------------------------------------
```

Test a legitimate ham message:
```bash
python src/predict.py --message "Are we still meeting at 5 pm?" --explain
```

**Output:**
```text
-------------------------------------------------------
Input Message : "Are we still meeting at 5 pm?"
Prediction    : [ HAM / NOT SPAM ] -> Ham
Confidence    : 99.64%
Probabilities : Ham=99.64% | Spam=0.36%

Feature Impact (Word Signals):
  • Pushing towards HAM  : meeting, are we, still, we still, at 5
-------------------------------------------------------
```

#### Interactive Console Mode:
Simply run `python src/predict.py` without arguments to launch an interactive loop where you can paste messages one by one.

### B. Interactive Web UI (`src/app.py`)

Launch the Streamlit web dashboard:
```bash
streamlit run src/app.py
```
**Web Dashboard Features:**
- Real-time text classification with confidence progress bar.
- Color-coded badges for Spam and Ham.
- **Threshold Tuning Slider**: dynamically observe how raising the spam threshold from $0.50$ to $0.80$ minimizes false alarms.
- **Feature Explainability Chart**: bar chart revealing which tokens drove the classification.
- One-click benchmark test loader.

---

## 8. Benchmark Example Predictions (Section 6 Requirements)

The 10 example messages from the project specification plus 3 real-world edge cases were evaluated through the saved model pipeline:

| # | Input Message | Predicted Label | Confidence | Ham Probability | Spam Probability |
| :-: | :--- | :---: | :---: | :---: | :---: |
| 1 | `Congratulations! You have won a prize. Claim now.` | **Spam** | 99.97% | 0.03% | 99.97% |
| 2 | `Are we still meeting at 5 pm?` | **Ham** | 99.64% | 99.64% | 0.36% |
| 3 | `URGENT: claim your free reward today!` | **Spam** | 99.77% | 0.23% | 99.77% |
| 4 | `Please send me the notes from class.` | **Ham** | 99.46% | 99.46% | 0.54% |
| 5 | `Your account has been selected for a bonus.` | **Spam** | 95.33% | 4.67% | 95.33% |
| 6 | `I will call you when I reach home.` | **Ham** | 99.97% | 99.97% | 0.03% |
| 7 | `Limited offer! Click the link to get a discount.` | **Spam** | 93.23% | 6.77% | 93.23% |
| 8 | `Can you share the assignment file?` | **Ham** | 99.25% | 99.25% | 0.75% |
| 9 | `You have been selected for a cash reward.` | **Spam** | 96.39% | 3.61% | 96.39% |
| 10 | `Thanks, see you tomorrow.` | **Ham** | 99.58% | 99.58% | 0.42% |
| 11 | `Hey, can you review the pull request on GitHub before tonight?` | **Ham** | 99.07% | 99.07% | 0.93% |
| 12 | `IMPORTANT: Your parcel delivery failed. Update your card details at http://scam-delivery.link` | **Spam** | 94.71% | 5.29% | 94.71% |
| 13 | `Your Google verification code is 482910. Do not share this with anyone.` | **Ham** | 92.70% | 92.70% | 7.30% |

All 10 benchmark test cases were classified with $100\%$ accuracy and high confidence.

---

## 9. Running Automated Tests

To run the automated test suite verifying model loading, probability normalization, edge case robustness, and metrics files:

```bash
python -m unittest tests/test_classifier.py -v
```

All 6 test suites execute in $<5$ seconds and verify that edge cases (empty strings, pure punctuation, very long strings) do not throw uncaught exceptions.

---

## 10. Known Limitations & Technical Discussion

1. **Short SMS vs. Long Email Distribution Shift**:
   - The primary dataset contains mobile SMS messages (character lengths typically between 20 and 160 characters).
   - Long email communications feature different structural components (subject lines, HTML headers, multi-paragraph bodies, disclaimers). While the TF-IDF pipeline handles arbitrary text length, classifying email corpora requires training on email-specific corpora (such as Enron Spam or Apache SpamAssassin) to avoid domain mismatch.
2. **Adversarial Spam Evasion (Concept Drift)**:
   - Modern spammers deliberately disguise trigger words using leetspeak (e.g. `fr33`, `pr!ze`), zero-width unicode characters, or image-embedded text.
   - Traditional bag-of-words TF-IDF models cannot identify unseen obfuscations unless paired with sub-word character n-gram tokenizers or Transformer embeddings (such as RoBERTa or DistilBERT).
3. **Naive Bayes Independence Assumption & Calibration**:
   - Multinomial Naive Bayes assumes conditional independence between words given the class label ($P(w_1, w_2 | y) = P(w_1 | y) P(w_2 | y)$). While this assumption rarely harms classification ranking, it frequently pushes posterior probabilities close to $0.0$ or $1.0$.
   - For high-stakes applications, using **Platt Scaling** or **Isotonic Regression** (via `CalibratedClassifierCV`) provides better calibrated probabilities.
4. **False Positive Cost Asymmetry**:
   - In real-world spam filtering, the cost of a False Positive (missing an urgent medical or banking alert) is significantly higher than a False Negative (a promotional spam reaching the inbox). We recommend operating the model with a higher decision threshold (e.g. `threshold = 0.85`) to strictly protect legitimate communications.

---

## 11. Final Submission Checklist Verification

- [x] **GitHub repository structure**: Clean layout with `data/`, `notebooks/`, `src/`, `models/`, `outputs/`, `tests/`, `requirements.txt`, and `README.md`.
- [x] **Saved serialized model file**: `models/spam_classifier.joblib` (TF-IDF + MultinomialNB pipeline).
- [x] **Metrics and confusion-matrix images**: Saved to `outputs/confusion_matrix.png`, `outputs/roc_pr_curve.png`, and `outputs/model_comparison.png`.
- [x] **Example predictions CSV with 10+ rows**: `outputs/example_predictions.csv` with 13 scored predictions.
- [x] **Short project summary**: Highlighted in Section 2 with exact measured results.
- [x] **Interactive Demo Web App**: `src/app.py` ready to launch via `streamlit run src/app.py`.
