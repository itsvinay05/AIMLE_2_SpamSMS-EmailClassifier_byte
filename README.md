# 🛡️ Spam SMS / Email Classifier

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.9.1-orange.svg)](https://scikit-learn.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.143.0-009688.svg)](https://fastapi.tiangolo.com/)
[![GitHub Repo](https://img.shields.io/badge/GitHub-Repository-181717.svg?logo=github&logoColor=white)](https://github.com/itsvinay05/spam-sms-email-classifier)
[![Vercel Ready](https://img.shields.io/badge/Vercel-Live-000000.svg?logo=vercel&logoColor=white)](https://spam-sms-email-classifier-three.vercel.app)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED.svg)](https://www.docker.com/)
[![License: CC BY 4.0](https://img.shields.io/badge/License-CC_BY_4.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)
[![Tests Passing](https://img.shields.io/badge/tests-16%2F16%20passed-brightgreen.svg)]()

A complete, production-grade Natural Language Processing (NLP) text classification system that detects whether incoming messages are **Spam** or **Ham** (legitimate). Built with a leak-free preprocessing pipeline, TF-IDF n-gram vectorization, Multinomial Naive Bayes baseline, multiple candidate model benchmarks, an explainability engine, an adversarial robustness security evaluation, a FastAPI microservice, an interactive Streamlit Web UI, and a zero-latency full-stack **Vercel Web Application** with serverless endpoints.

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

## 9. Production REST API Microservice (`src/api.py`)

A production-grade, asynchronous FastAPI backend service is provided for integrating the spam classifier into mobile applications, web relays, and enterprise microservices.

### Launch the REST API
```bash
python -m uvicorn src.api:app --reload --port 8000
```
* **Interactive OpenAPI / Swagger Documentation**: [http://localhost:8000/docs](http://localhost:8000/docs)
* **Alternative ReDoc UI**: [http://localhost:8000/redoc](http://localhost:8000/redoc)

### API Endpoints
| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/` | Service health status and endpoints map |
| `GET` | `/health` | Health probe checking model load status |
| `POST` | `/api/v1/predict` | Single message or raw email inference with optional explainability |
| `POST` | `/api/v1/batch_predict` | High-throughput batch message classification |
| `GET` | `/api/v1/metrics` | Returns held-out test evaluation JSON |

### Sample cURL Request (Single Prediction):
```bash
curl -X POST "http://localhost:8000/api/v1/predict" \
     -H "Content-Type: application/json" \
     -d '{
       "text": "Congratulations! You have won a $1000 prize. Claim now.",
       "explain": true,
       "threshold": 0.50
     }'
```

---

## 10. Email Parsing & Preprocessing (`src/email_parser.py`)

Real email communications feature MIME headers, HTML formatting, hidden tracking links, and subject lines. The [`src/email_parser.py`](src/email_parser.py) module handles:
* Extraction of **Subject**, **From**, **To**, and **Date** headers.
* Stripping of HTML formatting, scripts, and CSS stylesheets while retaining text and link anchors.
* Extraction of suspicious URLs (IP address URLs and shortened URLs like `bit.ly`, `tinyurl.com`).
* Combined feature representation with high weighting placed on Subject line keywords.

Run inference directly on raw emails or `.eml` files:
```bash
python src/predict.py --message "Subject: Urgent: Verify Account. Click http://192.168.1.1/login" --email --explain
python src/predict.py --file sample_email.eml --explain
```

---

## 11. Adversarial Robustness & Evasion Stress-Testing (`src/adversarial_eval.py`)

To evaluate how the model withstands real-world evasion attacks by spammers, we developed an automated adversarial stress-testing suite:

```bash
python src/adversarial_eval.py
```

### Evaluated Attack Vectors & Measured Bypass Rates
| Adversarial Attack Vector | Perturbation Method | Detection Rate | Evasion / Bypass Rate |
| :--- | :--- | :---: | :---: |
| **Clean Baseline** | No perturbation applied | **100.0%** | 0.0% |
| **Token Fragmentation** | Spacing urgent tokens (`"C L A I M"`) | **90.0%** | 10.0% |
| **Good-Word Dilution** | Stuffing innocent corporate vocabulary | **70.0%** | 30.0% |
| **Homoglyph Substitution** | Replacing characters with Cyrillic lookalikes | **50.0%** | 50.0% |
| **Zero-Width Unicode** | Inserting hidden `\u200b` characters | **40.0%** | 60.0% |
| **Leetspeak Obfuscation** | Substituting letters (`"pr!z3"`, `"cl@im"`) | **30.0%** | 70.0% |

*Results and comparative bar chart are saved to [`outputs/adversarial_report.json`](outputs/adversarial_report.json) and [`outputs/adversarial_report.png`](outputs/adversarial_report.png).*

---

## 12. Batch CSV & Bulk Scoring Mode

Process entire datasets or customer inquiry logs in parallel:

```bash
python src/predict.py --batch-input input_messages.csv --text-col "message_text" --output-csv outputs/flagged_spam.csv
```

---

## 13. Docker Containerization (`Dockerfile`)

The project includes a production multi-mode `Dockerfile` supporting both Streamlit Web UI and FastAPI microservice deployments:

### Build Container
```bash
docker build -t spam-classifier .
```

### Run Streamlit Web Application (Port 8501)
```bash
docker run -p 8501:8501 spam-classifier
```

### Run FastAPI Microservice (Port 8000)
```bash
docker run -p 8000:8000 -e APP_MODE=api spam-classifier
```

---

## 14. Vercel Web Application & Serverless API Deployment

The project includes a production-ready, full-stack **Vercel Web Application** featuring:

> **Live Production URL**: [https://spam-sms-email-classifier-three.vercel.app](https://spam-sms-email-classifier-three.vercel.app)

- **Instant Client-Side Edge Inference (0 ms latency)**: Using precomputed vocabulary IDF and log-likelihood weights (`public/model_export.json`), the browser classifies text in less than 1 millisecond with zero server roundtrips.
- **Vercel Serverless Python API (`/api/predict`)**: A FastAPI microservice running in Vercel Serverless Functions for automated external system integrations.
- **Interactive UI Tabs**:
  1. *Live Classifier*: Real-time SMS and text classification with token signal explainability.
  2. *Email Scanner*: RFC-822 / MIME email headers and HTML body parsing.
  3. *Benchmark Suite*: All 13 verified test cases preloaded with 1-click execution.
  4. *Confusion Matrix & Metrics*: Live rendering of the test matrix and ROC curves.
  5. *Batch CSV Scorer*: Multi-line message processing and downloadable reports.
  6. *Threshold Sensitivity*: Dynamic decision threshold tuning ($\theta \in [0.05, 0.95]$).
  7. *REST API Playground*: Copyable cURL, Python, and JavaScript code snippets.

### Running the Vercel App Locally
```bash
# Run local preview server with Uvicorn
python -m uvicorn api.index:app --host 127.0.0.1 --port 8000
```
Open [http://127.0.0.1:8000](http://127.0.0.1:8000) in your web browser.

### Deploying to Vercel (1-Command Deploy)
```bash
# 1. Login to your Vercel account
npx vercel login

# 2. Deploy directly to Vercel production
npx vercel --prod
```

---

## 15. Running Automated Tests (16/16 Passing)

To execute the entire automated test suite:

```bash
python -m unittest discover tests -v
```

All 16 unit and integration test suites pass with 100% success rate:
- `tests/test_classifier.py` (9 core pipeline, email parsing, model serialization, and adversarial tests)
- `tests/test_vercel_api.py` (7 Vercel API, serverless prediction, health check, batch scoring, and threshold sensitivity tests)

---

## 16. Known Limitations & Technical Discussion

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

## 17. Final Submission Checklist Verification

- [x] **GitHub repository structure**: Clean layout with `data/`, `notebooks/`, `src/`, `models/`, `outputs/`, `tests/`, `api/`, `public/`, `vercel.json`, `requirements.txt`, `Dockerfile`, and `README.md`.
- [x] **Saved serialized model file**: `models/spam_classifier.joblib` (TF-IDF + MultinomialNB pipeline).
- [x] **Metrics and confusion-matrix images**: Saved to `outputs/confusion_matrix.png`, `outputs/roc_pr_curve.png`, `outputs/model_comparison.png`, and `outputs/adversarial_report.png`.
- [x] **Example predictions CSV with 10+ rows**: `outputs/example_predictions.csv` with 13 scored predictions.
- [x] **Short project summary**: Highlighted in Section 2 with exact measured results.
- [x] **Vercel Web App**: `public/index.html`, `public/app.js`, `public/style.css`, and `api/index.py` ready for Vercel deployment and local preview.
- [x] **Interactive Streamlit Web App**: `src/app.py` for local exploration.
- [x] **FastAPI Microservice**: `src/api.py` and `api/index.py` with Swagger docs.
- [x] **Automated Test Suite**: 16/16 passing tests in `tests/`.

