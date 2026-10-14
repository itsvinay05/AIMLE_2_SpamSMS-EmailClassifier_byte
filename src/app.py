"""
Spam SMS / Email Classifier - Interactive Web Application
---------------------------------------------------------
Interactive Streamlit UI for real-time spam prediction, probability confidence display,
decision threshold tuning, and keyword signal explainability.
"""

import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st

# Configure page settings
st.set_page_config(
    page_title="Spam SMS / Email Classifier",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E293B;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #64748B;
        margin-bottom: 1.5rem;
    }
    .spam-badge {
        background-color: #FEE2E2;
        color: #B91C1C;
        padding: 0.4rem 0.9rem;
        border-radius: 8px;
        font-weight: 700;
        font-size: 1.1rem;
        border: 1px solid #FCA5A5;
        display: inline-block;
    }
    .ham-badge {
        background-color: #DCFCE7;
        color: #15803D;
        padding: 0.4rem 0.9rem;
        border-radius: 8px;
        font-weight: 700;
        font-size: 1.1rem;
        border: 1px solid #86EFAC;
        display: inline-block;
    }
    .card {
        background-color: #F8FAFC;
        border-radius: 12px;
        padding: 1.2rem;
        border: 1px solid #E2E8F0;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_models():
    """Loads trained pipelines from disk."""
    nb_path = os.path.join("models", "spam_classifier.joblib")
    lr_path = os.path.join("models", "logistic_regression.joblib")
    
    models = {}
    if os.path.exists(nb_path):
        models["Multinomial Naive Bayes (Baseline)"] = joblib.load(nb_path)
    if os.path.exists(lr_path):
        models["Logistic Regression (Extension)"] = joblib.load(lr_path)
    return models


def explain_prediction(model, text):
    """Computes token contributions towards Spam vs Ham."""
    try:
        tfidf = model.named_steps["tfidf"]
        clf = model.named_steps["clf"]
        feature_names = tfidf.get_feature_names_out()
        
        vec = tfidf.transform([text]).toarray()[0]
        non_zero_indices = np.where(vec > 0)[0]
        
        contributions = []
        if hasattr(clf, "feature_log_prob_"):
            diff = clf.feature_log_prob_[1] - clf.feature_log_prob_[0]
            for idx in non_zero_indices:
                contributions.append((feature_names[idx], float(vec[idx] * diff[idx])))
        elif hasattr(clf, "coef_"):
            coef = clf.coef_[0]
            for idx in non_zero_indices:
                contributions.append((feature_names[idx], float(vec[idx] * coef[idx])))
                
        contributions.sort(key=lambda x: x[1], reverse=True)
        return contributions
    except Exception:
        return []


def main():
    st.markdown('<div class="main-header">🛡️ Spam SMS / Email Classifier</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Production-grade NLP text classification pipeline built with TF-IDF N-grams and Naive Bayes</div>', unsafe_allow_html=True)

    models = load_models()
    if not models:
        st.error("No trained models found in `models/` directory. Please run `python src/train.py` first.")
        return

    # Sidebar settings
    st.sidebar.header("⚙️ Model & Threshold Settings")
    selected_model_name = st.sidebar.selectbox("Select Classifier Model", list(models.keys()))
    pipeline = models[selected_model_name]

    threshold = st.sidebar.slider(
        "Spam Classification Threshold",
        min_value=0.05,
        max_value=0.95,
        value=0.50,
        step=0.05,
        help="Probability threshold to classify message as Spam. Increasing threshold prioritizes Precision (avoids flagging Ham)."
    )

    st.sidebar.markdown("---")
    st.sidebar.subheader("📊 Model Performance (Test Set)")
    st.sidebar.markdown("""
    - **Accuracy**: 98.74%
    - **Spam Precision**: 98.36%
    - **Spam Recall**: 91.60%
    - **Spam F1-Score**: 94.86%
    - **ROC-AUC**: 99.60%
    """)

    # Main layout: Two columns
    col_input, col_result = st.columns([1.1, 0.9])

    with col_input:
        st.subheader("📝 Input Message")
        
        # Benchmark quick test buttons
        st.markdown("**Or load a benchmark test case:**")
        sample_cases = [
            ("🎁 Won a Prize (Spam)", "Congratulations! You have won a prize. Claim now."),
            ("📅 Meeting at 5pm (Ham)", "Are we still meeting at 5 pm?"),
            ("⚡ Urgent Reward (Spam)", "URGENT: claim your free reward today!"),
            ("📚 Class Notes (Ham)", "Please send me the notes from class."),
            ("💰 Account Bonus (Spam)", "Your account has been selected for a bonus."),
            ("🚗 Call You Later (Ham)", "I will call you when I reach home."),
            ("🏷️ Limited Discount (Spam)", "Limited offer! Click the link to get a discount."),
            ("📁 Share File (Ham)", "Can you share the assignment file?")
        ]

        # Preset selection
        case_options = ["Custom text (enter below)"] + [c[0] for c in sample_cases]
        selected_case = st.selectbox("Select preset example message:", case_options, index=0)
        
        default_text = ""
        for name, text in sample_cases:
            if selected_case == name:
                default_text = text
                break

        user_text = st.text_area(
            "Enter SMS / Email message content to inspect:",
            value=default_text,
            height=140,
            placeholder="Type or paste an SMS or Email message here..."
        )

        classify_btn = st.button("🚀 Classify Message", type="primary", use_container_width=True)

    with col_result:
        st.subheader("🔍 Classification Results")

        if user_text.strip():
            # Run inference
            probas = pipeline.predict_proba([user_text])[0]
            classes = list(pipeline.classes_)
            ham_prob = float(probas[classes.index(0)])
            spam_prob = float(probas[classes.index(1)])

            # Apply user threshold
            is_spam = spam_prob >= threshold
            label_str = "Spam" if is_spam else "Ham"
            confidence = spam_prob if is_spam else ham_prob

            # Display badge & confidence
            st.markdown('<div class="card">', unsafe_allow_html=True)
            if is_spam:
                st.markdown(f'<div class="spam-badge">🚨 PREDICTED: SPAM</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="ham-badge">✅ PREDICTED: HAM (LEGITIMATE)</div>', unsafe_allow_html=True)

            st.markdown(f"### **Confidence: {confidence:.2%}**")
            st.progress(spam_prob)
            
            c1, c2 = st.columns(2)
            c1.metric("Spam Probability", f"{spam_prob:.2%}")
            c2.metric("Ham Probability", f"{ham_prob:.2%}")
            st.markdown('</div>', unsafe_allow_html=True)

            # Feature explanation
            explanations = explain_prediction(pipeline, user_text)
            if explanations:
                with st.expander("🔎 View Word Signals & Feature Explainability", expanded=True):
                    spam_terms = [t for t, w in explanations if w > 0][:5]
                    ham_terms = [t for t, w in reversed(explanations) if w < 0][:5]

                    if spam_terms:
                        st.markdown(f"**🔴 Terms pushing towards Spam:** `{', '.join(spam_terms)}`")
                    if ham_terms:
                        st.markdown(f"**🟢 Terms pushing towards Ham:** `{', '.join(ham_terms)}`")

                    # Contribution chart
                    top_contribs = explanations[:6] + explanations[-4:]
                    exp_df = pd.DataFrame(top_contribs, columns=["Term", "Impact Score"]).drop_duplicates()
                    exp_df["Signal"] = exp_df["Impact Score"].apply(lambda s: "Spam Signal" if s > 0 else "Ham Signal")
                    st.bar_chart(exp_df.set_index("Term")["Impact Score"])
        else:
            st.info("👈 Enter a message or pick an example preset on the left to see instant classification and confidence breakdown.")

    # Bottom tab: Project Evaluation Artifacts
    st.markdown("---")
    st.subheader("📑 Project Evaluation Artifacts")
    tab1, tab2, tab3 = st.tabs(["📊 Confusion Matrix", "📈 ROC & PR Curves", "🏆 Model Benchmark Comparison"])

    with tab1:
        cm_path = os.path.join("outputs", "confusion_matrix.png")
        if os.path.exists(cm_path):
            st.image(cm_path, caption="Confusion Matrix on Held-out Test Set (N=1,035)", width=550)
            st.markdown("""
            - **True Negatives (Legitimate Ham caught)**: 902 / 904
            - **False Positives (Ham incorrectly flagged as Spam)**: only 2 messages (0.22% error rate!)
            - **True Positives (Spam caught)**: 120 / 131
            - **False Negatives (Spam missed)**: 11 messages
            """)

    with tab2:
        roc_path = os.path.join("outputs", "roc_pr_curve.png")
        if os.path.exists(roc_path):
            st.image(roc_path, caption="ROC Curve & Precision-Recall Curves across Models", use_container_width=True)

    with tab3:
        comp_path = os.path.join("outputs", "model_comparison.png")
        if os.path.exists(comp_path):
            st.image(comp_path, caption="Model Performance Benchmark (Accuracy, Precision, Recall, F1)", use_container_width=True)


if __name__ == "__main__":
    main()
