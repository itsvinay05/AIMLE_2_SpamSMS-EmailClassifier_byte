import json
import re
import math
import joblib
import numpy as np

with open('public/model_export.json', 'r', encoding='utf-8') as f:
    model_data = json.load(f)

vocab = model_data['vocabulary']
priors = model_data['class_log_prior']
token_pattern = re.compile(model_data['token_pattern'], re.UNICODE)

def predict_text(text):
    tokens = token_pattern.findall(text.lower())
    all_terms = list(tokens)
    for i in range(len(tokens) - 1):
        all_terms.append(f"{tokens[i]} {tokens[i+1]}")
    
    tf = {}
    for t in all_terms:
        if t in vocab:
            tf[t] = tf.get(t, 0) + 1
            
    norm_sq = 0.0
    for t, count in tf.items():
        idf = vocab[t][0]
        val = count * idf
        norm_sq += val * val
        
    norm = math.sqrt(norm_sq) if norm_sq > 0 else 1.0
    
    jll_ham = priors[0]
    jll_spam = priors[1]
    
    contributions = []
    for t, count in tf.items():
        idf, diff, ham_lp, spam_lp = vocab[t]
        weight = (count * idf) / norm
        jll_ham += weight * ham_lp
        jll_spam += weight * spam_lp
        contributions.append({
            "term": t,
            "weight": round(weight, 4),
            "spam_signal": round(diff * weight, 4),
            "favors": "Spam" if diff > 0 else "Ham"
        })
        
    max_jll = max(jll_ham, jll_spam)
    sum_exp = math.exp(jll_ham - max_jll) + math.exp(jll_spam - max_jll)
    log_sum = max_jll + math.log(sum_exp)
    
    prob_ham = math.exp(jll_ham - log_sum)
    prob_spam = math.exp(jll_spam - log_sum)
    
    contributions.sort(key=lambda x: abs(x["spam_signal"]), reverse=True)
    return prob_ham, prob_spam, contributions

# Compare with joblib pipeline
sk_model = joblib.load('models/spam_classifier.joblib')

test_sentences = [
    "Congratulations! You have won a prize. Claim now.",
    "Are we still meeting at 5 pm?",
    "URGENT: claim your free reward today!",
    "Please send me the notes from class.",
    "Your account has been selected for a bonus.",
    "I will call you when I reach home."
]

print("=== VERIFYING LIGHTWEIGHT ENGINE VS SCIKIT-LEARN PIPELINE ===")
all_matched = True
for s in test_sentences:
    h, sp, contribs = predict_text(s)
    sk_probs = sk_model.predict_proba([s])[0]
    diff = abs(sp - sk_probs[1])
    status = "OK" if diff < 0.005 else "MISMATCH"
    print(f"[{status}] \"{s[:35]}...\" -> Lightweight: {sp:.4f} | Sklearn: {sk_probs[1]:.4f} (diff: {diff:.6f})")
    if diff >= 0.005:
        all_matched = False

print("\nTop contributions for first spam sample:")
_, _, c = predict_text(test_sentences[0])
for item in c[:5]:
    print(f"  - {item['term']}: signal={item['spam_signal']}, favors={item['favors']}")

print(f"\nEngine Validated: {all_matched}")
