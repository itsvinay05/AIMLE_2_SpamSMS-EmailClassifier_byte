"""
Automated Comprehensive Test Suite for Spam Classifier
------------------------------------------------------
Verifies model inference, probability calibration, edge cases,
email parsing, adversarial transformations, and API contract logic.
"""

import os
import json
import unittest
import joblib
import numpy as np

from src.email_parser import parse_email_message, clean_html
from src.api import predict_single, predict_batch, MessageRequest, BatchMessageRequest, health_check
from src.adversarial_eval import attack_leetspeak, attack_char_spacing, attack_zero_width_chars


class TestSpamClassifierSuite(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.model_path = os.path.join("models", "spam_classifier.joblib")
        cls.metrics_path = os.path.join("outputs", "metrics.json")
        cls.example_csv_path = os.path.join("outputs", "example_predictions.csv")

        if not os.path.exists(cls.model_path):
            raise FileNotFoundError(
                f"Trained model not found at {cls.model_path}. Run 'python src/train.py' first."
            )
        cls.model = joblib.load(cls.model_path)

    def test_01_model_loading_and_pipeline_structure(self):
        """Verify pipeline contains tfidf vectorizer and classifier steps."""
        self.assertIsNotNone(self.model)
        self.assertIn("tfidf", self.model.named_steps)
        self.assertIn("clf", self.model.named_steps)

    def test_02_predict_and_proba_contract(self):
        """Verify output shape and valid probability distribution."""
        sample = ["Hello, this is a test message."]
        pred = self.model.predict(sample)
        self.assertEqual(len(pred), 1)
        self.assertIn(pred[0], [0, 1])

        probas = self.model.predict_proba(sample)
        self.assertEqual(probas.shape, (1, 2))
        self.assertAlmostEqual(float(np.sum(probas)), 1.0, places=5)
        self.assertTrue(0.0 <= probas[0][0] <= 1.0)
        self.assertTrue(0.0 <= probas[0][1] <= 1.0)

    def test_03_benchmark_spam_messages(self):
        """Verify standard spam messages receive Spam prediction with high confidence."""
        spam_samples = [
            "Congratulations! You have won a prize. Claim now.",
            "URGENT: claim your free reward today!",
            "WINNER!! As a valued customer you have been selected for a £1000 prize reward! Call now.",
            "Limited offer! Click the link to get a discount."
        ]
        for msg in spam_samples:
            pred = self.model.predict([msg])[0]
            proba = self.model.predict_proba([msg])[0][1]  # spam prob
            self.assertEqual(pred, 1, f"Failed to identify spam: '{msg}'")
            self.assertGreater(proba, 0.85, f"Spam confidence too low ({proba:.2f}) for: '{msg}'")

    def test_04_benchmark_ham_messages(self):
        """Verify standard legitimate messages receive Ham prediction with high confidence."""
        ham_samples = [
            "Are we still meeting at 5 pm?",
            "Please send me the notes from class.",
            "I will call you when I reach home.",
            "Can you share the assignment file?",
            "Thanks, see you tomorrow."
        ]
        for msg in ham_samples:
            pred = self.model.predict([msg])[0]
            proba = self.model.predict_proba([msg])[0][0]  # ham prob
            self.assertEqual(pred, 0, f"False positive spam on ham message: '{msg}'")
            self.assertGreater(proba, 0.85, f"Ham confidence too low ({proba:.2f}) for: '{msg}'")

    def test_05_edge_cases_robustness(self):
        """Verify pipeline handles edge cases without crashing."""
        edge_cases = [
            "",                        # empty string
            "   ",                     # whitespace only
            "??? !!! $$",              # punctuation / symbols only
            "1234567890",              # digits only
            "a" * 1000                 # very long string
        ]
        for ec in edge_cases:
            try:
                pred = self.model.predict([ec])
                proba = self.model.predict_proba([ec])
                self.assertIn(pred[0], [0, 1])
                self.assertAlmostEqual(float(np.sum(proba)), 1.0, places=4)
            except Exception as e:
                self.fail(f"Pipeline raised exception on edge case '{ec[:20]}...': {e}")

    def test_06_email_parser_html_and_headers(self):
        """Verify email parsing extracts headers and removes HTML markup cleanly."""
        raw_email = (
            "From: security@paypal-alerts.net\n"
            "Subject: Urgent: Verify Account\n"
            "Content-Type: text/html\n\n"
            "<html><body><h1>Security Notice</h1><p>Click <a href='http://192.168.1.1/login'>here</a> to verify.</p></body></html>"
        )
        parsed = parse_email_message(raw_email)
        self.assertEqual(parsed["subject"], "Urgent: Verify Account")
        self.assertIn("Security Notice", parsed["body"])
        self.assertNotIn("<html>", parsed["body"])
        self.assertTrue(parsed["has_ip_url"])
        self.assertEqual(len(parsed["extracted_urls"]), 1)

    def test_07_adversarial_perturbation_generators(self):
        """Verify adversarial attack generators transform text correctly."""
        base_text = "Claim your free reward prize now"
        leet = attack_leetspeak(base_text, intensity=1.0)
        self.assertNotEqual(base_text, leet)

        spaced = attack_char_spacing("CLAIM your FREE prize")
        self.assertIn("C L A I M", spaced)
        self.assertIn("F R E E", spaced)

        zw = attack_zero_width_chars(base_text)
        self.assertIn("\u200b", zw)

    def test_08_api_handler_endpoints(self):
        """Verify FastAPI handler logic for single and batch predictions."""
        # Health check
        h = health_check()
        self.assertEqual(h["status"], "healthy")

        # Single prediction
        req = MessageRequest(text="Congratulations! You won a $1000 prize. Claim now.", explain=True)
        resp = predict_single(req)
        self.assertEqual(resp.predicted_label, "Spam")
        self.assertGreater(resp.spam_probability, 0.90)
        self.assertIsNotNone(resp.explanation)

        # Batch prediction
        batch_req = BatchMessageRequest(messages=[
            "URGENT: Claim your reward today!",
            "Hey, let's have lunch tomorrow."
        ])
        batch_resp = predict_batch(batch_req)
        self.assertEqual(batch_resp.total_messages, 2)
        self.assertEqual(batch_resp.spam_count, 1)
        self.assertEqual(batch_resp.ham_count, 1)

    def test_09_artifacts_and_metrics_file_integrity(self):
        """Verify all project output files and audit reports exist and are non-empty."""
        expected_artifacts = [
            os.path.join("outputs", "confusion_matrix.png"),
            os.path.join("outputs", "roc_pr_curve.png"),
            os.path.join("outputs", "model_comparison.png"),
            os.path.join("outputs", "metrics.json"),
            os.path.join("outputs", "example_predictions.csv"),
            os.path.join("outputs", "adversarial_report.json"),
            os.path.join("outputs", "adversarial_report.png")
        ]
        for art in expected_artifacts:
            self.assertTrue(os.path.exists(art), f"Missing artifact: {art}")
            self.assertGreater(os.path.getsize(art), 0, f"Empty artifact: {art}")


if __name__ == "__main__":
    unittest.main()
