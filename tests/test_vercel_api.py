"""
Automated Integration Tests for Vercel App API & Inference Engine
-----------------------------------------------------------------
Validates that all Vercel serverless endpoints, thresholding logic,
token signal explainability, and metrics conform to specification.
"""

import os
import unittest
from api.index import app, load_model_data, classify_text, MessageRequest, BatchRequest, predict, batch_predict, health, get_metrics

class TestVercelAPI(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.model_data = load_model_data()

    def test_model_loading(self):
        self.assertIsNotNone(self.model_data)
        self.assertEqual(self.model_data["vocab_size"], 11432)
        self.assertIn("class_log_prior", self.model_data)

    def test_health_check(self):
        h = health()
        self.assertEqual(h["status"], "healthy")
        self.assertTrue(h["model_loaded"])

    def test_metrics_endpoint(self):
        m = get_metrics()
        self.assertIn("metrics", m)
        acc = m["metrics"]["accuracy"]
        self.assertGreater(acc, 0.98)
        self.assertGreater(m["metrics"]["precision_spam"], 0.98)
        self.assertGreater(m["metrics"]["recall_spam"], 0.91)

    def test_spam_prediction(self):
        req = MessageRequest(
            text="Congratulations! You have won a guaranteed prize. Claim now.",
            threshold=0.50,
            explain=True
        )
        res = predict(req)
        self.assertEqual(res["predicted_label"], "Spam")
        self.assertGreater(res["confidence"], 0.95)
        self.assertGreater(res["spam_probability"], 0.95)
        self.assertIsNotNone(res["explanation"])
        self.assertGreater(len(res["explanation"]), 0)

    def test_ham_prediction(self):
        req = MessageRequest(
            text="Are we still meeting at 5 pm for study group?",
            threshold=0.50,
            explain=False
        )
        res = predict(req)
        self.assertEqual(res["predicted_label"], "Ham")
        self.assertGreater(res["confidence"], 0.95)
        self.assertGreater(res["ham_probability"], 0.95)

    def test_threshold_sensitivity(self):
        borderline_text = "Limited offer! Click the link to get a discount."
        # Test default threshold (0.50 -> Spam)
        normal_res = predict(MessageRequest(text=borderline_text, threshold=0.50))
        # Test ultra-conservative high threshold (0.95 -> Ham)
        high_res = predict(MessageRequest(text=borderline_text, threshold=0.95))
        self.assertEqual(normal_res["predicted_label"], "Spam")
        self.assertEqual(high_res["predicted_label"], "Ham")

    def test_batch_prediction(self):
        req = BatchRequest(
            messages=[
                "URGENT: claim your free reward today!",
                "Please send me the lecture notes."
            ],
            threshold=0.50
        )
        res = batch_predict(req)
        self.assertEqual(res["total_messages"], 2)
        self.assertEqual(res["spam_count"], 1)
        self.assertEqual(res["ham_count"], 1)
        self.assertEqual(res["results"][0]["predicted_label"], "Spam")
        self.assertEqual(res["results"][1]["predicted_label"], "Ham")

if __name__ == "__main__":
    unittest.main()
