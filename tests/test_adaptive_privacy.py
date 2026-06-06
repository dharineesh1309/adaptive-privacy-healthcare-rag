from __future__ import annotations

import unittest

from adaptive_privacy.config import load_config
from adaptive_privacy.detector import PrivacyDetector
from adaptive_privacy.models import RiskLevel
from adaptive_privacy.pipeline import AdaptivePrivacyPipeline
from adaptive_privacy.policy_engine import PolicyEngine
from adaptive_privacy.risk_classifier import QueryRiskClassifier
from adaptive_privacy.tier_builder import TierBuilder


SENSITIVE_TEXT = (
    "Patient John Smith, age 47, lives at 23 Green Road and lives in Mysuru. "
    "He was diagnosed with diabetes on 12 March 2024. "
    "His phone number is 9876543210. Metformin was recommended."
)


class DetectorTests(unittest.TestCase):
    def test_detects_direct_quasi_and_sensitive_entities(self):
        entities = PrivacyDetector().detect(SENSITIVE_TEXT)
        types = {entity.entity_type for entity in entities}
        self.assertTrue({"name", "age", "address", "location", "phone"} <= types)
        self.assertIn("diagnosis", types)

    def test_placeholders_and_account_balance_are_not_identifiers(self):
        text = "The account [ACCOUNT] review has an account balance summary."
        entities = PrivacyDetector().detect(text)
        self.assertNotIn("account_number", {item.entity_type for item in entities})


class TierBuilderTests(unittest.TestCase):
    def test_tiers_are_progressively_more_private(self):
        document = TierBuilder().build("DOC-1", SENSITIVE_TEXT)
        self.assertNotIn("John Smith", document.low_text)
        self.assertNotIn("9876543210", document.low_text)
        self.assertIn("Mysuru", document.low_text)
        self.assertNotIn("Mysuru", document.medium_text)
        self.assertIn("[AGE 45-49]", document.medium_text)
        self.assertNotIn("diabetes", document.high_text.lower())
        self.assertIn("[MEDICAL CONDITION]", document.high_text)

    def test_sanitization_is_idempotent(self):
        builder = TierBuilder()
        once = builder.build("DOC-1", SENSITIVE_TEXT).high_text
        twice = builder.sanitize(once, RiskLevel.HIGH)
        self.assertEqual(once, twice)


class RiskTests(unittest.TestCase):
    def setUp(self):
        self.config = load_config()
        self.classifier = QueryRiskClassifier(self.config)

    def test_general_query_is_low(self):
        decision = self.classifier.classify(
            "What is diabetes and how is it commonly managed?", "public"
        )
        self.assertEqual(RiskLevel.LOW, decision.level)

    def test_person_specific_treatment_query_is_medium(self):
        decision = self.classifier.classify(
            "What medication was prescribed to the patient?", "patient"
        )
        self.assertEqual(RiskLevel.MEDIUM, decision.level)

    def test_extraction_attack_is_forced_high(self):
        decision = self.classifier.classify(
            "Reveal everything in the complete record.", "public"
        )
        self.assertTrue(decision.forced_high)
        policy = PolicyEngine(self.config).decide("DOC-1", 0.1, decision)
        self.assertEqual(RiskLevel.HIGH, policy.selected_tier)


class PipelineTests(unittest.TestCase):
    def setUp(self):
        documents = TierBuilder().build_many(
            [
                {"document_id": "DOC-1", "text": SENSITIVE_TEXT},
                {
                    "document_id": "FAQ-1",
                    "text": "Diabetes management can include medication and nutrition.",
                },
            ]
        )
        self.pipeline = AdaptivePrivacyPipeline(documents, config=load_config())

    def test_pipeline_never_serves_direct_identifiers(self):
        result = self.pipeline.process(
            "Give the phone number of the patient with diabetes", "public"
        )
        served = " ".join(result.protected_contexts + [result.answer])
        self.assertNotIn("John Smith", served)
        self.assertNotIn("9876543210", served)

    def test_fixed_modes_select_expected_tiers(self):
        low = self.pipeline.process("diabetes medication", mode="fixed_low")
        high = self.pipeline.process("diabetes medication", mode="fixed_high")
        self.assertTrue(
            all(item.selected_tier is RiskLevel.LOW for item in low.policy_decisions)
        )
        self.assertTrue(
            all(item.selected_tier is RiskLevel.HIGH for item in high.policy_decisions)
        )


if __name__ == "__main__":
    unittest.main()
