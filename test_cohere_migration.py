"""
FocusSense - Cohere Migration Comprehensive Test Suite
Tests:
1. CohereAIProvider initialization, configuration detection, and model defaults.
2. Graceful fallback to LocalMockAIProvider when COHERE_API_KEY is not set.
3. Graceful fallback when Cohere API encounters timeout, HTTP 500, or network error.
4. Multi-turn conversation history structuring.
5. Study material RAG context injection and grounding.
6. Structured Quiz generation and recall answer evaluation.
7. Assessment question generation and mistake explanation.
8. FocusScore and Focus Intelligence isolation (learning logic untouched).
"""

import os
import unittest
from unittest.mock import patch, MagicMock

from ai_service import (
    CohereAIProvider,
    LocalMockAIProvider,
    FocusSenseAIService,
    ai_service,
)
from learning_engine import compute_unified_focus_score


class TestCohereMigration(unittest.TestCase):
    def setUp(self):
        self.service = FocusSenseAIService()

    def test_01_cohere_provider_configuration(self):
        """Test configuration detection with and without API key."""
        with patch.dict(os.environ, {}, clear=True):
            provider = CohereAIProvider()
            self.assertFalse(provider.is_configured())
            info = provider.get_info()
            self.assertEqual(info["name"], "Cohere AI")
            self.assertEqual(info["model"], "command-a-plus-05-2026")
            self.assertFalse(info["active"])

        with patch.dict(os.environ, {"COHERE_API_KEY": "test_fake_cohere_key_12345"}):
            provider = CohereAIProvider()
            self.assertTrue(provider.is_configured())
            info = provider.get_info()
            self.assertTrue(info["active"])
            self.assertEqual(info["type"], "cloud")

    def test_02_custom_model_env_override(self):
        """Test model override via COHERE_MODEL environment variable."""
        with patch.dict(os.environ, {"COHERE_MODEL": "command-r-plus-08-2024"}):
            provider = CohereAIProvider()
            self.assertEqual(provider.model, "command-r-plus-08-2024")

    def test_03_fallback_when_unconfigured(self):
        """When COHERE_API_KEY is unset, service immediately falls back to LocalMockAIProvider."""
        with patch.dict(os.environ, {}, clear=True):
            service = FocusSenseAIService()
            res = service.generate_chat_reply(
                prompt="Explain the purpose of an operating system kernel.",
                conversation_history=[],
                study_topic="Operating Systems",
                student_name="Alice"
            )
            self.assertTrue(res["success"])
            self.assertEqual(res["provider"], "FocusSense Local AI")
            self.assertTrue(res.get("fallback_used"))
            self.assertIn("reply", res)
            self.assertTrue(len(res["reply"]) > 10)

    def test_04_fallback_on_cohere_timeout(self):
        """When Cohere times out, service immediately falls back to LocalMockAIProvider."""
        with patch.dict(os.environ, {"COHERE_API_KEY": "test_fake_key"}):
            service = FocusSenseAIService()
            
            # Mock Cohere provider to simulate a timeout error
            with patch.object(
                service.cohere_provider,
                "generate_reply",
                return_value={"success": False, "error": "Cohere API request timed out (timeout=5.5s)"}
            ):
                res = service.generate_chat_reply(
                    prompt="What is mutual exclusion in concurrency?",
                    conversation_history=[],
                    study_topic="Concurrency",
                    student_name="Bob"
                )
                self.assertTrue(res["success"], "Service should gracefully succeed via fallback")
                self.assertEqual(res["provider"], "FocusSense Local AI")
                self.assertTrue(res.get("fallback_used"))
                self.assertIn("reply", res)

    def test_05_fallback_on_cohere_http_error(self):
        """When Cohere returns HTTP 500 / 401, service falls back to LocalMockAIProvider."""
        with patch.dict(os.environ, {"COHERE_API_KEY": "test_fake_key"}):
            service = FocusSenseAIService()
            with patch.object(
                service.cohere_provider,
                "generate_reply",
                return_value={"success": False, "error": "HTTP 500: Internal Server Error"}
            ):
                res = service.generate_chat_reply(
                    prompt="Define deadlock.",
                    conversation_history=[],
                    study_topic="Operating Systems"
                )
                self.assertTrue(res["success"])
                self.assertEqual(res["provider"], "FocusSense Local AI")
                self.assertTrue(res.get("fallback_used"))

    def test_06_cohere_successful_chat_reply(self):
        """When Cohere is configured and succeeds, returns response from Cohere AI."""
        with patch.dict(os.environ, {"COHERE_API_KEY": "test_fake_key"}):
            service = FocusSenseAIService()
            mock_cohere_reply = {
                "success": True,
                "reply": "A kernel is the core component of an operating system, bridging hardware and software.",
                "provider": "Cohere AI",
                "model": "command-a-plus-05-2026"
            }
            with patch.object(service.cohere_provider, "generate_reply", return_value=mock_cohere_reply):
                res = service.generate_chat_reply(
                    prompt="What is a kernel?",
                    conversation_history=[],
                    study_topic="Operating Systems"
                )
                self.assertTrue(res["success"])
                self.assertEqual(res["provider"], "Cohere AI")
                self.assertEqual(res["model"], "command-a-plus-05-2026")
                self.assertIn("kernel", res["reply"].lower())

    def test_07_rag_study_material_grounding(self):
        """Test RAG study material context extraction in prompt formatting."""
        provider = CohereAIProvider()
        material_context = {
            "context_text": "Virtual memory is a memory management capability of an OS that uses hardware and software to allow a computer to compensate for physical memory shortages.",
            "source_count": 1
        }
        messages = provider._format_messages(
            prompt="How does virtual memory work?",
            conversation_history=[
                {"sender": "user", "text": "Hello"},
                {"sender": "bot", "text": "Hi! What topic are we reviewing today?"}
            ],
            study_topic="Operating Systems",
            student_name="Charlie",
            material_context=material_context
        )
        # Verify messages structure
        self.assertTrue(len(messages) >= 3)
        # Verify system prompt has grounding instructions
        system_content = messages[0]["content"]
        self.assertIn("STUDY MATERIAL / REFERENCE CONTEXT", system_content)
        self.assertIn("Virtual memory is a memory management capability", system_content)
        # Verify user prompt is present
        self.assertEqual(messages[-1]["role"], "user")
        self.assertEqual(messages[-1]["content"], "How does virtual memory work?")

    def test_08_quiz_generation_and_fallback(self):
        """Test quiz generation with fallback and mock."""
        # 1. Fallback when unconfigured
        with patch.dict(os.environ, {}, clear=True):
            service = FocusSenseAIService()
            questions = service.generate_quiz(topic="Process Scheduling", num_questions=3)
            self.assertIsInstance(questions, list)
            self.assertEqual(len(questions), 3)
            self.assertIn("question_text", questions[0])

        # 2. Cohere success
        with patch.dict(os.environ, {"COHERE_API_KEY": "test_fake_key"}):
            service = FocusSenseAIService()
            mock_questions = [
                {
                    "question_index": 1,
                    "question_text": "What is preemptive scheduling?",
                    "sample_answer": "OS interrupts running process.",
                    "rubric": "Process interruption",
                    "sub_concept": "Preemption",
                    "points_possible": 1.0
                }
            ]
            with patch.object(service.cohere_provider, "generate_quiz", return_value=mock_questions):
                q = service.generate_quiz(topic="Process Scheduling")
                self.assertIsInstance(q, list)
                self.assertEqual(len(q), 1)
                self.assertEqual(q[0]["sub_concept"], "Preemption")

    def test_09_answer_evaluation(self):
        """Test recall answer semantic grading."""
        # Fallback evaluation
        eval_res = self.service.evaluate_answer(
            question="What is a semaphore?",
            student_answer="A semaphore is a synchronization tool used to control access to shared resources.",
            sample_answer="A variable used to control access to common resources in concurrent programming.",
            rubric="Synchronization mechanism"
        )
        self.assertTrue(eval_res["success"])
        self.assertIn(eval_res["verdict"], ["correct", "partially_correct", "incorrect"])
        self.assertIn("feedback", eval_res)
        self.assertTrue(isinstance(eval_res["score"], (int, float)))

    def test_10_assessment_questions_and_mistake_explanation(self):
        """Test assessment generation and mistake explanations."""
        # Assessment questions
        questions = self.service.generate_assessment_questions(
            topic="Data Structures",
            num_questions=2
        )
        self.assertIsInstance(questions, list)
        self.assertGreaterEqual(len(questions), 1)

        # Mistake explanation
        explanation = self.service.explain_assessment_mistake(
            question_text="What is the average time complexity of searching a hash table?",
            student_answer="O(n)",
            correct_answer="O(1)",
            rubric="Direct hashing allows constant time lookup.",
            sub_concept="Hash Table Lookup Complexity",
            topic="Data Structures"
        )
        self.assertIsInstance(explanation, str)
        self.assertGreater(len(explanation), 15)

    def test_11_focus_intelligence_isolation(self):
        """Verify that learning analytics and FocusScore calculation remain completely intact."""
        metrics = {
            "presence_pct": 100.0,
            "retention_pct": 100.0,
            "ai_engagement_pct": 100.0,
            "quiz_score_pct": 100.0,
            "duration_sec": 1800,
            "completion_pct": 100.0,
            "consistency_pct": 100.0,
            "absence_count": 0,
            "longest_absence_sec": 0,
        }
        res = compute_unified_focus_score(metrics)
        self.assertEqual(res["unified_focus_score"], 100)
        self.assertEqual(res["quality_tier"].upper(), "EXCELLENT")
        self.assertFalse(res["is_passive_alert"])
        self.assertIn("factors", res)


if __name__ == "__main__":
    unittest.main()
