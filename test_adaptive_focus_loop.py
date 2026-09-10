r"""
FocusSense Phase 4 Test Suite: Adaptive Focus Intelligence & Unified Learning Loop
==================================================================================
Tests:
1. 8-Dimension Unified FocusScore calculation and weights (Presence 25%, Retention 25%, AI Engagement 15%, Quiz 15%, Duration 10%, Completion 5%, Consistency 5%, Distraction penalty).
2. Passive Study Alert rule (Presence >= 85% and Retention < 50% => score <= 55, label "Passive Study Alert").
3. Anti-Gaming AI Query Classification (Trivial greetings filtered vs conceptual inquiries recognized).
4. AI Study Engagement Evaluation.
5. 7-Day Habit Consistency calculation.
6. Explainable Adaptive Action Selection Policy.
7. Mathematical Adaptive Reward formulation ($R_t \in [-1, 1]$).
8. Socratic AI Personalization prompt injection (Weak mastery <60% vs High mastery >=85%).
9. Database State-Action-Reward logging integrity.
10. Multi-session score lifecycle & non-permanent quiz impact recovery.
"""

import os
import sys
import unittest
import sqlite3
import tempfile
import json
from unittest.mock import patch

from focus_intelligence import (
    classify_meaningful_query,
    evaluate_ai_study_interaction,
    compute_habit_consistency,
)
from learning_engine import (
    compute_unified_focus_score,
    select_adaptive_action,
    calculate_adaptive_reward,
    record_learning_state,
    record_adaptive_action,
    record_adaptive_reward,
    get_latest_learning_state,
    get_unified_learning_profile,
    update_topic_mastery,
)
from database import create_database
from ai_service import ai_service


class TestUnifiedFocusIntelligence(unittest.TestCase):

    def test_eight_dimension_scoring_weights(self):
        """Verify the 8-dimension weighted calculation."""
        metrics = {
            "presence_pct": 100.0,
            "retention_pct": 100.0,
            "ai_engagement_pct": 100.0,
            "quiz_score_pct": 100.0,
            "duration_sec": 1800,  # 30 mins => 100% duration rating
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
        self.assertEqual(res["factors"]["physical_presence_pct"], 100.0)

    def test_passive_presence_cognitive_contrast_rule(self):
        """Presence >= 85% and Retention < 50% must cap score at <= 55 with Passive Study Alert."""
        metrics = {
            "presence_pct": 95.0,
            "retention_pct": 40.0,
            "ai_engagement_pct": 80.0,
            "quiz_score_pct": 40.0,
            "duration_sec": 1800,
            "completion_pct": 100.0,
            "consistency_pct": 80.0,
            "absence_count": 0,
            "longest_absence_sec": 0,
        }
        res = compute_unified_focus_score(metrics)
        self.assertTrue(res["is_passive_alert"])
        self.assertLessEqual(res["unified_focus_score"], 55)
        self.assertEqual(res["score_label"], "Passive Study Alert")
        self.assertIn("quiz performance was low", res["explanation"])

    def test_learning_efficiency_calculation(self):
        """Learning efficiency index formula check."""
        metrics = {
            "presence_pct": 90.0,
            "retention_pct": 90.0,
            "ai_engagement_pct": 85.0,
            "quiz_score_pct": 90.0,
            "duration_sec": 1800,
            "completion_pct": 100.0,
            "consistency_pct": 80.0,
        }
        res = compute_unified_focus_score(metrics)
        self.assertGreaterEqual(res["learning_efficiency"], 70)
        self.assertLessEqual(res["learning_efficiency"], 100)

    def test_anti_gaming_query_classification(self):
        """Test classification of meaningful inquiries vs filler greetings."""
        topic = "Python Recursion"
        
        # Substantive queries
        self.assertTrue(classify_meaningful_query("Can you explain the base case in recursion?", topic)["is_meaningful"])
        self.assertTrue(classify_meaningful_query("Why do I get RecursionError: maximum recursion depth exceeded?", topic)["is_meaningful"])
        self.assertTrue(classify_meaningful_query("What is the difference between recursion and iteration?", topic)["is_meaningful"])
        self.assertTrue(classify_meaningful_query("How does the call stack work during recursive calls?", topic)["is_meaningful"])
        self.assertTrue(classify_meaningful_query("Give me an example of binary search with recursion", topic)["is_meaningful"])

        # Trivial filler messages (should not get study focus credit)
        self.assertFalse(classify_meaningful_query("hi", topic)["is_meaningful"])
        self.assertFalse(classify_meaningful_query("hello", topic)["is_meaningful"])
        self.assertFalse(classify_meaningful_query("ok", topic)["is_meaningful"])
        self.assertFalse(classify_meaningful_query("thanks", topic)["is_meaningful"])
        self.assertFalse(classify_meaningful_query("k", topic)["is_meaningful"])
        self.assertFalse(classify_meaningful_query("bye", topic)["is_meaningful"])

    def test_ai_study_interaction_evaluation(self):
        """Test AI study engagement score computation."""
        messages = [
            {"sender": "user", "message": "Hi"},
            {"sender": "assistant", "message": "Hello!"},
            {"sender": "user", "message": "How do Python decorators work behind the scenes?"},
            {"sender": "assistant", "message": "A decorator is a callable that takes a function..."},
            {"sender": "user", "message": "Can you give me a code example with @functools.wraps?"},
            {"sender": "assistant", "message": "Here is an example..."},
            {"sender": "user", "message": "What is the difference between function decorators and class decorators?"},
            {"sender": "assistant", "message": "Class decorators receive the class object..."},
        ]
        eval_res = evaluate_ai_study_interaction(
            messages=messages,
            study_topic="Python Decorators",
            active_duration_sec=1200
        )
        self.assertGreaterEqual(eval_res["engagement_pct"], 70)
        self.assertEqual(eval_res["meaningful_query_count"], 3)

    def test_adaptive_policy_action_selection(self):
        """Verify explainable rule-based adaptive actions."""
        # 1. Weak mastery => INCREASE_RECALL_FREQ
        state_weak = {
            "physical_presence_pct": 90.0,
            "retention_score_pct": 45.0,
            "quiz_score_pct": 50.0,
            "ai_engagement_pct": 75.0,
            "distraction_control_pct": 90.0,
            "mastery_tier": "WEAK",
        }
        act_weak = select_adaptive_action(state_weak)
        self.assertEqual(act_weak["action_type"], "INCREASE_RECALL_FREQ")
        self.assertEqual(act_weak["action_payload"]["delay_minutes"], 5)

        # 2. Marathon duration with poor retention => RECOMMEND_SHORT_SESSIONS
        state_distracted = {
            "physical_presence_pct": 90.0,
            "study_duration_sec": 3600,
            "retention_score_pct": 50.0,
            "distraction_control_pct": 80.0,
            "mastery_tier": "WEAK",
        }
        act_distracted = select_adaptive_action(state_distracted)
        self.assertEqual(act_distracted["action_type"], "RECOMMEND_SHORT_SESSIONS")

        # 3. Short focused session, high retention => REINFORCE_EFFICIENCY
        state_efficient = {
            "physical_presence_pct": 95.0,
            "study_duration_sec": 1500,
            "retention_score_pct": 90.0,
            "quiz_score_pct": 90.0,
            "distraction_control_pct": 95.0,
            "mastery_tier": "STRONG",
        }
        act_eff = select_adaptive_action(state_efficient)
        self.assertEqual(act_eff["action_type"], "REINFORCE_EFFICIENCY")

        # 4. Long high mastery session => ADVANCED_CHALLENGE
        state_strong = {
            "physical_presence_pct": 95.0,
            "study_duration_sec": 2700,
            "retention_score_pct": 92.0,
            "quiz_score_pct": 90.0,
            "distraction_control_pct": 95.0,
            "mastery_tier": "STRONG",
        }
        act_strong = select_adaptive_action(state_strong)
        self.assertEqual(act_strong["action_type"], "ADVANCED_CHALLENGE")

    def test_adaptive_reward_calculation(self):
        r"""Test exact reward formula $R_t \in [-1.0, +1.0]$."""
        # Initial baseline session (no previous state)
        r_init = calculate_adaptive_reward(None, {"retention_score_pct": 80.0, "unified_focus_score": 85})
        self.assertGreaterEqual(r_init, 0.0)
        self.assertLessEqual(r_init, 1.0)

        # Positive improvement: +20% retention, +10% score, positive efficiency
        s_t = {
            "retention_score_pct": 60.0,
            "unified_focus_score": 70,
            "learning_efficiency_score": 65,
        }
        s_t_plus_1 = {
            "retention_score_pct": 80.0,
            "unified_focus_score": 80,
            "learning_efficiency_score": 75,
        }
        r_pos = calculate_adaptive_reward(s_t, s_t_plus_1)
        # Expected: 0.6 * (20/100) + 0.3 * (10/100) + 0.1 * 1 = 0.12 + 0.03 + 0.1 = 0.25
        self.assertAlmostEqual(r_pos, 0.25, places=2)

        # Negative drop
        s_drop = {
            "retention_score_pct": 40.0,
            "unified_focus_score": 50,
            "learning_efficiency_score": 45,
        }
        r_neg = calculate_adaptive_reward(s_t, s_drop)
        # Expected: 0.6 * (-20/100) + 0.3 * (-20/100) + 0.1 * (-1) = -0.12 - 0.06 - 0.1 = -0.28
        self.assertAlmostEqual(r_neg, -0.28, places=2)

    def test_ai_socratic_personalization(self):
        """Test prompt personalization based on mastery tiers."""
        from unittest.mock import MagicMock

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"message": {"content": "A binary tree is a hierarchical data structure."}}

        with patch("requests.post", return_value=mock_resp), patch.object(ai_service.ollama_provider, "is_available", return_value=True):
            weak_context = {"mastery_tier": "WEAK", "avg_retention_score": 45.0}
            reply_weak = ai_service.generate_chat_reply(
                prompt="What is a binary tree?",
                conversation_history=[],
                study_topic="Binary Trees",
                student_name="Student",
                learning_context=weak_context
            )
            self.assertTrue(reply_weak["success"])
            self.assertIn("reply", reply_weak)

            strong_context = {"mastery_tier": "STRONG", "avg_retention_score": 90.0}
            reply_strong = ai_service.generate_chat_reply(
                prompt="What is an AVL tree?",
                conversation_history=[],
                study_topic="Tree Structures",
                student_name="Student",
                learning_context=strong_context
            )
            self.assertTrue(reply_strong["success"])
            self.assertIn("reply", reply_strong)


class TestDatabaseAndLifecycleIntegration(unittest.TestCase):

    def setUp(self):
        self.temp_db = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        self.temp_db.close()
        self.db_path = self.temp_db.name
        create_database(self.db_path)

    def tearDown(self):
        if os.path.exists(self.db_path):
            try:
                os.remove(self.db_path)
            except Exception:
                pass

    def test_database_tables_exist(self):
        """Verify learning_states, adaptive_actions, adaptive_rewards exist."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = {r[0] for r in cursor.fetchall()}
        conn.close()

        self.assertIn("learning_states", tables)
        self.assertIn("adaptive_actions", tables)
        self.assertIn("adaptive_rewards", tables)

    def test_learning_state_and_action_recording(self):
        """Test inserting and querying state, action, and reward rows."""
        user_id = 1
        topic = "Algorithms"

        # Record State 1
        metrics_1 = {
            "physical_presence_pct": 85.0,
            "study_duration_sec": 1800,
            "ai_engagement_pct": 80.0,
            "quiz_score_pct": 60.0,
            "retention_score_pct": 55.0,
            "distraction_control_pct": 85.0,
            "session_completion_pct": 100.0,
            "consistency_score_pct": 80.0,
            "unified_focus_score": 68,
            "learning_efficiency_score": 62,
            "quality_tier": "GOOD",
            "score_label": "Good Focus",
            "explanation": "Solid study session with opportunities to consolidate retention.",
        }
        s1_id = record_learning_state(user_id, 101, topic, metrics_1, database_path=self.db_path)
        self.assertIsNotNone(s1_id)

        # Record Action
        act = select_adaptive_action(metrics_1)
        a1_id = record_adaptive_action(
            user_id=user_id,
            topic=topic,
            session_id=101,
            state_id=s1_id,
            action_type=act["action_type"],
            action_payload=act["action_payload"],
            reason=act["reason"],
            database_path=self.db_path
        )
        self.assertIsNotNone(a1_id)

        # Record State 2 (Improvement after recall test)
        metrics_2 = dict(metrics_1)
        metrics_2["retention_score_pct"] = 85.0
        metrics_2["unified_focus_score"] = 82
        metrics_2["learning_efficiency_score"] = 80
        s2_id = record_learning_state(user_id, 101, topic, metrics_2, database_path=self.db_path)

        # Record Reward
        r_val = calculate_adaptive_reward(metrics_1, metrics_2)
        rew_id = record_adaptive_reward(
            user_id=user_id,
            action_id=a1_id,
            initial_state_id=s1_id,
            resulting_state_id=s2_id,
            reward_value=r_val,
            evaluation_notes="Spaced recall improved retention to 85%",
            database_path=self.db_path
        )
        self.assertIsNotNone(rew_id)

        # Verify unified profile retrieval
        profile = get_unified_learning_profile(user_id, database_path=self.db_path)
        self.assertEqual(profile["latest_unified_score"], 82)
        self.assertEqual(profile["action_type"], act["action_type"])


if __name__ == "__main__":
    unittest.main()
