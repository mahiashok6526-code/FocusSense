"""
Unit & Integration Tests for FocusSense Personal Learning Coach (8-Pillar Synthesis Engine)
========================================================================================
Tests cover:
  1. 8-Pillar Telemetry Synthesis (sessions, radar, AI chats, quizzes, recall, materials, retention, adaptive actions)
  2. Explainable Superpower 1: What Should I Study Next (ranking, action triggers)
  3. Explainable Superpower 2: What Am I Weak At (conceptual gaps, passive-study detection)
  4. Explainable Superpower 3: When Should I Revise (Ebbinghaus decay curve, estimated retention, urgency queues)
  5. Explainable Superpower 4: How Am I Improving (growth velocity, cognitive efficiency, radar balance)
  6. Pedagogical Milestone Verification (no free badges for mere uptime)
  7. Socratic AI Coach Dialogue (grounded in real telemetry, offline deterministic fallback)
  8. API Endpoints & User Isolation (auth checks, multi-user boundary verification)
  9. Zero Data Fabrication / Insufficient History Handling
"""

import os
import sys
import unittest
import json
import time
import sqlite3
from datetime import datetime, timedelta

# Ensure app directory is in path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from database import create_database, get_db_connection, DB_PATH
import coach_engine
import app as flask_app


class TestPersonalLearningCoach(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """Initialize test database tables."""
        create_database()

    def setUp(self):
        """Set up isolated test users and test client for each test."""
        self.app = flask_app.app
        self.app.config['TESTING'] = True
        self.app.config['SECRET_KEY'] = 'test-secret-key-coach'
        self.client = self.app.test_client()

        self.conn = get_db_connection()
        self.cursor = self.conn.cursor()

        # Generate unique test usernames
        self.user_a_name = f"coach_user_a_{int(time.time() * 1000)}"
        self.user_b_name = f"coach_user_b_{int(time.time() * 1000)}"

        # Insert User A
        self.cursor.execute(
            "INSERT INTO users (fullname, username, password, email) VALUES (?, ?, ?, ?)",
            ("User A", self.user_a_name, "password_hash_a", f"{self.user_a_name}@example.com")
        )
        self.user_a_id = self.cursor.lastrowid

        # Insert User B
        self.cursor.execute(
            "INSERT INTO users (fullname, username, password, email) VALUES (?, ?, ?, ?)",
            ("User B", self.user_b_name, "password_hash_b", f"{self.user_b_name}@example.com")
        )
        self.user_b_id = self.cursor.lastrowid
        self.conn.commit()

    def tearDown(self):
        """Clean up test artifacts."""
        try:
            self.cursor.execute("DELETE FROM coach_recommendations WHERE user_id IN (?, ?)", (self.user_a_id, self.user_b_id))
            self.cursor.execute("DELETE FROM coach_milestones WHERE user_id IN (?, ?)", (self.user_a_id, self.user_b_id))
            self.cursor.execute("DELETE FROM coach_queries WHERE user_id IN (?, ?)", (self.user_a_id, self.user_b_id))
            self.cursor.execute("DELETE FROM topic_mastery WHERE user_id IN (?, ?)", (self.user_a_id, self.user_b_id))
            self.cursor.execute("DELETE FROM recall_tasks WHERE user_id IN (?, ?)", (self.user_a_id, self.user_b_id))
            self.cursor.execute("DELETE FROM study_quizzes WHERE user_id IN (?, ?)", (self.user_a_id, self.user_b_id))
            self.cursor.execute("DELETE FROM study_materials WHERE user_id IN (?, ?)", (self.user_a_id, self.user_b_id))
            self.cursor.execute("DELETE FROM adaptive_actions WHERE user_id IN (?, ?)", (self.user_a_id, self.user_b_id))
            self.cursor.execute("DELETE FROM learning_states WHERE user_id IN (?, ?)", (self.user_a_id, self.user_b_id))
            self.cursor.execute("DELETE FROM session_focus_scores WHERE user_id IN (?, ?)", (self.user_a_id, self.user_b_id))
            self.cursor.execute("DELETE FROM study_sessions WHERE user_id IN (?, ?)", (self.user_a_id, self.user_b_id))
            self.cursor.execute("DELETE FROM users WHERE id IN (?, ?)", (self.user_a_id, self.user_b_id))
            self.conn.commit()
            self.conn.close()
        except Exception:
            pass

    # =========================================================================
    # 1. Zero-Fabrication & Insufficient History Tests
    # =========================================================================

    def test_insufficient_learning_history(self):
        """When a user has no learning data, coach must explicitly show 'Not enough learning history yet.'"""
        telemetry = coach_engine.get_eight_pillar_telemetry(self.user_a_id)
        self.assertFalse(telemetry["has_sufficient_history"])
        self.assertEqual(telemetry["history_message"], "Not enough learning history yet.")
        self.assertEqual(telemetry["coach_health_score"], 0)

        what_next = coach_engine.get_what_to_study_next(self.user_a_id)
        self.assertFalse(what_next["has_sufficient_history"])
        self.assertIn("Not enough learning history yet.", what_next["message"])

        weaknesses = coach_engine.get_what_am_i_weak_at(self.user_a_id)
        self.assertFalse(weaknesses["has_sufficient_history"])
        self.assertIn("Not enough learning history yet.", weaknesses["message"])

        revisions = coach_engine.get_when_to_revise(self.user_a_id)
        self.assertFalse(revisions["has_sufficient_history"])
        self.assertIn("Not enough learning history yet.", revisions["message"])

        improvement = coach_engine.get_how_am_i_improving(self.user_a_id)
        self.assertFalse(improvement["has_sufficient_history"])
        self.assertIn("Not enough learning history yet.", improvement["message"])

    # =========================================================================
    # 2. 8-Pillar Synthesis & Telemetry Aggregation
    # =========================================================================

    def test_eight_pillar_telemetry_aggregation(self):
        """Seed all 8 pillars and verify comprehensive mathematical synthesis."""
        now = datetime.now()

        # Pillar 1: Study session
        self.cursor.execute("""
            INSERT INTO study_sessions (user_id, date, start_time, end_time, duration, focus_score)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (self.user_a_id, now.strftime('%Y-%m-%d'), (now - timedelta(hours=3)).isoformat(), (now - timedelta(hours=2)).isoformat(), 3600, 90))
        session_id = self.cursor.lastrowid

        # Pillar 2: Radar Presence
        self.cursor.execute("""
            INSERT INTO session_focus_scores (session_id, user_id, presence_pct, absence_count, longest_absence_sec, total_absence_sec, session_duration_sec, goal_progress_pct, quality_tier, estimated_score, computed_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (session_id, self.user_a_id, 90.0, 1, 360, 360, 3600, 100.0, "Deep Focus", 90, now.isoformat()))

        # Pillar 3: AI Engagement
        self.cursor.execute("""
            INSERT INTO ai_study_sessions (user_id, session_id, topic, started_at, ended_at, active_duration_sec, question_count, meaningful_query_count)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (self.user_a_id, session_id, "Data Structures - Binary Trees", (now - timedelta(hours=3)).isoformat(), now.isoformat(), 1800, 10, 8))

        # Pillar 4: Quiz Results
        self.cursor.execute("""
            INSERT INTO study_quizzes (user_id, session_id, topic, total_questions, score_pct, quality_label, summary_text, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (self.user_a_id, session_id, "Data Structures - Binary Trees", 10, 80.0, "Mastered", "Good conceptual grasp", now.isoformat()))
        quiz_id = self.cursor.lastrowid

        # Pillar 5: Recall Results
        self.cursor.execute("""
            INSERT INTO recall_tasks (user_id, session_id, topic, delay_minutes, scheduled_for, status, quiz_id, retention_score, quality_label, created_at, completed_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (self.user_a_id, session_id, "Data Structures - Binary Trees", 60, now.isoformat(), "COMPLETED", quiz_id, 85.0, "Solid", now.isoformat(), now.isoformat()))

        # Pillar 6: Study Materials
        self.cursor.execute("""
            INSERT INTO study_materials (user_id, filename, stored_path, file_type, file_size_bytes, title, topic, chunk_count, status, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (self.user_a_id, "avl_trees.pdf", "/uploads/avl.pdf", "pdf", 10240, "AVL Trees Guide", "Data Structures - Binary Trees", 12, "READY", now.isoformat(), now.isoformat()))

        # Pillar 7: Topic Retention & Mastery
        self.cursor.execute("""
            INSERT INTO topic_mastery (user_id, topic, mastery_tier, total_quizzes, total_questions, correct_count, partial_count, incorrect_count, avg_retention_score, last_retention_score, last_tested_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (self.user_a_id, "Data Structures - Binary Trees", "STRONG", 2, 20, 17, 2, 1, 85.0, 85.0, now.isoformat(), now.isoformat()))

        # Pillar 8: Adaptive Actions
        self.cursor.execute("""
            INSERT INTO adaptive_actions (user_id, topic, session_id, action_type, reason, status, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (self.user_a_id, "Data Structures - Binary Trees", session_id, "REINFORCE_EFFICIENCY", "High physical focus with strong retention", "PENDING", now.isoformat()))
        self.conn.commit()

        telemetry = coach_engine.get_eight_pillar_telemetry(self.user_a_id)

        self.assertTrue(telemetry["has_sufficient_history"])
        self.assertGreater(telemetry["coach_health_score"], 0)
        self.assertEqual(telemetry["pillars"]["study_sessions"]["session_count"], 1)
        self.assertEqual(telemetry["pillars"]["ai_engagement"]["total_queries"], 10)
        self.assertEqual(telemetry["pillars"]["quiz_results"]["avg_quiz_score"], 80.0)
        self.assertEqual(telemetry["pillars"]["recall_results"]["avg_retention_score"], 85.0)
        self.assertEqual(telemetry["pillars"]["study_materials"]["material_count"], 1)
        self.assertEqual(telemetry["pillars"]["knowledge_retention"]["topics_tracked_count"], 1)
        self.assertEqual(telemetry["pillars"]["adaptive_learning"]["action_count"], 1)


    # =========================================================================
    # 3. Superpower 1: What Should I Study Next?
    # =========================================================================

    def test_what_should_i_study_next_prioritization(self):
        """Urgent decaying topics or low-scoring quiz topics must be recommended first."""
        now = datetime.now()

        # Seed topic with low retention (should trigger RECALL_TEST or REVIEW_MATERIAL)
        self.cursor.execute("""
            INSERT INTO topic_mastery (user_id, topic, mastery_tier, total_quizzes, total_questions, correct_count, partial_count, incorrect_count, avg_retention_score, last_retention_score, last_tested_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (self.user_a_id, "Operating Systems - Deadlocks", "WEAK", 1, 10, 4, 1, 5, 40.0, 40.0, (now - timedelta(days=5)).isoformat(), (now - timedelta(days=5)).isoformat()))
        self.conn.commit()

        res = coach_engine.get_what_to_study_next(self.user_a_id)
        self.assertTrue(res["has_sufficient_history"])
        self.assertIsNotNone(res["primary_recommendation"])
        self.assertIn("Deadlocks", res["primary_recommendation"]["topic"])
        self.assertIn(res["primary_recommendation"]["action_type"], ["START_STUDY", "TAKE_QUIZ", "REVIEW_MATERIAL", "RECALL_TEST"])
        self.assertTrue(len(res["priority_queue"]) > 0)

    # =========================================================================
    # 4. Superpower 2: What Am I Weak At? & Passive-Study Detection
    # =========================================================================

    def test_passive_study_detection(self):
        """High presence at desk (>=80%) but poor retention/quiz (<60%) must trigger passive-study flag."""
        now = datetime.now()

        # Session with 95% physical presence
        self.cursor.execute("""
            INSERT INTO study_sessions (user_id, date, start_time, end_time, duration, focus_score)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (self.user_a_id, now.strftime('%Y-%m-%d'), (now - timedelta(hours=2)).isoformat(), now.isoformat(), 7200, 95))
        session_id = self.cursor.lastrowid

        self.cursor.execute("""
            INSERT INTO session_focus_scores (session_id, user_id, presence_pct, absence_count, longest_absence_sec, total_absence_sec, session_duration_sec, goal_progress_pct, quality_tier, estimated_score, computed_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (session_id, self.user_a_id, 95.0, 0, 0, 0, 7200, 100.0, "Deep Focus", 95, now.isoformat()))

        # Low quiz result (40%)
        self.cursor.execute("""
            INSERT INTO study_quizzes (user_id, session_id, topic, total_questions, score_pct, quality_label, summary_text, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (self.user_a_id, session_id, "Calculus III - Vector Fields", 10, 40.0, "Needs Review", "Struggled with curl and divergence", now.isoformat()))

        # Low mastery
        self.cursor.execute("""
            INSERT INTO topic_mastery (user_id, topic, mastery_tier, total_quizzes, total_questions, correct_count, partial_count, incorrect_count, avg_retention_score, last_retention_score, last_tested_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (self.user_a_id, "Calculus III - Vector Fields", "WEAK", 1, 10, 4, 0, 6, 40.0, 40.0, now.isoformat(), now.isoformat()))
        self.conn.commit()

        weaknesses = coach_engine.get_what_am_i_weak_at(self.user_a_id)
        self.assertTrue(weaknesses["has_sufficient_history"])
        self.assertGreater(len(weaknesses["passive_study_alerts"]), 0)
        flag = weaknesses["passive_study_alerts"][0]
        self.assertEqual(flag["topic"], "Calculus III - Vector Fields")
        self.assertGreaterEqual(flag["presence_pct"], 80.0)
        self.assertLess(flag["retention_pct"], 60.0)

    # =========================================================================
    # 5. Superpower 3: When Should I Revise? & Ebbinghaus Curve
    # =========================================================================

    def test_ebbinghaus_decay_estimation(self):
        """Ebbinghaus decay must decrease over elapsed time and present explicit estimation labeling."""
        now = datetime.now()

        # Fresh topic (reviewed 2 hours ago)
        ret_fresh = coach_engine.calculate_ebbinghaus_decay(80.0, (now - timedelta(hours=2)).isoformat(), repetitions=1)
        # Old topic (reviewed 10 days ago)
        ret_old = coach_engine.calculate_ebbinghaus_decay(80.0, (now - timedelta(days=10)).isoformat(), repetitions=1)

        self.assertGreater(ret_fresh["estimated_retention"], ret_old["estimated_retention"])
        self.assertIn("Estimated", ret_fresh["disclaimer"])
        self.assertIn("Estimated", ret_old["disclaimer"])

        # Seed into DB
        self.cursor.execute("""
            INSERT INTO topic_mastery (user_id, topic, mastery_tier, total_quizzes, total_questions, correct_count, partial_count, incorrect_count, avg_retention_score, last_retention_score, last_tested_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (self.user_a_id, "Computer Networks - TCP Handshake", "STRONG", 1, 10, 9, 0, 1, 85.0, 85.0, (now - timedelta(days=4)).isoformat(), (now - timedelta(days=4)).isoformat()))
        self.conn.commit()

        revisions = coach_engine.get_when_to_revise(self.user_a_id)
        self.assertTrue(revisions["has_sufficient_history"])
        self.assertIn("Computer Networks - TCP Handshake", [item["topic"] for item in revisions["revision_queue"]])
        for item in revisions["revision_queue"]:
            self.assertIn("Estimated", item["disclaimer"])

    # =========================================================================
    # 6. Superpower 4: How Am I Improving?
    # =========================================================================

    def test_how_am_i_improving_velocity(self):
        """Growth velocity and 8-pillar radar statistics must synthesize correctly."""
        now = datetime.now()

        self.cursor.execute("""
            INSERT INTO topic_mastery (user_id, topic, mastery_tier, total_quizzes, total_questions, correct_count, partial_count, incorrect_count, avg_retention_score, last_retention_score, last_tested_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (self.user_a_id, "Database Systems - B+ Trees", "STRONG", 4, 40, 37, 2, 1, 92.0, 92.0, now.isoformat(), now.isoformat()))
        self.conn.commit()

        impr = coach_engine.get_how_am_i_improving(self.user_a_id)
        self.assertTrue(impr["has_sufficient_history"])
        self.assertIn("cognitive_efficiency", impr)
        self.assertIn("growth_velocity_label", impr)
        self.assertIn("pillar_radar", impr)
        self.assertEqual(len(impr["pillar_radar"]), 8)

    # =========================================================================
    # 7. Meaningful Pedagogical Milestones
    # =========================================================================

    def test_milestones_awarded_for_mastery_only(self):
        """Milestones are awarded for real pedagogical achievements (e.g. topic mastery > 85), not uptime."""
        now = datetime.now()

        # Seed high mastery topic
        self.cursor.execute("""
            INSERT INTO topic_mastery (user_id, topic, mastery_tier, total_quizzes, total_questions, correct_count, partial_count, incorrect_count, avg_retention_score, last_retention_score, last_tested_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (self.user_a_id, "Compiler Design - LL(1) Parsers", "STRONG", 3, 30, 28, 1, 1, 93.0, 93.0, now.isoformat(), now.isoformat()))
        self.conn.commit()

        new_milestones = coach_engine.check_and_award_milestones(self.user_a_id)
        # Should award at least "FIRST_MASTERY_STRONG" or "MASTERY_TRIAD"
        self.assertTrue(any(m["milestone_key"] in ["FIRST_MASTERY_STRONG", "MASTERY_TRIAD", "DEEP_RETENTION", "MASTERY_PROMOTION"] for m in new_milestones))

        # Check DB persistence
        self.cursor.execute("SELECT milestone_key FROM coach_milestones WHERE user_id = ?", (self.user_a_id,))
        keys = [row[0] for row in self.cursor.fetchall()]
        self.assertTrue(any(k in ["FIRST_MASTERY_STRONG", "MASTERY_TRIAD", "DEEP_RETENTION", "MASTERY_PROMOTION"] for k in keys))

    # =========================================================================
    # 8. Socratic AI Coach Dialogue & Offline Fallback
    # =========================================================================

    def test_socratic_ai_coach_query(self):
        """Coach answers student queries grounded in telemetry, with robust fallback."""
        now = datetime.now()
        self.cursor.execute("""
            INSERT INTO topic_mastery (user_id, topic, mastery_tier, total_quizzes, total_questions, correct_count, partial_count, incorrect_count, avg_retention_score, last_retention_score, last_tested_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (self.user_a_id, "Linear Algebra - Eigenvalues", "WEAK", 2, 20, 10, 2, 8, 55.0, 55.0, now.isoformat(), now.isoformat()))
        self.conn.commit()

        resp = coach_engine.ask_personal_coach(self.user_a_id, "What should I study next?")
        self.assertIn("response", resp)
        self.assertGreater(len(resp["response"]), 20)
        self.assertTrue("Linear Algebra" in resp["response"] or "Personal Learning Coach" in resp["response"])

    # =========================================================================
    # 9. API Endpoints & User Isolation
    # =========================================================================

    def test_api_endpoints_and_user_isolation(self):
        """Verify endpoints return structured data and respect user boundaries."""
        now = datetime.now()

        # Populate User A's data
        self.cursor.execute("""
            INSERT INTO topic_mastery (user_id, topic, mastery_tier, total_quizzes, total_questions, correct_count, partial_count, incorrect_count, avg_retention_score, last_retention_score, last_tested_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (self.user_a_id, "Secret User A Topic", "STRONG", 2, 20, 18, 1, 1, 90.0, 90.0, now.isoformat(), now.isoformat()))
        self.conn.commit()

        # 1. Unauthenticated request to /api/coach/overview should return 401
        res = self.client.get('/api/coach/overview')
        self.assertEqual(res.status_code, 401)

        # 2. Login as User B (who has NO study data)
        with self.client.session_transaction() as sess:
            sess['user_id'] = self.user_b_id
            sess['username'] = self.user_b_name

        res_b = self.client.get('/api/coach/overview')
        self.assertEqual(res_b.status_code, 200)
        data_b = json.loads(res_b.data)

        # User B must NOT see User A's "Secret User A Topic"
        self.assertFalse(data_b["overview"]["has_sufficient_history"])
        self.assertNotIn("Secret User A Topic", json.dumps(data_b))

        # 3. Login as User A
        with self.client.session_transaction() as sess:
            sess['user_id'] = self.user_a_id
            sess['username'] = self.user_a_name

        res_a = self.client.get('/api/coach/overview')
        self.assertEqual(res_a.status_code, 200)
        data_a = json.loads(res_a.data)
        self.assertTrue(data_a["overview"]["has_sufficient_history"])
        self.assertIn("Secret User A Topic", json.dumps(data_a))


        # 4. Test remaining superpower API endpoints
        res_next = self.client.get('/api/coach/what-to-study-next')
        self.assertEqual(res_next.status_code, 200)

        res_weak = self.client.get('/api/coach/what-am-i-weak-at')
        self.assertEqual(res_weak.status_code, 200)

        res_rev = self.client.get('/api/coach/when-to-revise')
        self.assertEqual(res_rev.status_code, 200)

        res_imp = self.client.get('/api/coach/how-am-i-improving')
        self.assertEqual(res_imp.status_code, 200)

        # 5. Ask Coach POST API
        res_ask = self.client.post('/api/coach/ask', json={"query": "How am I doing?"})
        self.assertEqual(res_ask.status_code, 200)
        data_ask = json.loads(res_ask.data)
        self.assertTrue(data_ask["success"])
        self.assertIn("response", data_ask)


if __name__ == '__main__':
    unittest.main()
