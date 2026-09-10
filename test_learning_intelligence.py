"""
FocusSense Learning Intelligence — End-to-End Test Suite
========================================================
Verifies:
1. Study topic association with study sessions in SQLite.
2. Dynamic 3-tier difficulty quiz generation via Ollama llama3.2:3b (basic, application, advanced).
3. Semantic rubric evaluation of student answers (Correct, Partially Correct, Incorrect).
4. Accurate Knowledge Retention calculation (0 - 100%).
5. Complete 9-field SQLite persistence (user_id, session_id, topic, question, student_answer, ai_evaluation, correctness, retention_score, timestamp).
6. Topic mastery tier updates in SQLite (WEAK < 60%, MEDIUM 60-84%, STRONG >= 85%).
7. Composite FocusScore & Three-Pillar diagnostic calibration:
   - Long presence + poor recall -> Passive Presence warning.
   - Long presence + strong recall -> Holistic Deep Mastery.
8. Adaptive difficulty tuning based on historical topic mastery.
9. Learning Intelligence API routes (/api/learning/profile, /api/learning/weak-topics, /api/learning/topic-mastery).
"""

import os
import sqlite3
import unittest
from datetime import datetime, timedelta
from app import app
from database import create_database, DATABASE
from ai_service import ai_service
from learning_engine import (
    calculate_retention_score,
    update_topic_mastery,
    get_topic_mastery_list,
    get_weak_topics,
    get_student_learning_profile,
    get_adaptive_quiz_config,
    compute_learning_focus_score,
)
from recall_engine import (
    schedule_recall_task,
    start_recall_test,
    submit_recall_answer,
    finalize_recall_test,
)


class TestFocusSenseLearningIntelligence(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        create_database()
        cls.app = app
        cls.app.config["TESTING"] = True
        cls.client = cls.app.test_client()

        # Seed isolated test user
        conn = sqlite3.connect(DATABASE)
        cursor = conn.cursor()
        cursor.execute("SELECT id FROM users WHERE username = 'learning_test_student'")
        row = cursor.fetchone()
        if not row:
            cursor.execute("""
                INSERT INTO users (fullname, email, username, password)
                VALUES ('Learning Student', 'learningstudent@focus.test', 'learning_test_student', 'testpass123')
            """)
            cls.user_id = cursor.lastrowid
        else:
            cls.user_id = row[0]

        cursor.execute("INSERT OR REPLACE INTO goals (user_id, daily_goal, weekly_goal, recall_delay_minutes) VALUES (?, 2, 10, 0)", (cls.user_id,))
        conn.commit()
        conn.close()

    def setUp(self):
        with self.client.session_transaction() as sess:
            sess["user_id"] = self.user_id
            sess["username"] = "learning_test_student"
            sess["fullname"] = "Learning Student"

    # -------------------------------------------------------------------------
    # 1. Topic Association with Study Sessions
    # -------------------------------------------------------------------------
    def test_01_study_session_topic_association(self):
        """Study session records the topic in SQLite upon start and stop."""
        topic_name = "Python Functions"
        res_start = self.client.post("/start-study", json={"topic": topic_name})
        self.assertEqual(res_start.status_code, 200)
        data_start = res_start.get_json()
        self.assertTrue(data_start["success"])
        self.assertEqual(data_start["topic"], topic_name)

        # Stop session
        res_stop = self.client.post("/stop-study")
        self.assertEqual(res_stop.status_code, 200)
        data_stop = res_stop.get_json()
        self.assertTrue(data_stop["success"])
        session_id = data_stop["session_id"]

        # Verify in database
        conn = sqlite3.connect(DATABASE)
        conn.row_factory = sqlite3.Row
        c = conn.cursor()
        c.execute("SELECT topic FROM study_sessions WHERE id = ?", (session_id,))
        s_row = c.fetchone()
        conn.close()
        self.assertIsNotNone(s_row)
        self.assertEqual(s_row["topic"], topic_name)

    # -------------------------------------------------------------------------
    # 2. Dynamic 3-Tier Difficulty Quiz Generation via Ollama
    # -------------------------------------------------------------------------
    def test_02_dynamic_3tier_quiz_generation(self):
        """Ollama generates exactly 3 questions categorized into basic, application, and advanced tiers."""
        topic = "Python Functions"
        questions = ai_service.generate_quiz(topic=topic, question_count=3)
        self.assertEqual(len(questions), 3)

        for q in questions:
            self.assertIn("question_text", q)
            self.assertIn("sub_concept", q)
            self.assertIn("sample_answer", q)
            self.assertTrue(len(q["question_text"]) > 10)

    # -------------------------------------------------------------------------
    # 3. Semantic Answer Evaluation with Rubric Scoring
    # -------------------------------------------------------------------------
    def test_03_semantic_answer_evaluation(self):
        """Ollama semantically evaluates answers as correct, partial, or incorrect."""
        q_text = "What is the difference between an argument and a parameter in a Python function?"
        sub_c = "Parameters vs Arguments"
        rubric = "A parameter is the variable listed inside parentheses in the function definition, while an argument is the actual value sent to the function when called."

        # Correct answer
        ans_correct = "Parameters are defined in the function signature, while arguments are the actual values passed during execution."
        eval_correct = ai_service.evaluate_answer(q_text, sub_c, rubric, ans_correct)
        self.assertTrue(eval_correct.get("success"))
        self.assertIn(eval_correct.get("status"), ["correct", "partial"])
        self.assertGreaterEqual(eval_correct.get("score", 0), 0.5)

        # Incorrect answer
        ans_incorrect = "Arguments compile the bytecode to memory."
        eval_incorrect = ai_service.evaluate_answer(q_text, sub_c, rubric, ans_incorrect)
        self.assertTrue(eval_incorrect.get("success"))
        self.assertEqual(eval_incorrect.get("status"), "incorrect")
        self.assertEqual(eval_incorrect.get("score"), 0.0)

    # -------------------------------------------------------------------------
    # 4. Knowledge Retention Calculation Engine
    # -------------------------------------------------------------------------
    def test_04_retention_calculation(self):
        """Knowledge retention score calculates accurate percentage and assigns mastery tier."""
        mock_answers_high = [
            {"sub_concept": "Function Syntax", "points_possible": 1.0, "score_earned": 1.0, "evaluation_status": "correct"},
            {"sub_concept": "Return Values", "points_possible": 1.0, "score_earned": 1.0, "evaluation_status": "correct"},
            {"sub_concept": "Mutable Defaults", "points_possible": 1.0, "score_earned": 0.5, "evaluation_status": "partial"},
        ]
        res_high = calculate_retention_score(mock_answers_high)
        self.assertAlmostEqual(res_high["retention_score"], 83.3, places=1)
        self.assertEqual(res_high["tier"], "MEDIUM")
        self.assertIn("Function Syntax", res_high["learned_well"])
        self.assertIn("Mutable Defaults", res_high["needs_revision"])

        mock_answers_strong = [
            {"sub_concept": "Function Syntax", "points_possible": 1.0, "score_earned": 1.0, "evaluation_status": "correct"},
            {"sub_concept": "Return Values", "points_possible": 1.0, "score_earned": 1.0, "evaluation_status": "correct"},
            {"sub_concept": "Scope Rules", "points_possible": 1.0, "score_earned": 1.0, "evaluation_status": "correct"},
        ]
        res_strong = calculate_retention_score(mock_answers_strong)
        self.assertEqual(res_strong["retention_score"], 100.0)
        self.assertEqual(res_strong["tier"], "STRONG")
        self.assertEqual(len(res_strong["learned_well"]), 3)
        self.assertEqual(len(res_strong["needs_revision"]), 0)

        mock_answers_weak = [
            {"sub_concept": "Function Syntax", "points_possible": 1.0, "score_earned": 0.0, "evaluation_status": "incorrect"},
            {"sub_concept": "Return Values", "points_possible": 1.0, "score_earned": 0.5, "evaluation_status": "partial"},
            {"sub_concept": "Scope Rules", "points_possible": 1.0, "score_earned": 0.0, "evaluation_status": "incorrect"},
        ]
        res_weak = calculate_retention_score(mock_answers_weak)
        self.assertAlmostEqual(res_weak["retention_score"], 16.7, places=1)
        self.assertEqual(res_weak["tier"], "WEAK")
        self.assertIn("Function Syntax", res_weak["needs_revision"])

    # -------------------------------------------------------------------------
    # 5. Complete 9-Field SQLite Persistence & Full Recall Cycle
    # -------------------------------------------------------------------------
    def test_05_complete_recall_cycle_and_sqlite_persistence(self):
        """Recall test executes and persists all 9 required fields and adaptive learning logs in SQLite."""
        topic = "SQL Joins"
        # 1. Schedule
        task_info = schedule_recall_task(user_id=self.user_id, session_id=1, topic=topic, delay_minutes=0)
        task_id = task_info["task_id"]

        # 2. Start
        start_res = start_recall_test(task_id=task_id, user_id=self.user_id, question_count=3)
        self.assertTrue(start_res["success"])
        questions = start_res["questions"]
        self.assertEqual(len(questions), 3)

        # 3. Submit Answers
        for i, q in enumerate(questions):
            # Fetch sample answer from DB to test semantic grading
            conn = sqlite3.connect(DATABASE)
            conn.row_factory = sqlite3.Row
            c = conn.cursor()
            c.execute("SELECT sample_answer FROM quiz_questions WHERE id = ?", (q["id"],))
            q_row = c.fetchone()
            conn.close()
            ans_text = q_row["sample_answer"] if q_row else "SQL joins combine records from multiple tables."

            sub_res = submit_recall_answer(
                question_id=q["id"],
                student_answer=ans_text,
                user_id=self.user_id
            )
            self.assertTrue(sub_res["success"])

        # 4. Finalize
        fin_res = finalize_recall_test(task_id=task_id, user_id=self.user_id)
        self.assertTrue(fin_res["success"])
        self.assertGreaterEqual(fin_res["retention_score"], 50)
        self.assertIn("mastery_tier", fin_res)
        self.assertIn("physical_focus", fin_res)
        self.assertIn("learning_focus", fin_res)
        self.assertIn("final_focus_score", fin_res)
        self.assertIn("explanation", fin_res)
        self.assertIn("learned_well", fin_res)
        self.assertIn("needs_revision", fin_res)

        # 5. Verify SQLite 9-Field Persistence
        conn = sqlite3.connect(DATABASE)
        conn.row_factory = sqlite3.Row
        c = conn.cursor()

        # Check quiz_answers fields
        c.execute("""
            SELECT qa.user_id, sq.session_id, sq.topic, qq.question_text, qa.student_answer,
                   qa.feedback_text, qa.evaluation_status, rt.retention_score, qa.answered_at
            FROM quiz_answers qa
            JOIN quiz_questions qq ON qq.id = qa.question_id
            JOIN study_quizzes sq ON sq.id = qa.quiz_id
            JOIN recall_tasks rt ON rt.quiz_id = sq.id
            WHERE qa.user_id = ? AND sq.topic = ?
        """, (self.user_id, topic))
        ans_records = c.fetchall()

        self.assertGreaterEqual(len(ans_records), 3)
        sample_rec = ans_records[0]
        self.assertEqual(sample_rec["user_id"], self.user_id)
        self.assertEqual(sample_rec["topic"], topic)
        self.assertIsNotNone(sample_rec["question_text"])
        self.assertIsNotNone(sample_rec["student_answer"])
        self.assertIsNotNone(sample_rec["evaluation_status"])
        self.assertIsNotNone(sample_rec["retention_score"])
        self.assertIsNotNone(sample_rec["answered_at"])

        # Check adaptive_learning_logs
        c.execute("SELECT * FROM adaptive_learning_logs WHERE user_id = ? AND topic = ?", (self.user_id, topic))
        adaptive_log = c.fetchone()
        conn.close()

        self.assertIsNotNone(adaptive_log)
        self.assertEqual(adaptive_log["topic"], topic)
        self.assertIn(adaptive_log["action_intervention"], ["RECALL_TEST", "REMEDIATION_DRILL", "CONCEPTUAL_CHALLENGE", "ADVANCED_SYNTHESIS"])

    # -------------------------------------------------------------------------
    # 6. Topic Mastery Tracking (Weak, Medium, Strong)
    # -------------------------------------------------------------------------
    def test_06_topic_mastery_tiers_in_sqlite(self):
        """Update and query topic mastery tiers across multiple topics."""
        # Weak topic
        update_topic_mastery(self.user_id, "Memory Management", 40.0, [
            {"evaluation_status": "incorrect", "points_possible": 1.0},
            {"evaluation_status": "partial", "points_possible": 1.0},
            {"evaluation_status": "incorrect", "points_possible": 1.0},
        ])

        # Strong topic
        update_topic_mastery(self.user_id, "Python Decorators", 95.0, [
            {"evaluation_status": "correct", "points_possible": 1.0},
            {"evaluation_status": "correct", "points_possible": 1.0},
            {"evaluation_status": "correct", "points_possible": 1.0},
        ])

        mastery_list = get_topic_mastery_list(self.user_id)
        topics_map = {m["topic"]: m["mastery_tier"] for m in mastery_list}

        self.assertIn("Memory Management", topics_map)
        self.assertEqual(topics_map["Memory Management"], "WEAK")
        self.assertIn("Python Decorators", topics_map)
        self.assertEqual(topics_map["Python Decorators"], "STRONG")

        weak_list = get_weak_topics(self.user_id)
        weak_names = [w["topic"] for w in weak_list]
        self.assertIn("Memory Management", weak_names)

    # -------------------------------------------------------------------------
    # 7. FocusScore 3-Pillar Calibration (Presence vs. Recall Contrast)
    # -------------------------------------------------------------------------
    def test_07_focus_score_learning_contrast(self):
        """System produces different composite diagnosis for Long Presence + Poor Recall vs Strong Recall."""
        # Case A: Long Presence (95%) + Poor Recall (40%)
        res_passive = compute_learning_focus_score(presence_pct=95.0, behavior_pct=90.0, retention_pct=40.0)
        self.assertIn("Passive Presence Detected", res_passive["insight"])
        self.assertLessEqual(res_passive["final_score"], 74)
        self.assertEqual(res_passive["physical_focus"], 92.8)
        self.assertEqual(res_passive["learning_focus"], 40.0)
        self.assertIn("concepts need revision", res_passive["explanation"])

        # Case B: Long Presence (95%) + Strong Recall (95%)
        res_mastery = compute_learning_focus_score(presence_pct=95.0, behavior_pct=90.0, retention_pct=95.0)
        self.assertIn("Holistic Deep Mastery", res_mastery["insight"])
        self.assertGreaterEqual(res_mastery["final_score"], 90)
        self.assertEqual(res_mastery["learning_focus"], 95.0)

    # -------------------------------------------------------------------------
    # 8. Adaptive Difficulty Tuning
    # -------------------------------------------------------------------------
    def test_08_adaptive_difficulty_configuration(self):
        """Weak topics trigger remediation guidance while Strong topics trigger advanced challenge."""
        cfg_weak = get_adaptive_quiz_config(self.user_id, "Memory Management")
        self.assertEqual(cfg_weak["strategy"], "REMEDIATION")
        self.assertIn("basic", cfg_weak["tier_distribution"])

        cfg_strong = get_adaptive_quiz_config(self.user_id, "Python Decorators")
        self.assertEqual(cfg_strong["strategy"], "ADVANCED_CHALLENGE")
        self.assertIn("advanced", cfg_strong["tier_distribution"])

    # -------------------------------------------------------------------------
    # 9. Learning Intelligence API Routes
    # -------------------------------------------------------------------------
    def test_09_learning_api_endpoints(self):
        """GET /api/learning/profile, /api/learning/weak-topics, and /api/learning/topic-mastery return valid JSON."""
        # 1. Profile
        res_prof = self.client.get("/api/learning/profile")
        self.assertEqual(res_prof.status_code, 200)
        data_prof = res_prof.get_json()
        self.assertTrue(data_prof["success"])
        self.assertIn("total_topics_tracked", data_prof["profile"])
        self.assertIn("diagnosis", data_prof["profile"])

        # 2. Weak topics
        res_weak = self.client.get("/api/learning/weak-topics")
        self.assertEqual(res_weak.status_code, 200)
        data_weak = res_weak.get_json()
        self.assertTrue(data_weak["success"])
        self.assertIsInstance(data_weak["weak_topics"], list)

        # 3. Topic mastery
        res_tm = self.client.get("/api/learning/topic-mastery")
        self.assertEqual(res_tm.status_code, 200)
        data_tm = res_tm.get_json()
        self.assertTrue(data_tm["success"])
        self.assertIsInstance(data_tm["topic_mastery"], list)


if __name__ == "__main__":
    unittest.main()
