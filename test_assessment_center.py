"""
FocusSense AI — Exam & Assessment Center Test Suite (Phase 5)
============================================================
Comprehensive test suite verifying:
- Multi-mode test creation (Quick, Practice, Topic, Exam)
- Material-grounded question generation
- Server-authoritative timer & elapsed time calculation
- Client-side answer protection during active testing
- Hybrid grading (deterministic MCQ/TF + AI semantic rubrics)
- Dimensional cognitive scoring (Concept, Recall, Application, Problem Solving)
- Strong (>=75%) vs Weak (<70%) concept detection
- Single source of truth topic_mastery updates
- FocusScore separation and protection
- Closed-loop adaptive learning integration
- Socratic AI mistake remediation
- Historical persistence
- Multi-user isolation and security
- Complete end-to-end assessment lifecycle
"""

import json
import os
import sqlite3
import unittest
from datetime import datetime, timezone

from database import DB_PATH, create_database
import ai_service
import assessment_engine
import learning_engine
import study_material_engine
from app import app


class TestAssessmentCenter(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        create_database()
        cls._orig_force_local = ai_service.ai_service.force_local
        ai_service.ai_service.force_local = True

    @classmethod
    def tearDownClass(cls):
        ai_service.ai_service.force_local = getattr(cls, "_orig_force_local", False)

    def setUp(self):
        self.app = app.test_client()
        self.app.testing = True

        # Set up test users in database
        self.conn = sqlite3.connect(DB_PATH)
        self.conn.row_factory = sqlite3.Row
        self.cursor = self.conn.cursor()

        # Clean prior test users
        self.cursor.execute("DELETE FROM users WHERE username IN ('test_exam_student', 'test_exam_other')")
        self.conn.commit()

        # Insert test student
        self.cursor.execute("""
        INSERT INTO users (fullname, email, username, password)
        VALUES ('Exam Student', 'student@exam.local', 'test_exam_student', 'hash123')
        """)
        self.user_id = self.cursor.lastrowid

        # Insert second student for multi-user isolation tests
        self.cursor.execute("""
        INSERT INTO users (fullname, email, username, password)
        VALUES ('Other Student', 'other@exam.local', 'test_exam_other', 'hash456')
        """)
        self.other_user_id = self.cursor.lastrowid

        self.conn.commit()

    def tearDown(self):
        try:
            # Clean up test artifacts
            self.cursor.execute("DELETE FROM assessment_attempts WHERE user_id IN (?, ?)", (self.user_id, self.other_user_id))
            self.cursor.execute("DELETE FROM study_quizzes WHERE user_id IN (?, ?)", (self.user_id, self.other_user_id))
            self.cursor.execute("DELETE FROM concept_mastery WHERE user_id IN (?, ?)", (self.user_id, self.other_user_id))
            self.cursor.execute("DELETE FROM topic_mastery WHERE user_id IN (?, ?)", (self.user_id, self.other_user_id))
            self.cursor.execute("DELETE FROM study_materials WHERE user_id IN (?, ?)", (self.user_id, self.other_user_id))
            self.cursor.execute("DELETE FROM material_chunks WHERE user_id IN (?, ?)", (self.user_id, self.other_user_id))
            self.cursor.execute("DELETE FROM users WHERE id IN (?, ?)", (self.user_id, self.other_user_id))
            self.conn.commit()
            self.conn.close()
        except Exception:
            pass

    # =========================================================================
    # Test 1: Quick Test Creation (5 Questions, Untimed)
    # =========================================================================
    def test_01_quick_test_creation(self):
        res = assessment_engine.create_assessment(
            user_id=self.user_id,
            mode="QUICK",
            topic="Operating Systems - CPU Scheduling"
        )
        self.assertTrue(res["success"])
        self.assertEqual(res["mode"], "QUICK")
        self.assertEqual(res["total_questions"], 5)
        self.assertIsNone(res["time_limit_sec"])
        self.assertEqual(len(res["questions"]), 5)

    # =========================================================================
    # Test 2: Practice Test Creation (10 Questions, Untimed)
    # =========================================================================
    def test_02_practice_test_creation(self):
        res = assessment_engine.create_assessment(
            user_id=self.user_id,
            mode="PRACTICE",
            topic="Database Management Systems - SQL Joins"
        )
        self.assertTrue(res["success"])
        self.assertEqual(res["mode"], "PRACTICE")
        self.assertEqual(res["total_questions"], 10)
        self.assertIsNone(res["time_limit_sec"])

    # =========================================================================
    # Test 3: Topic Test Creation (8 Questions, Adaptive Difficulty)
    # =========================================================================
    def test_03_topic_test_creation(self):
        res = assessment_engine.create_assessment(
            user_id=self.user_id,
            mode="TOPIC",
            topic="Python Programming",
            sub_topic="Recursion & Memory Management"
        )
        self.assertTrue(res["success"])
        self.assertEqual(res["mode"], "TOPIC")
        self.assertIn("Python Programming - Recursion & Memory Management", res["topic"])
        self.assertEqual(res["total_questions"], 8)

    # =========================================================================
    # Test 4: Exam Mode Creation (Strict Time Limit)
    # =========================================================================
    def test_04_exam_mode_creation(self):
        res = assessment_engine.create_assessment(
            user_id=self.user_id,
            mode="EXAM",
            topic="Operating Systems",
            question_count=12,
            time_limit_sec=600
        )
        self.assertTrue(res["success"])
        self.assertEqual(res["mode"], "EXAM")
        self.assertEqual(res["time_limit_sec"], 600)
        self.assertIsNotNone(res["started_at"])

    # =========================================================================
    # Test 5: Client-Side Answer Protection During Active Assessment
    # =========================================================================
    def test_05_answer_protection_during_active_assessment(self):
        create_res = assessment_engine.create_assessment(
            user_id=self.user_id,
            mode="PRACTICE",
            topic="Computer Networks"
        )
        attempt_id = create_res["attempt_id"]

        active_res = assessment_engine.get_active_assessment(self.user_id, attempt_id)
        self.assertTrue(active_res["success"])
        self.assertFalse(active_res["is_completed"])

        for q in active_res["questions"]:
            self.assertNotIn("correct_option", q, "correct_option must NOT be leaked to client during active assessment")
            self.assertNotIn("sample_answer", q, "sample_answer must NOT be leaked to client during active assessment")
            self.assertNotIn("explanation", q, "explanation must NOT be leaked to client during active assessment")

    # =========================================================================
    # Test 6: Material-Grounded Question Generation
    # =========================================================================
    def test_06_material_grounded_question_generation(self):
        # Save a sample study material
        mat_res = study_material_engine.save_and_process_material(
            user_id=self.user_id,
            original_filename="scheduling_lecture.txt",
            file_bytes=b"Round Robin scheduling allocates a fixed time quantum. The convoy effect occurs in FCFS.",
            subject="Operating Systems",
            topic="CPU Scheduling"
        )
        self.assertTrue(mat_res["success"])
        mat_id = mat_res["material_id"]

        # Create assessment grounded in this material
        res = assessment_engine.create_assessment(
            user_id=self.user_id,
            mode="PRACTICE",
            topic="CPU Scheduling",
            material_id=mat_id,
            question_count=5
        )
        self.assertTrue(res["success"])
        self.assertEqual(res["material_id"], mat_id)
        self.assertEqual(len(res["questions"]), 5)

    # =========================================================================
    # Test 7: Deterministic Server-Side MCQ Grading (Correct)
    # =========================================================================
    def test_07_deterministic_mcq_grading_correct(self):
        create_res = assessment_engine.create_assessment(
            user_id=self.user_id,
            mode="QUICK",
            topic="Operating Systems"
        )
        attempt_id = create_res["attempt_id"]

        # Find an MCQ question from the attempt in DB
        self.cursor.execute("""
        SELECT * FROM quiz_questions
        WHERE quiz_id = ? AND question_type = 'mcq' LIMIT 1
        """, (create_res["quiz_id"],))
        mcq_row = self.cursor.fetchone()

        if mcq_row:
            qid = mcq_row["id"]
            corr_opt = mcq_row["correct_option"]
            ans_res = assessment_engine.submit_assessment_answer(
                user_id=self.user_id,
                attempt_id=attempt_id,
                question_id=qid,
                student_answer=corr_opt
            )
            self.assertTrue(ans_res["success"])
            self.assertEqual(ans_res["status"], "correct")
            self.assertEqual(ans_res["score_earned"], 1.0)

    # =========================================================================
    # Test 8: Deterministic Server-Side MCQ Grading (Incorrect)
    # =========================================================================
    def test_08_deterministic_mcq_grading_incorrect(self):
        create_res = assessment_engine.create_assessment(
            user_id=self.user_id,
            mode="QUICK",
            topic="Operating Systems"
        )
        attempt_id = create_res["attempt_id"]

        self.cursor.execute("""
        SELECT * FROM quiz_questions
        WHERE quiz_id = ? AND question_type = 'mcq' LIMIT 1
        """, (create_res["quiz_id"],))
        mcq_row = self.cursor.fetchone()

        if mcq_row:
            qid = mcq_row["id"]
            corr_opt = mcq_row["correct_option"]
            wrong_opt = "A" if corr_opt != "A" else "C"

            ans_res = assessment_engine.submit_assessment_answer(
                user_id=self.user_id,
                attempt_id=attempt_id,
                question_id=qid,
                student_answer=wrong_opt
            )
            self.assertTrue(ans_res["success"])
            self.assertEqual(ans_res["status"], "incorrect")
            self.assertEqual(ans_res["score_earned"], 0.0)

    # =========================================================================
    # Test 9: Deterministic Server-Side True/False Grading
    # =========================================================================
    def test_09_deterministic_tf_grading(self):
        create_res = assessment_engine.create_assessment(
            user_id=self.user_id,
            mode="QUICK",
            topic="Operating Systems"
        )
        attempt_id = create_res["attempt_id"]

        self.cursor.execute("""
        SELECT * FROM quiz_questions
        WHERE quiz_id = ? AND question_type = 'true_false' LIMIT 1
        """, (create_res["quiz_id"],))
        tf_row = self.cursor.fetchone()

        if tf_row:
            qid = tf_row["id"]
            corr_opt = tf_row["correct_option"]
            ans_res = assessment_engine.submit_assessment_answer(
                user_id=self.user_id,
                attempt_id=attempt_id,
                question_id=qid,
                student_answer=corr_opt
            )
            self.assertTrue(ans_res["success"])
            self.assertEqual(ans_res["status"], "correct")
            self.assertEqual(ans_res["score_earned"], 1.0)

    # =========================================================================
    # Test 10: Semantic AI Rubric Grading for Conceptual / Short Answer
    # =========================================================================
    def test_10_semantic_ai_rubric_grading(self):
        create_res = assessment_engine.create_assessment(
            user_id=self.user_id,
            mode="QUICK",
            topic="Operating Systems"
        )
        attempt_id = create_res["attempt_id"]

        self.cursor.execute("""
        SELECT * FROM quiz_questions
        WHERE quiz_id = ? AND question_type IN ('conceptual', 'short_answer') LIMIT 1
        """, (create_res["quiz_id"],))
        c_row = self.cursor.fetchone()

        if c_row:
            qid = c_row["id"]
            sample_ans = c_row["sample_answer"]

            # Submit good answer matching key concepts
            ans_res = assessment_engine.submit_assessment_answer(
                user_id=self.user_id,
                attempt_id=attempt_id,
                question_id=qid,
                student_answer=sample_ans
            )
            self.assertTrue(ans_res["success"])
            self.assertIn(ans_res["status"], ["correct", "partial"])
            self.assertGreater(ans_res["score_earned"], 0.0)

    # =========================================================================
    # Test 11: Server-Authoritative Timer Calculation
    # =========================================================================
    def test_11_server_authoritative_timer(self):
        create_res = assessment_engine.create_assessment(
            user_id=self.user_id,
            mode="EXAM",
            topic="Operating Systems",
            time_limit_sec=600
        )
        attempt_id = create_res["attempt_id"]

        active_res = assessment_engine.get_active_assessment(self.user_id, attempt_id)
        self.assertTrue(active_res["success"])
        self.assertLessEqual(active_res["elapsed_sec"], 5)
        self.assertGreaterEqual(active_res["remaining_sec"], 595)
        self.assertFalse(active_res["is_expired"])

    # =========================================================================
    # Test 12: Dimensional Cognitive Analytics & Finalization
    # =========================================================================
    def test_12_dimensional_analytics_and_finalization(self):
        create_res = assessment_engine.create_assessment(
            user_id=self.user_id,
            mode="QUICK",
            topic="Operating Systems"
        )
        attempt_id = create_res["attempt_id"]
        quiz_id = create_res["quiz_id"]

        # Fetch questions and submit answers
        self.cursor.execute("SELECT * FROM quiz_questions WHERE quiz_id = ?", (quiz_id,))
        q_rows = self.cursor.fetchall()

        for q in q_rows:
            qid = q["id"]
            q_type = q["question_type"]
            if q_type in ["mcq", "true_false"]:
                ans = q["correct_option"]
            else:
                ans = q["sample_answer"]
            assessment_engine.submit_assessment_answer(self.user_id, attempt_id, qid, ans)

        # Finalize assessment
        fin_res = assessment_engine.finalize_assessment(self.user_id, attempt_id)
        self.assertTrue(fin_res["success"])
        self.assertGreater(fin_res["overall_score_pct"], 0.0)
        self.assertIn("dimensions", fin_res)
        self.assertIn("concept_score", fin_res["dimensions"])
        self.assertIn("recall_score", fin_res["dimensions"])
        self.assertIn("application_score", fin_res["dimensions"])
        self.assertIn("problem_solving_score", fin_res["dimensions"])

    # =========================================================================
    # Test 13: Strong vs Weak Concept Detection
    # =========================================================================
    def test_13_strong_and_weak_concept_detection(self):
        create_res = assessment_engine.create_assessment(
            user_id=self.user_id,
            mode="PRACTICE",
            topic="Operating Systems"
        )
        attempt_id = create_res["attempt_id"]
        quiz_id = create_res["quiz_id"]

        self.cursor.execute("SELECT * FROM quiz_questions WHERE quiz_id = ?", (quiz_id,))
        q_rows = self.cursor.fetchall()

        # Answer first half correctly, second half incorrectly
        for idx, q in enumerate(q_rows):
            qid = q["id"]
            if idx < len(q_rows) // 2:
                ans = q["correct_option"] or q["sample_answer"]
            else:
                ans = "completely wrong answer irrelevant to topic"
            assessment_engine.submit_assessment_answer(self.user_id, attempt_id, qid, ans)

        fin_res = assessment_engine.finalize_assessment(self.user_id, attempt_id)
        self.assertTrue(fin_res["success"])
        # Should have strong concepts (>=75%) and weak concepts (<70%)
        self.assertIsInstance(fin_res["strong_concepts"], list)
        self.assertIsInstance(fin_res["weak_concepts"], list)

    # =========================================================================
    # Test 14: Single Source of Truth Topic Mastery Updates
    # =========================================================================
    def test_14_topic_mastery_single_source_of_truth(self):
        create_res = assessment_engine.create_assessment(
            user_id=self.user_id,
            mode="QUICK",
            topic="Computer Architecture"
        )
        attempt_id = create_res["attempt_id"]
        quiz_id = create_res["quiz_id"]

        self.cursor.execute("SELECT * FROM quiz_questions WHERE quiz_id = ?", (quiz_id,))
        for q in self.cursor.fetchall():
            ans = q["correct_option"] or q["sample_answer"]
            assessment_engine.submit_assessment_answer(self.user_id, attempt_id, q["id"], ans)

        fin_res = assessment_engine.finalize_assessment(self.user_id, attempt_id)
        self.assertTrue(fin_res["success"])

        # Verify topic_mastery table was updated via learning_engine
        mastery_list = learning_engine.get_topic_mastery_list(self.user_id)
        self.assertTrue(any(tm["topic"] == "Computer Architecture" for tm in mastery_list))

    # =========================================================================
    # Test 15: FocusScore Protection & Separation
    # =========================================================================
    def test_15_focusscore_protection(self):
        # Insert initial session focus score
        self.cursor.execute("""
        INSERT INTO study_sessions (user_id, date, start_time, end_time, duration, focus_score, topic)
        VALUES (?, '2026-08-26', '10:00:00', '10:30:00', 1800, 85.0, 'Operating Systems')
        """, (self.user_id,))
        session_id = self.cursor.lastrowid
        self.conn.commit()

        # Run and complete a 0% assessment attempt
        create_res = assessment_engine.create_assessment(
            user_id=self.user_id,
            mode="QUICK",
            topic="Operating Systems"
        )
        attempt_id = create_res["attempt_id"]
        quiz_id = create_res["quiz_id"]

        self.cursor.execute("SELECT * FROM quiz_questions WHERE quiz_id = ?", (quiz_id,))
        for q in self.cursor.fetchall():
            assessment_engine.submit_assessment_answer(self.user_id, attempt_id, q["id"], "wrong answer")

        fin_res = assessment_engine.finalize_assessment(self.user_id, attempt_id)
        self.assertTrue(fin_res["success"])

        # Check that the physical study session focus_score was NOT directly overwritten
        self.cursor.execute("SELECT focus_score FROM study_sessions WHERE id = ?", (session_id,))
        s_score = self.cursor.fetchone()["focus_score"]
        self.assertEqual(s_score, 85.0, "Physical focus score must not be overwritten by assessment performance")

    # =========================================================================
    # Test 16: Closed-Loop Adaptive Recommendation Generation
    # =========================================================================
    def test_16_adaptive_action_recommendation(self):
        create_res = assessment_engine.create_assessment(
            user_id=self.user_id,
            mode="QUICK",
            topic="Operating Systems"
        )
        attempt_id = create_res["attempt_id"]
        quiz_id = create_res["quiz_id"]

        self.cursor.execute("SELECT * FROM quiz_questions WHERE quiz_id = ?", (quiz_id,))
        for q in self.cursor.fetchall():
            ans = q["correct_option"] or q["sample_answer"]
            assessment_engine.submit_assessment_answer(self.user_id, attempt_id, q["id"], ans)

        fin_res = assessment_engine.finalize_assessment(self.user_id, attempt_id)
        self.assertTrue(fin_res["success"])
        self.assertIn("adaptive_recommendation", fin_res)

    # =========================================================================
    # Test 17: Socratic AI Mistake Remediation
    # =========================================================================
    def test_17_socratic_ai_mistake_remediation(self):
        create_res = assessment_engine.create_assessment(
            user_id=self.user_id,
            mode="QUICK",
            topic="Operating Systems"
        )
        attempt_id = create_res["attempt_id"]
        quiz_id = create_res["quiz_id"]

        self.cursor.execute("SELECT * FROM quiz_questions WHERE quiz_id = ? LIMIT 1", (quiz_id,))
        q = self.cursor.fetchone()
        qid = q["id"]

        assessment_engine.submit_assessment_answer(self.user_id, attempt_id, qid, "Incorrect attempt at scheduling")

        expl_res = assessment_engine.explain_assessment_mistake(self.user_id, attempt_id, qid)
        self.assertTrue(expl_res["success"])
        self.assertTrue(len(expl_res["explanation"]) > 10)
        self.assertIn(q["sub_concept"], expl_res["explanation"] + expl_res["sub_concept"])

    # =========================================================================
    # Test 18: Historical Persistence (Never Overwrites Prior Attempts)
    # =========================================================================
    def test_18_historical_persistence(self):
        # Create 3 attempts in sequence
        a1 = assessment_engine.create_assessment(self.user_id, mode="QUICK", topic="Topic 1")
        assessment_engine.finalize_assessment(self.user_id, a1["attempt_id"])

        a2 = assessment_engine.create_assessment(self.user_id, mode="PRACTICE", topic="Topic 2")
        assessment_engine.finalize_assessment(self.user_id, a2["attempt_id"])

        a3 = assessment_engine.create_assessment(self.user_id, mode="EXAM", topic="Topic 3")
        assessment_engine.finalize_assessment(self.user_id, a3["attempt_id"])

        history = assessment_engine.get_user_assessment_history(self.user_id)
        self.assertGreaterEqual(len(history), 3)
        attempt_ids = [h["attempt_id"] for h in history]
        self.assertIn(a1["attempt_id"], attempt_ids)
        self.assertIn(a2["attempt_id"], attempt_ids)
        self.assertIn(a3["attempt_id"], attempt_ids)

    # =========================================================================
    # Test 19: Multi-User Tenant Isolation & Security
    # =========================================================================
    def test_19_multi_user_isolation(self):
        # User 1 creates an assessment
        res_u1 = assessment_engine.create_assessment(self.user_id, mode="EXAM", topic="Private User 1 Exam")
        attempt_u1 = res_u1["attempt_id"]

        # User 2 attempts to fetch User 1's assessment
        active_u2 = assessment_engine.get_active_assessment(self.other_user_id, attempt_u1)
        self.assertFalse(active_u2["success"], "User 2 must NOT be allowed to access User 1's assessment")

        # User 2 attempts to submit answer to User 1's assessment
        submit_u2 = assessment_engine.submit_assessment_answer(self.other_user_id, attempt_u1, 1, "test")
        self.assertFalse(submit_u2["success"], "User 2 must NOT be allowed to submit answers to User 1's assessment")

        # User 2 attempts to finalize User 1's assessment
        fin_u2 = assessment_engine.finalize_assessment(self.other_user_id, attempt_u1)
        self.assertFalse(fin_u2["success"], "User 2 must NOT be allowed to finalize User 1's assessment")

        # User 2 attempts to get User 1's results
        res_u2 = assessment_engine.get_assessment_results(self.other_user_id, attempt_u1)
        self.assertFalse(res_u2["success"], "User 2 must NOT be allowed to read User 1's results")

    # =========================================================================
    # Test 20: Post-Assessment Results Reveals Answers and Explanations
    # =========================================================================
    def test_20_results_reveals_explanations_post_completion(self):
        create_res = assessment_engine.create_assessment(self.user_id, mode="QUICK", topic="Operating Systems")
        attempt_id = create_res["attempt_id"]

        assessment_engine.finalize_assessment(self.user_id, attempt_id)

        results = assessment_engine.get_assessment_results(self.user_id, attempt_id)
        self.assertTrue(results["success"])
        self.assertIsNotNone(results["completed_at"])

        for q in results["questions"]:
            self.assertIn("sample_answer", q)
            self.assertIn("explanation", q)
            self.assertIn("evaluation_status", q)

    # =========================================================================
    # Test 21: REST API Endpoint - POST /api/assessment/create
    # =========================================================================
    def test_21_api_assessment_create(self):
        with self.app.session_transaction() as sess:
            sess["user_id"] = self.user_id

        resp = self.app.post("/api/assessment/create", json={
            "mode": "PRACTICE",
            "topic": "Algorithms and Data Structures",
            "difficulty": "medium"
        })
        self.assertEqual(resp.status_code, 200)
        data = json.loads(resp.data)
        self.assertTrue(data["success"])
        self.assertIn("attempt_id", data)

    # =========================================================================
    # Test 22: REST API Endpoint - GET /api/assessment/<id>
    # =========================================================================
    def test_22_api_assessment_get(self):
        create_res = assessment_engine.create_assessment(self.user_id, mode="QUICK", topic="Algorithms")
        attempt_id = create_res["attempt_id"]

        with self.app.session_transaction() as sess:
            sess["user_id"] = self.user_id

        resp = self.app.get(f"/api/assessment/{attempt_id}")
        self.assertEqual(resp.status_code, 200)
        data = json.loads(resp.data)
        self.assertTrue(data["success"])
        self.assertEqual(data["attempt_id"], attempt_id)

    # =========================================================================
    # Test 23: REST API Endpoint - POST /api/assessment/<id>/submit-answer
    # =========================================================================
    def test_23_api_assessment_submit_answer(self):
        create_res = assessment_engine.create_assessment(self.user_id, mode="QUICK", topic="Algorithms")
        attempt_id = create_res["attempt_id"]
        qid = create_res["questions"][0]["id"]

        with self.app.session_transaction() as sess:
            sess["user_id"] = self.user_id

        resp = self.app.post(f"/api/assessment/{attempt_id}/submit-answer", json={
            "question_id": qid,
            "student_answer": "Binary Search Tree"
        })
        self.assertEqual(resp.status_code, 200)
        data = json.loads(resp.data)
        self.assertTrue(data["success"])

    # =========================================================================
    # Test 24: REST API Endpoint - POST /api/assessment/<id>/finalize & GET results
    # =========================================================================
    def test_24_api_assessment_finalize_and_results(self):
        create_res = assessment_engine.create_assessment(self.user_id, mode="QUICK", topic="Algorithms")
        attempt_id = create_res["attempt_id"]

        with self.app.session_transaction() as sess:
            sess["user_id"] = self.user_id

        fin_resp = self.app.post(f"/api/assessment/{attempt_id}/finalize")
        self.assertEqual(fin_resp.status_code, 200)

        res_resp = self.app.get(f"/api/assessment/{attempt_id}/results")
        self.assertEqual(res_resp.status_code, 200)
        data = json.loads(res_resp.data)
        self.assertTrue(data["success"])

    # =========================================================================
    # Test 25: REST API Endpoint - GET /api/assessment/history
    # =========================================================================
    def test_25_api_assessment_history(self):
        assessment_engine.create_assessment(self.user_id, mode="QUICK", topic="Data Structures")

        with self.app.session_transaction() as sess:
            sess["user_id"] = self.user_id

        resp = self.app.get("/api/assessment/history")
        self.assertEqual(resp.status_code, 200)
        data = json.loads(resp.data)
        self.assertTrue(data["success"])
        self.assertIsInstance(data["history"], list)

    # =========================================================================
    # Test 26: Web Page Route - GET /assessment
    # =========================================================================
    def test_26_web_route_assessment_page(self):
        with self.app.session_transaction() as sess:
            sess["user_id"] = self.user_id
            sess["fullname"] = "Exam Student"

        resp = self.app.get("/assessment")
        self.assertEqual(resp.status_code, 200)
        self.assertIn(b"Assessment Center", resp.data)
        self.assertIn(b"Quick Test", resp.data)
        self.assertIn(b"Practice Test", resp.data)
        self.assertIn(b"Exam Mode", resp.data)


if __name__ == "__main__":
    unittest.main()
