"""
Automated Test Suite for FocusSense Recall Engine (Phase 3).
Verifies scheduling, question generation, answer grading, retention scoring,
weak concept identification, Three-Pillar intelligence, and Flask API routes.
"""

import sys
import unittest
from datetime import datetime, timedelta

from app import app, get_db
from recall_engine import (
    schedule_recall_task,
    get_pending_and_due_tasks,
    start_recall_test,
    submit_recall_answer,
    finalize_recall_test,
    get_weak_topics_summary,
    get_recall_history,
)
from focus_intelligence import compute_three_pillar_intelligence


class TestFocusSenseRecallEngine(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.client = app.test_client()
        app.config["TESTING"] = True

        # Find or create a test user
        conn = get_db()
        cursor = conn.cursor()
        cursor.execute("SELECT id, username, fullname FROM users LIMIT 1")
        user = cursor.fetchone()
        if not user:
            cursor.execute("""
                INSERT INTO users (username, password, fullname, email)
                VALUES ('test_recall_user', 'hash123', 'Recall Tester', 'recall@example.com')
            """)
            conn.commit()
            cursor.execute("SELECT id, username, fullname FROM users WHERE username = 'test_recall_user'")
            user = cursor.fetchone()
        cls.user_id = user["id"]
        cls.username = user["username"]
        cls.fullname = user["fullname"]
        conn.close()

    def login_session(self):
        with self.client.session_transaction() as sess:
            sess["user_id"] = self.user_id
            sess["username"] = self.username
            sess["fullname"] = self.fullname

    def test_01_three_pillar_intelligence_calculation(self):
        """Verify the 3-Pillar cognitive model (Presence + Behavior + Retention)."""
        # Case A: Full 3 pillars
        result_full = compute_three_pillar_intelligence(
            presence_pct=95.0,
            behavior_pct=85.0,
            retention_score_pct=88.0
        )
        self.assertIn("composite_score", result_full)
        self.assertGreaterEqual(result_full["composite_score"], 80)
        self.assertEqual(len(result_full["pillars"]), 3)
        self.assertTrue(len(result_full["insight"]) > 5)

        # Case B: Pending recall (no retention score yet)
        result_pending = compute_three_pillar_intelligence(
            presence_pct=90.0,
            behavior_pct=80.0,
            retention_score_pct=None
        )
        self.assertIn("composite_score", result_pending)
        self.assertEqual(result_pending["pillars"][2]["value"], None)
        self.assertIn("Pending", result_pending["pillars"][2]["badge"])

    def test_02_schedule_and_retrieve_recall_tasks(self):
        """Verify scheduling a recall task and transitioning from PENDING to READY."""
        # 1. Schedule instant task (delay = 0)
        task_instant = schedule_recall_task(
            user_id=self.user_id,
            session_id=1,
            topic="Operating Systems - CPU Scheduling",
            delay_minutes=0
        )
        self.assertTrue(task_instant["success"])
        self.assertEqual(task_instant["status"], "READY")

        # 2. Schedule delayed task (delay = 30)
        task_delayed = schedule_recall_task(
            user_id=self.user_id,
            session_id=1,
            topic="Python Data Structures - Hash Maps",
            delay_minutes=30
        )
        self.assertTrue(task_delayed["success"])
        self.assertEqual(task_delayed["status"], "PENDING")

        # 3. Retrieve tasks
        pending_data = get_pending_and_due_tasks(self.user_id)
        self.assertGreaterEqual(pending_data["ready_count"], 1)
        self.assertIsNotNone(pending_data["primary_task"])

    def test_03_full_recall_test_lifecycle(self):
        """Verify starting test, submitting answers turn-by-turn, and finalizing retention score."""
        # 1. Schedule a dedicated test task
        task = schedule_recall_task(
            user_id=self.user_id,
            session_id=1,
            topic="Operating Systems - Process Synchronization",
            delay_minutes=0
        )
        task_id = task["task_id"]

        # 2. Start recall test (request 3 questions)
        start_res = start_recall_test(
            task_id=task_id,
            user_id=self.user_id,
            question_count=3
        )
        self.assertTrue(start_res["success"])
        questions = start_res["questions"]
        self.assertEqual(len(questions), 3)

        # 3. Grade answers for each question
        for i, q in enumerate(questions):
            if i == 0:
                # Strong answer matching generated question from DB
                conn = get_db()
                c = conn.cursor()
                c.execute("SELECT sample_answer FROM quiz_questions WHERE id = ?", (q["id"],))
                q_row = c.fetchone()
                conn.close()
                strong_ans = q_row["sample_answer"] if q_row and q_row["sample_answer"] else "A mutex ensures mutual exclusion so only one process enters the critical section at any time."
                ans_res = submit_recall_answer(
                    question_id=q["id"],
                    student_answer=strong_ans,
                    user_id=self.user_id
                )
                self.assertTrue(ans_res["success"])
                self.assertIn(ans_res["evaluation"], ["Correct", "Partially Correct"])
            elif i == 1:
                # Partial answer
                ans_res = submit_recall_answer(
                    question_id=q["id"],
                    student_answer="Semaphores use wait and signal primitives.",
                    user_id=self.user_id
                )
                self.assertTrue(ans_res["success"])
            else:
                # Weak answer
                ans_res = submit_recall_answer(
                    question_id=q["id"],
                    student_answer="I forgot what race condition means.",
                    user_id=self.user_id
                )
                self.assertTrue(ans_res["success"])

        # 4. Finalize recall test
        fin_res = finalize_recall_test(task_id=task_id, user_id=self.user_id)
        self.assertTrue(fin_res["success"])
        self.assertIn("retention_score", fin_res)
        self.assertGreaterEqual(fin_res["retention_score"], 0)
        self.assertLessEqual(fin_res["retention_score"], 100)
        self.assertIn("quality_label", fin_res)
        self.assertIn("concepts_breakdown", fin_res)

        # 5. Check recall history
        history = get_recall_history(self.user_id)
        self.assertGreaterEqual(len(history), 1)

    def test_04_weak_concepts_summary(self):
        """Verify that weak concepts (<70% mastery) are identified for revision."""
        weak_list = get_weak_topics_summary(self.user_id)
        self.assertIsInstance(weak_list, list)
        for item in weak_list:
            self.assertLess(item["mastery_pct"], 70)
            self.assertIn("sub_concept", item)

    def test_05_flask_api_routes(self):
        """Verify that all Recall API endpoints respond correctly."""
        self.login_session()

        # GET /api/recall/pending
        res = self.client.get("/api/recall/pending")
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertTrue(data["success"])

        # POST /api/recall/schedule-custom
        res = self.client.post("/api/recall/schedule-custom", json={
            "topic": "Computer Networks - TCP Handshake",
            "delay_minutes": 0
        })
        self.assertEqual(res.status_code, 200)
        sched_data = res.get_json()
        self.assertTrue(sched_data["success"])
        task_id = sched_data["task_id"]

        # POST /api/recall/start
        res = self.client.post("/api/recall/start", json={
            "task_id": task_id,
            "question_count": 3
        })
        self.assertEqual(res.status_code, 200)
        start_data = res.get_json()
        self.assertTrue(start_data["success"])
        self.assertGreaterEqual(len(start_data["questions"]), 1)
        first_q = start_data["questions"][0]

        # POST /api/recall/submit-answer
        res = self.client.post("/api/recall/submit-answer", json={
            "question_id": first_q["id"],
            "student_answer": "SYN, SYN-ACK, ACK establishing reliable bidirectional communication."
        })
        self.assertEqual(res.status_code, 200)
        eval_data = res.get_json()
        self.assertTrue(eval_data["success"])

        # POST /api/recall/complete
        res = self.client.post("/api/recall/complete", json={
            "task_id": task_id
        })
        self.assertEqual(res.status_code, 200)
        comp_data = res.get_json()
        self.assertTrue(comp_data["success"])
        self.assertIn("retention_score", comp_data)

        # GET /api/recall/history
        res = self.client.get("/api/recall/history")
        self.assertEqual(res.status_code, 200)
        hist_data = res.get_json()
        self.assertTrue(hist_data["success"])

        # GET /api/recall/weak-concepts
        res = self.client.get("/api/recall/weak-concepts")
        self.assertEqual(res.status_code, 200)
        weak_data = res.get_json()
        self.assertTrue(weak_data["success"])

    def test_06_dashboard_and_reports_pages_render(self):
        """Verify that Dashboard, Reports, and Goals pages render without template errors."""
        self.login_session()

        # Dashboard
        res = self.client.get("/dashboard")
        self.assertEqual(res.status_code, 200)
        self.assertIn(b"Three-Pillar Learning Intelligence", res.data)
        self.assertIn(b"FocusSense Recall Engine", res.data)

        # Reports
        res = self.client.get("/reports")
        self.assertEqual(res.status_code, 200)
        self.assertIn(b"Delayed Recall Retention History", res.data)

        # Goals
        res = self.client.get("/goals")
        self.assertEqual(res.status_code, 200)
        self.assertIn(b"Delayed Recall Delay", res.data)

        # Status endpoint
        res = self.client.get("/status")
        self.assertEqual(res.status_code, 200)
        status_data = res.get_json()
        self.assertIn("running", status_data)
        self.assertIn("radar", status_data)


if __name__ == "__main__":
    unittest.main()
