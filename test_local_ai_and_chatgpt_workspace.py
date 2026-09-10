"""
FocusSense AI Comprehensive Upgrade Test Suite
=============================================
Tests:
1. Local Ollama AI Provider initialization & graceful offline handling.
2. Dynamic question generation across arbitrary topics (Java Datatypes, SQL Joins, SRS).
3. Semantic answer evaluation rubric.
4. Multi-turn conversation storage, title generation, and date grouping (Today/Yesterday/Older).
5. Anti-gaming AI study interaction tracking.
6. Recall Test dedicated UI workflow (start -> submit answer -> finalize).
"""

import os
import json
import sqlite3
import unittest
from datetime import datetime, timedelta
from app import app
from database import create_database, DATABASE
from ai_service import ai_service, OllamaAIProvider, GeminiAIProvider, LocalMockAIProvider
from recall_engine import schedule_recall_task, start_recall_test, submit_recall_answer, finalize_recall_test
from focus_intelligence import evaluate_ai_study_interaction


class TestFocusSenseAIUpgrade(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        create_database()
        cls.app = app
        cls.app.config["TESTING"] = True
        cls.client = cls.app.test_client()

        # Seed test user
        conn = sqlite3.connect(DATABASE)
        cursor = conn.cursor()
        cursor.execute("SELECT id FROM users WHERE username = 'test_ai_student'")
        row = cursor.fetchone()
        if not row:
            cursor.execute("""
                INSERT INTO users (fullname, email, username, password)
                VALUES ('AI Student', 'aistudent@focus.test', 'test_ai_student', 'testpass123')
            """)
            cls.user_id = cursor.lastrowid
        else:
            cls.user_id = row[0]

        # Insert standard goal
        cursor.execute("INSERT OR REPLACE INTO goals (user_id, daily_goal, weekly_goal, recall_delay_minutes) VALUES (?, 2, 10, 0)", (cls.user_id,))
        conn.commit()
        conn.close()

    def setUp(self):
        with self.client.session_transaction() as sess:
            sess["user_id"] = self.user_id
            sess["username"] = "test_ai_student"
            sess["fullname"] = "AI Student"

    # -------------------------------------------------------------------------
    # 1. Local AI & Ollama Provider Tests
    # -------------------------------------------------------------------------
    def test_ollama_provider_offline_graceful_message(self):
        """When Ollama daemon is offline, returns actionable setup instructions, never empty/fake responses."""
        provider = OllamaAIProvider(host="http://localhost:9999") # Non-existent port
        self.assertFalse(provider.is_available())
        reply_dict = provider.generate_reply(
            prompt="Explain polymorphism in Java",
            conversation_history=[],
            study_topic="Java Datatypes",
            student_name="AI Student"
        )
        self.assertFalse(reply_dict["success"])
        self.assertIn("Local AI Engine (Ollama) is Not Running", reply_dict["reply"])
        self.assertIn("ollama run llama3", reply_dict["reply"])

    def test_dynamic_question_generation_arbitrary_topics(self):
        """Questions must be generated dynamically for ANY topic (Java Datatypes, SQL Joins, SRS)."""
        topics_to_test = [
            "Java Datatypes",
            "SQL Joins",
            "Software Requirement Specification (SRS)",
            "Distributed Consensus Algorithms"
        ]
        for topic in topics_to_test:
            questions = ai_service.generate_quiz(topic, conversation_history=[], question_count=3)
            self.assertEqual(len(questions), 3, f"Failed for topic {topic}")
            self.assertTrue(all("question_text" in q and "sub_concept" in q for q in questions))
            self.assertTrue(any(topic.split()[0].lower() in q["question_text"].lower() or topic.split()[0].lower() in q["sub_concept"].lower() for q in questions))

    def test_semantic_answer_evaluations(self):
        """Verifies semantic answer evaluations return correct/partial/incorrect status."""
        # Correct answer
        q_text = "What is the difference between primitive and reference types in Java?"
        sub_c = "Primitive vs Reference Types"
        sample_ans = "Primitive types store values directly on stack, reference types store memory references pointing to objects on heap."
        
        eval_correct = ai_service.evaluate_answer(q_text, sub_c, sample_ans, "Primitives store values on the stack while reference types point to objects on the heap memory.")
        self.assertIn(eval_correct["status"], ["correct", "partial"])
        self.assertGreaterEqual(eval_correct["score"], 0.5)

        # Incorrect answer
        eval_incorrect = ai_service.evaluate_answer(q_text, sub_c, sample_ans, "Java is a programming language made by Sun Microsystems in 1995.")
        self.assertEqual(eval_incorrect["status"], "incorrect")
        self.assertEqual(eval_incorrect["score"], 0.0)

    # -------------------------------------------------------------------------
    # 2. Multi-Turn Conversation Memory & Grouping
    # -------------------------------------------------------------------------
    def test_multiturn_chat_and_title_generation(self):
        """Student starts a chat, follow-up conversation is stored, and title is generated."""
        res1 = self.client.post("/api/ai/chat", json={
            "message": "Explain Java Polymorphism in detail",
            "topic": "Java OOP"
        })
        self.assertEqual(res1.status_code, 200)
        data1 = res1.get_json()
        self.assertTrue(data1["success"])
        conv_id = data1["conversation_id"]
        self.assertIsNotNone(conv_id)
        self.assertIn("Polymorphism", data1["title"])

        # Follow-up turn referencing earlier conversation
        res2 = self.client.post("/api/ai/chat", json={
            "message": "I do not understand method overriding vs overloading in this context",
            "conversation_id": conv_id
        })
        self.assertEqual(res2.status_code, 200)
        data2 = res2.get_json()
        self.assertTrue(data2["success"])
        self.assertEqual(data2["conversation_id"], conv_id)

        # Fetch conversation history
        res_hist = self.client.get(f"/api/ai/conversations/{conv_id}")
        self.assertEqual(res_hist.status_code, 200)
        hist_data = res_hist.get_json()
        self.assertTrue(hist_data["success"])
        self.assertEqual(len(hist_data["messages"]), 4) # 2 user turns + 2 assistant turns

    def test_conversation_sidebar_date_grouping(self):
        """Conversations are grouped into Today, Yesterday, and Older."""
        conn = sqlite3.connect(DATABASE)
        cursor = conn.cursor()
        now = datetime.now()
        yesterday = (now - timedelta(days=1)).isoformat()
        older = (now - timedelta(days=10)).isoformat()

        cursor.execute("INSERT INTO ai_conversations (user_id, topic, title, created_at, updated_at) VALUES (?, 'Python', 'Yesterday Chat', ?, ?)",
                       (self.user_id, yesterday, yesterday))
        cursor.execute("INSERT INTO ai_conversations (user_id, topic, title, created_at, updated_at) VALUES (?, 'Databases', 'Older Chat', ?, ?)",
                       (self.user_id, older, older))
        conn.commit()
        conn.close()

        res = self.client.get("/api/ai/conversations")
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertTrue(data["success"])
        self.assertIn("grouped", data)
        self.assertIn("today", data["grouped"])
        self.assertIn("yesterday", data["grouped"])
        self.assertIn("older", data["grouped"])
        self.assertTrue(len(data["grouped"]["yesterday"]) >= 1)
        self.assertTrue(len(data["grouped"]["older"]) >= 1)

    # -------------------------------------------------------------------------
    # 3. Anti-Gaming AI Activity Tracking
    # -------------------------------------------------------------------------
    def test_anti_gaming_rules(self):
        """Idle open windows yield 0 focus credit; meaningful queries earn capped credit."""
        # Case A: 2 hours open, 0 questions asked -> 0 credit
        idle_eval = evaluate_ai_study_interaction(active_duration_sec=7200, question_count=0, meaningful_query_count=0)
        self.assertEqual(idle_eval["credit_score"], 0.0)
        self.assertEqual(idle_eval["capped_active_sec"], 0)
        self.assertEqual(idle_eval["status"], "idle_no_interaction")

        # Case B: 3 meaningful study queries -> bounded cognitive bonus
        active_eval = evaluate_ai_study_interaction(active_duration_sec=600, question_count=3, meaningful_query_count=3)
        self.assertGreater(active_eval["credit_score"], 0.0)
        self.assertLessEqual(active_eval["credit_score"], 5.0)
        self.assertEqual(active_eval["status"], "active_study")

    # -------------------------------------------------------------------------
    # 4. Recall Test Dedicated UI Workflow
    # -------------------------------------------------------------------------
    def test_recall_test_ui_workflow(self):
        """End-to-end recall test workflow: start -> submit answer -> finalize."""
        # 1. Schedule recall task
        task_info = schedule_recall_task(user_id=self.user_id, session_id=1, topic="Java Datatypes", delay_minutes=0)
        task_id = task_info["task_id"]
        self.assertIsNotNone(task_id)

        # 2. Start recall test (generates 3 questions)
        res_start = self.client.post("/api/recall/start", json={"task_id": task_id, "question_count": 3})
        self.assertEqual(res_start.status_code, 200)
        start_data = res_start.get_json()
        self.assertTrue(start_data["success"])
        questions = start_data["questions"]
        self.assertEqual(len(questions), 3)

        # 3. Submit answers to all questions
        for q in questions:
            res_ans = self.client.post("/api/recall/submit-answer", json={
                "question_id": q["id"],
                "student_answer": "In Java Datatypes, primitive variables store values directly on the stack while reference types point to objects on heap memory."
            })
            self.assertEqual(res_ans.status_code, 200)
            ans_data = res_ans.get_json()
            self.assertTrue(ans_data["success"])
            self.assertIn("evaluation", ans_data)

        # 4. Finalize recall test
        res_comp = self.client.post("/api/recall/complete", json={"task_id": task_id})
        self.assertEqual(res_comp.status_code, 200)
        comp_data = res_comp.get_json()
        self.assertTrue(comp_data["success"])
        self.assertIn("retention_score", comp_data)
        self.assertIn("quality_label", comp_data)
        self.assertIn("correct_count", comp_data)
        self.assertIn("partial_count", comp_data)
        self.assertIn("incorrect_count", comp_data)


if __name__ == "__main__":
    unittest.main()
