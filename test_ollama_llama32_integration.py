"""
FocusSense AI — Ollama llama3.2:3b Local Integration Test Suite
===============================================================
Verifies:
1. Live Ollama availability and model selection ('llama3.2:3b').
2. Multi-turn conversation memory with follow-up queries.
3. General academic & technical inquiries (algorithms, concepts, programming).
4. Topic context grounding without limiting general queries.
5. Dynamic 3-question recall quiz generation via Ollama llama3.2:3b.
6. Semantic rubric answer evaluation via Ollama llama3.2:3b.
7. Graceful diagnostic messaging when Ollama is offline.
8. Database persistence in SQLite (users.db).
"""

import os
import sqlite3
import unittest
from datetime import datetime
from app import app
from database import create_database, DATABASE
from ai_service import ai_service, OllamaAIProvider


class TestOllamaLlama32Integration(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        create_database()
        cls.app = app
        cls.app.config["TESTING"] = True
        cls.client = cls.app.test_client()

        # Seed isolated test user
        conn = sqlite3.connect(DATABASE)
        cursor = conn.cursor()
        cursor.execute("SELECT id FROM users WHERE username = 'ollama_test_student'")
        row = cursor.fetchone()
        if not row:
            cursor.execute("""
                INSERT INTO users (fullname, email, username, password)
                VALUES ('Llama Student', 'llamastudent@focus.test', 'ollama_test_student', 'testpass123')
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
            sess["username"] = "ollama_test_student"
            sess["fullname"] = "Llama Student"

    # -------------------------------------------------------------------------
    # 1. Ollama Provider Connection & Model Detection
    # -------------------------------------------------------------------------
    def test_01_ollama_provider_detection(self):
        """Ollama provider connects to http://localhost:11434 and selects llama3.2:3b."""
        provider = OllamaAIProvider()
        is_up = provider.is_available()
        self.assertTrue(is_up, "Local Ollama server is expected to be running on http://localhost:11434")

        active_model = provider.get_active_model()
        self.assertTrue("llama3.2:3b" in active_model or "llama3.2" in active_model,
                        f"Expected llama3.2:3b to be selected, found: {active_model}")
        
        info = provider.get_info()
        self.assertEqual(info["name"], "Ollama Local AI")
        self.assertEqual(info["type"], "local")
        self.assertTrue(info["active"])

    # -------------------------------------------------------------------------
    # 2. General Academic Inquiry & Multi-Turn Context Memory
    # -------------------------------------------------------------------------
    def test_02_general_academic_and_multiturn_dialogue(self):
        """FocusSense AI answers general CS questions and retains memory across follow-ups."""
        # Turn 1: Initial conceptual explanation
        res1 = self.client.post("/api/ai/chat", json={
            "message": "Explain Dijkstra's shortest path algorithm and its time complexity with a min-heap.",
            "topic": "Graph Algorithms"
        })
        self.assertEqual(res1.status_code, 200)
        data1 = res1.get_json()
        self.assertTrue(data1["success"])
        self.assertIn("Dijkstra", data1["reply"])
        conv_id = data1["conversation_id"]
        self.assertIsNotNone(conv_id)

        # Turn 2: Follow-up question relying on previous turn context
        res2 = self.client.post("/api/ai/chat", json={
            "message": "Can you give me a simple Python code snippet illustrating that?",
            "conversation_id": conv_id
        })
        self.assertEqual(res2.status_code, 200)
        data2 = res2.get_json()
        self.assertTrue(data2["success"])
        self.assertEqual(data2["conversation_id"], conv_id)
        # Should contain code or python references
        self.assertTrue("def " in data2["reply"] or "heapq" in data2["reply"] or "dijkstra" in data2["reply"].lower())

    # -------------------------------------------------------------------------
    # 3. Off-Topic & General Questions Are Not Blocked By Study Topic
    # -------------------------------------------------------------------------
    def test_03_topic_grounding_without_restricting_general_queries(self):
        """Active topic provides context, but the AI willingly answers general academic questions."""
        res = self.client.post("/api/ai/chat", json={
            "message": "How does photosynthesis convert light energy into chemical energy?",
            "topic": "Operating Systems"  # Active topic is OS, but question is Biology
        })
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertTrue(data["success"])
        # AI should explain photosynthesis rather than refusing or demanding OS context
        reply_lower = data["reply"].lower()
        self.assertTrue("chlorophyll" in reply_lower or "glucose" in reply_lower or "light" in reply_lower or "energy" in reply_lower)

    # -------------------------------------------------------------------------
    # 4. Dynamic Quiz Generation with llama3.2:3b
    # -------------------------------------------------------------------------
    def test_04_dynamic_quiz_generation(self):
        """Ollama dynamically generates 3 structured conceptual questions for a topic."""
        topic = "SQL Joins and Index Optimization"
        questions = ai_service.generate_quiz(topic=topic, question_count=3)
        self.assertEqual(len(questions), 3)
        for q in questions:
            self.assertIn("question_text", q)
            self.assertIn("sub_concept", q)
            self.assertIn("sample_answer", q)
            self.assertTrue(len(q["question_text"]) > 15)

    # -------------------------------------------------------------------------
    # 5. Semantic Rubric Evaluation with llama3.2:3b
    # -------------------------------------------------------------------------
    def test_05_semantic_rubric_evaluation(self):
        """Ollama evaluates a student answer semantically against ideal rubric."""
        q_text = "What is the difference between an INNER JOIN and a LEFT OUTER JOIN in relational databases?"
        sub_concept = "Inner vs Outer Joins"
        sample_rubric = "INNER JOIN returns only rows with matches in both tables, whereas LEFT OUTER JOIN returns all rows from the left table and matched rows from the right table, with NULLs for unmatched rows."
        
        # Test accurate student answer
        student_ans = "Inner join only includes records where keys match in both tables. Left join takes everything from the first table and fills missing right table columns with null."
        eval_res = ai_service.evaluate_answer(q_text, sub_concept, sample_rubric, student_ans)
        self.assertTrue(eval_res.get("success"))
        self.assertIn(eval_res.get("status"), ["correct", "partial"])
        self.assertGreaterEqual(eval_res.get("score", 0), 0.5)

    # -------------------------------------------------------------------------
    # 6. Graceful Offline Handling
    # -------------------------------------------------------------------------
    def test_06_offline_handling_diagnostic_guidance(self):
        """When Ollama is unreachable, returns clear setup guidance without crashing."""
        offline_provider = OllamaAIProvider(host="http://localhost:9999")
        self.assertFalse(offline_provider.is_available())

        res = offline_provider.generate_reply(
            prompt="Explain polymorphism",
            conversation_history=[],
            study_topic="Java",
            student_name="Llama Student"
        )
        self.assertFalse(res["success"])
        self.assertIn("Local AI Engine (Ollama) is Not Running", res["reply"])
        self.assertIn("ollama run", res["reply"])

    # -------------------------------------------------------------------------
    # 7. SQLite Conversation & Message Persistence
    # -------------------------------------------------------------------------
    def test_07_database_message_persistence(self):
        """All user and assistant turns persist in SQLite ai_messages and ai_conversations."""
        conn = sqlite3.connect(DATABASE)
        cursor = conn.cursor()
        cursor.execute("SELECT count(*) FROM ai_messages m JOIN ai_conversations c ON m.conversation_id = c.id WHERE c.user_id = ?", (self.user_id,))
        count = cursor.fetchone()[0]
        conn.close()
        self.assertGreaterEqual(count, 4, "Expected at least 4 messages stored for test student")


if __name__ == "__main__":
    unittest.main()
