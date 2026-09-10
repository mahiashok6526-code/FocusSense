"""
Automated Test Suite for FocusSense AI Dedicated Workspace (/ai)
Verifies:
1. Workspace route GET /ai
2. Conversation CRUD API (new, list grouped, get, rename, archive, delete)
3. Full-text search across titles and messages
4. Auto-title generation from first user turn
5. Multi-conversation isolation and persistence
6. Navigation links across all pages (/dashboard, /reports, /goals, /settings, /parent)
7. Non-regression on Radar Simulation & Focus Score lifecycle
"""

import os
import sys
import unittest
import json
import sqlite3

from app import app, get_db
from ai_service import ai_service


class TestFocusSenseAIWorkspace(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        app.config["TESTING"] = True
        app.config["SECRET_KEY"] = "test-secret-key-123"
        cls.client = app.test_client()

        # Create or fetch a test student user
        conn = get_db()
        cursor = conn.cursor()
        cursor.execute("SELECT id, fullname, username FROM users WHERE username = 'teststudent_ai'")
        user = cursor.fetchone()
        if not user:
            from werkzeug.security import generate_password_hash
            cursor.execute("""
                INSERT INTO users (fullname, email, username, password)
                VALUES (?, ?, ?, ?)
            """, ("AI Test Student", "ai_test@focussense.io", "teststudent_ai", generate_password_hash("password123")))
            conn.commit()
            cursor.execute("SELECT id, fullname, username FROM users WHERE username = 'teststudent_ai'")
            user = cursor.fetchone()

        cls.user_id = user["id"]
        cls.fullname = user["fullname"]
        cls.username = user["username"]
        conn.close()

    def setUp(self):
        # Log in test user in session
        with self.client.session_transaction() as sess:
            sess["user_id"] = self.user_id
            sess["fullname"] = self.fullname
            sess["username"] = self.username

    def test_01_workspace_route(self):
        """Verify GET /ai renders the full-screen AI workspace."""
        res = self.client.get("/ai")
        self.assertEqual(res.status_code, 200)
        html = res.get_data(as_text=True)
        self.assertIn("FocusSense AI", html)
        self.assertIn("welcome-avatar-orb", html)
        self.assertIn("btn-new-chat", html)
        self.assertIn("ai-search-input", html)
        self.assertIn("aiChatTextarea", html)

    def test_02_all_navigation_links(self):
        """Verify all workspace sidebars link to /ai and student pages have Monitoring section."""
        student_routes = ["/dashboard", "/reports", "/goals", "/settings"]
        for r in student_routes:
            res = self.client.get(r)
            self.assertEqual(res.status_code, 200, f"Failed on route {r}")
            html = res.get_data(as_text=True)
            self.assertIn('/ai', html, f"Route {r} missing link to /ai")
            self.assertIn("FocusSense AI", html, f"Route {r} missing FocusSense AI text")
            self.assertIn("Monitoring", html, f"Route {r} missing Monitoring section")

        # Parent route check
        res_parent = self.client.get("/parent")
        self.assertEqual(res_parent.status_code, 200)
        html_p = res_parent.get_data(as_text=True)
        self.assertIn('/ai', html_p, "Parent route missing link to /ai")
        self.assertIn("FocusSense AI", html_p, "Parent route missing FocusSense AI text")

    def test_03_create_new_conversation(self):
        """Verify POST /api/ai/conversations/new creates conversation with default title."""
        res = self.client.post("/api/ai/conversations/new", json={
            "topic": "Operating Systems - Process Scheduling"
        })
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertTrue(data["success"])
        self.assertIn("conversation", data)
        self.assertEqual(data["conversation"]["title"], "New Study Chat")
        self.assertEqual(data["conversation"]["topic"], "Operating Systems - Process Scheduling")
        self.assertIsNotNone(data["conversation"]["id"])

    def test_04_chat_and_auto_title_generation(self):
        """Verify sending a message auto-generates a smart title from 'New Study Chat'."""
        # 1. Create a fresh conversation
        res_new = self.client.post("/api/ai/conversations/new", json={
            "topic": "Operating Systems - CPU Scheduling"
        })
        conv_id = res_new.get_json()["conversation"]["id"]

        # 2. Send first message
        res_chat = self.client.post("/api/ai/chat", json={
            "conversation_id": conv_id,
            "message": "What is Round Robin scheduling with time quantum?"
        })
        self.assertEqual(res_chat.status_code, 200)
        chat_data = res_chat.get_json()
        self.assertTrue(chat_data["success"])
        self.assertIn("reply", chat_data)
        self.assertTrue(len(chat_data["reply"]) > 20)
        self.assertEqual(chat_data["conversation_id"], conv_id)
        # Verify title was auto-updated from 'New Study Chat'
        self.assertNotEqual(chat_data["title"], "New Study Chat")
        self.assertTrue(chat_data["title_updated"])

        # 3. Retrieve conversation history
        res_hist = self.client.get(f"/api/ai/conversations/{conv_id}")
        self.assertEqual(res_hist.status_code, 200)
        hist_data = res_hist.get_json()
        self.assertTrue(hist_data["success"])
        self.assertEqual(len(hist_data["messages"]), 2)  # 1 user + 1 assistant
        self.assertEqual(hist_data["messages"][0]["sender"], "user")
        self.assertEqual(hist_data["messages"][1]["sender"], "assistant")

    def test_05_list_grouped_conversations(self):
        """Verify GET /api/ai/conversations returns non-archived conversations grouped by date."""
        res = self.client.get("/api/ai/conversations")
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertTrue(data["success"])
        self.assertIn("conversations", data)
        self.assertIn("grouped", data)
        self.assertIn("today", data["grouped"])
        self.assertTrue(len(data["grouped"]["today"]) >= 1)

    def test_06_search_conversations(self):
        """Verify full-text search across titles, topics, and message content."""
        # Create a unique searchable topic & message
        res_new = self.client.post("/api/ai/conversations/new", json={
            "topic": "Quantum Computing Superposition"
        })
        conv_id = res_new.get_json()["conversation"]["id"]

        self.client.post("/api/ai/chat", json={
            "conversation_id": conv_id,
            "message": "Explain how qubits achieve entanglement in quantum circuits"
        })

        # Search by keyword in message
        res_search = self.client.get("/api/ai/conversations/search?q=entanglement")
        self.assertEqual(res_search.status_code, 200)
        data = res_search.get_json()
        self.assertTrue(data["success"])
        matched_ids = [c["id"] for c in data["conversations"]]
        self.assertIn(conv_id, matched_ids)

        # Search by keyword in topic
        res_search2 = self.client.get("/api/ai/conversations/search?q=Superposition")
        self.assertEqual(res_search2.status_code, 200)
        matched_ids2 = [c["id"] for c in res_search2.get_json()["conversations"]]
        self.assertIn(conv_id, matched_ids2)

    def test_07_rename_archive_delete_conversation(self):
        """Verify rename, archive, and delete operations."""
        # 1. Create conversation
        res_new = self.client.post("/api/ai/conversations/new", json={"topic": "Test Lifecycle"})
        conv_id = res_new.get_json()["conversation"]["id"]

        # 2. Rename
        res_rename = self.client.post(f"/api/ai/conversations/{conv_id}/rename", json={
            "title": "Renamed Custom Study Topic"
        })
        self.assertEqual(res_rename.status_code, 200)
        self.assertEqual(res_rename.get_json()["title"], "Renamed Custom Study Topic")

        # 3. Archive
        res_archive = self.client.post(f"/api/ai/conversations/{conv_id}/archive")
        self.assertEqual(res_archive.status_code, 200)
        # Ensure archived conversation does not appear in active list
        res_list = self.client.get("/api/ai/conversations")
        active_ids = [c["id"] for c in res_list.get_json()["conversations"]]
        self.assertNotIn(conv_id, active_ids)

        # 4. Delete
        res_del = self.client.delete(f"/api/ai/conversations/{conv_id}")
        self.assertEqual(res_del.status_code, 200)
        # Verify completely deleted from database
        conn = get_db()
        cursor = conn.cursor()
        cursor.execute("SELECT id FROM ai_conversations WHERE id = ?", (conv_id,))
        self.assertIsNone(cursor.fetchone())
        conn.close()

    def test_08_radar_and_focus_score_integrity(self):
        """Verify start study, radar simulation, and focus score lifecycle are unaffected."""
        # Start study
        res_start = self.client.post("/start-study", json={"topic": "Compiler Design - Lexical Analysis"})
        self.assertEqual(res_start.status_code, 200)

        # Check radar status via /status endpoint
        res_status = self.client.get("/status")
        self.assertEqual(res_status.status_code, 200)
        status_data = res_status.get_json()
        self.assertIn("radar", status_data)
        self.assertIn("presence", status_data)

        # Stop study
        res_stop = self.client.post("/stop-study")
        self.assertEqual(res_stop.status_code, 200)
        stop_data = res_stop.get_json()
        self.assertTrue(stop_data["success"])
        self.assertIn("focus_score", stop_data)


if __name__ == "__main__":
    unittest.main(verbosity=2)
