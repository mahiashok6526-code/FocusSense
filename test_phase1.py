import os
import sys
import time
import sqlite3
import json

# Ensure FocusSense is in python path
sys.path.insert(0, r"c:\Users\mahia\OneDrive\Documents\FocusSense")
sys.stdout.reconfigure(encoding='utf-8')

from app import app, get_db
from hardware.sensor import sensor_controller

def run_phase1_tests():
    print("=" * 70)
    print("FocusSense AI — Phase 1 Comprehensive Test Suite")
    print("=" * 70)

    client = app.test_client()

    # -------------------------------------------------------------------------
    # 1. Database Schema & Data Preservation Verification
    # -------------------------------------------------------------------------
    print("\n[TEST 1] Verifying Database Schema & Historical Data Integrity...")
    conn = get_db()
    c = conn.cursor()
    
    # Check new tables exist
    c.execute("SELECT name FROM sqlite_master WHERE type='table' AND name IN ('study_topics', 'ai_conversations', 'ai_messages')")
    new_tables = {r[0] for r in c.fetchall()}
    assert "study_topics" in new_tables, "FAIL: study_topics table missing"
    assert "ai_conversations" in new_tables, "FAIL: ai_conversations table missing"
    assert "ai_messages" in new_tables, "FAIL: ai_messages table missing"
    
    # Check study_sessions.topic column exists
    c.execute("PRAGMA table_info(study_sessions)")
    columns = [r["name"] for r in c.fetchall()]
    assert "topic" in columns, "FAIL: topic column missing in study_sessions"
    
    # Check historical user & session data is 100% preserved
    c.execute("SELECT count(*) FROM users")
    user_count = c.fetchone()[0]
    c.execute("SELECT count(*) FROM study_sessions")
    session_count = c.fetchone()[0]
    conn.close()
    
    assert user_count >= 3, f"FAIL: Expected at least 3 users, found {user_count}"
    assert session_count >= 26, f"FAIL: Expected at least 26 sessions, found {session_count}"
    print(f"PASS: Schema verified. Preserved {user_count} users and {session_count} existing study sessions.")

    # -------------------------------------------------------------------------
    # 2. Existing Routes & Regression Verification
    # -------------------------------------------------------------------------
    print("\n[TEST 2] Verifying Core Application Routes...")
    with client.session_transaction() as s:
        s["user_id"] = 1
        s["fullname"] = "ashok"
        s["username"] = "ashok1"

    for endpoint in ["/dashboard", "/reports", "/parent", "/goals", "/settings"]:
        res = client.get(endpoint)
        assert res.status_code == 200, f"FAIL: Route {endpoint} returned status {res.status_code}"
    print("PASS: /dashboard, /reports, /parent, /goals, /settings all returned HTTP 200.")

    # -------------------------------------------------------------------------
    # 3. Topic Grounding & Starting Study Session with Topic
    # -------------------------------------------------------------------------
    print("\n[TEST 3] Testing Topic Grounding at Study Session Start...")
    test_topic = "Operating Systems - Process Scheduling"
    res_start = client.post("/start-study", json={"topic": test_topic})
    assert res_start.status_code == 200, f"FAIL: /start-study returned {res_start.status_code}"
    data_start = res_start.get_json()
    assert data_start["success"] is True, "FAIL: start_study failed"
    assert data_start["topic"] == test_topic, f"FAIL: Expected topic {test_topic}, got {data_start.get('topic')}"
    assert data_start["conversation_id"] is not None, "FAIL: Conversation ID missing from start_study response"
    conv_id = data_start["conversation_id"]
    print(f"PASS: Session started with topic: '{test_topic}', conversation ID: {conv_id}.")

    # -------------------------------------------------------------------------
    # 4. In-Session Multi-Turn AI Chat Interaction
    # -------------------------------------------------------------------------
    print("\n[TEST 4] Testing In-Session Multi-Turn AI Chat Companion...")
    
    # Question 1: What is FCFS scheduling?
    res_chat1 = client.post("/api/ai/chat", json={
        "message": "What is FCFS scheduling?",
        "topic": test_topic
    })
    assert res_chat1.status_code == 200, f"FAIL: /api/ai/chat returned {res_chat1.status_code}"
    data_chat1 = res_chat1.get_json()
    assert data_chat1["success"] is True, "FAIL: Chat message 1 was not successful"
    assert len(data_chat1["reply"]) > 20, "FAIL: AI reply was empty or too short"
    print(f"PASS: AI Chat Q1 responded using provider '{data_chat1['provider']}' ({data_chat1['model']}).")

    # Question 2: Explain it simply (ELI5)
    res_chat2 = client.post("/api/ai/chat", json={
        "message": "Explain it simply.",
        "topic": test_topic
    })
    assert res_chat2.status_code == 200
    data_chat2 = res_chat2.get_json()
    assert data_chat2["success"] is True
    print("PASS: AI Chat Q2 (Explain simply) successfully processed.")

    # Question 3: Grounded query ("What should I understand first?")
    res_chat3 = client.post("/api/ai/chat", json={
        "message": "What should I understand first?",
        "topic": test_topic
    })
    assert res_chat3.status_code == 200
    data_chat3 = res_chat3.get_json()
    assert data_chat3["success"] is True
    assert len(data_chat3["reply"]) > 20, "FAIL: AI reply was empty or too short"
    print("PASS: AI Chat Q3 (Grounded roadmap) correctly generated pedagogical response.")

    # -------------------------------------------------------------------------
    # 5. Database Message & Conversation Storage Verification
    # -------------------------------------------------------------------------
    print("\n[TEST 5] Verifying SQLite Conversation & Message Storage...")
    conn = get_db()
    c = conn.cursor()
    c.execute("SELECT * FROM ai_conversations WHERE id = ?", (conv_id,))
    conv_row = c.fetchone()
    assert conv_row is not None, "FAIL: Conversation row not found in database"
    assert conv_row["topic"] == test_topic, "FAIL: Topic mismatch in ai_conversations"

    c.execute("SELECT sender, message, timestamp FROM ai_messages WHERE conversation_id = ? ORDER BY id ASC", (conv_id,))
    msg_rows = c.fetchall()
    conn.close()

    # Should have initial greeting + 3 user messages + 3 assistant replies = 7 messages
    assert len(msg_rows) >= 7, f"FAIL: Expected >= 7 messages in conversation, found {len(msg_rows)}"
    user_msgs = [r["message"] for r in msg_rows if r["sender"] == "user"]
    assert "What is FCFS scheduling?" in user_msgs, "FAIL: User message 1 missing from DB"
    assert "Explain it simply." in user_msgs, "FAIL: User message 2 missing from DB"
    assert "What should I understand first?" in user_msgs, "FAIL: User message 3 missing from DB"
    print(f"PASS: Conversation and all {len(msg_rows)} turns accurately stored in SQLite.")

    # -------------------------------------------------------------------------
    # 6. Chat History Route & Frontend Loading
    # -------------------------------------------------------------------------
    print("\n[TEST 6] Testing /api/ai/history Endpoint...")
    res_hist = client.get("/api/ai/history")
    assert res_hist.status_code == 200
    data_hist = res_hist.get_json()
    assert len(data_hist["messages"]) >= 7, "FAIL: History endpoint did not return stored messages"
    assert data_hist["active_topic"] == test_topic
    print(f"PASS: /api/ai/history returned {len(data_hist['messages'])} messages with active topic: {data_hist['active_topic']}.")

    # -------------------------------------------------------------------------
    # 7. Radar Simulation, Absence Tracking, and Stopping Study Session
    # -------------------------------------------------------------------------
    print("\n[TEST 7] Testing Study Session Lifecycle & Topic Attachment...")
    sensor_controller.set_simulated_presence(True)
    res_sim = client.get("/status")
    data_sim = res_sim.get_json()
    assert data_sim["running"] is True
    assert data_sim["radar"] == "PRESENT"

    time.sleep(1.0)
    res_stop = client.post("/stop-study")
    assert res_stop.status_code == 200
    data_stop = res_stop.get_json()
    assert data_stop["success"] is True
    assert data_stop["topic"] == test_topic
    saved_session_focus = data_stop["focus_score"]
    print(f"PASS: Session stopped. Focus Score: {saved_session_focus}%, Topic: '{data_stop['topic']}' saved.")

    # -------------------------------------------------------------------------
    # 8. Server Restart & Persistence Resilience
    # -------------------------------------------------------------------------
    print("\n[TEST 8] Testing Persistence Across Fresh Database Connections...")
    conn_fresh = sqlite3.connect("users.db")
    conn_fresh.row_factory = sqlite3.Row
    c_f = conn_fresh.cursor()
    c_f.execute("SELECT topic, focus_score FROM study_sessions WHERE user_id = 1 ORDER BY id DESC LIMIT 1")
    latest_saved_session = c_f.fetchone()
    assert latest_saved_session is not None
    assert latest_saved_session["topic"] == test_topic, "FAIL: Topic not saved in study_sessions row"

    c_f.execute("SELECT count(*) FROM ai_messages WHERE conversation_id = ?", (conv_id,))
    count_persisted_msgs = c_f.fetchone()[0]
    conn_fresh.close()
    assert count_persisted_msgs >= 7, "FAIL: AI messages lost after fresh connection"
    print(f"PASS: Fresh DB connection verified. Topic '{latest_saved_session['topic']}' and {count_persisted_msgs} chat messages persisted.")

    # -------------------------------------------------------------------------
    # 9. Dashboard Rendering with AI Workspace Link & Topic Controls
    # -------------------------------------------------------------------------
    print("\n[TEST 9] Verifying Dashboard & Dedicated FocusSense AI Workspace Rendering...")
    res_dash = client.get("/dashboard")
    assert res_dash.status_code == 200
    html_dash = res_dash.data.decode("utf-8")
    assert "studyTopicInput" in html_dash, "FAIL: studyTopicInput missing from dashboard HTML"
    assert "/ai" in html_dash, "FAIL: /ai link missing from dashboard HTML"

    res_ai = client.get("/ai")
    assert res_ai.status_code == 200
    html_ai = res_ai.data.decode("utf-8")
    assert "FocusSense AI" in html_ai, "FAIL: FocusSense AI header not rendered in AI workspace"
    print("PASS: Dashboard HTML contains Study Topic input and links to dedicated FocusSense AI workspace.")

    print("\n" + "=" * 70)
    print("ALL PHASE 1 VERIFICATION TESTS PASSED SUCCESSFULLY (9/9)!")
    print("=" * 70)

if __name__ == "__main__":
    run_phase1_tests()
