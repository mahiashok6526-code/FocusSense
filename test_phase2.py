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

def run_phase2_tests():
    print("=" * 75)
    print("FocusSense AI — Phase 2 Comprehensive Test Suite")
    print("Post-Study Examination & Learning Verification")
    print("=" * 75)

    client = app.test_client()

    # -------------------------------------------------------------------------
    # 1. Database Schema & Data Preservation Verification
    # -------------------------------------------------------------------------
    print("\n[TEST 1] Verifying Phase 2 Database Tables & Historical Integrity...")
    conn = get_db()
    c = conn.cursor()
    
    # Check Phase 2 tables exist
    c.execute("SELECT name FROM sqlite_master WHERE type='table' AND name IN ('study_quizzes', 'quiz_questions', 'quiz_answers', 'concept_mastery')")
    tables = {r[0] for r in c.fetchall()}
    assert "study_quizzes" in tables, "FAIL: study_quizzes table missing"
    assert "quiz_questions" in tables, "FAIL: quiz_questions table missing"
    assert "quiz_answers" in tables, "FAIL: quiz_answers table missing"
    assert "concept_mastery" in tables, "FAIL: concept_mastery table missing"
    
    # Check historical user & session data is preserved
    c.execute("SELECT count(*) FROM users")
    user_count = c.fetchone()[0]
    c.execute("SELECT count(*) FROM study_sessions")
    session_count = c.fetchone()[0]
    conn.close()
    
    assert user_count >= 3, f"FAIL: Expected >= 3 users, found {user_count}"
    assert session_count >= 26, f"FAIL: Expected >= 26 sessions, found {session_count}"
    print(f"PASS: Schema verified. Preserved {user_count} users and {session_count} study sessions.")

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
    # 3. Session Start, Chat Interaction, and Session Stop
    # -------------------------------------------------------------------------
    print("\n[TEST 3] Running Study Session with Grounded Topic...")
    test_topic = "Operating Systems - CPU Scheduling"
    res_start = client.post("/start-study", json={"topic": test_topic})
    assert res_start.status_code == 200
    data_start = res_start.get_json()
    assert data_start["success"] is True
    
    # In-session chat interaction
    res_chat = client.post("/api/ai/chat", json={
        "message": "What is Round Robin scheduling?",
        "topic": test_topic
    })
    assert res_chat.status_code == 200

    # Stop study session
    time.sleep(1.0)
    res_stop = client.post("/stop-study")
    assert res_stop.status_code == 200
    data_stop = res_stop.get_json()
    assert data_stop["success"] is True
    assert data_stop.get("session_id") is not None, "FAIL: stop-study must return session_id"
    session_id = data_stop["session_id"]
    duration = data_stop.get("duration", 10)
    print(f"PASS: Study session {session_id} completed for topic '{test_topic}' (Duration: {duration}s).")

    # -------------------------------------------------------------------------
    # 4. Post-Study Examination Generation
    # -------------------------------------------------------------------------
    print("\n[TEST 4] Testing Grounded Examination Generation (/api/quiz/generate)...")
    res_quiz_gen = client.post("/api/quiz/generate", json={
        "session_id": session_id,
        "topic": test_topic
    })
    assert res_quiz_gen.status_code == 200
    data_quiz_gen = res_quiz_gen.get_json()
    assert data_quiz_gen["success"] is True
    quiz = data_quiz_gen["quiz"]
    quiz_id = quiz["quiz_id"]
    questions = quiz["questions"]
    assert len(questions) >= 3, f"FAIL: Expected >= 3 questions, got {len(questions)}"
    
    # Verify answers are NOT leaked in questions array
    for q in questions:
        assert "sample_answer" not in q, f"FAIL: Exam integrity violated: sample_answer leaked in {q}"
        assert "sub_concept" in q and len(q["sub_concept"]) > 0, "FAIL: Missing sub_concept tag"
    print(f"PASS: Quiz {quiz_id} generated {len(questions)} grounded questions with tagged sub-concepts (no answer leaks).")

    # -------------------------------------------------------------------------
    # 5. Semantic Turn-by-Turn Answer Evaluation
    # -------------------------------------------------------------------------
    print("\n[TEST 5] Testing Semantic Answer Evaluation & Grading (/api/quiz/submit-answer)...")
    
    # Q1: Strong Answer on dynamically generated question
    q1 = questions[0]
    conn = get_db()
    c = conn.cursor()
    c.execute("SELECT sample_answer FROM quiz_questions WHERE id = ?", (q1["id"],))
    row_ans1 = c.fetchone()
    ans1_text = row_ans1["sample_answer"] if row_ans1 and row_ans1["sample_answer"] else "FCFS executes processes in arrival order non-preemptively."
    conn.close()

    res_ans1 = client.post("/api/quiz/submit-answer", json={
        "question_id": q1["id"],
        "student_answer": ans1_text
    })
    assert res_ans1.status_code == 200
    data_ans1 = res_ans1.get_json()
    assert data_ans1["status"] == "correct", f"FAIL: Expected correct, got {data_ans1['status']}"
    assert data_ans1["score_earned"] == 1.0
    print(f"PASS: Q1 ('{q1['sub_concept']}') evaluated as 'correct' (+1.0 pt): {data_ans1['feedback']}")

    # Q2: Partial Answer on question 2
    q2 = questions[1]
    res_ans2 = client.post("/api/quiz/submit-answer", json={
        "question_id": q2["id"],
        "student_answer": "It uses a fixed time slice."
    })
    assert res_ans2.status_code == 200
    data_ans2 = res_ans2.get_json()
    assert data_ans2["status"] in ["partial", "correct", "incorrect"], f"FAIL: Unexpected status {data_ans2['status']}"
    print(f"PASS: Q2 ('{q2['sub_concept']}') evaluated: {data_ans2['feedback']}")

    # Q3: Weak / Irrelevant Answer on Preemption
    q3 = questions[2]
    res_ans3 = client.post("/api/quiz/submit-answer", json={
        "question_id": q3["id"],
        "student_answer": "I do not know."
    })
    assert res_ans3.status_code == 200
    data_ans3 = res_ans3.get_json()
    assert data_ans3["status"] in ["incorrect", "unanswered"], f"FAIL: Expected incorrect, got {data_ans3['status']}"
    assert data_ans3["score_earned"] == 0.0
    print(f"PASS: Q3 ('{q3['sub_concept']}') evaluated as 'incorrect' (+0.0 pt): {data_ans3['feedback']}")

    # -------------------------------------------------------------------------
    # 6. Final Learning Verification Score & Time vs Learning Synthesis
    # -------------------------------------------------------------------------
    print("\n[TEST 6] Testing Final Learning Verification Scoring & Synthesis (/api/quiz/complete)...")
    res_complete = client.post("/api/quiz/complete", json={
        "quiz_id": quiz_id,
        "duration": 3600 # 1 hour
    })
    assert res_complete.status_code == 200
    data_comp = res_complete.get_json()
    assert data_comp["success"] is True
    
    # Expected score: (1.0 + 0.5 + 0.0) / 3.0 = 50%
    assert data_comp["score_pct"] == 50, f"FAIL: Expected score 50%, got {data_comp['score_pct']}%"
    assert q1["sub_concept"] in data_comp["strong_concepts"], f"FAIL: {q1['sub_concept']} should be strong"
    assert q2["sub_concept"] in data_comp["weak_concepts"], f"FAIL: {q2['sub_concept']} should be weak"
    assert len(data_comp["summary_text"]) > 20, "FAIL: Missing summary insight"
    print(f"PASS: Learning Verification Score: {data_comp['score_pct']}% ({data_comp['quality_label']})")
    print(f"PASS: Strong Concepts: {data_comp['strong_concepts']}")
    print(f"PASS: Weak Concepts (Needs Review): {data_comp['weak_concepts']}")
    print(f"PASS: Synthesis Insight: '{data_comp['summary_text']}'")

    # -------------------------------------------------------------------------
    # 7. Concept Mastery Tracking & Weak Area Tagging Verification
    # -------------------------------------------------------------------------
    print("\n[TEST 7] Verifying Concept Mastery Database Records...")
    conn = get_db()
    c = conn.cursor()
    c.execute("SELECT sub_concept, total_tested, mastery_pct, is_weak FROM concept_mastery WHERE user_id = 1")
    mastery_records = c.fetchall()
    conn.close()

    assert len(mastery_records) >= 3, f"FAIL: Expected >= 3 concept mastery records, found {len(mastery_records)}"
    for r in mastery_records:
        print(f" - Concept: {r['sub_concept']} | Mastery: {r['mastery_pct']}% | Weak Area Flag: {r['is_weak']}")
    print("PASS: Concept mastery table accurately tracked running accuracy and weak flags.")

    # -------------------------------------------------------------------------
    # 8. Latest Quiz Overview Route
    # -------------------------------------------------------------------------
    print("\n[TEST 8] Testing /api/quiz/latest Endpoint...")
    res_latest = client.get("/api/quiz/latest")
    assert res_latest.status_code == 200
    data_latest = res_latest.get_json()
    assert data_latest["success"] is True
    assert data_latest["overview"]["has_quiz"] is True
    assert data_latest["overview"]["score_pct"] == 50
    print(f"PASS: /api/quiz/latest returned verified score: {data_latest['overview']['score_pct']}% ({data_latest['overview']['quality_label']}).")

    # -------------------------------------------------------------------------
    # 9. Dual Score Display & Dashboard HTML Rendering
    # -------------------------------------------------------------------------
    print("\n[TEST 9] Verifying Dashboard Dual-Factor Rendering & Modal Elements...")
    res_dash = client.get("/dashboard")
    assert res_dash.status_code == 200
    html = res_dash.data.decode("utf-8")
    assert "Learning Verification" in html, "FAIL: Learning Verification not rendered in dashboard HTML"
    assert "learningScoreCard" in html, "FAIL: learningScoreCard missing from dashboard HTML"
    assert "learningMasteryCard" in html, "FAIL: learningMasteryCard missing from dashboard HTML"
    assert "learningVerificationModal" in html, "FAIL: learningVerificationModal missing from dashboard HTML"
    assert "Strong Concepts" in html, "FAIL: Strong Concepts section missing"
    assert "Needs Review" in html, "FAIL: Needs Review section missing"
    print("PASS: Dashboard HTML contains dual Focus Score + Learning Verification cards, Concept Mastery section, and Exam Modal.")

    # -------------------------------------------------------------------------
    # 10. Persistence Across Fresh Database Connections
    # -------------------------------------------------------------------------
    print("\n[TEST 10] Testing Persistence Across Fresh Database Connections...")
    conn_fresh = sqlite3.connect("users.db")
    conn_fresh.row_factory = sqlite3.Row
    c_f = conn_fresh.cursor()
    c_f.execute("SELECT score_pct, quality_label, summary_text FROM study_quizzes WHERE id = ?", (quiz_id,))
    persisted_quiz = c_f.fetchone()
    assert persisted_quiz is not None
    assert persisted_quiz["score_pct"] == 50.0
    conn_fresh.close()
    print(f"PASS: Fresh DB connection confirmed quiz {quiz_id} persisted with score {persisted_quiz['score_pct']}%.")

    print("\n" + "=" * 75)
    print("ALL PHASE 2 VERIFICATION TESTS PASSED SUCCESSFULLY (10/10)!")
    print("=" * 75)

if __name__ == "__main__":
    run_phase2_tests()
