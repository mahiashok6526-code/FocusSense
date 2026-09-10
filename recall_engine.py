"""
FocusSense Recall Engine (Phase 3)
==================================
Handles delayed, spaced recall task scheduling, question generation,
semantic evaluation, retention scoring, and weak-concept revision tracking.

Core Principles:
1. Physical Presence (Radar telemetry) != Knowledge Retention.
2. Delayed active recall tests verify actual learning after a study session.
3. Automatically tracks sub-concept mastery and detects weak concepts (<70%).
"""

import sqlite3
from datetime import datetime, timedelta
from ai_service import ai_service
from learning_engine import (
    calculate_retention_score,
    update_topic_mastery,
    get_topic_mastery_list,
    get_weak_topics,
    get_student_learning_profile,
    get_adaptive_quiz_config,
    compute_learning_focus_score,
    compute_unified_focus_score,
    log_adaptive_learning_event,
    record_learning_state,
    record_adaptive_action,
    record_adaptive_reward,
    calculate_adaptive_reward,
    get_latest_learning_state,
    select_adaptive_action,
)

from database import DATABASE, DB_PATH


def get_db():
    conn = sqlite3.connect(DATABASE, timeout=30.0)
    conn.row_factory = sqlite3.Row
    return conn


def schedule_recall_task(user_id: int, session_id: int, topic: str, delay_minutes: int = None) -> dict:
    """
    Schedule a delayed recall task upon completing a study session.
    Default delay is fetched from user goals setting (or 10 minutes default).
    """
    conn = get_db()
    cursor = conn.cursor()

    if delay_minutes is None:
        cursor.execute("SELECT recall_delay_minutes FROM goals WHERE user_id = ?", (user_id,))
        grow = cursor.fetchone()
        delay_minutes = grow["recall_delay_minutes"] if grow and grow["recall_delay_minutes"] is not None else 10

    now = datetime.now()
    scheduled_dt = now + timedelta(minutes=delay_minutes)
    now_iso = now.isoformat()
    sched_iso = scheduled_dt.isoformat()
    status = "READY" if delay_minutes <= 0 else "PENDING"

    cursor.execute("""
        INSERT INTO recall_tasks (user_id, session_id, topic, delay_minutes, scheduled_for, status, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (user_id, session_id, topic, delay_minutes, sched_iso, status, now_iso))
    task_id = cursor.lastrowid
    conn.commit()
    conn.close()

    return {
        "success": True,
        "task_id": task_id,
        "topic": topic,
        "session_id": session_id,
        "delay_minutes": delay_minutes,
        "scheduled_for": sched_iso,
        "status": status
    }


def get_pending_and_due_tasks(user_id: int) -> dict:
    """
    Retrieve active recall tasks, transitioning any PENDING task whose scheduled_for <= now to READY.
    """
    conn = get_db()
    cursor = conn.cursor()
    now_iso = datetime.now().isoformat()

    # Transition due tasks to READY
    cursor.execute("""
        UPDATE recall_tasks
        SET status = 'READY'
        WHERE user_id = ? AND status = 'PENDING' AND scheduled_for <= ?
    """, (user_id, now_iso))
    conn.commit()

    # Query active/due tasks (most recent first)
    cursor.execute("""
        SELECT rt.id, rt.session_id, rt.topic, rt.delay_minutes, rt.scheduled_for,
               rt.status, rt.quiz_id, rt.retention_score, rt.quality_label, rt.created_at,
               ss.duration as session_duration
        FROM recall_tasks rt
        LEFT JOIN study_sessions ss ON ss.id = rt.session_id
        WHERE rt.user_id = ? AND rt.status IN ('READY', 'IN_PROGRESS', 'PENDING')
        ORDER BY CASE rt.status WHEN 'READY' THEN 1 WHEN 'IN_PROGRESS' THEN 2 ELSE 3 END, rt.id DESC
    """, (user_id,))
    rows = cursor.fetchall()
    conn.close()

    active_tasks = []
    ready_count = 0
    pending_count = 0

    for r in rows:
        is_ready = r["status"] == "READY"
        if is_ready:
            ready_count += 1
        elif r["status"] == "PENDING":
            pending_count += 1

        active_tasks.append({
            "id": r["id"],
            "session_id": r["session_id"],
            "topic": r["topic"],
            "delay_minutes": r["delay_minutes"],
            "scheduled_for": r["scheduled_for"],
            "status": r["status"],
            "quiz_id": r["quiz_id"],
            "retention_score": r["retention_score"],
            "quality_label": r["quality_label"],
            "created_at": r["created_at"],
            "session_duration": r["session_duration"] or 0
        })

    primary_task = active_tasks[0] if active_tasks else None

    return {
        "has_tasks": len(active_tasks) > 0,
        "ready_count": ready_count,
        "pending_count": pending_count,
        "primary_task": primary_task,
        "tasks": active_tasks
    }


def start_recall_test(task_id: int, user_id: int, question_count: int = 3) -> dict:
    """
    Start or resume a recall test for a specific task. Generates 3-5 grounded questions.
    """
    conn = get_db()
    cursor = conn.cursor()

    cursor.execute("""
        SELECT id, session_id, topic, status, quiz_id
        FROM recall_tasks
        WHERE id = ? AND user_id = ?
    """, (task_id, user_id))
    task = cursor.fetchone()

    if not task:
        conn.close()
        return {"success": False, "error": "Recall task not found"}

    session_id = task["session_id"]
    topic = task["topic"] or "General Study"
    quiz_id = task["quiz_id"]

    # Check if a quiz was already generated for this task
    if quiz_id:
        cursor.execute("SELECT id, topic, total_questions, score_pct, quality_label, summary_text FROM study_quizzes WHERE id = ?", (quiz_id,))
        existing_quiz = cursor.fetchone()
        if existing_quiz:
            cursor.execute("""
                SELECT qq.id, qq.question_index, qq.question_text, qq.sub_concept, qq.points_possible,
                       qa.student_answer, qa.evaluation_status, qa.score_earned, qa.feedback_text
                FROM quiz_questions qq
                LEFT JOIN quiz_answers qa ON qa.question_id = qq.id AND qa.user_id = ?
                WHERE qq.quiz_id = ?
                ORDER BY qq.question_index ASC
            """, (user_id, quiz_id))
            q_rows = cursor.fetchall()
            conn.close()

            questions = [
                {
                    "id": r["id"],
                    "question_index": r["question_index"],
                    "question_text": r["question_text"],
                    "sub_concept": r["sub_concept"],
                    "points_possible": r["points_possible"],
                    "answered": r["student_answer"] is not None,
                    "student_answer": r["student_answer"] or "",
                    "evaluation_status": r["evaluation_status"],
                    "score_earned": r["score_earned"],
                    "feedback_text": r["feedback_text"],
                }
                for r in q_rows
            ]

            return {
                "success": True,
                "task_id": task_id,
                "quiz_id": quiz_id,
                "topic": existing_quiz["topic"],
                "total_questions": existing_quiz["total_questions"],
                "questions": questions,
                "is_completed": existing_quiz["score_pct"] > 0.0 or all(q["answered"] for q in questions),
                "score_pct": existing_quiz["score_pct"],
                "quality_label": existing_quiz["quality_label"],
            }

    # Fetch any prior weak concepts for this topic to target them in question generation
    cursor.execute("""
        SELECT sub_concept FROM concept_mastery
        WHERE user_id = ? AND topic = ? AND is_weak = 1
        ORDER BY mastery_pct ASC LIMIT 2
    """, (user_id, topic))
    weak_rows = cursor.fetchall()
    weak_subconcepts = [w["sub_concept"] for w in weak_rows]

    # Fetch recent AI conversation history for grounding
    cursor.execute("""
        SELECT sender, message FROM ai_messages
        WHERE conversation_id IN (
            SELECT id FROM ai_conversations WHERE user_id = ? ORDER BY id DESC LIMIT 1
        )
        ORDER BY id ASC LIMIT 8
    """, (user_id,))
    chat_rows = cursor.fetchall()
    conversation_history = [{"sender": r["sender"], "message": r["message"]} for r in chat_rows]

    # Adaptive difficulty configuration based on prior mastery history
    adaptive_cfg = get_adaptive_quiz_config(user_id, topic)

    # Check for matching material context
    material_excerpts = None
    try:
        cursor.execute("""
            SELECT id FROM study_materials
            WHERE user_id = ? AND (topic = ? OR title LIKE ?) AND status = 'READY'
            ORDER BY id DESC LIMIT 1
        """, (user_id, topic, f"%{topic}%"))
        mat_match = cursor.fetchone()
        if mat_match:
            cursor.execute("""
                SELECT chunk_text FROM material_chunks
                WHERE material_id = ? AND user_id = ?
                ORDER BY chunk_index ASC LIMIT 3
            """, (mat_match["id"], user_id))
            c_rows = cursor.fetchall()
            if c_rows:
                material_excerpts = "\n\n".join([r["chunk_text"] for r in c_rows])
    except Exception:
        material_excerpts = None

    # Generate 3 to 5 questions via AI service
    q_count = max(3, min(5, question_count))
    raw_questions = ai_service.generate_quiz(
        topic,
        conversation_history,
        question_count=q_count,
        adaptive_guidance=adaptive_cfg.get("guidance"),
        material_excerpts=material_excerpts
    )

    if not raw_questions:
        raw_questions = [
            {
                "question_index": 1,
                "question_text": f"What is the foundational definition and primary mechanism of '{topic}'?",
                "sub_concept": f"{topic} Core Principles",
                "difficulty": "basic",
                "sample_answer": f"The core purpose of {topic} is to organize operations and solve foundational challenges effectively.",
                "points_possible": 1.0,
            },
            {
                "question_index": 2,
                "question_text": f"Explain a critical trade-off or practical scenario encountered when applying '{topic}'.",
                "sub_concept": f"{topic} Application",
                "difficulty": "application",
                "sample_answer": f"Key trade-offs in {topic} involve balancing resource overhead, complexity, and practical execution.",
                "points_possible": 1.0,
            },
            {
                "question_index": 3,
                "question_text": f"How does '{topic}' behave in a real-world edge case, pitfall, or under complex workload?",
                "sub_concept": f"{topic} Edge Cases",
                "difficulty": "advanced",
                "sample_answer": f"Under complex conditions, {topic} requires careful boundary checks and deterministic error handling.",
                "points_possible": 1.0,
            }
        ][:q_count]

    # Insert into study_quizzes
    now_iso = datetime.now().isoformat()
    cursor.execute("""
        INSERT INTO study_quizzes (user_id, session_id, topic, total_questions, score_pct, created_at)
        VALUES (?, ?, ?, ?, 0.0, ?)
    """, (user_id, session_id, topic, len(raw_questions), now_iso))
    new_quiz_id = cursor.lastrowid

    # Insert questions
    client_questions = []
    for idx, q in enumerate(raw_questions, start=1):
        q_text = q.get("question_text", f"Explain {topic}")
        sub_c = q.get("sub_concept", topic)
        sample_ans = q.get("sample_answer", "Accurate conceptual explanation.")
        diff_str = q.get("difficulty", "conceptual")
        pts = float(q.get("points_possible", 1.0))

        cursor.execute("""
            INSERT INTO quiz_questions (quiz_id, question_index, question_text, sub_concept, sample_answer, difficulty, points_possible)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (new_quiz_id, idx, q_text, sub_c, sample_ans, diff_str, pts))
        q_id = cursor.lastrowid

        client_questions.append({
            "id": q_id,
            "question_index": idx,
            "question_text": q_text,
            "sub_concept": sub_c,
            "sample_answer": sample_ans,
            "difficulty": diff_str,
            "points_possible": pts,
            "answered": False,
            "student_answer": "",
            "evaluation_status": None,
            "score_earned": None,
            "feedback_text": None,
        })

    # Update recall_tasks with quiz_id and status 'IN_PROGRESS'
    cursor.execute("""
        UPDATE recall_tasks
        SET quiz_id = ?, status = 'IN_PROGRESS'
        WHERE id = ?
    """, (new_quiz_id, task_id))

    conn.commit()
    conn.close()

    return {
        "success": True,
        "task_id": task_id,
        "quiz_id": new_quiz_id,
        "topic": topic,
        "total_questions": len(client_questions),
        "questions": client_questions,
        "is_completed": False,
        "score_pct": 0.0,
    }


def submit_recall_answer(question_id: int, student_answer: str, user_id: int,
                         question_text: str = None, sub_concept: str = None,
                         sample_answer: str = None, points_possible: float = 1.0,
                         topic: str = None, task_id: int = None, quiz_id: int = None) -> dict:
    """
    Evaluate student's answer, assign points (1.0, 0.5, 0.0), and update concept mastery.
    Resilient to serverless multi-instance execution (Vercel/Lambda ephemeral storage).
    """
    conn = get_db()
    cursor = conn.cursor()

    cursor.execute("""
        SELECT qq.id, qq.quiz_id, qq.question_text, qq.sub_concept, qq.sample_answer, qq.points_possible,
               sq.topic, sq.session_id
        FROM quiz_questions qq
        JOIN study_quizzes sq ON sq.id = qq.quiz_id
        WHERE qq.id = ?
    """, (question_id,))
    q_row = cursor.fetchone()

    # If not found directly by ID (common in serverless ephemeral instances where DB is per-container),
    # try matching by question_text or recreate the question row in this container's DB!
    if not q_row and question_text:
        cursor.execute("""
            SELECT qq.id, qq.quiz_id, qq.question_text, qq.sub_concept, qq.sample_answer, qq.points_possible,
                   sq.topic, sq.session_id
            FROM quiz_questions qq
            JOIN study_quizzes sq ON sq.id = qq.quiz_id
            WHERE qq.question_text = ?
        """, (question_text,))
        q_row = cursor.fetchone()

    if not q_row:
        # Serverless Ephemeral Instance Resilience:
        # When deployed on Vercel/serverless platforms, Request 2 (answer submission)
        # may hit a different execution container whose local /tmp/users.db does not
        # have the question created by Request 1.
        # We gracefully reconstruct and persist the quiz, question, and task rows here.
        now_iso = datetime.now().isoformat()

        resolved_topic = topic
        resolved_session_id = None
        if task_id:
            try:
                cursor.execute("SELECT topic, session_id, quiz_id FROM recall_tasks WHERE id = ?", (task_id,))
                t_row = cursor.fetchone()
                if t_row:
                    resolved_topic = resolved_topic or t_row["topic"]
                    resolved_session_id = t_row["session_id"]
                    quiz_id = quiz_id or t_row["quiz_id"]
            except Exception:
                pass

        resolved_topic = (resolved_topic or "General Academic Study").strip()
        sub_concept = (sub_concept or "Core Concept").strip()
        question_text = (question_text or f"Explain the core mechanisms and significance of {sub_concept}.").strip()
        points_possible = float(points_possible or 1.0)

        # Resolve sample answer if missing
        if not sample_answer:
            sample_answer = f"Accurate conceptual explanation of {sub_concept} within {resolved_topic}."
            sub_c_lower = sub_concept.lower()
            if "fcfs" in sub_c_lower:
                sample_answer = "FCFS executes processes in arrival order non-preemptively. The convoy effect occurs when short CPU processes wait behind a long CPU-bound process, causing high average waiting time."
            elif "round robin" in sub_c_lower:
                sample_answer = "Round Robin assigns each process a fixed time quantum. If too small, context switching overhead degrades throughput."
            elif "primitive" in sub_c_lower:
                sample_answer = "Primitive types store raw values on the stack, while reference types store addresses pointing to heap objects."

        # Ensure a study_quizzes row exists
        if quiz_id:
            cursor.execute("SELECT id FROM study_quizzes WHERE id = ?", (quiz_id,))
            if not cursor.fetchone():
                cursor.execute("""
                    INSERT INTO study_quizzes (id, user_id, session_id, topic, total_questions, score_pct, created_at)
                    VALUES (?, ?, ?, ?, 3, 0.0, ?)
                """, (quiz_id, user_id, resolved_session_id, resolved_topic, now_iso))
        else:
            cursor.execute("""
                INSERT INTO study_quizzes (user_id, session_id, topic, total_questions, score_pct, created_at)
                VALUES (?, ?, ?, 3, 0.0, ?)
            """, (user_id, resolved_session_id, resolved_topic, now_iso))
            quiz_id = cursor.lastrowid

        # Insert or replace question into quiz_questions
        cursor.execute("""
            INSERT OR REPLACE INTO quiz_questions (id, quiz_id, question_index, question_text, sub_concept, sample_answer, difficulty, points_possible)
            VALUES (?, ?, 1, ?, ?, ?, 'conceptual', ?)
        """, (question_id, quiz_id, question_text, sub_concept, sample_answer, points_possible))

        # Ensure recall_tasks row exists/is updated
        if task_id:
            cursor.execute("SELECT id FROM recall_tasks WHERE id = ?", (task_id,))
            if cursor.fetchone():
                cursor.execute("UPDATE recall_tasks SET quiz_id = ?, status = 'IN_PROGRESS' WHERE id = ?", (quiz_id, task_id))
            else:
                cursor.execute("""
                    INSERT INTO recall_tasks (id, user_id, session_id, topic, quiz_id, status, scheduled_for, created_at)
                    VALUES (?, ?, ?, ?, ?, 'IN_PROGRESS', ?, ?)
                """, (task_id, user_id, resolved_session_id, resolved_topic, quiz_id, now_iso, now_iso))

        conn.commit()

        # Re-fetch q_row
        cursor.execute("""
            SELECT qq.id, qq.quiz_id, qq.question_text, qq.sub_concept, qq.sample_answer, qq.points_possible,
                   sq.topic, sq.session_id
            FROM quiz_questions qq
            JOIN study_quizzes sq ON sq.id = qq.quiz_id
            WHERE qq.id = ?
        """, (question_id,))
        q_row = cursor.fetchone()

    if not q_row:
        conn.close()
        return {"success": False, "error": "Question could not be resolved"}

    quiz_id = q_row["quiz_id"]
    topic = q_row["topic"]
    sub_concept = q_row["sub_concept"]
    sample_answer = q_row["sample_answer"]
    question_text = q_row["question_text"]
    points_possible = float(q_row["points_possible"] or 1.0)

    # Perform evaluation
    eval_result = ai_service.evaluate_answer(
        question_text=question_text,
        sub_concept=sub_concept,
        sample_answer=sample_answer,
        student_answer=student_answer
    )

    status = eval_result.get("status", "incorrect")
    score_factor = float(eval_result.get("score", 0.0))
    score_earned = round(score_factor * points_possible, 2)
    feedback_text = eval_result.get("feedback", "Answer evaluated.")
    now_iso = datetime.now().isoformat()

    # Record / update in quiz_answers
    cursor.execute("SELECT id FROM quiz_answers WHERE question_id = ? AND user_id = ?", (question_id, user_id))
    existing_ans = cursor.fetchone()
    if existing_ans:
        cursor.execute("""
            UPDATE quiz_answers
            SET student_answer = ?, evaluation_status = ?, score_earned = ?, feedback_text = ?, answered_at = ?
            WHERE id = ?
        """, (student_answer, status, score_earned, feedback_text, now_iso, existing_ans["id"]))
    else:
        cursor.execute("""
            INSERT INTO quiz_answers (question_id, quiz_id, user_id, student_answer, evaluation_status, score_earned, feedback_text, answered_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (question_id, quiz_id, user_id, student_answer, status, score_earned, feedback_text, now_iso))

    # Update running concept mastery
    cursor.execute("""
        SELECT id, total_tested, correct_count, partial_count, incorrect_count
        FROM concept_mastery
        WHERE user_id = ? AND topic = ? AND sub_concept = ?
    """, (user_id, topic, sub_concept))
    cm_row = cursor.fetchone()

    is_correct = 1 if status == "correct" else 0
    is_partial = 1 if status == "partial" else 0
    is_incorrect = 1 if status in ["incorrect", "unanswered"] else 0

    if cm_row:
        tot = cm_row["total_tested"] + 1
        corr = cm_row["correct_count"] + is_correct
        part = cm_row["partial_count"] + is_partial
        inc = cm_row["incorrect_count"] + is_incorrect
        mastery = round(((corr * 1.0 + part * 0.5) / max(1, tot)) * 100.0, 1)
        weak_flag = 1 if mastery < 70.0 else 0

        cursor.execute("""
            UPDATE concept_mastery
            SET total_tested = ?, correct_count = ?, partial_count = ?, incorrect_count = ?,
                mastery_pct = ?, is_weak = ?, last_tested_at = ?
            WHERE id = ?
        """, (tot, corr, part, inc, mastery, weak_flag, now_iso, cm_row["id"]))
    else:
        tot = 1
        corr = is_correct
        part = is_partial
        inc = is_incorrect
        mastery = round(((corr * 1.0 + part * 0.5) / 1.0) * 100.0, 1)
        weak_flag = 1 if mastery < 70.0 else 0

        cursor.execute("""
            INSERT INTO concept_mastery (user_id, topic, sub_concept, total_tested, correct_count, partial_count, incorrect_count, mastery_pct, is_weak, last_tested_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (user_id, topic, sub_concept, tot, corr, part, inc, mastery, weak_flag, now_iso))

    conn.commit()
    conn.close()

    eval_label = "Correct" if status == "correct" else ("Partially Correct" if status == "partial" else "Incorrect")

    return {
        "success": True,
        "question_id": question_id,
        "status": status,
        "evaluation": eval_label,
        "score_earned": score_earned,
        "points_possible": points_possible,
        "feedback": feedback_text,
        "ai_feedback": feedback_text,
        "sub_concept": sub_concept,
    }


def finalize_recall_test(task_id: int, user_id: int, session_answers: dict = None, topic: str = None) -> dict:
    """
    Finalize recall test, calculate retention score %, update recall_tasks & study_quizzes,
    and generate explainable retention and revision recommendations.
    Resilient to serverless multi-instance execution (Vercel/Lambda ephemeral storage).
    """
    conn = get_db()
    cursor = conn.cursor()

    cursor.execute("""
        SELECT rt.id, rt.session_id, rt.topic, rt.quiz_id, ss.duration
        FROM recall_tasks rt
        LEFT JOIN study_sessions ss ON ss.id = rt.session_id
        WHERE rt.id = ? AND rt.user_id = ?
    """, (task_id, user_id))
    task_row = cursor.fetchone()

    quiz_id = task_row["quiz_id"] if task_row and task_row["quiz_id"] else None
    resolved_topic = task_row["topic"] if task_row and task_row["topic"] else (topic or "General Academic Study")
    session_id = task_row["session_id"] if task_row else None
    session_duration = (task_row["duration"] if task_row else 0) or 0

    # Retrieve all questions and answers from database
    rows = []
    if quiz_id:
        cursor.execute("""
            SELECT qq.id, qq.sub_concept, qq.points_possible,
                   qa.evaluation_status, qa.score_earned, qa.feedback_text
            FROM quiz_questions qq
            LEFT JOIN quiz_answers qa ON qa.question_id = qq.id AND qa.user_id = ?
            WHERE qq.quiz_id = ?
            ORDER BY qq.question_index ASC
        """, (user_id, quiz_id))
        rows = cursor.fetchall()

    # Fallback to session_answers if DB rows were empty (cross-instance serverless jump)
    if not rows and session_answers:
        rows = [
            {
                "id": a.get("question_id"),
                "sub_concept": a.get("sub_concept") or "Core Concept",
                "points_possible": float(a.get("points_possible", 1.0)),
                "evaluation_status": a.get("status") or "unanswered",
                "score_earned": float(a.get("score_earned", 0.0)),
                "feedback_text": a.get("feedback") or "",
            }
            for a in session_answers.values()
        ]

    if not task_row and not rows:
        conn.close()
        return {"success": False, "error": "Recall task or answers not found"}

    topic = resolved_topic
    total_possible = sum(float(r["points_possible"] or 1.0) for r in rows) if rows else 3.0
    total_earned = sum(float(r["score_earned"] or 0.0) for r in rows) if rows else 0.0
    score_pct = int(round((total_earned / max(1.0, total_possible)) * 100.0))

    if score_pct >= 90:
        quality_label = "Deep Mastery"
    elif score_pct >= 75:
        quality_label = "Solid Comprehension"
    elif score_pct >= 50:
        quality_label = "Partial Retention"
    else:
        quality_label = "Needs Review"

    strong_concepts = []
    weak_concepts = []
    for r in rows:
        sc = r["sub_concept"]
        status = r["evaluation_status"] or "unanswered"
        if status == "correct":
            strong_concepts.append(sc)
        else:
            weak_concepts.append(sc)

    strong_concepts = list(dict.fromkeys(strong_concepts))
    weak_concepts = list(dict.fromkeys(weak_concepts))

    hrs = session_duration // 3600
    mins = (session_duration % 3600) // 60
    time_str = f"{hrs}h {mins}m" if hrs > 0 else f"{max(1, mins)}m"

    if score_pct >= 85:
        summary_text = (
            f"High retention verified ({score_pct}%). You demonstrated strong recall on {topic} after your study session."
        )
    elif weak_concepts:
        weak_str = ", ".join(weak_concepts[:2])
        summary_text = (
            f"Recall score of {score_pct}%. While you studied for {time_str}, your answers indicate that {weak_str} needs spaced review."
        )
    else:
        summary_text = (
            f"Recall score of {score_pct}%. Review key definitions and mechanism flows to boost retention."
        )

    now_iso = datetime.now().isoformat()

    # Compute structured retention data (learned well vs needs revision)
    retention_data = calculate_retention_score(rows)
    learned_well = retention_data.get("learned_well") or strong_concepts
    needs_revision = retention_data.get("needs_revision") or weak_concepts

    # Retrieve physical telemetry for this study session if present
    presence_val = 100.0
    behavior_val = 80.0
    duration_val = 1800
    absence_cnt = 0
    longest_abs = 0
    completion_val = 100.0

    if session_id:
        try:
            cursor.execute("SELECT * FROM session_focus_scores WHERE session_id = ?", (session_id,))
            sfs_row = cursor.fetchone()
            if sfs_row:
                presence_val = float(sfs_row["presence_pct"] or 100.0)
                behavior_val = float(sfs_row["estimated_score"] or 80.0)
                duration_val = int(sfs_row["session_duration_sec"] or 1800)
                absence_cnt = int(sfs_row["absence_count"] or 0)
                longest_abs = int(sfs_row["longest_absence_sec"] or 0)
                completion_val = float(sfs_row["goal_progress_pct"] or 100.0)
        except Exception:
            pass

    # Compute 8-dimension Unified FocusScore
    unified_metrics = {
        "presence_pct": presence_val,
        "retention_pct": score_pct,
        "ai_engagement_pct": behavior_val,
        "quiz_score_pct": score_pct,
        "duration_sec": duration_val,
        "completion_pct": completion_val,
        "consistency_pct": 80.0,
        "absence_count": absence_cnt,
        "longest_absence_sec": longest_abs,
    }
    unified_analysis = compute_unified_focus_score(unified_metrics)
    final_focus_score = unified_analysis.get("unified_focus_score", int(round(score_pct)))

    # Update study_quizzes
    cursor.execute("""
        UPDATE study_quizzes
        SET score_pct = ?, quality_label = ?, summary_text = ?
        WHERE id = ?
    """, (score_pct, quality_label, summary_text, quiz_id))

    # Update recall_tasks
    cursor.execute("""
        UPDATE recall_tasks
        SET status = 'COMPLETED', retention_score = ?, quality_label = ?, completed_at = ?
        WHERE id = ?
    """, (score_pct, quality_label, now_iso, task_id))

    # Update study_sessions with retention_score and synthesized focus_score
    if session_id:
        try:
            cursor.execute("""
                UPDATE study_sessions
                SET retention_score = ?, focus_score = ?
                WHERE id = ?
            """, (score_pct, final_focus_score, session_id))
        except Exception:
            pass

    conn.commit()
    conn.close()

    # Update topic-level mastery tiers in SQLite (topic_mastery table)
    mastery_info = update_topic_mastery(
        user_id=user_id,
        topic=topic,
        retention_score=score_pct,
        answers=rows
    )

    # 1. State Snapshot
    prior_state = get_latest_learning_state(user_id, topic)
    state_metrics = {
        "physical_presence_pct": presence_val,
        "study_duration_sec": duration_val,
        "ai_engagement_pct": behavior_val,
        "quiz_score_pct": score_pct,
        "retention_score_pct": score_pct,
        "distraction_control_pct": unified_analysis["factors"]["distraction_control_pct"],
        "session_completion_pct": completion_val,
        "consistency_score_pct": 80.0,
        "unified_focus_score": final_focus_score,
        "learning_efficiency_score": unified_analysis["learning_efficiency"],
        "quality_tier": unified_analysis["quality_tier"],
        "score_label": unified_analysis["score_label"],
        "explanation": unified_analysis["explanation"],
        "mastery_tier": mastery_info.get("mastery_tier", "MEDIUM"),
        "is_passive_alert": unified_analysis["is_passive_alert"],
    }
    state_id = record_learning_state(user_id, session_id or 0, topic, state_metrics)

    # 2. Adaptive Action Selection
    selected_act = select_adaptive_action(state_metrics)
    action_id = record_adaptive_action(
        user_id=user_id,
        topic=topic,
        session_id=session_id or 0,
        state_id=state_id,
        action_type=selected_act["action_type"],
        action_payload=selected_act["action_payload"],
        reason=selected_act["reason"]
    )

    # 3. Reward Calculation if prior state existed
    if prior_state and prior_state.get("id") != state_id:
        r_val = calculate_adaptive_reward(prior_state, state_metrics)
        # Fetch pending action if any
        conn_r = get_db()
        c_r = conn_r.cursor()
        c_r.execute("""
            SELECT id FROM adaptive_actions
            WHERE user_id = ? AND topic = ? AND status = 'PENDING' AND id != ?
            ORDER BY id DESC LIMIT 1
        """, (user_id, topic, action_id))
        pending_act = c_r.fetchone()
        conn_r.close()
        prev_act_id = pending_act["id"] if pending_act else action_id
        record_adaptive_reward(
            user_id=user_id,
            action_id=prev_act_id,
            initial_state_id=prior_state["id"],
            resulting_state_id=state_id,
            reward_value=r_val,
            evaluation_notes=f"Recall test outcome {score_pct}% evaluated."
        )

    # Log to adaptive_learning_logs for backward compatibility
    log_adaptive_learning_event(
        user_id=user_id,
        session_id=session_id or 0,
        topic=topic,
        prior_mastery_tier=mastery_info.get("mastery_tier", "MEDIUM"),
        action_intervention=selected_act["action_type"],
        retention_outcome_pct=score_pct
    )

    concepts_breakdown = []
    for r in rows:
        sc = r["sub_concept"]
        eval_st = r["evaluation_status"] or "unanswered"
        pts_poss = float(r["points_possible"] or 1.0)
        pts_earned = float(r["score_earned"] or 0.0)
        c_pct = int(round((pts_earned / max(0.1, pts_poss)) * 100))
        concepts_breakdown.append({
            "concept": sc,
            "score": c_pct,
            "evaluation": "Correct" if eval_st == "correct" else ("Partially Correct" if eval_st == "partial" else "Incorrect")
        })

    return {
        "success": True,
        "task_id": task_id,
        "quiz_id": quiz_id,
        "topic": topic,
        "retention_score": score_pct,
        "score_pct": score_pct,
        "quality_label": quality_label,
        "mastery_tier": mastery_info.get("mastery_tier", "MEDIUM"),
        "avg_retention_score": mastery_info.get("avg_retention_score", score_pct),
        "physical_focus": round(presence_val, 1),
        "learning_focus": score_pct,
        "final_focus_score": final_focus_score,
        "explanation": unified_analysis.get("explanation", "Your physical focus was strong, but some concepts need revision."),
        "summary": summary_text,
        "summary_text": summary_text,
        "learned_well": learned_well,
        "needs_revision": needs_revision,
        "strong_concepts": strong_concepts,
        "weak_concepts": weak_concepts,
        "concepts_breakdown": concepts_breakdown,
        "correct_count": sum(1 for r in rows if r["evaluation_status"] == "correct"),
        "partial_count": sum(1 for r in rows if r["evaluation_status"] == "partial"),
        "incorrect_count": sum(1 for r in rows if r["evaluation_status"] in ["incorrect", "unanswered", None]),
        "total_questions": len(rows),
    }


def get_weak_topics_summary(user_id: int) -> list:
    """
    Retrieve all concepts currently marked as weak (<70% mastery) for revision scheduling.
    """
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT topic, sub_concept, total_tested, correct_count, partial_count,
               incorrect_count, mastery_pct, last_tested_at
        FROM concept_mastery
        WHERE user_id = ? AND (is_weak = 1 OR mastery_pct < 70.0)
        ORDER BY mastery_pct ASC, last_tested_at DESC
        LIMIT 10
    """, (user_id,))
    rows = cursor.fetchall()
    conn.close()

    return [
        {
            "topic": r["topic"],
            "sub_concept": r["sub_concept"],
            "total_tested": r["total_tested"],
            "mastery_pct": r["mastery_pct"],
            "last_tested_at": r["last_tested_at"]
        }
        for r in rows
    ]


def get_recall_history(user_id: int) -> list:
    """
    Retrieve completed delayed recall tests with retention metrics.
    """
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT rt.id, rt.session_id, rt.topic, rt.delay_minutes, rt.retention_score,
               rt.quality_label, rt.completed_at, ss.date as session_date, ss.duration as session_duration
        FROM recall_tasks rt
        LEFT JOIN study_sessions ss ON ss.id = rt.session_id
        WHERE rt.user_id = ? AND rt.status = 'COMPLETED'
        ORDER BY rt.id DESC
        LIMIT 20
    """, (user_id,))
    rows = cursor.fetchall()
    conn.close()

    return [
        {
            "id": r["id"],
            "session_id": r["session_id"],
            "topic": r["topic"],
            "delay_minutes": r["delay_minutes"],
            "retention_score": r["retention_score"],
            "quality_label": r["quality_label"],
            "completed_at": r["completed_at"],
            "session_date": r["session_date"],
            "session_duration": r["session_duration"]
        }
        for r in rows
    ]
