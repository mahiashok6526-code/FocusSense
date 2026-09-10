"""
FocusSense AI — Post-Study Examination & Learning Verification Engine (Phase 2)
================================================================================
Evaluates actual student comprehension upon session completion.
- Generates grounded, topic-specific examinations from active session context.
- Semantically grades free-form student answers with partial-credit rubrics.
- Tracks granular sub-concept mastery and detects weak concepts (<70%).
- Synthesizes explainable "Time Spent vs. Demonstrated Learning" insights.
"""

import sqlite3
from datetime import datetime
from ai_service import ai_service

from database import DATABASE, DB_PATH


def get_db_connection():
    conn = sqlite3.connect(DATABASE, timeout=30.0)
    conn.row_factory = sqlite3.Row
    return conn


def create_or_get_session_quiz(user_id: int, session_id: int = None, topic: str = "General Study", question_count: int = 3) -> dict:
    """
    Generate a new grounded quiz for a completed study session or return existing active quiz.
    Ensures answers are not revealed in the returned question objects.
    """
    conn = get_db_connection()
    cursor = conn.cursor()

    # Check if a quiz was already created for this specific session
    if session_id:
        cursor.execute("""
            SELECT id, topic, total_questions, score_pct, quality_label, summary_text
            FROM study_quizzes
            WHERE user_id = ? AND session_id = ?
            ORDER BY id DESC LIMIT 1
        """, (user_id, session_id))
        existing_quiz = cursor.fetchone()
        if existing_quiz:
            quiz_id = existing_quiz["id"]
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

            questions = []
            for r in q_rows:
                questions.append({
                    "id": r["id"],
                    "question_index": r["question_index"],
                    "question_text": r["question_text"],
                    "sub_concept": r["sub_concept"],
                    "points_possible": r["points_possible"],
                    "answered": r["student_answer"] is not None,
                    "evaluation_status": r["evaluation_status"],
                    "score_earned": r["score_earned"],
                    "feedback_text": r["feedback_text"],
                })

            return {
                "quiz_id": quiz_id,
                "topic": existing_quiz["topic"],
                "total_questions": existing_quiz["total_questions"],
                "questions": questions,
                "is_completed": existing_quiz["score_pct"] > 0.0 or all(q["answered"] for q in questions),
                "score_pct": existing_quiz["score_pct"],
                "quality_label": existing_quiz["quality_label"],
                "summary_text": existing_quiz["summary_text"],
            }

    # Fetch recent chat conversation for context grounding
    cursor.execute("""
        SELECT sender, message FROM ai_messages
        WHERE conversation_id IN (
            SELECT id FROM ai_conversations WHERE user_id = ? ORDER BY id DESC LIMIT 1
        )
        ORDER BY id ASC LIMIT 10
    """, (user_id,))
    chat_rows = cursor.fetchall()
    conversation_history = [{"sender": r["sender"], "message": r["message"]} for r in chat_rows]

    # Generate grounded questions via AI Service
    raw_questions = ai_service.generate_quiz(topic, conversation_history, question_count=question_count)
    if not raw_questions:
        raw_questions = [
            {
                "question_index": 1,
                "question_text": f"What is the foundational definition and core mechanism of '{topic}'?",
                "sub_concept": f"{topic} Core Principles",
                "sample_answer": f"The core purpose of {topic} is to organize operations and solve foundational challenges effectively.",
                "points_possible": 1.0,
            }
        ]

    # Insert quiz record
    now_iso = datetime.now().isoformat()
    cursor.execute("""
        INSERT INTO study_quizzes (user_id, session_id, topic, total_questions, score_pct, created_at)
        VALUES (?, ?, ?, ?, 0.0, ?)
    """, (user_id, session_id, topic, len(raw_questions), now_iso))
    quiz_id = cursor.lastrowid

    # Insert individual questions
    client_questions = []
    for idx, q in enumerate(raw_questions, start=1):
        q_text = q.get("question_text", f"Explain {topic}")
        sub_c = q.get("sub_concept", topic)
        sample_ans = q.get("sample_answer", "Accurate conceptual explanation.")
        pts = float(q.get("points_possible", 1.0))

        cursor.execute("""
            INSERT INTO quiz_questions (quiz_id, question_index, question_text, sub_concept, sample_answer, points_possible)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (quiz_id, idx, q_text, sub_c, sample_ans, pts))
        q_id = cursor.lastrowid

        client_questions.append({
            "id": q_id,
            "question_index": idx,
            "question_text": q_text,
            "sub_concept": sub_c,
            "points_possible": pts,
            "answered": False,
        })

    conn.commit()
    conn.close()

    return {
        "quiz_id": quiz_id,
        "topic": topic,
        "total_questions": len(client_questions),
        "questions": client_questions,
        "is_completed": False,
        "score_pct": 0.0,
    }


def evaluate_student_answer(question_id: int, student_answer: str, user_id: int,
                            question_text: str = None, sub_concept: str = None,
                            sample_answer: str = None, points_possible: float = 1.0,
                            topic: str = None, quiz_id: int = None) -> dict:
    """
    Semantically grade a student's answer, record the attempt, and update running concept mastery.
    Resilient to serverless multi-instance execution (ephemeral per-container SQLite).
    """
    conn = get_db_connection()
    cursor = conn.cursor()

    # Retrieve question details
    cursor.execute("""
        SELECT qq.id, qq.quiz_id, qq.question_text, qq.sub_concept, qq.sample_answer, qq.points_possible,
               sq.topic, sq.session_id
        FROM quiz_questions qq
        JOIN study_quizzes sq ON sq.id = qq.quiz_id
        WHERE qq.id = ?
    """, (question_id,))
    q_row = cursor.fetchone()

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
        # Reconstruct quiz and question in this container's DB
        now_iso = datetime.now().isoformat()
        resolved_topic = (topic or "General Academic Studies").strip()
        sub_concept = (sub_concept or "Core Concept").strip()
        question_text = (question_text or f"Explain {sub_concept} in {resolved_topic}.").strip()
        sample_answer = sample_answer or f"Accurate conceptual explanation of {sub_concept}."
        points_possible = float(points_possible or 1.0)

        if quiz_id:
            cursor.execute("SELECT id FROM study_quizzes WHERE id = ?", (quiz_id,))
            if not cursor.fetchone():
                cursor.execute("""
                    INSERT INTO study_quizzes (id, user_id, topic, total_questions, score_pct, created_at)
                    VALUES (?, ?, ?, 3, 0.0, ?)
                """, (quiz_id, user_id, resolved_topic, now_iso))
        else:
            cursor.execute("""
                INSERT INTO study_quizzes (user_id, topic, total_questions, score_pct, created_at)
                VALUES (?, ?, 3, 0.0, ?)
            """, (user_id, resolved_topic, now_iso))
            quiz_id = cursor.lastrowid

        cursor.execute("""
            INSERT OR REPLACE INTO quiz_questions (id, quiz_id, question_index, question_text, sub_concept, sample_answer, difficulty, points_possible)
            VALUES (?, ?, 1, ?, ?, ?, 'conceptual', ?)
        """, (question_id, quiz_id, question_text, sub_concept, sample_answer, points_possible))
        conn.commit()

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
        return {"success": False, "error": "Question not found"}

    quiz_id = q_row["quiz_id"]
    topic = q_row["topic"]
    sub_concept = q_row["sub_concept"]
    sample_answer = q_row["sample_answer"]
    question_text = q_row["question_text"]
    points_possible = float(q_row["points_possible"] or 1.0)

    # Perform semantic grading via AI Service
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

    # Record or update in quiz_answers
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

    return {
        "success": True,
        "question_id": question_id,
        "status": status,
        "score_earned": score_earned,
        "points_possible": points_possible,
        "feedback": feedback_text,
        "sub_concept": sub_concept,
    }


def complete_quiz(quiz_id: int, user_id: int, session_duration_sec: int = None) -> dict:
    """
    Finalize quiz, compute total score %, categorize strong vs weak concepts,
    and generate explainable Time Spent vs Demonstrated Learning summary.
    """
    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute("SELECT id, session_id, topic, total_questions FROM study_quizzes WHERE id = ? AND user_id = ?", (quiz_id, user_id))
    quiz_row = cursor.fetchone()
    if not quiz_row:
        conn.close()
        return {"success": False, "error": "Quiz not found"}

    session_id = quiz_row["session_id"]
    topic = quiz_row["topic"]

    # Gather all question & answer pairs
    cursor.execute("""
        SELECT qq.id, qq.sub_concept, qq.points_possible,
               qa.evaluation_status, qa.score_earned, qa.feedback_text
        FROM quiz_questions qq
        LEFT JOIN quiz_answers qa ON qa.question_id = qq.id AND qa.user_id = ?
        WHERE qq.quiz_id = ?
        ORDER BY qq.question_index ASC
    """, (user_id, quiz_id))
    rows = cursor.fetchall()

    total_possible = sum(float(r["points_possible"] or 1.0) for r in rows)
    total_earned = sum(float(r["score_earned"] or 0.0) for r in rows)
    score_pct = int(round((total_earned / max(1.0, total_possible)) * 100.0))

    if score_pct >= 90:
        quality_label = "Deep Mastery"
    elif score_pct >= 75:
        quality_label = "Solid Comprehension"
    elif score_pct >= 50:
        quality_label = "Partial Retention"
    else:
        quality_label = "Needs Review"

    # Identify strong concepts (>= 75%) and weak concepts (< 70%)
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

    # Fetch session duration if not provided
    if session_duration_sec is None and session_id:
        cursor.execute("SELECT duration FROM study_sessions WHERE id = ?", (session_id,))
        s_row = cursor.fetchone()
        if s_row:
            session_duration_sec = s_row["duration"]
    duration_sec = session_duration_sec or 0

    hrs = duration_sec // 3600
    mins = (duration_sec % 3600) // 60
    time_str = f"{hrs}h {mins}m" if hrs > 0 else f"{max(1, mins)}m"

    # Synthesize "Time Spent vs Demonstrated Learning" insight
    if score_pct >= 80:
        summary_text = (
            f"You spent {time_str} studying {topic}, and your learning verification of {score_pct}% "
            f"demonstrates high conceptual mastery."
        )
    elif weak_concepts:
        weak_list_str = ", ".join(weak_concepts[:2])
        summary_text = (
            f"You spent {time_str} studying, but your answers show that {weak_list_str} "
            f"needs further review before your next study session."
        )
    else:
        summary_text = (
            f"You spent {time_str} studying. Review key definitions to strengthen overall retention."
        )

    # Save to study_quizzes
    cursor.execute("""
        UPDATE study_quizzes
        SET score_pct = ?, quality_label = ?, summary_text = ?
        WHERE id = ?
    """, (score_pct, quality_label, summary_text, quiz_id))

    conn.commit()
    conn.close()

    return {
        "success": True,
        "quiz_id": quiz_id,
        "topic": topic,
        "score_pct": score_pct,
        "quality_label": quality_label,
        "study_time_str": time_str,
        "summary_text": summary_text,
        "strong_concepts": strong_concepts,
        "weak_concepts": weak_concepts,
        "total_questions": len(rows),
        "correct_count": sum(1 for r in rows if r["evaluation_status"] == "correct"),
        "partial_count": sum(1 for r in rows if r["evaluation_status"] == "partial"),
        "incorrect_count": sum(1 for r in rows if r["evaluation_status"] in ["incorrect", "unanswered", None]),
    }


def get_latest_quiz_overview(user_id: int) -> dict:
    """Retrieve the latest completed examination overview for dashboard rendering."""
    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute("""
        SELECT sq.id, sq.topic, sq.score_pct, sq.quality_label, sq.summary_text, sq.created_at,
               ss.duration
        FROM study_quizzes sq
        LEFT JOIN study_sessions ss ON ss.id = sq.session_id
        WHERE sq.user_id = ? AND sq.score_pct > 0.0
        ORDER BY sq.id DESC LIMIT 1
    """, (user_id,))
    row = cursor.fetchone()

    if not row:
        conn.close()
        return {
            "has_quiz": False,
            "score_pct": None,
            "quality_label": "No Exam Yet",
            "strong_concepts": [],
            "weak_concepts": [],
        }

    quiz_id = row["id"]
    cursor.execute("""
        SELECT qq.sub_concept, qa.evaluation_status
        FROM quiz_questions qq
        LEFT JOIN quiz_answers qa ON qa.question_id = qq.id AND qa.user_id = ?
        WHERE qq.quiz_id = ?
    """, (user_id, quiz_id))
    q_rows = cursor.fetchall()

    cursor.execute("""
        SELECT sub_concept, mastery_pct, is_weak
        FROM concept_mastery
        WHERE user_id = ? AND topic = ?
        ORDER BY last_tested_at DESC LIMIT 6
    """, (user_id, row["topic"]))
    mastery_rows = cursor.fetchall()
    conn.close()

    strong = [r["sub_concept"] for r in mastery_rows if r["is_weak"] == 0]
    weak = [r["sub_concept"] for r in mastery_rows if r["is_weak"] == 1]

    return {
        "has_quiz": True,
        "quiz_id": quiz_id,
        "topic": row["topic"],
        "score_pct": int(round(row["score_pct"])),
        "quality_label": row["quality_label"] or "Completed",
        "summary_text": row["summary_text"],
        "strong_concepts": strong or [r["sub_concept"] for r in q_rows if r["evaluation_status"] == "correct"],
        "weak_concepts": weak or [r["sub_concept"] for r in q_rows if r["evaluation_status"] != "correct"],
    }


def create_or_get_material_quiz(user_id: int, material_id: int, question_count: int = 4) -> dict:
    """
    Generate a grounded quiz directly from an uploaded study material.
    Extracts high-yield chunks from the material and prompts the AI provider.
    Saves in standard study_quizzes and quiz_questions tables with material_id.
    """
    conn = get_db_connection()
    cursor = conn.cursor()

    # Validate material ownership
    cursor.execute("""
        SELECT id, title, subject, topic, chunk_count
        FROM study_materials
        WHERE id = ? AND user_id = ?
    """, (material_id, user_id))
    mat = cursor.fetchone()

    if not mat:
        conn.close()
        return {"success": False, "error": "Study material not found or access denied."}

    topic = mat["topic"] or mat["title"]

    # Check for existing incomplete quiz for this material
    cursor.execute("""
        SELECT id, topic, total_questions, score_pct, quality_label, summary_text
        FROM study_quizzes
        WHERE user_id = ? AND material_id = ? AND (score_pct = 0.0 OR score_pct IS NULL)
        ORDER BY id DESC LIMIT 1
    """, (user_id, material_id))
    existing_quiz = cursor.fetchone()

    if existing_quiz:
        quiz_id = existing_quiz["id"]
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
                "evaluation_status": r["evaluation_status"],
                "score_earned": r["score_earned"],
                "feedback_text": r["feedback_text"],
            }
            for r in q_rows
        ]

        return {
            "success": True,
            "quiz_id": quiz_id,
            "material_id": material_id,
            "topic": existing_quiz["topic"],
            "total_questions": existing_quiz["total_questions"],
            "questions": questions,
            "is_completed": False,
            "score_pct": 0.0,
        }

    # Fetch top/representative chunks from the material
    cursor.execute("""
        SELECT chunk_index, chunk_text
        FROM material_chunks
        WHERE material_id = ? AND user_id = ?
        ORDER BY chunk_index ASC LIMIT 5
    """, (material_id, user_id))
    chunk_rows = cursor.fetchall()

    excerpts_list = [f"=== Section #{r['chunk_index']} ===\n{r['chunk_text']}" for r in chunk_rows]
    material_excerpts = "\n\n".join(excerpts_list)

    # Generate questions via AI service
    raw_questions = ai_service.generate_quiz(
        topic=topic,
        conversation_history=[],
        question_count=question_count,
        material_excerpts=material_excerpts
    )

    if not raw_questions:
        raw_questions = [
            {
                "question_index": 1,
                "question_text": f"Based on '{mat['title']}', what is the core mechanism and primary objective discussed?",
                "sub_concept": f"{topic} Core Principles",
                "sample_answer": f"The material describes fundamental rules, systematic coordination, and practical workflows for {topic}.",
                "points_possible": 1.0,
            }
        ]

    # Save quiz record
    now_iso = datetime.now().isoformat()
    cursor.execute("""
        INSERT INTO study_quizzes (user_id, material_id, topic, total_questions, score_pct, created_at)
        VALUES (?, ?, ?, ?, 0.0, ?)
    """, (user_id, material_id, topic, len(raw_questions), now_iso))
    quiz_id = cursor.lastrowid

    client_questions = []
    for idx, q in enumerate(raw_questions, start=1):
        q_text = q.get("question_text", f"Explain {topic}")
        sub_c = q.get("sub_concept", topic)
        sample_ans = q.get("sample_answer", "Accurate conceptual explanation.")
        pts = float(q.get("points_possible", 1.0))

        cursor.execute("""
            INSERT INTO quiz_questions (quiz_id, question_index, question_text, sub_concept, sample_answer, points_possible)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (quiz_id, idx, q_text, sub_c, sample_ans, pts))
        q_id = cursor.lastrowid

        client_questions.append({
            "id": q_id,
            "question_index": idx,
            "question_text": q_text,
            "sub_concept": sub_c,
            "points_possible": pts,
            "answered": False,
        })

    conn.commit()
    conn.close()

    return {
        "success": True,
        "quiz_id": quiz_id,
        "material_id": material_id,
        "topic": topic,
        "total_questions": len(client_questions),
        "questions": client_questions,
        "is_completed": False,
        "score_pct": 0.0,
    }
