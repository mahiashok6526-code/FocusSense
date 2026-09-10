"""
FocusSense AI — Exam & Assessment Center Engine
=================================================
Orchestrates multi-mode student testing, server-authoritative timers,
hybrid grading (deterministic MCQ + AI semantic rubrics), material-grounded
question generation, dimensional cognitive analytics, and closed-loop
adaptive policy integration.
"""

import json
import logging
import re
import sqlite3
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple, Any

from database import DB_PATH
from ai_service import ai_service
import learning_engine
import study_material_engine

logger = logging.getLogger("FocusSense_AssessmentEngine")
logger.setLevel(logging.INFO)


# =============================================================================
# Helper: Database Connection
# =============================================================================

def get_db_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH, timeout=20.0)
    conn.row_factory = sqlite3.Row
    return conn


# =============================================================================
# Assessment Configuration Defaults
# =============================================================================

MODE_CONFIGS = {
    "QUICK": {
        "default_count": 5,
        "time_limit_sec": None,       # Untimed
        "difficulty": "mixed",
        "description": "Short 5-question pulse check covering basic & conceptual recall."
    },
    "PRACTICE": {
        "default_count": 10,
        "time_limit_sec": None,       # Untimed
        "difficulty": "mixed",
        "description": "Comprehensive 10-question practice with MCQs, short answer, and practical scenarios."
    },
    "TOPIC": {
        "default_count": 8,
        "time_limit_sec": None,       # Untimed
        "difficulty": "adaptive",
        "description": "Targeted topic drill with adaptive difficulty based on mastery history."
    },
    "EXAM": {
        "default_count": 12,
        "time_limit_sec": 900,        # 15 minutes default (configurable)
        "difficulty": "mixed",
        "description": "Timed, rigorous assessment with strict server countdown and comprehensive cognitive analytics."
    }
}


# =============================================================================
# 1. Assessment Creation & Question Generation
# =============================================================================

def create_assessment(
    user_id: int,
    mode: str = "PRACTICE",
    topic: str = "General Study",
    material_id: Optional[int] = None,
    sub_topic: Optional[str] = None,
    difficulty: Optional[str] = None,
    question_count: Optional[int] = None,
    time_limit_sec: Optional[int] = None
) -> Dict[str, Any]:
    """
    Creates a new assessment attempt, generates grounded questions,
    and returns a client-safe assessment payload (with answers stripped).
    """
    mode_upper = (mode or "PRACTICE").upper()
    if mode_upper not in MODE_CONFIGS:
        mode_upper = "PRACTICE"

    cfg = MODE_CONFIGS[mode_upper]
    count = int(question_count) if (question_count and int(question_count) > 0) else cfg["default_count"]
    count = max(1, min(25, count))

    diff = (difficulty or cfg["difficulty"]).lower()
    t_limit = time_limit_sec if time_limit_sec is not None else cfg["time_limit_sec"]
    if mode_upper == "EXAM" and (t_limit is None or t_limit <= 0):
        t_limit = 900  # Default 15 mins for Exam Mode

    topic_clean = (topic or "General Academic Studies").strip()
    if sub_topic and sub_topic.strip():
        topic_clean = f"{topic_clean} - {sub_topic.strip()}"

    # Fetch Material Grounding (if material_id is provided)
    material_excerpts = None
    if material_id:
        try:
            mat = study_material_engine.get_material_by_id(material_id, user_id)
            if mat:
                chunks = study_material_engine.retrieve_relevant_chunks(user_id=user_id, query=topic_clean, material_id=material_id, top_k=4)
                if chunks:
                    material_excerpts = "\n\n".join([f"=== EXCERPT (Chunk #{c['chunk_index']}) ===\n{c['chunk_text']}" for c in chunks])
                elif mat.get("sample_chunks"):
                    material_excerpts = "\n\n".join([f"=== EXCERPT (Chunk #{c['chunk_index']}) ===\n{c['chunk_text']}" for c in mat["sample_chunks"][:4]])
        except Exception as ex:
            logger.warning(f"Error fetching study material for assessment grounding: {ex}")

    # Generate questions via AI Service
    raw_questions = ai_service.generate_assessment_questions(
        topic=topic_clean,
        mode=mode_upper,
        difficulty=diff,
        question_count=count,
        material_excerpts=material_excerpts
    )

    if not raw_questions:
        # Fallback to standard quiz generator
        raw_questions = ai_service.generate_quiz(topic_clean, question_count=count, material_excerpts=material_excerpts)

    now_iso = datetime.now(timezone.utc).isoformat()

    conn = get_db_connection()
    cursor = conn.cursor()

    try:
        # 1. Insert into study_quizzes (Preserve existing quiz architecture)
        cursor.execute("""
        INSERT INTO study_quizzes (
            user_id, session_id, material_id, topic, total_questions,
            score_pct, quality_label, summary_text, created_at,
            assessment_mode, time_limit_sec, time_spent_sec
        ) VALUES (?, NULL, ?, ?, ?, 0.0, 'PENDING', 'Assessment in progress', ?, ?, ?, 0)
        """, (user_id, material_id, topic_clean, len(raw_questions), now_iso, mode_upper, t_limit))
        quiz_id = cursor.lastrowid

        # 2. Insert questions into quiz_questions
        created_questions = []
        for idx, q in enumerate(raw_questions, start=1):
            q_text = q.get("question_text") or f"Question {idx} on {topic_clean}"
            s_concept = q.get("sub_concept") or f"{topic_clean} Core"
            q_type = (q.get("question_type") or "short_answer").lower()
            cat = (q.get("category") or "concept").lower()
            q_diff = (q.get("difficulty") or "medium").lower()
            opts = q.get("options")
            opts_json = json.dumps(opts) if opts else None
            corr_opt = q.get("correct_option")
            sample_ans = q.get("sample_answer") or ""
            expl = q.get("explanation") or ""
            pts = float(q.get("points_possible") or 1.0)

            cursor.execute("""
            INSERT INTO quiz_questions (
                quiz_id, question_index, question_text, sub_concept,
                sample_answer, points_possible, difficulty,
                question_type, options_json, correct_option, category, explanation
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (quiz_id, idx, q_text, s_concept, sample_ans, pts, q_diff, q_type, opts_json, corr_opt, cat, expl))
            q_id = cursor.lastrowid

            created_questions.append({
                "id": q_id,
                "question_index": idx,
                "question_text": q_text,
                "sub_concept": s_concept,
                "question_type": q_type,
                "category": cat,
                "difficulty": q_diff,
                "options": opts,
                # Note: correct_option, sample_answer, explanation are NEVER returned in active exam
                "points_possible": pts
            })

        # 3. Insert into assessment_attempts
        cursor.execute("""
        INSERT INTO assessment_attempts (
            user_id, quiz_id, material_id, topic, sub_topic,
            assessment_mode, difficulty, time_limit_sec, time_spent_sec,
            started_at, completed_at, score_pct
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 0, ?, NULL, 0.0)
        """, (user_id, quiz_id, material_id, topic_clean, sub_topic, mode_upper, diff, t_limit, now_iso))
        attempt_id = cursor.lastrowid

        conn.commit()

        return {
            "success": True,
            "attempt_id": attempt_id,
            "quiz_id": quiz_id,
            "mode": mode_upper,
            "topic": topic_clean,
            "sub_topic": sub_topic,
            "material_id": material_id,
            "time_limit_sec": t_limit,
            "started_at": now_iso,
            "total_questions": len(created_questions),
            "questions": created_questions
        }

    except Exception as ex:
        conn.rollback()
        logger.error(f"Failed to create assessment: {ex}", exc_info=True)
        return {"success": False, "error": str(ex)}
    finally:
        conn.close()


# =============================================================================
# 2. Active Assessment Fetching & Server-Authoritative Timer
# =============================================================================

def get_active_assessment(user_id: int, attempt_id: int) -> Dict[str, Any]:
    """
    Fetches an active assessment attempt, calculates server-authoritative remaining
    time, and safely returns questions with answers stripped.
    """
    conn = get_db_connection()
    cursor = conn.cursor()

    try:
        cursor.execute("""
        SELECT * FROM assessment_attempts
        WHERE id = ? AND user_id = ?
        """, (attempt_id, user_id))
        attempt = cursor.fetchone()

        if not attempt:
            return {"success": False, "error": "Assessment attempt not found or access denied."}

        attempt_dict = dict(attempt)
        quiz_id = attempt_dict["quiz_id"]
        is_completed = bool(attempt_dict.get("completed_at"))

        # Calculate server-authoritative elapsed & remaining time
        started_dt = datetime.fromisoformat(attempt_dict["started_at"].replace("Z", "+00:00"))
        now_dt = datetime.now(timezone.utc)
        elapsed_sec = max(0, int((now_dt - started_dt).total_seconds()))

        time_limit = attempt_dict.get("time_limit_sec")
        remaining_sec = None
        is_expired = False

        if time_limit and time_limit > 0:
            remaining_sec = max(0, time_limit - elapsed_sec)
            if remaining_sec == 0 and not is_completed:
                is_expired = True

        # Fetch questions
        cursor.execute("""
        SELECT * FROM quiz_questions
        WHERE quiz_id = ?
        ORDER BY question_index ASC
        """, (quiz_id,))
        q_rows = cursor.fetchall()

        # Fetch existing student answers
        cursor.execute("""
        SELECT question_id, student_answer, evaluation_status, score_earned, feedback_text
        FROM quiz_answers
        WHERE quiz_id = ? AND user_id = ?
        """, (quiz_id, user_id))
        ans_rows = cursor.fetchall()
        answers_by_qid = {r["question_id"]: dict(r) for r in ans_rows}

        safe_questions = []
        for row in q_rows:
            r_dict = dict(row)
            qid = r_dict["id"]
            opts = json.loads(r_dict["options_json"]) if r_dict.get("options_json") else None

            q_data = {
                "id": qid,
                "question_index": r_dict["question_index"],
                "question_text": r_dict["question_text"],
                "sub_concept": r_dict["sub_concept"],
                "question_type": r_dict.get("question_type", "short_answer"),
                "category": r_dict.get("category", "concept"),
                "difficulty": r_dict.get("difficulty", "medium"),
                "options": opts,
                "points_possible": r_dict.get("points_possible", 1.0),
                "student_answer": answers_by_qid.get(qid, {}).get("student_answer", ""),
                "is_answered": qid in answers_by_qid
            }

            # Only reveal correct answers and explanations IF the assessment is completed
            if is_completed:
                q_data["correct_option"] = r_dict.get("correct_option")
                q_data["sample_answer"] = r_dict.get("sample_answer")
                q_data["explanation"] = r_dict.get("explanation")
                q_data["evaluation_status"] = answers_by_qid.get(qid, {}).get("evaluation_status")
                q_data["score_earned"] = answers_by_qid.get(qid, {}).get("score_earned", 0.0)
                q_data["feedback_text"] = answers_by_qid.get(qid, {}).get("feedback_text", "")

            safe_questions.append(q_data)

        return {
            "success": True,
            "attempt_id": attempt_dict["id"],
            "quiz_id": quiz_id,
            "topic": attempt_dict["topic"],
            "sub_topic": attempt_dict.get("sub_topic"),
            "mode": attempt_dict["assessment_mode"],
            "difficulty": attempt_dict["difficulty"],
            "time_limit_sec": time_limit,
            "elapsed_sec": elapsed_sec,
            "remaining_sec": remaining_sec,
            "is_expired": is_expired,
            "is_completed": is_completed,
            "started_at": attempt_dict["started_at"],
            "completed_at": attempt_dict.get("completed_at"),
            "total_questions": len(safe_questions),
            "answered_count": len(answers_by_qid),
            "questions": safe_questions
        }

    except Exception as ex:
        logger.error(f"Error fetching active assessment {attempt_id}: {ex}", exc_info=True)
        return {"success": False, "error": str(ex)}
    finally:
        conn.close()


# =============================================================================
# 3. Answer Submission & Hybrid Grading
# =============================================================================

def submit_assessment_answer(
    user_id: int,
    attempt_id: int,
    question_id: int,
    student_answer: str
) -> Dict[str, Any]:
    """
    Submits and server-side grades a single assessment question.
    - MCQ / True-False: Deterministic server-side check.
    - Short Answer / Conceptual: AI semantic rubric evaluation.
    Updates concept_mastery immediately.
    """
    conn = get_db_connection()
    cursor = conn.cursor()

    try:
        cursor.execute("""
        SELECT * FROM assessment_attempts
        WHERE id = ? AND user_id = ?
        """, (attempt_id, user_id))
        attempt = cursor.fetchone()

        if not attempt:
            return {"success": False, "error": "Assessment attempt not found."}

        attempt_dict = dict(attempt)
        if attempt_dict.get("completed_at"):
            return {"success": False, "error": "Assessment is already finalized."}

        quiz_id = attempt_dict["quiz_id"]
        cursor.execute("SELECT topic FROM study_quizzes WHERE id = ?", (quiz_id,))
        qz_row = cursor.fetchone()
        topic = qz_row["topic"] if qz_row and qz_row["topic"] else "General Academic Studies"

        # Fetch question details
        cursor.execute("""
        SELECT * FROM quiz_questions
        WHERE id = ? AND quiz_id = ?
        """, (question_id, quiz_id))
        question = cursor.fetchone()

        if not question:
            return {"success": False, "error": "Question not found in this assessment."}

        q_dict = dict(question)
        q_type = (q_dict.get("question_type") or "short_answer").lower()
        sub_concept = q_dict["sub_concept"]
        pts_possible = float(q_dict.get("points_possible") or 1.0)
        cleaned_answer = (student_answer or "").strip()

        # Hybrid Grading Logic
        if q_type in ["mcq", "true_false"]:
            # Deterministic Server-Side Checking
            corr_opt = (q_dict.get("correct_option") or "").strip().upper()
            std_opt = cleaned_answer.upper()

            # Normalize option letters (e.g., "A) Option" -> "A", "TRUE" -> "True")
            std_match = re.match(r"^([A-D]|TRUE|FALSE)\b", std_opt)
            std_choice = std_match.group(1) if std_match else std_opt

            corr_match = re.match(r"^([A-D]|TRUE|FALSE)\b", corr_opt)
            corr_choice = corr_match.group(1) if corr_match else corr_opt

            if std_choice and (std_choice == corr_choice):
                status = "correct"
                score_earned = pts_possible
                feedback = "Correct! Well done."
            elif not cleaned_answer:
                status = "unanswered"
                score_earned = 0.0
                feedback = "No answer was selected."
            else:
                status = "incorrect"
                score_earned = 0.0
                feedback = f"Incorrect. The correct choice is option {corr_choice}."
        else:
            # Semantic / Rubric AI Evaluation
            sample_ans = q_dict.get("sample_answer") or ""
            eval_res = ai_service.evaluate_answer(
                question_text=q_dict["question_text"],
                sub_concept=sub_concept,
                sample_answer=sample_ans,
                student_answer=cleaned_answer
            )
            status = eval_res.get("status", "incorrect")
            score_earned = float(eval_res.get("score", 0.0)) * pts_possible
            feedback = eval_res.get("feedback", "Answer evaluated.")

        now_iso = datetime.now(timezone.utc).isoformat()

        # Upsert answer into quiz_answers
        cursor.execute("""
        SELECT id FROM quiz_answers
        WHERE quiz_id = ? AND question_id = ? AND user_id = ?
        """, (quiz_id, question_id, user_id))
        existing_ans = cursor.fetchone()

        if existing_ans:
            cursor.execute("""
            UPDATE quiz_answers
            SET student_answer = ?, evaluation_status = ?, score_earned = ?, feedback_text = ?, answered_at = ?
            WHERE id = ?
            """, (cleaned_answer, status, score_earned, feedback, now_iso, existing_ans["id"]))
        else:
            cursor.execute("""
            INSERT INTO quiz_answers (
                question_id, quiz_id, user_id, student_answer,
                evaluation_status, score_earned, feedback_text, answered_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (question_id, quiz_id, user_id, cleaned_answer, status, score_earned, feedback, now_iso))

        # Update Concept Mastery (single source of truth matching FocusSense schema)
        is_correct = 1 if status == "correct" else 0
        is_partial = 1 if status == "partial" else 0
        is_incorrect = 1 if status in ["incorrect", "unanswered"] else 0

        cursor.execute("""
        SELECT id, total_tested, correct_count, partial_count, incorrect_count
        FROM concept_mastery
        WHERE user_id = ? AND topic = ? AND sub_concept = ?
        """, (user_id, topic, sub_concept))
        cm_row = cursor.fetchone()

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
            INSERT INTO concept_mastery (
                user_id, topic, sub_concept, total_tested, correct_count,
                partial_count, incorrect_count, mastery_pct, is_weak, last_tested_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (user_id, topic, sub_concept, tot, corr, part, inc, mastery, weak_flag, now_iso))

        conn.commit()

        return {
            "success": True,
            "question_id": question_id,
            "status": status,
            "score_earned": score_earned,
            "feedback": feedback
        }

    except Exception as ex:
        conn.rollback()
        logger.error(f"Error submitting answer for attempt {attempt_id}, question {question_id}: {ex}", exc_info=True)
        return {"success": False, "error": str(ex)}
    finally:
        conn.close()


# =============================================================================
# 4. Assessment Finalization, Dimensional Analytics & Adaptive Loop
# =============================================================================

def finalize_assessment(user_id: int, attempt_id: int) -> Dict[str, Any]:
    """
    Finalizes an assessment attempt:
    - Calculates server-authoritative elapsed time.
    - Computes overall score & 4 cognitive dimensional scores:
      (Concept Understanding, Recall, Application, Problem Solving).
    - Identifies Strong Concepts (>= 75%) and Weak Concepts (< 70%).
    - Updates topic_mastery using the single source of truth (learning_engine).
    - Triggers closed-loop adaptive action via select_adaptive_action().
    - Updates assessment_attempts with complete results.
    """
    conn = get_db_connection()
    cursor = conn.cursor()

    try:
        cursor.execute("""
        SELECT * FROM assessment_attempts
        WHERE id = ? AND user_id = ?
        """, (attempt_id, user_id))
        attempt = cursor.fetchone()

        if not attempt:
            return {"success": False, "error": "Assessment attempt not found."}

        attempt_dict = dict(attempt)
        quiz_id = attempt_dict["quiz_id"]
        topic = attempt_dict["topic"]
        mode = attempt_dict["assessment_mode"]

        now_dt = datetime.now(timezone.utc)
        now_iso = now_dt.isoformat()
        started_dt = datetime.fromisoformat(attempt_dict["started_at"].replace("Z", "+00:00"))
        time_spent_sec = max(0, int((now_dt - started_dt).total_seconds()))

        # Fetch all questions and answers
        cursor.execute("""
        SELECT * FROM quiz_questions
        WHERE quiz_id = ?
        ORDER BY question_index ASC
        """, (quiz_id,))
        questions = [dict(r) for r in cursor.fetchall()]

        cursor.execute("""
        SELECT * FROM quiz_answers
        WHERE quiz_id = ? AND user_id = ?
        """, (quiz_id, user_id))
        answers = {r["question_id"]: dict(r) for r in cursor.fetchall()}

        # 1. Calculate Dimensional & Overall Scores
        total_possible = 0.0
        total_earned = 0.0

        dimension_totals = {
            "concept": {"possible": 0.0, "earned": 0.0},
            "recall": {"possible": 0.0, "earned": 0.0},
            "application": {"possible": 0.0, "earned": 0.0},
            "problem_solving": {"possible": 0.0, "earned": 0.0}
        }

        concept_scores_map = {}

        for q in questions:
            qid = q["id"]
            pts = float(q.get("points_possible") or 1.0)
            cat = (q.get("category") or "concept").lower()
            if cat not in dimension_totals:
                cat = "concept"
            sub_c = q["sub_concept"]

            total_possible += pts
            dimension_totals[cat]["possible"] += pts

            ans = answers.get(qid)
            score_earned = float(ans["score_earned"]) if ans else 0.0

            total_earned += score_earned
            dimension_totals[cat]["earned"] += score_earned

            if sub_c not in concept_scores_map:
                concept_scores_map[sub_c] = {"possible": 0.0, "earned": 0.0}
            concept_scores_map[sub_c]["possible"] += pts
            concept_scores_map[sub_c]["earned"] += score_earned

        overall_score_pct = round((total_earned / total_possible) * 100.0, 1) if total_possible > 0 else 0.0

        # Compute dimension percentages (fall back to overall score if category has 0 questions)
        dim_scores = {}
        for dim_name, dim_data in dimension_totals.items():
            if dim_data["possible"] > 0:
                dim_scores[dim_name] = round((dim_data["earned"] / dim_data["possible"]) * 100.0, 1)
            else:
                dim_scores[dim_name] = overall_score_pct

        concept_score = dim_scores["concept"]
        recall_score = dim_scores["recall"]
        application_score = dim_scores["application"]
        problem_solving_score = dim_scores["problem_solving"]

        # 2. Identify Strong (>= 75%) vs Weak (< 70%) Concepts
        strong_concepts = []
        weak_concepts = []

        for sc_name, sc_data in concept_scores_map.items():
            sc_pct = (sc_data["earned"] / sc_data["possible"]) * 100.0 if sc_data["possible"] > 0 else 0.0
            if sc_pct >= 75.0:
                strong_concepts.append({"concept": sc_name, "score_pct": round(sc_pct, 1)})
            elif sc_pct < 70.0:
                weak_concepts.append({
                    "concept": sc_name,
                    "score_pct": round(sc_pct, 1),
                    "reason": f"Scored {round(sc_pct, 1)}% in assessment. Key operational mechanisms require review."
                })

        strong_json = json.dumps(strong_concepts)
        weak_json = json.dumps(weak_concepts)

        # Quality Label
        if overall_score_pct >= 85:
            quality_label = "EXCELLENT"
        elif overall_score_pct >= 70:
            quality_label = "PROFICIENT"
        elif overall_score_pct >= 50:
            quality_label = "DEVELOPING"
        else:
            quality_label = "NEEDS_REVISION"

        # Synthesis Summary
        synthesis = (
            f"Completed {mode} Assessment on '{topic}'. "
            f"Overall Score: {overall_score_pct}% ({quality_label}). "
            f"Concept Understanding: {concept_score}%, Recall: {recall_score}%, "
            f"Application: {application_score}%, Problem Solving: {problem_solving_score}%."
        )

        # 3. Update topic_mastery (Single source of truth via learning_engine)
        evaluated_answers_list = []
        for q in questions:
            qid = q["id"]
            ans = answers.get(qid, {})
            evaluated_answers_list.append({
                "question_id": qid,
                "sub_concept": q["sub_concept"],
                "score_earned": ans.get("score_earned", 0.0),
                "points_possible": q.get("points_possible", 1.0),
                "evaluation_status": ans.get("evaluation_status", "unanswered")
            })

        try:
            learning_engine.update_topic_mastery(
                user_id=user_id,
                topic=topic,
                retention_score=overall_score_pct,
                answers=evaluated_answers_list
            )
        except Exception as ex:
            logger.warning(f"Error updating topic mastery: {ex}")

        # 4. Trigger Closed-Loop Adaptive Action (Single source of truth)
        adaptive_action_id = None
        adaptive_recommendation = None
        try:
            current_state = {
                "user_id": user_id,
                "topic": topic,
                "retention_score_pct": overall_score_pct,
                "study_duration_sec": time_spent_sec,
                "mastery_tier": "STRONG" if overall_score_pct >= 85 else ("MEDIUM" if overall_score_pct >= 60 else "WEAK"),
                "is_passive_alert": False
            }
            action_res = learning_engine.select_adaptive_action(current_state)
            if action_res:
                adaptive_recommendation = action_res
                adaptive_action_id = learning_engine.record_adaptive_action(
                    user_id=user_id,
                    topic=topic,
                    session_id=None,
                    state_id=None,
                    action_type=action_res.get("action_type", "MAINTAIN_MASTERY"),
                    action_payload=action_res.get("action_payload", {}),
                    reason=action_res.get("reason", "Assessment performance recorded.")
                )
        except Exception as ex:
            logger.warning(f"Error selecting adaptive action: {ex}")

        # 5. Update study_quizzes & assessment_attempts
        cursor.execute("""
        UPDATE study_quizzes
        SET score_pct = ?, quality_label = ?, summary_text = ?,
            time_spent_sec = ?, concept_score = ?, application_score = ?,
            recall_score = ?, problem_solving_score = ?
        WHERE id = ?
        """, (overall_score_pct, quality_label, synthesis, time_spent_sec,
              concept_score, application_score, recall_score, problem_solving_score, quiz_id))

        cursor.execute("""
        UPDATE assessment_attempts
        SET completed_at = ?, time_spent_sec = ?, score_pct = ?,
            concept_score = ?, application_score = ?, recall_score = ?,
            problem_solving_score = ?, strong_topics_json = ?, weak_topics_json = ?,
            synthesis_summary = ?, adaptive_action_id = ?
        WHERE id = ?
        """, (now_iso, time_spent_sec, overall_score_pct,
              concept_score, application_score, recall_score,
              problem_solving_score, strong_json, weak_json,
              synthesis, adaptive_action_id, attempt_id))

        conn.commit()

        return {
            "success": True,
            "attempt_id": attempt_id,
            "quiz_id": quiz_id,
            "mode": mode,
            "topic": topic,
            "time_spent_sec": time_spent_sec,
            "overall_score_pct": overall_score_pct,
            "quality_label": quality_label,
            "dimensions": {
                "concept_score": concept_score,
                "recall_score": recall_score,
                "application_score": application_score,
                "problem_solving_score": problem_solving_score
            },
            "strong_concepts": strong_concepts,
            "weak_concepts": weak_concepts,
            "synthesis_summary": synthesis,
            "adaptive_recommendation": adaptive_recommendation
        }

    except Exception as ex:
        conn.rollback()
        logger.error(f"Error finalizing assessment {attempt_id}: {ex}", exc_info=True)
        return {"success": False, "error": str(ex)}
    finally:
        conn.close()


# =============================================================================
# 5. Comprehensive Assessment Results & Review
# =============================================================================

def get_assessment_results(user_id: int, attempt_id: int) -> Dict[str, Any]:
    """
    Returns full post-assessment performance report including dimensional
    analytics, strong/weak concept breakdowns, and question-by-question review
    with revealed correct answers and explanations.
    """
    conn = get_db_connection()
    cursor = conn.cursor()

    try:
        cursor.execute("""
        SELECT * FROM assessment_attempts
        WHERE id = ? AND user_id = ?
        """, (attempt_id, user_id))
        attempt = cursor.fetchone()

        if not attempt:
            return {"success": False, "error": "Assessment attempt not found."}

        attempt_dict = dict(attempt)
        quiz_id = attempt_dict["quiz_id"]

        # Fetch questions
        cursor.execute("""
        SELECT * FROM quiz_questions
        WHERE quiz_id = ?
        ORDER BY question_index ASC
        """, (quiz_id,))
        questions = [dict(r) for r in cursor.fetchall()]

        # Fetch answers
        cursor.execute("""
        SELECT * FROM quiz_answers
        WHERE quiz_id = ? AND user_id = ?
        """, (quiz_id, user_id))
        answers = {r["question_id"]: dict(r) for r in cursor.fetchall()}

        question_reviews = []
        for q in questions:
            qid = q["id"]
            opts = json.loads(q["options_json"]) if q.get("options_json") else None
            ans = answers.get(qid, {})

            question_reviews.append({
                "id": qid,
                "question_index": q["question_index"],
                "question_text": q["question_text"],
                "sub_concept": q["sub_concept"],
                "question_type": q.get("question_type", "short_answer"),
                "category": q.get("category", "concept"),
                "difficulty": q.get("difficulty", "medium"),
                "options": opts,
                "correct_option": q.get("correct_option"),
                "sample_answer": q.get("sample_answer"),
                "explanation": q.get("explanation"),
                "points_possible": q.get("points_possible", 1.0),
                "student_answer": ans.get("student_answer", ""),
                "evaluation_status": ans.get("evaluation_status", "unanswered"),
                "score_earned": ans.get("score_earned", 0.0),
                "feedback_text": ans.get("feedback_text", "No response submitted.")
            })

        strong_concepts = json.loads(attempt_dict.get("strong_topics_json") or "[]")
        weak_concepts = json.loads(attempt_dict.get("weak_topics_json") or "[]")

        # Fetch adaptive recommendation if linked
        adaptive_rec = None
        if attempt_dict.get("adaptive_action_id"):
            cursor.execute("""
            SELECT * FROM adaptive_actions WHERE id = ?
            """, (attempt_dict["adaptive_action_id"],))
            action_row = cursor.fetchone()
            if action_row:
                adaptive_rec = dict(action_row)

        return {
            "success": True,
            "attempt_id": attempt_dict["id"],
            "quiz_id": quiz_id,
            "topic": attempt_dict["topic"],
            "sub_topic": attempt_dict.get("sub_topic"),
            "mode": attempt_dict["assessment_mode"],
            "difficulty": attempt_dict["difficulty"],
            "time_limit_sec": attempt_dict.get("time_limit_sec"),
            "time_spent_sec": attempt_dict.get("time_spent_sec", 0),
            "started_at": attempt_dict["started_at"],
            "completed_at": attempt_dict.get("completed_at"),
            "score_pct": attempt_dict.get("score_pct", 0.0),
            "dimensions": {
                "concept_score": attempt_dict.get("concept_score", 0.0),
                "recall_score": attempt_dict.get("recall_score", 0.0),
                "application_score": attempt_dict.get("application_score", 0.0),
                "problem_solving_score": attempt_dict.get("problem_solving_score", 0.0)
            },
            "strong_concepts": strong_concepts,
            "weak_concepts": weak_concepts,
            "synthesis_summary": attempt_dict.get("synthesis_summary", ""),
            "adaptive_recommendation": adaptive_rec,
            "questions": question_reviews
        }

    except Exception as ex:
        logger.error(f"Error fetching results for attempt {attempt_id}: {ex}", exc_info=True)
        return {"success": False, "error": str(ex)}
    finally:
        conn.close()


# =============================================================================
# 6. Assessment History
# =============================================================================

def get_user_assessment_history(user_id: int, limit: int = 25) -> List[Dict[str, Any]]:
    """
    Returns persistent historical list of assessment attempts for the student.
    Never overwrites or deletes previous attempts.
    """
    conn = get_db_connection()
    cursor = conn.cursor()

    try:
        cursor.execute("""
        SELECT a.*, q.total_questions
        FROM assessment_attempts a
        JOIN study_quizzes q ON a.quiz_id = q.id
        WHERE a.user_id = ?
        ORDER BY a.id DESC
        LIMIT ?
        """, (user_id, limit))
        rows = cursor.fetchall()

        history = []
        for r in rows:
            r_dict = dict(r)
            history.append({
                "attempt_id": r_dict["id"],
                "quiz_id": r_dict["quiz_id"],
                "topic": r_dict["topic"],
                "sub_topic": r_dict.get("sub_topic"),
                "mode": r_dict["assessment_mode"],
                "difficulty": r_dict.get("difficulty", "mixed"),
                "total_questions": r_dict.get("total_questions", 0),
                "score_pct": r_dict.get("score_pct", 0.0),
                "concept_score": r_dict.get("concept_score", 0.0),
                "recall_score": r_dict.get("recall_score", 0.0),
                "application_score": r_dict.get("application_score", 0.0),
                "problem_solving_score": r_dict.get("problem_solving_score", 0.0),
                "time_spent_sec": r_dict.get("time_spent_sec", 0),
                "started_at": r_dict["started_at"],
                "completed_at": r_dict.get("completed_at"),
                "is_completed": bool(r_dict.get("completed_at"))
            })

        return history

    except Exception as ex:
        logger.error(f"Error fetching assessment history for user {user_id}: {ex}", exc_info=True)
        return []
    finally:
        conn.close()


# =============================================================================
# 7. Socratic AI Mistake Remediation
# =============================================================================

def explain_assessment_mistake(
    user_id: int,
    attempt_id: int,
    question_id: int
) -> Dict[str, Any]:
    """
    Provides a Socratic, empathetic FocusSense AI explanation for an incorrect
    or incomplete assessment answer.
    """
    conn = get_db_connection()
    cursor = conn.cursor()

    try:
        # Verify attempt ownership
        cursor.execute("""
        SELECT * FROM assessment_attempts
        WHERE id = ? AND user_id = ?
        """, (attempt_id, user_id))
        attempt = cursor.fetchone()

        if not attempt:
            return {"success": False, "error": "Assessment attempt not found."}

        attempt_dict = dict(attempt)
        quiz_id = attempt_dict["quiz_id"]
        topic = attempt_dict["topic"]

        # Fetch question
        cursor.execute("""
        SELECT * FROM quiz_questions
        WHERE id = ? AND quiz_id = ?
        """, (question_id, quiz_id))
        question = cursor.fetchone()

        if not question:
            return {"success": False, "error": "Question not found."}

        q_dict = dict(question)

        # Fetch student's answer
        cursor.execute("""
        SELECT * FROM quiz_answers
        WHERE quiz_id = ? AND question_id = ? AND user_id = ?
        """, (quiz_id, question_id, user_id))
        ans = cursor.fetchone()
        student_ans = dict(ans)["student_answer"] if ans else ""

        # Call AI service for Socratic explanation
        explanation = ai_service.explain_assessment_mistake(
            question_text=q_dict["question_text"],
            student_answer=student_ans,
            ideal_answer=q_dict.get("sample_answer") or q_dict.get("explanation") or "",
            sub_concept=q_dict["sub_concept"],
            topic=topic
        )

        return {
            "success": True,
            "question_id": question_id,
            "sub_concept": q_dict["sub_concept"],
            "explanation": explanation
        }

    except Exception as ex:
        logger.error(f"Error explaining mistake for attempt {attempt_id}, question {question_id}: {ex}", exc_info=True)
        return {"success": False, "error": str(ex)}
    finally:
        conn.close()
