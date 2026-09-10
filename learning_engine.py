"""
FocusSense Learning Intelligence Engine
=======================================
Distinguishes Physical Desk Presence, Study Behavior, and Knowledge Retention.

Core Pillars:
1. Physical Presence (Radar Telemetry) != Demonstrated Learning.
2. Evaluates dynamic recall test performance to compute true Knowledge Retention Scores.
3. Automatically tracks topic-level mastery tiers:
   - WEAK (< 60% retention)
   - MEDIUM (60% - 84% retention)
   - STRONG (>= 85% retention)
4. Adapts future quiz difficulty and revision recommendations based on historical mastery.
5. Integrates Knowledge Retention into the Composite 3-Pillar FocusScore.
"""

import json
import math
import sqlite3
from datetime import datetime

from database import DATABASE, DB_PATH


def get_db_connection(database_path=None):
    target_db = database_path or DATABASE
    conn = sqlite3.connect(target_db, timeout=30.0)
    conn.row_factory = sqlite3.Row
    return conn


def _get_val(obj, key, default=None):
    if hasattr(obj, "get"):
        val = obj.get(key, default)
        return val if val is not None else default
    try:
        if hasattr(obj, "keys") and key in obj.keys():
            val = obj[key]
            return val if val is not None else default
    except Exception:
        pass
    return default


# ---------------------------------------------------------------------------
# 1. Retention Calculation
# ---------------------------------------------------------------------------

def calculate_retention_score(answers: list) -> dict:
    """
    Calculate normalized knowledge retention score (0 - 100%) from evaluated question answers.
    Answers format: list of dicts or sqlite3.Row objects with 'score_earned' and 'points_possible'.
    """
    if not answers:
        return {
            "retention_score": 0.0,
            "quality_label": "No Answers Recorded",
            "tier": "WEAK",
            "total_questions": 0,
            "correct_count": 0,
            "partial_count": 0,
            "incorrect_count": 0,
            "learned_well": [],
            "needs_revision": [],
        }

    total_possible = 0.0
    total_earned = 0.0
    correct_count = 0
    partial_count = 0
    incorrect_count = 0
    learned_well = []
    needs_revision = []

    for a in answers:
        pts = float(_get_val(a, "points_possible", 1.0) or 1.0)
        status = (_get_val(a, "evaluation_status") or _get_val(a, "status") or "").lower()
        score = _get_val(a, "score_earned")
        sub_c = _get_val(a, "sub_concept") or _get_val(a, "concept") or _get_val(a, "question_text") or "Core Concept"

        if score is not None:
            earned = float(score)
        else:
            if status == "correct":
                earned = pts
            elif status == "partial":
                earned = pts * 0.5
            else:
                earned = 0.0

        total_possible += pts
        total_earned += earned

        if status == "correct":
            correct_count += 1
            if sub_c not in learned_well:
                learned_well.append(sub_c)
        elif status == "partial":
            partial_count += 1
            if sub_c not in needs_revision:
                needs_revision.append(sub_c)
        else:
            incorrect_count += 1
            if sub_c not in needs_revision:
                needs_revision.append(sub_c)

    retention_pct = round((total_earned / max(0.1, total_possible)) * 100.0, 1)
    retention_pct = max(0.0, min(100.0, retention_pct))

    if retention_pct >= 85.0:
        quality_label = "Strong Mastery"
        tier = "STRONG"
    elif retention_pct >= 60.0:
        quality_label = "Developing Mastery"
        tier = "MEDIUM"
    else:
        quality_label = "Needs Review"
        tier = "WEAK"

    return {
        "retention_score": retention_pct,
        "quality_label": quality_label,
        "tier": tier,
        "total_questions": len(answers),
        "correct_count": correct_count,
        "partial_count": partial_count,
        "incorrect_count": incorrect_count,
        "learned_well": learned_well,
        "needs_revision": needs_revision,
    }


# ---------------------------------------------------------------------------
# 2. Topic Mastery Aggregation & Persistence
# ---------------------------------------------------------------------------

def update_topic_mastery(user_id: int, topic: str, retention_score: float, answers: list = None) -> dict:
    """
    Update topic-level mastery tiers in SQLite (topic_mastery table) following a completed recall test.
    Categorizes topics as 'WEAK' (<60%), 'MEDIUM' (60-84%), or 'STRONG' (>=85%).
    """
    if not topic or not user_id:
        return {}

    topic = topic.strip()
    now_iso = datetime.now().isoformat()
    answers = answers or []

    q_count = len(answers)
    c_count = sum(1 for a in answers if (_get_val(a, "evaluation_status") or "").lower() == "correct")
    p_count = sum(1 for a in answers if (_get_val(a, "evaluation_status") or "").lower() == "partial")
    i_count = sum(1 for a in answers if (_get_val(a, "evaluation_status") or "").lower() in ["incorrect", "unanswered"])

    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute("SELECT * FROM topic_mastery WHERE user_id = ? AND topic = ?", (user_id, topic))
    existing = cursor.fetchone()

    if existing:
        new_total_quizzes = (existing["total_quizzes"] or 0) + 1
        new_total_questions = (existing["total_questions"] or 0) + q_count
        new_correct = (existing["correct_count"] or 0) + c_count
        new_partial = (existing["partial_count"] or 0) + p_count
        new_incorrect = (existing["incorrect_count"] or 0) + i_count

        # Running average retention score
        old_avg = float(existing["avg_retention_score"] or retention_score)
        new_avg = round(((old_avg * (new_total_quizzes - 1)) + retention_score) / float(new_total_quizzes), 1)

        # Classification based on current and running retention
        if new_avg >= 85.0:
            tier = "STRONG"
        elif new_avg >= 60.0:
            tier = "MEDIUM"
        else:
            tier = "WEAK"

        cursor.execute("""
            UPDATE topic_mastery
            SET mastery_tier = ?, total_quizzes = ?, total_questions = ?,
                correct_count = ?, partial_count = ?, incorrect_count = ?,
                avg_retention_score = ?, last_retention_score = ?,
                last_tested_at = ?, updated_at = ?
            WHERE id = ?
        """, (tier, new_total_quizzes, new_total_questions, new_correct, new_partial, new_incorrect,
              new_avg, retention_score, now_iso, now_iso, existing["id"]))
        mastery_id = existing["id"]
    else:
        new_avg = retention_score
        if retention_score >= 85.0:
            tier = "STRONG"
        elif retention_score >= 60.0:
            tier = "MEDIUM"
        else:
            tier = "WEAK"

        cursor.execute("""
            INSERT INTO topic_mastery (
                user_id, topic, mastery_tier, total_quizzes, total_questions,
                correct_count, partial_count, incorrect_count,
                avg_retention_score, last_retention_score, last_tested_at, updated_at
            ) VALUES (?, ?, ?, 1, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (user_id, topic, tier, q_count, c_count, p_count, i_count,
              retention_score, retention_score, now_iso, now_iso))
        mastery_id = cursor.lastrowid

    conn.commit()
    conn.close()

    return {
        "id": mastery_id,
        "user_id": user_id,
        "topic": topic,
        "mastery_tier": tier,
        "avg_retention_score": new_avg,
        "last_retention_score": retention_score,
        "updated_at": now_iso,
    }


def get_topic_mastery_list(user_id: int) -> list:
    """
    Retrieve all topic mastery records for a student sorted by weakest topics first.
    """
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT id, topic, mastery_tier, total_quizzes, total_questions,
               correct_count, partial_count, incorrect_count,
               avg_retention_score, last_retention_score, last_tested_at
        FROM topic_mastery
        WHERE user_id = ?
        ORDER BY CASE mastery_tier WHEN 'WEAK' THEN 1 WHEN 'MEDIUM' THEN 2 ELSE 3 END, avg_retention_score ASC
    """, (user_id,))
    rows = cursor.fetchall()
    conn.close()

    results = []
    for r in rows:
        results.append({
            "id": r["id"],
            "topic": r["topic"],
            "mastery_tier": r["mastery_tier"],
            "total_quizzes": r["total_quizzes"],
            "total_questions": r["total_questions"],
            "avg_retention_score": r["avg_retention_score"],
            "last_retention_score": r["last_retention_score"],
            "last_tested_at": r["last_tested_at"],
            "badge_color": "var(--accent-rose)" if r["mastery_tier"] == "WEAK" else ("var(--accent-amber)" if r["mastery_tier"] == "MEDIUM" else "var(--accent-emerald)"),
        })
    return results


def get_weak_topics(user_id: int) -> list:
    """
    Retrieve weak topics (<60% mastery or WEAK tier) that require focused spaced repetition.
    """
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT topic, mastery_tier, avg_retention_score, last_retention_score, total_quizzes
        FROM topic_mastery
        WHERE user_id = ? AND (mastery_tier = 'WEAK' OR avg_retention_score < 70.0)
        ORDER BY avg_retention_score ASC
    """, (user_id,))
    rows = cursor.fetchall()
    conn.close()

    return [dict(r) for r in rows]


# ---------------------------------------------------------------------------
# 3. Adaptive Quiz Difficulty Configuration
# ---------------------------------------------------------------------------

def get_adaptive_quiz_config(user_id: int, topic: str) -> dict:
    """
    Determine the optimal 3-tier question difficulty balance based on the student's mastery history.
    - WEAK: Prioritizes foundational remediation (2 Basic/Foundational, 1 Application).
    - STRONG: Challenges student with deeper concepts (1 Application, 2 Harder/Edge Case/Debugging).
    - MEDIUM or NEW: Standard 3-tier distribution (1 Basic, 1 Conceptual/Application, 1 Harder/Edge Case).
    """
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT mastery_tier, avg_retention_score, total_quizzes FROM topic_mastery WHERE user_id = ? AND topic = ?", (user_id, topic))
    row = cursor.fetchone()
    conn.close()

    if not row:
        return {
            "strategy": "STANDARD",
            "tier": "NEW",
            "tier_distribution": ["basic", "application", "advanced"],
            "guidance": "Standard 3-tier difficulty: 1 Basic definition, 1 Practical application, 1 Harder edge-case/trade-off question."
        }

    tier = row["mastery_tier"]
    if tier == "WEAK":
        return {
            "strategy": "REMEDIATION",
            "tier": "WEAK",
            "tier_distribution": ["basic", "basic", "application"],
            "guidance": "Remediation focus: Reinforce foundational definitions and basic mechanisms to resolve misconceptions."
        }
    elif tier == "STRONG":
        return {
            "strategy": "ADVANCED_CHALLENGE",
            "tier": "STRONG",
            "tier_distribution": ["application", "advanced", "advanced"],
            "guidance": "Advanced challenge: Deepen comprehension with complex scenarios, critical trade-offs, and edge-case debugging."
        }
    else:
        return {
            "strategy": "STANDARD",
            "tier": "MEDIUM",
            "tier_distribution": ["basic", "application", "advanced"],
            "guidance": "Progressive 3-tier difficulty: 1 Basic definition, 1 Practical application, 1 Harder edge-case/trade-off question."
        }


# ---------------------------------------------------------------------------
# 4. Student Learning Profile Aggregator
# ---------------------------------------------------------------------------

def get_student_learning_profile(user_id: int) -> dict:
    """
    Synthesizes overall learning intelligence for a student:
    - Topic mastery breakdown (Weak, Medium, Strong)
    - Average knowledge retention across all quizzes
    - Cognitive efficiency diagnosis (Physical Presence vs Demonstrated Retention)
    """
    conn = get_db_connection()
    cursor = conn.cursor()

    # Query all topic mastery records
    cursor.execute("SELECT * FROM topic_mastery WHERE user_id = ? ORDER BY avg_retention_score ASC", (user_id,))
    topics = cursor.fetchall()

    # Query total recall tasks completed
    cursor.execute("SELECT COUNT(*), AVG(retention_score) FROM recall_tasks WHERE user_id = ? AND status = 'COMPLETED'", (user_id,))
    task_stats = cursor.fetchone()
    conn.close()

    total_quizzes = task_stats[0] if task_stats else 0
    overall_avg = round(float(task_stats[1] or 0.0), 1) if task_stats and task_stats[1] is not None else 0.0

    weak_list = []
    medium_list = []
    strong_list = []

    for t in topics:
        t_dict = dict(t)
        tier = t["mastery_tier"]
        if tier == "WEAK":
            weak_list.append(t_dict)
        elif tier == "MEDIUM":
            medium_list.append(t_dict)
        else:
            strong_list.append(t_dict)

    # Diagnosis synthesis
    if total_quizzes == 0:
        diagnosis = "No recall tests completed yet. Take your first test after completing a study session."
    elif overall_avg >= 85.0:
        diagnosis = "⭐ Exceptional Knowledge Consolidation: High study effort is translating effectively into long-term recall."
    elif overall_avg >= 65.0:
        diagnosis = "📈 Steady Retention: Solid foundational comprehension. Target identified weak concepts for review."
    else:
        diagnosis = "⚠️ Retention Alert: High desk presence detected with lower recall scores. Shift from passive reading to active retrieval practice."

    return {
        "user_id": user_id,
        "total_topics_tracked": len(topics),
        "total_quizzes_completed": total_quizzes,
        "overall_retention_avg": overall_avg,
        "weak_count": len(weak_list),
        "medium_count": len(medium_list),
        "strong_count": len(strong_list),
        "weak_topics": weak_list,
        "medium_topics": medium_list,
        "strong_topics": strong_list,
        "all_topics": [dict(t) for t in topics],
        "diagnosis": diagnosis,
    }


# ---------------------------------------------------------------------------
# 5. Adaptive Learning Telemetry & Future RL Logging
# ---------------------------------------------------------------------------

def log_adaptive_learning_event(user_id: int, session_id: int, topic: str,
                               prior_mastery_tier: str, action_intervention: str,
                               retention_outcome_pct: float) -> dict:
    """
    Persists student learning state, action/intervention, and outcome/reward
    into adaptive_learning_logs table as a foundation for future Reinforcement Learning.
    """
    if not user_id or not topic:
        return {}

    topic = topic.strip()
    conn = get_db_connection()
    cursor = conn.cursor()

    # Get prior average retention score for reward calculation
    cursor.execute("SELECT avg_retention_score FROM topic_mastery WHERE user_id = ? AND topic = ?", (user_id, topic))
    row = cursor.fetchone()
    prior_avg = float(row["avg_retention_score"]) if row and row["avg_retention_score"] is not None else retention_outcome_pct
    reward_delta = round(retention_outcome_pct - prior_avg, 1)
    now_iso = datetime.now().isoformat()

    try:
        cursor.execute("""
            INSERT INTO adaptive_learning_logs (
                user_id, session_id, topic, prior_mastery_tier, action_intervention,
                retention_outcome_pct, reward_delta, created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            user_id, session_id, topic, prior_mastery_tier, action_intervention,
            retention_outcome_pct, reward_delta, now_iso
        ))
        conn.commit()
    except Exception as e:
        print(f"[AdaptiveLearning] Log error: {e}")
    finally:
        conn.close()

    return {
        "user_id": user_id,
        "session_id": session_id,
        "topic": topic,
        "prior_mastery_tier": prior_mastery_tier,
        "action_intervention": action_intervention,
        "retention_outcome_pct": retention_outcome_pct,
        "reward_delta": reward_delta,
        "created_at": now_iso,
    }


# ---------------------------------------------------------------------------
# 6. Unified 8-Dimension Focus Intelligence Model
# ---------------------------------------------------------------------------

def compute_unified_focus_score(metrics: dict) -> dict:
    """
    Computes the transparent 8-dimension Unified Focus Intelligence Score:
    1. Physical Presence (25% Weight)
    2. Knowledge Retention (25% Weight)
    3. AI Learning Engagement (15% Weight)
    4. Quiz Performance (15% Weight)
    5. Study Duration Endurance (10% Weight)
    6. Session Completion (5% Weight)
    7. Study Habit Consistency (5% Weight)
    Total = 100%
    Distraction Interruption Penalty: Subtracted from total (max 15 pts).

    Strict Cognitive Contrast Rule:
    - If Physical Presence >= 85% and Knowledge Retention < 50%,
      cap score at 55 and flag 'Passive Study Alert'.
    """
    presence_pct = max(0.0, min(100.0, float(_get_val(metrics, "presence_pct", 100.0) or 100.0)))
    retention_raw = _get_val(metrics, "retention_pct")
    quiz_raw = _get_val(metrics, "quiz_score_pct") or _get_val(metrics, "quiz_pct")
    ai_eng_pct = max(0.0, min(100.0, float(_get_val(metrics, "ai_engagement_pct", 0.0) or 0.0)))
    duration_sec = max(1, int(_get_val(metrics, "duration_sec", 1800) or 1800))
    completion_pct = max(0.0, min(100.0, float(_get_val(metrics, "completion_pct", 100.0) or 100.0)))
    consistency_pct = max(0.0, min(100.0, float(_get_val(metrics, "consistency_pct", 75.0) or 75.0)))
    absence_count = max(0, int(_get_val(metrics, "absence_count", 0) or 0))
    longest_absence_sec = max(0, int(_get_val(metrics, "longest_absence_sec", 0) or 0))

    # Resolve learning dimensions (with fallbacks if test is pending)
    has_retention = retention_raw is not None
    has_quiz = quiz_raw is not None

    if has_retention:
        retention_pct = max(0.0, min(100.0, float(retention_raw)))
    elif has_quiz:
        retention_pct = max(0.0, min(100.0, float(quiz_raw)))
    else:
        retention_pct = presence_pct  # Interim physical proxy

    if has_quiz:
        quiz_pct = max(0.0, min(100.0, float(quiz_raw)))
    elif has_retention:
        quiz_pct = retention_pct
    else:
        quiz_pct = 70.0  # Baseline neutral proxy

    # Duration rating (100% at 30 mins / 1800s)
    duration_rating_pct = min(100.0, (duration_sec / 1800.0) * 100.0)

    # Distraction Penalty (0 to 15 points deduction)
    if absence_count == 0:
        distraction_penalty = 0.0
    elif absence_count == 1:
        distraction_penalty = 2.0
    elif absence_count == 2:
        distraction_penalty = 4.0
    else:
        distraction_penalty = min(15.0, (absence_count * 2.5) + (longest_absence_sec / 300.0) * 3.0)

    distraction_control_pct = round(max(0.0, 100.0 - (distraction_penalty * 6.66)), 1)

    # -------------------------------------------------------------------------
    # Weighted Scoring Matrix (100 Points Total)
    # -------------------------------------------------------------------------
    w_presence   = presence_pct * 0.25
    w_retention  = retention_pct * 0.25
    w_ai_eng     = ai_eng_pct * 0.15
    w_quiz       = quiz_pct * 0.15
    w_duration   = duration_rating_pct * 0.10
    w_completion = completion_pct * 0.05
    w_consistency= consistency_pct * 0.05

    raw_score = (w_presence + w_retention + w_ai_eng + w_quiz +
                 w_duration + w_completion + w_consistency) - distraction_penalty
    raw_score = max(0.0, min(100.0, raw_score))

    # -------------------------------------------------------------------------
    # Cognitive Contrast & Passive Presence Protection
    # -------------------------------------------------------------------------
    is_passive_alert = False
    # If high physical presence (>= 85%) but verified poor learning (< 50%)
    if presence_pct >= 85.0 and (retention_pct < 50.0 or (has_quiz and quiz_pct < 50.0)):
        final_score = min(55, int(round(raw_score)))
        quality_tier = "Needs Improvement"
        score_label = "Passive Study Alert"
        explanation = (
            "Your desk presence was excellent, but your quiz performance was low. "
            "FocusSense detected that you spent significant time studying without demonstrating enough "
            "knowledge retention. A short recall session has been scheduled."
        )
        is_passive_alert = True
    elif raw_score >= 90.0:
        final_score = int(round(raw_score))
        quality_tier = "Excellent"
        score_label = "Excellent Focus"
        explanation = "Outstanding session: Consistent desk presence translated directly into high AI engagement and retention."
    elif raw_score >= 75.0:
        final_score = int(round(raw_score))
        quality_tier = "Strong"
        score_label = "Strong Focus"
        explanation = "Solid focus: Good presence consistency and learning engagement maintained throughout."
    elif raw_score >= 60.0:
        final_score = int(round(raw_score))
        quality_tier = "Moderate"
        score_label = "Moderate Focus"
        explanation = "Moderate focus: Presence was steady, but some distraction or lower retention was recorded."
    else:
        final_score = int(round(raw_score))
        quality_tier = "Needs Improvement"
        score_label = "Needs Improvement"
        explanation = "Needs improvement: Focus session was disrupted by absences or weak conceptual recall."

    # Learning Efficiency Calculation: (Retention % / max(15 mins, Duration Mins)) * 30
    duration_mins = max(15.0, duration_sec / 60.0)
    learning_efficiency = round(min(100.0, (retention_pct / duration_mins) * 30.0), 1)

    return {
        "unified_focus_score": final_score,
        "final_score": final_score,
        "quality_tier": quality_tier,
        "score_label": score_label,
        "is_passive_alert": is_passive_alert,
        "learning_efficiency": learning_efficiency,
        "explanation": explanation,
        "factors": {
            "physical_presence_pct": round(presence_pct, 1),
            "knowledge_retention_pct": round(retention_pct, 1),
            "ai_engagement_pct": round(ai_eng_pct, 1),
            "quiz_performance_pct": round(quiz_pct, 1),
            "duration_rating_pct": round(duration_rating_pct, 1),
            "session_completion_pct": round(completion_pct, 1),
            "habit_consistency_pct": round(consistency_pct, 1),
            "distraction_control_pct": round(distraction_control_pct, 1),
            "distraction_penalty": round(distraction_penalty, 1),
        },
        "weights": {
            "physical_presence": "25%",
            "knowledge_retention": "25%",
            "ai_learning_engagement": "15%",
            "quiz_performance": "15%",
            "study_duration": "10%",
            "session_completion": "5%",
            "study_habit_consistency": "5%",
        }
    }


def compute_learning_focus_score(presence_pct: float, behavior_pct: float, retention_pct: float = None) -> dict:
    """
    Backward-compatible 2-dimension bridge delegating to the Unified Model.
    """
    metrics = {
        "presence_pct": presence_pct,
        "retention_pct": retention_pct,
        "behavior_pct": behavior_pct,
        "ai_engagement_pct": behavior_pct,
        "quiz_score_pct": retention_pct,
    }
    res = compute_unified_focus_score(metrics)
    p_pct = float(presence_pct or 100.0)
    b_pct = float(behavior_pct or 80.0)
    physical_focus = round((p_pct * 0.55) + (b_pct * 0.45), 1)

    if retention_pct is not None:
        r_val = float(retention_pct)
        if p_pct >= 85 and r_val < 60:
            insight = "⚠️ Passive Presence Detected: Long study session logged with low recall score. Review foundational concepts."
            explanation = "Your physical focus was strong, but some concepts need revision."
        elif p_pct >= 85 and r_val >= 85:
            insight = "⭐ Holistic Deep Mastery: Excellent desk presence backed by high knowledge retention."
            explanation = "Outstanding session: Physical discipline translated directly into verified conceptual mastery."
        else:
            insight = f"Balanced study session with {res['unified_focus_score']}% combined score."
            explanation = "Good physical consistency with developing knowledge retention."
    else:
        insight = "Physical study telemetry logged. Recall test scheduled."
        explanation = "Desk presence recorded. Take the delayed recall test to verify learning."

    return {
        "composite_score": res["unified_focus_score"],
        "final_score": res["unified_focus_score"],
        "physical_focus": physical_focus,
        "learning_focus": res["factors"]["knowledge_retention_pct"] if retention_pct is not None else None,
        "label": res["score_label"],
        "has_retention": retention_pct is not None,
        "presence_pct": res["factors"]["physical_presence_pct"],
        "behavior_pct": round(b_pct, 1),
        "retention_pct": res["factors"]["knowledge_retention_pct"] if retention_pct is not None else None,
        "explanation": explanation,
        "insight": insight,
        "learning_efficiency": res["learning_efficiency"],
        "quality_tier": res["quality_tier"],
    }


# ---------------------------------------------------------------------------
# 7. Adaptive Policy & Reinforcement Learning Foundation
# ---------------------------------------------------------------------------

def select_adaptive_action(state: dict) -> dict:
    """
    Explainable Rule-Based Adaptive Policy pi(State_t) -> Action_t
    Selects targeted pedagogical intervention based on the multi-dimensional learning state:
    1. Remediation & High-Frequency Recall: Weak retention (< 60%) or passive alert.
    2. Marathon Reading / Low Efficiency: Long duration (>= 1hr) with poor retention.
    3. Short-Session High-Efficiency: Short duration (<= 30 min) with high retention (>= 85%).
    4. Advanced Challenge & Spaced Expansion: High retention (>= 85%) or Strong mastery.
    5. Maintain Mastery: Standard balanced session.
    """
    topic = (state.get("topic") or "General Study").strip()
    retention = float(state.get("retention_score_pct") if state.get("retention_score_pct") is not None else 70.0)
    duration_sec = int(state.get("study_duration_sec") or 1800)
    mastery_tier = (state.get("mastery_tier") or "MEDIUM").upper()
    is_passive_alert = bool(state.get("is_passive_alert") or (state.get("score_label") == "Passive Study Alert"))

    # Policy Rule 2: Marathon Reading / Low Efficiency Detection
    if duration_sec >= 3600 and retention < 60.0:
        return {
            "action_type": "RECOMMEND_SHORT_SESSIONS",
            "action_payload": {
                "recommended_block_minutes": 25,
                "delay_minutes": 10,
                "difficulty": "basic",
                "recommendation": "Switch to 25-minute Pomodoro blocks with active concept testing."
            },
            "reason": "Extended desk presence with low retention indicates passive fatigue. 25-minute active blocks recommended."
        }

    # Policy Rule 1: Weak Retention / Remediation Drill
    if retention < 60.0 or mastery_tier == "WEAK" or is_passive_alert:
        return {
            "action_type": "INCREASE_RECALL_FREQ",
            "action_payload": {
                "delay_minutes": 5,
                "difficulty": "basic",
                "recommendation": "Frequent active recall scheduled to consolidate fundamental concepts."
            },
            "reason": "Topic retention is below 60%. High-frequency recall and foundational review scheduled."
        }

    # Policy Rule 3: Short Session, High Cognitive Efficiency
    if duration_sec <= 1800 and retention >= 85.0:
        return {
            "action_type": "REINFORCE_EFFICIENCY",
            "action_payload": {
                "recommended_block_minutes": 30,
                "delay_minutes": 720,
                "difficulty": "conceptual",
                "recommendation": "Maintain high-efficiency 30-minute study sprints."
            },
            "reason": "Demonstrated strong concept retention in a focused short block. High cognitive efficiency reinforced."
        }

    # Policy Rule 4: Consistent High Mastery
    if retention >= 85.0 or mastery_tier == "STRONG":
        return {
            "action_type": "ADVANCED_CHALLENGE",
            "action_payload": {
                "delay_minutes": 1440,
                "difficulty": "advanced",
                "recommendation": "Spaced review in 24-48 hours with architectural trade-offs."
            },
            "reason": "High mastery confirmed. Spaced interval extended with challenging edge-case questions."
        }

    # Default Rule: Maintain Standard Progress
    return {
        "action_type": "MAINTAIN_MASTERY",
        "action_payload": {
            "delay_minutes": 60,
            "difficulty": "conceptual",
            "recommendation": "Standard spaced review."
        },
        "reason": "Balanced study session logged. Normal spaced review scheduled."
    }


def calculate_adaptive_reward(initial_state: dict, next_state: dict) -> float:
    """
    Calculates quantitative scalar reward R_t in [-1.0, +1.0]:
    R_t = 0.6 * ((Retention_t+1 - Retention_t) / 100)
        + 0.3 * ((Score_t+1 - Score_t) / 100)
        + 0.1 * Sign(Efficiency_t+1 - Efficiency_t)
    Handles first session gracefully when initial_state is None.
    """
    if not next_state:
        return 0.0

    ret_next = float(next_state.get("retention_score_pct") if next_state.get("retention_score_pct") is not None else 70.0)
    score_next = float(next_state.get("unified_focus_score") if next_state.get("unified_focus_score") is not None else 70.0)
    eff_next = float(next_state.get("learning_efficiency_score") if next_state.get("learning_efficiency_score") is not None else 50.0)

    if not initial_state:
        # First session baseline reward normalized 0 to +1
        base_r = (0.6 * (ret_next / 100.0)) + (0.4 * (score_next / 100.0))
        return round(max(0.0, min(1.0, base_r)), 3)

    ret_init = float(initial_state.get("retention_score_pct") if initial_state.get("retention_score_pct") is not None else ret_next)
    score_init = float(initial_state.get("unified_focus_score") if initial_state.get("unified_focus_score") is not None else score_next)
    eff_init = float(initial_state.get("learning_efficiency_score") if initial_state.get("learning_efficiency_score") is not None else eff_next)

    delta_ret = (ret_next - ret_init) / 100.0
    delta_score = (score_next - score_init) / 100.0

    eff_diff = eff_next - eff_init
    sign_eff = 1.0 if eff_diff > 0.5 else (-1.0 if eff_diff < -0.5 else 0.0)

    raw_r = (0.6 * delta_ret) + (0.3 * delta_score) + (0.1 * sign_eff)
    return round(max(-1.0, min(1.0, raw_r)), 3)


# ---------------------------------------------------------------------------
# 8. SQLite State-Action-Reward Persistence
# ---------------------------------------------------------------------------

def record_learning_state(user_id: int, session_id: int, topic: str, state_metrics: dict, database_path: str = None) -> int:
    """
    Persists a multi-dimensional state snapshot in learning_states table.
    """
    if not user_id or not topic:
        return 0

    conn = get_db_connection(database_path)
    c = conn.cursor()
    now_iso = datetime.now().isoformat()

    c.execute("""
        INSERT INTO learning_states (
            user_id, session_id, topic, physical_presence_pct, study_duration_sec,
            ai_engagement_pct, quiz_score_pct, retention_score_pct, distraction_control_pct,
            session_completion_pct, consistency_score_pct, unified_focus_score,
            learning_efficiency_score, quality_tier, score_label, explanation, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        user_id,
        session_id,
        topic.strip(),
        float(state_metrics.get("physical_presence_pct") or 100.0),
        int(state_metrics.get("study_duration_sec") or 1800),
        float(state_metrics.get("ai_engagement_pct") or 0.0),
        float(state_metrics.get("quiz_score_pct")) if state_metrics.get("quiz_score_pct") is not None else None,
        float(state_metrics.get("retention_score_pct")) if state_metrics.get("retention_score_pct") is not None else None,
        float(state_metrics.get("distraction_control_pct") or 100.0),
        float(state_metrics.get("session_completion_pct") or 100.0),
        float(state_metrics.get("consistency_score_pct") or 75.0),
        int(state_metrics.get("unified_focus_score") or 75),
        float(state_metrics.get("learning_efficiency_score") or 50.0),
        str(state_metrics.get("quality_tier") or "Strong"),
        str(state_metrics.get("score_label") or "Strong Focus"),
        str(state_metrics.get("explanation") or ""),
        now_iso
    ))
    conn.commit()
    state_id = c.lastrowid
    conn.close()
    return state_id


def record_adaptive_action(user_id: int, topic: str, session_id: int, state_id: int,
                           action_type: str, action_payload: dict, reason: str, database_path: str = None) -> int:
    """
    Persists selected adaptive policy action in adaptive_actions table.
    """
    if not user_id or not topic:
        return 0

    conn = get_db_connection(database_path)
    c = conn.cursor()
    now_iso = datetime.now().isoformat()
    payload_str = json.dumps(action_payload or {})

    c.execute("""
        INSERT INTO adaptive_actions (
            user_id, topic, session_id, state_id, action_type, action_payload, reason, status, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, 'PENDING', ?)
    """, (user_id, topic.strip(), session_id, state_id, action_type, payload_str, reason, now_iso))
    conn.commit()
    action_id = c.lastrowid
    conn.close()
    return action_id


def record_adaptive_reward(user_id: int, action_id: int, initial_state_id: int,
                           resulting_state_id: int, reward_value: float, evaluation_notes: str = None, database_path: str = None) -> int:
    """
    Persists computed reward delta in adaptive_rewards table and marks action as EVALUATED.
    """
    if not user_id or not action_id:
        return 0

    conn = get_db_connection(database_path)
    c = conn.cursor()
    now_iso = datetime.now().isoformat()

    c.execute("""
        INSERT INTO adaptive_rewards (
            user_id, action_id, initial_state_id, resulting_state_id, reward_value, evaluation_notes, recorded_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (user_id, action_id, initial_state_id, resulting_state_id, reward_value, evaluation_notes or "", now_iso))

    c.execute("UPDATE adaptive_actions SET status = 'EVALUATED' WHERE id = ?", (action_id,))
    conn.commit()
    reward_id = c.lastrowid
    conn.close()
    return reward_id


def get_latest_learning_state(user_id: int, topic: str = None, database_path: str = None) -> dict:
    """
    Retrieves the most recent learning state snapshot for a user and optional topic.
    """
    if not user_id:
        return {}

    conn = get_db_connection(database_path)
    c = conn.cursor()

    if topic:
        c.execute("""
            SELECT * FROM learning_states
            WHERE user_id = ? AND topic = ?
            ORDER BY id DESC LIMIT 1
        """, (user_id, topic.strip()))
    else:
        c.execute("""
            SELECT * FROM learning_states
            WHERE user_id = ?
            ORDER BY id DESC LIMIT 1
        """, (user_id,))

    row = c.fetchone()
    conn.close()
    return dict(row) if row else {}


def get_unified_learning_profile(user_id: int, database_path: str = None) -> dict:
    """
    Returns comprehensive student learning intelligence profile with efficiency,
    unified score trends, mastery breakdown, and active adaptive recommendations.
    """
    if not user_id:
        return {}

    conn = get_db_connection(database_path)
    c = conn.cursor()

    # Fetch latest learning state
    c.execute("SELECT * FROM learning_states WHERE user_id = ? ORDER BY id DESC LIMIT 1", (user_id,))
    latest_state_row = c.fetchone()
    latest_state = dict(latest_state_row) if latest_state_row else {}

    # Fetch latest adaptive action
    c.execute("SELECT * FROM adaptive_actions WHERE user_id = ? ORDER BY id DESC LIMIT 1", (user_id,))
    latest_action_row = c.fetchone()
    latest_action = dict(latest_action_row) if latest_action_row else {}

    # Calculate average learning efficiency across last 10 sessions
    c.execute("""
        SELECT AVG(learning_efficiency_score) as avg_eff,
               AVG(unified_focus_score) as avg_unified,
               AVG(retention_score_pct) as avg_ret
        FROM (
            SELECT learning_efficiency_score, unified_focus_score, retention_score_pct
            FROM learning_states
            WHERE user_id = ?
            ORDER BY id DESC LIMIT 10
        )
    """, (user_id,))
    stats_row = c.fetchone()
    conn.close()

    avg_eff = round(float(stats_row["avg_eff"] or 75.0), 1) if stats_row and stats_row["avg_eff"] is not None else 75.0
    avg_unified = int(round(float(stats_row["avg_unified"] or 78.0))) if stats_row and stats_row["avg_unified"] is not None else 78
    avg_ret = round(float(stats_row["avg_ret"] or 75.0), 1) if stats_row and stats_row["avg_ret"] is not None else 75.0

    action_payload = {}
    if latest_action.get("action_payload"):
        try:
            action_payload = json.loads(latest_action["action_payload"])
        except Exception:
            action_payload = {}

    recommendation_text = action_payload.get("recommendation") or latest_action.get("reason") or "Maintain consistent active study habits."

    return {
        "user_id": user_id,
        "latest_unified_score": latest_state.get("unified_focus_score", avg_unified),
        "latest_score_label": latest_state.get("score_label", "Strong Focus"),
        "latest_quality_tier": latest_state.get("quality_tier", "Strong"),
        "learning_efficiency": latest_state.get("learning_efficiency_score", avg_eff),
        "avg_learning_efficiency": avg_eff,
        "avg_retention_score": avg_ret,
        "active_recommendation": recommendation_text,
        "action_type": latest_action.get("action_type", "MAINTAIN_MASTERY"),
        "why_explanation": latest_state.get("explanation") or "Steady desk presence with consistent knowledge retention.",
        "latest_state": latest_state,
    }
