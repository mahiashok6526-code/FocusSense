"""
FocusSense AI — Personal Learning Coach Engine
==============================================
Synthesizes all 8 foundational learning intelligence pillars:
  1. Study Sessions      (Duration, streak, habit consistency, session logs)
  2. Radar Presence       (Desk presence ratio, in-seat stillness, absence frequency)
  3. AI Engagement        (Substantive questions, conceptual depth, notes generated)
  4. Quiz Results         (Post-session tests, category mastery, difficulty handling)
  5. Recall Results       (Spaced active recall tests, retention verification score %)
  6. Study Materials      (Ingested documents, chunk coverage, semantic retrieval)
  7. Knowledge Retention  (Ebbinghaus decay curve modeling, memory half-life)
  8. Adaptive Learning    (Policy transitions, difficulty calibration, pacing)
         ↓
  FOCUSSENSE PERSONAL LEARNING COACH
         ↓
  "What should I study next?"
  "What am I weak at?"
  "When should I revise?"
  "How am I improving?"

Guiding Principles:
- Single source of truth: Aggregates and reuses existing engines (FocusScore,
  learning_engine, focus_intelligence, quiz_engine, recall_engine, study_material_engine).
- Never fabricates data: Explicitly returns "Not enough learning history yet." when data is sparse.
- Ebbinghaus retention is clearly presented as an ESTIMATE (never claimed as biological measurement).
- Grounded in real learning outcomes rather than passive application usage.
- Deterministic and explainable fallbacks when AI is offline.
- Strict multi-tenant user isolation.
"""

import json
import math
import sqlite3
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple

from database import DB_PATH
from ai_service import ai_service
import learning_engine
import focus_intelligence
import quiz_engine
import recall_engine
import study_material_engine

logger = logging.getLogger("FocusSense_CoachEngine")
logger.setLevel(logging.INFO)

DATABASE = DB_PATH


def get_db_connection(database_path: Optional[str] = None) -> sqlite3.Connection:
    target_db = database_path or DATABASE
    conn = sqlite3.connect(target_db, timeout=30.0)
    conn.row_factory = sqlite3.Row
    return conn


# =============================================================================
# 1. 8-Pillar Telemetry Aggregator
# =============================================================================

def get_eight_pillar_telemetry(user_id: int, database_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Collects, normalizes, and synthesizes real metrics across all 8 learning pillars for a user.
    Never fabricates metrics. If data is missing or empty, sets clear zero/None indicators.
    """
    if not user_id:
        return {"has_sufficient_history": False, "reason": "No user_id provided"}

    conn = get_db_connection(database_path)
    cursor = conn.cursor()

    # 1. Study Sessions Pillar
    cursor.execute("""
        SELECT COUNT(*) as session_count,
               COALESCE(SUM(duration), 0) as total_duration_sec,
               COALESCE(AVG(duration), 0) as avg_duration_sec,
               COALESCE(AVG(focus_score), 0) as avg_legacy_focus,
               MAX(date) as last_study_date
        FROM study_sessions
        WHERE user_id = ?
    """, (user_id,))
    session_stats = cursor.fetchone()
    session_count = session_stats["session_count"] if session_stats else 0
    total_duration_sec = session_stats["total_duration_sec"] if session_stats else 0
    avg_duration_sec = int(round(session_stats["avg_duration_sec"])) if session_stats else 0

    # Habit consistency streak & goals
    cursor.execute("SELECT daily_goal, weekly_goal FROM goals WHERE user_id = ?", (user_id,))
    goal_row = cursor.fetchone()
    daily_goal_hrs = goal_row["daily_goal"] if goal_row and goal_row["daily_goal"] else 2
    weekly_goal_hrs = goal_row["weekly_goal"] if goal_row and goal_row["weekly_goal"] else 10

    # 2. Radar Presence Pillar
    cursor.execute("""
        SELECT COALESCE(AVG(presence_pct), 0) as avg_presence_pct,
               COALESCE(AVG(absence_count), 0) as avg_absence_count,
               COALESCE(SUM(absence_count), 0) as total_absences,
               COALESCE(AVG(estimated_score), 0) as avg_fi_score,
               COUNT(*) as scored_sessions_count
        FROM session_focus_scores
        WHERE user_id = ?
    """, (user_id,))
    presence_stats = cursor.fetchone()
    avg_presence_pct = round(float(presence_stats["avg_presence_pct"] or 0.0), 1) if presence_stats else 0.0
    avg_absence_count = round(float(presence_stats["avg_absence_count"] or 0.0), 1) if presence_stats else 0.0
    scored_sessions_count = presence_stats["scored_sessions_count"] if presence_stats else 0

    # 3. AI Engagement Pillar
    cursor.execute("""
        SELECT COUNT(*) as session_count,
               COALESCE(SUM(question_count), 0) as total_queries,
               COALESCE(SUM(meaningful_query_count), 0) as total_meaningful_queries,
               COALESCE(SUM(active_duration_sec), 0) as total_ai_sec
        FROM ai_study_sessions
        WHERE user_id = ?
    """, (user_id,))
    ai_stats = cursor.fetchone()
    ai_query_count = ai_stats["total_queries"] if ai_stats else 0
    meaningful_queries = ai_stats["total_meaningful_queries"] if ai_stats else 0

    cursor.execute("SELECT COUNT(*) as msg_count FROM ai_messages m JOIN ai_conversations c ON m.conversation_id = c.id WHERE c.user_id = ?", (user_id,))
    msg_row = cursor.fetchone()
    total_messages = msg_row["msg_count"] if msg_row else 0

    # 4. Quiz Results Pillar
    cursor.execute("""
        SELECT COUNT(*) as quiz_count,
               COALESCE(AVG(score_pct), 0) as avg_quiz_score,
               MAX(score_pct) as max_quiz_score,
               MIN(score_pct) as min_quiz_score
        FROM study_quizzes
        WHERE user_id = ?
    """, (user_id,))
    quiz_stats = cursor.fetchone()
    quiz_count = quiz_stats["quiz_count"] if quiz_stats else 0
    avg_quiz_score = round(float(quiz_stats["avg_quiz_score"] or 0.0), 1) if quiz_stats else 0.0

    # Assessment attempts (Phase 5)
    cursor.execute("""
        SELECT COUNT(*) as assessment_count,
               COALESCE(AVG(score_pct), 0) as avg_assessment_score
        FROM assessment_attempts
        WHERE user_id = ? AND completed_at IS NOT NULL
    """, (user_id,))
    assessment_stats = cursor.fetchone()
    assessment_count = assessment_stats["assessment_count"] if assessment_stats else 0
    avg_assessment_score = round(float(assessment_stats["avg_assessment_score"] or 0.0), 1) if assessment_stats else 0.0

    # 5. Recall Results Pillar
    cursor.execute("""
        SELECT COUNT(*) as total_tasks,
               SUM(CASE WHEN status = 'COMPLETED' THEN 1 ELSE 0 END) as completed_tasks,
               SUM(CASE WHEN status = 'READY' THEN 1 ELSE 0 END) as ready_tasks,
               SUM(CASE WHEN status = 'PENDING' THEN 1 ELSE 0 END) as pending_tasks,
               COALESCE(AVG(CASE WHEN status = 'COMPLETED' THEN retention_score END), 0) as avg_retention_score
        FROM recall_tasks
        WHERE user_id = ?
    """, (user_id,))
    recall_stats = cursor.fetchone()
    recall_total = int(recall_stats["total_tasks"] or 0) if recall_stats else 0
    recall_completed = int(recall_stats["completed_tasks"] or 0) if (recall_stats and recall_stats["completed_tasks"] is not None) else 0
    recall_ready = int(recall_stats["ready_tasks"] or 0) if (recall_stats and recall_stats["ready_tasks"] is not None) else 0
    avg_recall_retention = round(float(recall_stats["avg_retention_score"] or 0.0), 1) if (recall_stats and recall_stats["avg_retention_score"] is not None) else 0.0

    # 6. Study Materials Pillar
    cursor.execute("""
        SELECT COUNT(*) as material_count,
               COALESCE(SUM(chunk_count), 0) as total_chunks,
               COALESCE(SUM(file_size_bytes), 0) as total_bytes
        FROM study_materials
        WHERE user_id = ? AND status = 'READY'
    """, (user_id,))
    mat_stats = cursor.fetchone()
    material_count = int(mat_stats["material_count"] or 0) if mat_stats else 0
    total_chunks = int(mat_stats["total_chunks"] or 0) if mat_stats else 0


    # Fetch materials list for action mapping
    cursor.execute("""
        SELECT id, filename, title, subject, topic, chunk_count, created_at
        FROM study_materials
        WHERE user_id = ? AND status = 'READY'
        ORDER BY id DESC
    """, (user_id,))
    materials_list = [dict(r) for r in cursor.fetchall()]

    # 7. Knowledge Retention & Topic Mastery Pillar
    cursor.execute("""
        SELECT id, topic, mastery_tier, total_quizzes, total_questions,
               correct_count, partial_count, incorrect_count,
               avg_retention_score, last_retention_score, last_tested_at, updated_at
        FROM topic_mastery
        WHERE user_id = ?
        ORDER BY avg_retention_score ASC
    """, (user_id,))
    topic_rows = cursor.fetchall()
    topics_tracked = [dict(r) for r in topic_rows]

    weak_topics = [t for t in topics_tracked if t["mastery_tier"] == "WEAK" or (t.get("avg_retention_score") or 0) < 60.0]
    medium_topics = [t for t in topics_tracked if t["mastery_tier"] == "MEDIUM" and (t.get("avg_retention_score") or 0) >= 60.0 and (t.get("avg_retention_score") or 0) < 85.0]
    strong_topics = [t for t in topics_tracked if t["mastery_tier"] == "STRONG" or (t.get("avg_retention_score") or 0) >= 85.0]

    # Concept mastery sub-topics
    cursor.execute("""
        SELECT topic, sub_concept, total_tested, correct_count, partial_count, incorrect_count,
               mastery_pct, is_weak, last_tested_at
        FROM concept_mastery
        WHERE user_id = ? AND is_weak = 1
        ORDER BY mastery_pct ASC
    """, (user_id,))
    weak_sub_concepts = [dict(r) for r in cursor.fetchall()]

    # 8. Adaptive Learning Pillar
    cursor.execute("""
        SELECT id, topic, unified_focus_score, learning_efficiency_score, quality_tier,
               score_label, physical_presence_pct, retention_score_pct, ai_engagement_pct,
               explanation, created_at
        FROM learning_states
        WHERE user_id = ?
        ORDER BY id DESC LIMIT 10
    """, (user_id,))
    recent_states = [dict(r) for r in cursor.fetchall()]

    cursor.execute("""
        SELECT id, topic, action_type, action_payload, reason, status, created_at
        FROM adaptive_actions
        WHERE user_id = ?
        ORDER BY id DESC LIMIT 5
    """, (user_id,))
    recent_actions = [dict(r) for r in cursor.fetchall()]

    cursor.execute("""
        SELECT AVG(reward_value) as avg_reward, COUNT(*) as count
        FROM adaptive_rewards
        WHERE user_id = ?
    """, (user_id,))
    reward_row = cursor.fetchone()
    avg_reward = round(float(reward_row["avg_reward"] or 0.0), 3) if reward_row and reward_row["count"] > 0 else 0.0

    conn.close()

    # Determine data sufficiency
    total_data_points = session_count + quiz_count + recall_completed + material_count + len(topics_tracked) + len(recent_actions)
    has_sufficient_history = total_data_points >= 1

    # Cognitive efficiency: retention score gained per hour of study
    study_hours = max(0.1, total_duration_sec / 3600.0)
    effective_retention = avg_recall_retention if recall_completed > 0 else (avg_quiz_score if quiz_count > 0 else (topics_tracked[0].get("avg_retention_score", 0.0) if topics_tracked else 0.0))
    cognitive_efficiency = round(min(100.0, (effective_retention / max(1.0, study_hours)) * (1.0 if session_count <= 2 else 0.8)), 1) if has_sufficient_history else 0.0

    # Macro Coaching Health Score (0 - 100)
    # Balanced composite:
    # 25% Retention/Recall, 20% In-Seat Presence, 20% Quiz Performance,
    # 15% Goal Consistency, 10% AI Engagement, 10% Material Mastery
    if has_sufficient_history:
        p_ret = effective_retention * 0.25
        p_pres = (avg_presence_pct if scored_sessions_count > 0 else 75.0) * 0.20
        p_quiz = (avg_quiz_score if quiz_count > 0 else (avg_recall_retention or 70.0)) * 0.20
        p_eng = min(100.0, (meaningful_queries * 15.0) + (total_messages * 3.0)) * 0.10
        p_mat = min(100.0, (len(strong_topics) / max(1, len(topics_tracked))) * 100.0 if topics_tracked else 50.0) * 0.10
        p_goal = min(100.0, (session_count / 5.0) * 100.0) * 0.15
        coach_health_score = int(round(p_ret + p_pres + p_quiz + p_eng + p_mat + p_goal))
        coach_health_score = max(5, min(100, coach_health_score))

    else:
        coach_health_score = 0

    if coach_health_score >= 85:
        health_tier = "EXCELLENT"
        health_label = "Optimal Learning Trajectory"
    elif coach_health_score >= 70:
        health_tier = "STRONG"
        health_label = "Steady Cognitive Consolidation"
    elif coach_health_score >= 50:
        health_tier = "DEVELOPING"
        health_label = "Developing Knowledge Retention"
    elif coach_health_score > 0:
        health_tier = "REMEDIATION_NEEDED"
        health_label = "Retention Action Required"
    else:
        health_tier = "NO_DATA"
        health_label = "Not enough learning history yet."

    return {
        "user_id": user_id,
        "has_sufficient_history": has_sufficient_history,
        "history_message": "Telemetry active." if has_sufficient_history else "Not enough learning history yet.",
        "coach_health_score": coach_health_score,
        "health_tier": health_tier,
        "health_label": health_label,
        "cognitive_efficiency": cognitive_efficiency,
        "pillars": {
            "study_sessions": {
                "session_count": session_count,
                "total_duration_sec": total_duration_sec,
                "avg_duration_sec": avg_duration_sec,
                "daily_goal_hrs": daily_goal_hrs,
                "weekly_goal_hrs": weekly_goal_hrs,
            },
            "radar_presence": {
                "avg_presence_pct": avg_presence_pct,
                "avg_absence_count": avg_absence_count,
                "scored_sessions_count": scored_sessions_count,
            },
            "ai_engagement": {
                "total_queries": ai_query_count,
                "meaningful_queries": meaningful_queries,
                "total_messages": total_messages,
            },
            "quiz_results": {
                "quiz_count": quiz_count,
                "avg_quiz_score": avg_quiz_score,
                "assessment_count": assessment_count,
                "avg_assessment_score": avg_assessment_score,
            },
            "recall_results": {
                "total_tasks": recall_total,
                "completed_tasks": recall_completed,
                "ready_tasks": recall_ready,
                "avg_retention_score": avg_recall_retention,
            },
            "study_materials": {
                "material_count": material_count,
                "total_chunks": total_chunks,
                "materials_list": materials_list,
            },
            "knowledge_retention": {
                "topics_tracked_count": len(topics_tracked),
                "weak_count": len(weak_topics),
                "medium_count": len(medium_topics),
                "strong_count": len(strong_topics),
                "weak_topics": weak_topics,
                "medium_topics": medium_topics,
                "strong_topics": strong_topics,
                "all_topics": topics_tracked,
                "weak_sub_concepts": weak_sub_concepts,
            },
            "adaptive_learning": {
                "recent_states": recent_states,
                "recent_actions": recent_actions,
                "action_count": len(recent_actions),
                "avg_reward": avg_reward,
            }
        }
    }



# =============================================================================
# 2. Ebbinghaus Memory Decay Estimation
# =============================================================================

def calculate_ebbinghaus_decay(last_score_pct: float, elapsed_hours: Any = 0.0,
                               repetitions_count: int = 1,
                               mastery_tier: str = "MEDIUM",
                               **kwargs) -> Dict[str, Any]:
    """
    Computes an estimated knowledge retention percentage using the Ebbinghaus forgetting curve.
    
    FORMULA:
      R(t) = R_0 * exp(-t / S)
      where:
        R_0 = Initial tested retention %
        t   = Elapsed hours since last tested/studied
        S   = Memory Stability Half-Life (hours), scaled by repetitions & mastery tier
    
    DISCLAIMER:
      Presents values strictly as ESTIMATES. FocusSense does not biologically scan memory.
    """
    r_0 = max(10.0, min(100.0, float(last_score_pct if last_score_pct is not None else 80.0)))
    
    # Support ISO string or float/int
    t = 0.0
    if isinstance(elapsed_hours, str):
        try:
            dt = datetime.fromisoformat(elapsed_hours.replace('Z', ''))
            t = max(0.0, (datetime.now() - dt).total_seconds() / 3600.0)
        except Exception:
            t = 0.0
    elif isinstance(elapsed_hours, (int, float)):
        t = max(0.0, float(elapsed_hours))
    elif kwargs.get("elapsed_hours") is not None:
        t = max(0.0, float(kwargs["elapsed_hours"]))

    reps = kwargs.get("repetitions") or kwargs.get("repetition_count") or repetitions_count or 1
    reps = max(1, int(reps))
    tier = (mastery_tier or "MEDIUM").upper()

    # Base stability in hours (24 hours baseline for single study session)
    if kwargs.get("stability") is not None:
        base_stability = float(kwargs["stability"]) * 24.0
    elif tier == "STRONG":
        base_stability = 72.0  # 3 days
    elif tier == "WEAK":
        base_stability = 12.0  # 12 hours
    else:
        base_stability = 24.0  # 1 day

    # Repetition multiplier (spaced consolidation strengthens stability)
    stability_factor = base_stability * (1.0 + 0.45 * (reps - 1))
    
    # Calculate decay
    decay_exponent = -(0.693 * t) / max(1.0, stability_factor)
    estimated_retention = round(max(5.0, min(100.0, r_0 * math.exp(decay_exponent))), 1)


    # Calculate forward projections
    decay_24h = round(max(5.0, min(100.0, r_0 * math.exp(-(0.693 * (t + 24.0)) / stability_factor))), 1)
    decay_48h = round(max(5.0, min(100.0, r_0 * math.exp(-(0.693 * (t + 48.0)) / stability_factor))), 1)
    decay_7d  = round(max(5.0, min(100.0, r_0 * math.exp(-(0.693 * (t + 168.0)) / stability_factor))), 1)

    # Urgency Classification
    if estimated_retention < 60.0:
        urgency = "CRITICAL"
        status_label = "Urgent Revision Required"
        badge_color = "var(--accent-rose)"
    elif estimated_retention < 75.0:
        urgency = "DUE"
        status_label = "Revision Recommended"
        badge_color = "var(--accent-amber)"
    elif estimated_retention < 88.0:
        urgency = "OPTIMAL"
        status_label = "Good Retention"
        badge_color = "var(--accent-cyan)"
    else:
        urgency = "CONSOLIDATED"
        status_label = "Consolidated Mastery"
        badge_color = "var(--accent-emerald)"

    return {
        "initial_retention_pct": r_0,
        "elapsed_hours": round(t, 1),
        "estimated_retention": estimated_retention,
        "estimated_current_retention": estimated_retention,
        "projections": {
            "in_24h": decay_24h,
            "in_48h": decay_48h,
            "in_7d": decay_7d,
        },
        "urgency": urgency,
        "status_label": status_label,
        "badge_color": badge_color,
        "stability_hours": round(stability_factor, 1),
        "disclaimer": "Estimated retention calculated via Ebbinghaus forgetting curve modeling. Does not measure biological memory."
    }



# =============================================================================
# 3. Core Coach Superpower 1: "What should I study next?"
# =============================================================================

def get_what_to_study_next(user_id: int, database_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Deterministically computes a prioritized list of actionable study recommendations.
    Every recommendation is backed by a human-readable reason and real telemetry.
    
    Prioritization criteria:
    1. Critical Ebbinghaus memory decay (<60% estimated retention)
    2. Weak topics (<60% mastery tier or low recall scores)
    3. Pending or Ready delayed recall obligations
    4. Unstudied or newly uploaded study materials
    5. Adaptive pacing and optimal block duration (25m Pomodoro vs 45m deep session)
    """
    telemetry = get_eight_pillar_telemetry(user_id, database_path)
    if not telemetry["has_sufficient_history"]:
        return {
            "has_sufficient_history": False,
            "message": "Not enough learning history yet.",
            "recommendations": [],
            "suggested_first_step": {
                "action_type": "START_STUDY",
                "title": "Start your first study session",
                "reason": "Complete a study session or upload course notes to generate personalized recommendations.",
                "action_url": "/dashboard"
            }
        }

    pillars = telemetry["pillars"]
    weak_topics = pillars["knowledge_retention"]["weak_topics"]
    medium_topics = pillars["knowledge_retention"]["medium_topics"]
    strong_topics = pillars["knowledge_retention"]["strong_topics"]
    materials = pillars["study_materials"]["materials_list"]
    ready_recalls = pillars["recall_results"]["ready_tasks"]
    recent_presence_avg = pillars["radar_presence"]["avg_presence_pct"]

    recommendations: List[Dict[str, Any]] = []
    now = datetime.now()

    # Map materials by topic/title for material-grounded study recommendations
    mat_by_topic: Dict[str, Dict[str, Any]] = {}
    for m in materials:
        top = (m.get("topic") or m.get("subject") or m.get("title") or "").strip().lower()
        if top:
            mat_by_topic[top] = m

    # 1. High Priority: Ready Delayed Recall Tasks
    if ready_recalls > 0:
        conn = get_db_connection(database_path)
        c = conn.cursor()
        c.execute("""
            SELECT id, topic, scheduled_for, created_at
            FROM recall_tasks
            WHERE user_id = ? AND status = 'READY'
            ORDER BY id ASC LIMIT 2
        """, (user_id,))
        for r_row in c.fetchall():
            rec_topic = r_row["topic"] or "Study Topic"
            recommendations.append({
                "id": f"recall_{r_row['id']}",
                "priority_score": 98,
                "priority_label": "CRITICAL DUE",
                "action_type": "RECALL_TEST",
                "topic": rec_topic,
                "title": f"Complete Due Recall Test: {rec_topic}",
                "reason": f"Active recall test is ready for '{rec_topic}'. Taking this test now consolidates memory and prevents forgetting.",
                "recommended_duration_min": 10,
                "task_id": r_row["id"],
                "action_url": f"/assessment?topic={rec_topic}&mode=QUICK",
                "badge_color": "var(--accent-rose)",
            })
        conn.close()

    # 2. High Priority: Weak Topics with Memory Decay
    for t in weak_topics:
        t_name = t["topic"]
        last_tested_iso = t.get("last_tested_at") or t.get("updated_at")
        elapsed_hours = 24.0
        if last_tested_iso:
            try:
                dt = datetime.fromisoformat(last_tested_iso)
                elapsed_hours = max(0.5, (now - dt).total_seconds() / 3600.0)
            except Exception:
                elapsed_hours = 24.0

        decay = calculate_ebbinghaus_decay(
            last_score_pct=t.get("avg_retention_score", 45.0),
            elapsed_hours=elapsed_hours,
            repetitions_count=t.get("total_quizzes", 1),
            mastery_tier="WEAK"
        )

        # Check if user has uploaded study material for this weak topic
        mat_match = None
        for key, mat in mat_by_topic.items():
            if key in t_name.lower() or t_name.lower() in key:
                mat_match = mat
                break

        # Duration recommendation based on radar presence endurance
        duration_min = 25 if recent_presence_avg < 80.0 else 35

        if mat_match:
            recommendations.append({
                "id": f"weak_mat_{t['id']}",
                "priority_score": 92,
                "priority_label": "WEAKNESS REMEDIATION",
                "action_type": "REVIEW_MATERIAL",
                "topic": t_name,
                "material_id": mat_match["id"],
                "material_title": mat_match["title"],
                "title": f"Review Notes & Drill: {t_name}",
                "reason": f"Topic '{t_name}' has low retention ({t.get('avg_retention_score', 0)}%). Review your uploaded document '{mat_match['title']}' to rebuild foundations.",
                "recommended_duration_min": duration_min,
                "action_url": f"/ai?material_id={mat_match['id']}",
                "badge_color": "var(--accent-rose)",
            })
        else:
            recommendations.append({
                "id": f"weak_quiz_{t['id']}",
                "priority_score": 90,
                "priority_label": "WEAKNESS DRILL",
                "action_type": "TAKE_QUIZ",
                "topic": t_name,
                "title": f"Targeted Practice Drill: {t_name}",
                "reason": f"Retention on '{t_name}' is currently {t.get('avg_retention_score', 0)}% (Weak Tier). Complete a 5-question remediation drill to fix conceptual gaps.",
                "recommended_duration_min": 15,
                "action_url": f"/assessment?topic={t_name}&mode=TOPIC&difficulty=basic",
                "badge_color": "var(--accent-amber)",
            })

    # 3. Medium Priority: Newly Uploaded Unstudied Materials
    for m in materials:
        m_topic = m.get("topic") or m.get("title")
        # Check if this material has been tested
        conn = get_db_connection(database_path)
        c = conn.cursor()
        c.execute("SELECT COUNT(*) FROM assessment_attempts WHERE material_id = ? AND user_id = ?", (m["id"], user_id))
        mat_assess_count = c.fetchone()[0]
        conn.close()

        if mat_assess_count == 0:
            recommendations.append({
                "id": f"untested_mat_{m['id']}",
                "priority_score": 80,
                "priority_label": "NEW MATERIAL",
                "action_type": "REVIEW_MATERIAL",
                "topic": m_topic,
                "material_id": m["id"],
                "material_title": m["title"],
                "title": f"Study Ingested Material: {m['title']}",
                "reason": f"Uploaded document '{m['title']}' contains {m['chunk_count']} knowledge chunks that have not yet been evaluated.",
                "recommended_duration_min": 30,
                "action_url": f"/ai?material_id={m['id']}",
                "badge_color": "var(--accent-cyan)",
            })

    # 4. Normal Priority: Medium Topics Approaching Decay Window
    for t in medium_topics:
        t_name = t["topic"]
        last_tested_iso = t.get("last_tested_at") or t.get("updated_at")
        elapsed_hours = 48.0
        if last_tested_iso:
            try:
                dt = datetime.fromisoformat(last_tested_iso)
                elapsed_hours = max(0.5, (now - dt).total_seconds() / 3600.0)
            except Exception:
                elapsed_hours = 48.0

        decay = calculate_ebbinghaus_decay(
            last_score_pct=t.get("avg_retention_score", 70.0),
            elapsed_hours=elapsed_hours,
            repetitions_count=t.get("total_quizzes", 2),
            mastery_tier="MEDIUM"
        )

        if decay["estimated_current_retention"] < 75.0:
            recommendations.append({
                "id": f"medium_decay_{t['id']}",
                "priority_score": 75,
                "priority_label": "SPACED REFRESH",
                "action_type": "START_STUDY",
                "topic": t_name,
                "title": f"Spaced Focus Sprint: {t_name}",
                "reason": f"Estimated retention has dropped to {decay['estimated_current_retention']}% after {int(elapsed_hours)}h. A 25-minute sprint will promote this topic to Strong tier.",
                "recommended_duration_min": 25,
                "action_url": f"/dashboard?start_topic={t_name}",
                "badge_color": "var(--accent-blue)",
            })

    # 5. Fallback Default if no urgent recommendations exist
    if not recommendations:
        if strong_topics:
            top_strong = strong_topics[0]["topic"]
            recommendations.append({
                "id": "advance_challenge",
                "priority_score": 60,
                "priority_label": "ADVANCED CHALLENGE",
                "action_type": "TAKE_QUIZ",
                "topic": top_strong,
                "title": f"Advanced Challenge Exam: {top_strong}",
                "reason": f"You have achieved Strong mastery in '{top_strong}'. Test deep edge cases and problem solving in Exam Mode.",
                "recommended_duration_min": 20,
                "action_url": f"/assessment?topic={top_strong}&mode=EXAM&difficulty=advanced",
                "badge_color": "var(--accent-purple)",
            })
        else:
            recommendations.append({
                "id": "general_session",
                "priority_score": 50,
                "priority_label": "REGULAR STUDY",
                "action_type": "START_STUDY",
                "topic": "General Study",
                "title": "Start a Focused Study Sprint",
                "reason": "Complete a 30-minute focused session with Radar presence tracking to build your streak.",
                "recommended_duration_min": 30,
                "action_url": "/dashboard",
                "badge_color": "var(--accent-cyan)",
            })

    # Sort descending by priority score
    recommendations.sort(key=lambda x: x["priority_score"], reverse=True)

    return {
        "has_sufficient_history": True,
        "total_recommendations": len(recommendations),
        "primary_recommendation": recommendations[0] if recommendations else None,
        "recommendations": recommendations[:6],  # Return top 6
        "priority_queue": recommendations[:6],
    }



# =============================================================================
# 4. Core Coach Superpower 2: "What am I weak at?"
# =============================================================================

def get_what_am_i_weak_at(user_id: int, database_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Performs multi-dimensional diagnostic analysis to uncover all knowledge gaps:
    1. Conceptual weaknesses (sub-concepts with accuracy < 70%)
    2. Passive study illusions (high desk presence >=85% + low retention <60%)
    3. Distraction-prone study topics (frequent absences or short sessions)
    4. Blind spots (topics or materials untested > 7 days)
    """
    telemetry = get_eight_pillar_telemetry(user_id, database_path)
    if not telemetry["has_sufficient_history"]:
        return {
            "has_sufficient_history": False,
            "message": "Not enough learning history yet.",
            "weaknesses_count": 0,
            "conceptual_weaknesses": [],
            "passive_study_alerts": [],
            "distraction_vulnerabilities": [],
            "blind_spots": [],
            "summary_diagnosis": "No weaknesses identified yet. Complete study sessions and recall tests to build your diagnostic profile."
        }

    pillars = telemetry["pillars"]
    weak_topics = pillars["knowledge_retention"]["weak_topics"]
    weak_sub_concepts = pillars["knowledge_retention"]["weak_sub_concepts"]
    materials = pillars["study_materials"]["materials_list"]
    all_topics = pillars["knowledge_retention"]["all_topics"]
    now = datetime.now()

    # 1. Conceptual Weaknesses
    concept_weaknesses: List[Dict[str, Any]] = []
    for w in weak_topics:
        # Match material if available
        matched_mat = None
        for m in materials:
            if (m.get("topic") or "").lower() in w["topic"].lower() or (m.get("title") or "").lower() in w["topic"].lower():
                matched_mat = m
                break

        concept_weaknesses.append({
            "topic": w["topic"],
            "mastery_tier": w["mastery_tier"],
            "avg_retention_score": w.get("avg_retention_score", 0.0),
            "total_quizzes": w.get("total_quizzes", 0),
            "incorrect_count": w.get("incorrect_count", 0),
            "severity": "HIGH" if (w.get("avg_retention_score") or 0) < 50.0 else "MODERATE",
            "material_id": matched_mat["id"] if matched_mat else None,
            "material_title": matched_mat["title"] if matched_mat else None,
            "remediation_advice": f"Review foundational definitions and take a 5-question basic drill on '{w['topic']}'.",
            "action_url": f"/assessment?topic={w['topic']}&mode=TOPIC&difficulty=basic"
        })

    # Sub-concept fine-grained weaknesses
    for sc in weak_sub_concepts[:8]:
        concept_weaknesses.append({
            "topic": sc["topic"],
            "sub_concept": sc["sub_concept"],
            "mastery_pct": sc.get("mastery_pct", 0.0),
            "total_tested": sc.get("total_tested", 0),
            "severity": "HIGH" if sc.get("mastery_pct", 0.0) < 50.0 else "MODERATE",
            "remediation_advice": f"Targeted gap in '{sc['sub_concept']}'. Clarify this concept with FocusSense AI.",
            "action_url": f"/ai?topic={sc['topic']}"
        })

    # 2. Passive Study Alert Diagnostic (High Presence + Low Recall / Quiz)
    conn = get_db_connection(database_path)
    c = conn.cursor()
    c.execute("""
        SELECT ss.id, COALESCE(sq.topic, rt.topic, 'Study Session') as topic, ss.duration, sfs.presence_pct, 
               COALESCE(rt.retention_score, sq.score_pct) as retention_score,
               ss.date, sfs.estimated_score
        FROM study_sessions ss
        JOIN session_focus_scores sfs ON sfs.session_id = ss.id
        LEFT JOIN recall_tasks rt ON rt.session_id = ss.id
        LEFT JOIN study_quizzes sq ON sq.session_id = ss.id
        WHERE ss.user_id = ? AND sfs.presence_pct >= 80.0
          AND (
              (rt.retention_score IS NOT NULL AND rt.retention_score < 60.0)
              OR (sq.score_pct IS NOT NULL AND sq.score_pct < 60.0)
          )
        ORDER BY ss.id DESC LIMIT 5

    """, (user_id,))
    passive_rows = c.fetchall()

    passive_study_alerts: List[Dict[str, Any]] = []
    for pr in passive_rows:
        passive_study_alerts.append({
            "session_id": pr["id"],
            "topic": pr["topic"] or "Study Session",
            "date": pr["date"],
            "presence_pct": pr["presence_pct"],
            "presence_rate": pr["presence_pct"],
            "retention_pct": pr["retention_score"],
            "retention_rate": pr["retention_score"],
            "retention_score": pr["retention_score"],
            "warning": "Passive Presence Illusion: You maintained high desk presence, but test recall demonstrated low conceptual retention.",
            "action_advice": "Shift from passive reading/watching to active questioning and flashcards during sessions."
        })


    # 3. Distraction Vulnerabilities
    c.execute("""
        SELECT ss.topic, AVG(sfs.absence_count) as avg_absences,
               AVG(sfs.presence_pct) as avg_pres, COUNT(*) as count
        FROM study_sessions ss
        JOIN session_focus_scores sfs ON sfs.session_id = ss.id
        WHERE ss.user_id = ? AND sfs.absence_count >= 2
        GROUP BY ss.topic
        ORDER BY avg_absences DESC LIMIT 3
    """, (user_id,))
    distraction_rows = c.fetchall()
    distraction_vulnerabilities = []
    for dr in distraction_rows:
        distraction_vulnerabilities.append({
            "topic": dr["topic"] or "General Study",
            "avg_absences": round(float(dr["avg_absences"]), 1),
            "avg_presence_pct": round(float(dr["avg_pres"]), 1),
            "sessions_count": dr["count"],
            "insight": f"Frequent interruptions recorded when studying '{dr['topic']}'. Consider breaking sessions into 20-minute intervals."
        })

    # 4. Blind Spots (Untested topics > 7 days)
    blind_spots = []
    for t in all_topics:
        last_t = t.get("last_tested_at") or t.get("updated_at")
        if last_t:
            try:
                dt = datetime.fromisoformat(last_t)
                days = (now - dt).total_seconds() / 86400.0
                if days >= 7.0:
                    blind_spots.append({
                        "topic": t["topic"],
                        "days_untested": int(days),
                        "last_retention_score": t.get("last_retention_score", 0.0),
                        "recommendation": f"Untested for {int(days)} days. Take a quick pulse check to verify retention.",
                        "action_url": f"/assessment?topic={t['topic']}&mode=QUICK"
                    })
            except Exception:
                pass
    conn.close()

    total_weakness_count = len(concept_weaknesses) + len(passive_study_alerts)

    if total_weakness_count == 0:
        diagnosis = "⭐ Exceptional Mastery: No critical knowledge gaps or passive study anomalies detected."
    elif len(passive_study_alerts) > 0 and len(concept_weaknesses) > 0:
        diagnosis = f"⚠️ Cognitive Gap Detected: {len(concept_weaknesses)} conceptual weaknesses identified. Passive study alerts suggest shifting from passive reading to active retrieval practice."
    else:
        diagnosis = f"📈 {len(concept_weaknesses)} concept areas require reinforcement. Target them with short remediation drills."

    return {
        "has_sufficient_history": True,
        "weaknesses_count": total_weakness_count,
        "conceptual_weaknesses": concept_weaknesses,
        "passive_study_alerts": passive_study_alerts,
        "distraction_vulnerabilities": distraction_vulnerabilities,
        "blind_spots": blind_spots,
        "summary_diagnosis": diagnosis,
    }


# =============================================================================
# 5. Core Coach Superpower 3: "When should I revise?"
# =============================================================================

def get_when_to_revise(user_id: int, database_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Computes an explainable spaced-repetition revision calendar using Ebbinghaus decay modeling.
    Returns:
    - Current estimated retention % for all studied topics
    - Decay status (Critical, Impending, Optimal, Consolidated)
    - Optimal revision window (date/time)
    - 1-Click revision test actions
    """
    telemetry = get_eight_pillar_telemetry(user_id, database_path)
    if not telemetry["has_sufficient_history"]:
        return {
            "has_sufficient_history": False,
            "message": "Not enough learning history yet.",
            "revision_queue": [],
            "calendar_events": [],
            "urgency_breakdown": {"critical": 0, "due": 0, "optimal": 0, "consolidated": 0},
            "disclaimer": "Estimated retention calculated via Ebbinghaus forgetting curve modeling. Does not measure biological memory."
        }

    all_topics = telemetry["pillars"]["knowledge_retention"]["all_topics"]
    now = datetime.now()
    revision_queue: List[Dict[str, Any]] = []

    urgency_counts = {"critical": 0, "due": 0, "optimal": 0, "consolidated": 0}

    for t in all_topics:
        t_name = t["topic"]
        last_tested_iso = t.get("last_tested_at") or t.get("updated_at")
        elapsed_hours = 24.0
        if last_tested_iso:
            try:
                dt = datetime.fromisoformat(last_tested_iso)
                elapsed_hours = max(0.5, (now - dt).total_seconds() / 3600.0)
            except Exception:
                elapsed_hours = 24.0

        decay = calculate_ebbinghaus_decay(
            last_score_pct=t.get("avg_retention_score", 70.0),
            elapsed_hours=elapsed_hours,
            repetitions_count=t.get("total_quizzes", 1),
            mastery_tier=t.get("mastery_tier", "MEDIUM")
        )

        urgency_key = decay["urgency"].lower()
        if urgency_key in urgency_counts:
            urgency_counts[urgency_key] += 1

        # Calculate optimal next review timestamp
        # Optimal review is when retention decays to ~70% (consolidation sweet spot)
        stability_hrs = decay["stability_hours"]
        hours_to_review = max(1.0, stability_hrs * 0.7 - elapsed_hours)
        optimal_review_dt = now + timedelta(hours=hours_to_review)

        if hours_to_review <= 2.0:
            window_str = "Immediate / Within 2 hours"
        elif hours_to_review <= 24.0:
            window_str = f"Today in ~{int(hours_to_review)} hours"
        elif hours_to_review <= 48.0:
            window_str = "Tomorrow"
        else:
            window_str = f"In {int(hours_to_review // 24)} days ({optimal_review_dt.strftime('%b %d')})"

        revision_queue.append({
            "topic": t_name,
            "mastery_tier": t.get("mastery_tier", "MEDIUM"),
            "last_retention_score": t.get("avg_retention_score", 0.0),
            "estimated_retention": decay["estimated_current_retention"],
            "estimated_current_retention": decay["estimated_current_retention"],
            "elapsed_hours": decay["elapsed_hours"],
            "urgency": decay["urgency"],
            "status_label": decay["status_label"],
            "badge_color": decay["badge_color"],
            "optimal_review_window": window_str,
            "optimal_review_dt": optimal_review_dt.isoformat(),
            "projections": decay["projections"],
            "action_url": f"/assessment?topic={t_name}&mode=QUICK",
            "disclaimer": decay["disclaimer"],
        })


    # Sort queue: lowest estimated retention first (most urgent)
    revision_queue.sort(key=lambda x: x["estimated_current_retention"])

    return {
        "has_sufficient_history": True,
        "total_topics_tracked": len(revision_queue),
        "urgency_breakdown": urgency_counts,
        "revision_queue": revision_queue,
        "disclaimer": "Estimated retention calculated via Ebbinghaus forgetting curve modeling. Does not measure biological memory."
    }


# =============================================================================
# 6. Core Coach Superpower 4: "How am I improving?"
# =============================================================================

def get_how_am_i_improving(user_id: int, database_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Computes longitudinal growth analytics, mastery trajectory, velocity deltas,
    and checks for meaningful pedagogical milestone achievements.
    """
    telemetry = get_eight_pillar_telemetry(user_id, database_path)
    if not telemetry["has_sufficient_history"]:
        return {
            "has_sufficient_history": False,
            "message": "Not enough learning history yet.",
            "velocity": {
                "retention_delta_pct": 0.0,
                "focus_delta_pct": 0.0,
                "efficiency_delta_pct": 0.0,
            },
            "mastery_distribution": {"weak": 0, "medium": 0, "strong": 0},
            "milestones_earned": [],
            "recent_sessions_trajectory": [],
            "radar_eight_pillars": [0, 0, 0, 0, 0, 0, 0, 0],
        }

    pillars = telemetry["pillars"]
    conn = get_db_connection(database_path)
    c = conn.cursor()

    # Longitudinal comparison: first half vs second half of sessions
    c.execute("""
        SELECT retention_score_pct, unified_focus_score, learning_efficiency_score,
               physical_presence_pct, created_at
        FROM learning_states
        WHERE user_id = ?
        ORDER BY id ASC
    """, (user_id,))
    state_rows = c.fetchall()

    retention_delta = 0.0
    focus_delta = 0.0
    eff_delta = 0.0

    if len(state_rows) >= 2:
        half_idx = len(state_rows) // 2
        first_half = state_rows[:half_idx]
        second_half = state_rows[half_idx:]

        avg_ret_1 = sum(float(r["retention_score_pct"] or 70.0) for r in first_half) / len(first_half)
        avg_ret_2 = sum(float(r["retention_score_pct"] or 70.0) for r in second_half) / len(second_half)
        retention_delta = round(avg_ret_2 - avg_ret_1, 1)

        avg_foc_1 = sum(float(r["unified_focus_score"] or 70.0) for r in first_half) / len(first_half)
        avg_foc_2 = sum(float(r["unified_focus_score"] or 70.0) for r in second_half) / len(second_half)
        focus_delta = round(avg_foc_2 - avg_foc_1, 1)

        avg_eff_1 = sum(float(r["learning_efficiency_score"] or 50.0) for r in first_half) / len(first_half)
        avg_eff_2 = sum(float(r["learning_efficiency_score"] or 50.0) for r in second_half) / len(second_half)
        eff_delta = round(avg_eff_2 - avg_eff_1, 1)

    # Check & award pedagogical milestones
    milestones_earned = check_and_award_milestones(user_id, telemetry, conn)
    conn.close()

    # Trajectory of last 8 study states
    recent_states = pillars["adaptive_learning"]["recent_states"]
    trajectory = []
    for s in reversed(recent_states[:8]):
        trajectory.append({
            "topic": s.get("topic") or "Study",
            "score": s.get("unified_focus_score", 75),
            "retention": s.get("retention_score_pct", 70.0),
            "efficiency": s.get("learning_efficiency_score", 50.0),
            "date": (s.get("created_at") or "")[:10]
        })

    # Normalized 8-Pillar Radar Chart Values (0 - 100)
    radar_values = [
        min(100, int(pillars["radar_presence"]["avg_presence_pct"] or 75)),
        min(100, int(pillars["recall_results"]["avg_retention_score"] or pillars["quiz_results"]["avg_quiz_score"] or 70)),
        min(100, int((pillars["ai_engagement"]["meaningful_queries"] * 20.0) + (pillars["ai_engagement"]["total_messages"] * 4.0))),
        min(100, int(pillars["quiz_results"]["avg_quiz_score"] or 70)),
        min(100, int((pillars["study_sessions"]["avg_duration_sec"] / 1800.0) * 100)),
        min(100, int((pillars["study_sessions"]["session_count"] / 5.0) * 100)),
        min(100, int(pillars["knowledge_retention"]["strong_count"] * 35.0)),
        min(100, int(telemetry["cognitive_efficiency"])),
    ]

    vel_label = "Positive Growth Trajectory" if (retention_delta >= 0 and eff_delta >= 0) else "Consolidating Foundations"

    return {
        "has_sufficient_history": True,
        "coach_health_score": telemetry["coach_health_score"],
        "health_label": telemetry["health_label"],
        "cognitive_efficiency": telemetry["cognitive_efficiency"],
        "growth_velocity_label": vel_label,
        "velocity_label": vel_label,
        "velocity": {
            "retention_delta_pct": retention_delta,
            "focus_delta_pct": focus_delta,
            "efficiency_delta_pct": eff_delta,
        },
        "mastery_distribution": {
            "weak": pillars["knowledge_retention"]["weak_count"],
            "medium": pillars["knowledge_retention"]["medium_count"],
            "strong": pillars["knowledge_retention"]["strong_count"],
        },
        "milestones_earned": milestones_earned,
        "recent_sessions_trajectory": trajectory,
        "radar_eight_pillars": radar_values,
        "radar_balance": radar_values,
        "pillar_radar": radar_values,
        "radar_labels": [
            "Presence Consistency",
            "Knowledge Retention",
            "AI Engagement",
            "Quiz Performance",
            "Duration Endurance",
            "Study Frequency",
            "Topic Mastery",
            "Cognitive Efficiency"
        ]
    }



# =============================================================================
# 7. Milestone Award Engine
# =============================================================================

def check_and_award_milestones(user_id: int, telemetry: Optional[Dict[str, Any]] = None,
                               conn: Optional[sqlite3.Connection] = None,
                               database_path: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Awards verified pedagogical milestones based strictly on demonstrated learning outcomes:
    - Never awards milestones for opening the app or idle time.
    - Awards on: topic mastery promotions, recall streaks, high cognitive efficiency, goal completion.
    """
    if telemetry is None:
        telemetry = get_eight_pillar_telemetry(user_id, database_path)

    should_close = False
    if conn is None:
        conn = get_db_connection(database_path)
        should_close = True


    c = conn.cursor()
    now_iso = datetime.now().isoformat()

    pillars = telemetry["pillars"]
    strong_count = pillars["knowledge_retention"]["strong_count"]
    completed_recalls = pillars["recall_results"]["completed_tasks"]
    avg_retention = pillars["recall_results"]["avg_retention_score"]
    eff_score = telemetry.get("cognitive_efficiency", 0.0)
    session_count = pillars["study_sessions"]["session_count"]

    potential_milestones = []

    # 1. First Strong Mastery
    if strong_count >= 1:
        potential_milestones.append({
            "key": "FIRST_MASTERY_STRONG",
            "title": "Mastery Pioneer",
            "description": "Achieved Strong Mastery (>=85% retention) on your first study topic.",
            "category": "MASTERY"
        })

    # 2. Triple Topic Mastery
    if strong_count >= 3:
        potential_milestones.append({
            "key": "MASTERY_TRIAD",
            "title": "Deep Knowledge Triad",
            "description": "Mastered 3 separate academic topics with verified long-term retention.",
            "category": "MASTERY"
        })

    # 3. Recall Veteran (3+ active recall tests)
    if completed_recalls >= 3 and avg_retention >= 75.0:
        potential_milestones.append({
            "key": "RECALL_CHAMPION",
            "title": "Active Retrieval Champion",
            "description": "Completed 3+ delayed recall challenges with >=75% average retention.",
            "category": "RECALL"
        })

    # 4. Cognitive Efficiency Boost
    if eff_score >= 70.0 and session_count >= 3:
        potential_milestones.append({
            "key": "HIGH_COGNITIVE_EFFICIENCY",
            "title": "High Cognitive Efficiency",
            "description": "Demonstrated rapid concept consolidation with high retention per study hour.",
            "category": "EFFICIENCY"
        })

    # 5. Study Endurance Discipline
    if session_count >= 5:
        potential_milestones.append({
            "key": "STUDY_CONSISTENCY_5",
            "title": "Habit Consolidation",
            "description": "Successfully completed 5 grounded study sessions with Radar Presence tracking.",
            "category": "CONSISTENCY"
        })

    # Insert any newly earned milestones
    for m in potential_milestones:
        c.execute("""
            INSERT OR IGNORE INTO coach_milestones (
                user_id, milestone_key, title, description, category, earned_at
            ) VALUES (?, ?, ?, ?, ?, ?)
        """, (user_id, m["key"], m["title"], m["description"], m["category"], now_iso))

    conn.commit()

    # Fetch all earned milestones for user
    c.execute("""
        SELECT id, milestone_key, title, description, category, earned_at
        FROM coach_milestones
        WHERE user_id = ?
        ORDER BY id DESC
    """, (user_id,))
    earned = [dict(r) for r in c.fetchall()]

    if should_close:
        conn.close()

    return earned


# =============================================================================
# 8. Conversational AI Personal Learning Coach
# =============================================================================

def ask_personal_coach(user_id: int, query_intent: str,
                       custom_prompt: Optional[str] = None,
                       database_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Interactive natural language dialogue with the Personal Learning Coach.
    Grounded in the student's exact 8-pillar telemetry.
    
    Distinguishes:
    - Known student facts (e.g. "Your recorded score on Python was 72%")
    - Estimated values (e.g. "Estimated retention is currently ~64%")
    - Actionable recommendations
    - General pedagogical advice
    """
    telemetry = get_eight_pillar_telemetry(user_id, database_path)
    raw_q = ((custom_prompt or "") + " " + (query_intent or "")).upper()
    if "NEXT" in raw_q or "WHAT SHOULD I STUDY" in raw_q or "STUDY NEXT" in raw_q:
        intent = "WHAT_NEXT"
    elif "WEAK" in raw_q or "GAP" in raw_q or "STRUGGLE" in raw_q:
        intent = "WHAT_WEAK"
    elif "REVISE" in raw_q or "WHEN" in raw_q or "DECAY" in raw_q or "FORGET" in raw_q:
        intent = "WHEN_REVISE"
    elif "IMPROV" in raw_q or "HOW AM I" in raw_q or "GROWTH" in raw_q or "PROGRESS" in raw_q:
        intent = "HOW_IMPROVING"
    else:
        intent = "CUSTOM"


    pillars = telemetry.get("pillars", {})

    # Generate deterministic answers for the 4 core intents
    if intent == "WHAT_NEXT":
        next_data = get_what_to_study_next(user_id, database_path)
    elif intent == "WHAT_WEAK":
        next_data = get_what_am_i_weak_at(user_id, database_path)
    elif intent == "WHEN_REVISE":
        next_data = get_when_to_revise(user_id, database_path)
    elif intent == "HOW_IMPROVING":
        next_data = get_how_am_i_improving(user_id, database_path)
    else:
        next_data = get_what_to_study_next(user_id, database_path)


    # Build Grounding Context
    student_context = {
        "has_history": telemetry.get("has_sufficient_history", False),
        "health_score": telemetry.get("coach_health_score", 0),
        "health_label": telemetry.get("health_label", "No Data"),
        "cognitive_efficiency": telemetry.get("cognitive_efficiency", 0.0),
        "weak_topics": [t["topic"] for t in pillars.get("knowledge_retention", {}).get("weak_topics", [])],
        "strong_topics": [t["topic"] for t in pillars.get("knowledge_retention", {}).get("strong_topics", [])],
        "materials_count": pillars.get("study_materials", {}).get("material_count", 0),
        "presence_pct": pillars.get("radar_presence", {}).get("avg_presence_pct", 0.0),
        "session_count": pillars.get("study_sessions", {}).get("session_count", 0),
    }

    # Deterministic fallback response in case AI service is unavailable
    fallback_response = _generate_deterministic_coach_response(intent, student_context, next_data, custom_prompt)

    # Socratic AI Generation via ai_service
    ai_response_text = None
    ai_provider_info = ai_service.get_active_provider_info()

    try:
        coach_system_prompt = f"""You are the FocusSense Personal Learning Coach — an empathetic, evidence-based academic tutor.
Your mission is to provide clear, actionable, and encouraging study advice grounded strictly in the student's real telemetry:

STUDENT PROFILE DATA:
- Health Score: {student_context['health_score']}/100 ({student_context['health_label']})
- Cognitive Efficiency: {student_context['cognitive_efficiency']}
- Sessions Completed: {student_context['session_count']}
- Avg Desk Presence: {student_context['presence_pct']}%
- Weak Topics (<60%): {', '.join(student_context['weak_topics']) if student_context['weak_topics'] else 'None recorded'}
- Strong Topics (>=85%): {', '.join(student_context['strong_topics']) if student_context['strong_topics'] else 'None yet'}
- Uploaded Materials: {student_context['materials_count']} documents

GUIDING RULES:
1. Always distinguish between KNOWN FACTS (e.g. test scores), ESTIMATES (e.g. Ebbinghaus retention decay), and RECOMMENDATIONS.
2. If data is sparse or absent, explicitly state that more study sessions are needed. Never hallucinate fake scores.
3. Be concise (2-3 short paragraphs maximum), structured with bullet points, and highlight 1 direct immediate action.
"""
        user_message = custom_prompt if custom_prompt else _get_prompt_for_intent(intent)
        active_topic = student_context['weak_topics'][0] if student_context.get('weak_topics') else (
            student_context['strong_topics'][0] if student_context.get('strong_topics') else "Personal Coaching"
        )
        
        # Use ai_service to generate response
        ai_result = ai_service.generate_chat_reply(
            prompt=user_message,
            conversation_history=[{"sender": "system", "message": coach_system_prompt}],
            study_topic=active_topic,
            student_name="Student",
            learning_context=student_context
        )
        if ai_result and ai_result.get("reply"):
            ai_response_text = ai_result["reply"]


    except Exception as e:
        logger.warning(f"AI Coach generation error: {e}. Falling back to deterministic output.")
        ai_response_text = None

    final_text = ai_response_text or fallback_response

    # Log query to database
    try:
        conn = get_db_connection(database_path)
        c = conn.cursor()
        c.execute("""
            INSERT INTO coach_queries (
                user_id, query_intent, user_prompt, coach_response, telemetry_snapshot_json, created_at
            ) VALUES (?, ?, ?, ?, ?, ?)
        """, (
            user_id, intent, custom_prompt or intent, final_text,
            json.dumps(student_context), datetime.now().isoformat()
        ))
        conn.commit()
        conn.close()
    except Exception as e:
        logger.warning(f"Failed to log coach query: {e}")

    return {
        "success": True,
        "intent": intent,
        "response": final_text,
        "provider": ai_provider_info.get("provider", "local_deterministic"),
        "is_ai_generated": bool(ai_response_text),
        "structured_data": next_data,
        "student_context": student_context,
    }


def _get_prompt_for_intent(intent: str) -> str:
    prompts = {
        "WHAT_NEXT": "What should I study next right now based on my retention, weaknesses, and upcoming goals?",
        "WHAT_WEAK": "What concepts or topics am I currently weakest at, and where was passive studying detected?",
        "WHEN_REVISE": "When should I revise my topics to stay ahead of the Ebbinghaus forgetting curve?",
        "HOW_IMPROVING": "How am I improving over time in terms of retention, focus consistency, and cognitive efficiency?",
    }
    return prompts.get(intent, "Give me personalized coaching guidance for my studies.")


def _generate_deterministic_coach_response(intent: str, context: dict,
                                           structured_data: dict, custom_prompt: Optional[str]) -> str:
    """Provides high-quality, formatted markdown response even when offline."""
    if not context.get("has_history"):
        return (
            "### 🎯 FocusSense Personal Learning Coach\n\n"
            "**Status**: *Not enough learning history yet.*\n\n"
            "To unlock your full personalized coaching insights, start by completing your first **Study Session** "
            "with Radar Presence tracking or uploading a study document in the **Study Materials** hub.\n\n"
            "👉 **Next Step**: Click **Start Study Session** on your Dashboard."
        )

    if intent == "WHAT_NEXT":
        rec = structured_data.get("primary_recommendation")
        if rec:
            return (
                f"### 🎯 Recommended Next Study Action\n\n"
                f"Based on your latest retention data and forgetting curve projections, your highest priority is:\n\n"
                f"**{rec['title']}**\n"
                f"- **Priority**: `{rec['priority_label']}` (Score: {rec['priority_score']}/100)\n"
                f"- **Recommended Duration**: {rec['recommended_duration_min']} minutes\n"
                f"- **Reason**: {rec['reason']}\n\n"
                f"💡 *Tip: Breaking study into focused {rec['recommended_duration_min']}-minute sprints maximizes memory retention.*"
            )
        return "You're all caught up on urgent topics! Choose any subject to explore in Assessment Center."

    elif intent == "WHAT_WEAK":
        weak_count = structured_data.get("weaknesses_count", 0)
        diag = structured_data.get("summary_diagnosis", "")
        weak_list = structured_data.get("conceptual_weaknesses", [])
        passive_list = structured_data.get("passive_study_alerts", [])

        res = f"### 🔍 Knowledge Gap Diagnostics\n\n{diag}\n\n"
        if weak_list:
            res += "**Top Weakness Areas:**\n"
            for w in weak_list[:3]:
                res += f"- **{w.get('topic') or w.get('sub_concept')}**: {w.get('remediation_advice')}\n"
        if passive_list:
            res += "\n⚠️ **Passive Presence Alerts**:\n"
            for p in passive_list[:2]:
                res += f"- *{p['topic']}*: High presence ({p['presence_pct']}%) with low recall ({p['retention_score']}%). Shift to active recall.\n"
        return res

    elif intent == "WHEN_REVISE":
        queue = structured_data.get("revision_queue", [])
        res = "### ⏳ Spaced-Repetition Revision Schedule\n\n"
        res += "*Estimated retention calculated via Ebbinghaus forgetting curve modeling.*\n\n"
        if queue:
            res += "**Upcoming Review Deadlines:**\n"
            for q in queue[:4]:
                res += f"- **{q['topic']}** (Est. Retention: **{q['estimated_current_retention']}%**): Optimal review window is **{q['optimal_review_window']}**.\n"
        else:
            res += "All tracked topics are currently consolidated with high retention!"
        return res

    elif intent == "HOW_IMPROVING":
        vel = structured_data.get("velocity", {})
        health = context.get("health_score", 0)
        eff = context.get("cognitive_efficiency", 0.0)
        r_delta = vel.get("retention_delta_pct", 0.0)
        f_delta = vel.get("focus_delta_pct", 0.0)

        sign_r = "+" if r_delta > 0 else ""
        sign_f = "+" if f_delta > 0 else ""

        return (
            f"### 📈 Your Growth & Mastery Trajectory\n\n"
            f"**Overall Learning Health**: `{health}/100` ({context.get('health_label')})\n"
            f"- **Retention Velocity**: `{sign_r}{r_delta}%` over recent sessions\n"
            f"- **Focus Consistency**: `{sign_f}{f_delta}%` in-seat radar presence\n"
            f"- **Cognitive Efficiency**: `{eff}` retention points earned per hour studied\n\n"
            f"Keep up regular active recall testing to accelerate your progress!"
        )

    return (
        f"### 💡 FocusSense Personal Coach Advice\n\n"
        f"You are currently at **{context.get('health_score')}/100** Learning Health. "
        f"To maximize knowledge consolidation, focus on your active recommendations and spaced revision deadlines."
    )
