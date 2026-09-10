"""
FocusSense - Explainable Focus Intelligence Engine
===================================================
Estimates study-session focus quality using real multimodal session telemetry:
- In-seat desk presence consistency (Radar Sensor / Simulation)
- Session completion & daily goal adherence
- Distraction control & absence frequency
- Study duration & sustained effort

SCORING FORMULA (100-Point Weighted Transparent Model):
  1. Presence Consistency   -> 40% (Ratio of time physically present at desk)
  2. Session Completion     -> 25% (Adherence to daily study targets / standard session goals)
  3. Distraction Control    -> 20% (Absence frequency & interruption penalty)
  4. Study Duration Rating  -> 15% (Sustained focus block endurance up to 30 mins)

SCORE TIERS:
  90–100 -> Excellent Focus
  75–89  -> Strong Focus
  60–74  -> Moderate Focus
  0–59   -> Needs Improvement

DISCLAIMER:
  Focus quality is estimated from desk presence data and session telemetry.
  It does not measure attention, comprehension, or cognitive brain activity.
"""

import re
from datetime import datetime, timedelta


# ---------------------------------------------------------------------------
# Constants & Quality Tiers
# ---------------------------------------------------------------------------

DISCLAIMER = (
    "Focus quality is estimated from desk presence telemetry and session habits. "
    "It measures presence consistency and distraction control, not cognitive comprehension."
)

QUALITY_TIERS = {
    "Excellent": {"min": 90, "label": "Excellent Focus",    "color": "emerald", "badge": "● High Consistency"},
    "Strong":    {"min": 75, "label": "Strong Focus",       "color": "cyan",    "badge": "● Sustained In-Seat"},
    "Moderate":  {"min": 60, "label": "Moderate Focus",     "color": "amber",   "badge": "● Intermittent Breaks"},
    "Needs Improvement": {"min": 0, "label": "Needs Improvement", "color": "rose", "badge": "● Frequent Absences"},
}

# Standard recommended focus block duration (seconds) for 100% duration rating (e.g. 30 mins)
STANDARD_FOCUS_BLOCK_SEC = 1800 
MIN_SCOREABLE_SECONDS = 300  # 5 minutes threshold for full telemetry inference


# ---------------------------------------------------------------------------
# Core Scoring Algorithm
# ---------------------------------------------------------------------------

def compute_focus_score(session_data):
    """
    Compute a real, explainable 0-100 Focus Score from session telemetry.
    
    Safe against:
    - Zero/negative duration
    - Missing or None data fields
    - Division by zero
    
    Returns structured dict with score, tier, label, factor percentages, and explanation.
    """
    duration_sec = max(1, int(session_data.get("duration_sec") or 1))
    absence_count = max(0, int(session_data.get("absence_count") or 0))
    longest_absence_sec = max(0, int(session_data.get("longest_absence_sec") or 0))
    total_absence_sec = max(0, int(session_data.get("total_absence_sec") or 0))
    raw_goal_pct = float(session_data.get("goal_progress_pct") or 0.0)

    # Clamp total absence to not exceed duration
    total_absence_sec = min(total_absence_sec, duration_sec)
    presence_sec = max(0, duration_sec - total_absence_sec)

    # -------------------------------------------------------------------------
    # 1. Presence Consistency (40% Weight -> 40 Points Max)
    # -------------------------------------------------------------------------
    presence_ratio = presence_sec / float(duration_sec)
    presence_pct = round(presence_ratio * 100.0, 1)
    presence_score = presence_ratio * 40.0

    # -------------------------------------------------------------------------
    # 2. Session Completion / Goal Adherence (25% Weight -> 25 Points Max)
    # -------------------------------------------------------------------------
    # If daily goal is tracked, use goal progress clamped to 100%.
    # If no goal is set, reward completion ratio against target focus block.
    if raw_goal_pct > 0.0:
        completion_pct = min(100.0, max(0.0, raw_goal_pct))
    else:
        completion_pct = min(100.0, (duration_sec / float(STANDARD_FOCUS_BLOCK_SEC)) * 100.0)
    completion_pct = round(completion_pct, 1)
    completion_score = (completion_pct / 100.0) * 25.0

    # -------------------------------------------------------------------------
    # 3. Distraction Control & Interruption Rate (20% Weight -> 20 Points Max)
    # -------------------------------------------------------------------------
    # Evaluate absence frequency and severity of longest break
    if absence_count == 0:
        freq_factor = 100.0
    elif absence_count == 1:
        freq_factor = 90.0
    elif absence_count == 2:
        freq_factor = 80.0
    elif absence_count <= 4:
        freq_factor = 65.0
    else:
        freq_factor = max(20.0, 100.0 - (absence_count * 10.0))

    # Longest absence penalty if longest departure exceeds 5 minutes (300s)
    if longest_absence_sec <= 60:
        long_penalty = 0.0
    elif longest_absence_sec <= 300:
        long_penalty = (longest_absence_sec / 300.0) * 10.0
    else:
        long_penalty = min(25.0, 10.0 + ((longest_absence_sec - 300.0) / 600.0) * 15.0)

    distraction_control_pct = round(max(0.0, min(100.0, freq_factor - long_penalty)), 1)
    distraction_score = (distraction_control_pct / 100.0) * 20.0

    # -------------------------------------------------------------------------
    # 4. Study Duration & Effort (15% Weight -> 15 Points Max)
    # -------------------------------------------------------------------------
    duration_rating_pct = round(min(100.0, (duration_sec / float(STANDARD_FOCUS_BLOCK_SEC)) * 100.0), 1)
    duration_score = (duration_rating_pct / 100.0) * 15.0

    # -------------------------------------------------------------------------
    # Aggregate Final Score (Bounded strictly 0 - 100)
    # -------------------------------------------------------------------------
    raw_total = presence_score + completion_score + distraction_score + duration_score
    estimated_score = max(0, min(100, int(round(raw_total))))

    # Determine Quality Tier & Label
    quality_tier = "Needs Improvement"
    for tier_key, tier_meta in QUALITY_TIERS.items():
        if estimated_score >= tier_meta["min"]:
            quality_tier = tier_key
            break

    label = get_focus_label(estimated_score)
    short_session = duration_sec < MIN_SCOREABLE_SECONDS

    interpretation = _build_interpretation(
        score=estimated_score,
        quality_tier=quality_tier,
        presence_pct=presence_pct,
        completion_pct=completion_pct,
        distraction_control_pct=distraction_control_pct,
        absence_count=absence_count,
        longest_absence_sec=longest_absence_sec,
        short_session=short_session,
        duration_sec=duration_sec,
    )

    return {
        "estimated_score":         estimated_score,
        "quality_tier":            quality_tier,
        "label":                   label,
        "presence_pct":            presence_pct,
        "completion_pct":          completion_pct,
        "distraction_control_pct": distraction_control_pct,
        "duration_rating_pct":     duration_rating_pct,
        "absence_count":           absence_count,
        "longest_absence_sec":     longest_absence_sec,
        "total_absence_sec":       total_absence_sec,
        "session_duration_sec":    duration_sec,
        "goal_progress_pct":       round(raw_goal_pct, 1),
        "short_session":           short_session,
        "interpretation":          interpretation,
        "disclaimer":              DISCLAIMER,
        "factors": {
            "presence":            presence_pct,
            "completion":          completion_pct,
            "distraction_control": distraction_control_pct,
            "duration_rating":     duration_rating_pct,
        }
    }


# ---------------------------------------------------------------------------
# Explainable Dynamic Interpretation Generator
# ---------------------------------------------------------------------------

def _build_interpretation(score, quality_tier, presence_pct, completion_pct,
                          distraction_control_pct, absence_count,
                          longest_absence_sec, short_session, duration_sec):
    """
    Generate dynamic natural-language explanation grounded in real session factors.
    """
    mins = max(1, duration_sec // 60)

    if short_session:
        if presence_pct >= 90.0:
            return f"Brief session ({mins} min) recorded with {presence_pct}% in-seat presence. Extended study sessions offer deeper focus telemetry."
        else:
            return f"Short session ({mins} min) with {absence_count} desk absence(s). Consistent in-seat presence will boost future scores."

    if presence_pct >= 90.0 and absence_count == 0:
        return f"Exceptional focus sustained throughout {mins} minutes with 100% desk presence and zero interruptions."
    elif presence_pct >= 85.0 and absence_count <= 2:
        return f"Strong focus maintained with {presence_pct}% desk presence and minimal interruptions ({absence_count} quick break{'s' if absence_count > 1 else ''})."
    elif presence_pct >= 75.0 and distraction_control_pct >= 70.0:
        return f"Solid study interval with {presence_pct}% presence. Distraction control remained steady at {distraction_control_pct}%."
    elif presence_pct >= 60.0:
        abs_str = f" ({_fmt_seconds(longest_absence_sec)} longest)" if longest_absence_sec > 0 else ""
        return f"Moderate focus recorded with {absence_count} desk absence(s){abs_str}. Aim for longer uninterrupted study blocks."
    else:
        return f"Focus needs improvement due to frequent desk absence ({round(100.0 - presence_pct)}% time away). Minimizing departures will improve score."


# ---------------------------------------------------------------------------
# Label and UI Helpers
# ---------------------------------------------------------------------------

def get_focus_label(score_or_tier):
    """
    Return human-readable focus label for a score (int/float) or tier name (str).
    90–100 -> Excellent Focus
    75–89  -> Strong Focus
    60–74  -> Moderate Focus
    0–59   -> Needs Improvement
    """
    if isinstance(score_or_tier, (int, float)):
        score = float(score_or_tier)
        if score >= 90:
            return "Excellent Focus"
        elif score >= 75:
            return "Strong Focus"
        elif score >= 60:
            return "Moderate Focus"
        else:
            return "Needs Improvement"

    tier_str = str(score_or_tier).strip()
    if tier_str in QUALITY_TIERS:
        return QUALITY_TIERS[tier_str]["label"]
    elif "Excellent" in tier_str:
        return "Excellent Focus"
    elif "Good" in tier_str or "Strong" in tier_str:
        return "Strong Focus"
    elif "Moderate" in tier_str:
        return "Moderate Focus"
    return "Needs Improvement"


def get_tier_color(tier_or_score):
    """Return appropriate CSS variable or hex color for a tier or score."""
    label = get_focus_label(tier_or_score)
    if label == "Excellent Focus":
        return "var(--accent-emerald)"
    elif label == "Strong Focus":
        return "var(--accent-cyan)"
    elif label == "Moderate Focus":
        return "var(--accent-amber)"
    return "var(--accent-rose)"


def compute_live_preview(elapsed_sec, current_absence_sec, completed_absence_events):
    """
    Compute real-time live preview telemetry during an ongoing study session.
    Safe against short/early durations. Does not present a premature final score.
    """
    if elapsed_sec <= 0:
        return {
            "is_active":               True,
            "has_enough_data":         False,
            "estimated_score":         None,
            "live_preview_score":      None,
            "status_label":            "Calibrating...",
            "presence_pct":            100.0,
            "completion_pct":          0.0,
            "distraction_control_pct": 100.0,
            "duration_rating_pct":     0.0,
            "absence_count":           0,
            "longest_absence_sec":     0,
            "total_absence_sec":       0,
            "quality_tier":            "Excellent",
            "label":                   "Ready to Focus",
            "interpretation":          "Session started. Calibrating desk presence and focus stability in real time. Final score will be calculated when you stop the session.",
        }

    completed_absence_sec = sum(
        (e.get("duration_seconds") or 0) for e in completed_absence_events
    )
    total_absence_sec = completed_absence_sec + current_absence_sec
    absence_count = len(completed_absence_events) + (1 if current_absence_sec > 0 else 0)
    longest = max(
        [e.get("duration_seconds") or 0 for e in completed_absence_events] + [current_absence_sec],
        default=0,
    )

    result = compute_focus_score({
        "duration_sec":        elapsed_sec,
        "absence_count":       absence_count,
        "longest_absence_sec": longest,
        "total_absence_sec":   total_absence_sec,
        "goal_progress_pct":   0.0,
    })

    has_enough_data = elapsed_sec >= MIN_SCOREABLE_SECONDS
    status_label = "Calibrating..." if not has_enough_data else "Live Tracking"

    # Dynamic live status interpretation
    mins_str = _fmt_seconds(elapsed_sec)
    if elapsed_sec < 60:
        interpretation = f"Session started ({mins_str} elapsed). Calibrating desk presence and focus stability in real time. Final score will be calculated when you stop the session."
    elif absence_count == 0:
        interpretation = f"Active session in progress ({mins_str} elapsed). 100% desk presence with 0 interruptions so far. Final score will be calculated upon stopping."
    else:
        interpretation = f"Active session in progress ({mins_str} elapsed). {result['presence_pct']}% desk presence recorded with {absence_count} absence event(s). Final score will be calculated upon stopping."

    return {
        "is_active":               True,
        "has_enough_data":         has_enough_data,
        "estimated_score":         None,  # Do not present final score until session stops
        "live_preview_score":      result["estimated_score"] if has_enough_data else None,
        "status_label":            status_label,
        "quality_tier":            result["quality_tier"],
        "label":                   result["label"],
        "presence_pct":            result["presence_pct"],
        "completion_pct":          result["completion_pct"],
        "distraction_control_pct": result["distraction_control_pct"],
        "duration_rating_pct":     result["duration_rating_pct"],
        "absence_count":           absence_count,
        "longest_absence_sec":     longest,
        "total_absence_sec":       total_absence_sec,
        "interpretation":          interpretation,
    }


# ---------------------------------------------------------------------------
# Three-Pillar Focus & Retention Intelligence (Phase 3)
# ---------------------------------------------------------------------------

def compute_three_pillar_intelligence(presence_pct, behavior_pct, retention_score_pct=None):
    """
    Computes explainable 3-Pillar Learning Telemetry:
    1. Physical Presence (Radar Desk In-Seat Ratio)
    2. Study Behavior (Distraction Control & Endurance)
    3. Knowledge Retention (Delayed Recall Accuracy)
    
    If retention has not been tested yet, returns telemetry-only baseline.
    """
    p_pct = max(0.0, min(100.0, float(presence_pct or 0.0)))
    b_pct = max(0.0, min(100.0, float(behavior_pct or 0.0)))
    has_retention = retention_score_pct is not None
    r_pct = max(0.0, min(100.0, float(retention_score_pct))) if has_retention else None

    if has_retention:
        # 3-Pillar Weighted Composite: 35% Presence, 30% Behavior, 35% Retention
        composite = (p_pct * 0.35) + (b_pct * 0.30) + (r_pct * 0.35)
        composite_score = int(round(composite))
    else:
        # 2-Pillar Baseline: 55% Presence, 45% Behavior
        composite = (p_pct * 0.55) + (b_pct * 0.45)
        composite_score = int(round(composite))

    label = get_focus_label(composite_score)

    if has_retention:
        if r_pct >= 85 and p_pct >= 85:
            insight = "⭐ Holistic Mastery: High desk presence compounded into excellent long-term recall."
        elif p_pct >= 85 and r_pct < 60:
            insight = "⚠️ Passive Presence Detected: High desk time but lower recall. Focus on active recall during study."
        elif r_pct >= 80 and p_pct < 70:
            insight = "💡 High Efficiency: Solid retention despite frequent breaks. Aim to stabilize session endurance."
        else:
            insight = f"Balanced learning profile with {composite_score}% composite index."
    else:
        insight = "Physical session completed. Delayed recall test scheduled to evaluate long-term retention."

    return {
        "composite_score": composite_score,
        "label": label,
        "has_retention": has_retention,
        "presence_pct": round(p_pct, 1),
        "behavior_pct": round(b_pct, 1),
        "retention_pct": round(r_pct, 1) if has_retention else None,
        "insight": insight,
        "pillars": [
            {
                "id": "presence",
                "name": "Physical Presence",
                "value": round(p_pct, 1),
                "badge": "Radar Telemetry",
                "description": "In-seat physical consistency at desk",
                "color": "var(--accent-emerald)" if p_pct >= 80 else ("var(--accent-amber)" if p_pct >= 60 else "var(--accent-rose)")
            },
            {
                "id": "behavior",
                "name": "Study Behavior",
                "value": round(b_pct, 1),
                "badge": "Session Habit",
                "description": "Distraction control & sustained endurance",
                "color": "var(--accent-cyan)" if b_pct >= 75 else "var(--accent-amber)"
            },
            {
                "id": "retention",
                "name": "Knowledge Retention",
                "value": round(r_pct, 1) if has_retention else None,
                "badge": "Delayed Recall" if has_retention else "Pending Test",
                "description": "Demonstrated recall accuracy across sub-concepts" if has_retention else "Scheduled for post-study evaluation",
                "color": ("var(--accent-purple)" if (r_pct or 0) >= 80 else "var(--accent-amber)") if has_retention else "var(--text-muted)"
            }
        ]
    }


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Meaningful AI Learning Engagement & Anti-Gaming Classifier
# ---------------------------------------------------------------------------

MEANINGFUL_QUERY_PATTERNS = [
    r"\b(what|why|how|when|where|who|which)\b",
    r"\b(explain|describe|clarify|define|definition|meaning)\b",
    r"\b(example|sample|instance|demonstrate|show me|walk me through)\b",
    r"\b(difference|compare|contrast|vs|versus|tradeoff|pros and cons)\b",
    r"\b(solve|solution|debug|error|issue|bug|fix|code|implement|function)\b",
    r"\b(why does|how to|can you|help me understand|review|quiz me|test me)\b",
    r"\b(concept|algorithm|mechanism|syntax|architecture|formula|rule)\b",
    r"\b(mutable|immutable|scope|decorator|closure|recursion|pointer|class|object|inheritance)\b",
    r"\b(join|query|index|key|table|process|thread|deadlock|memory|stack|heap)\b",
]

TRIVIAL_FILLERS = {
    "hi", "hello", "hey", "sup", "bye", "goodbye", "ok", "okay", "k",
    "cool", "nice", "yeah", "yes", "no", "nope", "yep", "thanks", "thx",
    "thank you", "lol", "haha", "test", "asdf", "qwerty", "yo", "hmm", "ok cool"
}


def classify_meaningful_query(message: str, topic: str = None) -> dict:
    """
    Evaluates whether a student message constitutes a substantive pedagogical inquiry.
    Filters out idle greetings, single-word fillers, and spam messages.
    """
    if not message or not isinstance(message, str):
        return {"is_meaningful": False, "reason": "Empty message", "confidence": 0.0}

    clean_msg = message.strip()
    lower_msg = clean_msg.lower()

    # 1. Reject trivial fillers
    if lower_msg in TRIVIAL_FILLERS or len(lower_msg) <= 3:
        return {"is_meaningful": False, "reason": "Trivial greeting / filler phrase", "confidence": 0.1}

    # 2. Check topic keyword presence
    topic_matched = False
    if topic and topic.strip() and topic != "General Study":
        topic_words = [w.lower() for w in re.split(r"[\s\-_,]+", topic.strip()) if len(w) > 2]
        for tw in topic_words:
            if tw in lower_msg:
                topic_matched = True
                break

    # 3. Match learning inquiry patterns
    pattern_matches = 0
    for pat in MEANINGFUL_QUERY_PATTERNS:
        if re.search(pat, lower_msg):
            pattern_matches += 1

    # 4. Length and punctuation check (question mark, substantive prompt length)
    has_question_mark = "?" in clean_msg
    is_substantive_length = len(clean_msg) >= 15

    if topic_matched or pattern_matches >= 1 or (has_question_mark and is_substantive_length):
        return {
            "is_meaningful": True,
            "reason": "Substantive conceptual or topic inquiry",
            "confidence": min(1.0, 0.5 + (pattern_matches * 0.2) + (0.3 if topic_matched else 0.0))
        }

    if is_substantive_length and len(clean_msg.split()) >= 4:
        return {
            "is_meaningful": True,
            "reason": "Multi-word contextual study message",
            "confidence": 0.6
        }

    return {
        "is_meaningful": False,
        "reason": "Short non-inquiry chatter",
        "confidence": 0.3
    }


def evaluate_ai_study_interaction(active_duration_sec=0, question_count=0, meaningful_query_count=0,
                                  messages: list = None, study_topic: str = None) -> dict:
    """
    Evaluates AI study engagement under strict anti-gaming rules:
    - Zero credit for idle open tabs or non-meaningful chatter.
    - Caps active engagement based on valid conceptual queries.
    - Supports both legacy numerical inputs and raw message list analysis.
    """
    # If list of message objects is provided, analyze directly
    if messages is not None:
        user_msgs = [m.get("message", "") for m in messages if m.get("sender") == "user"]
        total_q = len(user_msgs)
        
        # Deduplicate to prevent spamming identical strings
        unique_msgs = set(m.strip().lower() for m in user_msgs if m.strip())
        meaningful_count = 0
        for u_msg in unique_msgs:
            classification = classify_meaningful_query(u_msg, study_topic)
            if classification["is_meaningful"]:
                meaningful_count += 1
        raw_duration = int(active_duration_sec or 0)
    else:
        meaningful_count = max(0, int(meaningful_query_count or 0))
        total_q = max(0, int(question_count or 0))
        raw_duration = max(0, int(active_duration_sec or 0))

    if meaningful_count == 0:
        return {
            "credit_score": 0.0,
            "engagement_pct": 0.0,
            "meaningful_queries": 0,
            "total_queries": total_q,
            "capped_active_sec": 0,
            "status": "idle_no_interaction",
            "is_gamed": raw_duration > 300 or (total_q > 5 and meaningful_count == 0),
            "explanation": "No meaningful conceptual queries submitted. Idle time and unrelated chatter yield zero focus credit."
        }

    # Award up to 20% engagement per distinct meaningful inquiry up to 100%
    engagement_pct = min(100.0, meaningful_count * 20.0 + min(20.0, (raw_duration / 300.0) * 10.0))
    engagement_pct = round(engagement_pct, 1)

    # Max 180 seconds (3 mins) of focus credit earned per meaningful question asked
    max_allowable_sec = meaningful_count * 180
    capped_sec = min(raw_duration, max_allowable_sec)
    bonus = min(5.0, (meaningful_count * 1.5) + (capped_sec / 360.0))

    return {
        "credit_score": round(bonus, 1),
        "engagement_pct": engagement_pct,
        "meaningful_queries": meaningful_count,
        "meaningful_query_count": meaningful_count,
        "total_queries": total_q,
        "capped_active_sec": capped_sec,
        "status": "active_study",
        "is_gamed": False,
        "explanation": f"Awarded focus engagement for {meaningful_count} meaningful conceptual inquiries."
    }


def compute_habit_consistency(user_id: int, current_date_str: str = None, database_path: str = None) -> dict:
    """
    Computes 7-day study habit consistency (0 - 100%) based on multi-day session frequency and streaks.
    """
    if not user_id:
        return {"consistency_pct": 70.0, "active_days": 1, "total_sessions_7d": 1, "streak_days": 1}

    if database_path is None:
        try:
            from database import DATABASE
            database_path = DATABASE
        except Exception:
            database_path = "users.db"

    import sqlite3
    try:
        conn = sqlite3.connect(database_path, timeout=10.0)
        conn.row_factory = sqlite3.Row
        c = conn.cursor()

        target_date = datetime.strptime(current_date_str, "%Y-%m-%d") if current_date_str else datetime.now()
        start_7d = (target_date - timedelta(days=6)).strftime("%Y-%m-%d")
        end_7d = target_date.strftime("%Y-%m-%d")

        c.execute("""
            SELECT date, COUNT(*) as sess_count, SUM(duration) as total_dur
            FROM study_sessions
            WHERE user_id = ? AND date >= ? AND date <= ?
            GROUP BY date
            ORDER BY date ASC
        """, (user_id, start_7d, end_7d))
        rows = c.fetchall()
        conn.close()

        active_days = len(rows)
        total_sessions = sum(r["sess_count"] for r in rows)

        # Baseline: 3+ active days out of 7 is solid consistency (80%+), 5+ days is 95%+
        if active_days >= 5:
            consistency_pct = min(100.0, 90.0 + (total_sessions * 1.5))
        elif active_days >= 3:
            consistency_pct = min(88.0, 75.0 + (active_days * 3.0))
        elif active_days >= 1:
            consistency_pct = min(74.0, 50.0 + (active_days * 10.0))
        else:
            consistency_pct = 40.0

        return {
            "consistency_pct": round(consistency_pct, 1),
            "active_days_7d": active_days,
            "total_sessions_7d": total_sessions,
            "explanation": f"Studied on {active_days} of the past 7 days across {total_sessions} sessions."
        }
    except Exception as e:
        return {
            "consistency_pct": 70.0,
            "active_days_7d": 1,
            "total_sessions_7d": 1,
            "explanation": f"Default habit consistency estimation ({e})"
        }


# ---------------------------------------------------------------------------
# Internal Utilities
# ---------------------------------------------------------------------------

def _fmt_seconds(sec):
    m, s = divmod(int(sec), 60)
    if m == 0:
        return f"{s}s"
    if s == 0:
        return f"{m} min"
    return f"{m} min {s}s"


