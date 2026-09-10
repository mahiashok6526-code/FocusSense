from flask import Flask, render_template, request, redirect, session, jsonify, flash, url_for
import os
import sqlite3
from datetime import datetime, timedelta
from werkzeug.security import generate_password_hash, check_password_hash
from hardware.sensor import check_radar, check_light, sensor_controller
from focus_intelligence import (
    compute_focus_score,
    compute_live_preview,
    compute_three_pillar_intelligence,
    get_focus_label,
    evaluate_ai_study_interaction,
    compute_habit_consistency,
    classify_meaningful_query,
    DISCLAIMER as FI_DISCLAIMER,
)
from database import create_database, DATABASE, DB_PATH, get_db_connection
from ai_service import ai_service
from quiz_engine import (
    create_or_get_session_quiz,
    create_or_get_material_quiz,
    evaluate_student_answer,
    complete_quiz,
    get_latest_quiz_overview,
)
from study_material_engine import (
    save_and_process_material,
    get_user_materials,
    get_material_by_id,
    delete_user_material,
    build_material_grounded_context,
    safe_resolve_material_path,
    validate_file_metadata,
)
from recall_engine import (
    schedule_recall_task,
    get_pending_and_due_tasks,
    start_recall_test,
    submit_recall_answer,
    finalize_recall_test,
    get_weak_topics_summary,
    get_recall_history,
)
from learning_engine import (
    calculate_retention_score,
    update_topic_mastery,
    get_topic_mastery_list,
    get_weak_topics,
    get_student_learning_profile,
    get_adaptive_quiz_config,
    compute_learning_focus_score,
    compute_unified_focus_score,
    select_adaptive_action,
    calculate_adaptive_reward,
    record_learning_state,
    record_adaptive_action,
    record_adaptive_reward,
    get_latest_learning_state,
    get_unified_learning_profile,
)
from assessment_engine import (
    create_assessment,
    get_active_assessment,
    submit_assessment_answer,
    finalize_assessment,
    get_assessment_results,
    get_user_assessment_history,
    explain_assessment_mistake as engine_explain_mistake,
)
from coach_engine import (
    get_eight_pillar_telemetry,
    get_what_to_study_next,
    get_what_am_i_weak_at,
    get_when_to_revise,
    get_how_am_i_improving,
    ask_personal_coach,
    calculate_ebbinghaus_decay,
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
app = Flask(
    __name__,
    template_folder=os.path.join(BASE_DIR, "templates"),
    static_folder=os.path.join(BASE_DIR, "static"),
)
app.secret_key = os.environ.get("SECRET_KEY", "FocusSense2026")

# Initialize database schema if tables do not exist
create_database()



def get_db():
    """Helper to establish a connection with row access by column name."""
    conn = sqlite3.connect(DATABASE, timeout=30.0)
    conn.row_factory = sqlite3.Row
    return conn


def calculate_streak(user_id):
    """Calculate the consecutive days study streak for a user."""
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT DISTINCT date 
        FROM study_sessions 
        WHERE user_id = ? 
        ORDER BY date DESC
    """, (user_id,))
    rows = cursor.fetchall()
    conn.close()

    if not rows:
        return 0

    dates = [datetime.strptime(row["date"], "%Y-%m-%d").date() for row in rows]
    today = datetime.now().date()
    yesterday = today - timedelta(days=1)

    # Check if the streak starts today or yesterday
    if dates[0] != today and dates[0] != yesterday:
        return 0

    streak = 1
    current_date = dates[0]

    for next_date in dates[1:]:
        if (current_date - next_date).days == 1:
            streak += 1
            current_date = next_date
        elif (current_date - next_date).days == 0:
            continue
        else:
            break

    return streak


def get_weekly_analytics(user_id):
    """Retrieve last 7 days study minutes and labels."""
    conn = get_db()
    cursor = conn.cursor()

    labels = []
    values = []  # in minutes
    
    today = datetime.now().date()
    for i in range(6, -1, -1):
        day_date = today - timedelta(days=i)
        day_str = day_date.strftime("%Y-%m-%d")
        day_label = day_date.strftime("%a")  # Mon, Tue, etc.

        cursor.execute("""
            SELECT COALESCE(SUM(duration), 0) AS total_sec
            FROM study_sessions
            WHERE user_id = ? AND date = ?
        """, (user_id, day_str))
        
        row = cursor.fetchone()
        sec = row["total_sec"] if row else 0
        minutes = round(sec / 60, 1)

        labels.append(day_label)
        values.append(minutes)

    conn.close()
    return labels, values


def get_ai_insight(progress, total_seconds, total_sessions, streak):
    """Generate dynamic AI study recommendation based on student telemetry."""
    if progress >= 100:
        return "🎯 Daily Goal Achieved! Your focus endurance is exceptional today. Consider taking a 15-minute restorative break."
    elif total_seconds > 5400: # > 1.5 hours
        return "🧠 Deep Work Zone: High cognitive continuity detected. Hydrate and do light stretching to maintain peak focus."
    elif streak >= 3:
        return f"🔥 Momentum Alert: You are on a {streak}-day study streak! Consistent micro-sessions compound into mastery."
    elif total_sessions == 0:
        return "🚀 Welcome! Start your first session today to calibrate radar presence detection and optimal focus lighting."
    elif progress < 50:
        return "💡 Optimal Focus Window: Afternoon study intervals show 18% higher retention when paired with 25-minute Pomodoro sprints."
    else:
        return "📈 Steady Progress: You're halfway through today's target. Keep up the rhythm for optimal habit consolidation."


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")

        conn = get_db()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM users WHERE username = ?", (username,))
        user = cursor.fetchone()
        conn.close()

        # Support both hashed passwords and legacy plaintext passwords
        if user:
            stored_pw = user["password"]
            password_valid = False
            try:
                password_valid = check_password_hash(stored_pw, password)
            except Exception:
                password_valid = False

            if not password_valid and stored_pw == password:
                password_valid = True

            if password_valid:
                session["user_id"] = user["id"]
                session["fullname"] = user["fullname"]
                session["username"] = user["username"]
                flash(f"Welcome back, {user['fullname']}!", "success")
                return redirect("/dashboard")

        flash("Invalid username or password.", "error")
        return render_template("login.html")

    return render_template("login.html")


@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        fullname = request.form.get("fullname", "").strip()
        email = request.form.get("email", "").strip()
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")

        if not (fullname and email and username and password):
            flash("All fields are required.", "error")
            return render_template("register.html")

        conn = get_db()
        cursor = conn.cursor()

        # Check for existing username or email
        cursor.execute("SELECT id FROM users WHERE username = ? OR email = ?", (username, email))
        existing_user = cursor.fetchone()

        if existing_user:
            conn.close()
            flash("Username or Email is already registered.", "error")
            return render_template("register.html")

        hashed_password = generate_password_hash(password)
        cursor.execute("""
            INSERT INTO users(fullname, email, username, password)
            VALUES (?, ?, ?, ?)
        """, (fullname, email, username, hashed_password))
        conn.commit()
        conn.close()

        flash("Registration successful! Please log in.", "success")
        return redirect("/login")

    return render_template("register.html")


@app.route("/dashboard")
def dashboard():
    if "user_id" not in session:
        return redirect("/login")

    today = datetime.now().strftime("%Y-%m-%d")
    user_id = session["user_id"]

    conn = get_db()
    cursor = conn.cursor()

    # Total study sessions
    cursor.execute("SELECT COUNT(*) AS total FROM study_sessions WHERE user_id = ?", (user_id,))
    total_sessions_row = cursor.fetchone()
    total_sessions = total_sessions_row["total"] if total_sessions_row else 0

    # Today's total study time (seconds)
    cursor.execute("""
        SELECT COALESCE(SUM(duration), 0) AS total_today
        FROM study_sessions
        WHERE user_id = ? AND date = ?
    """, (user_id, today))
    today_row = cursor.fetchone()
    total_seconds = today_row["total_today"] if today_row else 0

    # Scored Focus Intelligence statistics
    cursor.execute("""
        SELECT 
            COALESCE(AVG(sfs.estimated_score), 0) AS avg_focus,
            MAX(sfs.estimated_score) AS best_focus,
            MIN(sfs.estimated_score) AS lowest_focus,
            COUNT(sfs.id) AS scored_count
        FROM session_focus_scores sfs
        WHERE sfs.user_id = ?
    """, (user_id,))
    score_stats_row = cursor.fetchone()
    scored_count = score_stats_row["scored_count"] if score_stats_row else 0
    avg_focus = int(round(score_stats_row["avg_focus"])) if score_stats_row and scored_count > 0 else 0
    focus_label = get_focus_label(avg_focus) if scored_count > 0 else "No Score Yet"

    # Daily goal
    cursor.execute("SELECT daily_goal FROM goals WHERE user_id = ?", (user_id,))
    goal_row = cursor.fetchone()
    daily_goal = goal_row["daily_goal"] if goal_row else 2

    # Recent 3 sessions for dashboard activity feed — include FI data
    cursor.execute("""
        SELECT ss.date, ss.start_time, ss.end_time, ss.duration, ss.focus_score,
               sfs.estimated_score AS fi_score, sfs.quality_tier AS fi_tier,
               sfs.presence_pct AS fi_presence_pct
        FROM study_sessions ss
        LEFT JOIN session_focus_scores sfs ON sfs.session_id = ss.id
        WHERE ss.user_id = ?
        ORDER BY ss.id DESC LIMIT 3
    """, (user_id,))
    recent_sessions = cursor.fetchall()

    # Most recent session Focus Intelligence score and factors for dashboard breakdown
    cursor.execute("""
        SELECT sfs.estimated_score, sfs.quality_tier, sfs.presence_pct, sfs.absence_count,
               sfs.longest_absence_sec, sfs.total_absence_sec, sfs.session_duration_sec,
               sfs.goal_progress_pct, ss.date AS session_date
        FROM session_focus_scores sfs
        JOIN study_sessions ss ON ss.id = sfs.session_id
        WHERE sfs.user_id = ?
        ORDER BY sfs.id DESC LIMIT 1
    """, (user_id,))
    last_fi_row = cursor.fetchone()

    # Recent study topics for topic grounding autocomplete/suggestions
    cursor.execute("""
        SELECT topic_name FROM study_topics
        WHERE user_id = ?
        ORDER BY last_studied_at DESC LIMIT 5
    """, (user_id,))
    recent_topics = [r["topic_name"] for r in cursor.fetchall()]

    conn.close()

    goal_seconds = max(1, daily_goal * 3600)
    progress = min(int((total_seconds / goal_seconds) * 100), 100)

    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    today_study = f"{hours} hr {minutes} min"

    streak = calculate_streak(user_id)
    weekly_labels, weekly_values = get_weekly_analytics(user_id)
    ai_insight = get_ai_insight(progress, total_seconds, total_sessions, streak)

    # Format recent sessions
    recent_list = []
    for s in recent_sessions:
        mins = max(1, s["duration"] // 60)
        score_val = s["fi_score"] if s["fi_score"] is not None else s["focus_score"]
        recent_list.append({
            "date":            s["date"],
            "time":            s["start_time"],
            "duration":        f"{mins} min",
            "focus":           score_val,
            "fi_score":        s["fi_score"],
            "fi_tier":         s["fi_tier"],
            "fi_presence_pct": s["fi_presence_pct"],
            "fi_label":        get_focus_label(score_val) if score_val else "Recorded",
        })

    last_fi = None
    latest_session_score = None
    latest_session_label = "No Score Yet"
    has_scored_session = False

    if last_fi_row:
        fi_calc = compute_focus_score({
            "duration_sec":        last_fi_row["session_duration_sec"],
            "absence_count":       last_fi_row["absence_count"],
            "longest_absence_sec": last_fi_row["longest_absence_sec"],
            "total_absence_sec":   last_fi_row["total_absence_sec"],
            "goal_progress_pct":   last_fi_row["goal_progress_pct"],
        })
        last_fi = {
            "score":                   last_fi_row["estimated_score"],
            "tier":                    last_fi_row["quality_tier"],
            "label":                   get_focus_label(last_fi_row["estimated_score"]),
            "presence_pct":            last_fi_row["presence_pct"],
            "completion_pct":          fi_calc["completion_pct"],
            "distraction_control_pct": fi_calc["distraction_control_pct"],
            "duration_rating_pct":     fi_calc["duration_rating_pct"],
            "absence_count":           last_fi_row["absence_count"],
            "interpretation":          fi_calc["interpretation"],
            "date":                    last_fi_row["session_date"],
        }
        latest_session_score = last_fi_row["estimated_score"]
        latest_session_label = last_fi["label"]
        has_scored_session = True

    is_session_active = "study_start_time" in session
    current_study_topic = session.get("study_topic", "")
    ai_provider_info = ai_service.get_active_provider_info()
    latest_quiz = get_latest_quiz_overview(user_id)

    # FocusSense Recall Engine: Pending & due recall tasks
    recall_task_data = get_pending_and_due_tasks(user_id)
    weak_concepts_summary = get_weak_topics_summary(user_id)

    # Fetch latest completed recall test score for 3-Pillar computation
    conn_r = get_db()
    cursor_r = conn_r.cursor()
    cursor_r.execute("""
        SELECT retention_score FROM recall_tasks
        WHERE user_id = ? AND status = 'COMPLETED' AND retention_score IS NOT NULL
        ORDER BY id DESC LIMIT 1
    """, (user_id,))
    recent_recall_row = cursor_r.fetchone()
    conn_r.close()

    recent_recall_score = recent_recall_row["retention_score"] if recent_recall_row else (latest_quiz["score_pct"] if latest_quiz.get("has_quiz") else None)

    # Three-Pillar Intelligence: Physical Presence + Study Behavior + Knowledge Retention
    presence_val = last_fi["presence_pct"] if last_fi else 100.0
    behavior_val = ((last_fi["distraction_control_pct"] + last_fi["completion_pct"]) / 2.0) if last_fi else 80.0
    three_pillars = compute_three_pillar_intelligence(presence_val, behavior_val, recent_recall_score)

    # FocusSense Learning Intelligence: Student Profile & Topic Mastery
    learning_profile_data = get_student_learning_profile(user_id)
    topic_mastery_data = get_topic_mastery_list(user_id)

    return render_template(
        "dashboard.html",
        fullname=session.get("fullname", "Student"),
        today_study=today_study,
        total_sessions=total_sessions,
        daily_goal=daily_goal,
        progress=progress,
        streak=streak,
        avg_focus=avg_focus,
        focus_label=focus_label,
        ai_insight=ai_insight,
        weekly_labels=weekly_labels,
        weekly_values=weekly_values,
        recent_sessions=recent_list,
        simulation_mode=sensor_controller.is_simulation_mode(),
        radar_status=check_radar(is_session_active),
        last_fi=last_fi,
        latest_session_score=latest_session_score,
        latest_session_label=latest_session_label,
        has_scored_session=has_scored_session,
        is_session_active=is_session_active,
        fi_disclaimer=FI_DISCLAIMER,
        recent_topics=recent_topics,
        current_study_topic=current_study_topic,
        ai_provider_info=ai_provider_info,
        latest_quiz=latest_quiz,
        recall_tasks=recall_task_data,
        due_recall_task=recall_task_data.get("primary_task"),
        weak_concepts=weak_concepts_summary,
        three_pillars=three_pillars,
        learning_profile=learning_profile_data,
        topic_mastery=topic_mastery_data,
        unified_profile=get_unified_learning_profile(user_id),
        coach_telemetry=get_eight_pillar_telemetry(user_id),
        coach_next=get_what_to_study_next(user_id),
    )



@app.route("/logout")
def logout():
    session.clear()
    flash("You have been logged out.", "info")
    return redirect("/login")


@app.route("/start-study", methods=["POST"])
def start_study():
    if "user_id" not in session:
        return jsonify({"error": "Please login", "success": False}), 401

    user_id = session["user_id"]
    data = request.get_json(silent=True) or {}
    topic = (data.get("topic") or "").strip() or "General Study"

    if "study_start_time" not in session:
        session["study_start_time"] = datetime.now().isoformat()
        session["study_topic"] = topic
        # Focus Intelligence: reset in-session absence tracking
        session["fi_last_presence"] = "ABSENT"  # will be set on first /status poll
        session["fi_absence_start"] = None

        # Record study topic in catalog
        now_iso = datetime.now().isoformat()
        conn = get_db()
        cursor = conn.cursor()
        cursor.execute("SELECT id FROM study_topics WHERE user_id = ? AND topic_name = ?", (user_id, topic))
        t_row = cursor.fetchone()
        if t_row:
            cursor.execute("UPDATE study_topics SET last_studied_at = ?, session_count = session_count + 1 WHERE id = ?", (now_iso, t_row["id"]))
        else:
            cursor.execute("INSERT INTO study_topics (user_id, topic_name, created_at, last_studied_at, session_count) VALUES (?, ?, ?, ?, 1)", (user_id, topic, now_iso, now_iso))

        # Create active AI conversation for this session
        cursor.execute("""
            INSERT INTO ai_conversations (user_id, topic, title, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?)
        """, (user_id, topic, f"Study: {topic}", now_iso, now_iso))
        conv_id = cursor.lastrowid
        session["ai_conversation_id"] = conv_id

        # Insert initial AI greeting grounded in the topic
        student_name = session.get("fullname", "Student")
        initial_greeting = (
            f"Hello {student_name}! I'm **FocusSense AI**, your companion for this session.\n\n"
            f"🎯 **Target Topic**: **{topic}**\n\n"
            f"Ask me to explain concepts, clarify doubts, walk through examples, or summarize key takeaways as you study. How would you like to start?"
        )
        now_ts = datetime.now().strftime("%H:%M")
        cursor.execute("""
            INSERT INTO ai_messages (conversation_id, sender, message, timestamp)
            VALUES (?, 'assistant', ?, ?)
        """, (conv_id, initial_greeting, now_ts))

        conn.commit()
        conn.close()

    return jsonify({
        "success": True,
        "message": "Study session started",
        "topic": session.get("study_topic", "General Study"),
        "conversation_id": session.get("ai_conversation_id")
    })


@app.route("/stop-study", methods=["POST"])
def stop_study():
    if "user_id" not in session:
        return jsonify({"error": "Please login", "success": False}), 401

    if "study_start_time" not in session:
        return jsonify({
            "error": "No study session is running",
            "success": False
        }), 400

    user_id = session["user_id"]
    topic = session.get("study_topic", "General Study")
    try:
        start_time = datetime.fromisoformat(session["study_start_time"])
    except Exception:
        start_time = datetime.now()

    end_time = datetime.now()
    duration = max(1, int((end_time - start_time).total_seconds()))

    conn = get_db()
    cursor = conn.cursor()

    # --- Close any open absence event if the student was ABSENT at stop time ---
    fi_absence_start = session.get("fi_absence_start")
    if fi_absence_start:
        try:
            abs_start_dt = datetime.fromisoformat(fi_absence_start)
            abs_dur = max(0, int((end_time - abs_start_dt).total_seconds()))
            cursor.execute("""
                UPDATE session_absence_events
                SET absence_end = ?, duration_seconds = ?
                WHERE user_id = ? AND absence_start = ? AND session_id IS NULL
            """, (end_time.isoformat(), abs_dur, user_id, fi_absence_start))
        except Exception:
            pass
        session["fi_absence_start"] = None

    # --- Save the study session record with topic ---
    cursor.execute("""
        INSERT INTO study_sessions
        (user_id, date, start_time, end_time, duration, focus_score, topic)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (
        user_id,
        start_time.strftime("%Y-%m-%d"),
        start_time.strftime("%H:%M:%S"),
        end_time.strftime("%H:%M:%S"),
        duration,
        0,  # Will be updated right below with calculated score
        topic
    ))
    conn.commit()
    new_session_id = cursor.lastrowid

    # Link active AI conversation to this completed study session
    conv_id = session.get("ai_conversation_id")
    if conv_id:
        cursor.execute("UPDATE ai_conversations SET session_id = ? WHERE id = ?", (new_session_id, conv_id))

    # --- Collect completed absence events for this session ---
    cursor.execute("""
        SELECT id, duration_seconds FROM session_absence_events
        WHERE user_id = ? AND session_id IS NULL AND absence_end IS NOT NULL
    """, (user_id,))
    absence_rows = cursor.fetchall()

    absence_count       = len(absence_rows)
    total_absence_sec   = sum(r["duration_seconds"] or 0 for r in absence_rows)
    longest_absence_sec = max((r["duration_seconds"] or 0 for r in absence_rows), default=0)

    # Backfill session_id on all absence event rows
    if absence_rows:
        absence_ids = [r["id"] for r in absence_rows]
        cursor.executemany(
            "UPDATE session_absence_events SET session_id = ? WHERE id = ?",
            [(new_session_id, aid) for aid in absence_ids]
        )

    # --- Fetch daily goal progress at session end ---
    today_str = start_time.strftime("%Y-%m-%d")
    cursor.execute("""
        SELECT COALESCE(SUM(duration), 0) AS total_today
        FROM study_sessions
        WHERE user_id = ? AND date = ?
    """, (user_id, today_str))
    today_sec_row = cursor.fetchone()
    today_sec = (today_sec_row["total_today"] if today_sec_row else 0) + duration

    cursor.execute("SELECT daily_goal FROM goals WHERE user_id = ?", (user_id,))
    goal_row = cursor.fetchone()
    daily_goal = goal_row["daily_goal"] if goal_row else 2
    goal_seconds = max(1, daily_goal * 3600)
    goal_progress_pct = min(100.0, (today_sec / goal_seconds) * 100.0)

    # --- Fetch in-session AI study conversation messages for engagement evaluation ---
    ai_msgs = []
    if conv_id:
        cursor.execute("SELECT sender, message FROM ai_messages WHERE conversation_id = ? ORDER BY id ASC", (conv_id,))
        ai_msgs = [{"sender": r["sender"], "message": r["message"]} for r in cursor.fetchall()]

    ai_eng_eval = evaluate_ai_study_interaction(messages=ai_msgs, study_topic=topic, active_duration_sec=duration)
    habit_eval = compute_habit_consistency(user_id=user_id, current_date_str=today_str)

    # --- Check prior topic mastery / retention for topic ---
    cursor.execute("SELECT avg_retention_score, last_retention_score, mastery_tier FROM topic_mastery WHERE user_id = ? AND topic = ?", (user_id, topic))
    tm_row = cursor.fetchone()
    prior_retention = float(tm_row["avg_retention_score"]) if tm_row and tm_row["avg_retention_score"] is not None else None
    mastery_tier = tm_row["mastery_tier"] if tm_row else "MEDIUM"

    # --- Check if an in-session quiz was completed ---
    cursor.execute("""
        SELECT score_pct FROM study_quizzes
        WHERE session_id = ? AND score_pct IS NOT NULL
        ORDER BY id DESC LIMIT 1
    """, (new_session_id,))
    sq_row = cursor.fetchone()
    quiz_score_val = float(sq_row["score_pct"]) if sq_row and sq_row["score_pct"] is not None else None

    # --- Compute Physical Focus Intelligence score (Legacy/Physical dimension) ---
    fi_result = compute_focus_score({
        "duration_sec":        duration,
        "absence_count":       absence_count,
        "longest_absence_sec": longest_absence_sec,
        "total_absence_sec":   total_absence_sec,
        "goal_progress_pct":   goal_progress_pct,
    })

    # --- Compute 8-Dimension Unified Focus Intelligence Score ---
    unified_input = {
        "presence_pct": fi_result["presence_pct"],
        "retention_pct": prior_retention,
        "quiz_score_pct": quiz_score_val,
        "ai_engagement_pct": ai_eng_eval["engagement_pct"],
        "duration_sec": duration,
        "completion_pct": fi_result["completion_pct"],
        "consistency_pct": habit_eval["consistency_pct"],
        "absence_count": absence_count,
        "longest_absence_sec": longest_absence_sec,
    }
    unified_result = compute_unified_focus_score(unified_input)

    # Update study_sessions with the calculated unified focus score
    cursor.execute(
        "UPDATE study_sessions SET focus_score = ? WHERE id = ?",
        (unified_result["unified_focus_score"], new_session_id)
    )

    # --- Persist the physical focus score record ---
    cursor.execute("""
        INSERT OR REPLACE INTO session_focus_scores
        (session_id, user_id, presence_pct, absence_count, longest_absence_sec,
         total_absence_sec, session_duration_sec, goal_progress_pct,
         quality_tier, estimated_score, computed_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        new_session_id,
        user_id,
        fi_result["presence_pct"],
        fi_result["absence_count"],
        fi_result["longest_absence_sec"],
        fi_result["total_absence_sec"],
        fi_result["session_duration_sec"],
        fi_result["goal_progress_pct"],
        fi_result["quality_tier"],
        fi_result["estimated_score"],
        datetime.now().isoformat()
    ))
    conn.commit()
    conn.close()

    # --- Unified Learning Loop State-Action-Reward Telemetry ---
    prior_state = get_latest_learning_state(user_id, topic)
    state_metrics = {
        "physical_presence_pct": fi_result["presence_pct"],
        "study_duration_sec": duration,
        "ai_engagement_pct": ai_eng_eval["engagement_pct"],
        "quiz_score_pct": quiz_score_val,
        "retention_score_pct": prior_retention,
        "distraction_control_pct": fi_result["distraction_control_pct"],
        "session_completion_pct": fi_result["completion_pct"],
        "consistency_score_pct": habit_eval["consistency_pct"],
        "unified_focus_score": unified_result["unified_focus_score"],
        "learning_efficiency_score": unified_result["learning_efficiency"],
        "quality_tier": unified_result["quality_tier"],
        "score_label": unified_result["score_label"],
        "explanation": unified_result["explanation"],
        "mastery_tier": mastery_tier,
        "is_passive_alert": unified_result["is_passive_alert"],
    }
    state_id = record_learning_state(user_id, new_session_id, topic, state_metrics)

    # Adaptive Policy Action Selection
    selected_act = select_adaptive_action(state_metrics)
    action_id = record_adaptive_action(
        user_id=user_id,
        topic=topic,
        session_id=new_session_id,
        state_id=state_id,
        action_type=selected_act["action_type"],
        action_payload=selected_act["action_payload"],
        reason=selected_act["reason"]
    )

    # Reward calculation if prior state existed for this user/topic
    if prior_state and prior_state.get("id") != state_id:
        r_val = calculate_adaptive_reward(prior_state, state_metrics)
        conn_r = get_db()
        c_r = conn_r.cursor()
        c_r.execute("""
            SELECT id FROM adaptive_actions
            WHERE user_id = ? AND topic = ? AND status = 'PENDING' AND id != ?
            ORDER BY id DESC LIMIT 1
        """, (user_id, topic, action_id))
        p_act = c_r.fetchone()
        conn_r.close()
        p_act_id = p_act["id"] if p_act else action_id
        record_adaptive_reward(
            user_id=user_id,
            action_id=p_act_id,
            initial_state_id=prior_state["id"],
            resulting_state_id=state_id,
            reward_value=r_val,
            evaluation_notes=f"Post-session evaluation for {topic} completed."
        )

    # FocusSense Recall Engine: Automatically schedule delayed recall task
    recall_delay = selected_act.get("action_payload", {}).get("delay_minutes", 10)
    recall_task = schedule_recall_task(user_id=user_id, session_id=new_session_id, topic=topic, delay_minutes=recall_delay)

    # Clear Focus Intelligence and topic session state
    session.pop("study_start_time", None)
    session.pop("study_topic", None)
    session.pop("fi_last_presence", None)
    session.pop("fi_absence_start", None)
    session.pop("ai_conversation_id", None)

    return jsonify({
        "success":                 True,
        "message":                 "Study session saved",
        "session_id":              new_session_id,
        "duration":                duration,
        "topic":                   topic,
        "focus_score":             unified_result["unified_focus_score"],
        "unified_focus_score":     unified_result["unified_focus_score"],
        "quality_tier":            unified_result["quality_tier"],
        "label":                   unified_result["score_label"],
        "score_label":             unified_result["score_label"],
        "is_passive_alert":        unified_result["is_passive_alert"],
        "learning_efficiency":     unified_result["learning_efficiency"],
        "presence_pct":            unified_result["factors"]["physical_presence_pct"],
        "completion_pct":          unified_result["factors"]["session_completion_pct"],
        "distraction_control_pct": unified_result["factors"]["distraction_control_pct"],
        "duration_rating_pct":     unified_result["factors"]["duration_rating_pct"],
        "ai_engagement_pct":       unified_result["factors"]["ai_engagement_pct"],
        "knowledge_retention_pct": unified_result["factors"]["knowledge_retention_pct"],
        "quiz_performance_pct":    unified_result["factors"]["quiz_performance_pct"],
        "habit_consistency_pct":   unified_result["factors"]["habit_consistency_pct"],
        "factors":                 unified_result["factors"],
        "interpretation":          fi_result["interpretation"],
        "explanation":             unified_result["explanation"],
        "disclaimer":              FI_DISCLAIMER,
        "adaptive_action":         selected_act,
        "recommendation":          selected_act.get("action_payload", {}).get("recommendation", selected_act["reason"]),
        "recall_task":             recall_task,
        "recall_task_id":          recall_task.get("task_id"),
        "recall_scheduled_for":    recall_task.get("scheduled_for"),
        "recall_delay_minutes":    recall_task.get("delay_minutes"),
    })


# -----------------------------------------------------------------------------
# FocusSense AI Workspace & Chat Endpoints
# -----------------------------------------------------------------------------

@app.route("/ai")
def ai_workspace():
    """Dedicated FocusSense AI Full-Screen Workspace."""
    if "user_id" not in session:
        flash("Please log in to access FocusSense AI", "warning")
        return redirect("/login")

    user_id = session["user_id"]
    is_session_active = "study_start_time" in session
    active_study_topic = session.get("study_topic", "")
    ai_provider_info = ai_service.get_active_provider_info()
    user_materials = get_user_materials(user_id)
    active_material_id = session.get("active_material_id")
    active_material_title = session.get("active_material_title")

    return render_template(
        "ai.html",
        fullname=session.get("fullname", "Student"),
        is_session_active=is_session_active,
        active_study_topic=active_study_topic,
        ai_provider_info=ai_provider_info,
        materials=user_materials,
        active_material_id=active_material_id,
        active_material_title=active_material_title,
    )


@app.route("/api/ai/conversations", methods=["GET"])
def ai_list_conversations():
    """List all active (unarchived) conversations grouped by date."""
    if "user_id" not in session:
        return jsonify({"success": False, "error": "Please login"}), 401

    user_id = session["user_id"]
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT c.id, c.title, c.topic, c.created_at, c.updated_at, c.session_id,
               (SELECT message FROM ai_messages WHERE conversation_id = c.id ORDER BY id DESC LIMIT 1) as last_message,
               (SELECT count(*) FROM ai_messages WHERE conversation_id = c.id) as message_count
        FROM ai_conversations c
        WHERE c.user_id = ? AND (c.is_archived IS NULL OR c.is_archived = 0)
        ORDER BY c.updated_at DESC
    """, (user_id,))
    rows = cursor.fetchall()
    conn.close()

    now = datetime.now()
    today_date = now.date()
    yesterday_date = today_date - timedelta(days=1)
    seven_days_ago = today_date - timedelta(days=7)

    grouped = {
        "today": [],
        "yesterday": [],
        "previous_7_days": [],
        "older": []
    }
    all_convs = []

    for r in rows:
        conv_item = {
            "id": r["id"],
            "title": r["title"] or "New Study Chat",
            "topic": r["topic"] or "General Study",
            "created_at": r["created_at"],
            "updated_at": r["updated_at"],
            "session_id": r["session_id"],
            "last_message": r["last_message"] or "",
            "message_count": r["message_count"] or 0
        }
        all_convs.append(conv_item)

        try:
            conv_dt = datetime.fromisoformat(r["updated_at"]).date()
            if conv_dt == today_date:
                grouped["today"].append(conv_item)
            elif conv_dt == yesterday_date:
                grouped["yesterday"].append(conv_item)
            elif conv_dt >= seven_days_ago:
                grouped["previous_7_days"].append(conv_item)
            else:
                grouped["older"].append(conv_item)
        except Exception:
            grouped["older"].append(conv_item)

    return jsonify({
        "success": True,
        "conversations": all_convs,
        "grouped": grouped
    })


@app.route("/api/ai/conversations/new", methods=["POST"])
def ai_new_conversation():
    """Start a fresh conversation thread."""
    if "user_id" not in session:
        return jsonify({"success": False, "error": "Please login"}), 401

    user_id = session["user_id"]
    data = request.get_json(silent=True) or {}
    topic = (data.get("topic") or session.get("study_topic") or "General Study").strip()
    title = (data.get("title") or "New Study Chat").strip()
    now_iso = datetime.now().isoformat()

    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO ai_conversations (user_id, topic, title, created_at, updated_at, is_archived)
        VALUES (?, ?, ?, ?, ?, 0)
    """, (user_id, topic, title, now_iso, now_iso))
    conv_id = cursor.lastrowid
    conn.commit()
    conn.close()

    session["ai_conversation_id"] = conv_id

    return jsonify({
        "success": True,
        "conversation": {
            "id": conv_id,
            "title": title,
            "topic": topic,
            "created_at": now_iso,
            "updated_at": now_iso
        }
    })


@app.route("/api/ai/conversations/<int:conv_id>", methods=["GET"])
def ai_get_conversation(conv_id):
    """Retrieve full history for a specific conversation."""
    if "user_id" not in session:
        return jsonify({"success": False, "error": "Please login"}), 401

    user_id = session["user_id"]
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT id, title, topic, created_at, updated_at, session_id
        FROM ai_conversations
        WHERE id = ? AND user_id = ?
    """, (conv_id, user_id))
    conv_row = cursor.fetchone()

    if not conv_row:
        conn.close()
        return jsonify({"success": False, "error": "Conversation not found"}), 404

    cursor.execute("""
        SELECT id, sender, message, timestamp
        FROM ai_messages
        WHERE conversation_id = ?
        ORDER BY id ASC
    """, (conv_id,))
    msg_rows = cursor.fetchall()
    conn.close()

    session["ai_conversation_id"] = conv_id

    messages = [
        {
            "id": r["id"],
            "sender": r["sender"],
            "message": r["message"],
            "timestamp": r["timestamp"]
        }
        for r in msg_rows
    ]

    return jsonify({
        "success": True,
        "conversation": {
            "id": conv_row["id"],
            "title": conv_row["title"] or "Study Chat",
            "topic": conv_row["topic"] or "General Study",
            "created_at": conv_row["created_at"],
            "updated_at": conv_row["updated_at"],
            "session_id": conv_row["session_id"]
        },
        "messages": messages,
        "provider_info": ai_service.get_active_provider_info()
    })


@app.route("/api/ai/conversations/<int:conv_id>/rename", methods=["POST"])
def ai_rename_conversation(conv_id):
    """Rename a conversation."""
    if "user_id" not in session:
        return jsonify({"success": False, "error": "Please login"}), 401

    user_id = session["user_id"]
    data = request.get_json(silent=True) or {}
    new_title = (data.get("title") or "").strip()

    if not new_title:
        return jsonify({"success": False, "error": "Title cannot be empty"}), 400

    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("""
        UPDATE ai_conversations
        SET title = ?, updated_at = ?
        WHERE id = ? AND user_id = ?
    """, (new_title, datetime.now().isoformat(), conv_id, user_id))
    conn.commit()
    conn.close()

    return jsonify({"success": True, "title": new_title})


@app.route("/api/ai/conversations/<int:conv_id>/archive", methods=["POST"])
def ai_archive_conversation(conv_id):
    """Archive a conversation (hidden from main list)."""
    if "user_id" not in session:
        return jsonify({"success": False, "error": "Please login"}), 401

    user_id = session["user_id"]
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("""
        UPDATE ai_conversations
        SET is_archived = 1, updated_at = ?
        WHERE id = ? AND user_id = ?
    """, (datetime.now().isoformat(), conv_id, user_id))
    conn.commit()
    conn.close()

    return jsonify({"success": True, "message": "Conversation archived"})


@app.route("/api/ai/conversations/<int:conv_id>", methods=["DELETE"])
def ai_delete_conversation(conv_id):
    """Delete a conversation and its messages."""
    if "user_id" not in session:
        return jsonify({"success": False, "error": "Please login"}), 401

    user_id = session["user_id"]
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM ai_messages WHERE conversation_id = ? AND conversation_id IN (SELECT id FROM ai_conversations WHERE user_id = ?)", (conv_id, user_id))
    cursor.execute("DELETE FROM ai_conversations WHERE id = ? AND user_id = ?", (conv_id, user_id))
    conn.commit()
    conn.close()

    if session.get("ai_conversation_id") == conv_id:
        session.pop("ai_conversation_id", None)

    return jsonify({"success": True, "message": "Conversation deleted"})


@app.route("/api/ai/conversations/search", methods=["GET"])
def ai_search_conversations():
    """Full-text search across titles, topics, and message content."""
    if "user_id" not in session:
        return jsonify({"success": False, "error": "Please login"}), 401

    user_id = session["user_id"]
    query = (request.args.get("q") or "").strip()
    if not query:
        return ai_list_conversations()

    search_pattern = f"%{query}%"
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT DISTINCT c.id, c.title, c.topic, c.created_at, c.updated_at, c.session_id,
               (SELECT message FROM ai_messages WHERE conversation_id = c.id ORDER BY id DESC LIMIT 1) as last_message,
               (SELECT count(*) FROM ai_messages WHERE conversation_id = c.id) as message_count
        FROM ai_conversations c
        LEFT JOIN ai_messages m ON m.conversation_id = c.id
        WHERE c.user_id = ? AND (c.is_archived IS NULL OR c.is_archived = 0)
          AND (c.title LIKE ? OR c.topic LIKE ? OR m.message LIKE ?)
        ORDER BY c.updated_at DESC
        LIMIT 30
    """, (user_id, search_pattern, search_pattern, search_pattern))
    rows = cursor.fetchall()
    conn.close()

    results = [
        {
            "id": r["id"],
            "title": r["title"] or "Study Chat",
            "topic": r["topic"] or "General Study",
            "created_at": r["created_at"],
            "updated_at": r["updated_at"],
            "session_id": r["session_id"],
            "last_message": r["last_message"] or "",
            "message_count": r["message_count"] or 0
        }
        for r in rows
    ]

    return jsonify({
        "success": True,
        "query": query,
        "conversations": results
    })


@app.route("/api/ai/chat", methods=["POST"])
def ai_chat():
    """FocusSense AI multi-turn dialogue endpoint with automatic title generation."""
    if "user_id" not in session:
        return jsonify({"error": "Please login", "success": False}), 401

    user_id = session["user_id"]
    student_name = session.get("fullname") or session.get("username") or "Student"
    data = request.get_json(silent=True) or {}
    prompt = (data.get("message") or "").strip()
    conv_id = data.get("conversation_id") or session.get("ai_conversation_id")
    override_topic = (data.get("topic") or "").strip()

    if not prompt:
        return jsonify({"error": "Message cannot be empty", "success": False}), 400

    conn = get_db()
    cursor = conn.cursor()
    now_iso = datetime.now().isoformat()
    now_ts = datetime.now().strftime("%H:%M")

    # Find or create conversation
    current_title = "New Study Chat"
    study_topic = override_topic or session.get("study_topic") or "General Study"

    if conv_id:
        cursor.execute("SELECT id, title, topic FROM ai_conversations WHERE id = ? AND user_id = ?", (conv_id, user_id))
        conv_row = cursor.fetchone()
        if conv_row:
            current_title = conv_row["title"]
            study_topic = override_topic or conv_row["topic"] or study_topic
        else:
            conv_id = None

    if not conv_id:
        cursor.execute("""
            INSERT INTO ai_conversations (user_id, topic, title, created_at, updated_at, is_archived)
            VALUES (?, ?, ?, ?, ?, 0)
        """, (user_id, study_topic, current_title, now_iso, now_iso))
        conv_id = cursor.lastrowid
        session["ai_conversation_id"] = conv_id
        conn.commit()

    # Fetch recent history for multi-turn context
    cursor.execute("""
        SELECT sender, message, timestamp FROM ai_messages
        WHERE conversation_id = ?
        ORDER BY id ASC LIMIT 20
    """, (conv_id,))
    history_rows = cursor.fetchall()
    history = [{"sender": r["sender"], "message": r["message"]} for r in history_rows]

    # Save user message
    cursor.execute("""
        INSERT INTO ai_messages (conversation_id, sender, message, timestamp)
        VALUES (?, 'user', ?, ?)
    """, (conv_id, prompt, now_ts))
    conn.commit()

    # Determine if question is substantive (meaningful study question vs trivial greeting)
    is_meaningful = len(prompt.strip()) >= 8 and prompt.strip().lower() not in ["hi", "hello", "hey", "ok", "k", "thanks", "thank you", "bye"]

    # Log/update AI study telemetry (anti-gaming tracking)
    active_session_id = session.get("study_session_id")
    cursor.execute("""
        SELECT id, question_count, meaningful_query_count, active_duration_sec
        FROM ai_study_sessions
        WHERE user_id = ? AND conversation_id = ?
        ORDER BY id DESC LIMIT 1
    """, (user_id, conv_id))
    sess_row = cursor.fetchone()

    if sess_row:
        m_inc = 1 if is_meaningful else 0
        cursor.execute("""
            UPDATE ai_study_sessions
            SET question_count = question_count + 1,
                meaningful_query_count = meaningful_query_count + ?,
                ended_at = ?
            WHERE id = ?
        """, (m_inc, now_iso, sess_row["id"]))
    else:
        cursor.execute("""
            INSERT INTO ai_study_sessions (user_id, session_id, conversation_id, topic, started_at, ended_at, active_duration_sec, question_count, meaningful_query_count)
            VALUES (?, ?, ?, ?, ?, ?, 60, 1, ?)
        """, (user_id, active_session_id, conv_id, study_topic, now_iso, now_iso, 1 if is_meaningful else 0))
    conn.commit()

    # Generate title if this is the first turn or title is default
    title_updated = False
    if current_title in ["New Study Chat", "New Chat", ""]:
        new_generated_title = ai_service.generate_title(prompt, study_topic)
        if new_generated_title:
            current_title = new_generated_title
            title_updated = True
            cursor.execute("UPDATE ai_conversations SET title = ? WHERE id = ?", (current_title, conv_id))
            conn.commit()

    # Ground with Study Material if selected or passed
    material_id = data.get("material_id")
    if material_id is None:
        material_id = session.get("active_material_id")
    material_context = None
    if material_id:
        try:
            material_id = int(material_id)
            material_context = build_material_grounded_context(user_id, prompt, material_id)
            if material_context and material_context.get("found"):
                if not override_topic and material_context.get("topic"):
                    study_topic = material_context.get("topic")
        except (ValueError, TypeError):
            material_context = None

    # Generate AI response
    result = ai_service.generate_chat_reply(
        prompt=prompt,
        conversation_history=history,
        study_topic=study_topic,
        student_name=student_name,
        learning_context=None,
        material_context=material_context,
    )

    reply_text = result.get("reply") or "I'm focusing on your study topic. Let me know what you'd like to explore next!"
    ai_ts = datetime.now().strftime("%H:%M")

    # Save assistant turn
    cursor.execute("""
        INSERT INTO ai_messages (conversation_id, sender, message, timestamp)
        VALUES (?, 'assistant', ?, ?)
    """, (conv_id, reply_text, ai_ts))

    cursor.execute("UPDATE ai_conversations SET updated_at = ? WHERE id = ?", (now_iso, conv_id))
    conn.commit()
    conn.close()

    return jsonify({
        "success": True,
        "reply": reply_text,
        "timestamp": ai_ts,
        "topic": study_topic,
        "conversation_id": conv_id,
        "title": current_title,
        "title_updated": title_updated,
        "provider": result.get("provider", "Cohere AI"),
        "model": result.get("model", "command-a-plus-05-2026"),
        "material_id": material_id if (material_context and material_context.get("found")) else None,
        "material_title": material_context.get("material_title") if material_context else None,
        "grounded": bool(material_context and material_context.get("is_relevant")),
    })


@app.route("/api/ai/track-session", methods=["POST"])
def ai_track_session():
    """Records AI workspace heartbeat while enforcing anti-gaming limits."""
    if "user_id" not in session:
        return jsonify({"success": False, "error": "Please login"}), 401

    user_id = session["user_id"]
    data = request.get_json(silent=True) or {}
    conv_id = data.get("conversation_id")
    elapsed_sec = min(60, max(0, int(data.get("duration_sec") or 30)))

    if not conv_id:
        return jsonify({"success": True, "tracked": False})

    conn = get_db()
    cursor = conn.cursor()
    now_iso = datetime.now().isoformat()

    cursor.execute("""
        SELECT id, meaningful_query_count, active_duration_sec
        FROM ai_study_sessions
        WHERE user_id = ? AND conversation_id = ?
        ORDER BY id DESC LIMIT 1
    """, (user_id, conv_id))
    row = cursor.fetchone()

    if row:
        meaningful_q = row["meaningful_query_count"]
        # Anti-gaming rule: Max 3 mins active focus credit per meaningful question asked
        max_allowed = max(60, meaningful_q * 180)
        current_dur = row["active_duration_sec"]
        if current_dur < max_allowed:
            new_dur = min(max_allowed, current_dur + elapsed_sec)
            cursor.execute("UPDATE ai_study_sessions SET active_duration_sec = ?, ended_at = ? WHERE id = ?", (new_dur, now_iso, row["id"]))
            conn.commit()

    conn.close()
    return jsonify({"success": True, "tracked": True})


@app.route("/api/ai/history", methods=["GET"])
def ai_history():
    """Retrieve message history for active or latest conversation."""
    if "user_id" not in session:
        return jsonify({"messages": [], "topic": ""}), 401

    user_id = session["user_id"]
    conn = get_db()
    cursor = conn.cursor()

    conv_id = session.get("ai_conversation_id")
    topic = session.get("study_topic", "")

    if not conv_id:
        cursor.execute("""
            SELECT id, topic FROM ai_conversations
            WHERE user_id = ? AND (is_archived IS NULL OR is_archived = 0)
            ORDER BY updated_at DESC LIMIT 1
        """, (user_id,))
        latest_conv = cursor.fetchone()
        if latest_conv:
            conv_id = latest_conv["id"]
            if not topic:
                topic = latest_conv["topic"]

    msgs = []
    if conv_id:
        cursor.execute("""
            SELECT id, sender, message, timestamp FROM ai_messages
            WHERE conversation_id = ?
            ORDER BY id ASC LIMIT 50
        """, (conv_id,))
        msgs = [
            {
                "id": r["id"],
                "sender": r["sender"],
                "message": r["message"],
                "timestamp": r["timestamp"]
            }
            for r in cursor.fetchall()
        ]

    conn.close()

    return jsonify({
        "messages": msgs,
        "topic": topic,
        "active_topic": session.get("study_topic", topic),
        "conversation_id": conv_id,
        "provider_info": ai_service.get_active_provider_info()
    })


@app.route("/api/ai/clear", methods=["POST"])
def ai_clear():
    """Reset active conversation and start a fresh chat thread."""
    return ai_new_conversation()


# -----------------------------------------------------------------------------
# FocusSense AI Study Materials & Knowledge Base Endpoints
# -----------------------------------------------------------------------------

@app.route("/api/materials/upload", methods=["POST"])
def materials_upload():
    """Upload, extract, chunk, and index a study material (PDF, TXT, DOCX)."""
    if "user_id" not in session:
        return jsonify({"success": False, "error": "Please login"}), 401

    user_id = session["user_id"]
    if "file" not in request.files:
        return jsonify({"success": False, "error": "No file part in request."}), 400

    file_storage = request.files["file"]
    if not file_storage or not file_storage.filename:
        return jsonify({"success": False, "error": "No file selected."}), 400

    title = request.form.get("title", "").strip()
    subject = request.form.get("subject", "").strip()
    topic = request.form.get("topic", "").strip()

    result = save_and_process_material(
        user_id=user_id,
        file_storage=file_storage,
        title=title,
        subject=subject,
        topic=topic
    )

    if not result.get("success"):
        return jsonify(result), 400

    # Auto-select the newly uploaded material
    session["active_material_id"] = result["material_id"]
    session["active_material_title"] = result["title"]

    return jsonify(result)


@app.route("/api/materials", methods=["GET"])
def materials_list():
    """List all study materials for current authenticated student."""
    if "user_id" not in session:
        return jsonify({"success": False, "error": "Please login"}), 401

    user_id = session["user_id"]
    materials = get_user_materials(user_id)
    return jsonify({
        "success": True,
        "materials": materials,
        "active_material_id": session.get("active_material_id"),
        "active_material_title": session.get("active_material_title")
    })


@app.route("/api/materials/<int:material_id>", methods=["GET"])
def materials_get(material_id):
    """Retrieve metadata and status of a single study material."""
    if "user_id" not in session:
        return jsonify({"success": False, "error": "Please login"}), 401

    user_id = session["user_id"]
    material = get_material_by_id(material_id, user_id=user_id)
    if not material:
        return jsonify({"success": False, "error": "Study material not found."}), 404

    return jsonify({"success": True, "material": material})


@app.route("/api/materials/<int:material_id>", methods=["DELETE"])
def materials_delete(material_id):
    """Delete a study material, its indexed chunks, and file from disk."""
    if "user_id" not in session:
        return jsonify({"success": False, "error": "Please login"}), 401

    user_id = session["user_id"]
    result = delete_user_material(material_id, user_id=user_id)
    if not result.get("success"):
        return jsonify(result), 400

    if session.get("active_material_id") == material_id:
        session.pop("active_material_id", None)
        session.pop("active_material_title", None)

    return jsonify(result)


@app.route("/api/materials/<int:material_id>/select", methods=["POST"])
def materials_select(material_id):
    """Select a study material as the active grounding source for chat & exam."""
    if "user_id" not in session:
        return jsonify({"success": False, "error": "Please login"}), 401

    user_id = session["user_id"]
    material = get_material_by_id(material_id, user_id=user_id)
    if not material:
        return jsonify({"success": False, "error": "Study material not found."}), 404

    session["active_material_id"] = material["id"]
    session["active_material_title"] = material["title"]

    return jsonify({
        "success": True,
        "message": f"Active study material set to '{material['title']}'",
        "active_material": material
    })


@app.route("/api/materials/clear-selection", methods=["POST"])
def materials_clear_selection():
    """Clear active study material selection."""
    if "user_id" not in session:
        return jsonify({"success": False, "error": "Please login"}), 401

    session.pop("active_material_id", None)
    session.pop("active_material_title", None)

    return jsonify({"success": True, "message": "Study material selection cleared."})


@app.route("/api/materials/<int:material_id>/generate-quiz", methods=["POST"])
def materials_generate_quiz(material_id):
    """Generate a knowledge verification quiz grounded in a study material."""
    if "user_id" not in session:
        return jsonify({"success": False, "error": "Please login"}), 401

    user_id = session["user_id"]
    data = request.get_json(silent=True) or {}
    q_count = int(data.get("question_count") or 4)

    quiz_data = create_or_get_material_quiz(user_id, material_id, question_count=q_count)
    if not quiz_data.get("success", True):
        return jsonify(quiz_data), 400

    return jsonify(quiz_data)


# -----------------------------------------------------------------------------
# FocusSense AI Post-Study Examination Endpoints (Phase 2)
# -----------------------------------------------------------------------------

@app.route("/api/quiz/generate", methods=["POST"])
def quiz_generate():
    """Generate or retrieve grounded learning verification questions."""
    if "user_id" not in session:
        return jsonify({"success": False, "error": "Please login"}), 401

    user_id = session["user_id"]
    data = request.get_json(silent=True) or {}
    session_id = data.get("session_id")
    topic = data.get("topic")

    # If topic not explicitly provided, find from session or latest study session
    if not topic:
        topic = session.get("study_topic")
    if not topic:
        conn = get_db()
        cursor = conn.cursor()
        if session_id:
            cursor.execute("SELECT topic FROM study_sessions WHERE id = ?", (session_id,))
            s_row = cursor.fetchone()
            if s_row and s_row["topic"]:
                topic = s_row["topic"]
        if not topic:
            cursor.execute("SELECT topic FROM study_sessions WHERE user_id = ? AND topic IS NOT NULL ORDER BY id DESC LIMIT 1", (user_id,))
            s_row = cursor.fetchone()
            if s_row and s_row["topic"]:
                topic = s_row["topic"]
        conn.close()

    topic = topic or "General Academic Study"
    quiz_data = create_or_get_session_quiz(user_id=user_id, session_id=session_id, topic=topic, question_count=3)
    return jsonify({
        "success": True,
        "quiz": quiz_data
    })


@app.route("/api/quiz/submit-answer", methods=["POST"])
def quiz_submit_answer():
    """Grade an individual question answer turn-by-turn."""
    if "user_id" not in session:
        return jsonify({"success": False, "error": "Please login"}), 401

    user_id = session["user_id"]
    data = request.get_json(silent=True) or {}
    question_id = data.get("question_id")
    student_answer = data.get("student_answer", "")
    question_text = data.get("question_text")
    sub_concept = data.get("sub_concept")
    sample_answer = data.get("sample_answer")
    points_possible = data.get("points_possible", 1.0)
    topic = data.get("topic") or session.get("study_topic")
    quiz_id = data.get("quiz_id")

    if not question_id:
        return jsonify({"success": False, "error": "Missing question_id"}), 400

    result = evaluate_student_answer(
        question_id=question_id,
        student_answer=student_answer,
        user_id=user_id,
        question_text=question_text,
        sub_concept=sub_concept,
        sample_answer=sample_answer,
        points_possible=float(points_possible or 1.0),
        topic=topic,
        quiz_id=quiz_id
    )
    return jsonify(result)


@app.route("/api/quiz/complete", methods=["POST"])
def quiz_complete():
    """Finalize examination, compute learning verification score, and summarize weak concepts."""
    if "user_id" not in session:
        return jsonify({"success": False, "error": "Please login"}), 401

    user_id = session["user_id"]
    data = request.get_json(silent=True) or {}
    quiz_id = data.get("quiz_id")
    duration_sec = data.get("duration")

    if not quiz_id:
        return jsonify({"success": False, "error": "Missing quiz_id"}), 400

    result = complete_quiz(quiz_id=quiz_id, user_id=user_id, session_duration_sec=duration_sec)
    return jsonify(result)


@app.route("/api/quiz/latest", methods=["GET"])
def quiz_latest():
    """Retrieve latest learning verification metrics and concept mastery summary."""
    if "user_id" not in session:
        return jsonify({"success": False, "error": "Please login"}), 401

    user_id = session["user_id"]
    overview = get_latest_quiz_overview(user_id=user_id)
    return jsonify({
        "success": True,
        "overview": overview
    })


# -----------------------------------------------------------------------------
# FocusSense Recall Engine Endpoints (Phase 3)
# -----------------------------------------------------------------------------

@app.route("/api/recall/pending", methods=["GET"])
def recall_pending():
    """Retrieve pending and ready recall tasks for the logged-in student."""
    if "user_id" not in session:
        return jsonify({"success": False, "error": "Please login"}), 401

    user_id = session["user_id"]
    data = get_pending_and_due_tasks(user_id)
    return jsonify({
        "success": True,
        **data
    })


@app.route("/api/recall/start", methods=["POST"])
def recall_start():
    """Start or resume a delayed recall test for a specific recall task."""
    if "user_id" not in session:
        return jsonify({"success": False, "error": "Please login"}), 401

    user_id = session["user_id"]
    data = request.get_json(silent=True) or {}
    task_id = data.get("task_id")
    question_count = int(data.get("question_count", 3))

    if not task_id:
        # Fallback to the latest ready/pending task if task_id was not explicitly passed
        pending_data = get_pending_and_due_tasks(user_id)
        if pending_data.get("primary_task"):
            task_id = pending_data["primary_task"]["id"]
        else:
            return jsonify({"success": False, "error": "No active recall task found"}), 404

    result = start_recall_test(task_id=task_id, user_id=user_id, question_count=question_count)
    if result.get("success"):
        # Store in signed Flask session cookie for serverless multi-instance persistence
        session["active_recall_quiz"] = {
            "task_id": task_id,
            "quiz_id": result.get("quiz_id"),
            "topic": result.get("topic"),
            "questions": {
                str(q["id"]): q for q in result.get("questions", [])
            }
        }
        session["active_recall_answers"] = {}

    return jsonify(result)


@app.route("/api/recall/submit-answer", methods=["POST"])
def recall_submit_answer():
    """Grade an individual recall test question turn-by-turn."""
    if "user_id" not in session:
        return jsonify({"success": False, "error": "Please login"}), 401

    user_id = session["user_id"]
    data = request.get_json(silent=True) or {}
    question_id = data.get("question_id")
    student_answer = data.get("student_answer", "")

    if not question_id:
        return jsonify({"success": False, "error": "Missing question_id"}), 400

    # Retrieve metadata from request or fallback to signed session cookie
    sess_quiz = session.get("active_recall_quiz") or {}
    sess_q = sess_quiz.get("questions", {}).get(str(question_id)) or {}

    question_text = data.get("question_text") or sess_q.get("question_text")
    sub_concept = data.get("sub_concept") or sess_q.get("sub_concept")
    sample_answer = data.get("sample_answer") or sess_q.get("sample_answer")
    points_possible = data.get("points_possible") or sess_q.get("points_possible") or 1.0
    topic = data.get("topic") or sess_quiz.get("topic") or session.get("study_topic")
    task_id = data.get("task_id") or sess_quiz.get("task_id")
    quiz_id = sess_quiz.get("quiz_id")

    result = submit_recall_answer(
        question_id=question_id,
        student_answer=student_answer,
        user_id=user_id,
        question_text=question_text,
        sub_concept=sub_concept,
        sample_answer=sample_answer,
        points_possible=float(points_possible or 1.0),
        topic=topic,
        task_id=task_id,
        quiz_id=quiz_id
    )

    if result.get("success"):
        answers = session.get("active_recall_answers") or {}
        answers[str(question_id)] = {
            "question_id": question_id,
            "student_answer": student_answer,
            "sub_concept": result.get("sub_concept") or sub_concept,
            "points_possible": float(points_possible or 1.0),
            "score_earned": result.get("score_earned", 0.0),
            "status": result.get("status", "incorrect"),
            "feedback": result.get("feedback") or result.get("ai_feedback"),
        }
        session["active_recall_answers"] = answers

    return jsonify(result)


@app.route("/api/recall/complete", methods=["POST"])
def recall_complete():
    """Finalize recall test, compute retention score %, and summarize weak concepts."""
    if "user_id" not in session:
        return jsonify({"success": False, "error": "Please login"}), 401

    user_id = session["user_id"]
    data = request.get_json(silent=True) or {}
    task_id = data.get("task_id") or session.get("active_recall_quiz", {}).get("task_id")
    sess_answers = session.get("active_recall_answers") or {}
    topic = data.get("topic") or session.get("active_recall_quiz", {}).get("topic") or session.get("study_topic")

    if not task_id and not sess_answers:
        return jsonify({"success": False, "error": "Missing task_id"}), 400

    result = finalize_recall_test(task_id=task_id, user_id=user_id, session_answers=sess_answers, topic=topic)
    return jsonify(result)


@app.route("/api/recall/history", methods=["GET"])
def recall_history():
    """Retrieve historical completed recall tests and retention scores."""
    if "user_id" not in session:
        return jsonify({"success": False, "error": "Please login"}), 401

    user_id = session["user_id"]
    history = get_recall_history(user_id)
    return jsonify({
        "success": True,
        "history": history
    })


@app.route("/api/recall/weak-concepts", methods=["GET"])
def recall_weak_concepts():
    """Retrieve list of concepts needing spaced revision (<70% mastery)."""
    if "user_id" not in session:
        return jsonify({"success": False, "error": "Please login"}), 401

    user_id = session["user_id"]
    weak_list = get_weak_topics_summary(user_id)
    return jsonify({
        "success": True,
        "weak_concepts": weak_list
    })


@app.route("/api/recall/schedule-custom", methods=["POST"])
def recall_schedule_custom():
    """Manually schedule a recall task for any study topic."""
    if "user_id" not in session:
        return jsonify({"success": False, "error": "Please login"}), 401

    user_id = session["user_id"]
    data = request.get_json(silent=True) or {}
    topic = (data.get("topic") or "General Study").strip()
    delay_minutes = int(data.get("delay_minutes", 0))

    # If linked to recent session
    session_id = data.get("session_id")
    if not session_id:
        conn = get_db()
        cursor = conn.cursor()
        cursor.execute("SELECT id FROM study_sessions WHERE user_id = ? ORDER BY id DESC LIMIT 1", (user_id,))
        srow = cursor.fetchone()
        session_id = srow["id"] if srow else 0
        conn.close()

    result = schedule_recall_task(user_id=user_id, session_id=session_id, topic=topic, delay_minutes=delay_minutes)
    return jsonify(result)


# -----------------------------------------------------------------------------
# FocusSense Learning Intelligence API Endpoints
# -----------------------------------------------------------------------------

@app.route("/api/learning/profile", methods=["GET"])
def learning_profile():
    """Retrieve full student learning intelligence profile and retention breakdown."""
    if "user_id" not in session:
        return jsonify({"success": False, "error": "Please login"}), 401
    
    user_id = session["user_id"]
    profile_data = get_student_learning_profile(user_id)
    return jsonify({
        "success": True,
        "profile": profile_data
    })


@app.route("/api/learning/weak-topics", methods=["GET"])
def learning_weak_topics():
    """Retrieve weak topics (<60% mastery) requiring spaced repetition review."""
    if "user_id" not in session:
        return jsonify({"success": False, "error": "Please login"}), 401

    user_id = session["user_id"]
    weak_topics = get_weak_topics(user_id)
    return jsonify({
        "success": True,
        "weak_topics": weak_topics
    })


@app.route("/api/learning/topic-mastery", methods=["GET"])
def learning_topic_mastery():
    """Retrieve all topic-level mastery tiers (Weak, Medium, Strong) for the student."""
    if "user_id" not in session:
        return jsonify({"success": False, "error": "Please login"}), 401

    user_id = session["user_id"]
    mastery_list = get_topic_mastery_list(user_id)
    return jsonify({
        "success": True,
        "topic_mastery": mastery_list
    })


@app.route("/api/learning/unified-loop", methods=["GET"])
def learning_unified_loop():
    """Retrieve unified focus intelligence loop state, active adaptive recommendation, and telemetry."""
    if "user_id" not in session:
        return jsonify({"success": False, "error": "Please login"}), 401

    user_id = session["user_id"]
    profile = get_unified_learning_profile(user_id)
    return jsonify({
        "success": True,
        "unified_profile": profile
    })


@app.route("/api/learning/efficiency", methods=["GET"])
def learning_efficiency_breakdown():
    """Retrieve learning efficiency index and historical state transitions."""
    if "user_id" not in session:
        return jsonify({"success": False, "error": "Please login"}), 401

    user_id = session["user_id"]
    conn = get_db()
    c = conn.cursor()
    c.execute("""
        SELECT topic, physical_presence_pct, study_duration_sec, ai_engagement_pct,
               quiz_score_pct, retention_score_pct, unified_focus_score, learning_efficiency_score,
               quality_tier, score_label, created_at
        FROM learning_states
        WHERE user_id = ?
        ORDER BY id DESC LIMIT 10
    """, (user_id,))
    rows = c.fetchall()
    conn.close()

    history = [dict(r) for r in rows]
    return jsonify({
        "success": True,
        "efficiency_history": history
    })


# -----------------------------------------------------------------------------
# FocusSense AI Exam & Assessment Center Endpoints (Phase 5)
# -----------------------------------------------------------------------------

@app.route("/assessment")
def assessment_page():
    """Dedicated FocusSense AI Full-Screen Exam & Assessment Center."""
    if "user_id" not in session:
        flash("Please log in to access the Assessment Center", "warning")
        return redirect("/login")

    user_id = session["user_id"]
    fullname = session.get("fullname", "Student")
    user_materials = get_user_materials(user_id)
    history = get_user_assessment_history(user_id, limit=10)
    topic_mastery = get_topic_mastery_list(user_id)
    active_material_id = session.get("active_material_id")
    active_study_topic = session.get("study_topic", "")

    return render_template(
        "assessment.html",
        fullname=fullname,
        materials=user_materials,
        history=history,
        topic_mastery=topic_mastery,
        active_material_id=active_material_id,
        active_study_topic=active_study_topic
    )


@app.route("/api/assessment/create", methods=["POST"])
def api_assessment_create():
    """Create and start a new assessment attempt."""
    if "user_id" not in session:
        return jsonify({"success": False, "error": "Please login"}), 401

    user_id = session["user_id"]
    data = request.get_json(silent=True) or {}

    mode = data.get("mode", "PRACTICE")
    topic = data.get("topic") or session.get("study_topic") or "General Academic Studies"
    material_id = data.get("material_id")
    sub_topic = data.get("sub_topic")
    difficulty = data.get("difficulty")
    question_count = data.get("question_count")
    time_limit_sec = data.get("time_limit_sec")

    # If material_id is provided as string, convert to int
    if material_id:
        try:
            material_id = int(material_id)
        except (ValueError, TypeError):
            material_id = None

    result = create_assessment(
        user_id=user_id,
        mode=mode,
        topic=topic,
        material_id=material_id,
        sub_topic=sub_topic,
        difficulty=difficulty,
        question_count=question_count,
        time_limit_sec=time_limit_sec
    )

    status_code = 200 if result.get("success") else 400
    return jsonify(result), status_code


@app.route("/api/assessment/<int:attempt_id>", methods=["GET"])
def api_assessment_get(attempt_id):
    """Retrieve active assessment state with server-authoritative timer."""
    if "user_id" not in session:
        return jsonify({"success": False, "error": "Please login"}), 401

    user_id = session["user_id"]
    result = get_active_assessment(user_id=user_id, attempt_id=attempt_id)
    status_code = 200 if result.get("success") else 404
    return jsonify(result), status_code


@app.route("/api/assessment/<int:attempt_id>/submit-answer", methods=["POST"])
def api_assessment_submit_answer(attempt_id):
    """Submit and server-side grade an answer to a question in the assessment."""
    if "user_id" not in session:
        return jsonify({"success": False, "error": "Please login"}), 401

    user_id = session["user_id"]
    data = request.get_json(silent=True) or {}
    question_id = data.get("question_id")
    student_answer = data.get("student_answer", "")

    if not question_id:
        return jsonify({"success": False, "error": "question_id is required."}), 400

    result = submit_assessment_answer(
        user_id=user_id,
        attempt_id=attempt_id,
        question_id=int(question_id),
        student_answer=student_answer,
        question_text=data.get("question_text"),
        sub_concept=data.get("sub_concept"),
        sample_answer=data.get("sample_answer"),
        points_possible=float(data.get("points_possible") or 1.0),
        topic=data.get("topic") or session.get("study_topic")
    )

    status_code = 200 if result.get("success") else 400
    return jsonify(result), status_code


@app.route("/api/assessment/<int:attempt_id>/finalize", methods=["POST"])
def api_assessment_finalize(attempt_id):
    """Finalize assessment attempt, compute dimensional analytics, and trigger adaptive loop."""
    if "user_id" not in session:
        return jsonify({"success": False, "error": "Please login"}), 401

    user_id = session["user_id"]
    result = finalize_assessment(user_id=user_id, attempt_id=attempt_id)
    status_code = 200 if result.get("success") else 400
    return jsonify(result), status_code


@app.route("/api/assessment/<int:attempt_id>/results", methods=["GET"])
def api_assessment_results(attempt_id):
    """Retrieve full post-exam report, dimensional scores, and revealed answer explanations."""
    if "user_id" not in session:
        return jsonify({"success": False, "error": "Please login"}), 401

    user_id = session["user_id"]
    result = get_assessment_results(user_id=user_id, attempt_id=attempt_id)
    status_code = 200 if result.get("success") else 404
    return jsonify(result), status_code


@app.route("/api/assessment/history", methods=["GET"])
def api_assessment_history():
    """Retrieve student's persistent assessment attempt history."""
    if "user_id" not in session:
        return jsonify({"success": False, "error": "Please login"}), 401

    user_id = session["user_id"]
    limit = int(request.args.get("limit", 25))
    history = get_user_assessment_history(user_id=user_id, limit=limit)
    return jsonify({"success": True, "history": history})


@app.route("/api/assessment/explain-mistake", methods=["POST"])
def api_assessment_explain_mistake():
    """Generate Socratic AI remediation for an incorrect assessment question."""
    if "user_id" not in session:
        return jsonify({"success": False, "error": "Please login"}), 401

    user_id = session["user_id"]
    data = request.get_json(silent=True) or {}
    attempt_id = data.get("attempt_id")
    question_id = data.get("question_id")

    if not attempt_id or not question_id:
        return jsonify({"success": False, "error": "attempt_id and question_id are required."}), 400

    result = engine_explain_mistake(
        user_id=user_id,
        attempt_id=int(attempt_id),
        question_id=int(question_id)
    )

    status_code = 200 if result.get("success") else 400
    return jsonify(result), status_code


# -----------------------------------------------------------------------------
# FocusSense AI Personal Learning Coach Endpoints (8-Pillar Synthesis)
# -----------------------------------------------------------------------------

@app.route("/coach")
def coach_page():
    """Dedicated FocusSense AI Full-Screen Personal Learning Coach Hub."""
    if "user_id" not in session:
        flash("Please log in to access the Personal Learning Coach", "warning")
        return redirect("/login")

    user_id = session["user_id"]
    fullname = session.get("fullname", "Student")
    is_session_active = "study_start_time" in session
    active_study_topic = session.get("study_topic", "")

    # Aggregate 8-pillar telemetry and 4 core diagnostic superpower answers
    telemetry = get_eight_pillar_telemetry(user_id)
    what_next = get_what_to_study_next(user_id)
    what_weak = get_what_am_i_weak_at(user_id)
    when_revise = get_when_to_revise(user_id)
    how_improving = get_how_am_i_improving(user_id)

    return render_template(
        "coach.html",
        fullname=fullname,
        is_session_active=is_session_active,
        active_study_topic=active_study_topic,
        telemetry=telemetry,
        what_next=what_next,
        what_weak=what_weak,
        when_revise=when_revise,
        how_improving=how_improving,
    )


@app.route("/api/coach/overview", methods=["GET"])
def api_coach_overview():
    """Retrieve full 8-pillar telemetry synthesis and coach health score."""
    if "user_id" not in session:
        return jsonify({"success": False, "error": "Please login"}), 401

    user_id = session["user_id"]
    telemetry = get_eight_pillar_telemetry(user_id)
    return jsonify({
        "success": True,
        "overview": telemetry
    })


@app.route("/api/coach/what-to-study-next", methods=["GET"])
def api_coach_what_to_study_next():
    """Retrieve prioritized, deterministic next study recommendations."""
    if "user_id" not in session:
        return jsonify({"success": False, "error": "Please login"}), 401

    user_id = session["user_id"]
    result = get_what_to_study_next(user_id)
    return jsonify({
        "success": True,
        **result
    })


@app.route("/api/coach/what-am-i-weak-at", methods=["GET"])
def api_coach_what_am_i_weak_at():
    """Retrieve multi-dimensional weakness diagnostics and passive-study alerts."""
    if "user_id" not in session:
        return jsonify({"success": False, "error": "Please login"}), 401

    user_id = session["user_id"]
    result = get_what_am_i_weak_at(user_id)
    return jsonify({
        "success": True,
        **result
    })


@app.route("/api/coach/when-to-revise", methods=["GET"])
def api_coach_when_to_revise():
    """Retrieve Ebbinghaus decay estimations and spaced repetition calendar."""
    if "user_id" not in session:
        return jsonify({"success": False, "error": "Please login"}), 401

    user_id = session["user_id"]
    result = get_when_to_revise(user_id)
    return jsonify({
        "success": True,
        **result
    })


@app.route("/api/coach/how-am-i-improving", methods=["GET"])
def api_coach_how_am_i_improving():
    """Retrieve longitudinal improvement metrics, velocity deltas, and earned milestones."""
    if "user_id" not in session:
        return jsonify({"success": False, "error": "Please login"}), 401

    user_id = session["user_id"]
    result = get_how_am_i_improving(user_id)
    return jsonify({
        "success": True,
        **result
    })


@app.route("/api/coach/ask", methods=["POST"])
def api_coach_ask():
    """Interactive conversational query with the Personal Learning Coach."""
    if "user_id" not in session:
        return jsonify({"success": False, "error": "Please login"}), 401

    user_id = session["user_id"]
    data = request.get_json(silent=True) or {}
    intent = data.get("intent", "CUSTOM")
    prompt = data.get("prompt") or data.get("message")

    result = ask_personal_coach(
        user_id=user_id,
        query_intent=intent,
        custom_prompt=prompt
    )
    return jsonify(result)


@app.route("/api/coach/action-complete", methods=["POST"])
def api_coach_action_complete():
    """Log or update completion status for a coach recommendation."""
    if "user_id" not in session:
        return jsonify({"success": False, "error": "Please login"}), 401

    user_id = session["user_id"]
    data = request.get_json(silent=True) or {}
    rec_id = data.get("recommendation_id")
    topic = data.get("topic") or "General Study"
    action_type = data.get("action_type") or "STUDY"
    now_iso = datetime.now().isoformat()

    conn = get_db()
    c = conn.cursor()
    c.execute("""
        INSERT INTO coach_recommendations (
            user_id, topic, action_type, title, reason, status, created_at, completed_at
        ) VALUES (?, ?, ?, ?, ?, 'COMPLETED', ?, ?)
    """, (user_id, topic, action_type, f"Completed: {action_type}", "User completed recommended action", now_iso, now_iso))
    conn.commit()
    conn.close()

    return jsonify({"success": True, "message": "Action marked as completed"})


@app.route("/reports")
def reports():

    if "user_id" not in session:
        return redirect("/login")

    user_id = session["user_id"]
    conn = get_db()
    cursor = conn.cursor()

    cursor.execute("""
        SELECT ss.id, ss.date, ss.start_time, ss.end_time, ss.duration, ss.focus_score,
               sfs.estimated_score, sfs.quality_tier, sfs.presence_pct,
               sfs.absence_count, sfs.longest_absence_sec, sfs.total_absence_sec
        FROM study_sessions ss
        LEFT JOIN session_focus_scores sfs ON sfs.session_id = ss.id
        WHERE ss.user_id = ?
        ORDER BY ss.id DESC
    """, (user_id,))

    sessions = cursor.fetchall()

    # Aggregates for report analytics cards
    cursor.execute("""
        SELECT 
            COUNT(ss.id) AS total_count,
            COALESCE(SUM(ss.duration), 0) AS total_sec,
            COALESCE(AVG(COALESCE(sfs.estimated_score, ss.focus_score)), 0) AS avg_score,
            MAX(COALESCE(sfs.estimated_score, ss.focus_score)) AS best_score,
            MIN(COALESCE(sfs.estimated_score, ss.focus_score)) AS lowest_score
        FROM study_sessions ss
        LEFT JOIN session_focus_scores sfs ON sfs.session_id = ss.id
        WHERE ss.user_id = ?
    """, (user_id,))
    stats_row = cursor.fetchone()

    # Focus Intelligence aggregate stats
    cursor.execute("""
        SELECT
            COALESCE(AVG(estimated_score), 0) AS avg_fi_score,
            COALESCE(AVG(presence_pct), 0)    AS avg_presence_pct,
            COUNT(*) AS fi_session_count
        FROM session_focus_scores
        WHERE user_id = ?
    """, (user_id,))
    fi_stats_row = cursor.fetchone()

    # Tier distribution for donut chart
    cursor.execute("""
        SELECT quality_tier, COUNT(*) AS cnt
        FROM session_focus_scores
        WHERE user_id = ?
        GROUP BY quality_tier
    """, (user_id,))
    tier_rows = cursor.fetchall()
    conn.close()

    total_count = stats_row["total_count"] if stats_row else 0
    total_sec   = stats_row["total_sec"]   if stats_row else 0
    avg_score   = int(round(stats_row["avg_score"])) if stats_row and total_count > 0 else 0
    best_score  = int(round(stats_row["best_score"])) if stats_row and stats_row["best_score"] is not None else 0
    lowest_score = int(round(stats_row["lowest_score"])) if stats_row and stats_row["lowest_score"] is not None else 0
    avg_score_label = get_focus_label(avg_score) if total_count > 0 else "No Data Yet"

    fi_avg_score     = round(fi_stats_row["avg_fi_score"],  1) if fi_stats_row else 0
    fi_avg_presence  = round(fi_stats_row["avg_presence_pct"], 1) if fi_stats_row else 0
    fi_session_count = fi_stats_row["fi_session_count"] if fi_stats_row else 0

    tier_distribution = {r["quality_tier"]: r["cnt"] for r in tier_rows}

    tot_hours = total_sec // 3600
    tot_mins  = (total_sec % 3600) // 60
    total_time_formatted = f"{tot_hours}h {tot_mins}m"
    avg_session_mins = round((total_sec / max(1, total_count)) / 60, 1)

    report_data = []
    for study in sessions:
        total_secs = study["duration"]
        hrs  = total_secs // 3600
        mins = (total_secs % 3600) // 60
        secs = total_secs % 60
        duration_formatted = f"{hrs:02d}:{mins:02d}:{secs:02d}"
        score_val = study["estimated_score"] if study["estimated_score"] is not None else study["focus_score"]
        presence_val = study["presence_pct"] if study["presence_pct"] is not None else 100.0
        tier_val = study["quality_tier"] or ("Excellent" if score_val >= 90 else ("Strong" if score_val >= 75 else ("Moderate" if score_val >= 60 else "Needs Improvement")))

        report_data.append({
            "date":              study["date"],
            "start":             study["start_time"],
            "end":               study["end_time"],
            "duration":          duration_formatted,
            "duration_sec":      total_secs,
            "focus":             score_val,
            "fi_score":          score_val,
            "fi_tier":           tier_val,
            "fi_presence_pct":   presence_val,
            "fi_absence_count":  study["absence_count"] if study["absence_count"] is not None else 0,
            "fi_label":          get_focus_label(score_val),
        })

    weekly_labels, weekly_values = get_weekly_analytics(user_id)
    recall_history_list = get_recall_history(user_id)
    weak_concepts_list = get_weak_topics_summary(user_id)

    return render_template(
        "reports.html",
        fullname=session.get("fullname", "Student"),
        reports=report_data,
        total_sessions=total_count,
        total_time=total_time_formatted,
        avg_score=avg_score,
        best_score=best_score,
        lowest_score=lowest_score,
        avg_score_label=avg_score_label,
        avg_session_mins=avg_session_mins,
        weekly_labels=weekly_labels,
        weekly_values=weekly_values,
        fi_avg_score=fi_avg_score,
        fi_avg_presence=fi_avg_presence,
        fi_session_count=fi_session_count,
        fi_tier_distribution=tier_distribution,
        fi_disclaimer=FI_DISCLAIMER,
        recall_history=recall_history_list,
        weak_concepts=weak_concepts_list,
    )


@app.route("/parent")
def parent():
    if "user_id" not in session:
        return redirect("/login")

    user_id = session["user_id"]
    today = datetime.now().strftime("%Y-%m-%d")

    conn = get_db()
    cursor = conn.cursor()

    # Query today's completed study duration and count
    cursor.execute("""
        SELECT COUNT(*) AS today_count, COALESCE(SUM(duration), 0) AS total_today
        FROM study_sessions
        WHERE user_id = ? AND date = ?
    """, (user_id, today))
    today_row = cursor.fetchone()
    today_sessions_count = today_row["today_count"] if today_row else 0
    total_seconds = today_row["total_today"] if today_row else 0

    # Query total sessions and avg focus
    cursor.execute("""
        SELECT 
            COUNT(ss.id) AS total_count,
            COALESCE(AVG(COALESCE(sfs.estimated_score, ss.focus_score)), 0) AS avg_score,
            MAX(COALESCE(sfs.estimated_score, ss.focus_score)) AS best_score,
            MIN(COALESCE(sfs.estimated_score, ss.focus_score)) AS lowest_score
        FROM study_sessions ss
        LEFT JOIN session_focus_scores sfs ON sfs.session_id = ss.id
        WHERE ss.user_id = ?
    """, (user_id,))
    stat_row = cursor.fetchone()
    total_sessions = stat_row["total_count"] if stat_row else 0
    avg_score = int(round(stat_row["avg_score"])) if stat_row and total_sessions > 0 else 0

    # Query daily goal
    cursor.execute("SELECT daily_goal FROM goals WHERE user_id = ?", (user_id,))
    goal_row = cursor.fetchone()
    daily_goal = goal_row["daily_goal"] if goal_row else 2

    # Focus Intelligence: most recent session score
    cursor.execute("""
        SELECT sfs.estimated_score, sfs.quality_tier, sfs.presence_pct,
               sfs.absence_count, sfs.longest_absence_sec, sfs.total_absence_sec,
               sfs.session_duration_sec, sfs.goal_progress_pct, ss.date AS session_date
        FROM session_focus_scores sfs
        JOIN study_sessions ss ON ss.id = sfs.session_id
        WHERE sfs.user_id = ?
        ORDER BY sfs.id DESC LIMIT 1
    """, (user_id,))
    parent_fi_row = cursor.fetchone()

    conn.close()

    goal_seconds = max(1, daily_goal * 3600)
    progress = min(int((total_seconds / goal_seconds) * 100), 100)

    running = "study_start_time" in session
    session_time = "00:00:00"
    if running:
        try:
            start_time = datetime.fromisoformat(session["study_start_time"])
            elapsed = max(0, int((datetime.now() - start_time).total_seconds()))
            hrs = elapsed // 3600
            mins = (elapsed % 3600) // 60
            secs = elapsed % 60
            session_time = f"{hrs:02d}:{mins:02d}:{secs:02d}"
        except Exception:
            session_time = "00:00:00"

    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    today_study_str = f"{hours} hr {minutes} min"

    radar_status = check_radar(running)
    streak = calculate_streak(user_id)
    weekly_labels, weekly_values = get_weekly_analytics(user_id)

    parent_fi = None
    latest_focus_score = 0
    latest_focus_label = "No Score Yet"
    latest_presence_pct = 100.0

    if parent_fi_row:
        fi_calc = compute_focus_score({
            "duration_sec":        parent_fi_row["session_duration_sec"],
            "absence_count":       parent_fi_row["absence_count"],
            "longest_absence_sec": parent_fi_row["longest_absence_sec"],
            "total_absence_sec":   parent_fi_row["total_absence_sec"],
            "goal_progress_pct":   parent_fi_row["goal_progress_pct"],
        })
        latest_focus_score = parent_fi_row["estimated_score"]
        latest_focus_label = get_focus_label(latest_focus_score)
        latest_presence_pct = parent_fi_row["presence_pct"]
        parent_fi = {
            "score":            latest_focus_score,
            "tier":             parent_fi_row["quality_tier"],
            "label":            latest_focus_label,
            "presence_pct":     latest_presence_pct,
            "absence_count":    parent_fi_row["absence_count"],
            "longest_absence":  parent_fi_row["longest_absence_sec"],
            "session_date":     parent_fi_row["session_date"],
            "interpretation":   fi_calc["interpretation"],
        }

    return render_template(
        "parent.html",
        fullname=session.get("fullname", "Student"),
        status=radar_status,
        today=today_study_str,
        today_sessions_count=today_sessions_count,
        latest_focus_score=latest_focus_score,
        latest_focus_label=latest_focus_label,
        latest_presence_pct=latest_presence_pct,
        session_time=session_time,
        progress=progress,
        daily_goal=daily_goal,
        streak=streak,
        avg_score=avg_score,
        total_sessions=total_sessions,
        weekly_labels=weekly_labels,
        weekly_values=weekly_values,
        simulation_mode=sensor_controller.is_simulation_mode(),
        parent_fi=parent_fi,
        fi_disclaimer=FI_DISCLAIMER,
    )


@app.route("/goals", methods=["GET", "POST"])
def goals():
    if "user_id" not in session:
        return redirect("/login")

    user_id = session["user_id"]
    today = datetime.now().strftime("%Y-%m-%d")
    conn = get_db()
    cursor = conn.cursor()

    if request.method == "POST":
        try:
            daily_goal = int(request.form.get("daily_goal", 2))
            daily_goal = max(1, min(24, daily_goal))
        except ValueError:
            daily_goal = 2

        try:
            recall_delay = int(request.form.get("recall_delay_minutes", 10))
            recall_delay = max(0, min(10080, recall_delay))
        except ValueError:
            recall_delay = 10

        cursor.execute("SELECT id FROM goals WHERE user_id = ?", (user_id,))
        existing_goal = cursor.fetchone()

        if existing_goal:
            cursor.execute("""
                UPDATE goals
                SET daily_goal = ?, weekly_goal = ?, recall_delay_minutes = ?
                WHERE user_id = ?
            """, (daily_goal, daily_goal * 7, recall_delay, user_id))
        else:
            cursor.execute("""
                INSERT INTO goals (user_id, daily_goal, weekly_goal, recall_delay_minutes)
                VALUES (?, ?, ?, ?)
            """, (user_id, daily_goal, daily_goal * 7, recall_delay))

        conn.commit()
        flash("Study and recall goals updated successfully!", "success")

    cursor.execute("SELECT daily_goal, weekly_goal, recall_delay_minutes FROM goals WHERE user_id = ?", (user_id,))
    goal_row = cursor.fetchone()
    daily_goal = goal_row["daily_goal"] if goal_row else 2
    weekly_goal = goal_row["weekly_goal"] if goal_row else daily_goal * 7
    recall_delay_minutes = goal_row["recall_delay_minutes"] if goal_row and goal_row["recall_delay_minutes"] is not None else 10

    # Today's study time
    cursor.execute("""
        SELECT COALESCE(SUM(duration), 0) AS total_today
        FROM study_sessions
        WHERE user_id = ? AND date = ?
    """, (user_id, today))
    today_row = cursor.fetchone()
    total_seconds = today_row["total_today"] if today_row else 0
    conn.close()

    goal_seconds = max(1, daily_goal * 3600)
    progress = min(int((total_seconds / goal_seconds) * 100), 100)
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    today_study = f"{hours} hr {minutes} min"
    streak = calculate_streak(user_id)

    return render_template(
        "goals.html",
        fullname=session.get("fullname", "Student"),
        daily_goal=daily_goal,
        weekly_goal=weekly_goal,
        recall_delay_minutes=recall_delay_minutes,
        progress=progress,
        today_study=today_study,
        streak=streak
    )


@app.route("/settings", methods=["GET", "POST"])
def settings():
    if "user_id" not in session:
        return redirect("/login")

    user_id = session["user_id"]
    conn = get_db()
    cursor = conn.cursor()

    if request.method == "POST":
        fullname = request.form.get("fullname", "").strip()
        email = request.form.get("email", "").strip()

        if fullname and email:
            cursor.execute("""
                UPDATE users
                SET fullname = ?, email = ?
                WHERE id = ?
            """, (fullname, email, user_id))
            conn.commit()
            session["fullname"] = fullname
            flash("Profile updated successfully!", "success")
        else:
            flash("Full Name and Email cannot be blank.", "error")

    cursor.execute("SELECT fullname, email, username FROM users WHERE id = ?", (user_id,))
    user = cursor.fetchone()
    conn.close()

    return render_template(
        "settings.html",
        fullname=user["fullname"] if user else session.get("fullname", ""),
        email=user["email"] if user else "",
        username=user["username"] if user else ""
    )


@app.route("/status")
def status():
    if "user_id" not in session:
        return jsonify({"error": "Not logged in"}), 401

    running = "study_start_time" in session
    elapsed_seconds = 0

    if running:
        try:
            start_time = datetime.fromisoformat(session["study_start_time"])
            elapsed_seconds = max(0, int((datetime.now() - start_time).total_seconds()))
        except Exception:
            elapsed_seconds = 0

    radar_status = check_radar(running)
    light_status = check_light(running)
    user_id = session["user_id"]

    # -----------------------------------------------------------------------
    # Focus Intelligence: detect PRESENT <-> ABSENT transitions during a session
    # -----------------------------------------------------------------------
    fi_preview = None
    current_absence_sec = 0

    if running:
        prev_presence = session.get("fi_last_presence", "PRESENT")
        now_str = datetime.now().isoformat()

        if radar_status == "ABSENT" and prev_presence == "PRESENT":
            # Transition: student just left the desk — open a new absence event
            session["fi_absence_start"] = now_str
            conn = get_db()
            conn.execute("""
                INSERT INTO session_absence_events (user_id, absence_start)
                VALUES (?, ?)
            """, (user_id, now_str))
            conn.commit()
            conn.close()

        elif radar_status == "PRESENT" and prev_presence == "ABSENT":
            # Transition: student returned — close the open absence event
            fi_absence_start = session.get("fi_absence_start")
            if fi_absence_start:
                try:
                    abs_start_dt = datetime.fromisoformat(fi_absence_start)
                    abs_dur = max(0, int((datetime.now() - abs_start_dt).total_seconds()))
                    conn = get_db()
                    conn.execute("""
                        UPDATE session_absence_events
                        SET absence_end = ?, duration_seconds = ?
                        WHERE user_id = ? AND absence_start = ? AND session_id IS NULL
                    """, (now_str, abs_dur, user_id, fi_absence_start))
                    conn.commit()
                    conn.close()
                except Exception:
                    pass
                session["fi_absence_start"] = None

        # Track current open absence duration
        fi_absence_start = session.get("fi_absence_start")
        if fi_absence_start and radar_status == "ABSENT":
            try:
                current_absence_sec = max(0, int(
                    (datetime.now() - datetime.fromisoformat(fi_absence_start)).total_seconds()
                ))
            except Exception:
                current_absence_sec = 0

        # Compute live preview score from completed events so far
        conn = get_db()
        cursor = conn.cursor()
        cursor.execute("""
            SELECT duration_seconds FROM session_absence_events
            WHERE user_id = ? AND session_id IS NULL AND absence_end IS NOT NULL
        """, (user_id,))
        completed_events = [dict(r) for r in cursor.fetchall()]
        conn.close()

        fi_preview = compute_live_preview(
            elapsed_sec=elapsed_seconds,
            current_absence_sec=current_absence_sec,
            completed_absence_events=completed_events,
        )

        # Update last known presence state in session
        session["fi_last_presence"] = radar_status

    # Check for due recall obligations when not in session
    due_recall_task = None
    try:
        tasks_data = get_pending_and_due_tasks(user_id)
        if tasks_data.get("ready_count", 0) > 0:
            due_recall_task = tasks_data.get("primary_task")
    except Exception:
        pass

    return jsonify({
        "running":          running,
        "radar":            radar_status,
        "light":            light_status,
        "elapsed_seconds":  elapsed_seconds,
        "simulation_mode":  sensor_controller.is_simulation_mode(),
        "presence":         radar_status == "PRESENT",
        # Focus Intelligence live preview (only present when session is running)
        "fi_preview":       fi_preview,
        "fi_absence_sec":   current_absence_sec,
        "due_recall_task":  due_recall_task,
    })


@app.route("/focus-intelligence/<int:session_id>")
def focus_intelligence_detail(session_id):
    """Return the stored Focus Intelligence data for a completed study session."""
    if "user_id" not in session:
        return jsonify({"error": "Not logged in"}), 401

    user_id = session["user_id"]
    conn = get_db()
    cursor = conn.cursor()

    cursor.execute("""
        SELECT sfs.*, ss.date AS session_date, ss.start_time, ss.end_time
        FROM session_focus_scores sfs
        JOIN study_sessions ss ON ss.id = sfs.session_id
        WHERE sfs.session_id = ? AND sfs.user_id = ?
    """, (session_id, user_id))
    row = cursor.fetchone()
    conn.close()

    if not row:
        return jsonify({"error": "No focus intelligence data for this session"}), 404

    fi_calc = compute_focus_score({
        "duration_sec":        row["session_duration_sec"],
        "absence_count":       row["absence_count"],
        "longest_absence_sec": row["longest_absence_sec"],
        "total_absence_sec":   row["total_absence_sec"],
        "goal_progress_pct":   row["goal_progress_pct"],
    })

    return jsonify({
        "session_id":              session_id,
        "session_date":            row["session_date"],
        "estimated_score":         row["estimated_score"],
        "quality_tier":            row["quality_tier"],
        "focus_label":             get_focus_label(row["estimated_score"]),
        "presence_pct":            row["presence_pct"],
        "completion_pct":          fi_calc["completion_pct"],
        "distraction_control_pct": fi_calc["distraction_control_pct"],
        "duration_rating_pct":     fi_calc["duration_rating_pct"],
        "absence_count":           row["absence_count"],
        "longest_absence_sec":     row["longest_absence_sec"],
        "total_absence_sec":       row["total_absence_sec"],
        "session_duration_sec":    row["session_duration_sec"],
        "goal_progress_pct":       row["goal_progress_pct"],
        "interpretation":          fi_calc["interpretation"],
        "disclaimer":              FI_DISCLAIMER,
        "computed_at":             row["computed_at"],
    })




@app.route("/simulation/presence", methods=["POST"])
def set_simulation_presence():
    """API endpoint to set the simulated radar presence state (PRESENT / ABSENT)."""
    if "user_id" not in session:
        return jsonify({"error": "Unauthorized", "success": False}), 401

    data = request.get_json(silent=True) or {}
    
    # Support 'presence': bool or 'status': 'PRESENT'/'ABSENT'
    if "presence" in data:
        is_present = bool(data["presence"])
    elif "status" in data:
        is_present = str(data["status"]).upper() == "PRESENT"
    else:
        # Default to toggle if no explicit parameter provided
        is_present = not sensor_controller.get_simulated_presence()

    result = sensor_controller.set_simulated_presence(is_present)
    running = "study_start_time" in session
    light_state = check_light(running)

    return jsonify({
        "success": True,
        "simulation_mode": True,
        "presence": result["present"],
        "status": result["status"],
        "light": light_state,
        "message": f"Simulated radar presence set to {result['status']}"
    })


@app.route("/simulation/toggle", methods=["POST"])
def toggle_simulation():
    """API endpoint to quickly toggle simulated presence between PRESENT and ABSENT."""
    if "user_id" not in session:
        return jsonify({"error": "Unauthorized", "success": False}), 401

    result = sensor_controller.toggle_simulated_presence()
    running = "study_start_time" in session
    light_state = check_light(running)

    return jsonify({
        "success": True,
        "simulation_mode": True,
        "presence": result["present"],
        "status": result["status"],
        "light": light_state,
        "message": f"Simulated radar presence toggled to {result['status']}"
    })


@app.route("/simulation/status", methods=["GET"])
def simulation_status():
    """API endpoint to query current simulation parameters."""
    if "user_id" not in session:
        return jsonify({"error": "Unauthorized", "success": False}), 401

    running = "study_start_time" in session
    radar_info = sensor_controller.check_radar_presence(running)
    light_state = check_light(running)

    return jsonify({
        "simulation_mode": sensor_controller.is_simulation_mode(),
        "presence": radar_info["present"],
        "status": radar_info["status"],
        "light": light_state,
        "running": running
    })


if __name__ == "__main__":
    app.run(debug=True)