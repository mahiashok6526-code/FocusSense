import os
import shutil
import sqlite3

def _resolve_database_path():
    """
    Resolves SQLite database path:
    1. DATABASE_PATH environment variable if explicitly set.
    2. Vercel / AWS Lambda serverless execution (read-only filesystem):
       - Uses /tmp/users.db (writable scratch space).
       - On cold boot, if seed users.db exists in project root and /tmp/users.db does not exist,
         copies the seed database to /tmp/users.db.
       - Does NOT overwrite /tmp/users.db on subsequent requests.
    3. Local execution:
       - Uses users.db in project directory.
    """
    db_env = os.environ.get("DATABASE_PATH")
    if db_env:
        return db_env

    is_serverless = bool(os.environ.get("VERCEL") or os.environ.get("AWS_LAMBDA_FUNCTION_NAME"))
    if is_serverless:
        tmp_db = "/tmp/users.db"
        if not os.path.exists(tmp_db):
            local_seed = os.path.abspath(os.path.join(os.path.dirname(__file__), "users.db"))
            if os.path.exists(local_seed):
                try:
                    shutil.copy2(local_seed, tmp_db)
                except Exception:
                    pass
        return tmp_db

    return os.path.abspath(os.path.join(os.path.dirname(__file__), "users.db"))

DATABASE = _resolve_database_path()
DB_PATH = DATABASE

def get_db_connection(db_name=None):
    target_db = db_name or DATABASE
    conn = sqlite3.connect(target_db, timeout=30.0)
    conn.row_factory = sqlite3.Row
    return conn

def create_database(db_name=None):
    target_db = db_name or DATABASE
    db_dir = os.path.dirname(target_db)
    if db_dir:
        os.makedirs(db_dir, exist_ok=True)
    connection = sqlite3.connect(target_db, timeout=30.0)
    cursor = connection.cursor()

    # Users Table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS users(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        fullname TEXT,
        email TEXT UNIQUE,
        username TEXT UNIQUE,
        password TEXT
    )
    """)

    # Study Sessions
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS study_sessions(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER,
        date TEXT,
        start_time TEXT,
        end_time TEXT,
        duration INTEGER,
        focus_score INTEGER,
        FOREIGN KEY(user_id) REFERENCES users(id)
    )
    """)

    # Study Goals
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS goals(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER,
        daily_goal INTEGER,
        weekly_goal INTEGER,
        FOREIGN KEY(user_id) REFERENCES users(id)
    )
    """)

    # Notifications
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS notifications(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER,
        message TEXT,
        created_at TEXT,
        status TEXT,
        FOREIGN KEY(user_id) REFERENCES users(id)
    )
    """)

    # Focus Intelligence — Absence Events
    # Records each individual PRESENT→ABSENT→PRESENT event during a session.
    # session_id is NULL while the session is still active; backfilled at /stop-study.
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS session_absence_events(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        session_id INTEGER,
        user_id INTEGER NOT NULL,
        absence_start TEXT NOT NULL,
        absence_end TEXT,
        duration_seconds INTEGER,
        FOREIGN KEY(user_id) REFERENCES users(id)
    )
    """)

    # Focus Intelligence — Per-Session Computed Scores
    # One row per completed study session. Joined to study_sessions via session_id.
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS session_focus_scores(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        session_id INTEGER UNIQUE NOT NULL,
        user_id INTEGER NOT NULL,
        presence_pct REAL,
        absence_count INTEGER,
        longest_absence_sec INTEGER,
        total_absence_sec INTEGER,
        session_duration_sec INTEGER,
        goal_progress_pct REAL,
        quality_tier TEXT,
        estimated_score INTEGER,
        computed_at TEXT,
        FOREIGN KEY(session_id) REFERENCES study_sessions(id),
        FOREIGN KEY(user_id) REFERENCES users(id)
    )
    """)

    # -------------------------------------------------------------------------
    # FocusSense AI — Topic Grounding & In-Session Chat (Phase 1)
    # -------------------------------------------------------------------------

    # 1. Study Topics Catalog
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS study_topics(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER NOT NULL,
        topic_name TEXT NOT NULL,
        created_at TEXT NOT NULL,
        last_studied_at TEXT,
        session_count INTEGER DEFAULT 1,
        FOREIGN KEY(user_id) REFERENCES users(id)
    )
    """)

    # 2. AI Conversations (one per session or topical thread)
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS ai_conversations(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER NOT NULL,
        session_id INTEGER,
        topic TEXT NOT NULL,
        title TEXT,
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL,
        FOREIGN KEY(user_id) REFERENCES users(id),
        FOREIGN KEY(session_id) REFERENCES study_sessions(id)
    )
    """)

    # 3. AI Chat Messages (multi-turn conversation history)
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS ai_messages(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        conversation_id INTEGER NOT NULL,
        sender TEXT NOT NULL, -- 'user' | 'assistant' | 'system'
        message TEXT NOT NULL,
        timestamp TEXT NOT NULL,
        tokens INTEGER DEFAULT 0,
        FOREIGN KEY(conversation_id) REFERENCES ai_conversations(id)
    )
    """)

    # -------------------------------------------------------------------------
    # FocusSense AI — Post-Study Examination & Learning Verification (Phase 2)
    # -------------------------------------------------------------------------

    # 1. Study Quizzes (one per examination)
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS study_quizzes(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER NOT NULL,
        session_id INTEGER,
        topic TEXT NOT NULL,
        total_questions INTEGER DEFAULT 0,
        score_pct REAL DEFAULT 0.0,
        quality_label TEXT,
        summary_text TEXT,
        created_at TEXT NOT NULL,
        FOREIGN KEY(user_id) REFERENCES users(id),
        FOREIGN KEY(session_id) REFERENCES study_sessions(id)
    )
    """)

    # 2. Quiz Questions (individual questions within an exam)
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS quiz_questions(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        quiz_id INTEGER NOT NULL,
        question_index INTEGER NOT NULL,
        question_text TEXT NOT NULL,
        sub_concept TEXT NOT NULL,
        sample_answer TEXT NOT NULL,
        points_possible REAL DEFAULT 1.0,
        FOREIGN KEY(quiz_id) REFERENCES study_quizzes(id)
    )
    """)

    # 3. Quiz Answers (student responses & semantic evaluations)
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS quiz_answers(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        question_id INTEGER NOT NULL,
        quiz_id INTEGER NOT NULL,
        user_id INTEGER NOT NULL,
        student_answer TEXT NOT NULL,
        evaluation_status TEXT NOT NULL, -- 'correct' | 'partial' | 'incorrect' | 'unanswered'
        score_earned REAL DEFAULT 0.0,
        feedback_text TEXT,
        answered_at TEXT NOT NULL,
        FOREIGN KEY(question_id) REFERENCES quiz_questions(id),
        FOREIGN KEY(quiz_id) REFERENCES study_quizzes(id),
        FOREIGN KEY(user_id) REFERENCES users(id)
    )
    """)

    # 4. Concept Mastery (running sub-concept accuracy & weak-concept detection)
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS concept_mastery(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER NOT NULL,
        topic TEXT NOT NULL,
        sub_concept TEXT NOT NULL,
        total_tested INTEGER DEFAULT 0,
        correct_count INTEGER DEFAULT 0,
        partial_count INTEGER DEFAULT 0,
        incorrect_count INTEGER DEFAULT 0,
        mastery_pct REAL DEFAULT 0.0,
        is_weak INTEGER DEFAULT 0, -- 1 if weak concept (< 70%), 0 if solid
        last_tested_at TEXT,
        FOREIGN KEY(user_id) REFERENCES users(id)
    )
    """)

    # -------------------------------------------------------------------------
    # FocusSense Recall Engine — Delayed Spaced Recall (Phase 3)
    # -------------------------------------------------------------------------
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS recall_tasks (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER NOT NULL,
        session_id INTEGER NOT NULL,
        topic TEXT NOT NULL,
        delay_minutes INTEGER DEFAULT 0,
        scheduled_for TEXT NOT NULL,
        status TEXT DEFAULT 'PENDING', -- 'PENDING' | 'READY' | 'IN_PROGRESS' | 'COMPLETED' | 'EXPIRED'
        quiz_id INTEGER,
        retention_score REAL DEFAULT NULL,
        quality_label TEXT DEFAULT NULL,
        created_at TEXT NOT NULL,
        completed_at TEXT DEFAULT NULL,
        FOREIGN KEY(user_id) REFERENCES users(id),
        FOREIGN KEY(session_id) REFERENCES study_sessions(id),
        FOREIGN KEY(quiz_id) REFERENCES study_quizzes(id)
    )
    """)

    # -------------------------------------------------------------------------
    # FocusSense AI — Active AI Study Session Tracking (Anti-Gaming Analytics)
    # -------------------------------------------------------------------------
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS ai_study_sessions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER NOT NULL,
        session_id INTEGER,
        conversation_id INTEGER,
        topic TEXT NOT NULL,
        started_at TEXT NOT NULL,
        ended_at TEXT,
        active_duration_sec INTEGER DEFAULT 0,
        question_count INTEGER DEFAULT 0,
        meaningful_query_count INTEGER DEFAULT 0,
        FOREIGN KEY(user_id) REFERENCES users(id),
        FOREIGN KEY(session_id) REFERENCES study_sessions(id),
        FOREIGN KEY(conversation_id) REFERENCES ai_conversations(id)
    )
    """)

    # -------------------------------------------------------------------------
    # FocusSense Learning Intelligence — Topic Mastery & Adaptive Progress
    # -------------------------------------------------------------------------
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS topic_mastery (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER NOT NULL,
        topic TEXT NOT NULL,
        mastery_tier TEXT NOT NULL DEFAULT 'MEDIUM', -- 'WEAK' (<60%), 'MEDIUM' (60-84%), 'STRONG' (>=85%)
        total_quizzes INTEGER DEFAULT 0,
        total_questions INTEGER DEFAULT 0,
        correct_count INTEGER DEFAULT 0,
        partial_count INTEGER DEFAULT 0,
        incorrect_count INTEGER DEFAULT 0,
        avg_retention_score REAL DEFAULT 0.0,
        last_retention_score REAL DEFAULT 0.0,
        last_tested_at TEXT,
        updated_at TEXT NOT NULL,
        FOREIGN KEY(user_id) REFERENCES users(id),
        UNIQUE(user_id, topic)
    )
    """)

    # -------------------------------------------------------------------------
    # FocusSense Learning Intelligence — Adaptive Learning & RL Telemetry Foundation
    # -------------------------------------------------------------------------
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS adaptive_learning_logs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER NOT NULL,
        session_id INTEGER,
        topic TEXT NOT NULL,
        prior_mastery_tier TEXT NOT NULL, -- 'WEAK' | 'MEDIUM' | 'STRONG'
        action_intervention TEXT NOT NULL, -- 'REMEDIATION_DRILL' | 'CONCEPTUAL_CHALLENGE' | 'ADVANCED_SYNTHESIS'
        retention_outcome_pct REAL NOT NULL,
        reward_delta REAL NOT NULL,        -- Outcome % minus prior average %
        created_at TEXT NOT NULL,
        FOREIGN KEY(user_id) REFERENCES users(id),
        FOREIGN KEY(session_id) REFERENCES study_sessions(id)
    )
    """)

    # -------------------------------------------------------------------------
    # FocusSense Unified Learning Loop & Adaptive Policy Tables (RL Foundation)
    # -------------------------------------------------------------------------

    # 1. State snapshot per study/recall episode
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS learning_states (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER NOT NULL,
        session_id INTEGER,
        topic TEXT NOT NULL,
        physical_presence_pct REAL,
        study_duration_sec INTEGER,
        ai_engagement_pct REAL,
        quiz_score_pct REAL,
        retention_score_pct REAL,
        distraction_control_pct REAL,
        session_completion_pct REAL,
        consistency_score_pct REAL,
        unified_focus_score INTEGER,
        learning_efficiency_score REAL,
        quality_tier TEXT,
        score_label TEXT,
        explanation TEXT,
        created_at TEXT NOT NULL,
        FOREIGN KEY(user_id) REFERENCES users(id),
        FOREIGN KEY(session_id) REFERENCES study_sessions(id)
    )
    """)

    # 2. Adaptive policy actions selected
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS adaptive_actions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER NOT NULL,
        topic TEXT NOT NULL,
        session_id INTEGER,
        state_id INTEGER,
        action_type TEXT NOT NULL, -- 'INCREASE_RECALL_FREQ' | 'REMEDIATION_DRILL' | 'ADVANCED_CHALLENGE' | 'RECOMMEND_SHORT_SESSIONS' | 'REINFORCE_EFFICIENCY' | 'MAINTAIN_MASTERY'
        action_payload TEXT,      -- JSON config (delay_mins, difficulty_tier, recommendation)
        reason TEXT NOT NULL,
        status TEXT DEFAULT 'PENDING', -- 'PENDING' | 'EVALUATED' | 'DISMISSED'
        created_at TEXT NOT NULL,
        FOREIGN KEY(user_id) REFERENCES users(id),
        FOREIGN KEY(session_id) REFERENCES study_sessions(id),
        FOREIGN KEY(state_id) REFERENCES learning_states(id)
    )
    """)

    # 3. Measured rewards connecting state transitions (State_t -> Action_t -> Reward_t -> State_t+1)
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS adaptive_rewards (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER NOT NULL,
        action_id INTEGER NOT NULL,
        initial_state_id INTEGER NOT NULL,
        resulting_state_id INTEGER,
        reward_value REAL NOT NULL, -- Clamped scalar [-1.0, +1.0]
        evaluation_notes TEXT,
        recorded_at TEXT NOT NULL,
        FOREIGN KEY(user_id) REFERENCES users(id),
        FOREIGN KEY(action_id) REFERENCES adaptive_actions(id),
        FOREIGN KEY(initial_state_id) REFERENCES learning_states(id),
        FOREIGN KEY(resulting_state_id) REFERENCES learning_states(id)
    )
    """)

    # -------------------------------------------------------------------------
    # FocusSense AI — Study Materials & Knowledge Base
    # -------------------------------------------------------------------------

    # 1. Study Materials Table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS study_materials (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER NOT NULL,
        filename TEXT NOT NULL,           -- Original client filename (e.g., "Python Unit 3.pdf")
        stored_path TEXT NOT NULL,        -- Safe server storage path
        file_type TEXT NOT NULL,          -- 'pdf' | 'txt' | 'docx'
        file_size_bytes INTEGER NOT NULL,
        title TEXT NOT NULL,              -- Display title
        subject TEXT,                     -- Optional subject
        topic TEXT,                       -- Optional topic
        chunk_count INTEGER DEFAULT 0,
        status TEXT DEFAULT 'READY',      -- 'PROCESSING' | 'READY' | 'FAILED'
        error_message TEXT,
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL,
        FOREIGN KEY(user_id) REFERENCES users(id)
    )
    """)

    # 2. Material Chunks Table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS material_chunks (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        material_id INTEGER NOT NULL,
        user_id INTEGER NOT NULL,
        chunk_index INTEGER NOT NULL,
        chunk_text TEXT NOT NULL,
        char_count INTEGER NOT NULL,
        created_at TEXT NOT NULL,
        FOREIGN KEY(material_id) REFERENCES study_materials(id),
        FOREIGN KEY(user_id) REFERENCES users(id)
    )
    """)

    # Study Materials Indexes
    cursor.execute("""
    CREATE INDEX IF NOT EXISTS idx_study_materials_user
    ON study_materials(user_id)
    """)

    cursor.execute("""
    CREATE INDEX IF NOT EXISTS idx_material_chunks_material
    ON material_chunks(material_id)
    """)

    cursor.execute("""
    CREATE INDEX IF NOT EXISTS idx_material_chunks_user
    ON material_chunks(user_id)
    """)

    # -------------------------------------------------------------------------
    # FocusSense AI — Assessment Center & Examination Platform (Phase 5)
    # -------------------------------------------------------------------------
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS assessment_attempts (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER NOT NULL,
        quiz_id INTEGER NOT NULL,
        material_id INTEGER DEFAULT NULL,
        topic TEXT NOT NULL,
        sub_topic TEXT DEFAULT NULL,
        assessment_mode TEXT NOT NULL,       -- 'QUICK' | 'PRACTICE' | 'TOPIC' | 'EXAM'
        difficulty TEXT DEFAULT 'mixed',     -- 'basic' | 'medium' | 'advanced' | 'mixed' | 'adaptive'
        time_limit_sec INTEGER DEFAULT NULL,
        time_spent_sec INTEGER DEFAULT 0,
        started_at TEXT NOT NULL,
        completed_at TEXT DEFAULT NULL,
        score_pct REAL DEFAULT 0.0,
        concept_score REAL DEFAULT 0.0,
        application_score REAL DEFAULT 0.0,
        recall_score REAL DEFAULT 0.0,
        problem_solving_score REAL DEFAULT 0.0,
        strong_topics_json TEXT DEFAULT '[]',
        weak_topics_json TEXT DEFAULT '[]',
        synthesis_summary TEXT,
        adaptive_action_id INTEGER DEFAULT NULL,
        FOREIGN KEY(user_id) REFERENCES users(id),
        FOREIGN KEY(quiz_id) REFERENCES study_quizzes(id),
        FOREIGN KEY(material_id) REFERENCES study_materials(id)
    )
    """)

    cursor.execute("""
    CREATE INDEX IF NOT EXISTS idx_assessment_attempts_user
    ON assessment_attempts(user_id)
    """)

    cursor.execute("""
    CREATE INDEX IF NOT EXISTS idx_assessment_attempts_quiz
    ON assessment_attempts(quiz_id)
    """)

    # Safe additive migrations
    try:
        cursor.execute("ALTER TABLE ai_conversations ADD COLUMN is_archived INTEGER DEFAULT 0")
    except sqlite3.OperationalError:
        pass  # Column already exists

    try:
        cursor.execute("ALTER TABLE goals ADD COLUMN recall_delay_minutes INTEGER DEFAULT 10")
    except sqlite3.OperationalError:
        pass  # Column already exists

    try:
        cursor.execute("ALTER TABLE quiz_questions ADD COLUMN difficulty TEXT DEFAULT 'conceptual'")
    except sqlite3.OperationalError:
        pass  # Column already exists

    try:
        cursor.execute("ALTER TABLE quiz_questions ADD COLUMN question_type TEXT DEFAULT 'short_answer'")
    except sqlite3.OperationalError:
        pass

    try:
        cursor.execute("ALTER TABLE quiz_questions ADD COLUMN options_json TEXT DEFAULT NULL")
    except sqlite3.OperationalError:
        pass

    try:
        cursor.execute("ALTER TABLE quiz_questions ADD COLUMN correct_option TEXT DEFAULT NULL")
    except sqlite3.OperationalError:
        pass

    try:
        cursor.execute("ALTER TABLE quiz_questions ADD COLUMN category TEXT DEFAULT 'concept'")
    except sqlite3.OperationalError:
        pass

    try:
        cursor.execute("ALTER TABLE quiz_questions ADD COLUMN explanation TEXT DEFAULT NULL")
    except sqlite3.OperationalError:
        pass

    try:
        cursor.execute("ALTER TABLE study_sessions ADD COLUMN retention_score REAL DEFAULT NULL")
    except sqlite3.OperationalError:
        pass  # Column already exists

    try:
        cursor.execute("ALTER TABLE study_sessions ADD COLUMN material_id INTEGER DEFAULT NULL")
    except sqlite3.OperationalError:
        pass  # Column already exists

    try:
        cursor.execute("ALTER TABLE study_quizzes ADD COLUMN material_id INTEGER DEFAULT NULL")
    except sqlite3.OperationalError:
        pass  # Column already exists

    try:
        cursor.execute("ALTER TABLE study_quizzes ADD COLUMN assessment_mode TEXT DEFAULT 'STANDARD'")
    except sqlite3.OperationalError:
        pass

    try:
        cursor.execute("ALTER TABLE study_quizzes ADD COLUMN time_limit_sec INTEGER DEFAULT NULL")
    except sqlite3.OperationalError:
        pass

    try:
        cursor.execute("ALTER TABLE study_quizzes ADD COLUMN time_spent_sec INTEGER DEFAULT 0")
    except sqlite3.OperationalError:
        pass

    try:
        cursor.execute("ALTER TABLE study_quizzes ADD COLUMN concept_score REAL DEFAULT NULL")
    except sqlite3.OperationalError:
        pass

    try:
        cursor.execute("ALTER TABLE study_quizzes ADD COLUMN application_score REAL DEFAULT NULL")
    except sqlite3.OperationalError:
        pass

    try:
        cursor.execute("ALTER TABLE study_quizzes ADD COLUMN recall_score REAL DEFAULT NULL")
    except sqlite3.OperationalError:
        pass

    try:
        cursor.execute("ALTER TABLE study_quizzes ADD COLUMN problem_solving_score REAL DEFAULT NULL")
    except sqlite3.OperationalError:
        pass

    try:
        cursor.execute("ALTER TABLE recall_tasks ADD COLUMN material_id INTEGER DEFAULT NULL")
    except sqlite3.OperationalError:
        pass  # Column already exists

    # -------------------------------------------------------------------------
    # FocusSense Personal Learning Coach (8-Pillar Synthesis Engine)
    # -------------------------------------------------------------------------
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS coach_recommendations (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER NOT NULL,
        topic TEXT NOT NULL,
        action_type TEXT NOT NULL, -- 'START_STUDY' | 'TAKE_QUIZ' | 'REVIEW_MATERIAL' | 'RECALL_TEST'
        material_id INTEGER DEFAULT NULL,
        title TEXT NOT NULL,
        reason TEXT NOT NULL,
        priority_score INTEGER DEFAULT 50,
        recommended_duration_min INTEGER DEFAULT 25,
        status TEXT DEFAULT 'ACTIVE', -- 'ACTIVE' | 'STARTED' | 'COMPLETED' | 'DISMISSED'
        created_at TEXT NOT NULL,
        completed_at TEXT DEFAULT NULL,
        FOREIGN KEY(user_id) REFERENCES users(id),
        FOREIGN KEY(material_id) REFERENCES study_materials(id)
    )
    """)

    cursor.execute("""
    CREATE INDEX IF NOT EXISTS idx_coach_rec_user
    ON coach_recommendations(user_id)
    """)

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS coach_milestones (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER NOT NULL,
        milestone_key TEXT NOT NULL,
        title TEXT NOT NULL,
        description TEXT NOT NULL,
        category TEXT NOT NULL, -- 'MASTERY' | 'RECALL' | 'CONSISTENCY' | 'EFFICIENCY'
        earned_at TEXT NOT NULL,
        metadata_json TEXT DEFAULT '{}',
        FOREIGN KEY(user_id) REFERENCES users(id),
        UNIQUE(user_id, milestone_key)
    )
    """)

    cursor.execute("""
    CREATE INDEX IF NOT EXISTS idx_coach_milestones_user
    ON coach_milestones(user_id)
    """)

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS coach_queries (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER NOT NULL,
        query_intent TEXT NOT NULL, -- 'WHAT_NEXT' | 'WHAT_WEAK' | 'WHEN_REVISE' | 'HOW_IMPROVING' | 'CUSTOM'
        user_prompt TEXT,
        coach_response TEXT NOT NULL,
        telemetry_snapshot_json TEXT,
        created_at TEXT NOT NULL,
        FOREIGN KEY(user_id) REFERENCES users(id)
    )
    """)

    cursor.execute("""
    CREATE INDEX IF NOT EXISTS idx_coach_queries_user
    ON coach_queries(user_id)
    """)

    connection.commit()
    connection.close()

if __name__ == "__main__":
    create_database()
    print("Database Created / Updated Successfully with FocusSense Personal Learning Coach Tables!")
