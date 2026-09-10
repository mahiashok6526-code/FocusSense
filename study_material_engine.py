"""
FocusSense AI — Study Materials & Knowledge Base Engine
======================================================
Local-first Study Material Ingestion, Text Extraction, Semantic Chunking,
and Lightweight Local BM25 / TF-IDF Retrieval-Augmented Generation (RAG).

Security Guarantees:
- Strict extension allowlist (.pdf, .txt, .docx)
- Server-side 15 MB file size enforcement
- Safe UUID stored filenames
- Absolute path canonicalization and directory traversal defense
- Corrupted/unreadable files fail gracefully with error status
- Multi-tenant user isolation on all database and filesystem queries
"""

import os
import io
import re
import math
import uuid
import zipfile
import sqlite3
import logging
from datetime import datetime
import xml.etree.ElementTree as ET

try:
    import pypdf
except ImportError:
    pypdf = None

try:
    import docx
except ImportError:
    docx = None

logger = logging.getLogger("FocusSenseMaterials")

from database import DATABASE, DB_PATH

def _resolve_upload_dir():
    """
    Resolves a writable upload directory:
    - If UPLOAD_DIR environment variable is set, use it.
    - If running on Vercel or AWS Lambda serverless (read-only container):
      Use /tmp/uploads/study_materials.
    - Local execution:
      Use project uploads/study_materials.
    """
    custom = os.environ.get("UPLOAD_DIR")
    if custom:
        os.makedirs(custom, exist_ok=True)
        return custom

    is_serverless = bool(os.environ.get("VERCEL") or os.environ.get("AWS_LAMBDA_FUNCTION_NAME"))
    if is_serverless:
        tmp_dir = "/tmp/uploads/study_materials"
        os.makedirs(tmp_dir, exist_ok=True)
        return tmp_dir

    default_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "uploads", "study_materials"))
    os.makedirs(default_dir, exist_ok=True)
    return default_dir

UPLOAD_DIR = _resolve_upload_dir()

ALLOWED_EXTENSIONS = {"pdf", "txt", "docx"}
MAX_FILE_SIZE_BYTES = 15 * 1024 * 1024  # 15 MB

# Common English stopwords for BM25 keyword extraction
STOPWORDS = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are", "aren't",
    "as", "at", "be", "because", "been", "before", "being", "below", "between", "both", "but", "by",
    "can", "can't", "cannot", "could", "couldn't", "did", "didn't", "do", "does", "doesn't", "doing",
    "don't", "down", "during", "each", "few", "for", "from", "further", "had", "hadn't", "has", "hasn't",
    "have", "haven't", "having", "he", "he'd", "he'll", "he's", "her", "here", "here's", "hers", "herself",
    "him", "himself", "his", "how", "how's", "i", "i'd", "i'll", "i'm", "i've", "if", "in", "into", "is",
    "isn't", "it", "it's", "its", "itself", "let's", "me", "more", "most", "mustn't", "my", "myself", "no",
    "nor", "not", "of", "off", "on", "once", "only", "or", "other", "ought", "our", "ours", "ourselves",
    "out", "over", "own", "same", "shan't", "she", "she'd", "she'll", "she's", "should", "shouldn't", "so",
    "some", "such", "than", "that", "that's", "the", "their", "theirs", "them", "themselves", "then", "there",
    "there's", "these", "they", "they'd", "they'll", "they're", "they've", "this", "those", "through", "to",
    "too", "under", "until", "up", "very", "was", "wasn't", "we", "we'd", "we'll", "we're", "we've", "were",
    "weren't", "what", "what's", "when", "when's", "where", "where's", "which", "while", "who", "who's",
    "whom", "why", "why's", "with", "won't", "would", "wouldn't", "you", "you'd", "you'll", "you're", "you've",
    "your", "yours", "yourself", "yourselves", "tell", "explain", "describe", "give", "show", "please", "summary",
    "summarize", "detail", "details"
}


def get_db():
    conn = sqlite3.connect(DATABASE, timeout=30.0)
    conn.row_factory = sqlite3.Row
    return conn


# ---------------------------------------------------------------------------
# Path & Security Validation Helpers
# ---------------------------------------------------------------------------

def safe_resolve_material_path(stored_path: str) -> str:
    """
    Canonicalize and verify that path is strictly inside UPLOAD_DIR.
    Rejects any path outside UPLOAD_DIR (defense against directory traversal).
    """
    if not stored_path:
        raise ValueError("Stored path cannot be empty.")

    if not os.path.isabs(stored_path):
        resolved = os.path.abspath(os.path.join(UPLOAD_DIR, stored_path))
    else:
        resolved = os.path.abspath(stored_path)

    upload_dir_real = os.path.realpath(UPLOAD_DIR)
    resolved_real = os.path.realpath(resolved)

    # Must be within UPLOAD_DIR
    if not (resolved_real == upload_dir_real or resolved_real.startswith(upload_dir_real + os.path.sep)):
        raise PermissionError(f"Access Denied: Path traversal detected for '{stored_path}'")

    return resolved_real


def validate_file_metadata(filename: str, file_size: int = 0) -> tuple[bool, str, str]:
    """
    Validates file extension and size constraints.
    Returns (is_valid, sanitized_extension, error_message).
    """
    if not filename or "." not in filename:
        return False, "", "Filename is missing or does not have a valid extension."

    ext = filename.rsplit(".", 1)[-1].lower().strip()
    if ext not in ALLOWED_EXTENSIONS:
        return False, ext, f"Unsupported file format '.{ext}'. Supported formats: PDF, TXT, DOCX."

    if file_size > MAX_FILE_SIZE_BYTES:
        max_mb = MAX_FILE_SIZE_BYTES // (1024 * 1024)
        return False, ext, f"File size exceeds maximum limit of {max_mb} MB."

    return True, ext, ""


def generate_safe_filename(original_filename: str) -> str:
    """Generate a collision-free, safe storage filename with UUID prefix."""
    ext = original_filename.rsplit(".", 1)[-1].lower().strip() if "." in original_filename else "bin"
    base = os.path.splitext(os.path.basename(original_filename))[0]
    safe_base = re.sub(r'[^a-zA-Z0-9_-]', '_', base)[:40].strip("_") or "material"
    unique_id = uuid.uuid4().hex[:12]
    return f"{unique_id}_{safe_base}.{ext}"


# ---------------------------------------------------------------------------
# Text Extraction Functions
# ---------------------------------------------------------------------------

def extract_text_from_pdf(file_input) -> str:
    """Extract text from a PDF file (path or bytes) using pypdf."""
    if pypdf is None:
        raise RuntimeError("pypdf library is not installed.")

    text_parts = []
    stream = io.BytesIO(file_input) if isinstance(file_input, bytes) else open(file_input, "rb")
    try:
        reader = pypdf.PdfReader(stream)
        if reader.is_encrypted:
            try:
                reader.decrypt("")
            except Exception:
                raise ValueError("PDF is password protected and cannot be extracted.")

        for page_idx, page in enumerate(reader.pages):
            try:
                page_text = page.extract_text() or ""
                if page_text.strip():
                    text_parts.append(page_text.strip())
            except Exception as ex:
                logger.warning(f"Error extracting page {page_idx}: {ex}")
    except Exception as ex:
        logger.warning(f"pypdf extraction error: {ex}")
        if not text_parts:
            return ""
    finally:
        if stream and not isinstance(file_input, bytes):
            stream.close()

    full_text = "\n\n".join(text_parts).strip()
    return full_text


def extract_text_from_docx(file_input) -> str:
    """Extract text from a DOCX file (path or bytes) using python-docx with zip/xml fallback."""
    stream = io.BytesIO(file_input) if isinstance(file_input, bytes) else file_input

    # Method 1: python-docx
    if docx is not None:
        try:
            doc = docx.Document(stream)
            paragraphs = [p.text.strip() for p in doc.paragraphs if p.text.strip()]
            for table in doc.tables:
                for row in table.rows:
                    row_text = " | ".join(cell.text.strip() for cell in row.cells if cell.text.strip())
                    if row_text:
                        paragraphs.append(row_text)
            if paragraphs:
                return "\n\n".join(paragraphs).strip()
        except Exception as ex:
            logger.warning(f"python-docx extraction failed, attempting XML fallback: {ex}")

    # Reset stream if BytesIO
    if isinstance(stream, io.BytesIO):
        stream.seek(0)

    # Method 2: Pure Python Zip/XML fallback
    try:
        with zipfile.ZipFile(stream) as z:
            xml_content = z.read("word/document.xml")
            tree = ET.fromstring(xml_content)
            ns = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
            paragraphs = []
            for p in tree.findall(".//w:p", ns):
                texts = [node.text for node in p.findall(".//w:t", ns) if node.text]
                if texts:
                    paragraphs.append("".join(texts).strip())
            return "\n\n".join(paragraphs).strip()
    except Exception as ex:
        logger.warning(f"DOCX XML extraction error: {ex}")
        return ""


def extract_text_from_txt(file_input) -> str:
    """Extract text from TXT file (path or bytes) supporting UTF-8, UTF-8-SIG, Latin-1, and CP1252."""
    if isinstance(file_input, bytes):
        encodings = ["utf-8", "utf-8-sig", "latin-1", "cp1252"]
        for enc in encodings:
            try:
                decoded = file_input.decode(enc)
                if decoded:
                    return decoded.strip()
            except (UnicodeDecodeError, UnicodeError):
                continue
        return file_input.decode("utf-8", errors="replace").strip()

    encodings = ["utf-8", "utf-8-sig", "latin-1", "cp1252"]
    for enc in encodings:
        try:
            with open(file_input, "r", encoding=enc, errors="strict") as f:
                content = f.read()
                if content:
                    return content.strip()
        except (UnicodeDecodeError, UnicodeError):
            continue

    with open(file_input, "r", encoding="utf-8", errors="replace") as f:
        return f.read().strip()


def extract_text_from_file(file_path: str, file_type: str) -> str:
    """Dispatcher for extracting clean text across PDF, DOCX, and TXT."""
    safe_path = safe_resolve_material_path(file_path)
    if not os.path.exists(safe_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    f_type = file_type.lower().strip()
    if f_type == "pdf":
        raw_text = extract_text_from_pdf(safe_path)
    elif f_type == "docx":
        raw_text = extract_text_from_docx(safe_path)
    elif f_type == "txt":
        raw_text = extract_text_from_txt(safe_path)
    else:
        raise ValueError(f"Unsupported file type: {file_type}")

    # Clean text: remove excessive control characters and redundant whitespace
    cleaned = re.sub(r'[\r\f\v]', '\n', raw_text)
    cleaned = re.sub(r'\n{3,}', '\n\n', cleaned)
    cleaned = re.sub(r'[ \t]+', ' ', cleaned).strip()

    if len(cleaned) < 20:
        raise ValueError("Document contains insufficient extractable text (empty or non-text content).")

    return cleaned


# ---------------------------------------------------------------------------
# Semantic Chunking
# ---------------------------------------------------------------------------

def chunk_text(text: str, chunk_size: int = 700, overlap: int = 100) -> list[str]:
    """
    Paragraph and sentence-aware sliding window chunker.
    Target: ~500–800 characters (~120–180 words) with ~100 characters overlap.
    Preserves sentence and paragraph boundaries rather than splitting words in half.
    """
    if not text:
        return []

    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    if not paragraphs:
        paragraphs = [text.strip()]

    # Break large paragraphs into sentences
    sentences = []
    for para in paragraphs:
        # Split by sentence boundaries (.!?) followed by space or newline
        para_sents = re.split(r'(?<=[.!?])\s+', para)
        for s in para_sents:
            s_clean = s.strip()
            if s_clean:
                sentences.append(s_clean)
        # Sentinel to separate paragraphs
        sentences.append("\n\n")

    chunks = []
    current_chunk = []
    current_len = 0

    for item in sentences:
        if item == "\n\n":
            # If paragraph break and we have substantial content, finalize chunk
            if current_len >= chunk_size - 150:
                chunk_str = " ".join(current_chunk).strip()
                if chunk_str:
                    chunks.append(chunk_str)
                # Keep last 1-2 sentences for overlap
                overlap_items = []
                overlap_len = 0
                for prev in reversed(current_chunk):
                    if overlap_len + len(prev) <= overlap:
                        overlap_items.insert(0, prev)
                        overlap_len += len(prev)
                    else:
                        break
                current_chunk = overlap_items
                current_len = sum(len(x) + 1 for x in current_chunk)
            continue

        sent_len = len(item)

        # If a single sentence is larger than chunk_size, split by sub-clauses or words
        if sent_len > chunk_size:
            words = item.split()
            word_buf = []
            word_len = 0
            for w in words:
                if word_len + len(w) + 1 > chunk_size and word_buf:
                    chunks.append(" ".join(word_buf).strip())
                    word_buf = word_buf[-10:]  # small word overlap
                    word_len = sum(len(x) + 1 for x in word_buf)
                word_buf.append(w)
                word_len += len(w) + 1
            if word_buf:
                current_chunk.append(" ".join(word_buf))
                current_len += word_len
            continue

        if current_len + sent_len + 1 > chunk_size and current_chunk:
            chunk_str = " ".join(current_chunk).strip()
            if chunk_str:
                chunks.append(chunk_str)
            # Create overlap
            overlap_items = []
            overlap_len = 0
            for prev in reversed(current_chunk):
                if overlap_len + len(prev) <= overlap:
                    overlap_items.insert(0, prev)
                    overlap_len += len(prev)
                else:
                    break
            current_chunk = overlap_items + [item]
            current_len = sum(len(x) + 1 for x in current_chunk)
        else:
            current_chunk.append(item)
            current_len += sent_len + 1

    if current_chunk:
        chunk_str = " ".join(current_chunk).strip()
        if chunk_str and (not chunks or chunks[-1] != chunk_str):
            chunks.append(chunk_str)

    # Filter out empty or tiny trailing chunks
    clean_chunks = [c for c in chunks if len(c.strip()) >= 20]
    return clean_chunks if clean_chunks else [text[:chunk_size]]


# ---------------------------------------------------------------------------
# Lightweight Local BM25 / TF-IDF Retrieval
# ---------------------------------------------------------------------------

def tokenize(text: str) -> list[str]:
    """Tokenize text into lowercase alphanumeric keywords excluding stopwords."""
    words = re.findall(r'[a-z0-9_]+', text.lower())
    return [w for w in words if len(w) > 2 and w not in STOPWORDS]


def compute_bm25_scores(query: str, chunks: list[dict], title_boost: str = "", topic_boost: str = "") -> list[tuple[dict, float]]:
    """
    Computes BM25 relevance scores for candidate chunks against the user's query.
    Includes keyword frequency, length normalization, topic/title bonus, and phrase match bonus.
    """
    query_tokens = tokenize(query)
    if not query_tokens or not chunks:
        return [(c, 0.0) for c in chunks]

    # Term IDF across the candidate chunks
    num_docs = len(chunks)
    doc_freqs = {}
    doc_tokens_list = []
    doc_lens = []

    for c in chunks:
        tokens = tokenize(c["chunk_text"])
        doc_tokens_list.append(tokens)
        doc_lens.append(len(tokens))
        unique_tokens = set(tokens)
        for t in unique_tokens:
            doc_freqs[t] = doc_freqs.get(t, 0) + 1

    avg_dl = max(1.0, sum(doc_lens) / max(1, num_docs))
    k1 = 1.5
    b = 0.75

    # Context boost tokens from title, subject, and topic
    boost_tokens = set(tokenize(f"{title_boost} {topic_boost}"))

    # Lowercase query for exact phrase match
    query_clean = " ".join(query_tokens)

    results = []
    for idx, c in enumerate(chunks):
        doc_tokens = doc_tokens_list[idx]
        doc_len = doc_lens[idx]
        text_lower = c["chunk_text"].lower()

        tf_dict = {}
        for t in doc_tokens:
            tf_dict[t] = tf_dict.get(t, 0) + 1

        bm25_score = 0.0
        for q_term in query_tokens:
            if q_term in tf_dict:
                tf = tf_dict[q_term]
                df = doc_freqs.get(q_term, 1)
                # Standard BM25 IDF formulation
                idf = math.log((num_docs - df + 0.5) / (df + 0.5) + 1.0)
                # BM25 Term Weight
                term_weight = (tf * (k1 + 1.0)) / (tf + k1 * (1.0 - b + b * (doc_len / avg_dl)))
                bm25_score += idf * term_weight

                # Bonus if term is in topic/title
                if q_term in boost_tokens:
                    bm25_score += 0.4

        # Bonus for contiguous exact substring match
        if len(query_clean) > 8 and query_clean in text_lower:
            bm25_score += 2.5
        elif any(len(q_term) > 4 and q_term in text_lower for q_term in query_tokens):
            bm25_score += 0.2

        results.append((c, bm25_score))

    return results


def retrieve_relevant_chunks(user_id: int, query: str, material_id: int = None, top_k: int = 4) -> list[dict]:
    """
    Retrieve the top-K most relevant chunks for a user query.
    Enforces user isolation on all SQL queries.
    """
    conn = get_db()
    cursor = conn.cursor()

    if material_id:
        # Verify material belongs to user
        cursor.execute("SELECT id, title, subject, topic FROM study_materials WHERE id = ? AND user_id = ?", (material_id, user_id))
        mat = cursor.fetchone()
        if not mat:
            conn.close()
            return []

        title_boost = mat["title"] or ""
        topic_boost = f"{mat['subject'] or ''} {mat['topic'] or ''}"

        cursor.execute("""
            SELECT id, material_id, chunk_index, chunk_text, char_count
            FROM material_chunks
            WHERE material_id = ? AND user_id = ?
            ORDER BY chunk_index ASC
        """, (material_id, user_id))
    else:
        title_boost = ""
        topic_boost = ""
        cursor.execute("""
            SELECT id, material_id, chunk_index, chunk_text, char_count
            FROM material_chunks
            WHERE user_id = ?
            ORDER BY id ASC LIMIT 300
        """, (user_id,))

    rows = cursor.fetchall()
    conn.close()

    if not rows:
        return []

    candidate_chunks = [
        {
            "id": r["id"],
            "material_id": r["material_id"],
            "chunk_index": r["chunk_index"],
            "chunk_text": r["chunk_text"],
            "char_count": r["char_count"]
        }
        for r in rows
    ]

    scored = compute_bm25_scores(query, candidate_chunks, title_boost=title_boost, topic_boost=topic_boost)
    scored.sort(key=lambda x: x[1], reverse=True)

    top_results = []
    for chunk, score in scored[:top_k]:
        chunk_copy = dict(chunk)
        chunk_copy["relevance_score"] = round(score, 4)
        top_results.append(chunk_copy)

    return top_results


def build_material_grounded_context(user_id: int, query: str, material_id: int) -> dict:
    """
    Build structured grounding payload for AI prompts.
    Performs honest grounding check: flags `is_relevant` based on BM25 match quality.
    """
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT id, title, subject, topic, filename, chunk_count, status
        FROM study_materials
        WHERE id = ? AND user_id = ?
    """, (material_id, user_id))
    material = cursor.fetchone()
    conn.close()

    if not material:
        return {
            "found": False,
            "material_id": material_id,
            "is_relevant": False,
            "error": "Study material not found or access denied."
        }

    # If query is a general high-level material prompt (e.g. summarize, explain material)
    q_lower = query.lower().strip()
    is_general_material_request = any(w in q_lower for w in [
        "summarize", "summary", "explain this material", "explain this chapter",
        "what is this about", "important concepts", "important topics",
        "generate 10 questions", "generate questions", "generate mcqs", "test me", "teach me"
    ])

    retrieved = retrieve_relevant_chunks(user_id=user_id, query=query, material_id=material_id, top_k=5)

    # Determine relevance:
    # If general material request, the representative chunks are unconditionally relevant.
    # Otherwise, check if at least one chunk scored above a meaningful BM25 threshold (> 0.25).
    max_score = max([c.get("relevance_score", 0.0) for c in retrieved], default=0.0)
    is_relevant = is_general_material_request or (max_score >= 0.25)

    # If general request and retrieval scored low, pick top 4 initial/representative chunks
    if is_general_material_request and max_score < 0.1:
        conn = get_db()
        cursor = conn.cursor()
        cursor.execute("""
            SELECT id, material_id, chunk_index, chunk_text, char_count
            FROM material_chunks
            WHERE material_id = ? AND user_id = ?
            ORDER BY chunk_index ASC LIMIT 4
        """, (material_id, user_id))
        rows = cursor.fetchall()
        conn.close()
        retrieved = [
            {
                "id": r["id"],
                "material_id": r["material_id"],
                "chunk_index": r["chunk_index"],
                "chunk_text": r["chunk_text"],
                "char_count": r["char_count"],
                "relevance_score": 1.0
            }
            for r in rows
        ]
        is_relevant = True

    # Construct context block with max context cap (~6000 chars)
    formatted_excerpts = []
    total_len = 0
    for idx, c in enumerate(retrieved, start=1):
        excerpt = f"=== EXCERPT {idx} (Section Chunk #{c['chunk_index']}) ===\n{c['chunk_text']}"
        if total_len + len(excerpt) > 6000:
            break
        formatted_excerpts.append(excerpt)
        total_len += len(excerpt)

    context_text = "\n\n".join(formatted_excerpts)

    return {
        "found": True,
        "material_id": material["id"],
        "material_title": material["title"],
        "subject": material["subject"] or "Academic Study",
        "topic": material["topic"] or material["title"],
        "filename": material["filename"],
        "is_relevant": is_relevant,
        "max_score": max_score,
        "chunks": retrieved,
        "context_text": context_text,
        "total_chars": len(context_text)
    }


# ---------------------------------------------------------------------------
# Material Management & Database Operations
# ---------------------------------------------------------------------------

def save_and_process_material(user_id: int, file_storage=None, original_filename: str = "", file_bytes: bytes = None, title: str = "", subject: str = "", topic: str = "") -> dict:
    """
    Complete pipeline:
    1. Validates file metadata and constraints.
    2. Saves file safely with UUID filename in UPLOAD_DIR.
    3. Extracts text.
    4. Generates chunks.
    5. Saves records in SQLite (study_materials, material_chunks).
    """
    if file_storage is not None and hasattr(file_storage, "filename") and file_storage.filename:
        orig_name = file_storage.filename.strip()
        file_storage.seek(0, os.SEEK_END)
        file_size = file_storage.tell()
        file_storage.seek(0)
        raw_bytes = None
    elif file_bytes is not None and original_filename:
        orig_name = original_filename.strip()
        file_size = len(file_bytes)
        raw_bytes = file_bytes
    else:
        return {"success": False, "error": "No file was selected for upload."}

    is_valid, ext, err_msg = validate_file_metadata(orig_name, file_size)
    if not is_valid:
        return {"success": False, "error": err_msg}

    # Generate safe unique filename and target path
    safe_filename = generate_safe_filename(orig_name)
    target_path = os.path.join(UPLOAD_DIR, safe_filename)

    # Save to disk
    try:
        if file_storage is not None:
            file_storage.save(target_path)
        else:
            with open(target_path, "wb") as f:
                f.write(raw_bytes)
    except Exception as ex:
        logger.error(f"Failed to save uploaded file: {ex}")
        return {"success": False, "error": f"Failed to save file to server: {str(ex)}"}

    display_title = (title or "").strip() or os.path.splitext(orig_name)[0].replace("_", " ").title()
    subject_str = (subject or "").strip()
    topic_str = (topic or "").strip() or display_title

    now_iso = datetime.now().isoformat()

    conn = get_db()
    cursor = conn.cursor()

    # Insert initial PROCESSING record
    cursor.execute("""
        INSERT INTO study_materials
        (user_id, filename, stored_path, file_type, file_size_bytes, title, subject, topic, chunk_count, status, created_at, updated_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, 0, 'PROCESSING', ?, ?)
    """, (user_id, orig_name, safe_filename, ext, file_size, display_title, subject_str, topic_str, now_iso, now_iso))
    material_id = cursor.lastrowid
    conn.commit()

    # Process extraction and chunking
    try:
        extracted_text = extract_text_from_file(target_path, ext)
        chunks = chunk_text(extracted_text)

        if not chunks:
            raise ValueError("No usable text chunks could be generated from the document.")

        for idx, chk in enumerate(chunks, start=1):
            cursor.execute("""
                INSERT INTO material_chunks (material_id, user_id, chunk_index, chunk_text, char_count, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (material_id, user_id, idx, chk, len(chk), now_iso))

        cursor.execute("""
            UPDATE study_materials
            SET chunk_count = ?, status = 'READY', updated_at = ?
            WHERE id = ? AND user_id = ?
        """, (len(chunks), datetime.now().isoformat(), material_id, user_id))
        conn.commit()

        return {
            "success": True,
            "material_id": material_id,
            "title": display_title,
            "subject": subject_str,
            "topic": topic_str,
            "filename": orig_name,
            "file_type": ext,
            "file_size_bytes": file_size,
            "chunk_count": len(chunks),
            "stored_path": target_path,
            "status": "READY"
        }

    except Exception as ex:
        logger.error(f"Error processing material {material_id}: {ex}")
        err_str = str(ex)[:250]
        cursor.execute("""
            UPDATE study_materials
            SET status = 'FAILED', error_message = ?, updated_at = ?
            WHERE id = ? AND user_id = ?
        """, (err_str, datetime.now().isoformat(), material_id, user_id))
        conn.commit()
        return {
            "success": False,
            "material_id": material_id,
            "error": f"Text processing failed: {err_str}",
            "status": "FAILED"
        }
    finally:
        conn.close()


def get_user_materials(user_id: int) -> list[dict]:
    """Retrieve all study materials uploaded by the user."""
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT id, user_id, filename, file_type, file_size_bytes, title, subject, topic,
               chunk_count, status, error_message, created_at, updated_at
        FROM study_materials
        WHERE user_id = ?
        ORDER BY id DESC
    """, (user_id,))
    rows = cursor.fetchall()
    conn.close()

    materials = []
    for r in rows:
        materials.append({
            "id": r["id"],
            "user_id": r["user_id"],
            "filename": r["filename"],
            "file_type": r["file_type"],
            "file_size_bytes": r["file_size_bytes"],
            "file_size_kb": round(r["file_size_bytes"] / 1024, 1),
            "title": r["title"],
            "subject": r["subject"] or "",
            "topic": r["topic"] or "",
            "chunk_count": r["chunk_count"],
            "status": r["status"],
            "error_message": r["error_message"] or "",
            "created_at": r["created_at"],
            "updated_at": r["updated_at"]
        })
    return materials


def get_material_by_id(material_id: int, user_id: int) -> dict:
    """Retrieve single material details and sample chunks with strict ownership check."""
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT id, user_id, filename, stored_path, file_type, file_size_bytes, title, subject, topic,
               chunk_count, status, error_message, created_at, updated_at
        FROM study_materials
        WHERE id = ? AND user_id = ?
    """, (material_id, user_id))
    r = cursor.fetchone()

    if not r:
        conn.close()
        return None

    cursor.execute("""
        SELECT id, chunk_index, chunk_text, char_count
        FROM material_chunks
        WHERE material_id = ? AND user_id = ?
        ORDER BY chunk_index ASC LIMIT 20
    """, (material_id, user_id))
    chunk_rows = cursor.fetchall()
    conn.close()

    return {
        "id": r["id"],
        "user_id": r["user_id"],
        "filename": r["filename"],
        "file_type": r["file_type"],
        "file_size_bytes": r["file_size_bytes"],
        "file_size_kb": round(r["file_size_bytes"] / 1024, 1),
        "title": r["title"],
        "subject": r["subject"] or "",
        "topic": r["topic"] or "",
        "chunk_count": r["chunk_count"],
        "status": r["status"],
        "error_message": r["error_message"] or "",
        "created_at": r["created_at"],
        "updated_at": r["updated_at"],
        "sample_chunks": [
            {
                "id": c["id"],
                "chunk_index": c["chunk_index"],
                "chunk_text": c["chunk_text"],
                "char_count": c["char_count"]
            }
            for c in chunk_rows
        ]
    }


def get_material_chunks(material_id: int, user_id: int) -> list[dict]:
    """Retrieve all semantic chunks for a specific material owned by user."""
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT id, material_id, user_id, chunk_index, chunk_text, char_count, created_at
        FROM material_chunks
        WHERE material_id = ? AND user_id = ?
        ORDER BY chunk_index ASC
    """, (material_id, user_id))
    rows = cursor.fetchall()
    conn.close()

    return [
        {
            "id": r["id"],
            "material_id": r["material_id"],
            "user_id": r["user_id"],
            "chunk_index": r["chunk_index"],
            "chunk_text": r["chunk_text"],
            "char_count": r["char_count"],
            "created_at": r["created_at"]
        }
        for r in rows
    ]


def delete_user_material(material_id: int, user_id: int) -> dict:
    """
    Deletes material, associated chunks, and stored physical file.
    Validates ownership and verifies physical path safety.
    """
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("SELECT id, stored_path FROM study_materials WHERE id = ? AND user_id = ?", (material_id, user_id))
    mat = cursor.fetchone()

    if not mat:
        conn.close()
        return {"success": False, "error": "Material not found or unauthorized access."}

    stored_path = mat["stored_path"]

    # Delete DB records
    cursor.execute("DELETE FROM material_chunks WHERE material_id = ? AND user_id = ?", (material_id, user_id))
    cursor.execute("DELETE FROM study_materials WHERE id = ? AND user_id = ?", (material_id, user_id))
    conn.commit()
    conn.close()

    # Safely delete physical file
    try:
        resolved_file = safe_resolve_material_path(stored_path)
        if os.path.exists(resolved_file):
            os.remove(resolved_file)
    except Exception as ex:
        logger.warning(f"Error safely removing physical file for material {material_id}: {ex}")

    return {"success": True, "message": "Study material deleted successfully."}
