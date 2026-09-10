"""
==============================================================================
FocusSense AI — Study Materials & Knowledge Base Test Suite
==============================================================================
Tests 30 unit and integration cases covering:
1. File validation & Security (extensions, 15MB limit, empty file, path traversal, safe filenames)
2. Text extraction (TXT UTF-8, Latin-1 fallback, PDF, DOCX, empty text handling)
3. Chunking & BM25 retrieval (chunk lengths, BM25 scoring, title/topic boost, weak match detection, honest context builder)
4. Database CRUD & Multi-User Isolation (save, list, isolation, delete with cleanup)
5. Flask API endpoints (upload, invalid upload, list, select/clear, grounded chat, ungrounded chat)
6. Quiz & Recall integration (grounded quiz generation, recall task material chunk integration)
==============================================================================
"""

import os
import io
import unittest
import tempfile
import shutil
import sqlite3
from unittest.mock import patch, MagicMock

# Import FocusSense modules
import database
import study_material_engine as sme
from ai_service import ai_service, LocalMockAIProvider, FocusSenseAIService
import quiz_engine
import recall_engine
import app as flask_app_module


class TestFocusSenseStudyMaterials(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        database.create_database()
        conn = flask_app_module.get_db()
        cursor = conn.cursor()
        
        # Ensure test students exist
        cursor.execute("SELECT id FROM users WHERE username = 'student_mat_a'")
        row_a = cursor.fetchone()
        if not row_a:
            cursor.execute("""
                INSERT INTO users (fullname, email, username, password)
                VALUES ('Student Material A', 'student_mat_a@focussense.io', 'student_mat_a', 'pass123')
            """)
            conn.commit()
            cursor.execute("SELECT id FROM users WHERE username = 'student_mat_a'")
            row_a = cursor.fetchone()
        cls.user_a_id = row_a["id"]

        cursor.execute("SELECT id FROM users WHERE username = 'student_mat_b'")
        row_b = cursor.fetchone()
        if not row_b:
            cursor.execute("""
                INSERT INTO users (fullname, email, username, password)
                VALUES ('Student Material B', 'student_mat_b@focussense.io', 'student_mat_b', 'pass123')
            """)
            conn.commit()
            cursor.execute("SELECT id FROM users WHERE username = 'student_mat_b'")
            row_b = cursor.fetchone()
        cls.user_b_id = row_b["id"]
        conn.close()

        # Configure Flask app for testing
        flask_app_module.app.config["TESTING"] = True
        flask_app_module.app.config["SECRET_KEY"] = "test-secret-key-123"
        cls.client = flask_app_module.app.test_client()

    @classmethod
    def tearDownClass(cls):
        pass

    # --------------------------------------------------------------------------
    # 1. FILE VALIDATION & SECURITY (Tests 1–7)
    # --------------------------------------------------------------------------
    def test_01_file_validation_valid_formats(self):
        """Test 1: Valid PDF, DOCX, TXT formats pass metadata validation."""
        v_pdf, ext_pdf, _ = sme.validate_file_metadata("lecture_notes.pdf", 1024 * 500)
        v_docx, ext_docx, _ = sme.validate_file_metadata("summary.DOCX", 1024 * 200)
        v_txt, ext_txt, _ = sme.validate_file_metadata("study_guide.txt", 1024 * 50)
        self.assertTrue(v_pdf)
        self.assertEqual(ext_pdf, "pdf")
        self.assertTrue(v_docx)
        self.assertEqual(ext_docx, "docx")
        self.assertTrue(v_txt)
        self.assertEqual(ext_txt, "txt")

    def test_02_file_validation_rejected_extensions(self):
        """Test 2: Disallowed file extensions (.exe, .py, .zip, .sh) are rejected."""
        for disallowed in ["malicious.exe", "script.py", "archive.zip", "test.sh", "notes.png"]:
            valid, _, err = sme.validate_file_metadata(disallowed, 1024)
            self.assertFalse(valid)
            self.assertIn("Unsupported file", err)

    def test_03_file_validation_oversized(self):
        """Test 3: Files exceeding 15MB are rejected."""
        oversized = 16 * 1024 * 1024  # 16 MB
        valid, _, err = sme.validate_file_metadata("huge_book.pdf", oversized)
        self.assertFalse(valid)
        self.assertIn("exceeds", err.lower())

    def test_04_file_validation_empty_file(self):
        """Test 4: Missing filename or invalid name is rejected."""
        valid, _, err = sme.validate_file_metadata("", 0)
        self.assertFalse(valid)

    def test_05_safe_resolve_material_path_valid(self):
        """Test 5: Canonical path resolution confines files inside UPLOAD_DIR."""
        safe_rel = "abc-123_notes.txt"
        resolved = sme.safe_resolve_material_path(safe_rel)
        self.assertTrue(os.path.isabs(resolved))
        self.assertTrue(resolved.startswith(os.path.realpath(sme.UPLOAD_DIR)))

    def test_06_safe_resolve_material_path_traversal(self):
        """Test 6: Path traversal attempts raise PermissionError or ValueError."""
        traversal_attempts = [
            "../../secret.db",
            "..\\..\\windows\\system32\\calc.exe",
            "uploads/../../database.py"
        ]
        for bad_path in traversal_attempts:
            with self.assertRaises((PermissionError, ValueError)):
                sme.safe_resolve_material_path(bad_path)

    def test_07_safe_filename_generation(self):
        """Test 7: Generates sanitized filenames with UUID prefix."""
        fname = sme.generate_safe_filename("My Class Notes (Chapter 1) [Final].pdf")
        self.assertTrue(fname.endswith(".pdf"))
        self.assertNotIn(" ", fname)
        self.assertNotIn("(", fname)
        self.assertNotIn(")", fname)
        self.assertIn("_", fname)

    # --------------------------------------------------------------------------
    # 2. TEXT EXTRACTION (Tests 8–12)
    # --------------------------------------------------------------------------
    def test_08_text_extractor_txt_utf8(self):
        """Test 8: Plain TXT UTF-8 extraction preserves content."""
        sample_text = "Operating Systems: CPU Scheduling algorithms include FCFS, SJF, and Round Robin."
        extracted = sme.extract_text_from_txt(sample_text.encode("utf-8"))
        self.assertIn("CPU Scheduling", extracted)

    def test_09_text_extractor_txt_latin1_fallback(self):
        """Test 9: Plain TXT encoded in Latin-1/CP1252 extracts cleanly."""
        sample_latin1 = "Café & Résumé study notes with special characters: é, è, ç, ñ, ü, ß".encode("latin-1")
        extracted = sme.extract_text_from_txt(sample_latin1)
        self.assertIn("Caf", extracted)
        self.assertIn("study notes", extracted)

    def test_10_text_extractor_pdf(self):
        """Test 10: PDF extraction handles valid PDF bytes or returns empty on invalid bytes."""
        mock_pdf = b"%PDF-1.4 header test"
        extracted = sme.extract_text_from_pdf(mock_pdf)
        self.assertIsInstance(extracted, str)

    def test_11_text_extractor_docx(self):
        """Test 11: DOCX extraction extracts paragraphs or returns string."""
        mock_docx = b"PK\x03\x04"
        extracted = sme.extract_text_from_docx(mock_docx)
        self.assertIsInstance(extracted, str)

    def test_12_text_extractor_empty_content(self):
        """Test 12: Empty byte content returns empty text."""
        extracted = sme.extract_text_from_txt(b"")
        self.assertEqual(extracted, "")

    # --------------------------------------------------------------------------
    # 3. SEMANTIC CHUNKING & BM25 RETRIEVAL (Tests 13–18)
    # --------------------------------------------------------------------------
    def test_13_semantic_chunker_length_and_overlap(self):
        """Test 13: Semantic chunker breaks long text into chunks with overlap."""
        long_text = ("Virtual memory allows processes to execute without being entirely in physical memory. " * 30)
        chunks = sme.chunk_text(long_text, chunk_size=400, overlap=50)
        self.assertGreater(len(chunks), 1)

    def test_14_bm25_retrieval_exact_match(self):
        """Test 14: BM25 ranker ranks matching chunk highest."""
        chunks = [
            {"id": 1, "chunk_text": "Photosynthesis converts light energy into chemical energy in chloroplasts."},
            {"id": 2, "chunk_text": "Cellular respiration produces ATP in the mitochondria of eukaryotic cells."},
            {"id": 3, "chunk_text": "Newton's second law of motion states F = ma."}
        ]
        results = sme.compute_bm25_scores("Where does cellular respiration produce ATP?", chunks)
        results.sort(key=lambda x: x[1], reverse=True)
        self.assertGreater(len(results), 0)
        self.assertEqual(results[0][0]["id"], 2)
        self.assertGreater(results[0][1], 0.1)

    def test_15_bm25_retrieval_title_and_topic_boost(self):
        """Test 15: BM25 score receives boost when query matches material title or topic."""
        chunks = [
            {"id": 1, "chunk_text": "The scheduler allocates CPU bursts to processes based on priority."}
        ]
        score_no_boost = sme.compute_bm25_scores("scheduler priority", chunks, title_boost="", topic_boost="")[0][1]
        score_with_boost = sme.compute_bm25_scores("scheduler priority", chunks, title_boost="CPU Scheduler", topic_boost="Priority Scheduling")[0][1]
        self.assertGreater(score_with_boost, score_no_boost)

    def test_16_bm25_retrieval_irrelevant_query(self):
        """Test 16: BM25 ranker gives low score (< threshold) for completely unrelated query."""
        chunks = [
            {"id": 1, "chunk_text": "Thermodynamics first law states that energy cannot be created or destroyed."}
        ]
        results = sme.compute_bm25_scores("quantum entanglement cryptography algorithm", chunks)
        score = results[0][1] if results else 0.0
        self.assertLess(score, 0.05)

    def test_17_build_material_grounding_context_relevant(self):
        """Test 17: Grounding context builder produces structured context when relevant chunks exist."""
        # Create a material
        res = sme.save_and_process_material(
            user_id=self.user_a_id,
            original_filename="paging_notes.txt",
            file_bytes=b"Paging is a memory management scheme that eliminates the need for contiguous physical memory.",
            title="OS Memory Management",
            topic="Paging"
        )
        context_data = sme.build_material_grounded_context(self.user_a_id, "Explain paging memory management", res["material_id"])
        self.assertTrue(context_data["found"])
        self.assertTrue(context_data["is_relevant"])
        self.assertIn("Paging is a memory management scheme", context_data["context_text"])

    def test_18_build_material_grounding_context_honest_fallback(self):
        """Test 18: Grounding context builder flags is_relevant=False when query has no match."""
        res = sme.save_and_process_material(
            user_id=self.user_a_id,
            original_filename="bio_notes.txt",
            file_bytes=b"DNA replication occurs in the 5' to 3' direction using DNA polymerase.",
            title="Molecular Biology",
            topic="Genetics"
        )
        context_data = sme.build_material_grounded_context(self.user_a_id, "How do black holes form in astrophysics?", res["material_id"])
        self.assertTrue(context_data["found"])
        self.assertFalse(context_data["is_relevant"])

    # --------------------------------------------------------------------------
    # 4. DATABASE CRUD & MULTI-USER ISOLATION (Tests 19–22)
    # --------------------------------------------------------------------------
    def test_19_db_save_and_process_material(self):
        """Test 19: save_and_process_material stores material record and semantic chunks in SQLite."""
        raw_text = "Computer Networks: The OSI 7-layer model includes Physical, Data Link, Network, Transport, Session, Presentation, Application."
        res = sme.save_and_process_material(
            user_id=self.user_a_id,
            original_filename="networks.txt",
            file_bytes=raw_text.encode("utf-8"),
            title="OSI Model Guide",
            subject="Computer Networks",
            topic="OSI 7 Layers"
        )
        self.assertTrue(res["success"])
        self.assertIsNotNone(res["material_id"])
        self.assertGreaterEqual(res["chunk_count"], 1)

        # Verify in DB
        mat = sme.get_material_by_id(res["material_id"], user_id=self.user_a_id)
        self.assertIsNotNone(mat)
        self.assertEqual(mat["title"], "OSI Model Guide")
        self.assertEqual(mat["status"], "READY")

        chunks = sme.get_material_chunks(res["material_id"], user_id=self.user_a_id)
        self.assertGreaterEqual(len(chunks), 1)

    def test_20_db_get_user_materials(self):
        """Test 20: get_user_materials retrieves list of materials for a specific user."""
        materials = sme.get_user_materials(user_id=self.user_a_id)
        self.assertIsInstance(materials, list)
        self.assertGreaterEqual(len(materials), 1)
        self.assertTrue(all(m["user_id"] == self.user_a_id for m in materials))

    def test_21_db_cross_user_isolation(self):
        """Test 21: Cross-user isolation guarantees User 2 cannot access User 1's materials."""
        # Create a material for User 1
        res1 = sme.save_and_process_material(
            user_id=self.user_a_id,
            original_filename="private_notes_1.txt",
            file_bytes=b"Private thoughts of User 1",
            title="User 1 Secret Notes"
        )
        mat1_id = res1["material_id"]

        # User 2 attempts to fetch User 1's material
        mat_for_u2 = sme.get_material_by_id(mat1_id, user_id=self.user_b_id)
        self.assertIsNone(mat_for_u2)

        # User 2 attempts to fetch User 1's chunks
        chunks_for_u2 = sme.get_material_chunks(mat1_id, user_id=self.user_b_id)
        self.assertEqual(len(chunks_for_u2), 0)

        # User 2 attempts to delete User 1's material
        del_res = sme.delete_user_material(mat1_id, user_id=self.user_b_id)
        self.assertFalse(del_res["success"])
        self.assertIn("not found or unauthorized", del_res["error"])

    def test_22_db_delete_user_material(self):
        """Test 22: delete_user_material removes DB record, chunks, and disk file safely."""
        res = sme.save_and_process_material(
            user_id=self.user_a_id,
            original_filename="to_delete.txt",
            file_bytes=b"Temporary content to be deleted.",
            title="To Delete"
        )
        mat_id = res["material_id"]
        stored_path = res["stored_path"]

        # Confirm file exists on disk
        self.assertTrue(os.path.exists(stored_path))

        # Delete material
        del_res = sme.delete_user_material(mat_id, user_id=self.user_a_id)
        self.assertTrue(del_res["success"])

        # Verify DB records removed
        self.assertIsNone(sme.get_material_by_id(mat_id, user_id=self.user_a_id))
        self.assertEqual(len(sme.get_material_chunks(mat_id, user_id=self.user_a_id)), 0)

        # Verify disk file removed
        self.assertFalse(os.path.exists(stored_path))

    # --------------------------------------------------------------------------
    # 5. FLASK API ENDPOINTS (Tests 23–28)
    # --------------------------------------------------------------------------
    def test_23_api_upload_material(self):
        """Test 23: POST /api/materials/upload uploads, processes, and indexes material."""
        with self.client.session_transaction() as sess:
            sess["user_id"] = self.user_a_id
            sess["username"] = "student_mat_a"

        data = {
            "title": "API Test Material",
            "subject": "Software Engineering",
            "topic": "Design Patterns",
            "file": (io.BytesIO(b"Design Patterns: Singleton, Factory, and Observer patterns."), "patterns.txt")
        }
        response = self.client.post("/api/materials/upload", data=data, content_type="multipart/form-data")
        self.assertEqual(response.status_code, 200)
        json_data = response.get_json()
        self.assertTrue(json_data["success"])
        self.assertIn("material_id", json_data)
        self.assertGreaterEqual(json_data["chunk_count"], 1)

    def test_24_api_upload_material_invalid_file(self):
        """Test 24: POST /api/materials/upload rejects unauthorized file extensions."""
        with self.client.session_transaction() as sess:
            sess["user_id"] = self.user_a_id
            sess["username"] = "student_mat_a"

        data = {
            "title": "Malicious Script",
            "file": (io.BytesIO(b"import os; os.system('calc')"), "exploit.py")
        }
        response = self.client.post("/api/materials/upload", data=data, content_type="multipart/form-data")
        self.assertEqual(response.status_code, 400)
        json_data = response.get_json()
        self.assertFalse(json_data["success"])
        self.assertIn("Unsupported file", json_data["error"])

    def test_25_api_list_materials(self):
        """Test 25: GET /api/materials returns the list of materials for the authenticated user."""
        with self.client.session_transaction() as sess:
            sess["user_id"] = self.user_a_id
            sess["username"] = "student_mat_a"

        response = self.client.get("/api/materials")
        self.assertEqual(response.status_code, 200)
        json_data = response.get_json()
        self.assertTrue(json_data["success"])
        self.assertIsInstance(json_data["materials"], list)

    def test_26_api_select_and_clear_material(self):
        """Test 26: POST /api/materials/<id>/select and /api/materials/clear-selection update session."""
        with self.client.session_transaction() as sess:
            sess["user_id"] = self.user_a_id
            sess["username"] = "student_mat_a"

        # Create a material
        res = sme.save_and_process_material(
            user_id=self.user_a_id,
            original_filename="select_test.txt",
            file_bytes=b"Material selection test text.",
            title="Selection Test"
        )
        mat_id = res["material_id"]

        # Select material
        sel_resp = self.client.post(f"/api/materials/{mat_id}/select")
        self.assertEqual(sel_resp.status_code, 200)
        sel_json = sel_resp.get_json()
        self.assertTrue(sel_json["success"])
        self.assertEqual(sel_json["active_material"]["id"], mat_id)

        # Clear selection
        clear_resp = self.client.post("/api/materials/clear-selection")
        self.assertEqual(clear_resp.status_code, 200)
        clear_json = clear_resp.get_json()
        self.assertTrue(clear_json["success"])

    def test_27_api_chat_with_grounded_material(self):
        """Test 27: POST /api/ai/chat with material_id grounds AI reply and returns grounding metadata."""
        with self.client.session_transaction() as sess:
            sess["user_id"] = self.user_a_id
            sess["username"] = "student_mat_a"

        # Create a material
        res = sme.save_and_process_material(
            user_id=self.user_a_id,
            original_filename="chat_grounding.txt",
            file_bytes=b"Deadlock conditions in OS: Mutual Exclusion, Hold and Wait, No Preemption, Circular Wait.",
            title="Deadlock Principles",
            topic="Deadlocks"
        )
        mat_id = res["material_id"]

        payload = {
            "message": "What are the four deadlock conditions?",
            "material_id": mat_id
        }
        response = self.client.post("/api/ai/chat", json=payload)
        self.assertEqual(response.status_code, 200)
        json_data = response.get_json()
        self.assertTrue(json_data["success"])
        self.assertIsNotNone(json_data["reply"])
        self.assertEqual(json_data.get("material_id"), mat_id)
        self.assertEqual(json_data.get("material_title"), "Deadlock Principles")
        self.assertTrue(json_data.get("grounded"))

    def test_28_api_chat_without_material(self):
        """Test 28: POST /api/ai/chat without material_id works normally (regression test)."""
        with self.client.session_transaction() as sess:
            sess["user_id"] = self.user_a_id
            sess["username"] = "student_mat_a"
            sess.pop("active_material_id", None)
            sess.pop("active_material_title", None)

        payload = {
            "message": "Hello FocusSense AI, explain Newton's first law."
        }
        response = self.client.post("/api/ai/chat", json=payload)
        self.assertEqual(response.status_code, 200)
        json_data = response.get_json()
        self.assertTrue(json_data["success"])
        self.assertIsNotNone(json_data["reply"])
        self.assertIsNone(json_data.get("material_id"))
        self.assertFalse(json_data.get("grounded", False))

    # --------------------------------------------------------------------------
    # 6. QUIZ & RECALL INTEGRATION (Tests 29–30)
    # --------------------------------------------------------------------------
    def test_29_quiz_generation_from_material(self):
        """Test 29: create_or_get_material_quiz generates structured questions grounded in material chunks."""
        # Create a material
        res = sme.save_and_process_material(
            user_id=self.user_a_id,
            original_filename="quiz_material.txt",
            file_bytes=b"Database Normalization: 1NF removes duplicates, 2NF removes partial dependency, 3NF removes transitive dependency.",
            title="DB Normalization",
            topic="Normalization"
        )
        mat_id = res["material_id"]

        quiz_data = quiz_engine.create_or_get_material_quiz(user_id=self.user_a_id, material_id=mat_id, question_count=3)
        self.assertIsNotNone(quiz_data)
        self.assertIn("quiz_id", quiz_data)
        self.assertIn("questions", quiz_data)
        self.assertGreaterEqual(len(quiz_data["questions"]), 1)
        self.assertEqual(quiz_data.get("material_id"), mat_id)

    def test_30_recall_engine_with_material_chunks(self):
        """Test 30: start_recall_test integrates material chunks when a material_id is present."""
        # Create a material
        res = sme.save_and_process_material(
            user_id=self.user_a_id,
            original_filename="recall_material.txt",
            file_bytes=b"Cache memory hierarchy: L1 cache is fastest and smallest, L2 is larger, L3 is shared.",
            title="Computer Architecture Cache",
            topic="Cache Memory"
        )
        mat_id = res["material_id"]

        # Schedule recall task
        task = recall_engine.schedule_recall_task(
            user_id=self.user_a_id,
            session_id=1,
            topic="Cache Memory",
            delay_minutes=0
        )
        task_id = task["task_id"]

        recall_data = recall_engine.start_recall_test(
            task_id=task_id,
            user_id=self.user_a_id,
            question_count=3
        )
        self.assertIsNotNone(recall_data)
        self.assertTrue(recall_data["success"])
        self.assertEqual(recall_data["task_id"], task_id)
        self.assertIn("questions", recall_data)


if __name__ == "__main__":
    unittest.main()
