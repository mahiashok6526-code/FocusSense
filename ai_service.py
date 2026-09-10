"""
FocusSense AI — Study Companion & Topic Grounding Service
=========================================================
Modular AI Service Architecture:
- BaseAIProvider: Abstract interface for AI backends.
- GeminiAIProvider: Official Google Gemini REST API integration.
- LocalMockAIProvider: Intelligent offline fallback with grounded pedagogical reasoning.
- FocusSenseAIService: High-level orchestrator and facade.
"""

import os
import re
import json
import logging
from abc import ABC, abstractmethod

try:
    import requests
except ImportError:
    requests = None

logger = logging.getLogger("FocusSenseAI")


# ---------------------------------------------------------------------------
# Abstract Base Provider
# ---------------------------------------------------------------------------

class BaseAIProvider(ABC):
    """Abstract interface for FocusSense AI backend providers."""

    @abstractmethod
    def generate_reply(self, prompt: str, conversation_history: list,
                       study_topic: str, student_name: str, learning_context: dict = None,
                       material_context: dict = None) -> dict:
        """
        Generate grounded pedagogical response.
        """
        pass

    @abstractmethod
    def generate_quiz(self, topic: str, conversation_history: list, question_count: int = 3) -> list:
        """Generate structured questions for learning verification."""
        pass

    @abstractmethod
    def evaluate_answer(self, question_text: str, sub_concept: str,
                        sample_answer: str, student_answer: str) -> dict:
        """Semantically grade student answer against rubric."""
        pass

    def generate_assessment_questions(self, topic: str, mode: str = "PRACTICE", difficulty: str = "mixed",
                                      question_count: int = 5, conversation_history: list = None,
                                      material_excerpts: str = None) -> list:
        """Generate diverse assessment questions across multiple types and categories."""
        return []

    def explain_assessment_mistake(self, question_text: str, student_answer: str,
                                  ideal_answer: str, sub_concept: str, topic: str) -> str:
        """Provide a Socratic, constructive explanation for an incorrect or incomplete assessment answer."""
        return ""


# ---------------------------------------------------------------------------
# Local Ollama AI Provider (Private, Free Offline LLM Inference)
# ---------------------------------------------------------------------------

class OllamaAIProvider(BaseAIProvider):
    """
    Local Ollama Inference Engine.
    Connects to local Ollama daemon (default http://localhost:11434).
    Supports models like llama3.2:3b, llama3.2, llama3, mistral, gemma2, phi3, qwen2.5, deepseek-r1, etc.
    """

    DEFAULT_HOST = "http://localhost:11434"
    DEFAULT_MODEL = "llama3.2:3b"

    def __init__(self, host: str = None, model: str = None):
        self.host = (host or os.environ.get("OLLAMA_HOST") or self.DEFAULT_HOST).rstrip("/")
        self.model = model or os.environ.get("OLLAMA_MODEL") or self.DEFAULT_MODEL
        self._last_check_time = 0
        self._is_available_cache = False
        self._models_cache = []

    def is_available(self) -> bool:
        """Check if local Ollama server is running and reachable (with 3-second TTL cache)."""
        # In serverless environments (Vercel, AWS Lambda), local Ollama daemon is not present
        if os.environ.get("VERCEL") or os.environ.get("AWS_LAMBDA_FUNCTION_NAME"):
            return False

        import time
        now = time.time()
        if (now - self._last_check_time) < 3.0:
            return self._is_available_cache

        self._last_check_time = now
        try:
            res = requests.get(f"{self.host}/api/tags", timeout=0.8)
            self._is_available_cache = (res.status_code == 200)
            if self._is_available_cache:
                data = res.json()
                self._models_cache = [m.get("name") for m in data.get("models", [])]
            else:
                self._models_cache = []
        except Exception:
            self._is_available_cache = False
            self._models_cache = []

        return self._is_available_cache

    def get_installed_models(self) -> list:
        """Return list of models downloaded in local Ollama."""
        if self.is_available():
            return self._models_cache
        return []

    def get_active_model(self) -> str:
        """Select the best available model from installed models or user default."""
        models = self.get_installed_models()
        if not models:
            return self.model

        # 1. Exact match (e.g. "llama3.2:3b" or "llama3.2:3b-instruct-q4_K_M")
        for m in models:
            if m == self.model or m == f"{self.model}:latest" or m.startswith(f"{self.model}:"):
                return m

        # 2. Substring match
        for m in models:
            if self.model in m or m.startswith(self.model):
                return m

        # 3. Preferred family fallbacks (llama3.2:3b, llama3.2, llama3, mistral, gemma, phi3)
        for prefix in ["llama3.2:3b", "llama3.2", "llama3", "mistral", "gemma", "phi3"]:
            for m in models:
                if prefix in m:
                    return m

        return models[0]

    def get_info(self) -> dict:
        available = self.is_available()
        active_model = self.get_active_model() if available else self.model
        return {
            "name": "Ollama Local AI",
            "model": active_model,
            "type": "local",
            "active": available,
            "host": self.host
        }

    def generate_reply(self, prompt: str, conversation_history: list,
                       study_topic: str, student_name: str, learning_context: dict = None,
                       material_context: dict = None) -> dict:
        active_model = self.get_active_model()
        if not self.is_available():
            topic_hint = f" for **{study_topic}**" if study_topic and study_topic != "General Study" else ""
            return {
                "reply": (
                    f"⚠️ **Local AI Engine (Ollama) is Not Running**{topic_hint}\n\n"
                    f"FocusSense AI uses a local, private LLM for offline academic assistance on {study_topic or 'your studies'} without external API costs.\n\n"
                    "**To start FocusSense AI with the local model:**\n"
                    "1. Install Ollama from [ollama.com](https://ollama.com) if you haven't already.\n"
                    "2. Open your terminal and run:\n"
                    "   ```bash\n"
                    f"   ollama run {active_model or 'llama3.2:3b'}\n"
                    "   ```\n"
                    "3. Return to this chat and ask your question!"
                ),
                "provider": "Ollama (Offline)",
                "model": active_model,
                "success": False,
                "error": f"Ollama daemon not reachable at {self.host}",
            }

        topic_str = study_topic.strip() if study_topic and study_topic != "General Study" else ""
        student_str = student_name.strip() if student_name else "Student"

        # Adaptive pedagogical guidance based on student mastery
        personalization_clause = ""
        if learning_context:
            tier = (learning_context.get("mastery_tier") or "").upper()
            ret_score = learning_context.get("avg_retention_score") or learning_context.get("last_retention_score")
            score_str = f"{ret_score}%" if ret_score is not None else ""
            if tier == "WEAK" or (ret_score is not None and float(ret_score) < 60.0):
                personalization_clause = (
                    f"\n\nADAPTIVE PERSONALIZATION (Developing Mastery {score_str}): "
                    f"The student previously struggled with this topic. Use simpler explanations, relatable intuitive analogies, "
                    f"and concrete step-by-step examples before introducing formal technical terms. Ask gentle check questions."
                )
            elif tier == "STRONG" or (ret_score is not None and float(ret_score) >= 85.0):
                personalization_clause = (
                    f"\n\nADAPTIVE PERSONALIZATION (High Mastery {score_str}): "
                    f"The student has demonstrated strong mastery in this topic. Provide deeper technical reasoning, "
                    f"practical edge cases, architectural trade-offs, optimization techniques, and challenging conceptual connections."
                )

        # Material Grounding Clause
        material_clause = ""
        if material_context and material_context.get("found"):
            mat_title = material_context.get("material_title") or "Study Material"
            mat_subject = material_context.get("subject") or "Academic Study"
            mat_topic = material_context.get("topic") or mat_title
            is_rel = material_context.get("is_relevant", True)
            excerpts = material_context.get("context_text", "")

            if is_rel and excerpts:
                material_clause = (
                    f"\n\n=======================================================\n"
                    f"STUDY MATERIAL MODE: ACTIVATED\n"
                    f"Active Material: '{mat_title}' (Subject: {mat_subject}, Topic: {mat_topic})\n"
                    f"=======================================================\n"
                    f"RELEVANT EXCERPTS FROM THE STUDENT'S UPLOADED STUDY MATERIAL:\n"
                    f"{excerpts}\n\n"
                    f"STRICT GROUNDING INSTRUCTIONS:\n"
                    f"1. Base your answer primarily on the uploaded study material excerpts provided above.\n"
                    f"2. Cite key terminology, definitions, and concepts directly from the material.\n"
                    f"3. If the answer is not present in the excerpts, clearly state:\n"
                    f"   '⚠️ Note: I couldn't find this specific detail in your uploaded study material.'\n"
                    f"   and then provide concise general educational guidance.\n"
                )
            else:
                material_clause = (
                    f"\n\n=======================================================\n"
                    f"STUDY MATERIAL MODE: '{mat_title}'\n"
                    f"=======================================================\n"
                    f"The student asked a question, but NO relevant excerpts were found in the uploaded material '{mat_title}'.\n"
                    f"HONEST GROUNDING INSTRUCTION:\n"
                    f"Start your response by explicitly informing the student:\n"
                    f"'I couldn't find this clearly in the selected study material ({mat_title}).'\n"
                    f"Then provide a helpful general academic explanation from your broader knowledge base."
                )

        context_guidance = (
            f"The student's active study session topic is: '{topic_str}'. Connect concepts to this topic when relevant, "
            f"but ALWAYS thoroughly answer any general academic, programming, mathematics, scientific, or conceptual questions the student asks."
            if topic_str else
            "Answer all academic, technical, programming, scientific, mathematical, and conceptual inquiries thoroughly."
        )

        system_instruction = (
            f"You are FocusSense AI, an intelligent, empathetic, rigorous, and supportive Socratic educational AI study companion.\n"
            f"You are working with {student_str}.\n"
            f"{context_guidance}"
            f"{personalization_clause}"
            f"{material_clause}\n\n"
            f"CORE PEDAGOGICAL CAPABILITIES & GUIDELINES:\n"
            f"1. Conversational Memory: Maintain continuity with previous messages in this conversation. When the student asks follow-up questions (e.g. 'Can you explain that simpler?', 'Show me an example', 'How does this compare to X?', 'What about overriding?', 'Debug this code'), interpret their question in the direct context of earlier turns.\n"
            f"2. Concept Explanations & Socratic Progression: Break down complex topics using first principles, intuitive analogies, and concise step-by-step reasoning.\n"
            f"3. Programming & Technical Guidance: Provide clean, well-commented code snippets and explain complexity.\n"
            f"4. Conciseness & Readability: Keep answers focused, structured, and punchy (around 2-3 paragraphs or concise bullet points). Do not be overly verbose.\n"
            f"5. Socratic Checks: End your explanations with a brief, thoughtful check question to encourage active learning.\n"
            f"6. Formatting: Use clean GitHub Markdown: bold key terminology, numbered lists for sequence, bullet points for lists, and syntax-highlighted code blocks.\n"
            f"7. Tone: Academic, encouraging, clear, and focused."
        )

        messages = [{"role": "system", "content": system_instruction}]
        for msg in conversation_history[-10:]:
            role = "user" if msg.get("sender") == "user" else "assistant"
            content = (msg.get("message") or "").strip()
            if content:
                messages.append({"role": role, "content": content})

        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": active_model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": 0.6,
                "top_p": 0.9,
                "num_predict": 280,
            }
        }

        try:
            res = requests.post(f"{self.host}/api/chat", json=payload, timeout=120)
            if res.status_code == 200:
                data = res.json()
                reply_text = data.get("message", {}).get("content", "").strip()
                if reply_text:
                    return {
                        "reply": reply_text,
                        "provider": "Ollama Local AI",
                        "model": active_model,
                        "success": True,
                        "error": None,
                    }
        except Exception as ex:
            logger.warning(f"Ollama chat error: {ex}")

        return {
            "reply": "⚠️ FocusSense AI encountered an error communicating with the local Ollama server. Please ensure Ollama is running (`ollama run llama3.2:3b`).",
            "provider": "Ollama Local AI",
            "model": active_model,
            "success": False,
            "error": "Inference failure with local Ollama daemon",
        }

    def generate_quiz(self, topic: str, conversation_history: list = None, question_count: int = 3, adaptive_guidance: str = None, material_excerpts: str = None) -> list:
        if not self.is_available():
            return []

        active_model = self.get_active_model()
        history = conversation_history or []
        chat_context = "\n".join([f"{m.get('sender')}: {m.get('message')}" for m in history[-4:]]) if history else ""
        
        guidance_clause = f"Adaptive Guidance: {adaptive_guidance}\n" if adaptive_guidance else ""
        material_clause = f"\nGROUNDED STUDY MATERIAL EXCERPTS:\n{material_excerpts}\nGenerate questions strictly based on the facts and mechanisms in these excerpts.\n" if material_excerpts else ""
        prompt = (
            f"You are an expert academic examiner. Generate exactly {question_count} conceptual verification questions for a student who studied: '{topic}'.\n"
            f"Session Context:\n{chat_context}\n"
            f"{guidance_clause}"
            f"{material_clause}\n"
            f"STRUCTURE BY 3 DIFFICULTY TIERS:\n"
            f"- Question 1 (Basic): Core definition, terminology, or fundamental mechanism.\n"
            f"- Question 2 (Conceptual / Application): Concrete scenario-based problem or practical application.\n"
            f"- Question 3 (Harder / Edge Case / Debugging): Nuanced trade-off, common pitfall, or advanced edge-case.\n\n"
            f"Format response as a JSON array of objects with keys:\n"
            f"- question_index (integer, 1-indexed)\n"
            f"- question_text (string, clear question)\n"
            f"- sub_concept (string, concise sub-concept name e.g. 'Function Arguments', 'Inner Joins')\n"
            f"- difficulty (string: 'basic', 'application', or 'advanced')\n"
            f"- sample_answer (string, brief 1-2 sentence rubric explanation)\n"
            f"- points_possible (float, 1.0)\n\n"
            f"Return ONLY the valid JSON array."
        )

        payload = {
            "model": active_model,
            "prompt": prompt,
            "format": "json",
            "stream": False,
            "options": {
                "temperature": 0.3,
                "num_predict": 320
            }
        }

        try:
            res = requests.post(f"{self.host}/api/generate", json=payload, timeout=90)
            if res.status_code == 200:
                raw_json = res.json().get("response", "").strip()
                clean_json = re.sub(r"^```json\s*|\s*```$", "", raw_json, flags=re.MULTILINE).strip()
                parsed = json.loads(clean_json)
                if isinstance(parsed, list) and len(parsed) > 0:
                    return parsed[:question_count]
                elif isinstance(parsed, dict) and "questions" in parsed:
                    return parsed["questions"][:question_count]
        except Exception as ex:
            logger.warning(f"Ollama dynamic quiz generation failed: {ex}")

        return []

    def evaluate_answer(self, question_text: str, sub_concept: str,
                        sample_answer: str, student_answer: str) -> dict:
        if not self.is_available():
            return {}

        active_model = self.get_active_model()
        prompt = (
            f"You are an expert academic evaluator. Grade the student's answer.\n"
            f"Sub-Concept: {sub_concept}\n"
            f"Question: {question_text}\n"
            f"Sample Ideal Answer: {sample_answer}\n"
            f"Student Answer: {student_answer}\n\n"
            f"Evaluate conceptually. Return JSON with:\n"
            f"- status: 'correct', 'partial', or 'incorrect'\n"
            f"- score: float (1.0 for correct, 0.5 for partial, 0.0 for incorrect)\n"
            f"- feedback: 1 sentence of encouraging feedback\n\n"
            f"Return ONLY valid JSON."
        )

        payload = {
            "model": active_model,
            "prompt": prompt,
            "format": "json",
            "stream": False,
            "options": {
                "temperature": 0.2,
                "num_predict": 120
            }
        }

        try:
            res = requests.post(f"{self.host}/api/generate", json=payload, timeout=60)
            if res.status_code == 200:
                raw_json = res.json().get("response", "").strip()
                clean_json = re.sub(r"^```json\s*|\s*```$", "", raw_json, flags=re.MULTILINE).strip()
                evaluation = json.loads(clean_json)
                if "status" in evaluation and "score" in evaluation:
                    evaluation["success"] = True
                    return evaluation
        except Exception as ex:
            logger.warning(f"Ollama answer evaluation failed: {ex}")

        return {}

    def generate_assessment_questions(self, topic: str, mode: str = "PRACTICE", difficulty: str = "mixed",
                                      question_count: int = 5, conversation_history: list = None,
                                      material_excerpts: str = None) -> list:
        if not self.is_available():
            return []

        active_model = self.get_active_model()
        history = conversation_history or []
        chat_context = "\n".join([f"{m.get('sender')}: {m.get('message')}" for m in history[-4:]]) if history else ""
        mat_clause = f"\nGROUNDED STUDY MATERIAL EXCERPTS:\n{material_excerpts}\nGenerate questions strictly based on the facts, mechanisms, and rules in these excerpts.\n" if material_excerpts else ""

        mode_instructions = {
            "QUICK": "Generate a concise 5-question mix of fundamental definitions, True/False, and short conceptual questions.",
            "PRACTICE": "Generate a balanced 10-question mix: 4 MCQs, 3 Conceptual Short Answer, 2 Practical Application, 1 Problem Solving.",
            "TOPIC": f"Generate questions strictly restricted to the topic '{topic}' across adaptive difficulty tiers.",
            "EXAM": "Generate rigorous, multi-type exam questions with diverse difficulty (Recall, Conceptual, Application, Problem Solving) without hinting at answers."
        }.get(mode.upper(), "Generate diverse questions across mixed difficulty.")

        prompt = (
            f"You are a master academic assessment designer. Generate exactly {question_count} high-quality questions on: '{topic}'.\n"
            f"Assessment Mode: {mode}\n"
            f"Mode Directive: {mode_instructions}\n"
            f"Difficulty Policy: {difficulty}\n"
            f"Session Context:\n{chat_context}\n"
            f"{mat_clause}\n"
            f"QUESTION TYPES TO INCLUDE (Mix where appropriate):\n"
            f"- 'mcq': Multiple Choice (4 distinct options 'A) ...', 'B) ...', 'C) ...', 'D) ...')\n"
            f"- 'true_false': True / False statement\n"
            f"- 'short_answer': Concise free-form answer\n"
            f"- 'conceptual': Explains mechanisms, definitions, or architectural trade-offs\n"
            f"- 'application': Scenario-based problem or practical application\n"
            f"- 'problem_solving': Concrete problem, calculation, code analysis, or debugging\n\n"
            f"CATEGORIES (Assign one per question):\n"
            f"- 'recall', 'concept', 'application', 'problem_solving'\n\n"
            f"Format response as a JSON array of objects with keys:\n"
            f"- question_index (integer, 1-indexed)\n"
            f"- question_text (string)\n"
            f"- sub_concept (string, e.g. 'Process States', 'Index B-Trees')\n"
            f"- question_type (string: 'mcq' | 'true_false' | 'short_answer' | 'conceptual' | 'application' | 'problem_solving')\n"
            f"- category (string: 'recall' | 'concept' | 'application' | 'problem_solving')\n"
            f"- difficulty (string: 'basic' | 'medium' | 'advanced')\n"
            f"- options (list of strings for mcq or true_false, e.g. ['A) ...', 'B) ...', 'C) ...', 'D) ...'] or ['A) True', 'B) False']; null for free text)\n"
            f"- correct_option (string e.g. 'A' or 'True'; null for free text)\n"
            f"- sample_answer (string, complete rubric explanation)\n"
            f"- explanation (string, brief 1-2 sentence rationale for the correct answer)\n"
            f"- points_possible (float, 1.0)\n\n"
            f"Return ONLY the valid JSON array."
        )

        payload = {
            "model": active_model,
            "prompt": prompt,
            "format": "json",
            "stream": False,
            "options": {
                "temperature": 0.3,
                "num_predict": 750
            }
        }

        try:
            res = requests.post(f"{self.host}/api/generate", json=payload, timeout=25)
            if res.status_code == 200:
                raw_json = res.json().get("response", "").strip()
                clean_json = re.sub(r"^```json\s*|\s*```$", "", raw_json, flags=re.MULTILINE).strip()
                parsed = json.loads(clean_json)
                if isinstance(parsed, list) and len(parsed) >= question_count:
                    return parsed[:question_count]
                elif isinstance(parsed, dict) and "questions" in parsed and len(parsed["questions"]) >= question_count:
                    return parsed["questions"][:question_count]
        except Exception as ex:
            logger.warning(f"Ollama assessment generation failed: {ex}")

        return []

    def explain_assessment_mistake(self, question_text: str, student_answer: str,
                                  ideal_answer: str, sub_concept: str, topic: str) -> str:
        if not self.is_available():
            return ""

        active_model = self.get_active_model()
        prompt = (
            f"You are FocusSense AI, an empathetic and supportive Socratic academic tutor.\n"
            f"The student answered an assessment question incorrectly on Topic: '{topic}' (Sub-Concept: '{sub_concept}').\n\n"
            f"Question: {question_text}\n"
            f"Ideal Answer / Concept: {ideal_answer}\n"
            f"Student's Submitted Answer: {student_answer}\n\n"
            f"INSTRUCTIONS:\n"
            f"1. Empathize constructively with what the student attempted.\n"
            f"2. Explain clearly where the reasoning diverged or what key mechanism was missed.\n"
            f"3. Provide a simple, memorable 1-2 sentence conceptual takeaway or intuitive rule of thumb.\n"
            f"4. Keep the total response to 2 concise, focused paragraphs in clean Markdown.\n"
            f"Do not output internal chain-of-thought."
        )

        payload = {
            "model": active_model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.4,
                "num_predict": 250
            }
        }

        try:
            res = requests.post(f"{self.host}/api/generate", json=payload, timeout=60)
            if res.status_code == 200:
                return res.json().get("response", "").strip()
        except Exception as ex:
            logger.warning(f"Ollama mistake explanation error: {ex}")

        return ""


# ---------------------------------------------------------------------------
# Google Gemini Provider
# ---------------------------------------------------------------------------

class GeminiAIProvider(BaseAIProvider):
    """
    Google Gemini Cloud Provider.
    Connects securely using GEMINI_API_KEY or GOOGLE_API_KEY from environment variables.
    """

    DEFAULT_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")
    MODELS = ["gemini-2.5-flash", "gemini-1.5-flash", "gemini-1.5-pro"]

    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        self.endpoint_base = "https://generativelanguage.googleapis.com/v1beta/models"

    def is_configured(self) -> bool:
        return bool(self.api_key and len(self.api_key.strip()) > 10)

    def get_info(self) -> dict:
        return {
            "name": "Gemini AI",
            "model": self.DEFAULT_MODEL,
            "type": "cloud",
            "active": self.is_configured(),
        }

    def _call_gemini_api(self, payload: dict, timeout: float = 5.0) -> tuple:
        """
        Execute API call across candidate models (DEFAULT_MODEL + fallbacks).
        Includes application-level responsiveness safeguards to preserve fast user experience.
        Returns (response_json_dict, successful_model_name) or (None, None).
        """
        if not self.is_configured():
            return None, None

        candidates = [self.DEFAULT_MODEL] + [m for m in self.MODELS if m != self.DEFAULT_MODEL]
        seen = set()
        models_to_try = [m for m in candidates if not (m in seen or seen.add(m))]

        for model in models_to_try:
            url = f"{self.endpoint_base}/{model}:generateContent?key={self.api_key}"
            try:
                res = requests.post(url, json=payload, headers={"Content-Type": "application/json"}, timeout=timeout)
                if res.status_code == 200:
                    return res.json(), model
                elif res.status_code in (404, 429):
                    logger.warning(f"Gemini model {model} returned HTTP {res.status_code}, trying next model...")
                    continue
                elif res.status_code in (400, 403):
                    # Key, auth, or model deprecation: log and do not stall the user experience
                    logger.warning(f"Gemini API auth/permission issue on model {model} (HTTP {res.status_code}): {res.text[:200]}")
                    break
                else:
                    logger.warning(f"Gemini API error on model {model} (HTTP {res.status_code}): {res.text[:200]}")
                    break
            except Exception as ex:
                logger.warning(f"Gemini request exception for model {model}: {ex}")
                # Responsiveness safeguard: avoid cumulative multi-model timeouts
                break

        return None, None

    def generate_reply(self, prompt: str, conversation_history: list,
                       study_topic: str, student_name: str, learning_context: dict = None,
                       material_context: dict = None) -> dict:
        if not self.is_configured():
            return {
                "reply": "",
                "provider": "Gemini AI",
                "model": self.DEFAULT_MODEL,
                "success": False,
                "error": "GEMINI_API_KEY is not configured.",
            }

        topic_str = study_topic.strip() if study_topic else "General Academic Studies"
        student_str = student_name.strip() if student_name else "Student"

        # Material Grounding Clause
        material_clause = ""
        if material_context and material_context.get("found"):
            mat_title = material_context.get("material_title") or "Study Material"
            mat_subject = material_context.get("subject") or "Academic Study"
            mat_topic = material_context.get("topic") or mat_title
            is_rel = material_context.get("is_relevant", True)
            excerpts = material_context.get("context_text", "")

            if is_rel and excerpts:
                material_clause = (
                    f"\n\nSTUDY MATERIAL MODE: ACTIVATED\n"
                    f"Active Material: '{mat_title}' (Subject: {mat_subject}, Topic: {mat_topic})\n"
                    f"EXCERPTS FROM UPLOADED STUDY MATERIAL:\n{excerpts}\n\n"
                    f"GROUNDING INSTRUCTIONS:\n"
                    f"1. Base your answer primarily on the uploaded study material excerpts provided above.\n"
                    f"2. Cite key terminology and concepts from the material.\n"
                    f"3. If the answer is not present in the excerpts, clearly state:\n"
                    f"   '⚠️ Note: I couldn't find this specific detail in your uploaded study material.'\n"
                    f"   and then provide concise general educational guidance.\n"
                )
            else:
                material_clause = (
                    f"\n\nSTUDY MATERIAL MODE: '{mat_title}'\n"
                    f"The student asked a question, but no relevant excerpts were found in '{mat_title}'.\n"
                    f"HONEST GROUNDING: Start your response with:\n"
                    f"'I couldn't find this clearly in the selected study material ({mat_title}).'\n"
                    f"Then provide a helpful general academic explanation."
                )

        system_instruction = (
            f"You are FocusSense AI, an empathetic, intellectually rigorous, and encouraging personal Socratic study companion.\n"
            f"You are currently working directly with {student_str}, who is studying the topic: '{topic_str}'.\n\n"
            f"GUIDELINES:\n"
            f"1. Conversational Memory: Always maintain context from previous turns in this conversation. When the student asks follow-up questions, understand them in context.\n"
            f"2. Ground all explanations, examples, and answers in '{topic_str}' whenever relevant.\n"
            f"3. Use clear formatting with concise bullet points, bold key terms, and intuitive real-world analogies.\n"
            f"4. When asked to explain simply (ELI5), strip out unnecessary jargon and build intuition from first principles.\n"
            f"5. If code or technical structures are needed, provide clear, well-commented snippets.\n"
            f"6. End with a quick reflective check question to verify comprehension.\n"
            f"7. Maintain an encouraging, focused, academic tone."
            f"{material_clause}"
        )

        # Build multi-turn contents payload
        contents = []
        for msg in conversation_history[-10:]:  # keep last 10 turns for context window efficiency
            role = "user" if msg.get("sender") == "user" else "model"
            text_content = msg.get("message", "").strip()
            if text_content:
                contents.append({
                    "role": role,
                    "parts": [{"text": text_content}]
                })

        contents.append({
            "role": "user",
            "parts": [{"text": prompt}]
        })

        payload = {
            "contents": contents,
            "systemInstruction": {
                "parts": [{"text": system_instruction}]
            },
            "generationConfig": {
                "temperature": 0.7,
                "maxOutputTokens": 1000,
                "topP": 0.95,
            }
        }

        # Application-level responsiveness safeguard: 5.0s timeout to maintain fast UX
        data, model = self._call_gemini_api(payload, timeout=5.0)
        if data:
            candidates = data.get("candidates", [])
            if candidates:
                parts = candidates[0].get("content", {}).get("parts", [])
                if parts:
                    reply_text = parts[0].get("text", "").strip()
                    if reply_text:
                        return {
                            "reply": reply_text,
                            "provider": "Gemini AI",
                            "model": model,
                            "success": True,
                            "error": None,
                        }

        return {
            "reply": "",
            "provider": "Gemini AI",
            "model": self.DEFAULT_MODEL,
            "success": False,
            "error": "Gemini API request failed.",
        }

    def generate_quiz(self, topic: str, conversation_history: list = None, question_count: int = 3, adaptive_guidance: str = None, material_excerpts: str = None) -> list:
        if not self.is_configured():
            return []

        history = conversation_history or []
        chat_context = "\n".join([f"{m.get('sender')}: {m.get('message')}" for m in history[-6:]]) if history else ""
        mat_clause = f"\nStudy Material Excerpts:\n{material_excerpts}\n" if material_excerpts else ""
        prompt = (
            f"Generate exactly {question_count} conceptual learning verification questions for a student who studied '{topic}'.\n"
            f"Session Context:\n{chat_context}\n"
            f"{mat_clause}\n"
            f"Format as valid JSON array of objects with keys:\n"
            f"- question_index (integer, 1-indexed)\n"
            f"- question_text (string)\n"
            f"- sub_concept (string, e.g. 'FCFS Scheduling', 'Time Quantum')\n"
            f"- sample_answer (string, complete rubric explanation)\n"
            f"- points_possible (float, 1.0)\n\n"
            f"Return ONLY valid JSON."
        )

        payload = {
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": 0.3, "maxOutputTokens": 1000}
        }

        # Application-level responsiveness safeguard: 5.0s timeout
        data, model = self._call_gemini_api(payload, timeout=5.0)
        if data:
            try:
                candidates = data.get("candidates", [])
                if candidates:
                    parts = candidates[0].get("content", {}).get("parts", [])
                    if parts:
                        raw = parts[0].get("text", "").strip()
                        clean_json = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw, flags=re.MULTILINE).strip()
                        questions = json.loads(clean_json)
                        if isinstance(questions, list) and len(questions) > 0:
                            return questions
            except Exception as ex:
                logger.warning(f"Gemini quiz generation parse failed: {ex}")
        return []

    def evaluate_answer(self, question_text: str, sub_concept: str,
                        sample_answer: str, student_answer: str) -> dict:
        if not self.is_configured():
            return {}

        prompt = (
            f"You are an expert academic evaluator. Grade the student's answer.\n"
            f"Topic/Concept: {sub_concept}\n"
            f"Question: {question_text}\n"
            f"Sample Ideal Answer / Rubric: {sample_answer}\n"
            f"Student Answer: {student_answer}\n\n"
            f"Evaluate conceptually (do NOT require exact phrasing). Return JSON with keys:\n"
            f"- status: 'correct' (accurate, complete), 'partial' (partially accurate/missing key aspect), or 'incorrect' (wrong, irrelevant, or major misconception)\n"
            f"- score: float (1.0 for correct, 0.5 for partial, 0.0 for incorrect)\n"
            f"- feedback: 1-2 sentences of encouraging, precise feedback explaining what was right or missing\n\n"
            f"Return ONLY valid JSON."
        )

        payload = {
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": 0.2, "maxOutputTokens": 400}
        }

        # Application-level responsiveness safeguard: 4.5s evaluation timeout
        data, model = self._call_gemini_api(payload, timeout=4.5)
        if data:
            try:
                candidates = data.get("candidates", [])
                if candidates:
                    parts = candidates[0].get("content", {}).get("parts", [])
                    if parts:
                        raw = parts[0].get("text", "").strip()
                        clean_json = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw, flags=re.MULTILINE).strip()
                        evaluation = json.loads(clean_json)
                        if "status" in evaluation and "score" in evaluation:
                            evaluation["success"] = True
                            return evaluation
            except Exception as ex:
                logger.warning(f"Gemini answer evaluation parse failed: {ex}")
        return {}

    def generate_assessment_questions(self, topic: str, mode: str = "PRACTICE", difficulty: str = "mixed",
                                      question_count: int = 5, conversation_history: list = None,
                                      material_excerpts: str = None) -> list:
        if not self.is_configured():
            return []

        history = conversation_history or []
        chat_context = "\n".join([f"{m.get('sender')}: {m.get('message')}" for m in history[-4:]]) if history else ""
        mat_clause = f"\nStudy Material Excerpts:\n{material_excerpts}\n" if material_excerpts else ""

        prompt = (
            f"Generate exactly {question_count} high-quality assessment questions on: '{topic}'.\n"
            f"Mode: {mode} | Difficulty: {difficulty}\n"
            f"{chat_context}\n{mat_clause}\n"
            f"Format as valid JSON array of objects with keys:\n"
            f"- question_index (integer, 1-indexed)\n"
            f"- question_text (string)\n"
            f"- sub_concept (string)\n"
            f"- question_type (string: 'mcq' | 'true_false' | 'short_answer' | 'conceptual' | 'application' | 'problem_solving')\n"
            f"- category (string: 'recall' | 'concept' | 'application' | 'problem_solving')\n"
            f"- difficulty (string: 'basic' | 'medium' | 'advanced')\n"
            f"- options (list of strings for mcq or true_false, e.g. ['A) ...', 'B) ...', 'C) ...', 'D) ...']; null for free text)\n"
            f"- correct_option (string e.g. 'A' or 'True'; null for free text)\n"
            f"- sample_answer (string, rubric explanation)\n"
            f"- explanation (string, brief rationale)\n"
            f"- points_possible (float, 1.0)\n\n"
            f"Return ONLY valid JSON."
        )

        payload = {
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": 0.3, "maxOutputTokens": 1500}
        }

        # Application-level responsiveness safeguard: 6.0s timeout
        data, model = self._call_gemini_api(payload, timeout=6.0)
        if data:
            try:
                candidates = data.get("candidates", [])
                if candidates:
                    parts = candidates[0].get("content", {}).get("parts", [])
                    if parts:
                        raw = parts[0].get("text", "").strip()
                        clean_json = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw, flags=re.MULTILINE).strip()
                        parsed = json.loads(clean_json)
                        if isinstance(parsed, list):
                            return parsed[:question_count]
                        elif isinstance(parsed, dict) and "questions" in parsed:
                            return parsed["questions"][:question_count]
            except Exception as ex:
                logger.warning(f"Gemini assessment generation parse failed: {ex}")

        return []

    def explain_assessment_mistake(self, question_text: str, student_answer: str,
                                  ideal_answer: str, sub_concept: str, topic: str) -> str:
        if not self.is_configured():
            return ""

        prompt = (
            f"You are FocusSense AI, an empathetic academic tutor.\n"
            f"The student answered an assessment question incorrectly on Topic: '{topic}' (Concept: '{sub_concept}').\n\n"
            f"Question: {question_text}\n"
            f"Ideal Answer: {ideal_answer}\n"
            f"Student Answer: {student_answer}\n\n"
            f"Provide a constructive 2-paragraph Socratic explanation in clean Markdown clarifying what was missed and giving a memorable rule of thumb."
        )

        payload = {
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": 0.4, "maxOutputTokens": 400}
        }

        # Application-level responsiveness safeguard: 4.5s timeout
        data, model = self._call_gemini_api(payload, timeout=4.5)
        if data:
            try:
                candidates = data.get("candidates", [])
                if candidates:
                    parts = candidates[0].get("content", {}).get("parts", [])
                    if parts:
                        return parts[0].get("text", "").strip()
            except Exception as ex:
                logger.warning(f"Gemini mistake explanation parse failed: {ex}")

        return ""


# ---------------------------------------------------------------------------
# Local Offline Dynamic Heuristic Fallback Provider
# ---------------------------------------------------------------------------

class LocalMockAIProvider(BaseAIProvider):
    """
    Intelligent offline pedagogical assistant and dynamic topic question generator.
    Provides structured guidance when neither local Ollama nor cloud API is available.
    """

    def get_info(self) -> dict:
        return {
            "name": "FocusSense Local AI",
            "model": "offline-guidance-v2",
            "type": "offline",
            "active": True,
        }

    def generate_reply(self, prompt: str, conversation_history: list,
                       study_topic: str, student_name: str, learning_context: dict = None,
                       material_context: dict = None) -> dict:
        topic = (study_topic or "Your Topic").strip()
        student = (student_name or "Student").strip()
        p_lower = prompt.lower().strip()

        # Check for multi-turn pronouns / follow-ups
        previous_context = ""
        if conversation_history:
            last_msg = conversation_history[-1].get("message", "")
            if last_msg:
                previous_context = f" *(Continuing from previous discussion on: {last_msg[:40]}...)*\n\n"

        # Check for Material Grounding Mode in Local Fallback
        if material_context and material_context.get("found"):
            mat_title = material_context.get("material_title") or topic
            mat_topic = material_context.get("topic") or topic
            mat_subject = material_context.get("subject") or "Academic Subject"
            is_rel = material_context.get("is_relevant", True)
            chunks = material_context.get("chunks", [])
            sample_content = " ".join([c.get("chunk_text", "") for c in chunks[:3]])[:400]

            if not is_rel:
                disclaimer = f"I couldn't find this clearly in the selected study material (**{mat_title}**).\n\n"
                general_reply = (
                    f"### 🧠 FocusSense AI Guidance\n\n"
                    f"{disclaimer}Regarding **'{prompt}'**:\n\n"
                    f"- **Core Concept**: This relates to foundational mechanisms and systemic coordination in academic study.\n"
                    f"- **Mechanism**: It operates by tracking constraints, evaluating incoming inputs against deterministic rules, and executing the next appropriate operation.\n"
                    f"- **Why it matters**: Understanding this foundational principle prevents performance degradation and ensures scalable architecture.\n\n"
                    f"Would you like to switch to General Chat mode or explore another section of **{mat_title}**?"
                )
                return {
                    "reply": general_reply,
                    "provider": "FocusSense Local AI",
                    "model": "offline-guidance-v2",
                    "success": True,
                    "error": None,
                }

            # Grounded Material Quick Actions
            if any(w in p_lower for w in ["summarize", "summary", "overview"]):
                reply = (
                    f"### 📝 Study Material Summary: **{mat_title}**\n\n"
                    f"**Subject**: {mat_subject} | **Topic**: {mat_topic}\n\n"
                    f"#### 🎯 Key Takeaways & Core Themes:\n"
                    f"1. **Primary Focus**: The uploaded material focuses on foundational definitions, structural rules, and system behavior for **{mat_topic}**.\n"
                    f"2. **Core Mechanisms**: Emphasizes step-by-step state transitions, resource coordination, and systematic problem solving.\n"
                    f"3. **Practical Scope**: Outlines design trade-offs, constraints, and standard implementation practices.\n\n"
                    f"💡 *Excerpt Highlight*: *\"{sample_content[:180]}...\"*\n\n"
                    f"Would you like to deep-dive into a specific sub-topic or generate practice questions from this material?"
                )
            elif any(w in p_lower for w in ["important concept", "important topic", "key concept", "key topic", "what topics"]):
                reply = (
                    f"### 🎯 Important Concepts in **{mat_title}**\n\n"
                    f"Based on your uploaded material for **{mat_topic}**, here are the most high-yield areas to review:\n\n"
                    f"- **1. Foundational Architecture & Definitions**: Core terminology and operational prerequisites.\n"
                    f"- **2. Execution Lifecycle & Data Flow**: How inputs are transformed and processed step-by-step.\n"
                    f"- **3. Boundary Conditions & Edge Cases**: Handling constraints, bottlenecks, and error states.\n"
                    f"- **4. Practical Synthesis**: Real-world application and verification methods.\n\n"
                    f"Which of these concepts would you like me to test you on or explain in detail?"
                )
            elif any(w in p_lower for w in ["generate 10 questions", "generate questions", "generate mcqs", "practice questions"]):
                reply = (
                    f"### 📋 Practice Questions for **{mat_title}**\n\n"
                    f"Here are 10 conceptual and application questions based on **{mat_title}**:\n\n"
                    f"1. **Core Mechanism**: What is the primary objective of {mat_topic} as described in the material?\n"
                    f"2. **Terminology**: Define the foundational data structures and parameters used in {mat_topic}.\n"
                    f"3. **Comparison**: How does the primary approach in {mat_title} compare to alternative methodologies?\n"
                    f"4. **Lifecycle**: Outline the sequential steps involved in initializing and executing this process.\n"
                    f"5. **Constraints**: What are the main resource bottlenecks and performance trade-offs?\n"
                    f"6. **Edge Case**: How does the system respond when invalid or boundary inputs are provided?\n"
                    f"7. **Optimization**: What strategy is recommended to reduce overhead and improve throughput?\n"
                    f"8. **Application**: In a practical production scenario, when would you select this specific mechanism?\n"
                    f"9. **Debugging**: What are common failure modes and diagnostic steps mentioned in the material?\n"
                    f"10. **Synthesis**: Explain the relationship between the theoretical model and practical implementation.\n\n"
                    f"Type your answer to any question above or click **Test Me** to launch an interactive graded quiz!"
                )
            elif any(w in p_lower for w in ["beginner", "eli5", "simple", "simpler", "easy"]):
                reply = (
                    f"### 🐣 Beginner Breakdown: **{mat_title}**\n\n"
                    f"Let's break down **{mat_topic}** using a simple intuition from your material:\n\n"
                    f"- **The Big Picture**: Think of **{mat_topic}** as an organized set of instructions designed to keep operations predictable and error-free.\n"
                    f"- **Step 1 (Input)**: The system receives incoming requests or data.\n"
                    f"- **Step 2 (Processing)**: It checks rules, verifies permissions, and executes tasks in order.\n"
                    f"- **Step 3 (Output)**: It produces a deterministic, reliable result.\n\n"
                    f"Does this high-level picture make sense, or would you like to explore a concrete example?"
                )
            else:
                reply = (
                    f"### 📚 Grounded Study Analysis: **{mat_title}**\n\n"
                    f"{previous_context}Regarding **'{prompt}'** in your uploaded material for **{mat_topic}**:\n\n"
                    f"- **Relevant Insight**: The material describes how {mat_topic} establishes core constraints, processes data flows, and structures state transitions.\n"
                    f"- **Key Excerpt**: *\"{sample_content[:200]}...\"*\n"
                    f"- **Practical Application**: Grounding your understanding in these core mechanisms ensures proper implementation and avoids common architectural pitfalls.\n\n"
                    f"Would you like a deeper breakdown, an illustrative code/flow example, or practice quiz questions on this?"
                )

            return {
                "reply": reply,
                "provider": "FocusSense Local AI",
                "model": "offline-guidance-v2",
                "success": True,
                "error": None,
            }

        # Standard General Chat without Material
        if any(w in p_lower for w in ["what should i understand first", "where do i start", "start", "overview", "key concepts", "roadmap", "intro"]):
            reply = (
                f"### 🎯 Core Roadmap for **{topic}**\n\n"
                f"Welcome, **{student}**!{previous_context}When tackling **{topic}**, here is the optimal first-principles progression:\n\n"
                f"1. **Core Problem Definition**: Why was this concept or mechanism created? What challenge does it solve?\n"
                f"2. **Fundamental Working Mechanism**: The step-by-step lifecycle and foundational rules.\n"
                f"3. **Trade-offs & Edge Cases**: What are the performance advantages, bottlenecks, and failure modes?\n"
                f"4. **Practical Implementation**: Real-world application in modern systems.\n\n"
                f"💡 *Action item*: What specific sub-concept in **{topic}** would you like to break down first?"
            )
        elif any(w in p_lower for w in ["explain simply", "eli5", "simple", "simpler", "easy terms", "basics"]):
            reply = (
                f"### 💡 Simple Breakdown: **{topic}**\n\n"
                f"{previous_context}Imagine **{topic}** using a simple real-world analogy:\n\n"
                f"- **The Goal**: Accomplishing tasks reliably without causing bottlenecks or chaos.\n"
                f"- **The Mechanism**: Instead of doing everything at once, the system manages incoming requests step-by-step using clear priorities and structured queues.\n"
                f"- **The Key Takeaway**: Efficiency is about minimizing wait time while ensuring fairness and stability.\n\n"
                f"Does this high-level intuition make sense, or would you like me to walk through a specific practical example?"
            )
        elif any(w in p_lower for w in ["example", "practical", "real world", "code", "scenario", "show me"]):
            reply = (
                f"### 🔍 Practical Example for **{topic}**\n\n"
                f"{previous_context}Here is a concrete scenario to illustrate how **{topic}** operates in practice:\n\n"
                f"- **Scenario**: 3 concurrent tasks arrive at the system ($T_1, T_2, T_3$).\n"
                f"- **Input Flow**:\n"
                f"  - $T_1$ requires 8 units of work.\n"
                f"  - $T_2$ requires 4 units of work.\n"
                f"  - $T_3$ requires 2 units of work.\n"
                f"- **Execution Decision**: The system evaluates arrival order, priority, and remaining time to schedule execution without starving any single task.\n\n"
                f"📌 *Check question*: How would the total turnaround time change if $T_3$ was executed before $T_1$?"
            )
        elif any(w in p_lower for w in ["difference", "compare", "vs", "versus", "distinction", "pros and cons"]):
            reply = (
                f"### ⚖️ Comparison in the context of **{topic}**\n\n"
                f"{previous_context}When comparing approaches within **{topic}**, evaluate these critical dimensions:\n\n"
                f"| Evaluation Metric | Approach A (Strict / Ordered) | Approach B (Dynamic / Adaptive) |\n"
                f"| :--- | :--- | :--- |\n"
                f"| **Fairness** | High (First come, first served) | Balanced across dynamic priorities |\n"
                f"| **Throughput** | Can suffer if long tasks block | Maximized for responsive tasks |\n"
                f"| **Overhead** | Minimal context switching | Higher state-tracking overhead |\n\n"
                f"Which specific two concepts in **{topic}** are you contrasting?"
            )
        elif any(phrase in p_lower for phrase in ["c programming", "c language", "about c", "what is c", "learn c", "explain c"]) or p_lower == "c":
            reply = (
                f"### 💻 C Programming Language Overview\n\n"
                f"{previous_context}**C** is a foundational, general-purpose procedural programming language created by Dennis Ritchie in 1972 at Bell Labs. It provides low-level memory access while maintaining structured programming principles.\n\n"
                f"#### 🔑 Core Pillars of C:\n"
                f"1. **Compiled & Ultra-Fast**: Translates directly to native machine code via compilers (GCC, Clang, MSVC) with near-zero runtime overhead.\n"
                f"2. **Pointers & Direct Memory Control**: Directly access and manipulate memory addresses, hardware registers, and system buffers.\n"
                f"3. **Dynamic Memory Allocation**: Manual heap allocation via `malloc()`, `calloc()`, `realloc()`, and deallocation with `free()`.\n"
                f"4. **Systems Software Pillar**: Serves as the bedrock for the Linux kernel, Windows, macOS, Git, Python runtimes, and embedded microcontrollers.\n\n"
                f"#### 📝 Hello World Example in C:\n"
                f"```c\n"
                f"#include <stdio.h>\n\n"
                f"int main(void) {{\n"
                f"    printf(\"Hello from FocusSense AI!\\n\");\n"
                f"    return 0;\n"
                f"}}\n"
                f"```\n\n"
                f"What would you like to explore next? Pointers, memory allocation, structs, or functions?"
            )
        else:
            reply = (
                f"### 🧠 FocusSense AI: **{topic}**\n\n"
                f"{previous_context}Regarding **'{prompt}'** in the context of **{topic}**:\n\n"
                f"- **Core Concept**: In **{topic}**, this relates directly to how the underlying system organizes resources and coordinates state transitions.\n"
                f"- **Mechanism**: It operates by tracking constraints, evaluating incoming inputs against deterministic rules, and executing the next appropriate operation.\n"
                f"- **Why it matters**: Understanding this foundational principle prevents performance degradation and ensures scalable architecture.\n\n"
                f"Would you like a deeper breakdown, an illustrative diagram/example, or a quick practice question on this?"
            )

        return {
            "reply": reply,
            "provider": "FocusSense Local AI",
            "model": "offline-guidance-v2",
            "success": True,
            "error": None,
        }

    def generate_quiz(self, topic: str, conversation_history: list = None, question_count: int = 3, adaptive_guidance: str = None, material_excerpts: str = None) -> list:
        t_clean = (topic or "General Study").strip()
        t_lower = t_clean.lower()

        # If material excerpts are provided, construct grounded material questions
        if material_excerpts:
            return [
                {
                    "question_index": 1,
                    "question_text": f"Based on the study material for '{t_clean}', what is the core operational principle and primary objective described?",
                    "sub_concept": f"{t_clean} Core Principles",
                    "sample_answer": f"The study material for {t_clean} outlines foundational parameters, structured workflows, and deterministic system behavior.",
                    "difficulty": "basic",
                    "points_possible": 1.0
                },
                {
                    "question_index": 2,
                    "question_text": f"According to the material on '{t_clean}', how are state transitions, inputs, or execution flows coordinated?",
                    "sub_concept": f"{t_clean} Mechanism & Execution Flow",
                    "sample_answer": f"Inputs are processed sequentially against validation rules and constraints to ensure correct state transitions in {t_clean}.",
                    "difficulty": "application",
                    "points_possible": 1.0
                },
                {
                    "question_index": 3,
                    "question_text": f"What key constraints, performance trade-offs, or common pitfalls are highlighted in the '{t_clean}' material?",
                    "sub_concept": f"{t_clean} Constraints & Edge Cases",
                    "sample_answer": f"Key trade-offs include balancing execution overhead against resource limits and handling boundary exceptions properly in {t_clean}.",
                    "difficulty": "advanced",
                    "points_possible": 1.0
                },
                {
                    "question_index": 4,
                    "question_text": f"How does the material on '{t_clean}' recommend verifying correctness or optimizing practical implementation?",
                    "sub_concept": f"{t_clean} Verification & Optimization",
                    "sample_answer": f"Verification involves rigorous testing of boundary conditions, monitoring state consistency, and following recommended optimization patterns for {t_clean}.",
                    "difficulty": "application",
                    "points_possible": 1.0
                }
            ][:question_count]

        # Domain: Java Datatypes / OOP
        if any(w in t_lower for w in ["java", "datatype", "type", "variable", "primitive", "polymorph"]):
            return [
                {
                    "question_index": 1,
                    "question_text": f"In {t_clean}, what is the fundamental difference between primitive data types and reference data types in memory storage?",
                    "sub_concept": "Primitive vs Reference Types",
                    "sample_answer": "Primitive types (e.g., int, char, boolean) store their raw binary values directly on the stack, whereas reference types store memory addresses pointing to objects allocated on the heap.",
                    "points_possible": 1.0
                },
                {
                    "question_index": 2,
                    "question_text": f"Why does Java define fixed bit-widths for numerical types (such as byte, short, int, long) across all operating systems?",
                    "sub_concept": "Platform Independence & Type Bounds",
                    "sample_answer": "Java enforces strict fixed sizes (e.g., 32-bit int, 64-bit long) to ensure deterministic WORA (Write Once, Run Anywhere) behavior and prevent architecture-dependent numeric overflow.",
                    "points_possible": 1.0
                },
                {
                    "question_index": 3,
                    "question_text": f"Explain the difference between implicit type casting (widening) and explicit type casting (narrowing) in {t_clean}.",
                    "sub_concept": "Type Casting & Data Precision",
                    "sample_answer": "Widening conversion happens automatically when converting a smaller type to a larger type without data loss. Narrowing requires explicit casting because it can cause truncation or loss of precision.",
                    "points_possible": 1.0
                }
            ][:question_count]

        # Domain: SQL / Databases
        elif any(w in t_lower for w in ["sql", "join", "dbms", "database", "query", "relation", "key"]):
            return [
                {
                    "question_index": 1,
                    "question_text": f"In {t_clean}, what is the difference between an INNER JOIN and a LEFT OUTER JOIN?",
                    "sub_concept": "Join Semantics & Matching",
                    "sample_answer": "INNER JOIN returns only rows that have matching values in both tables, whereas LEFT OUTER JOIN returns all rows from the left table and matched rows from the right table, filling NULLs where no match exists.",
                    "points_possible": 1.0
                },
                {
                    "question_index": 2,
                    "question_text": f"What is a Cartesian Product (CROSS JOIN) in {t_clean}, and why should unintended Cartesian joins be avoided in large tables?",
                    "sub_concept": "Cartesian Multiplicities",
                    "sample_answer": "A CROSS JOIN combines every row of the first table with every row of the second table ($M \\times N$ rows), leading to combinatorial explosion and severe query performance degradation.",
                    "points_possible": 1.0
                },
                {
                    "question_index": 3,
                    "question_text": f"How do indexing and ON clause predicates optimize join processing in relational database engines?",
                    "sub_concept": "Query Optimization & Indexing",
                    "sample_answer": "Indexes on join foreign keys allow the query optimizer to perform fast Index Hash Joins or Merge Joins instead of expensive Nested Loop full table scans.",
                    "points_possible": 1.0
                }
            ][:question_count]

        # Domain: Operating Systems / Scheduling
        elif any(w in t_lower for w in ["schedul", "operating system", "cpu", "process", "fcfs", "round robin", "os"]):
            return [
                {
                    "question_index": 1,
                    "question_text": "What is First-Come, First-Served (FCFS) scheduling, and what is the 'convoy effect'?",
                    "sub_concept": "FCFS Scheduling",
                    "sample_answer": "FCFS executes processes in arrival order non-preemptively. The convoy effect occurs when short CPU processes wait behind a long CPU-bound process, causing high average waiting time.",
                    "points_possible": 1.0
                },
                {
                    "question_index": 2,
                    "question_text": "How does Round Robin (RR) scheduling use a 'time quantum', and what happens if the quantum is set too small?",
                    "sub_concept": "Round Robin",
                    "sample_answer": "Round Robin assigns each process a fixed time slice (quantum). If the quantum is too small, excessive context switching creates massive CPU overhead and reduces throughput.",
                    "points_possible": 1.0
                },
                {
                    "question_index": 3,
                    "question_text": "What is the key difference between preemptive and non-preemptive scheduling algorithms?",
                    "sub_concept": "Preemption Mechanism",
                    "sample_answer": "Preemptive scheduling can interrupt a currently running process when a higher-priority task arrives, whereas non-preemptive allows a process to run until it finishes or yields.",
                    "points_possible": 1.0
                }
            ][:question_count]

        # Domain: C Programming / Systems
        elif any(w in t_lower for w in ["c programming", "c language", "pointer", "malloc", "struct", "memory leak"]) or t_lower == "c":
            return [
                {
                    "question_index": 1,
                    "question_text": "In C programming, what is a pointer, and how does the dereference operator (*) work?",
                    "sub_concept": "Pointers and Dereferencing",
                    "sample_answer": "A pointer is a variable that stores the memory address of another variable. The dereference operator (*) accesses or modifies the value stored at that memory address.",
                    "points_possible": 1.0
                },
                {
                    "question_index": 2,
                    "question_text": "What is the difference between stack and heap memory allocation in C, and why must malloc() always be paired with free()?",
                    "sub_concept": "Dynamic Memory Management",
                    "sample_answer": "Stack memory is automatically managed for local variables and function frames. Heap memory is manually allocated at runtime via malloc(). Unfreed heap memory causes memory leaks.",
                    "points_possible": 1.0
                },
                {
                    "question_index": 3,
                    "question_text": "What is the purpose of the C preprocessor and header files (#include <stdio.h>) before actual code compilation?",
                    "sub_concept": "C Preprocessor & Header Files",
                    "sample_answer": "The C preprocessor runs before compilation to expand macros, include header file function declarations and types, and handle conditional compilation directives.",
                    "points_possible": 1.0
                }
            ][:question_count]

        # Universal Dynamic Topic Generator
        return [
            {
                "question_index": 1,
                "question_text": f"What is the foundational principle behind '{t_clean}', and what core engineering or scientific challenge does it address?",
                "sub_concept": f"{t_clean} First Principles",
                "sample_answer": f"The core mechanism of {t_clean} solves fundamental coordination, representation, or efficiency challenges by establishing clear structural rules.",
                "points_possible": 1.0
            },
            {
                "question_index": 2,
                "question_text": f"In practical application of '{t_clean}', what are the primary trade-offs, bottlenecks, or constraints to consider?",
                "sub_concept": f"{t_clean} Architectural Constraints",
                "sample_answer": f"Working with {t_clean} requires balancing overhead, complexity, and performance against system scalability and correctness.",
                "points_possible": 1.0
            },
            {
                "question_index": 3,
                "question_text": f"How do you verify correctness or validate optimal performance when implementing '{t_clean}' in real-world environments?",
                "sub_concept": f"{t_clean} Verification & Edge Cases",
                "sample_answer": f"Validation requires testing boundary conditions, handling edge cases, and verifying state transitions under stress.",
                "points_possible": 1.0
            }
        ][:question_count]

    def evaluate_answer(self, question_text: str, sub_concept: str,
                        sample_answer: str, student_answer: str) -> dict:
        ans = (student_answer or "").strip()
        if not ans:
            return {
                "status": "unanswered",
                "score": 0.0,
                "feedback": "No answer was submitted for this question.",
                "success": True
            }

        ans_lower = ans.lower()
        sample_lower = sample_answer.lower()

        stop_words = {"the", "and", "that", "this", "with", "from", "when", "what", "which", "into", "over", "each", "were", "then", "have", "more", "also"}
        sample_words = [w for w in re.findall(r"[a-z0-9]+", sample_lower) if len(w) > 3 and w not in stop_words]
        
        matched_words = [w for w in sample_words if w in ans_lower]
        match_ratio = len(matched_words) / max(1, len(set(sample_words)))
        word_count = len(ans.split())

        if match_ratio >= 0.35 or (word_count >= 8 and match_ratio >= 0.20):
            status = "correct"
            score = 1.0
            feedback = f"Excellent comprehension! Your response accurately captures the core mechanisms of {sub_concept}."
        elif match_ratio >= 0.10 and word_count >= 4:
            status = "partial"
            score = 0.5
            feedback = f"Good foundational explanation for {sub_concept}, but consider expanding on key mechanics or edge cases."
        else:
            status = "incorrect"
            score = 0.0
            feedback = f"Needs review. Re-examine the key principles and operational definitions of {sub_concept}."

        return {
            "status": status,
            "score": score,
            "feedback": feedback,
            "success": True
        }

    def generate_assessment_questions(self, topic: str, mode: str = "PRACTICE", difficulty: str = "mixed",
                                      question_count: int = 5, conversation_history: list = None,
                                      material_excerpts: str = None) -> list:
        t_clean = (topic or "General Study").strip()
        t_lower = t_clean.lower()
        q_cnt = max(1, min(25, int(question_count or 5)))

        # Domain 1: Operating Systems & Process Management
        if any(w in t_lower for w in ["schedul", "operating system", "cpu", "process", "fcfs", "round robin", "os", "memory", "deadlock"]):
            bank = [
                {
                    "question_index": 1,
                    "question_text": "Which scheduling algorithm is non-preemptive and executes processes strictly in their arrival order?",
                    "sub_concept": "FCFS Scheduling",
                    "question_type": "mcq",
                    "category": "recall",
                    "difficulty": "basic",
                    "options": ["A) Round Robin", "B) First-Come, First-Served (FCFS)", "C) Shortest Remaining Time First", "D) Multilevel Feedback Queue"],
                    "correct_option": "B",
                    "sample_answer": "First-Come, First-Served (FCFS) assigns the CPU to the process that requests it first without preemption.",
                    "explanation": "FCFS is a non-preemptive algorithm that allocates the CPU based strictly on queue arrival time.",
                    "points_possible": 1.0
                },
                {
                    "question_index": 2,
                    "question_text": "Round Robin scheduling eliminates starvation by assigning each process a fixed time quantum.",
                    "sub_concept": "Round Robin Starvation Prevention",
                    "question_type": "true_false",
                    "category": "recall",
                    "difficulty": "basic",
                    "options": ["A) True", "B) False"],
                    "correct_option": "True",
                    "sample_answer": "True. Round Robin guarantees every ready process gets regular CPU turns, preventing indefinite starvation.",
                    "explanation": "Because of cyclic preemption every time quantum, no ready process waits indefinitely.",
                    "points_possible": 1.0
                },
                {
                    "question_index": 3,
                    "question_text": "Explain what the 'convoy effect' is in CPU scheduling and how it impacts average waiting time.",
                    "sub_concept": "Convoy Effect Dynamics",
                    "question_type": "conceptual",
                    "category": "concept",
                    "difficulty": "medium",
                    "options": None,
                    "correct_option": None,
                    "sample_answer": "The convoy effect occurs when short I/O-bound processes wait behind a long CPU-bound process in FCFS scheduling, causing high average wait times and poor device utilization.",
                    "explanation": "Convoy effect is a major limitation of non-preemptive FCFS where short tasks are blocked by CPU-heavy tasks.",
                    "points_possible": 1.0
                },
                {
                    "question_index": 4,
                    "question_text": "Given 3 processes with burst times P1=24ms, P2=3ms, P3=3ms arriving simultaneously at t=0. Calculate the average waiting time under FCFS.",
                    "sub_concept": "FCFS Waiting Time Calculation",
                    "question_type": "problem_solving",
                    "category": "problem_solving",
                    "difficulty": "medium",
                    "options": ["A) 17 ms", "B) 27 ms", "C) 30 ms", "D) 10 ms"],
                    "correct_option": "A",
                    "sample_answer": "Wait times: P1=0, P2=24, P3=27. Average = (0 + 24 + 27)/3 = 51/3 = 17 ms.",
                    "explanation": "Under FCFS, P1 waits 0ms, P2 waits 24ms, and P3 waits 24+3=27ms. (0+24+27)/3 = 17ms.",
                    "points_possible": 1.0
                },
                {
                    "question_index": 5,
                    "question_text": "In a real-time OS requiring high responsiveness, what trade-off occurs if the Round Robin time quantum is set to 1 microsecond?",
                    "sub_concept": "Quantum Size Trade-offs",
                    "question_type": "application",
                    "category": "application",
                    "difficulty": "advanced",
                    "options": None,
                    "correct_option": None,
                    "sample_answer": "Setting the quantum too small results in excessive context switching overhead, where the CPU spends more time saving/restoring registers than executing actual user code, degrading overall system throughput.",
                    "explanation": "Extremely small quantums degrade throughput due to context-switch overhead.",
                    "points_possible": 1.0
                },
                {
                    "question_index": 6,
                    "question_text": "What are the four Coffman conditions necessary for a deadlock to occur in an operating system?",
                    "sub_concept": "Deadlock Coffman Conditions",
                    "question_type": "short_answer",
                    "category": "concept",
                    "difficulty": "medium",
                    "options": None,
                    "correct_option": None,
                    "sample_answer": "The four conditions are Mutual Exclusion, Hold and Wait, No Preemption, and Circular Wait.",
                    "explanation": "All four conditions must hold simultaneously for a deadlock to exist.",
                    "points_possible": 1.0
                },
                {
                    "question_index": 7,
                    "question_text": "How does virtual memory paging eliminate external fragmentation in physical RAM?",
                    "sub_concept": "Paging & Memory Management",
                    "question_type": "conceptual",
                    "category": "concept",
                    "difficulty": "advanced",
                    "options": None,
                    "correct_option": None,
                    "sample_answer": "Paging divides physical memory into fixed-sized frames and logical memory into pages. Since any free frame can be allocated to any process page, contiguous physical memory is not required, eliminating external fragmentation.",
                    "explanation": "Fixed-size allocation avoids variable-sized gaps in physical memory.",
                    "points_possible": 1.0
                },
                {
                    "question_index": 8,
                    "question_text": "Which page replacement algorithm suffers from Belady's Anomaly where adding more frames increases page faults?",
                    "sub_concept": "Belady's Anomaly",
                    "question_type": "mcq",
                    "category": "recall",
                    "difficulty": "medium",
                    "options": ["A) Least Recently Used (LRU)", "B) First-In, First-Out (FIFO)", "C) Optimal Page Replacement (OPT)", "D) Clock Algorithm"],
                    "correct_option": "B",
                    "sample_answer": "FIFO page replacement can exhibit Belady's Anomaly because it is not a stack algorithm.",
                    "explanation": "FIFO does not maintain the stack property, meaning an increase in frame capacity can lead to more faults.",
                    "points_possible": 1.0
                },
                {
                    "question_index": 9,
                    "question_text": "A process requests memory causing a Translation Lookaside Buffer (TLB) miss. Trace the step-by-step resolution by the MMU.",
                    "sub_concept": "TLB Miss Resolution",
                    "question_type": "application",
                    "category": "application",
                    "difficulty": "advanced",
                    "options": None,
                    "correct_option": None,
                    "sample_answer": "On a TLB miss, the MMU accesses the Page Table in physical memory (page table walk), retrieves the frame number, updates the TLB cache, and then accesses the actual physical memory address.",
                    "explanation": "The hardware walks the page table structure to populate the TLB before accessing data.",
                    "points_possible": 1.0
                },
                {
                    "question_index": 10,
                    "question_text": "Implement or describe the condition to detect a cycle in a Resource Allocation Graph with single instance resources.",
                    "sub_concept": "Resource Allocation Graph Cycle",
                    "question_type": "problem_solving",
                    "category": "problem_solving",
                    "difficulty": "advanced",
                    "options": None,
                    "correct_option": None,
                    "sample_answer": "In single-instance resource systems, a directed cycle in the resource allocation graph is both a necessary and sufficient condition for deadlock. It can be detected using depth-first search cycle detection.",
                    "explanation": "Single-instance resource graphs deadlocked if and only if a directed cycle exists.",
                    "points_possible": 1.0
                }
            ]

        # Domain 2: Databases & SQL
        elif any(w in t_lower for w in ["sql", "join", "dbms", "database", "query", "relation", "normalization", "index"]):
            bank = [
                {
                    "question_index": 1,
                    "question_text": "Which SQL clause is used to filter aggregated grouped results created by a GROUP BY statement?",
                    "sub_concept": "SQL Aggregation & Having",
                    "question_type": "mcq",
                    "category": "recall",
                    "difficulty": "basic",
                    "options": ["A) WHERE", "B) HAVING", "C) ORDER BY", "D) GROUP FILTER"],
                    "correct_option": "B",
                    "sample_answer": "HAVING filters aggregated groups, whereas WHERE filters individual rows prior to grouping.",
                    "explanation": "HAVING operates on aggregated rows produced by GROUP BY.",
                    "points_possible": 1.0
                },
                {
                    "question_index": 2,
                    "question_text": "An INNER JOIN returns unmatched rows from the left table with NULL placeholders for the right table.",
                    "sub_concept": "SQL Join Mechanics",
                    "question_type": "true_false",
                    "category": "recall",
                    "difficulty": "basic",
                    "options": ["A) True", "B) False"],
                    "correct_option": "False",
                    "sample_answer": "False. LEFT OUTER JOIN retains unmatched left rows; INNER JOIN returns only matched rows.",
                    "explanation": "INNER JOIN discards rows that do not satisfy the join predicate in both tables.",
                    "points_possible": 1.0
                },
                {
                    "question_index": 3,
                    "question_text": "Explain the difference between Second Normal Form (2NF) and Third Normal Form (3NF).",
                    "sub_concept": "Database Normalization (2NF vs 3NF)",
                    "question_type": "conceptual",
                    "category": "concept",
                    "difficulty": "medium",
                    "options": None,
                    "correct_option": None,
                    "sample_answer": "2NF eliminates partial functional dependencies on composite primary keys. 3NF eliminates transitive functional dependencies where a non-prime attribute depends on another non-prime attribute.",
                    "explanation": "2NF removes partial key dependencies; 3NF removes transitive dependencies.",
                    "points_possible": 1.0
                },
                {
                    "question_index": 4,
                    "question_text": "Write a query to find all departments having more than 5 employees from an 'employees' table (dept_id, salary).",
                    "sub_concept": "SQL Group By Having Query",
                    "question_type": "problem_solving",
                    "category": "problem_solving",
                    "difficulty": "medium",
                    "options": None,
                    "correct_option": None,
                    "sample_answer": "SELECT dept_id, COUNT(*) FROM employees GROUP BY dept_id HAVING COUNT(*) > 5;",
                    "explanation": "Grouping by dept_id with a HAVING condition filters out departments with <= 5 employees.",
                    "points_possible": 1.0
                },
                {
                    "question_index": 5,
                    "question_text": "When is a B+ Tree index preferred over a Hash index in relational databases?",
                    "sub_concept": "Database Indexing Selection",
                    "question_type": "application",
                    "category": "application",
                    "difficulty": "advanced",
                    "options": None,
                    "correct_option": None,
                    "sample_answer": "B+ Tree indexes store keys in sorted sequential order, making them optimal for range queries (BETWEEN, <, >) and ORDER BY operations, whereas Hash indexes only support exact equality matches ($O(1)$).",
                    "explanation": "B+ Trees support efficient range traversals via linked leaf nodes.",
                    "points_possible": 1.0
                }
            ]

        # Domain 3: Programming, Python, Java, Data Structures & Algorithms
        elif any(w in t_lower for w in ["python", "java", "code", "function", "class", "algorithm", "tree", "graph", "recursion", "oop"]):
            bank = [
                {
                    "question_index": 1,
                    "question_text": "What is the time complexity of searching for an element in a balanced Binary Search Tree with $N$ nodes?",
                    "sub_concept": "BST Search Complexity",
                    "question_type": "mcq",
                    "category": "recall",
                    "difficulty": "basic",
                    "options": ["A) O(1)", "B) O(log N)", "C) O(N)", "D) O(N log N)"],
                    "correct_option": "B",
                    "sample_answer": "Searching a balanced BST has an average and worst-case time complexity of O(log N).",
                    "explanation": "Each comparison halves the remaining search space in a balanced tree.",
                    "points_possible": 1.0
                },
                {
                    "question_index": 2,
                    "question_text": "In Python, default mutable arguments (e.g. def func(x=[])) are evaluated once at function definition time.",
                    "sub_concept": "Python Mutable Defaults",
                    "question_type": "true_false",
                    "category": "concept",
                    "difficulty": "medium",
                    "options": ["A) True", "B) False"],
                    "correct_option": "True",
                    "sample_answer": "True. Python binds default argument expressions when the function definition is executed, mutating the shared object across calls.",
                    "explanation": "Default parameter values are evaluated once when the function is defined, not each time it is called.",
                    "points_possible": 1.0
                },
                {
                    "question_index": 3,
                    "question_text": "Explain the role of the base case in a recursive algorithm and describe what happens if it is omitted.",
                    "sub_concept": "Recursion & Base Cases",
                    "question_type": "conceptual",
                    "category": "concept",
                    "difficulty": "medium",
                    "options": None,
                    "correct_option": None,
                    "sample_answer": "The base case provides a termination condition where recursion stops. Without a base case, the function calls itself indefinitely, exhausting stack memory and causing a stack overflow (RecursionError).",
                    "explanation": "Base cases prevent unbounded call stack growth.",
                    "points_possible": 1.0
                },
                {
                    "question_index": 4,
                    "question_text": "Given an array of integers, design a linear $O(N)$ algorithm to find two numbers that sum to a target value.",
                    "sub_concept": "Two Sum Hash Map Technique",
                    "question_type": "problem_solving",
                    "category": "problem_solving",
                    "difficulty": "medium",
                    "options": None,
                    "correct_option": None,
                    "sample_answer": "Iterate through the array while maintaining a hash set or dictionary of seen values. For each element $x$, check if $(target - x)$ exists in the hash map. If found, return the pair; otherwise insert $x$.",
                    "explanation": "Hash table lookups achieve O(N) time with O(N) auxiliary space.",
                    "points_possible": 1.0
                },
                {
                    "question_index": 5,
                    "question_text": "How does Polymorphism via Interfaces promote loose coupling and extensibility in software architecture?",
                    "sub_concept": "Polymorphism & Loose Coupling",
                    "question_type": "application",
                    "category": "application",
                    "difficulty": "advanced",
                    "options": None,
                    "correct_option": None,
                    "sample_answer": "Polymorphism allows calling code to depend on abstract interface contracts rather than concrete implementations. New classes can be added without modifying existing consumer modules (Open/Closed Principle).",
                    "explanation": "Interfaces decouple the caller from specific implementation details.",
                    "points_possible": 1.0
                }
            ]

        # Domain 4: General Topic / Universal Fallback
        else:
            bank = [
            {
                "question_index": 1,
                "question_text": f"What is the foundational definition and primary purpose of '{t_clean}'?",
                "sub_concept": f"{t_clean} Core Definition",
                "question_type": "mcq",
                "category": "recall",
                "difficulty": "basic",
                "options": [
                    f"A) A systematic framework for organizing operations in {t_clean}",
                    f"B) An obsolete legacy mechanism superseded by hardware",
                    f"C) A temporary testing tool with no production use",
                    f"D) An unregulated theoretical concept without practical rules"
                ],
                "correct_option": "A",
                "sample_answer": f"The core purpose of {t_clean} is to establish structured rules, coordinate resources, and resolve operational bottlenecks.",
                "explanation": f"{t_clean} serves as a foundational structural discipline.",
                "points_possible": 1.0
            },
            {
                "question_index": 2,
                "question_text": f"Applying the principles of '{t_clean}' guarantees deterministic behavior across varying system constraints.",
                "sub_concept": f"{t_clean} Determinism",
                "question_type": "true_false",
                "category": "concept",
                "difficulty": "basic",
                "options": ["A) True", "B) False"],
                "correct_option": "True",
                "sample_answer": f"True. Proper application of {t_clean} enforces consistent and predictable outcomes.",
                "explanation": f"Structural rules in {t_clean} establish repeatable state transitions.",
                "points_possible": 1.0
            },
            {
                "question_index": 3,
                "question_text": f"Explain the key operational mechanism of '{t_clean}' and why it is critical for system correctness.",
                "sub_concept": f"{t_clean} Mechanism",
                "question_type": "conceptual",
                "category": "concept",
                "difficulty": "medium",
                "options": None,
                "correct_option": None,
                "sample_answer": f"In {t_clean}, incoming requests or state transitions are evaluated against invariant rules to prevent inconsistency and optimize performance.",
                "explanation": f"Understanding how {t_clean} enforces rules ensures reliable system operation.",
                "points_possible": 1.0
            },
            {
                "question_index": 4,
                "question_text": f"Describe a practical scenario where incorrect implementation of '{t_clean}' leads to failure or bottleneck.",
                "sub_concept": f"{t_clean} Practical Application",
                "question_type": "application",
                "category": "application",
                "difficulty": "advanced",
                "options": None,
                "correct_option": None,
                "sample_answer": f"If {t_clean} is misconfigured without handling edge cases, resource starvation or concurrency conflicts can cause cascading system degradation.",
                "explanation": f"Real-world application requires active constraint management.",
                "points_possible": 1.0
            },
            {
                "question_index": 5,
                "question_text": f"How do you evaluate and verify that an implementation of '{t_clean}' satisfies required performance benchmarks?",
                "sub_concept": f"{t_clean} Verification",
                "question_type": "problem_solving",
                "category": "problem_solving",
                "difficulty": "advanced",
                "options": None,
                "correct_option": None,
                "sample_answer": f"Verification is achieved by benchmarking stress conditions, monitoring throughput and error rates, and validating state consistency across edge cases.",
                "explanation": f"Empirical testing verifies that {t_clean} meets operational requirements.",
                "points_possible": 1.0
            }
        ]

        # Ensure bank has at least q_cnt questions for any requested length
        while len(bank) < q_cnt:
            idx = len(bank) + 1
            cat_cycle = ["recall", "concept", "application", "problem_solving"][(idx - 1) % 4]
            q_type_cycle = ["mcq", "true_false", "conceptual", "short_answer", "application", "problem_solving"][(idx - 1) % 6]

            if q_type_cycle == "mcq":
                opts = [
                    f"A) Principle {idx} of {t_clean} optimizing state operations",
                    f"B) An unstructured pattern avoiding standard invariants",
                    f"C) A deprecated hardware routine not used in software",
                    f"D) Non-deterministic random scheduling"
                ]
                corr = "A"
            elif q_type_cycle == "true_false":
                opts = ["A) True", "B) False"]
                corr = "True"
            else:
                opts = None
                corr = None

            bank.append({
                "question_index": idx,
                "question_text": f"Evaluate question #{idx} on '{t_clean}': describe how sub-concept '{t_clean} Part {idx}' enforces correct execution under resource constraints.",
                "sub_concept": f"{t_clean} Concept {idx}",
                "question_type": q_type_cycle,
                "category": cat_cycle,
                "difficulty": "medium" if idx <= 6 else "advanced",
                "options": opts,
                "correct_option": corr,
                "sample_answer": f"In {t_clean}, maintaining proper constraints and bounds ensures correct state transitions and eliminates race conditions or resource leaks.",
                "explanation": f"Properly applying {t_clean} guarantees deterministic and reliable behavior.",
                "points_possible": 1.0
            })

        return bank[:q_cnt]

    def explain_assessment_mistake(self, question_text: str, student_answer: str,
                                  ideal_answer: str, sub_concept: str, topic: str) -> str:
        s_ans = (student_answer or "").strip()
        if not s_ans:
            return (
                f"### 💡 FocusSense AI Socratic Guidance: **{sub_concept}**\n\n"
                f"It looks like this question was left unanswered. In the context of **{topic}**, **{sub_concept}** "
                f"focuses on the foundational mechanism: *\"{ideal_answer}\"*.\n\n"
                f"**Rule of Thumb**: When approaching {sub_concept}, always identify the input conditions and step-by-step rules before committing to an answer."
            )

        return (
            f"### 💡 FocusSense AI Socratic Review: **{sub_concept}**\n\n"
            f"You attempted: *\"{s_ans}\"*. While you're thinking along the right lines, your answer missed a key distinction in **{topic}**.\n\n"
            f"**The Core Mechanism**: {ideal_answer}\n\n"
            f"📌 **Key Takeaway**: Remember that in **{sub_concept}**, the fundamental invariant is maintaining deterministic ordering and avoiding unbounded overhead."
        )


# ---------------------------------------------------------------------------
# FocusSense AI Service Facade
# ---------------------------------------------------------------------------

class FocusSenseAIService:
    """
    Main Service orchestrator.
    Prioritizes local Ollama inference, falls back to Gemini API (if key is set),
    or provides high-quality offline guidance.
    """

    def __init__(self):
        self.ollama_provider = OllamaAIProvider()
        self.gemini_provider = GeminiAIProvider()
        self.local_provider = LocalMockAIProvider()
        self.force_local = os.environ.get("FORCE_LOCAL_AI", "").lower() in ("1", "true", "yes")

    def get_active_provider_info(self) -> dict:
        if not self.force_local and self.ollama_provider.is_available():
            return self.ollama_provider.get_info()
        if not self.force_local and self.gemini_provider.is_configured():
            return self.gemini_provider.get_info()
        return self.local_provider.get_info()

    def generate_chat_reply(self, prompt: str, conversation_history: list,
                            study_topic: str, student_name: str, learning_context: dict = None,
                            material_context: dict = None) -> dict:
        if not self.force_local:
            # Priority 1: Local Ollama LLM
            if self.ollama_provider.is_available():
                result = self.ollama_provider.generate_reply(
                    prompt=prompt,
                    conversation_history=conversation_history,
                    study_topic=study_topic,
                    student_name=student_name,
                    learning_context=learning_context,
                    material_context=material_context,
                )
                if result.get("success"):
                    return result

            # Priority 2: Gemini Cloud API (if configured)
            if self.gemini_provider.is_configured():
                result = self.gemini_provider.generate_reply(
                    prompt=prompt,
                    conversation_history=conversation_history,
                    study_topic=study_topic,
                    student_name=student_name,
                    learning_context=learning_context,
                    material_context=material_context,
                )
                if result.get("success"):
                    return result

        # Priority 3: Local intelligent fallback guidance
        local_res = self.local_provider.generate_reply(
            prompt=prompt,
            conversation_history=conversation_history,
            study_topic=study_topic,
            student_name=student_name,
            learning_context=learning_context,
            material_context=material_context,
        )
        if local_res.get("success"):
            return local_res

        if self.ollama_provider.is_available():
            return self.ollama_provider.generate_reply(
                prompt=prompt,
                conversation_history=conversation_history,
                study_topic=study_topic,
                student_name=student_name,
                learning_context=learning_context,
                material_context=material_context,
            )

        return local_res

    def generate_quiz(self, topic: str, conversation_history: list = None, question_count: int = 3, adaptive_guidance: str = None, material_excerpts: str = None) -> list:
        if not self.force_local:
            # 1. Try Ollama local LLM
            if self.ollama_provider.is_available():
                questions = self.ollama_provider.generate_quiz(topic, conversation_history, question_count, adaptive_guidance, material_excerpts)
                if questions:
                    return questions

            # 2. Try Gemini API
            if self.gemini_provider.is_configured():
                questions = self.gemini_provider.generate_quiz(topic, conversation_history, question_count, adaptive_guidance, material_excerpts)
                if questions:
                    return questions

        # 3. Dynamic grounded heuristic fallback
        return self.local_provider.generate_quiz(topic, conversation_history, question_count, adaptive_guidance, material_excerpts)

    def generate_assessment_questions(self, topic: str, mode: str = "PRACTICE", difficulty: str = "mixed",
                                      question_count: int = 5, conversation_history: list = None,
                                      material_excerpts: str = None) -> list:
        if not self.force_local:
            # 1. Try Ollama local LLM
            if self.ollama_provider.is_available():
                questions = self.ollama_provider.generate_assessment_questions(topic, mode, difficulty, question_count, conversation_history, material_excerpts)
                if questions:
                    return questions

            # 2. Try Gemini API
            if self.gemini_provider.is_configured():
                questions = self.gemini_provider.generate_assessment_questions(topic, mode, difficulty, question_count, conversation_history, material_excerpts)
                if questions:
                    return questions

        # 3. Local intelligent fallback
        return self.local_provider.generate_assessment_questions(topic, mode, difficulty, question_count, conversation_history, material_excerpts)

    def evaluate_answer(self, question_text: str, sub_concept: str,
                        sample_answer: str, student_answer: str) -> dict:
        if not self.force_local:
            # 1. Try Ollama local LLM
            if self.ollama_provider.is_available():
                res = self.ollama_provider.evaluate_answer(question_text, sub_concept, sample_answer, student_answer)
                if res and res.get("success"):
                    return res

            # 2. Try Gemini API
            if self.gemini_provider.is_configured():
                res = self.gemini_provider.evaluate_answer(question_text, sub_concept, sample_answer, student_answer)
                if res and res.get("success"):
                    return res

        # 3. Local semantic rubric engine
        return self.local_provider.evaluate_answer(question_text, sub_concept, sample_answer, student_answer)

    def explain_assessment_mistake(self, question_text: str, student_answer: str,
                                  ideal_answer: str, sub_concept: str, topic: str) -> str:
        if not self.force_local:
            if self.ollama_provider.is_available():
                res = self.ollama_provider.explain_assessment_mistake(question_text, student_answer, ideal_answer, sub_concept, topic)
                if res:
                    return res

            if self.gemini_provider.is_configured():
                res = self.gemini_provider.explain_assessment_mistake(question_text, student_answer, ideal_answer, sub_concept, topic)
                if res:
                    return res

        return self.local_provider.explain_assessment_mistake(question_text, student_answer, ideal_answer, sub_concept, topic)

    def generate_title(self, prompt: str, topic: str = "") -> str:
        """Generate a concise 2 to 4 word title for a study conversation thread."""
        clean = re.sub(r'[^\w\s-]', '', prompt).strip()
        words = clean.split()
        starters = {"what", "is", "how", "to", "why", "can", "you", "explain", "tell", "me", "about", "the", "a", "an", "in", "for", "please", "detail", "details"}
        meaningful = [w for w in words if w.lower() not in starters]

        if meaningful:
            return " ".join(meaningful[:4]).title()[:35]

        if topic and topic.strip() and topic != "General Study":
            parts = [pt.strip() for pt in topic.split("-") if pt.strip()]
            if len(parts) > 1:
                return parts[-1][:35]
            return topic[:35]

        if len(words) <= 4 and words:
            return clean.title()[:35]

        return " ".join(words[:4]).title()[:35] if words else "Study Discussion"


# Global service instance
ai_service = FocusSenseAIService()
