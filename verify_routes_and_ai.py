import unittest
from app import app
from ai_service import ai_service
from recall_engine import schedule_recall_task, start_recall_test, submit_recall_answer, finalize_recall_test

class TestFocusSenseSystemIntegrity(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = app
        cls.app.config["TESTING"] = True
        cls.client = cls.app.test_client()

    def test_routes(self):
        with self.client.session_transaction() as sess:
            sess["user_id"] = 1
            sess["username"] = "testuser"
            sess["fullname"] = "Test Student"

        for route in ["/dashboard", "/ai", "/reports", "/parent"]:
            res = self.client.get(route)
            self.assertEqual(res.status_code, 200, f"Route {route} failed with status {res.status_code}")
            print(f"PASS: {route} returned HTTP 200")

    def test_ai_chat(self):
        res = ai_service.generate_chat_reply(
            prompt="What is process scheduling in operating systems in one short sentence?",
            conversation_history=[],
            study_topic="Operating Systems",
            student_name="Test Student"
        )
        self.assertTrue(res.get("success"), "AI chat failed")
        self.assertIn("reply", res)
        self.assertTrue(len(res["reply"]) > 10)
        provider = res.get("provider")
        self.assertIn(provider, ["Cohere AI", "FocusSense Local AI"], f"Unexpected provider: {provider}")
        print(f"PASS: AI Chat returned valid reply from provider '{provider}'")

    def test_quiz_and_recall_flow(self):
        topic = "Operating Systems"
        task_info = schedule_recall_task(user_id=1, session_id=1, topic=topic, delay_minutes=0)
        task_id = task_info["task_id"]

        start_res = start_recall_test(task_id=task_id, user_id=1, question_count=3)
        self.assertTrue(start_res.get("success"))
        self.assertEqual(len(start_res["questions"]), 3)

        q = start_res["questions"][0]
        sub_res = submit_recall_answer(question_id=q["id"], student_answer="Process scheduling manages CPU execution order.", user_id=1)
        self.assertTrue(sub_res.get("success"))

        fin_res = finalize_recall_test(task_id=task_id, user_id=1)
        self.assertTrue(fin_res.get("success"))
        self.assertIn("physical_focus", fin_res)
        self.assertIn("learning_focus", fin_res)
        self.assertIn("final_focus_score", fin_res)
        self.assertIn("learned_well", fin_res)
        self.assertIn("needs_revision", fin_res)
        print(f"PASS: Recall engine cycle completed with 2-dimension score: Physical={fin_res['physical_focus']}%, Learning={fin_res['learning_focus']}%, Final={fin_res['final_focus_score']}%")

if __name__ == "__main__":
    unittest.main()
