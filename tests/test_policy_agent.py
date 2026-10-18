import copy
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
from answer_check import problems
from agent import answer_from_passages, evidence_answer
from index_store import inspect_contents, open_store
from langchain_core.documents import Document
from policy_loader import load_policy_documents

PASSAGES = [{"policy_id": "doc-1", "source": "policy.txt",
             "text": "Unreleased print jobs are retained for 24 hours."}]
SUPPORTED = {"unknown": False, "answer": PASSAGES[0]["text"], "policy_id": "doc-1",
             "quote": PASSAGES[0]["text"]}
UNKNOWN = {"unknown": True, "answer": "I don't know.", "policy_id": "", "quote": ""}
PASS = {"fact_available": True, "answer_supported": True, "reason": "Matches the evidence"}
REVISE = {"fact_available": True, "answer_supported": False, "reason": "Unsupported response"}
ABSENT = {"fact_available": False, "answer_supported": True, "reason": "The fact is absent"}


class EvidenceTests(unittest.TestCase):
    def test_supported_answer(self):
        self.assertEqual(problems(SUPPORTED, PASSAGES), [])

    def test_unknown_is_bare(self):
        self.assertEqual(problems(UNKNOWN, PASSAGES), [])
        for field, value in (("answer", "I don't know. Contact IT."), ("quote", "Unrelated"),
                             ("policy_id", "doc-1")):
            self.assertTrue(problems({**UNKNOWN, field: value}, PASSAGES))

    def test_wrong_citation(self):
        self.assertTrue(problems({**SUPPORTED, "policy_id": "missing"}, PASSAGES))

    def test_invented_quote(self):
        self.assertTrue(problems({**SUPPORTED, "quote": "Keep for 48 hours."}, PASSAGES))

    def test_unsupported_number(self):
        self.assertTrue(problems({**SUPPORTED, "answer": "48 hours."}, PASSAGES))

    def test_answer_cannot_change_the_cited_fact(self):
        self.assertTrue(problems({**SUPPORTED, "answer": "Use Campus-Private."}, PASSAGES))

    def test_empty_answer(self):
        self.assertTrue(problems({**SUPPORTED, "answer": ""}, PASSAGES))

    def test_inconsistent_unknown_flag(self):
        self.assertTrue(problems({**SUPPORTED, "answer": "I don't know."}, PASSAGES))


class ReflectionTests(unittest.TestCase):
    @patch("agent.invoke_json")
    def test_accepts_reviewed_answer(self, model):
        model.side_effect = [SUPPORTED, PASS]
        result = answer_from_passages("How long?", PASSAGES, [])
        self.assertEqual(result["answer"], SUPPORTED["quote"])
        self.assertEqual(model.call_count, 2)

    @patch("agent.invoke_json")
    def test_revises_false_unknown(self, model):
        model.side_effect = [UNKNOWN, REVISE, SUPPORTED, PASS]
        result = answer_from_passages("How long?", PASSAGES, [])
        self.assertFalse(result["draft"]["unknown"])
        self.assertEqual(len(result["reviews"]), 2)

    @patch("agent.invoke_json")
    def test_local_rejection_overrides_model_pass(self, model):
        model.side_effect = [{**SUPPORTED, "quote": "48 hours."}, PASS, SUPPORTED, PASS]
        self.assertEqual(answer_from_passages("How long?", PASSAGES, [])["answer"], SUPPORTED["quote"])
        self.assertEqual(model.call_count, 4)

    @patch("agent.invoke_json")
    def test_second_failure_is_an_error(self, model):
        model.side_effect = [UNKNOWN, REVISE, UNKNOWN, REVISE]
        with self.assertRaisesRegex(RuntimeError, "did not pass"):
            answer_from_passages("How long?", PASSAGES, [])

    @patch("agent.invoke_json")
    def test_missing_fact_can_be_unknown(self, model):
        model.side_effect = [UNKNOWN, ABSENT]
        self.assertEqual(answer_from_passages("Which room?", PASSAGES, [])["answer"], "I don't know.")

    @patch("agent.invoke_json")
    def test_known_answer_requires_available_fact(self, model):
        model.side_effect = [SUPPORTED, ABSENT, UNKNOWN, ABSENT]
        result = answer_from_passages("Which room?", PASSAGES, [])
        self.assertTrue(result["draft"]["unknown"])
        self.assertEqual(len(result["reviews"]), 2)

    @patch("agent.invoke_json")
    def test_model_failure_is_not_unknown(self, model):
        model.side_effect = ConnectionError("offline")
        with self.assertRaises(ConnectionError):
            answer_from_passages("Which room?", PASSAGES, [])

    def test_empty_retrieval_fails(self):
        with self.assertRaises(ValueError):
            answer_from_passages("Question", [], [])

    def test_answer_is_the_model_selected_quote(self):
        self.assertEqual(evidence_answer(SUPPORTED)["answer"], SUPPORTED["quote"])

    def test_empty_evidence_uses_generic_unknown_response(self):
        draft = evidence_answer({"quote": "", "policy_id": ""})
        self.assertEqual(draft, UNKNOWN)

    def test_empty_evidence_cannot_cite_an_unrelated_policy(self):
        draft = evidence_answer({"quote": "", "policy_id": "doc-1"})
        self.assertEqual(draft, UNKNOWN)


class RuntimeConfigTests(unittest.TestCase):
    def test_embedding_configuration_matches_installed_library(self):
        with TemporaryDirectory() as directory, patch("langchain_chroma.Chroma") as chroma:
            open_store("http://127.0.0.1:11529", Path(directory))
            embedding = chroma.call_args.kwargs["embedding_function"]
            self.assertEqual(embedding.keep_alive, 120)
            self.assertEqual(embedding.num_thread, 2)
            self.assertEqual(embedding.num_gpu, 99)


class StoreTests(unittest.TestCase):
    def setUp(self):
        self.docs = [Document(page_content="Original", metadata={"policy_id": "doc-1", "source": "policy.txt"})]
        self.content = {"ids": ["doc-1"], "documents": ["Original"],
                        "metadatas": [{"policy_id": "doc-1", "source": "policy.txt"}]}
        class Store:
            def __init__(self, outer):
                self.outer = outer
            def get(self, **kwargs):
                return self.outer.content
        self.store = Store(self)

    def test_exact_contents(self):
        self.assertEqual(inspect_contents(self.store, self.docs), {"doc-1"})

    def test_changed_text(self):
        self.content["documents"] = ["Altered"]
        with self.assertRaises(ValueError):
            inspect_contents(self.store, self.docs)

    def test_changed_metadata(self):
        self.content["metadatas"][0]["source"] = "another.txt"
        with self.assertRaises(ValueError):
            inspect_contents(self.store, self.docs)

    def test_unknown_document(self):
        self.content["ids"] = ["doc-9"]
        with self.assertRaises(ValueError):
            inspect_contents(self.store, self.docs)

    def test_partial_index(self):
        self.content = {"ids": [], "documents": [], "metadatas": []}
        self.assertEqual(inspect_contents(self.store, self.docs, complete=False), set())
        with self.assertRaises(ValueError):
            inspect_contents(self.store, self.docs)

    def test_course_loader_has_unique_ids(self):
        docs = load_policy_documents()
        self.assertGreater(len(docs), 1)
        self.assertEqual(len(docs), len({doc.metadata["policy_id"] for doc in docs}))
        self.assertEqual({doc.metadata["source"] for doc in docs}, {"university_it_policies.txt"})


if __name__ == "__main__":
    unittest.main()
