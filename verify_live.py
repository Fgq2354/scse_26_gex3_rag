"""Run real-model checks and retain failures as well as successes."""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
from agent import ask, review_answer
from settings import BASE_URL, MODEL, ROOT, server_json

QUESTIONS = [
    ("phone", "What exact phone number should I call for the University IT Service Desk?", None),
    ("printing", "How long are unreleased print jobs retained?", "24"),
    ("wifi", "What Wi-Fi network should a campus visitor use?", "Campus-Guest"),
    ("accounts", "How long should a temporarily locked account be left before another sign-in attempt?", "15"),
]


def verify(base_url, record_path):
    report = {
        "model": MODEL, "status": "running", "questions": [], "controls": [],
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "source_sha256": hashlib.sha256((ROOT / "university_it_policies.txt").read_bytes()).hexdigest(),
        "code_sha256": {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                       for path in sorted(ROOT.glob("*.py"))},
        "packages": {name: importlib.metadata.version(name) for name in (
            "langchain-core", "langchain-ollama", "langchain-chroma", "chromadb", "pydantic")},
    }
    record_path.parent.mkdir(parents=True, exist_ok=True)
    def save():
        record_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    try:
        for name, question, expected in QUESTIONS:
            result = ask(question, base_url)
            result["case"] = name
            report["questions"].append(result)
            save()
            if result["status"] != "accepted":
                raise AssertionError(f"{name}: answer review failed")
            if name == "phone":
                assert result["answer"] == "I don't know.", result["answer"]
                assert result["draft"]["unknown"]
            else:
                assert not result["draft"]["unknown"], result["answer"]
                assert expected.casefold() in result["answer"].casefold(), result["answer"]
        printing, wifi = report["questions"][1:3]
        false_unknown = {"unknown": True, "answer": "I don't know.", "policy_id": "", "quote": ""}
        wrong_network = {**wifi["draft"], "answer": "Use Campus-Private.", "unknown": False}
        for name, source, draft in [
            ("unknown_despite_evidence", printing, false_unknown),
            ("unsupported_network", wifi, wrong_network),
        ]:
            calls = []
            verdict = review_answer(source["question"], draft, source["passages"], calls, base_url)
            report["controls"].append({"case": name, "draft": draft, "review": verdict, "model_calls": calls})
            save()
            assert not verdict["accepted"], name
        processes = server_json(base_url + "/api/ps")["models"]
        report["loaded_models"] = processes
        qwen = next(item for item in processes if item["name"] == MODEL)
        assert qwen["size_vram"] >= qwen["size"] * 0.95, "Qwen is not fully GPU-offloaded"
        report["status"] = "passed"
    except Exception as error:
        report.update(status="failed", error=str(error))
        raise
    finally:
        report["finished_utc"] = datetime.now(timezone.utc).isoformat()
        save()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default=BASE_URL)
    parser.add_argument("--record", type=Path, required=True)
    args = parser.parse_args()
    verify(args.base_url, args.record)
    print("Four live questions and two negative controls passed")
