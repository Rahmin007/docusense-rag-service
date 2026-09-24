import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fastapi.testclient import TestClient

# Keep the demo deterministic and offline by default.
os.environ.setdefault("EMBEDDING_PROVIDER", "tfidf")
os.environ.setdefault("LLM_PROVIDER", "extractive")
os.environ.setdefault("SIMILARITY_THRESHOLD", "0.20")

from app.main import app

client = TestClient(app)


def run(question: str) -> None:
    response = client.post("/api/query", json={"question": question})
    response.raise_for_status()
    print(json.dumps(response.json(), indent=2))


if __name__ == "__main__":
    print("=== In-scope query ===")
    run("What is the policy on database backup retention periods?")
    print("\n=== Out-of-scope query ===")
    run("What is the office gym membership reimbursement amount?")
