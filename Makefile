install:
	python -m pip install -r requirements.txt

run:
	uvicorn app.main:app --reload --port 8000

test:
	pytest -q

demo:
	python scripts/demo.py

curl-in:
	curl -s -X POST http://localhost:8000/api/query -H "Content-Type: application/json" -d '{"question":"What is the policy on database backup retention periods?"}'

curl-out:
	curl -s -X POST http://localhost:8000/api/query -H "Content-Type: application/json" -d '{"question":"What is the travel reimbursement limit?"}'
