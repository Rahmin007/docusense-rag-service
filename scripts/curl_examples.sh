#!/usr/bin/env bash
set -euo pipefail

curl -s http://localhost:8000/health | python -m json.tool

echo "\n=== In-scope query ==="
curl -s -X POST http://localhost:8000/api/query \
  -H 'Content-Type: application/json' \
  -d '{"question":"What is the policy on database backup retention periods?"}' | python -m json.tool

echo "\n=== Out-of-scope fallback ==="
curl -s -X POST http://localhost:8000/api/query \
  -H 'Content-Type: application/json' \
  -d '{"question":"What is the office gym membership reimbursement amount?"}' | python -m json.tool
