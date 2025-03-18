#!/usr/bin/env bash

# Exit immediately if a command exits with a non-zero status.
set -e

# Activate your virtual environment. Adjust the path if needed.
# For example, if your venv is in .venv:
source .venv/bin/activate

# Run the FastAPI app with uvicorn on port 8200
python -m uvicorn main:app --reload --port 8200
