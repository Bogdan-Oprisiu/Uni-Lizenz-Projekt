@echo off

REM Activate the virtual environment. Adjust the path if needed.
REM For example, if your venv is in .venv:
call .venv\Scripts\activate

REM Run the FastAPI app with uvicorn on port 8200
python -m uvicorn main:app --reload --port 8200
