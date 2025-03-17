@echo off
REM Activate your virtual environment if not already activated.
REM For example, if you are using CMD and your virtual environment is located at ".venv":
call .venv\Scripts\activate

REM Install dependencies using pip
pip install -r requirements.txt

echo.
echo "All dependencies have been installed successfully."
pause
