@echo off
echo Activating virtual environment...
call .venv\Scripts\activate.bat

echo Starting Streamlit front end...
cd front-end
start cmd /k "streamlit run app.py"
cd ..

echo Streamlit service has been started.
pause
