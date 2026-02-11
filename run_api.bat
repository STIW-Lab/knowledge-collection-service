@echo off
REM Startup script for FastAPI RAG service (Windows)

echo Starting Knowledge Collection RAG API...
echo.

REM Check if .env exists
if not exist .env (
    echo Warning: .env file not found
    echo Please create a .env file with required variables:
    echo   - OPENAI_API_KEY
    echo   - POSTGRES_HOST, POSTGRES_PORT, POSTGRES_DB
    echo   - POSTGRES_USER, POSTGRES_PASSWORD
    echo.
    set /p CONTINUE="Continue anyway? (y/n) "
    if /i not "%CONTINUE%"=="y" exit /b 1
)

REM Default port
if "%PORT%"=="" set PORT=8000

echo Starting FastAPI on port %PORT%...
echo.

REM Run with uvicorn
uv run uvicorn api.main:app --host 0.0.0.0 --port %PORT% --reload
