@echo off
:: ============================================================
::  BinaryBridge RAG – Chatbot Launcher
::  Double-click this file OR type: run
::  Works from any directory.
:: ============================================================

SET "SCRIPT_DIR=%~dp0"
SET "VENV_PY=%SCRIPT_DIR%.venv\Scripts\python.exe"

echo.
echo  ============================================================
echo    BinaryBridge RAG Chatbot  ^|  PMKVY Scheme Q^&A
echo  ============================================================
echo.

IF NOT EXIST "%VENV_PY%" (
    echo  [ERROR] Virtual environment not found at:
    echo          %VENV_PY%
    echo.
    echo  Please run: python -m venv .venv
    echo              .venv\Scripts\pip install -r requirements.txt
    pause
    exit /b 1
)

"%VENV_PY%" "%SCRIPT_DIR%RAG.py"
pause
