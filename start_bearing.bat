@echo off
cd /d "%~dp0"
python bearing_view.py
if errorlevel 1 pause

