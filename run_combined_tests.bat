@echo off
setlocal
cd /d "%~dp0"
set PYTHONPATH=.
echo === repository tests ===
python -m pytest -q tests
if errorlevel 1 exit /b 1
python scripts\build_combined_results.py --check
if errorlevel 1 exit /b 1
echo === USA tests ===
cd markets\usa
set PYTHONPATH=.
python -m pytest -q tests
if errorlevel 1 exit /b 1
echo === India tests ===
cd ..\india
set PYTHONPATH=.
python -m pytest -q tests
if errorlevel 1 exit /b 1
endlocal
