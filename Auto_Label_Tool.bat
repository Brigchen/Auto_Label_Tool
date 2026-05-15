@echo off
REM Prefer All Users Anaconda (common on Windows). Override: set AUTO_LABEL_PYTHON=C:\path\to\python.exe
REM Fallback: py -3, then python on PATH.
pushd "%~dp0"
if defined AUTO_LABEL_PYTHON if exist "%AUTO_LABEL_PYTHON%" goto run_custom
if exist "C:\ProgramData\Anaconda3\python.exe" goto run_conda
where py >nul 2>nul && goto run_py
goto run_python

:run_custom
start "Auto_Label_Tool" cmd /K cd /d "%CD%" ^&^& "%AUTO_LABEL_PYTHON%" src\ALT.py
goto eof

:run_conda
start "Auto_Label_Tool" cmd /K cd /d "%CD%" ^&^& "C:\ProgramData\Anaconda3\python.exe" src\ALT.py
goto eof

:run_py
start "Auto_Label_Tool" cmd /K cd /d "%CD%" ^&^& py -3 src\ALT.py
goto eof

:run_python
start "Auto_Label_Tool" cmd /K cd /d "%CD%" ^&^& python src\ALT.py

:eof
popd
