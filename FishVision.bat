@echo off
REM Default: conda env py313. Override: set AUTO_LABEL_PYTHON=C:\path\to\python.exe
set "FVT_PYTHON=C:\ProgramData\Anaconda3\envs\py313\python.exe"
pushd "%~dp0"
if defined AUTO_LABEL_PYTHON if exist "%AUTO_LABEL_PYTHON%" goto run_custom
if exist "%FVT_PYTHON%" goto run_py313
if exist "C:\ProgramData\Anaconda3\python.exe" goto run_conda
where py >nul 2>nul && goto run_py
goto run_python

:run_custom
start "FishVision" cmd /K cd /d "%CD%" ^&^& "%AUTO_LABEL_PYTHON%" src\FishVision_GUI.py
goto eof

:run_py313
start "FishVision" cmd /K cd /d "%CD%" ^&^& "%FVT_PYTHON%" src\FishVision_GUI.py
goto eof

:run_conda
start "FishVision" cmd /K cd /d "%CD%" ^&^& "C:\ProgramData\Anaconda3\python.exe" src\FishVision_GUI.py
goto eof

:run_py
start "FishVision" cmd /K cd /d "%CD%" ^&^& py -3 src\FishVision_GUI.py
goto eof

:run_python
start "FishVision" cmd /K cd /d "%CD%" ^&^& python src\FishVision_GUI.py

:eof
popd
