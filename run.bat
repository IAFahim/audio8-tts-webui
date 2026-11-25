@echo off
setlocal enabledelayedexpansion
cd /d "%~dp0"

if not defined AUDIO8_HOST set "AUDIO8_HOST=127.0.0.1"
if not defined AUDIO8_PORT set "AUDIO8_PORT=8999"
if not defined AUDIO8_MODEL_DIR set "AUDIO8_MODEL_DIR=%CD%\models\Audio8-TTS-Realistic-Air"
if not defined AUDIO8_DATA_DIR set "AUDIO8_DATA_DIR=%CD%\data"
if not defined AUDIO8_DEVICE set "AUDIO8_DEVICE=auto"
if not defined PYTORCH_CUDA_ALLOC_CONF set "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"

set "VENV_DIR=%CD%\.venv"
set "VENV_PY=%VENV_DIR%\Scripts\python.exe"

where uv >nul 2>nul
if %errorlevel%==0 (
  set "HAS_UV=1"
) else (
  set "HAS_UV=0"
)

if not exist "%VENV_PY%" (
  if "%HAS_UV%"=="1" (
    echo [run.bat] Creating .venv with uv...
    uv venv --python 3.11 "%VENV_DIR%" || uv venv "%VENV_DIR%"
  ) else (
    where py >nul 2>nul
    if %errorlevel%==0 (
      py -3.11 -m venv "%VENV_DIR%" 2>nul || py -3 -m venv "%VENV_DIR%"
    ) else (
      python -m venv "%VENV_DIR%"
    )
  )
)

if not exist "%VENV_PY%" (
  echo [run.bat] Error: Failed to create Python virtual environment. Install Python 3.11 or uv.
  pause
  exit /b 1
)

set "USE_CUDA=0"
where nvidia-smi >nul 2>nul
if %errorlevel%==0 (
  nvidia-smi -L >nul 2>nul
  if %errorlevel%==0 set "USE_CUDA=1"
)

set "MODE_KEY=cpu"
if "%USE_CUDA%"=="1" set "MODE_KEY=cu124"
set "STAMP_FILE=%VENV_DIR%\.deps_installed"
set "NEED_INSTALL=1"
if exist "%STAMP_FILE%" (
  set /p CURR_STAMP=<"%STAMP_FILE%"
  if "!CURR_STAMP!"=="%MODE_KEY%" set "NEED_INSTALL=0"
)

if "%NEED_INSTALL%"=="1" (
  if "%USE_CUDA%"=="1" (
    echo [run.bat] Installing PyTorch 2.5.0+cu124...
    set "TORCH_INDEX=--index-url https://download.pytorch.org/whl/cu124"
  ) else (
    echo [run.bat] Installing PyTorch 2.5.0 ^(CPU mode^)...
    set "TORCH_INDEX="
  )
  if "%HAS_UV%"=="1" (
    set "UV_NO_CONFIG=1"
    set "UV_HTTP_TIMEOUT=300"
    set "UV_CONCURRENT_DOWNLOADS=2"
    uv pip install --python "%VENV_PY%" torch==2.5.0 torchaudio==2.5.0 !TORCH_INDEX!
    if errorlevel 1 goto :err
    uv pip install --python "%VENV_PY%" -r requirements.txt
    if errorlevel 1 goto :err
  ) else (
    "%VENV_PY%" -m pip install --upgrade pip
    "%VENV_PY%" -m pip install torch==2.5.0 torchaudio==2.5.0 !TORCH_INDEX!
    if errorlevel 1 goto :err
    "%VENV_PY%" -m pip install -r requirements.txt
    if errorlevel 1 goto :err
  )
  > "%STAMP_FILE%" echo %MODE_KEY%
)

if not exist "%AUDIO8_MODEL_DIR%\model.safetensors" (
  echo [run.bat] Downloading Audio8-TTS Realistic-Air weights...
  "%VENV_PY%" download_model.py --dir "%AUDIO8_MODEL_DIR%"
  if errorlevel 1 goto :err
)

set "APP_URL=http://127.0.0.1:%AUDIO8_PORT%"
start "" /b powershell -NoProfile -Command "for ($i=0; $i -lt 120; $i++) { try { $r = Invoke-RestMethod -Uri '%APP_URL%/api/health' -TimeoutSec 2; if ($r.ready) { Start-Process '%APP_URL%'; break } } catch {}; Start-Sleep -Seconds 1 }"

echo [run.bat] Starting Audio8-TTS Web UI at %APP_URL%...
"%VENV_PY%" -m uvicorn --app-dir "%CD%" server:app --host "%AUDIO8_HOST%" --port "%AUDIO8_PORT%"
goto :eof

:err
echo [run.bat] Setup or launch failed.
pause
exit /b 1
