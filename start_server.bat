@echo off
REM Quick start script for the backend server (Windows)

echo === Quest Controller Tracking Server ===
echo.

REM Check if virtual environment exists
if not exist "backend\venv" (
    echo Setting up virtual environment...
    cd backend
    python -m venv venv
    call venv\Scripts\activate
    pip install -r requirements.txt
    cd ..
) else (
    echo Activating virtual environment...
    call backend\venv\Scripts\activate
)

REM Get local IP
echo.
echo Detecting network interface...
for /f "tokens=2 delims=:" %%a in ('ipconfig ^| findstr /C:"IPv4"') do (
    set LOCAL_IP=%%a
    goto :found
)
:found
set LOCAL_IP=%LOCAL_IP:~1%

REM Check and generate SSL certificates
echo.
echo Checking SSL certificates...
cd backend
if not exist "certs\cert.pem" (
    echo Generating SSL certificates (required for WebXR)...
    python generate_cert.py --ip %LOCAL_IP%
    echo.
)
cd ..

echo.
echo === Server Information ===
echo Local IP: %LOCAL_IP%
echo HTTPS URL: https://%LOCAL_IP%:8000
echo.
echo On your Meta Quest 3:
echo 1. Open browser and go to: https://%LOCAL_IP%:8000
echo 2. Accept the security warning (self-signed certificate)
echo 3. Update WebSocket URL to: wss://%LOCAL_IP%:8000/ws
echo 4. Click 'Connect to Server'
echo 5. Click 'Enter VR'
echo.
echo Note: HTTPS is required for WebXR to work!
echo.
echo Starting server...
echo.

REM Start server
cd backend
python server.py
