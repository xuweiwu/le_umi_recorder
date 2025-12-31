#!/bin/bash
# Quick start script for the backend server

echo "=== Quest Controller Tracking Server ==="
echo ""

# Check if virtual environment exists
if [ ! -d "backend/venv" ]; then
    echo "Setting up virtual environment..."
    cd backend
    python3 -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt
    cd ..
else
    echo "Activating virtual environment..."
    source backend/venv/bin/activate
fi

# Get local IP
echo ""
echo "Detecting network interfaces..."
if [[ "$OSTYPE" == "darwin"* ]]; then
    LOCAL_IP=$(ifconfig | grep "inet " | grep -v 127.0.0.1 | head -1 | awk '{print $2}')
elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
    LOCAL_IP=$(hostname -I | awk '{print $1}')
else
    LOCAL_IP="localhost"
fi

# Check and generate SSL certificates
echo ""
echo "Checking SSL certificates..."
cd backend
if [ ! -f "certs/cert.pem" ] || [ ! -f "certs/key.pem" ]; then
    echo "Generating SSL certificates (required for WebXR)..."
    python3 generate_cert.py --ip "$LOCAL_IP"
    echo ""
fi
cd ..

echo ""
echo "=== Server Information ==="
echo "Local IP: $LOCAL_IP"
echo "HTTPS URL: https://$LOCAL_IP:8000"
echo ""
echo "On your Meta Quest 3:"
echo "1. Open browser and go to: https://$LOCAL_IP:8000"
echo "2. Accept the security warning (self-signed certificate)"
echo "3. Update WebSocket URL to: wss://$LOCAL_IP:8000/ws"
echo "4. Click 'Connect to Server'"
echo "5. Click 'Enter VR'"
echo ""
echo "Note: HTTPS is required for WebXR to work!"
echo ""
echo "Starting server..."
echo ""

# Start server
cd backend
python server.py
