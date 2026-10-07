#!/bin/bash
# Azure App Service startup command for the backend bundle.
#
# Starts Tamil and Sinhala backends in the background exactly as they
# already run on the local machine (unchanged, same ports, same code),
# then runs Gateway in the foreground as the main process Azure watches
# and routes public traffic to. Gateway keeps talking to Tamil/Sinhala
# over 127.0.0.1 because all three are in this same container.
set -e

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

echo "Starting Tamil backend on port 5000..."
cd "$REPO_ROOT/tamil-hate-speech-detection inte 23rd/tamil-hate-speech-detection"
python api/app.py > /tmp/tamil_backend.log 2>&1 &

echo "Starting Sinhala backend on port 5001..."
cd "$REPO_ROOT/AT final-4/AT final/backend"
python app.py > /tmp/sinhala_backend.log 2>&1 &

# Give both a few seconds to finish loading their models before Gateway
# starts accepting public traffic.
sleep 10

echo "Starting Gateway on port 8000 (main process)..."
cd "$REPO_ROOT/gateway"
exec python app.py
