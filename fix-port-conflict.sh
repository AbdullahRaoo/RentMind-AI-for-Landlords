#!/bin/bash

echo "🔍 Port 8000 Conflict Resolution"

echo "📋 Step 1: Check what's using port 8000"
echo "Processes on port 8000:"
sudo netstat -tulpn | grep :8000

echo ""
echo "More detailed process info:"
sudo lsof -i :8000

echo ""
echo "📋 Step 2: Check for running Python/Daphne processes"
echo "All Python processes:"
ps aux | grep python | grep -v grep

echo ""
echo "All Daphne processes:"
ps aux | grep daphne | grep -v grep

echo ""
echo "📋 Step 3: Stop conflicting processes"
echo "Stopping systemd service first..."
sudo systemctl stop rentmind-backend

echo ""
echo "Killing any remaining processes on port 8000..."
# Kill processes using port 8000
sudo fuser -k 8000/tcp 2>/dev/null || echo "No processes to kill on port 8000"

echo ""
echo "Killing any stray daphne processes..."
sudo pkill -f daphne 2>/dev/null || echo "No daphne processes to kill"

echo ""
echo "Killing any stray python processes running backend..."
sudo pkill -f "backend.asgi" 2>/dev/null || echo "No backend.asgi processes to kill"

echo ""
echo "📋 Step 4: Wait and verify port is free"
sleep 3
echo "Checking if port 8000 is now free:"
sudo netstat -tulpn | grep :8000 || echo "✅ Port 8000 is now free"

echo ""
echo "📋 Step 5: Start the service cleanly"
echo "Starting rentmind-backend service..."
sudo systemctl start rentmind-backend

echo ""
echo "📋 Step 6: Monitor startup"
echo "Waiting for service to start..."
sleep 10

echo ""
echo "Service status:"
sudo systemctl status rentmind-backend --no-pager | head -10

echo ""
echo "📋 Step 7: Test connection"
for i in {1..5}; do
    STATUS=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:8000/)
    if [ "$STATUS" = "200" ] || [ "$STATUS" = "404" ]; then
        echo "✅ Backend responding (attempt $i): HTTP $STATUS"
        break
    else
        echo "⏳ Attempt $i: HTTP $STATUS - waiting..."
        sleep 2
    fi
done

echo ""
echo "📋 Step 8: Final verification"
echo "Processes now on port 8000:"
sudo netstat -tulpn | grep :8000

echo ""
echo "Recent logs:"
sudo journalctl -u rentmind-backend -n 5 --no-pager

echo ""
echo "✅ Port conflict resolution complete!"
