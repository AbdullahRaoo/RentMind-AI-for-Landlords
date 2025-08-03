#!/bin/bash

echo "🧪 Quick Backend Test"

echo "📋 Testing if backend is running..."
BACKEND_STATUS=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:8000/)

if [ "$BACKEND_STATUS" = "200" ] || [ "$BACKEND_STATUS" = "404" ]; then
    echo "✅ Backend is responding (HTTP $BACKEND_STATUS)"
    
    echo ""
    echo "📋 Testing WebSocket endpoint..."
    WS_TEST=$(curl -s -I -H "Connection: Upgrade" -H "Upgrade: websocket" http://localhost:8000/ws/chat/ | head -1)
    echo "WebSocket response: $WS_TEST"
    
    echo ""
    echo "📋 Testing from external..."
    EXT_STATUS=$(curl -s -o /dev/null -w "%{http_code}" http://srv889806.hstgr.cloud:8000/)
    echo "External access (port 8000): HTTP $EXT_STATUS"
    
    PROXY_STATUS=$(curl -s -o /dev/null -w "%{http_code}" http://srv889806.hstgr.cloud/api/)
    echo "Nginx proxy to backend: HTTP $PROXY_STATUS"
    
else
    echo "❌ Backend not responding (HTTP $BACKEND_STATUS)"
    echo ""
    echo "📋 Checking service status..."
    sudo systemctl status rentmind-backend --no-pager | head -10
    
    echo ""
    echo "📋 Recent logs..."
    sudo journalctl -u rentmind-backend -n 20 --no-pager | tail -10
    
    echo ""
    echo "📋 Manual backend test..."
    cd /var/www/rentmind/backend
    source ../airlords/bin/activate
    timeout 10s python manage.py runserver 127.0.0.1:8002 --settings=backend.production_settings &
    sleep 3
    
    TEST_STATUS=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:8002/)
    if [ "$TEST_STATUS" = "200" ] || [ "$TEST_STATUS" = "404" ]; then
        echo "✅ Backend works manually (HTTP $TEST_STATUS) - service configuration issue"
    else
        echo "❌ Backend fails even manually - dependency/code issue"
    fi
    
    pkill -f "runserver"
fi
