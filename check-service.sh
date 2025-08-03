#!/bin/bash

echo "📊 Backend Service Status Check"

echo "📋 Service Status:"
sudo systemctl status rentmind-backend --no-pager | head -10

echo ""
echo "📋 Service Restart Count:"
sudo systemctl show rentmind-backend -p NRestarts

echo ""
echo "📋 Recent Logs (last 10 lines):"
sudo journalctl -u rentmind-backend -n 10 --no-pager

echo ""
echo "📋 Connection Test:"
STATUS=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:8000/)
echo "HTTP Status: $STATUS"

if [ "$STATUS" = "200" ] || [ "$STATUS" = "404" ]; then
    echo "✅ Backend is responding"
else
    echo "❌ Backend not responding"
    echo ""
    echo "📋 Checking what's running on port 8000:"
    sudo netstat -tulpn | grep :8000 || echo "Nothing on port 8000"
    
    echo ""
    echo "📋 Checking virtual environment:"
    if [ -f "/var/www/rentmind/airlords/bin/python" ]; then
        echo "✅ Virtual environment Python exists"
        /var/www/rentmind/airlords/bin/python --version
    else
        echo "❌ Virtual environment Python missing"
    fi
    
    if [ -f "/var/www/rentmind/airlords/bin/daphne" ]; then
        echo "✅ Daphne executable exists"
    else
        echo "❌ Daphne executable missing"
    fi
fi
