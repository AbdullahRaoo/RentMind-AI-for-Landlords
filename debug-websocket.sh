#!/bin/bash

echo "🔍 Frontend WebSocket Debug Guide"

echo "📋 Step 1: Check WebSocket Test Page"
echo "🌐 Open this in your browser: http://srv889806.hstgr.cloud/websocket-test.html"
echo "   Click 'Test WebSocket Connection' and watch the logs"
echo ""

echo "📋 Step 2: Check Nginx Configuration"
echo "Active Nginx config:"
sudo nginx -T 2>/dev/null | grep -A 20 "location /ws"

echo ""
echo "📋 Step 3: Test Nginx WebSocket Proxy"
echo "Testing direct backend WebSocket:"
timeout 5s websocat ws://localhost:8000/ws/chat/ 2>/dev/null || echo "websocat not available - using curl"

echo ""
echo "Testing Nginx proxy WebSocket:"
timeout 5s websocat ws://localhost/ws/chat/ 2>/dev/null || echo "websocat not available - using curl"

echo ""
echo "📋 Step 4: Check Nginx Logs"
echo "Recent Nginx errors:"
sudo tail -20 /var/log/nginx/error.log | grep -E "(ws|websocket|backend|proxy)" || echo "No WebSocket-related errors"

echo ""
echo "📋 Step 5: Check Nginx Access Logs"
echo "Recent WebSocket requests:"
sudo tail -20 /var/log/nginx/access.log | grep -E "(ws|chat)" || echo "No WebSocket requests logged"

echo ""
echo "📋 Step 6: Test Direct Connection vs Proxy"
echo "Direct backend test:"
curl -I -H "Connection: Upgrade" -H "Upgrade: websocket" -H "Sec-WebSocket-Version: 13" -H "Sec-WebSocket-Key: dGhlIHNhbXBsZSBub25jZQ==" http://localhost:8000/ws/chat/ 2>/dev/null | head -3

echo ""
echo "Nginx proxy test:"
curl -I -H "Connection: Upgrade" -H "Upgrade: websocket" -H "Sec-WebSocket-Version: 13" -H "Sec-WebSocket-Key: dGhlIHNhbXBsZSBub25jZQ==" http://localhost/ws/chat/ 2>/dev/null | head -3

echo ""
echo "📋 Step 7: Check Frontend Build"
cd /var/www/rentmind/front
echo "Frontend WebSocket URL in built files:"
grep -r "srv889806.hstgr.cloud" out/ 2>/dev/null | head -3 || echo "No WebSocket URLs found in build"

echo ""
echo "📋 Step 8: Live Debug Commands"
echo "🔍 Watch Nginx errors live:"
echo "   sudo tail -f /var/log/nginx/error.log"
echo ""
echo "🔍 Watch backend logs live:"
echo "   sudo journalctl -u rentmind-backend -f"
echo ""
echo "🔍 Test WebSocket in browser console:"
echo "   ws = new WebSocket('ws://srv889806.hstgr.cloud/ws/chat/')"
echo "   ws.onopen = () => console.log('Connected!')"
echo "   ws.onerror = (e) => console.log('Error:', e)"
echo "   ws.onclose = (e) => console.log('Closed:', e.code, e.reason)"
