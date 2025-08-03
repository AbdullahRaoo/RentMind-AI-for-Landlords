#!/bin/bash

echo "🚀 Complete RentMind Deployment"
echo "================================"

cd /var/www/rentmind

echo "📋 Step 1: Deploy Backend..."
./deploy-backend.sh

echo ""
echo "📋 Step 2: Setup Complete Nginx..."
./setup-nginx-complete.sh

echo ""
echo "📋 Step 3: Testing complete setup..."

# Test backend
echo "🧪 Testing backend..."
if curl -s -o /dev/null -w "%{http_code}" http://localhost:8000/ | grep -q "200\|404"; then
    echo "✅ Backend is responding"
else
    echo "❌ Backend not responding"
fi

# Test frontend
echo "🧪 Testing frontend..."
if curl -s -o /dev/null -w "%{http_code}" http://localhost/ | grep -q "200"; then
    echo "✅ Frontend is responding"
else
    echo "❌ Frontend not responding"
fi

echo ""
echo "🎉 Deployment Complete!"
echo "========================"
echo ""
echo "🌐 Your application is now live at:"
echo "   - Main site: http://srv889806.hstgr.cloud"
echo "   - Backend API: http://srv889806.hstgr.cloud/api/"
echo "   - WebSocket: ws://srv889806.hstgr.cloud/ws/chat/"
echo ""
echo "📋 Service management:"
echo "   - Backend status: sudo systemctl status rentmind-backend"
echo "   - Backend logs: sudo journalctl -u rentmind-backend -f"
echo "   - Nginx status: sudo systemctl status nginx"
echo "   - Nginx logs: sudo tail -f /var/log/nginx/error.log"
