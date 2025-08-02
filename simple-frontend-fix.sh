#!/bin/bash

# Simple Frontend Fix Script - Based on your previous commands
# This uses your existing service names and setup

echo "🚀 Simple Frontend Fix - Using Your Existing Setup"
echo "================================================="

# Check if we're in the right directory
if [ ! -d "front" ]; then
    echo "❌ Not in landlord-app directory. Please run from /srv/landlord-app"
    exit 1
fi

echo "📍 Current directory: $(pwd)"

# Stop frontend service if it exists
echo "🛑 Stopping frontend service..."
sudo systemctl stop rentmind-frontend 2>/dev/null || true

# Go to frontend directory and rebuild
echo "🎨 Rebuilding frontend..."
cd front

# Clean and rebuild
echo "🧹 Cleaning old build..."
rm -rf .next node_modules package-lock.json

echo "📦 Installing dependencies..."
npm cache clean --force
npm install

echo "🏗️ Building production frontend..."
npm run build

# Go back to main directory
cd ..

echo "🔄 Restarting services..."

# Restart your existing Django service
echo "🔄 Restarting Django backend..."
sudo systemctl restart daphne-landlord.service

# Test nginx configuration and reload
echo "🌐 Testing and reloading Nginx..."
sudo nginx -t && sudo systemctl reload nginx

# Start frontend service
echo "🎨 Starting frontend service..."
if [ -f "/etc/systemd/system/rentmind-frontend.service" ]; then
    sudo systemctl start rentmind-frontend
else
    echo "⚙️ Creating frontend service..."
    # Create frontend service if it doesn't exist
    sudo tee /etc/systemd/system/rentmind-frontend.service > /dev/null << EOF
[Unit]
Description=RentMind AI Frontend
After=network.target
Wants=daphne-landlord.service

[Service]
Type=simple
User=$USER
WorkingDirectory=$(pwd)/front
Environment=NODE_ENV=production
Environment=PORT=3000
ExecStart=/usr/bin/npm start
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF
    
    sudo systemctl daemon-reload
    sudo systemctl enable rentmind-frontend
    sudo systemctl start rentmind-frontend
fi

# Wait for services to start
sleep 5

echo ""
echo "🔍 CHECKING SERVICES"
echo "==================="

# Check service status
echo "🔍 Service Status:"
sudo systemctl is-active daphne-landlord.service && echo "✅ Django backend: Running" || echo "❌ Django backend: Not running"
sudo systemctl is-active rentmind-frontend.service && echo "✅ Frontend: Running" || echo "❌ Frontend: Not running"
sudo systemctl is-active nginx && echo "✅ Nginx: Running" || echo "❌ Nginx: Not running"

# Check ports
echo ""
echo "🔍 Port Status:"
netstat -tulpn | grep ":3000 " > /dev/null && echo "✅ Port 3000: Open" || echo "❌ Port 3000: Closed"
netstat -tulpn | grep ":8000 " > /dev/null && echo "✅ Port 8000: Open" || echo "❌ Port 8000: Closed"
netstat -tulpn | grep ":80 " > /dev/null && echo "✅ Port 80: Open" || echo "❌ Port 80: Closed"

# Test endpoints
echo ""
echo "🔍 Testing Endpoints:"
if curl -s http://localhost:3000 | head -n 1 | grep -q "<!DOCTYPE html"; then
    echo "✅ Frontend: Serving HTML"
else
    echo "❌ Frontend: Not serving properly"
fi

if curl -s http://localhost:8000 > /dev/null; then
    echo "✅ Backend: Responding"
else
    echo "❌ Backend: Not responding"
fi

# Show recent logs if there are issues
echo ""
echo "📋 Recent Logs (if issues found):"
echo "Frontend logs:"
sudo journalctl -u rentmind-frontend -n 5 --no-pager 2>/dev/null || echo "No frontend logs"

echo ""
echo "Backend logs:"
sudo journalctl -u daphne-landlord -n 5 --no-pager 2>/dev/null || echo "No backend logs"

echo ""
echo "✅ FRONTEND FIX COMPLETED"
echo "========================"
echo "🌐 Your application should be accessible at:"
echo "   Main site: http://srv889806.hstgr.cloud"
echo "   Direct frontend: http://srv889806.hstgr.cloud:3000"
echo "   Backend API: http://srv889806.hstgr.cloud:8000"
echo ""
echo "🔧 If issues persist:"
echo "   sudo journalctl -u daphne-landlord -f"
echo "   sudo journalctl -u rentmind-frontend -f"
echo "   sudo systemctl status daphne-landlord rentmind-frontend nginx"
