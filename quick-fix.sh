#!/bin/bash

# Quick Frontend Fix - Using Your Exact Commands
# Based on your previous deployment: daphne-landlord.service + landlord-app nginx config

echo "🚀 Quick Frontend Fix - Using Your Existing Setup"
echo "================================================="

# Check if we're in the right directory
if [ ! -d "front" ]; then
    echo "❌ Not in landlord-app directory. Please run from /srv/landlord-app"
    exit 1
fi

echo "📍 Current directory: $(pwd)"

echo "🎨 Rebuilding frontend..."
cd front

# Clean and rebuild (your exact process)
echo "🧹 Cleaning old build..."
rm -rf .next node_modules package-lock.json

echo "📦 Installing dependencies..."
npm cache clean --force
npm install

echo "🏗️ Building production frontend..."
NODE_ENV=production npm run build

# Verify build was successful
if [ ! -d ".next" ]; then
    echo "❌ Build failed - .next directory not created"
    echo "🔧 Trying alternative build process..."
    npm run build 2>&1 | tail -10
    exit 1
else
    echo "✅ Build successful - .next directory created"
fi

cd ..

echo "🔄 Restarting services with your exact commands..."

# Your exact commands:
echo "🔄 Restarting Django backend..."
sudo systemctl restart daphne-landlord.service

echo "🌐 Testing and reloading Nginx..."
sudo nginx -t && sudo systemctl reload nginx

# Wait for services
sleep 3

echo ""
echo "🔍 CHECKING STATUS"
echo "=================="

# Check services
echo "🔍 Service Status:"
if sudo systemctl is-active --quiet daphne-landlord.service; then
    echo "✅ Django backend (daphne-landlord): Running"
else
    echo "❌ Django backend (daphne-landlord): Not running"
    echo "🔧 Checking logs:"
    sudo journalctl -u daphne-landlord -n 5 --no-pager
fi

if sudo systemctl is-active --quiet nginx; then
    echo "✅ Nginx: Running"
else
    echo "❌ Nginx: Not running"
fi

# Check ports
echo ""
echo "🔍 Port Status:"
if netstat -tulpn | grep ":8000 " > /dev/null; then
    echo "✅ Port 8000 (Django): Open"
else
    echo "❌ Port 8000 (Django): Closed"
fi

if netstat -tulpn | grep ":80 " > /dev/null; then
    echo "✅ Port 80 (Nginx): Open"
else
    echo "❌ Port 80 (Nginx): Closed"
fi

# Test endpoints
echo ""
echo "🔍 Testing Application:"
if curl -s http://localhost:8000 > /dev/null 2>&1; then
    echo "✅ Backend: Responding"
else
    echo "❌ Backend: Not responding"
fi

if curl -s http://localhost > /dev/null 2>&1; then
    echo "✅ Nginx proxy: Working"
else
    echo "❌ Nginx proxy: Not working"
fi

echo ""
echo "✅ FRONTEND FIX COMPLETED"
echo "========================"
echo "🌐 Your application should be accessible at:"
echo "   Main site: http://srv889806.hstgr.cloud"
echo "   Backend API: http://srv889806.hstgr.cloud:8000"
echo ""
echo "🔧 Your Nginx config: /etc/nginx/sites-available/landlord-app"
echo "🔧 Your Django service: daphne-landlord.service"
echo ""
echo "🔧 If issues persist, run:"
echo "   sudo journalctl -u daphne-landlord -f"
echo "   sudo systemctl status daphne-landlord nginx"
echo "   sudo nano /etc/nginx/sites-available/landlord-app"
