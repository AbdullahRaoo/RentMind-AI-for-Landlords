#!/bin/bash

# RentMind AI Frontend Fix Script
# This script diagnoses and fixes Next.js frontend issues

echo "🔧 RentMind AI Frontend Troubleshooting & Fix Script"
echo "=================================================="

# Check current directory
echo "📍 Current directory: $(pwd)"
if [ ! -f "package.json" ] && [ ! -d "front" ]; then
    echo "❌ Not in the right directory. Please run from /srv/landlord-app"
    exit 1
fi

# Function to check service status
check_service() {
    local service=$1
    echo "🔍 Checking $service status..."
    if systemctl is-active --quiet $service; then
        echo "✅ $service is running"
        return 0
    else
        echo "❌ $service is not running"
        return 1
    fi
}

# Function to check port
check_port() {
    local port=$1
    local service=$2
    echo "🔍 Checking if port $port is open for $service..."
    if netstat -tulpn | grep ":$port " > /dev/null; then
        echo "✅ Port $port is open"
        return 0
    else
        echo "❌ Port $port is not open"
        return 1
    fi
}

# Check system services
echo ""
echo "🔍 SYSTEM DIAGNOSTICS"
echo "====================="

check_service "nginx"
# Check for existing services (could be either name)
if systemctl list-units --full -all | grep -q "daphne-landlord.service"; then
    check_service "daphne-landlord"
elif systemctl list-units --full -all | grep -q "rentmind.service"; then
    check_service "rentmind"
else
    echo "❌ No Django service found (looking for daphne-landlord or rentmind)"
fi

if systemctl list-units --full -all | grep -q "rentmind-frontend.service"; then
    check_service "rentmind-frontend"
else
    echo "❌ No frontend service found"
fi

check_port "80" "nginx"
check_port "3000" "frontend"
check_port "8000" "backend"

# Check Docker containers
echo ""
echo "🐳 DOCKER DIAGNOSTICS"
echo "====================="
if command -v docker &> /dev/null; then
    echo "🔍 Checking Docker containers..."
    docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"
    
    if ! docker ps | grep -q milvus; then
        echo "❌ Milvus containers not running"
    else
        echo "✅ Milvus containers are running"
    fi
else
    echo "❌ Docker not installed"
fi

# Check frontend directory and build
echo ""
echo "🎨 FRONTEND DIAGNOSTICS"
echo "======================"
cd front 2>/dev/null || { echo "❌ Frontend directory not found"; exit 1; }

echo "🔍 Checking frontend files..."
if [ ! -f "package.json" ]; then
    echo "❌ package.json not found"
    exit 1
fi

if [ ! -d "node_modules" ]; then
    echo "❌ node_modules not found - need to install dependencies"
else
    echo "✅ node_modules found"
fi

if [ ! -d ".next" ]; then
    echo "❌ .next build directory not found - need to build"
else
    echo "✅ .next build directory found"
fi

# Fix the issues
echo ""
echo "🔧 FIXING ISSUES"
echo "================"

# Stop services first
echo "🛑 Stopping services..."
sudo systemctl stop rentmind-frontend 2>/dev/null || true

# Clean and rebuild frontend
echo "🧹 Cleaning frontend..."
rm -rf node_modules .next package-lock.json

echo "📦 Installing frontend dependencies..."
npm cache clean --force
npm install

echo "🏗️ Building frontend..."
NODE_ENV=production npm run build

# Go back to main directory
cd ..

# Check if backend is working
echo "🔍 Testing backend connectivity..."
if curl -f http://localhost:8000/api/ 2>/dev/null; then
    echo "✅ Backend is responding"
else
    echo "❌ Backend not responding - restarting..."
    # Try to restart the correct service
    if systemctl list-units --full -all | grep -q "daphne-landlord.service"; then
        sudo systemctl restart daphne-landlord
    elif systemctl list-units --full -all | grep -q "rentmind.service"; then
        sudo systemctl restart rentmind
    else
        echo "❌ No Django service found to restart"
    fi
    sleep 5
fi

# Update systemd service with correct paths
echo "⚙️ Updating frontend service configuration..."
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

# Reload and restart services
echo "🔄 Reloading and starting services..."
sudo systemctl daemon-reload
sudo systemctl enable rentmind-frontend
sudo systemctl start rentmind-frontend

# Wait a moment for services to start
sleep 10

# Test services
echo ""
echo "🧪 TESTING SERVICES"
echo "=================="

echo "🔍 Service status after restart:"
# Check the correct services
if systemctl list-units --full -all | grep -q "daphne-landlord.service"; then
    check_service "daphne-landlord"
elif systemctl list-units --full -all | grep -q "rentmind.service"; then
    check_service "rentmind"
fi
check_service "rentmind-frontend"

echo "🔍 Port status after restart:"
check_port "3000" "frontend"
check_port "8000" "backend"

# Test endpoints
echo "🔍 Testing endpoints..."
if curl -f http://localhost:3000 2>/dev/null | grep -q "html\|HTML"; then
    echo "✅ Frontend is serving content"
else
    echo "❌ Frontend not serving content properly"
fi

if curl -f http://localhost:8000 2>/dev/null; then
    echo "✅ Backend is responding"
else
    echo "❌ Backend not responding"
fi

# Check logs for errors
echo ""
echo "📋 RECENT LOGS"
echo "=============="
echo "🔍 Frontend service logs (last 20 lines):"
sudo journalctl -u rentmind-frontend -n 20 --no-pager

echo ""
echo "🔍 Backend service logs (last 10 lines):"
if systemctl list-units --full -all | grep -q "daphne-landlord.service"; then
    sudo journalctl -u daphne-landlord -n 10 --no-pager
elif systemctl list-units --full -all | grep -q "rentmind.service"; then
    sudo journalctl -u rentmind -n 10 --no-pager
else
    echo "No Django service logs found"
fi

echo ""
echo "🔍 Nginx error logs (last 10 lines):"
sudo tail -n 10 /var/log/nginx/error.log 2>/dev/null || echo "No nginx error logs found"

# Final status
echo ""
echo "✅ FRONTEND FIX COMPLETED"
echo "========================"
echo "🌐 Your application should now be accessible at:"
echo "   Frontend: http://$(curl -s ifconfig.me)"
echo "   Direct Frontend: http://$(curl -s ifconfig.me):3000"
echo "   Backend API: http://$(curl -s ifconfig.me):8000"
echo ""
echo "🔧 If issues persist, check:"
echo "   sudo journalctl -u rentmind-frontend -f"
echo "   sudo journalctl -u rentmind -f"
echo "   sudo systemctl status rentmind rentmind-frontend"
