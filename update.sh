#!/bin/bash

# Easy update script for RentMind AI (Git-based)

echo "🔄 Updating RentMind AI from GitHub"
echo "==================================="

# Check if this is a git repository
if [ ! -d ".git" ]; then
    echo "❌ Not a git repository. Please clone from:"
    echo "   git clone https://github.com/AbdullahRaoo/RentMind-AI-for-Landlords.git"
    exit 1
fi

# Pull latest changes from GitHub
echo "📥 Pulling latest changes from GitHub..."
git pull origin main

if [ $? -eq 0 ]; then
    echo "✅ Successfully pulled latest changes"
else
    echo "❌ Failed to pull changes from GitHub"
    exit 1
fi

# Check which deployment method is being used
if [ -f "docker-compose.yml" ] && command -v docker-compose &> /dev/null; then
    echo "📦 Detected Docker deployment"
    echo "Rebuilding and restarting services..."
    
    # Rebuild and restart services
    docker-compose down
    docker-compose build --parallel
    docker-compose up -d
    
    echo "✅ Docker deployment updated!"
    echo "📋 View logs: docker-compose logs -f"
    echo "🌐 App URL: https://srv889806.hstgr.cloud"

elif [ -f "/etc/systemd/system/rentmind-backend.service" ]; then
    echo "🖥️  Detected simple Ubuntu deployment"
    
    # Update backend
    source venv/bin/activate
    pip install -r requirements-production.txt
    
    # Update frontend
    cd front
    pnpm install
    pnpm build
    cd ..
    
    # Restart services
    sudo systemctl restart rentmind-backend
    sudo systemctl reload caddy
    
    echo "✅ Simple deployment updated!"
    echo "📋 View logs: sudo journalctl -u rentmind-backend -f"

else
    echo "❌ No recognized deployment found"
    echo "Please use the deploy-git.sh script first"
    exit 1
fi

echo ""
echo "🎉 Update complete! Your application is running with the latest changes from GitHub."
echo "📊 Check status: docker-compose ps"
echo "🔍 Monitor resources: ./monitor.sh"
