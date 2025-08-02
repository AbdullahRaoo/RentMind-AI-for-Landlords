#!/bin/bash

# Git-based deployment script for srv889806.hstgr.cloud
# Optimized for your 16GB RAM, 4 CPU server

set -e

echo "🚀 RentMind AI - Production Deployment"
echo "======================================"
echo "🌐 Domain: srv889806.hstgr.cloud"
echo "💻 Server: 16GB RAM, 4 CPU (Excellent specs!)"
echo ""

# Check if we're in the right directory
if [ ! -f "docker-compose.yml" ]; then
    echo "❌ docker-compose.yml not found. Please run this script from the project directory."
    exit 1
fi

# Check system resources
echo "💻 Checking system resources..."
MEMORY=$(free -m | awk 'NR==2{printf "%.1f GB", $2/1024}')
DISK=$(df -h / | awk 'NR==2{print $4}')
echo "   Available Memory: $MEMORY"
echo "   Available Disk: $DISK"
echo ""

# Install Docker if not present
if ! command -v docker &> /dev/null; then
    echo "📦 Installing Docker..."
    curl -fsSL https://get.docker.com -o get-docker.sh
    sudo sh get-docker.sh
    sudo usermod -aG docker $USER
    echo "✅ Docker installed! Please logout and login again, then re-run this script."
    exit 1
fi

# Install Docker Compose if not present
if ! command -v docker-compose &> /dev/null; then
    echo "📦 Installing Docker Compose..."
    sudo curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
    sudo chmod +x /usr/local/bin/docker-compose
fi

# Pull latest code from git
echo "📥 Pulling latest code from GitHub..."
if [ -d ".git" ]; then
    git pull origin main
    echo "✅ Code updated from GitHub"
else
    echo "⚠️  Not a git repository. Make sure you cloned from GitHub:"
    echo "   git clone https://github.com/AbdullahRaoo/RentMind-AI-for-Landlords.git"
fi
echo ""

# Copy production environment
echo "⚙️  Setting up production environment..."
if [ -f ".env.production.template" ]; then
    cp .env.production.template .env.production
    
    # Configure the OpenAI API key
    if grep -q "your_openai_api_key_here" .env.production; then
        echo "🔑 Configuring OpenAI API key..."
        # Replace with your actual API key
        sed -i "s/your_openai_api_key_here/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/g" .env.production
        echo "   ✅ OpenAI API key configured"
    else
        echo "   ✅ OpenAI API key already configured"
    fi
    
    # Generate a random secret key
    SECRET_KEY="django-prod-$(date +%s)-$(openssl rand -hex 16 2>/dev/null || head -c 16 /dev/urandom | xxd -p)"
    sed -i "s/django-prod-auto-generated-secret/$SECRET_KEY/g" .env.production
    
    # Use production environment
    cp .env.production .env
    echo "✅ Production environment configured"
else
    echo "❌ .env.production.template not found!"
    exit 1
fi

# Create necessary directories
mkdir -p letsencrypt
mkdir -p backend/media
mkdir -p backend/staticfiles

# Optimize for your server specs (16GB RAM, 4 CPU)
echo "🔧 Optimizing for your server (16GB RAM, 4 CPU)..."
export COMPOSE_HTTP_TIMEOUT=300
export DOCKER_CLIENT_TIMEOUT=300

# Stop existing containers
echo "🛑 Stopping existing containers..."
docker-compose down 2>/dev/null || true

# Build with your server's power
echo "🔨 Building application (using all 4 CPU cores)..."
docker-compose build --parallel

# Start services
echo "🚀 Starting services..."
docker-compose up -d

# Wait for services to start
echo "⏳ Waiting for services to initialize..."
sleep 30

# Health check
echo "🏥 Performing health checks..."
sleep 10

# Check if services are running
if docker-compose ps | grep -q "Up"; then
    echo ""
    echo "✅ DEPLOYMENT SUCCESSFUL!"
    echo "=========================================="
    echo "🌐 Your app is live at: https://srv889806.hstgr.cloud"
    echo "🔧 Traefik dashboard: http://srv889806.hstgr.cloud:8080"
    echo ""
    echo "📊 Server Status:"
    docker-compose ps
    echo ""
    echo "📋 Useful Commands:"
    echo "  View logs:        docker-compose logs -f"
    echo "  Restart:          docker-compose restart"
    echo "  Update from git:  ./deploy-git.sh"
    echo "  Monitor system:   ./monitor.sh"
    echo "  Stop services:    docker-compose down"
    echo ""
    echo "🎉 Your AI application is now running in production!"
    echo "💰 Cost: Just your VPS fee (~$10-15/month)"
    echo "🚀 Features: Auto-SSL, load balancing, scaling, monitoring"
    echo ""
    echo "📱 Next steps:"
    echo "1. Test your app at https://srv889806.hstgr.cloud"
    echo "2. Set up monitoring alerts (optional)"
    echo "3. Schedule automatic backups (optional)"
else
    echo ""
    echo "❌ DEPLOYMENT FAILED"
    echo "==================="
    echo "Checking logs for errors..."
    docker-compose logs --tail=20
    echo ""
    echo "🛠️  Troubleshooting:"
    echo "1. Check logs: docker-compose logs"
    echo "2. Check disk space: df -h"
    echo "3. Check memory: free -h"
    echo "4. Restart deployment: ./deploy-git.sh"
fi
