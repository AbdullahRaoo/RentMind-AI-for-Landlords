#!/bin/bash

# Easy deployment script for Ubuntu VPS
# Optimized for budget/free hosting - no PM2, Nginx, or systemd needed!

set -e

echo "🚀 RentMind AI - Budget-Friendly Ubuntu VPS Deployment"
echo "======================================================"

# Check system resources
echo "💻 Checking system resources..."
MEMORY=$(free -m | awk 'NR==2{printf "%.1f GB", $2/1024}')
DISK=$(df -h / | awk 'NR==2{print $4}')
echo "   Available Memory: $MEMORY"
echo "   Available Disk: $DISK"

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "📦 Installing Docker (lightweight installation)..."
    curl -fsSL https://get.docker.com -o get-docker.sh
    sudo sh get-docker.sh
    sudo usermod -aG docker $USER
    echo "✅ Docker installed! Please logout and login again, then re-run this script."
    exit 1
fi

# Check if Docker Compose is installed
if ! command -v docker-compose &> /dev/null; then
    echo "📦 Installing Docker Compose..."
    sudo curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
    sudo chmod +x /usr/local/bin/docker-compose
fi

# Prompt for domain name
read -p "🌐 Enter your domain name (e.g., yourdomain.com): " DOMAIN
read -p "📧 Enter your email for SSL certificates: " EMAIL
read -p "🔑 Enter your OpenAI API key: " OPENAI_KEY

# Update docker-compose.yml with user's domain
echo "⚙️  Configuring domain and SSL..."
sed -i "s/your-domain.com/$DOMAIN/g" docker-compose.yml
sed -i "s/your-email@example.com/$EMAIL/g" docker-compose.yml

# Create environment file
echo "📝 Creating environment configuration..."
cat > .env << EOF
DOMAIN=$DOMAIN
EMAIL=$EMAIL
OPENAI_API_KEY=$OPENAI_KEY
DEBUG=False
DJANGO_SETTINGS_MODULE=backend.production_settings
EOF

# Create letsencrypt directory
mkdir -p letsencrypt

# Optimize for low-resource VPS
echo "🔧 Optimizing for budget VPS..."
export COMPOSE_HTTP_TIMEOUT=200
export DOCKER_CLIENT_TIMEOUT=200

# Build and start services with resource optimization
echo "🔨 Building and starting services (this may take a few minutes)..."
docker-compose build --parallel
docker-compose up -d

# Wait for services to start
echo "⏳ Waiting for services to initialize..."
sleep 30

# Check if services are running
if docker-compose ps | grep -q "Up"; then
    echo ""
    echo "✅ Deployment Complete!"
    echo "======================================"
    echo "🌐 Your app will be available at: https://$DOMAIN"
    echo "🔧 Traefik dashboard: http://$DOMAIN:8080"
    echo ""
    echo "📋 Management Commands:"
    echo "  View logs:     docker-compose logs -f"
    echo "  Stop services: docker-compose down"
    echo "  Update app:    ./update.sh"
    echo "  Restart:       docker-compose restart"
    echo ""
    echo "⚠️  Important Notes:"
    echo "1. Point your domain DNS to this server's IP: $(curl -s https://ipinfo.io/ip)"
    echo "2. Wait 2-3 minutes for SSL certificates to generate"
    echo "3. If using a budget VPS, monitor resource usage with: docker stats"
    echo ""
    echo "💰 Cost: Only your VPS hosting fee (~$3-5/month)"
    echo "🎉 Features: Full AI app with auto-SSL, load balancing, and scaling!"
else
    echo "❌ Some services failed to start. Check logs:"
    docker-compose logs
fi
