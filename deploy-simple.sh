#!/bin/bash

# Simple Ubuntu deployment without Docker
# Uses Caddy for automatic HTTPS - no Nginx configuration needed!

set -e

echo "🚀 RentMind AI - Simple Ubuntu Deployment"
echo "=========================================="

# Update system
sudo apt update && sudo apt upgrade -y

# Install Python, Node.js, and dependencies
echo "📦 Installing system dependencies..."
sudo apt install -y python3 python3-pip python3-venv nodejs npm git redis-server

# Install Caddy (automatic HTTPS server)
sudo apt install -y debian-keyring debian-archive-keyring apt-transport-https
curl -1sLf 'https://dl.cloudsmith.io/public/caddy/stable/gpg.key' | sudo gpg --dearmor -o /usr/share/keyrings/caddy-stable-archive-keyring.gpg
curl -1sLf 'https://dl.cloudsmith.io/public/caddy/stable/debian.deb.txt' | sudo tee /etc/apt/sources.list.d/caddy-stable.list
sudo apt update && sudo apt install caddy

# Install pnpm
sudo npm install -g pnpm

# Get domain and email
read -p "🌐 Enter your domain name: " DOMAIN
read -p "📧 Enter your email: " EMAIL

# Create project directory
PROJECT_DIR="/opt/rentmind"
sudo mkdir -p $PROJECT_DIR
sudo chown $USER:$USER $PROJECT_DIR

# Clone or copy project (assuming you'll upload it)
echo "📁 Please ensure your project files are in $PROJECT_DIR"

# Setup Python environment
cd $PROJECT_DIR
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Setup environment variables
cat > .env << EOF
DEBUG=False
ALLOWED_HOSTS=$DOMAIN,www.$DOMAIN
SECRET_KEY=$(python3 -c 'from django.core.management.utils import get_random_secret_key; print(get_random_secret_key())')
REDIS_URL=redis://localhost:6379/0
OPENAI_API_KEY=your_openai_key_here
EOF

# Build frontend
cd front
pnpm install
pnpm build
cd ..

# Configure Caddyfile
sudo tee /etc/caddy/Caddyfile > /dev/null << EOF
$DOMAIN {
    # Frontend (static files)
    root * /opt/rentmind/front/out
    file_server

    # API and WebSocket routes to backend
    reverse_proxy /ws/* localhost:8000
    reverse_proxy /api/* localhost:8000
    reverse_proxy /admin/* localhost:8000

    # Enable compression
    encode gzip

    # Security headers
    header {
        Strict-Transport-Security "max-age=31536000; includeSubDomains; preload"
        X-Content-Type-Options "nosniff"
        X-Frame-Options "DENY"
        Referrer-Policy "strict-origin-when-cross-origin"
    }
}
EOF

# Create systemd services
sudo tee /etc/systemd/system/rentmind-backend.service > /dev/null << EOF
[Unit]
Description=RentMind Backend
After=network.target redis.service

[Service]
Type=simple
User=$USER
WorkingDirectory=$PROJECT_DIR/backend
Environment=PATH=$PROJECT_DIR/venv/bin
EnvironmentFile=$PROJECT_DIR/.env
ExecStart=$PROJECT_DIR/venv/bin/daphne -b 0.0.0.0 -p 8000 backend.asgi:application
Restart=always
RestartSec=3

[Install]
WantedBy=multi-user.target
EOF

# Start services
sudo systemctl daemon-reload
sudo systemctl enable --now redis-server
sudo systemctl enable --now caddy
sudo systemctl enable --now rentmind-backend

echo ""
echo "✅ Deployment Complete!"
echo "======================="
echo "🌐 Your app is available at: https://$DOMAIN"
echo "🔧 Status: sudo systemctl status rentmind-backend caddy"
echo "📋 Logs: sudo journalctl -u rentmind-backend -f"
echo ""
echo "⚠️  Don't forget to:"
echo "1. Point your domain DNS to this server's IP"
echo "2. Update .env file with your OpenAI API key"
echo "3. Run: sudo systemctl restart rentmind-backend"
EOF
