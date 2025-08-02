#!/bin/bash

# Production Deployment Script for RentMind AI
# Run this script on your Ubuntu server

set -e  # Exit on any error

echo "🚀 Starting RentMind AI Deployment..."

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
APP_DIR="/srv/landlord-app"
DOMAIN="${1:-localhost}"  # Use first argument or default to localhost
USER="www-data"

print_status "Using domain/IP: $DOMAIN"

print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if running as root
if [[ $EUID -eq 0 ]]; then
   print_error "This script should not be run as root"
   exit 1
fi

# Check if directory exists
if [ ! -d "$APP_DIR" ]; then
    print_error "Application directory $APP_DIR does not exist"
    exit 1
fi

cd $APP_DIR

print_status "Updating system packages..."
sudo apt update

print_status "Installing required packages..."
sudo apt install -y python3 python3-pip python3-venv nodejs npm nginx git curl redis-server

# Install Node.js 18+
print_status "Installing Node.js 18+..."
if ! command -v node &> /dev/null || [ $(node -v | cut -d'v' -f2 | cut -d'.' -f1) -lt 18 ]; then
    curl -fsSL https://deb.nodesource.com/setup_18.x | sudo -E bash -
    sudo apt-get install -y nodejs
fi

# Install PM2
print_status "Installing PM2..."
sudo npm install -g pm2

print_status "Setting up backend..."
cd backend

# Create virtual environment
if [ ! -d "venv" ]; then
    python3 -m venv venv
fi

source venv/bin/activate

# Install Python dependencies
print_status "Installing Python dependencies..."
pip install --upgrade pip
pip install -r ../requirements.txt
pip install python-dotenv dj-database-url django-redis channels-redis gunicorn

# Create .env file
print_status "Creating environment file..."
cat > .env << EOF
DEBUG=False
SECRET_KEY=$(python -c 'from django.core.management.utils import get_random_secret_key; print(get_random_secret_key())')
ALLOWED_HOSTS=$DOMAIN,www.$DOMAIN,localhost,127.0.0.1
DATABASE_URL=sqlite:///db.sqlite3
EOF

# Database setup
print_status "Setting up database..."
python manage.py makemigrations
python manage.py migrate
python manage.py collectstatic --noinput

print_status "Setting up frontend..."
cd ../front

# Install frontend dependencies
npm install

# Build frontend
npm run build

print_status "Setting up services..."

# Create systemd service for Daphne
sudo tee /etc/systemd/system/daphne-landlord.service > /dev/null << EOF
[Unit]
Description=Daphne ASGI Server for Landlord AI
After=network.target

[Service]
Type=simple
User=$USER
WorkingDirectory=$APP_DIR/backend
Environment=DJANGO_SETTINGS_MODULE=backend.production_settings
ExecStart=$APP_DIR/backend/venv/bin/daphne -b 0.0.0.0 -p 8000 backend.asgi:application
Restart=always
RestartSec=3

[Install]
WantedBy=multi-user.target
EOF

# Create PM2 ecosystem
cat > ecosystem.config.js << EOF
module.exports = {
  apps: [{
    name: 'landlord-frontend',
    script: 'npm',
    args: 'start',
    cwd: '$APP_DIR/front',
    instances: 1,
    autorestart: true,
    watch: false,
    max_memory_restart: '1G',
    env: {
      NODE_ENV: 'production',
      PORT: 3000
    }
  }]
}
EOF

print_status "Configuring Nginx..."

# Create Nginx configuration
sudo tee /etc/nginx/sites-available/landlord-ai > /dev/null << 'EOF'
upstream django_backend {
    server 127.0.0.1:8000;
}

upstream nextjs_frontend {
    server 127.0.0.1:3000;
}

server {
    listen 80;
    server_name your-domain.com www.your-domain.com;
    
    # Security headers
    add_header X-Frame-Options "SAMEORIGIN" always;
    add_header X-XSS-Protection "1; mode=block" always;
    add_header X-Content-Type-Options "nosniff" always;
    
    # Main frontend (Next.js)
    location / {
        proxy_pass http://nextjs_frontend;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection 'upgrade';
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_cache_bypass $http_upgrade;
    }
    
    # Django API endpoints
    location /api/ {
        proxy_pass http://django_backend;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
    
    # WebSocket connections
    location /ws/ {
        proxy_pass http://django_backend;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_read_timeout 86400;
    }
    
    # Static files
    location /static/ {
        alias /srv/landlord-app/backend/staticfiles/;
        expires 30d;
    }
}
EOF

# Update domain in Nginx config
sudo sed -i "s/your-domain.com/$DOMAIN/g" /etc/nginx/sites-available/landlord-ai

# Enable site
sudo ln -sf /etc/nginx/sites-available/landlord-ai /etc/nginx/sites-enabled/
sudo rm -f /etc/nginx/sites-enabled/default

print_status "Setting permissions..."
sudo chown -R $USER:$USER $APP_DIR

print_status "Starting services..."

# Enable and start Redis
sudo systemctl enable redis-server
sudo systemctl start redis-server

# Enable and start Daphne
sudo systemctl daemon-reload
sudo systemctl enable daphne-landlord
sudo systemctl start daphne-landlord

# Start PM2
pm2 start ecosystem.config.js
pm2 save
pm2 startup

# Test Nginx configuration and restart
sudo nginx -t
sudo systemctl enable nginx
sudo systemctl restart nginx

print_status "Setting up firewall..."
sudo ufw allow OpenSSH
sudo ufw allow 'Nginx Full'
sudo ufw --force enable

print_status "Creating health check script..."
cat > $APP_DIR/health-check.sh << 'EOF'
#!/bin/bash
echo "=== Health Check $(date) ==="
echo "1. Daphne service: $(sudo systemctl is-active daphne-landlord)"
echo "2. PM2 processes:"
pm2 list
echo "3. Nginx: $(sudo systemctl is-active nginx)"
echo "4. Frontend: $(curl -s -o /dev/null -w "%{http_code}" http://localhost:3000/)"
echo "5. Backend: $(curl -s -o /dev/null -w "%{http_code}" http://localhost:8000/)"
EOF

chmod +x $APP_DIR/health-check.sh

print_status "✅ Deployment completed successfully!"
echo
echo "Next steps:"
echo "1. Update DNS to point $DOMAIN to this server"
echo "2. Install SSL certificate: sudo certbot --nginx -d $DOMAIN"
echo "3. Run health check: $APP_DIR/health-check.sh"
echo
echo "Service commands:"
echo "- Check backend: sudo systemctl status daphne-landlord"
echo "- Check frontend: pm2 status"
echo "- View logs: sudo journalctl -u daphne-landlord -f"
echo
echo "Your application should be available at:"
echo "- http://$DOMAIN (or http://your-server-ip)"
echo "- Backend API: http://$DOMAIN/api/"
echo "- WebSocket: ws://$DOMAIN/ws/"
