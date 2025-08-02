# Ubuntu Server Deployment Guide for RentMind AI

This guide covers deploying your Next.js frontend and Django (Daphne) backend on an Ubuntu-based server.

## Prerequisites

1. Ubuntu server (18.04+) with sudo access
2. Domain name (optional but recommended)
3. Basic knowledge of Linux command line

## Step 1: Server Setup

### Update System
```bash
sudo apt update && sudo apt upgrade -y
```

### Install Required Software
```bash
# Install Python, Node.js, Nginx, and other dependencies
sudo apt install -y python3 python3-pip python3-venv nodejs npm nginx git curl
sudo apt install -y build-essential python3-dev

# Install PM2 for process management
sudo npm install -g pm2

# Install Node.js 18+ (if needed)
curl -fsSL https://deb.nodesource.com/setup_18.x | sudo -E bash -
sudo apt-get install -y nodejs
```

## Step 2: Project Setup

### Clone Repository
```bash
cd /srv
sudo mkdir -p landlord-app
sudo chown $USER:$USER landlord-app
cd landlord-app
git clone https://github.com/your-username/RentMind-AI-for-Landlords.git .
```

### Backend Setup (Django + Daphne)
```bash
# Create virtual environment
cd backend
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install -r ../requirements.txt

# Environment variables
cat > .env << 'EOF'
DEBUG=False
SECRET_KEY=your-super-secret-key-here-change-this
ALLOWED_HOSTS=your-domain.com,your-server-ip,localhost,127.0.0.1
DATABASE_URL=sqlite:///db.sqlite3
EOF

# Database migrations
python manage.py makemigrations
python manage.py migrate
python manage.py collectstatic --noinput

# Create superuser (optional)
python manage.py createsuperuser
```

### Frontend Setup (Next.js)
```bash
cd ../front

# Install dependencies
npm install

# Build for production
npm run build
```

## Step 3: Production Configuration

### Update Django Settings for Production
Create `backend/backend/production_settings.py`:
```python
from .settings import *
import os
from dotenv import load_dotenv

load_dotenv()

DEBUG = False
SECRET_KEY = os.getenv('SECRET_KEY', 'change-this-in-production')
ALLOWED_HOSTS = os.getenv('ALLOWED_HOSTS', 'localhost').split(',')

# Security settings
SECURE_BROWSER_XSS_FILTER = True
SECURE_CONTENT_TYPE_NOSNIFF = True
X_FRAME_OPTIONS = 'DENY'
SECURE_HSTS_SECONDS = 31536000
SECURE_HSTS_INCLUDE_SUBDOMAINS = True
SECURE_HSTS_PRELOAD = True

# Static files
STATIC_ROOT = os.path.join(BASE_DIR, 'staticfiles')
STATIC_URL = '/static/'

# Channel layers for production (Redis recommended)
CHANNEL_LAYERS = {
    'default': {
        'BACKEND': 'channels_redis.core.RedisChannelLayer',
        'CONFIG': {
            "hosts": [('127.0.0.1', 6379)],
        },
    },
}

# Database (consider PostgreSQL for production)
if os.getenv('DATABASE_URL'):
    import dj_database_url
    DATABASES['default'] = dj_database_url.parse(os.getenv('DATABASE_URL'))
```

### Install Redis (Recommended for Production)
```bash
sudo apt install -y redis-server
sudo systemctl enable redis-server
sudo systemctl start redis-server

# Add to requirements
echo "channels_redis" >> requirements.txt
echo "dj-database-url" >> requirements.txt
pip install channels_redis dj-database-url
```

## Step 4: Process Management

### Create Systemd Service for Daphne
```bash
sudo tee /etc/systemd/system/daphne-landlord.service > /dev/null << 'EOF'
[Unit]
Description=Daphne ASGI Server for Landlord AI
After=network.target

[Service]
Type=simple
User=www-data
WorkingDirectory=/srv/landlord-app/backend
Environment=DJANGO_SETTINGS_MODULE=backend.production_settings
ExecStart=/srv/landlord-app/backend/venv/bin/daphne -b 0.0.0.0 -p 8000 backend.asgi:application
Restart=always
RestartSec=3

[Install]
WantedBy=multi-user.target
EOF

# Set proper permissions
sudo chown -R www-data:www-data /srv/landlord-app
sudo systemctl daemon-reload
sudo systemctl enable daphne-landlord
```

### Create PM2 Ecosystem for Next.js
```bash
cd /srv/landlord-app/front
cat > ecosystem.config.js << 'EOF'
module.exports = {
  apps: [{
    name: 'landlord-frontend',
    script: 'npm',
    args: 'start',
    cwd: '/srv/landlord-app/front',
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

# Start with PM2
pm2 start ecosystem.config.js
pm2 save
pm2 startup
```

## Step 5: Nginx Configuration

### Create Nginx Configuration
```bash
sudo tee /etc/nginx/sites-available/landlord-ai > /dev/null << 'EOF'
# Upstream servers
upstream django_backend {
    server 127.0.0.1:8000;
}

upstream nextjs_frontend {
    server 127.0.0.1:3000;
}

server {
    listen 80;
    server_name your-domain.com www.your-domain.com;  # Replace with your domain
    
    # Security headers
    add_header X-Frame-Options "SAMEORIGIN" always;
    add_header X-XSS-Protection "1; mode=block" always;
    add_header X-Content-Type-Options "nosniff" always;
    add_header Referrer-Policy "no-referrer-when-downgrade" always;
    
    # Rate limiting
    limit_req_zone $binary_remote_addr zone=api:10m rate=10r/s;
    limit_req_zone $binary_remote_addr zone=ws:10m rate=5r/s;
    
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
        proxy_read_timeout 86400;
    }
    
    # Django API endpoints
    location /api/ {
        limit_req zone=api burst=20 nodelay;
        proxy_pass http://django_backend;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_read_timeout 300;
        proxy_connect_timeout 300;
        proxy_send_timeout 300;
    }
    
    # WebSocket connections
    location /ws/ {
        limit_req zone=ws burst=10 nodelay;
        proxy_pass http://django_backend;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_read_timeout 86400;
        proxy_send_timeout 86400;
        proxy_connect_timeout 86400;
    }
    
    # Django admin (optional)
    location /admin/ {
        proxy_pass http://django_backend;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
    
    # Static files
    location /static/ {
        alias /srv/landlord-app/backend/staticfiles/;
        expires 30d;
        add_header Cache-Control "public, immutable";
    }
    
    # Media files (if any)
    location /media/ {
        alias /srv/landlord-app/backend/media/;
        expires 30d;
        add_header Cache-Control "public, immutable";
    }
    
    # Security.txt
    location /.well-known/security.txt {
        return 200 "Contact: your-email@domain.com\nExpires: 2025-12-31T23:59:59.000Z\n";
        add_header Content-Type text/plain;
    }
}
EOF

# Enable site
sudo ln -sf /etc/nginx/sites-available/landlord-ai /etc/nginx/sites-enabled/
sudo rm -f /etc/nginx/sites-enabled/default
sudo nginx -t
sudo systemctl reload nginx
```

## Step 6: SSL/TLS with Let's Encrypt

```bash
# Install Certbot
sudo apt install -y certbot python3-certbot-nginx

# Get SSL certificate
sudo certbot --nginx -d your-domain.com -d www.your-domain.com

# Auto-renewal
sudo systemctl enable certbot.timer
```

## Step 7: Firewall Configuration

```bash
# Configure UFW
sudo ufw allow OpenSSH
sudo ufw allow 'Nginx Full'
sudo ufw enable
```

## Step 8: Start Services

```bash
# Start Django backend
sudo systemctl start daphne-landlord
sudo systemctl status daphne-landlord

# Start Next.js frontend
cd /srv/landlord-app/front
pm2 restart landlord-frontend
pm2 status

# Restart Nginx
sudo systemctl restart nginx
```

## Step 9: Environment Variables & Configuration

### Update Frontend WebSocket URL
Edit `front/components/chatbot.tsx`:
```typescript
// Change from localhost to your domain
const WS_URL = process.env.NODE_ENV === 'production' 
  ? "wss://your-domain.com/ws/chat/" 
  : "ws://localhost:8000/ws/chat/";
```

### Create Frontend Environment File
```bash
cd /srv/landlord-app/front
cat > .env.production << 'EOF'
NEXT_PUBLIC_API_URL=https://your-domain.com/api
NEXT_PUBLIC_WS_URL=wss://your-domain.com/ws
EOF
```

## Step 10: Monitoring & Maintenance

### Log Files
- Django: `sudo journalctl -u daphne-landlord -f`
- Next.js: `pm2 logs landlord-frontend`
- Nginx: `sudo tail -f /var/log/nginx/access.log`

### Health Check Script
```bash
cat > /srv/landlord-app/health-check.sh << 'EOF'
#!/bin/bash
echo "=== Health Check $(date) ==="

# Check services
echo "1. Checking Daphne service..."
sudo systemctl is-active daphne-landlord

echo "2. Checking PM2 processes..."
pm2 list

echo "3. Checking Nginx..."
sudo systemctl is-active nginx

echo "4. Checking disk space..."
df -h /

echo "5. Checking memory..."
free -h

echo "6. Testing endpoints..."
curl -s -o /dev/null -w "%{http_code}" http://localhost:3000/
curl -s -o /dev/null -w "%{http_code}" http://localhost:8000/
EOF

chmod +x /srv/landlord-app/health-check.sh
```

### Backup Script
```bash
cat > /srv/landlord-app/backup.sh << 'EOF'
#!/bin/bash
BACKUP_DIR="/srv/backups/landlord-$(date +%Y%m%d)"
mkdir -p $BACKUP_DIR

# Backup database
cp /srv/landlord-app/backend/db.sqlite3 $BACKUP_DIR/

# Backup static files
tar -czf $BACKUP_DIR/staticfiles.tar.gz -C /srv/landlord-app/backend staticfiles/

# Clean old backups (keep 7 days)
find /srv/backups -type d -name "landlord-*" -mtime +7 -exec rm -rf {} \;
EOF

chmod +x /srv/landlord-app/backup.sh

# Add to crontab for daily backups
(crontab -l 2>/dev/null; echo "0 2 * * * /srv/landlord-app/backup.sh") | crontab -
```

## Troubleshooting

### Common Issues

1. **WebSocket Connection Failed**
   - Check if Nginx proxy_pass for /ws/ is correct
   - Verify Daphne is running on port 8000
   - Check firewall rules

2. **Static Files Not Loading**
   - Run `python manage.py collectstatic`
   - Check Nginx static file configuration

3. **Permission Issues**
   - Ensure www-data owns files: `sudo chown -R www-data:www-data /srv/landlord-app`

4. **Memory Issues**
   - Monitor with `htop`
   - Consider increasing server resources
   - Optimize Django queries

### Useful Commands
```bash
# Restart all services
sudo systemctl restart daphne-landlord nginx
pm2 restart landlord-frontend

# View logs
sudo journalctl -u daphne-landlord -f
pm2 logs landlord-frontend
sudo tail -f /var/log/nginx/error.log

# Update deployment
cd /srv/landlord-app
git pull origin main
cd backend && source venv/bin/activate && pip install -r ../requirements.txt
python manage.py migrate && python manage.py collectstatic --noinput
cd ../front && npm install && npm run build
sudo systemctl restart daphne-landlord
pm2 restart landlord-frontend
```

## Performance Optimization

1. **Enable Gzip Compression** (add to Nginx config):
```nginx
gzip on;
gzip_types text/plain text/css application/json application/javascript text/xml application/xml application/xml+rss text/javascript;
```

2. **Database Optimization**:
   - Consider PostgreSQL for production
   - Add database indexes for frequently queried fields
   - Set up connection pooling

3. **Caching**:
   - Implement Redis caching for Django
   - Use Next.js built-in caching
   - Add Nginx caching for static content

4. **Monitoring**:
   - Set up monitoring with tools like Grafana/Prometheus
   - Configure error reporting (Sentry)
   - Set up log aggregation

This deployment guide provides a production-ready setup for your RentMind AI application on Ubuntu server!
