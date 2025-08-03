#!/bin/bash

echo "🚀 Fast Frontend Deployment"

# Clean up Docker first
echo "🧹 Cleaning up Docker..."
./cleanup-docker.sh

echo "📦 Building Next.js for static export..."
cd front
npm run build
npm run export 2>/dev/null || echo "Export command not available, using build output"
cd ..

echo "🌐 Setting up Nginx for fast serving..."

# Install Nginx
sudo apt update
sudo apt install -y nginx

# Stop Nginx
sudo systemctl stop nginx

# Create optimized config for Next.js
sudo tee /etc/nginx/sites-available/rentmind << 'EOF'
server {
    listen 80;
    server_name srv889806.hstgr.cloud;
    
    # Main document root - serve from out folder if exists, otherwise from .next
    root /var/www/rentmind/front;
    index index.html;
    
    # Gzip compression for faster loading
    gzip on;
    gzip_vary on;
    gzip_min_length 1024;
    gzip_types text/plain text/css text/xml text/javascript application/javascript application/xml+rss application/json;
    
    # Main location - try multiple fallbacks
    location / {
        try_files $uri $uri.html $uri/index.html /index.html;
        add_header Cache-Control "public, max-age=3600";
    }
    
    # Static assets from _next folder
    location /_next/static/ {
        expires 1y;
        add_header Cache-Control "public, immutable";
        try_files $uri =404;
    }
    
    # Other static files
    location ~* \.(js|css|png|jpg|jpeg|gif|ico|svg|woff|woff2|ttf|eot)$ {
        expires 1y;
        add_header Cache-Control "public, max-age=31536000";
        try_files $uri =404;
    }
    
    # Security headers
    add_header X-Frame-Options "SAMEORIGIN" always;
    add_header X-Content-Type-Options "nosniff" always;
    add_header Referrer-Policy "no-referrer-when-downgrade" always;
    
    # Error pages
    error_page 404 /404.html;
    error_page 500 502 503 504 /50x.html;
}
EOF

# Enable the site
sudo ln -sf /etc/nginx/sites-available/rentmind /etc/nginx/sites-enabled/
sudo rm -f /etc/nginx/sites-enabled/default

# Test config
sudo nginx -t

# Start Nginx
sudo systemctl start nginx
sudo systemctl enable nginx

# Show status
sudo systemctl status nginx --no-pager

echo "✅ Fast deployment complete!"
echo "🌐 Your site: http://srv889806.hstgr.cloud"
echo "📊 Check status: sudo systemctl status nginx"
echo "📋 Check logs: sudo tail -f /var/log/nginx/access.log"
echo "🔍 Debug: curl -I http://srv889806.hstgr.cloud"
