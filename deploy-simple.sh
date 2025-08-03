#!/bin/bash

echo "🚀 Simple Frontend Deployment"

# Clean up Docker first
echo "🧹 Cleaning up Docker..."
./cleanup-docker.sh

echo "🌐 Setting up Nginx for static files..."

# Install Nginx
sudo apt update
sudo apt install -y nginx

# Stop Nginx
sudo systemctl stop nginx

# Create a simple static site config
sudo tee /etc/nginx/sites-available/rentmind << 'EOF'
server {
    listen 80;
    server_name srv889806.hstgr.cloud;
    
    root /var/www/rentmind/front/.next/static;
    index index.html;
    
    # Serve the main app files
    location / {
        root /var/www/rentmind/front;
        try_files $uri $uri.html /.next/server/app/index.html;
    }
    
    # Serve Next.js static assets
    location /_next/static/ {
        root /var/www/rentmind/front;
        expires 1y;
        add_header Cache-Control "public, immutable";
    }
    
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

echo "✅ Deployment complete!"
echo "🌐 Check: http://srv889806.hstgr.cloud"
echo "📊 Status: sudo systemctl status nginx"
