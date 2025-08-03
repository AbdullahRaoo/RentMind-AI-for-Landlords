#!/bin/bash

echo "🚀 Deploying Next.js Frontend to srv889806.hstgr.cloud"

# Clean up any existing Docker stuff
echo "🧹 Cleaning up Docker..."
./cleanup-docker.sh

# Export the Next.js app as static files
echo "📦 Exporting Next.js as static files..."
cd front
npm run build

# Check if out folder exists, if not create it from .next
if [ ! -d "out" ]; then
    echo "Creating out directory from .next build..."
    cp -r .next/static ./out-temp/
    mkdir -p out
    cp -r .next/* out/ 2>/dev/null || true
    # Copy static assets
    if [ -d ".next/static" ]; then
        mkdir -p out/_next
        cp -r .next/static out/_next/
    fi
    # Copy standalone files if they exist
    if [ -d ".next/standalone" ]; then
        cp -r .next/standalone/* out/ 2>/dev/null || true
    fi
fi

cd ..

# Install Nginx if not installed
echo "🌐 Setting up Nginx..."
sudo apt update
sudo apt install -y nginx

# Stop Nginx
sudo systemctl stop nginx

# Copy our config
sudo cp nginx-simple.conf /etc/nginx/sites-available/rentmind
sudo ln -sf /etc/nginx/sites-available/rentmind /etc/nginx/sites-enabled/
sudo rm -f /etc/nginx/sites-enabled/default

# Test Nginx config
sudo nginx -t

# Start Nginx
sudo systemctl start nginx
sudo systemctl enable nginx

# Check status
sudo systemctl status nginx --no-pager

echo "✅ Deployment complete!"
echo "🌐 Your site should be live at: http://srv889806.hstgr.cloud"
echo "📊 Check Nginx status: sudo systemctl status nginx"
echo "📋 Check Nginx logs: sudo tail -f /var/log/nginx/error.log"
