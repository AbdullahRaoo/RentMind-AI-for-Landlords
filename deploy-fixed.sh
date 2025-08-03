#!/bin/bash

echo "🚀 Fixed Frontend Deployment"

cd /var/www/rentmind/front

# Clean everything first
echo "🧹 Cleaning build cache..."
rm -rf .next out node_modules/.cache

# Install dependencies if needed
echo "📦 Installing dependencies..."
npm install

# Build with proper export
echo "📦 Building Next.js..."
npm run build

# Check if out directory was created
if [ -d "out" ]; then
    echo "✅ Out directory found, using static export"
    BUILD_DIR="out"
    ROOT_PATH="/var/www/rentmind/front/out"
else
    echo "⚠️ Out directory not found, using .next/server/app"
    BUILD_DIR=".next/server/app"
    ROOT_PATH="/var/www/rentmind/front/.next/server/app"
fi

echo "🌐 Setting up Nginx configuration..."

# Create optimized Nginx config
sudo tee /etc/nginx/sites-available/rentmind-fixed > /dev/null << 'EOF'
server {
    listen 80;
    server_name srv889806.hstgr.cloud;
    
    # Use the correct build directory
    root /var/www/rentmind/front;
    
    # Gzip compression
    gzip on;
    gzip_vary on;
    gzip_min_length 1024;
    gzip_types text/plain text/css text/xml text/javascript application/javascript application/xml+rss application/json;
    
    # Cache static assets aggressively
    location /_next/static/ {
        expires 1y;
        add_header Cache-Control "public, immutable";
        try_files $uri =404;
    }
    
    # Handle static files
    location ~* \.(js|css|png|jpg|jpeg|gif|ico|svg|woff|woff2|ttf|eot)$ {
        expires 1y;
        add_header Cache-Control "public, max-age=31536000";
        try_files $uri =404;
    }
    
    # Main location - serve from out if exists, otherwise from .next/server/app
    location / {
        # Try out directory first, then .next/server/app
        try_files /out/$uri /out/$uri.html /out/$uri/index.html /.next/server/app/$uri /.next/server/app/$uri.html /.next/server/app/index.html =404;
        
        # No cache for HTML files to avoid stale content
        add_header Cache-Control "no-cache, no-store, must-revalidate";
        add_header Pragma "no-cache";
        add_header Expires "0";
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

# Enable the new config
sudo ln -sf /etc/nginx/sites-available/rentmind-fixed /etc/nginx/sites-enabled/rentmind
sudo rm -f /etc/nginx/sites-enabled/default

# Test and reload Nginx
echo "🔧 Testing Nginx configuration..."
sudo nginx -t

if [ $? -eq 0 ]; then
    echo "✅ Nginx config is valid, reloading..."
    sudo systemctl reload nginx
    
    # Clear browser cache by adding a timestamp
    echo "🔄 Adding cache busting..."
    echo "<!-- Cache bust: $(date) -->" | sudo tee -a $ROOT_PATH/index.html > /dev/null
    
    echo "✅ Fixed deployment complete!"
    echo "🌐 Your site: http://srv889806.hstgr.cloud"
    echo "💡 Try force refresh (Ctrl+F5) in your browser to clear cache"
    
    # Show what files are actually being served
    echo ""
    echo "📂 Files being served:"
    echo "Build directory: $BUILD_DIR"
    if [ -f "$ROOT_PATH/index.html" ]; then
        echo "✅ Main HTML file found"
    else
        echo "❌ Main HTML file NOT found"
        echo "Available files in $ROOT_PATH:"
        ls -la "$ROOT_PATH" 2>/dev/null || echo "Directory not accessible"
    fi
else
    echo "❌ Nginx configuration error"
    sudo nginx -t
fi
