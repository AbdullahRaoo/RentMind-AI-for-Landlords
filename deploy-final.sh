#!/bin/bash

echo "🎯 Final Frontend Deployment Fix"

cd /var/www/rentmind/front

# Check if out directory exists
if [ ! -d "out" ]; then
    echo "❌ Out directory not found. Running build first..."
    npm run build
fi

echo "🌐 Setting up Nginx for static export..."

# Create the final optimized Nginx config
sudo tee /etc/nginx/sites-available/rentmind-final > /dev/null << 'EOF'
server {
    listen 80;
    server_name srv889806.hstgr.cloud;
    
    # Serve from the out directory (Next.js static export)
    root /var/www/rentmind/front/out;
    index index.html;
    
    # Gzip compression for faster loading
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
    
    # Handle other static files
    location ~* \.(js|css|png|jpg|jpeg|gif|ico|svg|woff|woff2|ttf|eot|avif)$ {
        expires 1y;
        add_header Cache-Control "public, max-age=31536000";
        try_files $uri =404;
    }
    
    # Main location - serve HTML files
    location / {
        try_files $uri $uri.html $uri/index.html /index.html;
        
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
}
EOF

# Enable the new config and remove conflicting ones
sudo rm -f /etc/nginx/sites-enabled/default
sudo rm -f /etc/nginx/sites-enabled/rentmind
sudo rm -f /etc/nginx/sites-enabled/rentmind-fixed
sudo ln -sf /etc/nginx/sites-available/rentmind-final /etc/nginx/sites-enabled/rentmind-final

# Test and reload Nginx
echo "🔧 Testing Nginx configuration..."
sudo nginx -t

if [ $? -eq 0 ]; then
    echo "✅ Nginx config is valid, reloading..."
    sudo systemctl reload nginx
    
    echo "✅ Final deployment complete!"
    echo "🌐 Your site: http://srv889806.hstgr.cloud"
    echo "💡 Try force refresh (Ctrl+F5) in your browser to clear cache"
    
    # Show the structure
    echo ""
    echo "📂 Serving from: /var/www/rentmind/front/out"
    echo "📂 Files available:"
    ls -la out/ | head -10
    
    echo ""
    echo "🔍 Static assets:"
    ls -la out/_next/static/ | head -5
    
    # Test the setup
    echo ""
    echo "🧪 Testing setup:"
    curl -s -o /dev/null -w "Main page: HTTP %{http_code}\n" http://localhost/
    
    if [ -f "out/_next/static/css/"*.css ]; then
        CSS_FILE=$(ls out/_next/static/css/*.css | head -1 | sed 's|out||')
        curl -s -o /dev/null -w "CSS file: HTTP %{http_code}\n" http://localhost$CSS_FILE
    fi
    
else
    echo "❌ Nginx configuration error"
    sudo nginx -t
fi
