#!/bin/bash

echo "🌐 Setting up complete Nginx configuration (Frontend + Backend)"

# Create complete Nginx config for both frontend and backend
sudo tee /etc/nginx/sites-available/rentmind-complete > /dev/null << 'EOF'
# Frontend server (port 80)
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
    
    # Proxy WebSocket connections to backend
    location /ws/ {
        proxy_pass http://127.0.0.1:8000;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
    
    # Proxy API calls to backend
    location /api/ {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
    
    # Proxy admin interface to backend
    location /admin/ {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
    
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

# Backend server (direct access on port 8000 if needed)
server {
    listen 8001;
    server_name srv889806.hstgr.cloud;
    
    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
    
    location /ws/ {
        proxy_pass http://127.0.0.1:8000;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
EOF

# Remove old configurations and enable the complete one
sudo rm -f /etc/nginx/sites-enabled/default
sudo rm -f /etc/nginx/sites-enabled/rentmind*
sudo ln -sf /etc/nginx/sites-available/rentmind-complete /etc/nginx/sites-enabled/rentmind-complete

# Test and reload Nginx
echo "🔧 Testing Nginx configuration..."
sudo nginx -t

if [ $? -eq 0 ]; then
    echo "✅ Nginx config is valid, reloading..."
    sudo systemctl reload nginx
    
    echo "✅ Complete Nginx setup done!"
    echo ""
    echo "🌐 Services:"
    echo "   - Frontend: http://srv889806.hstgr.cloud (port 80)"
    echo "   - Backend API: http://srv889806.hstgr.cloud/api/ (proxied to :8000)"
    echo "   - WebSocket: ws://srv889806.hstgr.cloud/ws/ (proxied to :8000)"
    echo "   - Direct Backend: http://srv889806.hstgr.cloud:8001 (optional)"
    
else
    echo "❌ Nginx configuration error"
    sudo nginx -t
fi
