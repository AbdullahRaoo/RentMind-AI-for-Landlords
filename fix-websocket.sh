#!/bin/bash

echo "🔧 Fixing WebSocket Connection Issues"

# Create improved Nginx config with better WebSocket handling
sudo tee /etc/nginx/sites-available/rentmind-websocket-fixed > /dev/null << 'EOF'
# Frontend server (port 80)
server {
    listen 80;
    server_name srv889806.hstgr.cloud;
    
    # Serve from the out directory (Next.js static export)
    root /var/www/rentmind/front/out;
    index index.html;
    
    # Increase client timeouts for WebSocket
    client_max_body_size 64M;
    client_body_timeout 60s;
    client_header_timeout 60s;
    keepalive_timeout 65s;
    
    # Gzip compression for faster loading
    gzip on;
    gzip_vary on;
    gzip_min_length 1024;
    gzip_types text/plain text/css text/xml text/javascript application/javascript application/xml+rss application/json;
    
    # Enhanced WebSocket configuration
    location /ws/ {
        proxy_pass http://127.0.0.1:8000;
        proxy_http_version 1.1;
        
        # Essential WebSocket headers
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        
        # WebSocket specific timeouts
        proxy_read_timeout 86400s;
        proxy_send_timeout 86400s;
        proxy_connect_timeout 60s;
        
        # Disable buffering for WebSocket
        proxy_buffering off;
        proxy_cache off;
        
        # Allow large WebSocket frames
        proxy_max_temp_file_size 0;
        
        # Add CORS headers for WebSocket
        add_header Access-Control-Allow-Origin *;
        add_header Access-Control-Allow-Methods "GET, POST, OPTIONS";
        add_header Access-Control-Allow-Headers "Origin, X-Requested-With, Content-Type, Accept, Authorization, Cache-Control";
    }
    
    # Handle preflight requests for WebSocket
    location /ws/chat/ {
        if ($request_method = 'OPTIONS') {
            add_header Access-Control-Allow-Origin *;
            add_header Access-Control-Allow-Methods "GET, POST, OPTIONS";
            add_header Access-Control-Allow-Headers "Origin, X-Requested-With, Content-Type, Accept, Authorization, Cache-Control";
            add_header Access-Control-Max-Age 1728000;
            add_header Content-Type "text/plain; charset=utf-8";
            add_header Content-Length 0;
            return 204;
        }
        
        proxy_pass http://127.0.0.1:8000;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        
        proxy_read_timeout 86400s;
        proxy_send_timeout 86400s;
        proxy_connect_timeout 60s;
        proxy_buffering off;
        proxy_cache off;
    }
    
    # Proxy API calls to backend
    location /api/ {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        
        # Add CORS headers for API
        add_header Access-Control-Allow-Origin *;
        add_header Access-Control-Allow-Methods "GET, POST, PUT, DELETE, OPTIONS";
        add_header Access-Control-Allow-Headers "Origin, X-Requested-With, Content-Type, Accept, Authorization, Cache-Control";
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
EOF

# Remove old configurations and enable the fixed one
sudo rm -f /etc/nginx/sites-enabled/rentmind*
sudo ln -sf /etc/nginx/sites-available/rentmind-websocket-fixed /etc/nginx/sites-enabled/rentmind-websocket-fixed

# Test and reload Nginx
echo "🔧 Testing Nginx configuration..."
sudo nginx -t

if [ $? -eq 0 ]; then
    echo "✅ Nginx config is valid, reloading..."
    sudo systemctl reload nginx
    
    echo "✅ WebSocket configuration updated!"
    echo ""
    echo "🔍 Testing WebSocket connection..."
    
    # Test if backend is running
    if curl -s -o /dev/null -w "%{http_code}" http://localhost:8000/ | grep -q "200\|404"; then
        echo "✅ Backend is responding on port 8000"
    else
        echo "❌ Backend not responding - restarting..."
        sudo systemctl restart rentmind-backend
        sleep 3
    fi
    
    # Test WebSocket upgrade
    echo "🧪 Testing WebSocket upgrade headers..."
    curl -i -N -H "Connection: Upgrade" -H "Upgrade: websocket" -H "Sec-WebSocket-Version: 13" -H "Sec-WebSocket-Key: test" http://localhost/ws/chat/ 2>/dev/null | head -10
    
else
    echo "❌ Nginx configuration error"
    sudo nginx -t
fi

echo ""
echo "📋 Debugging commands:"
echo "   - Check WebSocket logs: sudo journalctl -u rentmind-backend -f"
echo "   - Check Nginx errors: sudo tail -f /var/log/nginx/error.log"
echo "   - Test backend directly: curl http://localhost:8000/"
