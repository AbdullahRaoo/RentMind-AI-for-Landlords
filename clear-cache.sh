#!/bin/bash

echo "🧹 Browser Cache Buster"

cd /var/www/rentmind/front

# Add cache busting comment to index.html
if [ -f "out/index.html" ]; then
    echo "<!-- Cache bust: $(date) -->" >> out/index.html
    echo "✅ Added cache buster to index.html"
fi

# Clear Nginx access logs to see fresh requests
sudo truncate -s 0 /var/log/nginx/access.log
sudo truncate -s 0 /var/log/nginx/error.log

echo "✅ Cleared Nginx logs"
echo "💡 Now try refreshing your browser with Ctrl+F5"

# Show what's being served
echo ""
echo "📂 Current serving structure:"
echo "Root: /var/www/rentmind/front/out"
echo "Index file size: $(wc -c < out/index.html) bytes"
echo "Static assets:"
find out/_next/static/ -name "*.css" -o -name "*.js" | head -5
