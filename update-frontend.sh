#!/bin/bash

echo "🔄 Quick Frontend Update"

cd /var/www/rentmind

# Pull the latest changes
echo "⬇️ Pulling latest changes..."
git pull origin prod

# Rebuild frontend with the fixed WebSocket URL
echo "🔨 Rebuilding frontend..."
cd front
npm run build

echo "✅ Frontend updated!"
echo "🌐 Test your site: http://srv889806.hstgr.cloud"
echo "🔗 WebSocket should now connect via Nginx proxy"
