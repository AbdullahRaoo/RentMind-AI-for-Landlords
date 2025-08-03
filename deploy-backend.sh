#!/bin/bash

echo "🚀 Backend Deployment Script"

cd /var/www/rentmind/backend

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source ../airlords/bin/activate

# Install/update requirements
echo "📦 Installing requirements..."
pip install -r ../requirements.txt

# Run migrations
echo "🗄️ Running database migrations..."
python manage.py migrate --settings=backend.production_settings

# Collect static files
echo "📁 Collecting static files..."
python manage.py collectstatic --noinput --settings=backend.production_settings

# Create systemd service file for the backend
echo "🔧 Creating systemd service..."
sudo tee /etc/systemd/system/rentmind-backend.service > /dev/null << EOF
[Unit]
Description=RentMind Backend (Daphne)
After=network.target

[Service]
Type=simple
User=root
WorkingDirectory=/var/www/rentmind/backend
Environment=PATH=/var/www/rentmind/airlords/bin
ExecStart=/var/www/rentmind/airlords/bin/daphne -b 0.0.0.0 -p 8000 backend.asgi:application
ExecReload=/bin/kill -s HUP \$MAINPID
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
EOF

# Set environment variable for production
echo "🌍 Setting up environment..."
export DJANGO_SETTINGS_MODULE=backend.production_settings

# Enable and start the service
echo "▶️ Starting backend service..."
sudo systemctl daemon-reload
sudo systemctl enable rentmind-backend
sudo systemctl restart rentmind-backend

# Wait a moment for the service to start
sleep 3

# Check service status
echo "📊 Checking service status..."
sudo systemctl status rentmind-backend --no-pager -l

# Test if the backend is responding
echo "🧪 Testing backend connection..."
if curl -s -o /dev/null -w "%{http_code}" http://localhost:8000/ | grep -q "200\|404"; then
    echo "✅ Backend is running!"
else
    echo "❌ Backend might not be responding. Check logs:"
    echo "📋 Recent logs:"
    sudo journalctl -u rentmind-backend -n 10 --no-pager
fi

echo ""
echo "🌐 Backend should be available at:"
echo "   - Local: http://localhost:8000"
echo "   - Public: http://srv889806.hstgr.cloud:8000"
echo ""
echo "📋 Useful commands:"
echo "   - Check status: sudo systemctl status rentmind-backend"
echo "   - View logs: sudo journalctl -u rentmind-backend -f"
echo "   - Restart: sudo systemctl restart rentmind-backend"
