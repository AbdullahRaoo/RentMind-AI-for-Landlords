#!/bin/bash

echo "🔍 Backend Diagnosis and Setup"

cd /var/www/rentmind

echo "📋 Step 1: Checking virtual environment..."
if [ -d "airlords" ]; then
    echo "✅ Virtual environment 'airlords' exists"
else
    echo "❌ Virtual environment 'airlords' not found - creating it..."
    python3 -m venv airlords
fi

echo ""
echo "📋 Step 2: Activating virtual environment and installing requirements..."
source airlords/bin/activate

# Check if requirements.txt exists
if [ -f "requirements.txt" ]; then
    echo "✅ Found requirements.txt - installing packages..."
    pip install --upgrade pip
    pip install -r requirements.txt
else
    echo "❌ requirements.txt not found - installing essential packages..."
    pip install django channels daphne langchain openai faiss-cpu spacy scikit-learn pandas numpy
fi

echo ""
echo "📋 Step 3: Checking backend directory and dependencies..."
cd backend

# Check if manage.py exists
if [ -f "manage.py" ]; then
    echo "✅ Django project found"
else
    echo "❌ Django project not found in backend directory"
    exit 1
fi

echo ""
echo "📋 Step 4: Testing Django installation..."
python -c "import django; print(f'Django version: {django.get_version()}')" || echo "❌ Django not properly installed"
python -c "import channels; print('✅ Channels installed')" || echo "❌ Channels not installed"
python -c "import daphne; print('✅ Daphne installed')" || echo "❌ Daphne not installed"

echo ""
echo "📋 Step 5: Running Django migrations..."
python manage.py migrate --settings=backend.production_settings

echo ""
echo "📋 Step 6: Collecting static files..."
python manage.py collectstatic --noinput --settings=backend.production_settings

echo ""
echo "📋 Step 7: Testing Django startup..."
timeout 10s python manage.py runserver 127.0.0.1:8001 --settings=backend.production_settings &
TEST_PID=$!
sleep 3

if kill -0 $TEST_PID 2>/dev/null; then
    echo "✅ Django can start successfully"
    kill $TEST_PID
else
    echo "❌ Django failed to start"
fi

echo ""
echo "📋 Step 8: Checking systemd service..."
sudo systemctl status rentmind-backend --no-pager || echo "Service not found"

echo ""
echo "📋 Step 9: Recreating systemd service with proper environment..."
sudo tee /etc/systemd/system/rentmind-backend.service > /dev/null << EOF
[Unit]
Description=RentMind Backend (Daphne)
After=network.target

[Service]
Type=simple
User=root
WorkingDirectory=/var/www/rentmind/backend
Environment=PATH=/var/www/rentmind/airlords/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
Environment=DJANGO_SETTINGS_MODULE=backend.production_settings
Environment=PYTHONPATH=/var/www/rentmind
ExecStart=/var/www/rentmind/airlords/bin/daphne -b 0.0.0.0 -p 8000 backend.asgi:application
ExecReload=/bin/kill -s HUP \$MAINPID
Restart=always
RestartSec=5
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
EOF

echo ""
echo "📋 Step 10: Restarting backend service..."
sudo systemctl daemon-reload
sudo systemctl enable rentmind-backend
sudo systemctl restart rentmind-backend

echo ""
echo "📋 Step 11: Waiting for service to start..."
sleep 5

echo ""
echo "📋 Step 12: Checking service status..."
sudo systemctl status rentmind-backend --no-pager -l

echo ""
echo "📋 Step 13: Testing backend connection..."
for i in {1..10}; do
    if curl -s -o /dev/null -w "%{http_code}" http://localhost:8000/ | grep -q "200\|404"; then
        echo "✅ Backend is responding on attempt $i"
        break
    else
        echo "⏳ Attempt $i: Backend not responding yet..."
        sleep 2
    fi
done

echo ""
echo "📋 Step 14: Testing WebSocket endpoint..."
curl -i -N -H "Connection: Upgrade" -H "Upgrade: websocket" -H "Sec-WebSocket-Version: 13" -H "Sec-WebSocket-Key: test" http://localhost:8000/ws/chat/ 2>/dev/null | head -5

echo ""
echo "📋 Step 15: Showing recent logs..."
echo "🔍 Backend logs (last 10 lines):"
sudo journalctl -u rentmind-backend -n 10 --no-pager

echo ""
echo "📋 Diagnosis Complete!"
echo "🌐 Test your backend: http://srv889806.hstgr.cloud:8000/"
echo "🔧 Check logs: sudo journalctl -u rentmind-backend -f"
