#!/bin/bash

echo "🔧 Fixed Backend Setup"

cd /var/www/rentmind

echo "📋 Step 1: Cleaning up and recreating virtual environment..."
rm -rf airlords
python3 -m venv airlords

echo "📋 Step 2: Activating virtual environment..."
source airlords/bin/activate

echo "📋 Step 3: Upgrading pip and installing requirements..."
python -m pip install --upgrade pip
pip install --no-cache-dir -r requirements.txt

echo "📋 Step 4: Installing spaCy model separately..."
python -m spacy download en_core_web_sm

echo "📋 Step 5: Testing package installations..."
python -c "import django; print(f'✅ Django {django.get_version()}')"
python -c "import channels; print('✅ Channels installed')"
python -c "import daphne; print('✅ Daphne installed')"
python -c "import openai; print('✅ OpenAI installed')"
python -c "import langchain; print('✅ LangChain installed')"

echo "📋 Step 6: Setting up Django..."
cd backend
export DJANGO_SETTINGS_MODULE=backend.production_settings

echo "📋 Step 7: Running migrations..."
python manage.py migrate --settings=backend.production_settings

echo "📋 Step 8: Collecting static files..."
python manage.py collectstatic --noinput --settings=backend.production_settings

echo "📋 Step 9: Testing Django manually..."
timeout 10s python manage.py runserver 127.0.0.1:8002 --settings=backend.production_settings &
DJANGO_PID=$!
sleep 5

if curl -s -o /dev/null -w "%{http_code}" http://localhost:8002/ | grep -q "200\|404"; then
    echo "✅ Django works manually!"
    kill $DJANGO_PID 2>/dev/null
else
    echo "❌ Django still not working"
    kill $DJANGO_PID 2>/dev/null
fi

echo "📋 Step 10: Stopping the failing service..."
sudo systemctl stop rentmind-backend

echo "📋 Step 11: Creating fixed systemd service..."
sudo tee /etc/systemd/system/rentmind-backend.service > /dev/null << EOF
[Unit]
Description=RentMind Backend (Daphne)
After=network.target

[Service]
Type=simple
User=root
WorkingDirectory=/var/www/rentmind/backend
Environment=PATH=/var/www/rentmind/airlords/bin:/usr/local/bin:/usr/bin:/bin
Environment=VIRTUAL_ENV=/var/www/rentmind/airlords
Environment=DJANGO_SETTINGS_MODULE=backend.production_settings
Environment=PYTHONPATH=/var/www/rentmind:/var/www/rentmind/backend
ExecStart=/var/www/rentmind/airlords/bin/python /var/www/rentmind/airlords/bin/daphne -b 0.0.0.0 -p 8000 backend.asgi:application
Restart=always
RestartSec=10
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
EOF

echo "📋 Step 12: Reloading and starting service..."
sudo systemctl daemon-reload
sudo systemctl enable rentmind-backend
sudo systemctl start rentmind-backend

echo "📋 Step 13: Waiting for service to stabilize..."
sleep 10

echo "📋 Step 14: Checking service status..."
sudo systemctl status rentmind-backend --no-pager -l

echo "📋 Step 15: Testing connections..."
for i in {1..5}; do
    STATUS=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:8000/)
    if [ "$STATUS" = "200" ] || [ "$STATUS" = "404" ]; then
        echo "✅ Backend responding (attempt $i): HTTP $STATUS"
        break
    else
        echo "⏳ Attempt $i: HTTP $STATUS - waiting..."
        sleep 3
    fi
done

echo "📋 Step 16: Recent service logs..."
sudo journalctl -u rentmind-backend -n 5 --no-pager

echo ""
echo "✅ Setup complete!"
echo "🌐 Test: http://srv889806.hstgr.cloud:8000/"
echo "🔍 Logs: sudo journalctl -u rentmind-backend -f"
