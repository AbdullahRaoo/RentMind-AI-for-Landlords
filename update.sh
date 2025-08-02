#!/bin/bash

# Update script for RentMind AI deployment
# Run this when you want to deploy updates to your application

set -e

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

APP_DIR="/srv/landlord-app"

print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_status "🔄 Starting update process..."

cd $APP_DIR

# Backup current version
print_status "Creating backup..."
BACKUP_DIR="/srv/backups/landlord-$(date +%Y%m%d-%H%M%S)"
sudo mkdir -p $BACKUP_DIR
sudo cp backend/db.sqlite3 $BACKUP_DIR/ 2>/dev/null || true
sudo cp -r backend/staticfiles $BACKUP_DIR/ 2>/dev/null || true

# Pull latest code
print_status "Pulling latest code..."
git stash
git pull origin main

# Update backend
print_status "Updating backend..."
cd backend
source venv/bin/activate
pip install -r ../requirements.txt
python manage.py makemigrations
python manage.py migrate
python manage.py collectstatic --noinput

# Update frontend
print_status "Updating frontend..."
cd ../front
npm install
npm run build

# Restart services
print_status "Restarting services..."
sudo systemctl restart daphne-landlord
pm2 restart landlord-frontend
sudo systemctl reload nginx

# Health check
print_status "Running health check..."
sleep 5
$APP_DIR/health-check.sh

print_status "✅ Update completed successfully!"
echo "Application should be available in a few moments."
