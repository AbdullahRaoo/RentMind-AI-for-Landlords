#!/bin/bash

echo "🚀 Starting RentMind Production Deployment..."

# Stop any existing Redis service that might conflict
echo "🛑 Stopping any existing Redis services..."
sudo systemctl stop redis-server 2>/dev/null || true
sudo pkill redis-server 2>/dev/null || true
sudo fuser -k 6379/tcp 2>/dev/null || true

# Stop any running containers
echo "📦 Stopping existing containers..."
docker-compose -f docker-compose.production.yml down

# Remove old images (optional - uncomment if needed)
echo "🗑️  Removing old images..."
docker rmi rentmind_backend rentmind_frontend 2>/dev/null || true

# Clean up any dangling containers/networks
echo "🧹 Cleaning up..."
docker system prune -f

# Build and start services
echo "🔨 Building and starting services..."
docker-compose -f docker-compose.production.yml up --build -d

# Wait for services to be ready
echo "⏳ Waiting for services to start..."
sleep 45

# Check service health
echo "🏥 Checking service health..."
docker-compose -f docker-compose.production.yml ps

# Test the application
echo "🧪 Testing application..."
echo "Testing health endpoint..."
sleep 10
curl -s http://localhost/health/ || echo "Health check endpoint not ready yet"

# Show logs
echo "📋 Recent logs:"
docker-compose -f docker-compose.production.yml logs --tail=30

echo "✅ Deployment complete!"
echo "🌐 Frontend: https://srv889806.hstgr.cloud"
echo "🔧 Backend API: https://srv889806.hstgr.cloud/api/"
echo "🏥 Health Check: https://srv889806.hstgr.cloud/health/"
echo "📊 Monitor logs: docker-compose -f docker-compose.production.yml logs -f"
echo "📊 Monitor cost tracking: docker-compose -f docker-compose.production.yml logs -f backend | grep -E '(🔧|💰|📊)'"
