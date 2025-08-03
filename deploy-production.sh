#!/bin/bash

echo "🚀 Starting RentMind Production Deployment..."

# Stop any running containers
echo "📦 Stopping existing containers..."
docker-compose -f docker-compose.production.yml down

# Remove old images (optional - uncomment if needed)
echo "🗑️  Removing old images..."
docker rmi rentmind_backend rentmind_frontend 2>/dev/null || true

# Build and start services
echo "🔨 Building and starting services..."
docker-compose -f docker-compose.production.yml up --build -d

# Wait for services to be ready
echo "⏳ Waiting for services to start..."
sleep 30

# Check service health
echo "🏥 Checking service health..."
docker-compose -f docker-compose.production.yml ps

# Show logs
echo "📋 Recent logs:"
docker-compose -f docker-compose.production.yml logs --tail=20

echo "✅ Deployment complete!"
echo "🌐 Frontend: http://localhost"
echo "🔧 Backend API: http://localhost/api/"
echo "📊 Check logs: docker-compose -f docker-compose.production.yml logs -f"
