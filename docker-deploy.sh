#!/bin/bash

# Docker Deployment Script for RentMind AI
# Alternative deployment method using Docker containers

set -e

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    print_error "Docker is not installed. Please install Docker first."
    exit 1
fi

if ! command -v docker-compose &> /dev/null; then
    print_error "Docker Compose is not installed. Please install Docker Compose first."
    exit 1
fi

print_status "🐳 Starting Docker deployment for RentMind AI..."

# Create environment file if it doesn't exist
if [ ! -f .env ]; then
    print_status "Creating .env file..."
    cat > .env << EOF
SECRET_KEY=$(openssl rand -base64 32)
ALLOWED_HOSTS=localhost,127.0.0.1,your-domain.com
DEBUG=False
EOF
    print_warning "Please update the .env file with your domain and other settings"
fi

# Create directories for volumes
print_status "Creating directories..."
mkdir -p ssl logs

# Build and start containers
print_status "Building and starting containers..."
docker-compose down --remove-orphans
docker-compose build --no-cache
docker-compose up -d

# Wait for services to start
print_status "Waiting for services to start..."
sleep 30

# Check service health
print_status "Checking service health..."
docker-compose ps

# Run database migrations
print_status "Running database migrations..."
docker-compose exec backend python manage.py makemigrations
docker-compose exec backend python manage.py migrate

# Create superuser (optional)
print_status "Creating superuser (optional)..."
docker-compose exec backend python manage.py createsuperuser --noinput || true

# Show logs
print_status "Showing recent logs..."
docker-compose logs --tail=50

print_status "✅ Docker deployment completed!"
echo
echo "Your application should be available at:"
echo "- Frontend: http://localhost"
echo "- Backend API: http://localhost/api/"
echo "- Django Admin: http://localhost/admin/"
echo
echo "Useful commands:"
echo "- View logs: docker-compose logs -f"
echo "- Stop services: docker-compose down"
echo "- Restart services: docker-compose restart"
echo "- Update: docker-compose pull && docker-compose up -d"
