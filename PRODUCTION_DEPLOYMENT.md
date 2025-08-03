# RentMind Production Deployment Guide

## Quick Deployment Steps

### 1. First, push these new files to GitHub
```bash
git add .
git commit -m "Add production deployment files with cost tracking"
git push origin production
```

### 2. On your server, pull the latest changes
```bash
ssh root@srv889806.hstgr.cloud
cd /var/www/rentmind
git pull origin production
```

### 3. Set your environment variables
```bash
# Edit the .env.production file with your actual values
nano .env.production

# Set your OpenAI API key
export OPENAI_API_KEY="your-actual-openai-api-key"
```

### 4. Make the deployment script executable
```bash
chmod +x deploy-production.sh
```

### 5. Deploy
```bash
./deploy-production.sh
```

## What This Deployment Includes

✅ **Cost Tracking Features**: Your OpenAI API cost tracking is included  
✅ **No Path Issues**: Proper Docker context and volume mounting  
✅ **Health Checks**: Built-in health monitoring  
✅ **Nginx Proxy**: Proper routing for frontend/backend  
✅ **Redis**: For caching and session management  
✅ **Production Ready**: Environment variables and security settings  

## Monitoring Commands

```bash
# Check container status
docker-compose -f docker-compose.production.yml ps

# View logs
docker-compose -f docker-compose.production.yml logs -f

# Check specific service
docker-compose -f docker-compose.production.yml logs backend
docker-compose -f docker-compose.production.yml logs frontend

# Restart a service
docker-compose -f docker-compose.production.yml restart backend
```

## Accessing Your Application

- **Frontend**: http://srv889806.hstgr.cloud
- **Backend API**: http://srv889806.hstgr.cloud/api/
- **Health Check**: http://srv889806.hstgr.cloud/health/
- **Admin Panel**: http://srv889806.hstgr.cloud/admin/

## Troubleshooting

If you encounter issues:

1. **Check logs**: `docker-compose -f docker-compose.production.yml logs`
2. **Restart services**: `docker-compose -f docker-compose.production.yml restart`
3. **Rebuild**: `docker-compose -f docker-compose.production.yml up --build -d`

## Files Created

- `Dockerfile.backend` - Backend container definition
- `Dockerfile.frontend` - Frontend container definition  
- `docker-compose.production.yml` - Complete service orchestration
- `nginx.conf` - Reverse proxy configuration
- `.env.production` - Environment variables template
- `deploy-production.sh` - Automated deployment script
- `backend/chatbot/health_views.py` - Health check endpoint
- Updated `backend/backend/urls.py` - Added health check route
