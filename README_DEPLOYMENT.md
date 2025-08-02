# RentMind AI - Ubuntu Server Deployment

This project contains an AI-powered landlord assistant with Next.js frontend and Django (Daphne) backend. This guide provides multiple deployment options for Ubuntu servers.

## 🚀 Quick Start (Recommended)

### Option 1: Traditional Deployment (Systemd + PM2)

1. **Prepare your Ubuntu server:**
   ```bash
   # Clone the repository
   sudo mkdir -p /srv/landlord-app
   sudo chown $USER:$USER /srv/landlord-app
   cd /srv/landlord-app
   git clone https://github.com/your-username/RentMind-AI-for-Landlords.git .
   ```

2. **Run the deployment script:**
   ```bash
   chmod +x deploy.sh
   ./deploy.sh
   ```

3. **Configure your domain:**
   ```bash
   # Update the domain in deploy.sh before running
   DOMAIN="your-domain.com"
   ```

4. **Install SSL certificate:**
   ```bash
   sudo certbot --nginx -d your-domain.com -d www.your-domain.com
   ```

### Option 2: Docker Deployment

1. **Install Docker and Docker Compose:**
   ```bash
   # Install Docker
   curl -fsSL https://get.docker.com -o get-docker.sh
   sh get-docker.sh

   # Install Docker Compose
   sudo curl -L "https://github.com/docker/compose/releases/download/v2.20.0/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
   sudo chmod +x /usr/local/bin/docker-compose
   ```

2. **Deploy with Docker:**
   ```bash
   chmod +x docker-deploy.sh
   ./docker-deploy.sh
   ```

## 📁 Project Structure

```
RentMind-AI-for-Landlords/
├── backend/                    # Django backend
│   ├── backend/               # Django settings and configuration
│   ├── chatbot/              # Main chatbot app
│   └── manage.py
├── front/                     # Next.js frontend
│   ├── components/           # React components
│   ├── app/                  # Next.js app directory
│   └── package.json
├── AI Assistant/             # AI modules
├── predictive_maintenance_ai/ # Maintenance prediction
├── Rent Pricing AI/          # Rent prediction
├── deploy.sh                 # Traditional deployment script
├── docker-deploy.sh          # Docker deployment script
├── update.sh                 # Update script
└── DEPLOYMENT_GUIDE.md       # Detailed deployment guide
```

## 🔧 Configuration

### Environment Variables

Copy `.env.production` to `backend/.env` and update:

```bash
DEBUG=False
SECRET_KEY=your-super-secret-key
ALLOWED_HOSTS=your-domain.com,www.your-domain.com,localhost
DATABASE_URL=sqlite:///db.sqlite3  # or PostgreSQL URL
REDIS_URL=redis://localhost:6379/0
```

### Frontend Configuration

The frontend automatically detects the environment and configures WebSocket URLs accordingly:
- Development: `ws://localhost:8000/ws/chat/`
- Production: `wss://your-domain.com/ws/chat/`

## 🎯 Key Features

- **AI Rent Prediction**: ML-powered rent estimation
- **Tenant Screening**: AI-assisted tenant evaluation
- **Maintenance Alerts**: Predictive maintenance system
- **Real-time Chat**: WebSocket-based communication
- **Responsive UI**: Modern React/Next.js interface

## 📊 Services Overview

| Service | Port | Purpose |
|---------|------|---------|
| Next.js Frontend | 3000 | User interface |
| Django Backend | 8000 | API and WebSocket server |
| Redis | 6379 | Channel layers and caching |
| Nginx | 80/443 | Reverse proxy and static files |

## 🔍 Monitoring & Maintenance

### Health Checks
```bash
# Run health check
./health-check.sh

# Check individual services
sudo systemctl status daphne-landlord
pm2 status
sudo systemctl status nginx
```

### Logs
```bash
# Backend logs
sudo journalctl -u daphne-landlord -f

# Frontend logs
pm2 logs landlord-frontend

# Nginx logs
sudo tail -f /var/log/nginx/access.log
sudo tail -f /var/log/nginx/error.log
```

### Updates
```bash
# Update deployment
./update.sh

# Manual update
cd /srv/landlord-app
git pull origin main
cd backend && source venv/bin/activate
pip install -r ../requirements.txt
python manage.py migrate
python manage.py collectstatic --noinput
cd ../front && npm install && npm run build
sudo systemctl restart daphne-landlord
pm2 restart landlord-frontend
```

## 🔐 Security Considerations

1. **SSL/TLS**: Always use HTTPS in production
2. **Firewall**: Configure UFW to limit access
3. **Environment Variables**: Keep sensitive data in .env files
4. **Database**: Consider PostgreSQL for production
5. **Backup**: Regular database and file backups
6. **Updates**: Keep all dependencies updated

## 🐳 Docker Commands

```bash
# Start services
docker-compose up -d

# Stop services
docker-compose down

# View logs
docker-compose logs -f

# Update images
docker-compose pull
docker-compose up -d

# Database migrations
docker-compose exec backend python manage.py migrate

# Access container shell
docker-compose exec backend bash
docker-compose exec frontend sh
```

## 🔧 Troubleshooting

### Common Issues

1. **WebSocket Connection Failed**
   - Check Nginx WebSocket proxy configuration
   - Verify Daphne is running on port 8000
   - Check firewall rules

2. **Static Files Not Loading**
   - Run `python manage.py collectstatic`
   - Check Nginx static file configuration
   - Verify file permissions

3. **Database Connection Issues**
   - Check database configuration in settings
   - Verify database file permissions
   - Run migrations: `python manage.py migrate`

4. **Permission Denied Errors**
   - Ensure www-data owns application files
   - Check file permissions: `ls -la /srv/landlord-app`

### Performance Optimization

1. **Enable Redis**: Use Redis for channel layers and caching
2. **Database**: Consider PostgreSQL for better performance
3. **Static Files**: Use CDN for static file delivery
4. **Caching**: Implement application-level caching
5. **Load Balancing**: Use multiple backend instances for high traffic

## 📞 Support

For issues and questions:
1. Check the logs for error messages
2. Review the troubleshooting section
3. Ensure all dependencies are installed
4. Verify configuration files are correct

## 📝 License

This project is configured for deployment on Ubuntu servers with production-ready settings including:
- Security headers and HTTPS
- Process management with systemd and PM2
- Reverse proxy with Nginx
- Container deployment with Docker
- Automated deployment and update scripts

Choose the deployment method that best fits your infrastructure and requirements.
