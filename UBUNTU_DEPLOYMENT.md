# Quick Ubuntu Server Deployment Guide

## 📋 What You Have Ready:
- ✅ Django backend with Daphne
- ✅ Next.js frontend 
- ✅ Deployment scripts created
- ✅ Docker configuration ready
- ✅ Production settings configured

## 🚀 Deployment Steps on Ubuntu Server:

### Step 1: Push to GitHub (Do this first on Windows)
```bash
git add .
git commit -m "Add production deployment configuration"
git push origin main
```

### Step 2: On Your Ubuntu Server
```bash
# 1. Clone your repository
sudo mkdir -p /srv/landlord-app
sudo chown $USER:$USER /srv/landlord-app
cd /srv/landlord-app
git clone https://github.com/AbdullahRaoo/RentMind-AI-for-Landlords.git .

# 2. Make scripts executable
chmod +x deploy.sh update.sh docker-deploy.sh

# 3. Edit the domain in deploy.sh (replace 'your-domain.com' with your actual domain/IP)
nano deploy.sh  # Change DOMAIN="your-domain.com" to your actual domain

# 4. Run deployment
./deploy.sh
```

### Step 3: Configure Domain (Optional)
If you have a domain, update it in the deploy.sh file before running:
```bash
DOMAIN="your-actual-domain.com"  # Replace with your domain
```

If you don't have a domain, use your server IP:
```bash
DOMAIN="your-server-ip"  # Replace with your server IP
```

## 🎯 Two Deployment Options:

### Option A: Traditional (Recommended)
```bash
./deploy.sh
```
This sets up:
- Django with Daphne (port 8000)
- Next.js with PM2 (port 3000)  
- Nginx reverse proxy (port 80/443)
- Redis for WebSocket channels

### Option B: Docker
```bash
./docker-deploy.sh
```
This sets up everything in containers.

## 🔧 What the Scripts Do:

1. **Install dependencies** (Python, Node.js, Redis, Nginx)
2. **Set up virtual environment** for Django
3. **Install Python packages** from requirements.txt
4. **Build Next.js frontend**
5. **Configure services** (systemd for backend, PM2 for frontend)
6. **Set up Nginx** as reverse proxy
7. **Configure firewall**

## 🌐 Access Your App:
- **Frontend**: http://your-domain-or-ip
- **Backend API**: http://your-domain-or-ip/api/
- **WebSocket**: ws://your-domain-or-ip/ws/
- **Django Admin**: http://your-domain-or-ip/admin/

## 🔍 Check Status:
```bash
# Check if services are running
sudo systemctl status daphne-landlord  # Django backend
pm2 status                             # Next.js frontend  
sudo systemctl status nginx            # Web server

# View logs
sudo journalctl -u daphne-landlord -f  # Backend logs
pm2 logs landlord-frontend             # Frontend logs
```

## 🚨 Important Notes:
- Make sure your server has at least 2GB RAM
- Ports 80, 443, 3000, 8000 should be open
- Update your domain DNS to point to server IP
- For SSL: run `sudo certbot --nginx -d your-domain.com` after deployment
