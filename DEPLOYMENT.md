# 🚀 RentMind AI - Easy Ubuntu VPS Deployment Guide

**No PM2, Nginx, or systemd configuration hassles!**

## 🎯 Quick Start Options

### Option 1: Docker + Traefik (Recommended) ⭐
**Automatic SSL, load balancing, zero configuration**

```bash
# 1. Upload your project to VPS
scp -r RentMind-AI-for-Landlords user@your-server:/opt/

# 2. SSH into your server
ssh user@your-server

# 3. Run the deployment script
cd /opt/RentMind-AI-for-Landlords
chmod +x deploy.sh
./deploy.sh

# 4. Enter your domain and email when prompted
# 5. Update .env file with your OpenAI API key
# 6. Done! App available at https://yourdomain.com
```

**What it includes:**
- ✅ Automatic HTTPS/SSL certificates
- ✅ Load balancing and reverse proxy
- ✅ Redis for WebSockets
- ✅ Auto-restart on failure
- ✅ Docker container isolation
- ✅ One-command updates

### Option 2: Cloud Platforms (Easiest)

#### Railway.app (Recommended for beginners)
1. Push code to GitHub
2. Connect Railway to your repo
3. Deploy automatically
4. Add Redis service
5. Set environment variables

#### Render.com (Free tier available)
1. Push code to GitHub  
2. Import repo to Render
3. Use the included `render.yaml`
4. Deploy automatically

### Option 3: Simple Ubuntu Setup
**Uses Caddy for automatic HTTPS - no Nginx needed**

```bash
chmod +x deploy-simple.sh
./deploy-simple.sh
```

## 🔧 Environment Variables

Create `.env` file with:

```env
# Required
OPENAI_API_KEY=your_openai_api_key_here
DOMAIN=yourdomain.com
EMAIL=your@email.com

# Optional
DEBUG=False
ALLOWED_HOSTS=yourdomain.com,www.yourdomain.com
SECRET_KEY=your_secret_key_here
```

## 🔄 Updates & Maintenance

```bash
# Update your application
chmod +x update.sh
./update.sh

# View logs (Docker)
docker-compose logs -f

# View logs (Simple)
sudo journalctl -u rentmind-backend -f

# Restart services (Docker)
docker-compose restart

# Restart services (Simple)
sudo systemctl restart rentmind-backend
```

## 🌐 DNS Configuration

Point your domain to your server's IP:

```
A     @              YOUR_SERVER_IP
A     www            YOUR_SERVER_IP
```

## 🛠 Troubleshooting

### SSL Certificate Issues
```bash
# Check Traefik logs
docker-compose logs traefik

# Check Caddy logs  
sudo journalctl -u caddy -f
```

### Backend Issues
```bash
# Check backend logs
docker-compose logs backend

# Or for simple deployment
sudo journalctl -u rentmind-backend -f
```

### WebSocket Connection Issues
- Ensure your domain DNS is correctly pointed
- Check firewall allows ports 80, 443
- Verify SSL certificates are working

## 📊 Monitoring

### Health Checks
```bash
# Check all services
docker-compose ps

# Check system resources
htop
df -h
```

### Performance
- Frontend: Static files served by Caddy/Traefik
- Backend: Django with Daphne ASGI server
- WebSockets: Redis for scaling
- Database: SQLite (upgrade to PostgreSQL for production)

## 🔒 Security Features

- ✅ Automatic HTTPS/SSL
- ✅ Security headers configured
- ✅ CORS protection
- ✅ Container isolation (Docker)
- ✅ Non-root user execution

## 💰 Cost Comparison

| Option | Cost | Effort | Features |
|--------|------|--------|----------|
| Railway | $5-20/month | Minimal | Auto-scaling, monitoring |
| Render | Free-$25/month | Minimal | Free tier available |
| VPS + Docker | $5-10/month | Low | Full control |
| VPS Simple | $5-10/month | Medium | Direct Ubuntu setup |

## 🚀 Production Recommendations

1. **Use Docker deployment** for easiest management
2. **Set up monitoring** with UptimeRobot or similar
3. **Regular backups** of database and uploads
4. **Environment variables** for all sensitive data
5. **Domain with SSL** for production use

## 🆘 Need Help?

Common issues and solutions:

1. **"Cannot connect to Docker daemon"**
   ```bash
   sudo usermod -aG docker $USER
   # Logout and login again
   ```

2. **"Domain not resolving"**
   - Check DNS propagation: https://dnschecker.org
   - Wait up to 24 hours for DNS propagation

3. **"SSL certificate failed"**
   - Ensure domain points to server IP
   - Check ports 80, 443 are open
   - Wait 2-3 minutes for certificate generation

4. **"WebSocket connection failed"**
   - Verify backend is running: `docker-compose logs backend`
   - Check Redis: `docker-compose logs redis`
   - Ensure domain has proper SSL certificate

---

**🎉 That's it! Your RentMind AI application should now be running smoothly on your Ubuntu VPS with minimal configuration.**
