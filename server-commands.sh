#!/bin/bash

# Simple deployment commands for Ubuntu server
# Run these commands one by one on your Ubuntu server

echo "🚀 RentMind AI - Ubuntu Deployment Commands"
echo "Run these commands on your Ubuntu server:"
echo ""

echo "1. Clone repository:"
echo "sudo mkdir -p /srv/landlord-app"
echo "sudo chown \$USER:\$USER /srv/landlord-app"
echo "cd /srv/landlord-app"
echo "git clone https://github.com/AbdullahRaoo/RentMind-AI-for-Landlords.git ."
echo ""

echo "2. Make scripts executable:"
echo "chmod +x deploy.sh update.sh docker-deploy.sh"
echo ""

echo "3. Deploy (choose one):"
echo "# Option A: Traditional deployment"
echo "./deploy.sh your-domain-or-ip"
echo ""
echo "# Option B: Docker deployment"  
echo "./docker-deploy.sh"
echo ""

echo "4. Check status:"
echo "sudo systemctl status daphne-landlord"
echo "pm2 status"
echo "sudo systemctl status nginx"
echo ""

echo "5. View logs:"
echo "sudo journalctl -u daphne-landlord -f"
echo "pm2 logs landlord-frontend"
echo ""

echo "6. Access your app:"
echo "http://your-domain-or-ip"
