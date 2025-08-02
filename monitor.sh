#!/bin/bash

# Budget VPS monitoring script
# Helps you keep track of resource usage on low-cost hosting

echo "📊 RentMind AI - VPS Resource Monitor"
echo "======================================"

# System information
echo "🖥️  SYSTEM INFO:"
echo "   OS: $(lsb_release -d | cut -f2)"
echo "   Kernel: $(uname -r)"
echo "   Uptime: $(uptime -p)"
echo ""

# Memory usage
echo "💾 MEMORY USAGE:"
free -h | grep -E "Mem|Swap"
echo ""

# Disk usage
echo "💿 DISK USAGE:"
df -h / | tail -1 | awk '{print "   Root: " $3 " used / " $2 " total (" $5 " full)"}'
echo ""

# CPU usage
echo "🔥 CPU USAGE (last 1 min):"
echo "   Load Average: $(uptime | awk -F'load average:' '{print $2}' | awk '{print $1}' | sed 's/,//')"
echo ""

# Docker containers status
if command -v docker &> /dev/null; then
    echo "🐳 DOCKER CONTAINERS:"
    docker-compose ps 2>/dev/null || echo "   Docker Compose not running"
    echo ""
    
    echo "📈 CONTAINER RESOURCE USAGE:"
    if docker ps -q | wc -l | grep -q "^[1-9]"; then
        docker stats --no-stream --format "table {{.Name}}\t{{.CPUPerc}}\t{{.MemUsage}}\t{{.MemPerc}}"
    else
        echo "   No containers running"
    fi
    echo ""
fi

# Network usage
echo "🌐 NETWORK:"
if command -v vnstat &> /dev/null; then
    vnstat -i eth0 --oneline 2>/dev/null | tail -1 | awk -F';' '{print "   Today: " $4 " received, " $5 " sent"}'
else
    echo "   Install vnstat for network monitoring: sudo apt install vnstat"
fi
echo ""

# Application health check
echo "🏥 APPLICATION HEALTH:"
if curl -s -o /dev/null -w "%{http_code}" http://localhost:8000 | grep -q "200\|302"; then
    echo "   ✅ Backend: Running"
else
    echo "   ❌ Backend: Not responding"
fi

if curl -s -o /dev/null -w "%{http_code}" http://localhost:3000 | grep -q "200\|302"; then
    echo "   ✅ Frontend: Running"
else
    echo "   ❌ Frontend: Not responding"
fi
echo ""

# Recent errors in logs
echo "🚨 RECENT ERRORS (last 10):"
if [ -f "$(docker-compose config --services | head -1).log" ]; then
    docker-compose logs --tail=50 | grep -i error | tail -10 || echo "   No recent errors found"
else
    echo "   Run: docker-compose logs | grep error"
fi
echo ""

# Recommendations
echo "💡 RECOMMENDATIONS:"
MEMORY_USAGE=$(free | awk 'FNR==2{printf "%.0f", $3/($3+$4)*100}')
DISK_USAGE=$(df / | awk 'FNR==2{print $5}' | sed 's/%//')

if [ "$MEMORY_USAGE" -gt 80 ]; then
    echo "   ⚠️  Memory usage high ($MEMORY_USAGE%) - consider restarting containers"
fi

if [ "$DISK_USAGE" -gt 80 ]; then
    echo "   ⚠️  Disk usage high ($DISK_USAGE%) - clean up logs: docker system prune"
fi

if [ "$MEMORY_USAGE" -lt 50 ] && [ "$DISK_USAGE" -lt 50 ]; then
    echo "   ✅ System running efficiently!"
fi

echo ""
echo "🔄 To restart application: docker-compose restart"
echo "🧹 To clean up space: docker system prune -f"
echo "📋 To view live logs: docker-compose logs -f"
