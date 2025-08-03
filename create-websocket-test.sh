#!/bin/bash

echo "🔍 WebSocket Connection Debugger"

cd /var/www/rentmind/front

# Create a simple HTML test page for WebSocket debugging
cat > websocket-test.html << 'EOF'
<!DOCTYPE html>
<html>
<head>
    <title>WebSocket Test</title>
    <style>
        body { font-family: Arial, sans-serif; padding: 20px; }
        .log { background: #f5f5f5; padding: 10px; margin: 10px 0; border-radius: 5px; }
        .error { background: #ffebee; color: #c62828; }
        .success { background: #e8f5e9; color: #2e7d32; }
        .info { background: #e3f2fd; color: #1565c0; }
    </style>
</head>
<body>
    <h1>WebSocket Connection Test</h1>
    <button onclick="testConnection()">Test WebSocket Connection</button>
    <button onclick="clearLogs()">Clear Logs</button>
    
    <div id="logs"></div>
    
    <script>
        let ws = null;
        
        function log(message, type = 'info') {
            const logs = document.getElementById('logs');
            const div = document.createElement('div');
            div.className = `log ${type}`;
            div.innerHTML = `[${new Date().toLocaleTimeString()}] ${message}`;
            logs.appendChild(div);
            logs.scrollTop = logs.scrollHeight;
        }
        
        function clearLogs() {
            document.getElementById('logs').innerHTML = '';
        }
        
        function testConnection() {
            if (ws) {
                ws.close();
                ws = null;
            }
            
            log('🔄 Attempting to connect to WebSocket...', 'info');
            
            const wsUrl = 'ws://srv889806.hstgr.cloud/ws/chat/';
            log(`📡 Connecting to: ${wsUrl}`, 'info');
            
            ws = new WebSocket(wsUrl);
            
            ws.onopen = function(event) {
                log('✅ WebSocket connected successfully!', 'success');
                log(`🔗 ReadyState: ${ws.readyState}`, 'success');
                
                // Send a test message
                ws.send(JSON.stringify({
                    type: 'text',
                    message: 'WebSocket test message'
                }));
                log('📤 Test message sent', 'info');
            };
            
            ws.onmessage = function(event) {
                log(`📥 Received: ${event.data}`, 'success');
            };
            
            ws.onerror = function(error) {
                log(`❌ WebSocket error: ${error}`, 'error');
                log(`🔗 ReadyState: ${ws.readyState}`, 'error');
            };
            
            ws.onclose = function(event) {
                log(`🔴 WebSocket closed. Code: ${event.code}, Reason: ${event.reason}`, 'error');
                log(`🔗 ReadyState: ${ws.readyState}`, 'error');
                
                if (event.code === 1006) {
                    log('💡 Code 1006 usually means connection failed or was abnormally closed', 'info');
                }
            };
            
            // Set a timeout to check connection status
            setTimeout(() => {
                if (ws.readyState === WebSocket.CONNECTING) {
                    log('⏰ Still connecting after 5 seconds...', 'error');
                } else if (ws.readyState === WebSocket.CLOSED) {
                    log('⏰ Connection closed within 5 seconds', 'error');
                }
            }, 5000);
        }
        
        // Auto-test on page load
        window.onload = function() {
            log('🚀 WebSocket Debugger loaded', 'info');
            log('🌐 Testing from: ' + window.location.origin, 'info');
        };
    </script>
</body>
</html>
EOF

echo "✅ Created WebSocket test page"
echo "🌐 Access it at: http://srv889806.hstgr.cloud/websocket-test.html"
echo ""
echo "📋 This will help debug:"
echo "   - Connection establishment"
echo "   - WebSocket readyState changes"
echo "   - Error codes and reasons"
echo "   - Message sending/receiving"
