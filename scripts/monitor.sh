#!/bin/bash

echo "🔍 System Monitoring - Press Ctrl+C to exit"
echo "=========================================="

while true; do
    clear
    echo "🖥️  GPU Status:"
    nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits
    echo ""
    
    echo "🐍 Python Processes:"
    ps aux | grep python | grep -v grep | head -5
    echo ""
    
    echo "🌐 Server Status:"
    curl -s http://localhost:8000/health | jq . 2>/dev/null || echo "Server not responding"
    echo ""
    
    echo "📊 Last 5 log lines:"
    tail -5 logs/server.log 2>/dev/null || echo "No logs yet"
    
    sleep 5
done