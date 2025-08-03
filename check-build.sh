#!/bin/bash

echo "📊 Checking Next.js Build Structure"

cd /var/www/rentmind/front

echo "🔍 Current directory structure:"
ls -la

echo ""
echo "🔍 .next directory contents:"
ls -la .next/ 2>/dev/null || echo ".next directory not found"

echo ""
echo "🔍 .next/static contents:"
ls -la .next/static/ 2>/dev/null || echo ".next/static directory not found"

echo ""
echo "🔍 .next/server contents:"
ls -la .next/server/ 2>/dev/null || echo ".next/server directory not found"

echo ""
echo "🔍 out directory contents (if exists):"
ls -la out/ 2>/dev/null || echo "out directory not found"

echo ""
echo "🔍 Looking for HTML files:"
find .next/ -name "*.html" 2>/dev/null | head -10

echo ""
echo "🔍 Looking for main entry files:"
find . -maxdepth 3 -name "index.html" 2>/dev/null
find . -maxdepth 3 -name "page.html" 2>/dev/null

echo ""
echo "📊 Directory sizes:"
du -sh .next/ 2>/dev/null || echo ".next size: unknown"
du -sh out/ 2>/dev/null || echo "out size: unknown"
