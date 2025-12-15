#!/bin/bash

# CSE475 Assignment 2 - GitHub Push Script
# This script will help you push to GitHub with proper authentication

echo "🚀 CSE475 Assignment 2 - GitHub Push"
echo "======================================"
echo ""
echo "Repository: https://github.com/ShahriarKhan016/CSE475-Assignment-2-SSL-Semi-"
echo ""
echo "⚠️  You will need to authenticate with GitHub"
echo ""
echo "Choose your authentication method:"
echo "1. Personal Access Token (Recommended)"
echo "2. SSH (if you have SSH keys set up)"
echo ""
read -p "Enter choice (1 or 2): " choice

if [ "$choice" = "1" ]; then
    echo ""
    echo "📝 To create a Personal Access Token:"
    echo "   1. Go to: https://github.com/settings/tokens"
    echo "   2. Click 'Generate new token (classic)'"
    echo "   3. Select 'repo' scope"
    echo "   4. Copy the token"
    echo ""
    echo "When prompted:"
    echo "   Username: ShahriarKhan016"
    echo "   Password: [Paste your token]"
    echo ""
    read -p "Press Enter when ready to push..."
    git push -u origin main
    
elif [ "$choice" = "2" ]; then
    echo ""
    echo "🔑 Switching to SSH..."
    git remote set-url origin git@github.com:ShahriarKhan016/CSE475-Assignment-2-SSL-Semi-.git
    git push -u origin main
else
    echo "Invalid choice. Exiting."
    exit 1
fi

echo ""
echo "✅ Done!"
echo ""
echo "Visit your repository:"
echo "https://github.com/ShahriarKhan016/CSE475-Assignment-2-SSL-Semi-"
