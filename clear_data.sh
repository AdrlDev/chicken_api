#!/bin/bash

echo "🗑️  ChickenAI Clear Data Script"
echo "================================"

cd ~/chicken_api || exit 1

# Confirm before proceeding
read -p "⚠️  This will delete all dataset and training data. Are you sure? (y/N): " confirm
if [[ "$confirm" != "y" && "$confirm" != "Y" ]]; then
    echo "❌ Aborted."
    exit 0
fi

# Step 1: Clear inside Docker container
echo ""
echo "🐳 Clearing data inside Docker container..."
docker exec -it chickenapi rm -rf /app/runs/*
docker exec -it chickenapi rm -rf /app/dataset/images/*
docker exec -it chickenapi rm -rf /app/dataset/labels/*
docker exec -it chickenapi rm -f /app/dataset/notes.json
docker exec -it chickenapi rm -f /app/dataset/classes.txt
docker exec -it chickenapi rm -f /app/dataset/labels/train.cache
docker exec -it chickenapi rm -f /app/dataset/labels/val.cache
echo "✅ Docker data cleared."

# Step 2: Clear on host machine
echo ""
echo "🖥️  Clearing data on host machine..."
rm -rf ./runs/*
rm -rf ./dataset/images/*
rm -rf ./dataset/labels/*
rm -f ./dataset/notes.json
rm -f ./dataset/classes.txt
echo "✅ Host data cleared."

# Step 3: Clear public images
echo ""
echo "🌐 Clearing public images..."
rm -rf /var/www/dataset/images/*
echo "✅ Public images cleared."

echo ""
echo "✅ All data cleared successfully!"
