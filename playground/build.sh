#!/usr/bin/env bash
# Render build script for LLM Playground
# This script installs Node.js and builds the frontend

set -e

echo "=== Installing Node.js ==="
curl -fsSL https://deb.nodesource.com/setup_20.x | bash -
apt-get install -y nodejs

echo "=== Building Frontend ==="
cd frontend
npm install
npm run build
cd ..

echo "=== Installing Python Dependencies ==="
pip install -r requirements.txt
pip install groq huggingface_hub

echo "=== Build Complete ==="
