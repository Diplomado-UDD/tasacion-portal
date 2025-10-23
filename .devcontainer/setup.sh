#!/bin/bash

# Codespaces automatic setup script
echo "🚀 Setting up Property Price Analysis environment..."

# Install uv
echo "📦 Installing uv package manager..."
curl -LsSf https://astral.sh/uv/install.sh | sh

# Add uv to PATH for current session
export PATH="$HOME/.local/bin:$PATH"

# Install project dependencies
echo "📚 Installing project dependencies..."
uv sync

# Create data directories
echo "📁 Creating data directories..."
mkdir -p data/raw data/processed outputs/data outputs/plots outputs/reports

echo "✅ Setup complete! You can now run:"
echo "   python run.py"
echo ""
echo "📖 See README.md for more information"
