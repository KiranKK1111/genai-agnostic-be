#!/bin/bash
# Quick start script for GenAI Dashboard Backend
echo "=================================="
echo "GenAI Dashboard Backend"
echo "=================================="

# Copy env if not exists
if [ ! -f .env ]; then
    cp .env.example .env
    echo "Created .env from .env.example — please edit with your PostgreSQL credentials"
fi

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements.txt -q

# Start server
echo "Starting server..."
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
