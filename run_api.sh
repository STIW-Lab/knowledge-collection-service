#!/bin/bash
# Startup script for FastAPI RAG service

echo "🚀 Starting Knowledge Collection RAG API..."
echo ""

# Check if .env exists
if [ ! -f .env ]; then
    echo "⚠️  Warning: .env file not found"
    echo "Please create a .env file with required variables:"
    echo "  - OPENAI_API_KEY"
    echo "  - POSTGRES_HOST, POSTGRES_PORT, POSTGRES_DB"
    echo "  - POSTGRES_USER, POSTGRES_PASSWORD"
    echo ""
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Default port
PORT=${PORT:-8000}

echo "Starting FastAPI on port $PORT..."
echo ""

# Run with uvicorn
uv run uvicorn api.main:app --host 0.0.0.0 --port $PORT --reload
