#!/bin/bash
set -e

if [ "$ROLE" = "api" ]; then
    echo "Starting RAG API..."
    # use --loop asyncio to prevent uvloop/nest_asyncio crash
    exec uvicorn api:app --host 0.0.0.0 --port 8000 --workers 1 --loop asyncio

elif [ "$ROLE" = "dashboard" ]; then
    echo "Starting Streamlit Dashboard..."
    exec streamlit run dashboard.py --server.port=8501 --server.address=0.0.0.0

else
    echo "ERROR: No ROLE specified. Please set ROLE=api or ROLE=dashboard."
    exit 1
fi