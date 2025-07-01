#!/bin/bash
echo "🚀 Starting StockBot..."
echo "📁 Working directory: $(pwd)"
echo "🐍 Python version: $(python --version)"
echo "📦 Streamlit version: $(python -c 'import streamlit; print(streamlit.__version__)')"
echo "🔍 Checking file structure..."
ls -la
echo "📂 Checking src directory..."
ls -la src/
echo "🏁 Launching app..."
streamlit run app.py --server.port=7860 --server.address=0.0.0.0 --server.headless=true --server.enableCORS=false
