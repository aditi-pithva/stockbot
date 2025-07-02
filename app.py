# Main app entry point for Hugging Face Spaces
import sys
import os

# Add src directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Import and run the main app with UTF-8 encoding
with open(os.path.join(os.path.dirname(__file__), 'src', 'app.py'), 'r', encoding='utf-8') as f:
    exec(f.read())
