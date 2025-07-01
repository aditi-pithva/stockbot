# Main app entry point for Hugging Face Spaces
import sys
import os

# Add src directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Import and run the main app
exec(open(os.path.join(os.path.dirname(__file__), 'src', 'app.py')).read())
