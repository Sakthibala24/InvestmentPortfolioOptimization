#!/usr/bin/env python3
"""
Simple script to run the Flask backend server
"""
import os
import sys
from pathlib import Path

# Add the backend directory to Python path
backend_dir = Path(__file__).parent
sys.path.insert(0, str(backend_dir))

# Set environment variables
os.environ.setdefault('FLASK_ENV', 'development')
os.environ.setdefault('FLASK_DEBUG', '1')

try:
    from app import app
    
    if __name__ == '__main__':
        print("🚀 Starting Portfolio Optimization Backend...")
        print("📊 Backend will be available at: http://localhost:5000")
        print("🔗 Frontend should connect automatically")
        print("⚡ Press Ctrl+C to stop the server")
        print("-" * 50)
        
        app.run(
            host='0.0.0.0',
            port=5000,
            debug=True,
            use_reloader=True
        )
        
except ImportError as e:
    print(f"❌ Error importing Flask app: {e}")
    print("💡 Make sure you have installed the requirements:")
    print("   pip install -r requirements.txt")
    sys.exit(1)
except Exception as e:
    print(f"❌ Error starting server: {e}")
    sys.exit(1)