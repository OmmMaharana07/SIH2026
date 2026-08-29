"""
SatQuery AI — Production WSGI Entrypoint
"""

import os
import sys

# Ensure prototype root is in python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app import app

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    host = os.environ.get("HOST", "0.0.0.0")

    print(f"\n========================================================")
    print(f"  SatQuery AI Production WSGI Server (Waitress)")
    print(f"  Listening on: http://{host}:{port}")
    print(f"========================================================\n")

    from waitress import serve
    serve(app, host=host, port=port, threads=8)
