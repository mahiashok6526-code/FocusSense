"""
FocusSense Vercel Serverless Function WSGI Entrypoint
=====================================================
This module acts solely as the WSGI bridge between Vercel Serverless Functions
and the existing FocusSense Flask application in app.py.

It does NOT re-create the Flask app. The root app.py remains the single
source of truth for application configuration, routes, and intelligence engines.
"""

import sys
import os

# Reliably locate project root (parent directory of api/) and add to sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Import and expose the existing Flask application instance
from app import app
