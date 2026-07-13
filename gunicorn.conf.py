import os

# Bind Gunicorn to the PORT environment variable set by Render (default to 5000 locally)
bind = f"0.0.0.0:{os.environ.get('PORT', '5000')}"
workers = 2
