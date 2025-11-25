#!/usr/bin/env python3
"""
Main entry point for the Local Discovery Agent backend
"""

import uvicorn
from app.core import create_app
from app.config import settings

def main():
    """Main function to run the application"""
    app = create_app()
    
    uvicorn.run(
        app,
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
        log_level=settings.log_level.lower()
    )

if __name__ == "__main__":
    main()