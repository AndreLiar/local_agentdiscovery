"""
FastAPI application factory
"""

import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from app.utils import setup_logging
from app.services import ollama_service
from app.agents import LocalDiscoveryAgent
from app.api import router
from app.api.routes import set_agent_instance

logger = logging.getLogger(__name__)

# Global agent instance
agent_instance = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    global agent_instance
    
    # Startup
    logger.info("Starting Local Discovery Agent application...")
    
    try:
        # Start Ollama service
        if not ollama_service.start():
            logger.error("Failed to start Ollama service")
            raise RuntimeError("Ollama service initialization failed")
        
        # Initialize agent
        logger.info("Initializing Local Discovery Agent...")
        agent_instance = LocalDiscoveryAgent()
        
        # Set agent instance in routes
        set_agent_instance(agent_instance)
        
        logger.info("Agent initialized successfully")
        
    except Exception as e:
        logger.error(f"Failed to initialize application: {e}")
        raise
    
    yield
    
    # Shutdown
    logger.info("Shutting down Local Discovery Agent application...")
    
    try:
        # Stop Ollama service
        ollama_service.stop()
        logger.info("Application shutdown complete")
        
    except Exception as e:
        logger.error(f"Error during shutdown: {e}")

def create_app() -> FastAPI:
    """Create and configure FastAPI application"""
    
    # Setup logging
    setup_logging()
    
    # Create FastAPI app
    app = FastAPI(
        title=settings.app_name,
        version=settings.version,
        description="Production-grade API for local LLM-powered place discovery",
        docs_url="/docs" if settings.debug else None,
        redoc_url="/redoc" if settings.debug else None,
        lifespan=lifespan
    )
    
    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Configure appropriately for production
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Include API routes
    app.include_router(router)
    
    # Add root endpoint
    @app.get("/")
    async def root():
        return {
            "message": f"Welcome to {settings.app_name}",
            "version": settings.version,
            "status": "running"
        }
    
    logger.info(f"FastAPI application created - {settings.app_name} v{settings.version}")
    
    return app