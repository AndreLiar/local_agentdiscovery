"""
API routes for the Local Discovery Agent
"""

import logging
from fastapi import APIRouter, HTTPException
from typing import Dict, Any

from app.models import (
    SearchRequest, SearchResponse, HealthResponse,
    ConversationHistory, MemoryInfo, MemoryTypesResponse, MemoryTypeInfo,
    AgentResetResponse, ModelsResponse, ModelInfo
)
from app.config import settings
from app.services import ollama_service

logger = logging.getLogger(__name__)

# Global agent instance (will be initialized in main.py)
agent_instance = None

def set_agent_instance(agent):
    """Set the global agent instance"""
    global agent_instance
    agent_instance = agent

# Create router
router = APIRouter()

@router.post("/search", response_model=SearchResponse)
async def search_places(request: SearchRequest) -> SearchResponse:
    """Search for places using the agent"""
    try:
        if not agent_instance:
            raise HTTPException(status_code=503, detail="Agent not initialized")
        
        logger.info(f"Search request: {request.query}")
        
        # Use the agent to search
        result = agent_instance.search_places(request.query, request.location)
        
        return SearchResponse(**result)
        
    except Exception as e:
        logger.error(f"Search error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """Health check endpoint"""
    try:
        # Check Ollama service
        ollama_status = "healthy" if ollama_service.is_running() else "unhealthy"
        
        # Check agent
        agent_status = "healthy" if agent_instance else "not_initialized"
        
        overall_status = "healthy" if ollama_status == "healthy" and agent_status == "healthy" else "unhealthy"
        
        return HealthResponse(
            status=overall_status,
            agent_status=f"ollama: {ollama_status}, agent: {agent_status}",
            version=settings.version
        )
        
    except Exception as e:
        logger.error(f"Health check error: {e}")
        return HealthResponse(
            status="unhealthy",
            agent_status=f"error: {str(e)}",
            version=settings.version
        )

@router.get("/conversation/history", response_model=ConversationHistory)
async def get_conversation_history() -> ConversationHistory:
    """Get conversation history"""
    try:
        if not agent_instance:
            raise HTTPException(status_code=503, detail="Agent not initialized")
        
        messages = agent_instance.get_conversation_history()
        
        return ConversationHistory(
            messages=messages,
            total_messages=len(messages)
        )
        
    except Exception as e:
        logger.error(f"Error getting conversation history: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/agent/reset", response_model=AgentResetResponse)
async def reset_agent() -> AgentResetResponse:
    """Reset agent and clear memory"""
    try:
        if not agent_instance:
            raise HTTPException(status_code=503, detail="Agent not initialized")
        
        agent_instance.clear_memory()
        
        return AgentResetResponse(
            success=True,
            message="Agent memory cleared successfully",
            memory_cleared=True
        )
        
    except Exception as e:
        logger.error(f"Error resetting agent: {e}")
        return AgentResetResponse(
            success=False,
            message=f"Error resetting agent: {str(e)}",
            memory_cleared=False
        )

@router.get("/agent/memory/info", response_model=MemoryInfo)
async def get_memory_info() -> MemoryInfo:
    """Get current memory information"""
    try:
        if not agent_instance:
            raise HTTPException(status_code=503, detail="Agent not initialized")
        
        memory_info = agent_instance.get_memory_info()
        
        return MemoryInfo(**memory_info)
        
    except Exception as e:
        logger.error(f"Error getting memory info: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/agent/memory/types", response_model=MemoryTypesResponse)
async def get_memory_types() -> MemoryTypesResponse:
    """Get available memory types"""
    memory_types = [
        MemoryTypeInfo(
            type="buffer",
            name="Buffer Memory",
            description="Stores all conversation messages",
            pros=["Complete conversation history", "Context-aware responses"],
            cons=["Memory usage grows over time", "May become slow with long conversations"]
        ),
        MemoryTypeInfo(
            type="window",
            name="Window Memory", 
            description=f"Stores last {settings.max_memory_messages} conversation messages",
            pros=["Fixed memory usage", "Good for long conversations"],
            cons=["Forgets older context", "May lose important information"]
        ),
        MemoryTypeInfo(
            type="summary",
            name="Summary Memory",
            description="Summarizes conversation history",
            pros=["Compact memory usage", "Retains key information"],
            cons=["May lose details", "Requires LLM for summarization"]
        )
    ]
    
    return MemoryTypesResponse(memory_types=memory_types)

@router.post("/agent/memory/switch")
async def switch_memory_type(memory_type: str):
    """Switch agent memory type"""
    try:
        if not agent_instance:
            raise HTTPException(status_code=503, detail="Agent not initialized")
        
        valid_types = ["buffer", "window", "summary"]
        if memory_type not in valid_types:
            raise HTTPException(
                status_code=400, 
                detail=f"Invalid memory type. Must be one of: {valid_types}"
            )
        
        agent_instance.switch_memory_type(memory_type)
        
        return {
            "success": True,
            "message": f"Switched to {memory_type} memory type",
            "memory_type": memory_type
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error switching memory type: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/models/available", response_model=ModelsResponse)
async def get_available_models() -> ModelsResponse:
    """Get available Ollama models"""
    try:
        models_data = ollama_service.get_models()
        
        models = []
        for model_data in models_data:
            model = ModelInfo(
                name=model_data.get("name", "unknown"),
                size=model_data.get("size"),
                modified_at=model_data.get("modified_at"),
                digest=model_data.get("digest")
            )
            models.append(model)
        
        return ModelsResponse(models=models)
        
    except Exception as e:
        logger.error(f"Error getting available models: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/config/debug")
async def get_config_debug():
    """Debug endpoint to show actual configuration values being used"""
    import os
    from app.config import settings
    
    return {
        "message": "Current runtime configuration values",
        "agent_settings": {
            "agent_temperature": {
                "value": settings.agent_temperature,
                "source": "env variable" if os.getenv("AGENT_TEMPERATURE") else "settings.py default"
            },
            "agent_timeout": {
                "value": settings.agent_timeout,
                "source": "env variable" if os.getenv("AGENT_TIMEOUT") else "settings.py default"
            },
            "max_memory_messages": {
                "value": settings.max_memory_messages,
                "source": "env variable" if os.getenv("MAX_MEMORY_MESSAGES") else "settings.py default"
            },
            "ollama_model": {
                "value": settings.ollama_model,
                "source": "env variable" if os.getenv("OLLAMA_MODEL") else "settings.py default"
            }
        },
        "environment_variables": {
            "AGENT_TEMPERATURE": os.getenv("AGENT_TEMPERATURE"),
            "AGENT_TIMEOUT": os.getenv("AGENT_TIMEOUT"),
            "MAX_MEMORY_MESSAGES": os.getenv("MAX_MEMORY_MESSAGES"),
            "OLLAMA_MODEL": os.getenv("OLLAMA_MODEL")
        },
        "settings_py_defaults": {
            "agent_temperature": 0.3,
            "agent_timeout": 180,
            "max_memory_messages": 10,
            "ollama_model": "llama3:latest"
        }
    }