"""
Pydantic models for request/response schemas
"""

from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from enum import Enum

class MemoryType(str, Enum):
    """Available memory types for the agent"""
    BUFFER = "buffer"
    WINDOW = "window"
    SUMMARY = "summary"

class SearchRequest(BaseModel):
    """Request model for place search"""
    query: str = Field(..., description="Search query for places")
    location: Optional[str] = Field(default=None, description="Optional location context")

class PlaceResult(BaseModel):
    """Model for a single place result"""
    name: Optional[str] = Field(default="Unknown", description="Place name")
    rating: Optional[float] = None
    address: Optional[str] = None
    coordinates: Optional[List[float]] = Field(default=None, description="[longitude, latitude]")
    distance_km: Optional[float] = None
    description: Optional[str] = None
    price: Optional[str] = None
    reviews: Optional[int] = None
    hours: Optional[str] = None
    type: Optional[str] = None
    phone: Optional[str] = None
    website: Optional[str] = None

class SearchResponse(BaseModel):
    """Response model for place search"""
    success: bool
    response: Optional[str] = None
    places: List[PlaceResult] = []
    error: Optional[str] = None
    query: str
    processing_time: Optional[float] = None

class HealthResponse(BaseModel):
    """Response model for health check"""
    status: str
    agent_status: str
    version: str
    timestamp: Optional[str] = None

class ConversationMessage(BaseModel):
    """Model for conversation history"""
    role: str = Field(..., description="Role: 'user' or 'assistant'")
    content: str = Field(..., description="Message content")
    timestamp: Optional[str] = None

class ConversationHistory(BaseModel):
    """Response model for conversation history"""
    messages: List[ConversationMessage] = []
    total_messages: int = 0

class MemoryInfo(BaseModel):
    """Response model for memory information"""
    type: str
    description: str
    message_count: int

class MemoryTypeInfo(BaseModel):
    """Information about a memory type"""
    type: str
    name: str
    description: str
    pros: List[str] = []
    cons: List[str] = []

class MemoryTypesResponse(BaseModel):
    """Response model for available memory types"""
    memory_types: List[MemoryTypeInfo] = []

class AgentResetResponse(BaseModel):
    """Response model for agent reset"""
    success: bool
    message: str
    memory_cleared: bool = False

class ModelInfo(BaseModel):
    """Information about an available model"""
    name: str
    size: Optional[str] = None
    modified_at: Optional[str] = None
    digest: Optional[str] = None

class ModelsResponse(BaseModel):
    """Response model for available models"""
    models: List[ModelInfo] = []