"""
Configuration settings for the Local Discovery Agent
"""

import os
from typing import Optional
from pydantic_settings import BaseSettings
from pydantic import Field
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Settings(BaseSettings):
    """Application settings"""
    
    # App Configuration
    app_name: str = "Local Discovery Agent"
    version: str = "1.0.0"
    debug: bool = Field(default=False, env="DEBUG")
    
    # Server Configuration
    host: str = Field(default="0.0.0.0", env="HOST")
    port: int = Field(default=8000, env="PORT")
    
    # Ollama Configuration
    ollama_host: str = Field(default="localhost", env="OLLAMA_HOST")
    ollama_port: int = Field(default=11434, env="OLLAMA_PORT")
    ollama_model: str = Field(default="llama3.2", env="OLLAMA_MODEL")
    
    # API Keys
    serp_api_key: Optional[str] = Field(default=None, env="SERP_API_KEY")
    mapbox_access_token: Optional[str] = Field(default=None, env="MAPBOX_ACCESS_TOKEN")
    
    # Agent Configuration
    max_memory_messages: int = Field(default=20, env="MAX_MEMORY_MESSAGES")
    agent_temperature: float = Field(default=0.1, env="AGENT_TEMPERATURE")
    agent_timeout: int = Field(default=120, env="AGENT_TIMEOUT")
    
    # Rate Limiting
    rate_limit_requests: int = Field(default=100, env="RATE_LIMIT_REQUESTS")
    rate_limit_window: int = Field(default=60, env="RATE_LIMIT_WINDOW")
    
    # Logging
    log_level: str = Field(default="INFO", env="LOG_LEVEL")
    log_file: Optional[str] = Field(default=None, env="LOG_FILE")
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

    @property
    def ollama_base_url(self) -> str:
        """Get the full Ollama base URL"""
        return f"http://{self.ollama_host}:{self.ollama_port}"

# Global settings instance
settings = Settings()