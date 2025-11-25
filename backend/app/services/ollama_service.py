"""
Ollama service for managing local LLM interactions
"""

import logging
import subprocess
import requests
import time
from typing import Optional

from app.config import settings

logger = logging.getLogger(__name__)

class OllamaService:
    """Service for managing Ollama LLM server"""
    
    def __init__(self):
        self.process: Optional[subprocess.Popen] = None
        self.base_url = settings.ollama_base_url
        
    def is_running(self) -> bool:
        """Check if Ollama service is running"""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=2)
            return response.status_code == 200
        except Exception:
            return False
    
    def start(self) -> bool:
        """Start Ollama service as a subprocess"""
        try:
            if self.is_running():
                logger.info("Ollama service is already running")
                return True
            
            logger.info("Starting Ollama service...")
            self.process = subprocess.Popen(
                ["ollama", "serve"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            # Wait for service to start (max 30 seconds)
            for _ in range(30):
                if self.is_running():
                    logger.info("Ollama service started successfully")
                    return True
                time.sleep(1)
            
            logger.error("Ollama service failed to start within timeout")
            return False
            
        except Exception as e:
            logger.error(f"Failed to start Ollama service: {e}")
            return False
    
    def stop(self):
        """Stop Ollama service"""
        if self.process:
            logger.info("Stopping Ollama service...")
            self.process.terminate()
            try:
                self.process.wait(timeout=10)
                logger.info("Ollama service stopped successfully")
            except subprocess.TimeoutExpired:
                logger.warning("Ollama service didn't stop gracefully, forcing shutdown")
                self.process.kill()
            self.process = None
    
    def get_models(self) -> list:
        """Get list of available models"""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            if response.status_code == 200:
                data = response.json()
                return data.get("models", [])
            return []
        except Exception as e:
            logger.error(f"Failed to get models: {e}")
            return []
    
    def pull_model(self, model_name: str) -> bool:
        """Pull a model from Ollama registry"""
        try:
            logger.info(f"Pulling model: {model_name}")
            response = requests.post(
                f"{self.base_url}/api/pull",
                json={"name": model_name},
                timeout=300  # 5 minutes timeout for model download
            )
            return response.status_code == 200
        except Exception as e:
            logger.error(f"Failed to pull model {model_name}: {e}")
            return False

# Global instance
ollama_service = OllamaService()