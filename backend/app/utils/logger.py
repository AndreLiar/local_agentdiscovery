"""
Logging utilities
"""

import logging
import sys
from typing import Optional

from app.config import settings

def setup_logging(log_level: Optional[str] = None, log_file: Optional[str] = None):
    """Setup application logging"""
    
    level = getattr(logging, (log_level or settings.log_level).upper(), logging.INFO)
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Setup root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    
    # Remove existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    
    # File handler if specified
    if log_file or settings.log_file:
        file_handler = logging.FileHandler(log_file or settings.log_file)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
    
    # Suppress some noisy loggers
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("requests").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)