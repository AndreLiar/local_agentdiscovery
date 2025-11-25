# Local Discovery Agent - Backend Architecture

## 📁 Project Structure

The backend has been restructured into a scalable, modular architecture following best practices for FastAPI applications:

```
backend/
├── app/                          # Main application package
│   ├── __init__.py              # Package initialization
│   ├── agents/                   # AI Agent logic
│   │   ├── __init__.py
│   │   ├── discovery_agent.py   # Main agent implementation
│   │   ├── prompts.py           # Agent prompts and templates
│   │   └── tools.py             # Agent tools (search, geocoding)
│   ├── api/                     # API routes and endpoints
│   │   ├── __init__.py
│   │   └── routes.py            # FastAPI route definitions
│   ├── config/                  # Configuration management
│   │   ├── __init__.py
│   │   └── settings.py          # Application settings (Pydantic)
│   ├── core/                    # Core application logic
│   │   ├── __init__.py
│   │   └── app.py               # FastAPI app factory
│   ├── models/                  # Data models and schemas
│   │   ├── __init__.py
│   │   └── schemas.py           # Pydantic models for API
│   ├── services/                # External service integrations
│   │   ├── __init__.py
│   │   └── ollama_service.py    # Ollama LLM service management
│   └── utils/                   # Utility functions
│       ├── __init__.py
│       └── logger.py            # Logging configuration
├── main_new.py                  # New application entry point
├── main.py                      # Legacy entry point (backup)
├── local_discovery_agent.py     # Legacy monolithic agent (backup)
├── requirements.txt             # Python dependencies
└── PROJECT_STRUCTURE.md         # This documentation
```

## 🏗️ Architecture Components

### 1. **app/agents/** - AI Agent Logic
- **`discovery_agent.py`**: Main `LocalDiscoveryAgent` class with memory management
- **`tools.py`**: LangChain tools for place search and geocoding
- **`prompts.py`**: Optimized prompts for local LLM interactions

### 2. **app/api/** - API Layer
- **`routes.py`**: All FastAPI endpoints (search, health, memory management, etc.)
- Clean separation of API logic from business logic

### 3. **app/config/** - Configuration
- **`settings.py`**: Centralized configuration using Pydantic Settings
- Environment variable management with defaults
- Type-safe configuration

### 4. **app/core/** - Application Core
- **`app.py`**: FastAPI application factory with lifespan management
- Centralized middleware setup (CORS, logging, etc.)
- Clean startup/shutdown procedures

### 5. **app/models/** - Data Models
- **`schemas.py`**: Pydantic models for request/response validation
- Type-safe API contracts
- Comprehensive data models for all endpoints

### 6. **app/services/** - External Services
- **`ollama_service.py`**: Ollama LLM server management
- Service abstraction for external dependencies
- Health checking and lifecycle management

### 7. **app/utils/** - Utilities
- **`logger.py`**: Centralized logging configuration
- Shared utility functions

## 🚀 Key Improvements

### **Scalability**
- **Modular Design**: Each component has a single responsibility
- **Dependency Injection**: Services can be easily swapped or mocked
- **Configuration Management**: Environment-based configuration
- **Clean Architecture**: Separation of concerns across layers

### **Maintainability**
- **Type Safety**: Pydantic models for all data structures
- **Error Handling**: Centralized error handling with proper HTTP status codes
- **Logging**: Structured logging throughout the application
- **Code Organization**: Logical grouping of related functionality

### **Production Ready**
- **Lifecycle Management**: Proper startup/shutdown procedures
- **Health Checks**: Comprehensive health monitoring
- **Environment Configuration**: Production-ready settings management
- **Resource Management**: Proper cleanup of resources

### **Developer Experience**
- **FastAPI Integration**: Automatic API documentation
- **Type Hints**: Full type coverage for better IDE support
- **Modular Testing**: Easy to test individual components
- **Hot Reload**: Development mode with automatic reloading

## 🔧 Usage

### **Development**
```bash
# Install dependencies
pip install -r requirements.txt

# Run development server
python main_new.py
```

### **Production**
```bash
# Set environment variables
export OLLAMA_MODEL=llama3.2
export SERP_API_KEY=your_key_here
export DEBUG=false

# Run production server
python main_new.py
```

## 📊 Configuration

All configuration is managed through environment variables in `app/config/settings.py`:

```env
# Server Configuration
HOST=0.0.0.0
PORT=8000
DEBUG=false

# Ollama Configuration  
OLLAMA_HOST=localhost
OLLAMA_PORT=11434
OLLAMA_MODEL=llama3.2

# API Keys
SERP_API_KEY=your_serp_api_key
MAPBOX_ACCESS_TOKEN=your_mapbox_token

# Agent Configuration
MAX_MEMORY_MESSAGES=20
AGENT_TEMPERATURE=0.1
AGENT_TIMEOUT=120

# Logging
LOG_LEVEL=INFO
LOG_FILE=app.log
```

## 🧪 API Endpoints

The restructured API maintains all existing endpoints:

- `POST /search` - Place search
- `GET /health` - Health check
- `GET /conversation/history` - Conversation history
- `POST /agent/reset` - Reset agent memory
- `GET /agent/memory/info` - Memory information
- `GET /agent/memory/types` - Available memory types
- `POST /agent/memory/switch` - Switch memory type
- `GET /models/available` - Available Ollama models

## 🔄 Migration

The new structure is **backward compatible**. The old `main.py` and `local_discovery_agent.py` files are preserved as backups. To migrate:

1. **Test the new structure**: `python main_new.py`
2. **Update deployment scripts** to use `main_new.py`
3. **Remove legacy files** once migration is complete

## 🎯 Benefits

1. **Easier Testing**: Each module can be tested in isolation
2. **Better Debugging**: Clear separation makes issues easier to trace  
3. **Team Collaboration**: Multiple developers can work on different modules
4. **Feature Extensions**: New features can be added without affecting existing code
5. **Production Deployment**: Better suited for containerization and scaling