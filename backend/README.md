# 🤖 Local Discovery Agent - Restructured Backend

A **production-grade, scalable backend** for local AI-powered place discovery using:

- **Local LLM** via Ollama (Llama3.2, Mixtral, Gemma2, etc.)
- **SerpAPI** for real Google Local search results  
- **Mapbox** for geocoding and mapping
- **LangChain** architecture with conversational memory
- **FastAPI** with modular, enterprise-ready structure
- **Docker** support for easy deployment

## 🏗️ Architecture

The backend follows a **clean, modular architecture** with clear separation of concerns:

```
backend/
├── app/                          # Main application package
│   ├── agents/                   # AI Agent logic
│   │   ├── discovery_agent.py    # Main agent implementation  
│   │   ├── tools.py              # LangChain tools (search, geocoding)
│   │   └── prompts.py            # Optimized prompts for local LLMs
│   ├── api/                      # API routes and endpoints
│   │   └── routes.py             # FastAPI route definitions
│   ├── config/                   # Configuration management
│   │   └── settings.py           # Environment-based settings
│   ├── core/                     # Core application logic
│   │   └── app.py                # FastAPI app factory
│   ├── models/                   # Data models and schemas
│   │   └── schemas.py            # Pydantic models for API
│   ├── services/                 # External service integrations
│   │   └── ollama_service.py     # Ollama LLM service management
│   └── utils/                    # Utility functions
│       └── logger.py             # Logging configuration
├── main.py                       # Application entry point
├── requirements.txt              # Python dependencies
├── Dockerfile                    # Container configuration
├── docker-compose.yml            # Multi-service orchestration
└── PROJECT_STRUCTURE.md          # Detailed architecture docs
```

## 🚀 Quick Start

### **Development Mode**

1. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Set Environment Variables** (optional)
   ```bash
   export SERP_API_KEY=your_serp_api_key
   export MAPBOX_ACCESS_TOKEN=your_mapbox_token
   export OLLAMA_MODEL=llama3.2
   ```

3. **Run the Server**
   ```bash
   python main.py
   ```
   
   Server will start at `http://localhost:8000`

### **Production with Docker**

1. **Production Mode**
   ```bash
   docker-compose up backend
   ```

2. **Development Mode with Auto-reload**
   ```bash
   docker-compose --profile dev up backend-dev
   ```

## 📊 API Endpoints

- **`POST /search`** - Search for places
- **`GET /health`** - Health check
- **`GET /conversation/history`** - Get conversation history  
- **`POST /agent/reset`** - Reset agent memory
- **`GET /agent/memory/info`** - Memory information
- **`POST /agent/memory/switch`** - Switch memory type
- **`GET /models/available`** - Available Ollama models

**API Documentation**: Visit `http://localhost:8000/docs` when running

## ⚙️ Configuration

All settings are managed via environment variables in `app/config/settings.py`:

```env
# Server Configuration
HOST=0.0.0.0
PORT=8000
DEBUG=false

# Ollama Configuration  
OLLAMA_HOST=localhost
OLLAMA_PORT=11434
OLLAMA_MODEL=llama3.2

# API Keys (optional but recommended)
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

## 🎯 Key Features

### **Production Ready**
- ✅ Modular, scalable architecture
- ✅ Type-safe with Pydantic models
- ✅ Comprehensive error handling
- ✅ Structured logging
- ✅ Health monitoring
- ✅ Docker containerization
- ✅ Environment-based configuration

### **AI Agent Capabilities**
- 🧠 Local LLM reasoning via Ollama
- 🔍 Real-time place search via SerpAPI
- 🗺️ Geocoding via Mapbox
- 💭 Conversational memory (buffer/window/summary)
- 🔧 Tool-based architecture with LangChain

### **Developer Experience**
- 🚀 Hot reload in development
- 📚 Automatic API documentation
- 🧪 Easy testing and debugging
- 📝 Full type hints and validation
- 🏗️ Clean separation of concerns

## 🛠️ Development

### **Project Structure**
See `PROJECT_STRUCTURE.md` for detailed architecture documentation.

### **Adding New Features**
1. **New API endpoints**: Add to `app/api/routes.py`
2. **New agent tools**: Add to `app/agents/tools.py`
3. **New data models**: Add to `app/models/schemas.py`
4. **New services**: Add to `app/services/`

### **Testing**
```bash
# Install test dependencies
pip install pytest pytest-asyncio httpx

# Run tests (when test suite is added)
pytest
```

## 🔄 Migration from Legacy

This is a **complete restructure** of the original monolithic backend. The old structure has been refactored into this modular architecture while maintaining **100% API compatibility**.

**What Changed:**
- ❌ **Removed**: `local_discovery_agent.py` (monolithic file)
- ❌ **Removed**: Old `main.py` (mixed concerns)
- ✅ **Added**: Modular `app/` package structure
- ✅ **Added**: Docker support
- ✅ **Added**: Production-ready configuration

**API Compatibility**: All existing endpoints work exactly the same!

## 📈 Benefits

1. **🚀 Scalability**: Each component can be scaled independently
2. **🧪 Testability**: Each module can be tested in isolation  
3. **👥 Team Collaboration**: Multiple developers can work on different modules
4. **🔧 Maintainability**: Clear separation makes debugging easier
5. **📦 Deployment**: Ready for containerization and cloud deployment
6. **🔒 Production**: Enterprise-grade error handling and logging

## 🤝 Contributing

1. Follow the modular architecture patterns
2. Add type hints for all functions
3. Update documentation for new features
4. Test your changes thoroughly

---

**Powered by**: FastAPI, LangChain, Ollama, SerpAPI, Mapbox