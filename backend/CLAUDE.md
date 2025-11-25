# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Commands

### Running the Application
- **Development**: `python main.py` - Starts the FastAPI server with auto-reload and debug mode
- **Production**: Set `DEBUG=false` and run `python main.py`
- **Docker Development**: `docker-compose --profile dev up backend-dev` - Runs with hot reload
- **Docker Production**: `docker-compose up backend` - Production deployment with Ollama

### Testing
- Tests not yet implemented. When added, use `pytest` to run tests.
- Install test dependencies first: `pip install pytest pytest-asyncio httpx`

### Dependencies
- Install: `pip install -r requirements.txt`
- Dependencies are managed via `requirements.txt` with specific versions for production stability

## Architecture Overview

This is a **Local Discovery Agent** backend built with FastAPI and LangChain that uses local LLMs via Ollama for AI-powered place discovery.

### Core Architecture Patterns

**Modular FastAPI Structure**: The codebase follows a clean, production-ready architecture:
- `app/agents/` - AI agent logic (discovery_agent.py, tools.py, prompts.py)
- `app/api/` - FastAPI routes and endpoints 
- `app/config/` - Pydantic settings management
- `app/core/` - Application factory and lifecycle
- `app/models/` - Pydantic schemas for type safety
- `app/services/` - External service integrations (Ollama)
- `app/utils/` - Logging and utilities

**Agent Architecture**: Uses LangChain's ReAct agent pattern with:
- **Memory Types**: Buffer, Window, or Summary memory for conversation context
- **Tools**: Place search via SerpAPI, geocoding via Mapbox
- **LLM**: Local models via Ollama (Llama3.2, Mixtral, Gemma2)

### Key Components

**LocalDiscoveryAgent** (`app/agents/discovery_agent.py`): Main agent class that handles:
- LLM initialization via Ollama
- Memory management (buffer/window/summary types)
- Tool integration and ReAct reasoning
- Place search with structured output parsing

**Settings** (`app/config/settings.py`): Environment-based configuration using Pydantic Settings:
- Ollama connection settings
- API keys for SerpAPI and Mapbox
- Agent parameters (temperature, timeout, memory limits)
- Server configuration

**API Routes** (`app/api/routes.py`): FastAPI endpoints with full type validation:
- `POST /search` - Main place search
- `GET /health` - Health checks for Ollama and agent
- Memory management endpoints (history, reset, switch types)
- Model information endpoints

## Development Guidelines

### Configuration
All configuration is environment-based via `app/config/settings.py`. Required environment variables:
- `OLLAMA_HOST`, `OLLAMA_PORT`, `OLLAMA_MODEL` - Ollama connection
- `SERP_API_KEY`, `MAPBOX_ACCESS_TOKEN` - API keys (optional but recommended)
- `DEBUG`, `LOG_LEVEL` - Development settings

### Adding New Features
- **New API endpoints**: Add to `app/api/routes.py` with proper Pydantic models
- **New agent tools**: Add to `app/agents/tools.py` following LangChain patterns
- **New data models**: Add to `app/models/schemas.py` with full type hints
- **New external services**: Add to `app/services/` with health check methods

### Memory Management
The agent supports three memory types switchable at runtime:
- **Buffer**: Stores all conversation history (default)
- **Window**: Stores last N messages (configurable via `MAX_MEMORY_MESSAGES`)
- **Summary**: Compresses history using LLM summarization

### Tool Integration
Agent tools use LangChain's tool decorator pattern. Current tools:
- **search_places**: SerpAPI integration for Google Local search
- **geocode_location**: Mapbox geocoding for location resolution

### Error Handling
- Structured error responses with proper HTTP status codes
- Comprehensive logging via `app/utils/logger.py`
- Graceful fallbacks for parsing agent responses (JSON → text patterns)

## Docker Integration

The project includes multi-stage Docker builds and docker-compose orchestration:
- **Production**: Optimized builds with dependency caching
- **Development**: Volume mounts for hot reload
- **Ollama Integration**: Automatic model downloading and health checks
- **GPU Support**: Configurable NVIDIA GPU support for Ollama

Use `docker-compose.yml` profiles:
- Default: Production backend + Ollama
- `dev` profile: Development backend with auto-reload

## Service Dependencies

**Critical**: Ollama must be running and have models downloaded before starting the agent.
- Ollama health checks are integrated into the application startup
- Models can be managed via the `/models/available` endpoint
- The agent will fail gracefully if Ollama is unavailable

**Optional**: SerpAPI and Mapbox improve search quality but the agent can function without them using fallback mechanisms.