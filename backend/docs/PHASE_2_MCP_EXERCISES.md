# 🔧 Phase 2: MCP + RAG Hands-On Exercises
*Building Real-Time AI Agent with Model Context Protocol*

## 🎯 **Learning Objectives**

By completing these exercises, you'll master:
- **RAG Integration**: Local knowledge base for cultural insights
- **MCP Protocol**: Real-time data access through standardized tools
- **Tool Chaining**: Intelligent combination of multiple data sources
- **Performance Optimization**: Parallel execution and caching

---

## 📚 **Exercise 1: RAG Knowledge Base Setup**

### **🎯 Objective**
Build a local knowledge base with Paris restaurant insights.

### **Step-by-Step Implementation**

#### **1. Install Dependencies (10 min)**
```bash
# Add to requirements.txt
pip install chromadb sentence-transformers langchain-chroma

# Create data directories
mkdir -p data/vectorstore
mkdir -p data/knowledge
```

#### **2. Create Knowledge Content (15 min)**
```python
# File: data/knowledge/paris_dining.json
{
  "dining_etiquette": [
    "Always say 'Bonjour' when entering a restaurant",
    "Don't ask for substitutions - French chefs take pride in their recipes",
    "Tipping 5-10% is appreciated but not mandatory",
    "Ask for 'une carafe d'eau' for free water",
    "Lunch: 12:00-14:00, Dinner: 19:30-22:00 (many close between)"
  ],
  "neighborhood_insights": {
    "Marais": "Historic Jewish quarter, trendy restaurants, many open Sundays",
    "Saint-Germain": "Classic bistros, expensive, tourist-heavy around cafés",
    "Montmartre": "Avoid tourist traps near Sacré-Cœur, locals eat down the hill",
    "Belleville": "Multicultural, affordable, authentic ethnic food",
    "Latin Quarter": "Student-friendly prices, cramped but authentic bistros"
  },
  "timing_tips": [
    "Book popular restaurants 2-3 days ahead",
    "Many restaurants close Sundays and/or Mondays", 
    "Avoid 15:00-18:00 when most places are closed",
    "French dine late - 19:30 earliest for dinner",
    "Last orders usually 30 minutes before closing"
  ]
}
```

#### **3. Build RAG Service (30 min)**
```python
# File: app/services/rag_service.py

import json
from pathlib import Path
from langchain_chroma import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings

class ParisRAGService:
    def __init__(self):
        self.embeddings = OllamaEmbeddings(
            model="nomic-embed-text",
            base_url="http://localhost:11434"
        )
        
        self.vectorstore = Chroma(
            persist_directory="./data/vectorstore",
            embedding_function=self.embeddings
        )
        
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=300,
            chunk_overlap=50,
            separators=["\n\n", "\n", ".", "!"]
        )
        
        # Load knowledge on startup
        self._load_paris_knowledge()
    
    def _load_paris_knowledge(self):
        """Load curated Paris dining knowledge"""
        knowledge_file = Path("data/knowledge/paris_dining.json")
        
        if not knowledge_file.exists():
            print("⚠️ Knowledge file not found, creating sample data...")
            self._create_sample_knowledge()
        
        with open(knowledge_file, 'r', encoding='utf-8') as f:
            knowledge_data = json.load(f)
        
        # Convert to text chunks
        texts = self._convert_to_text_chunks(knowledge_data)
        
        # Add to vector store
        if texts:
            self.vectorstore.add_texts(texts)
            print(f"✅ Loaded {len(texts)} knowledge chunks")
    
    def _convert_to_text_chunks(self, knowledge_data: dict) -> list:
        """Convert structured knowledge to searchable text"""
        chunks = []
        
        # Dining etiquette
        for tip in knowledge_data.get("dining_etiquette", []):
            chunks.append(f"Dining Etiquette: {tip}")
        
        # Neighborhood insights
        for neighborhood, info in knowledge_data.get("neighborhood_insights", {}).items():
            chunks.append(f"{neighborhood} Neighborhood: {info}")
        
        # Timing tips
        for tip in knowledge_data.get("timing_tips", []):
            chunks.append(f"Timing Advice: {tip}")
        
        return chunks
    
    def query_knowledge(self, query: str, k: int = 3) -> list:
        """Query the knowledge base"""
        try:
            docs = self.vectorstore.similarity_search_with_score(query, k=k)
            
            relevant_knowledge = []
            for doc, score in docs:
                if score < 0.8:  # Only high relevance (lower score = more similar)
                    relevant_knowledge.append({
                        "text": doc.page_content,
                        "relevance_score": round(1 - score, 3)  # Convert to 0-1 scale
                    })
            
            return relevant_knowledge
            
        except Exception as e:
            print(f"❌ RAG query error: {e}")
            return []
```

#### **4. Test RAG System (15 min)**
```python
# File: test_rag.py

from app.services.rag_service import ParisRAGService

def test_rag_queries():
    """Test RAG system with various queries"""
    rag = ParisRAGService()
    
    test_queries = [
        "What should I know about dining etiquette in Paris?",
        "Best neighborhood for authentic food",
        "When should I make restaurant reservations?",
        "What time do restaurants serve dinner?"
    ]
    
    for query in test_queries:
        print(f"\n🔍 Query: {query}")
        results = rag.query_knowledge(query, k=2)
        
        for i, result in enumerate(results, 1):
            print(f"  {i}. {result['text']} (relevance: {result['relevance_score']})")

if __name__ == "__main__":
    test_rag_queries()
```

---

## 🌐 **Exercise 2: MCP Server Development**

### **🎯 Objective**
Build an MCP server that provides real-time restaurant data.

### **Step-by-Step Implementation**

#### **1. Install MCP Dependencies (5 min)**
```bash
pip install mcp anthropic-mcp-tools httpx beautifulsoup4 aiohttp
```

#### **2. Create Basic MCP Server (45 min)**
```python
# File: app/mcp/restaurant_server.py

import asyncio
import json
import aiohttp
from typing import Any, List
from mcp.server.models import InitializeResult
from mcp.server.server import NotificationOptions, Server
from mcp.server.stdio import stdio_server
from mcp.types import (
    CallToolRequest, CallToolResult, ListToolsRequest, ListToolsResult,
    Tool, TextContent, ImageContent, EmbeddedResource
)

class RestaurantMCPServer:
    def __init__(self):
        self.server = Server("restaurant-discovery")
        self._setup_handlers()
    
    def _setup_handlers(self):
        """Setup MCP request handlers"""
        
        @self.server.list_tools()
        async def handle_list_tools() -> List[Tool]:
            """List available MCP tools"""
            return [
                Tool(
                    name="get_restaurant_hours",
                    description="Get real-time restaurant opening hours and current status",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "restaurant_name": {
                                "type": "string", 
                                "description": "Name of the restaurant"
                            },
                            "location": {
                                "type": "string",
                                "description": "Location/address for context"
                            }
                        },
                        "required": ["restaurant_name"]
                    }
                ),
                Tool(
                    name="get_current_reviews",
                    description="Get latest customer reviews from multiple platforms",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "restaurant_name": {"type": "string"},
                            "limit": {
                                "type": "integer", 
                                "default": 5,
                                "description": "Number of reviews to fetch"
                            }
                        },
                        "required": ["restaurant_name"]
                    }
                ),
                Tool(
                    name="check_menu_prices",
                    description="Get current menu items and pricing information",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "restaurant_name": {"type": "string"},
                            "cuisine_type": {"type": "string", "description": "Optional filter"}
                        },
                        "required": ["restaurant_name"]
                    }
                ),
                Tool(
                    name="check_reservations",
                    description="Check table availability and reservation options",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "restaurant_name": {"type": "string"},
                            "date": {"type": "string", "description": "Date in YYYY-MM-DD format"},
                            "time": {"type": "string", "description": "Time in HH:MM format"},
                            "party_size": {"type": "integer"}
                        },
                        "required": ["restaurant_name", "date", "time", "party_size"]
                    }
                )
            ]
        
        @self.server.call_tool()
        async def handle_call_tool(name: str, arguments: dict) -> List[TextContent]:
            """Handle tool execution"""
            try:
                if name == "get_restaurant_hours":
                    return await self._get_restaurant_hours(arguments)
                elif name == "get_current_reviews":
                    return await self._get_current_reviews(arguments)
                elif name == "check_menu_prices":
                    return await self._check_menu_prices(arguments)
                elif name == "check_reservations":
                    return await self._check_reservations(arguments)
                else:
                    return [TextContent(
                        type="text",
                        text=f"Unknown tool: {name}"
                    )]
            except Exception as e:
                return [TextContent(
                    type="text",
                    text=f"Error executing {name}: {str(e)}"
                )]
    
    async def _get_restaurant_hours(self, args: dict) -> List[TextContent]:
        """Get restaurant opening hours - Mock implementation"""
        restaurant_name = args["restaurant_name"]
        location = args.get("location", "Paris")
        
        # Mock data for demo (replace with real API calls)
        mock_hours = {
            "Monday": "12:00 - 14:30, 19:00 - 22:30",
            "Tuesday": "12:00 - 14:30, 19:00 - 22:30", 
            "Wednesday": "12:00 - 14:30, 19:00 - 22:30",
            "Thursday": "12:00 - 14:30, 19:00 - 22:30",
            "Friday": "12:00 - 14:30, 19:00 - 23:00",
            "Saturday": "12:00 - 14:30, 19:00 - 23:00",
            "Sunday": "Closed"
        }
        
        # Simulate API delay
        await asyncio.sleep(0.5)
        
        response = f"""📍 {restaurant_name} - Opening Hours:

📅 **Weekly Schedule:**
"""
        
        for day, hours in mock_hours.items():
            response += f"• {day}: {hours}\n"
        
        # Add current status
        import datetime
        current_hour = datetime.datetime.now().hour
        if 12 <= current_hour <= 14 or 19 <= current_hour <= 22:
            response += "\n✅ **Currently OPEN**"
        else:
            response += "\n❌ **Currently CLOSED**"
        
        response += f"\n\n🔄 *Last updated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}*"
        
        return [TextContent(type="text", text=response)]
    
    async def _get_current_reviews(self, args: dict) -> List[TextContent]:
        """Get latest reviews - Mock implementation"""
        restaurant_name = args["restaurant_name"]
        limit = args.get("limit", 5)
        
        # Mock reviews data
        mock_reviews = [
            {
                "platform": "Google",
                "rating": 4.5,
                "text": "Amazing coq au vin and excellent service. Authentic French bistro experience!",
                "date": "2024-11-20",
                "author": "Sarah M."
            },
            {
                "platform": "TripAdvisor", 
                "rating": 4.0,
                "text": "Good food but a bit pricey. The atmosphere is wonderful for a romantic dinner.",
                "date": "2024-11-18",
                "author": "Mike R."
            },
            {
                "platform": "Yelp",
                "rating": 5.0,
                "text": "Best escargot in Paris! Staff speaks English and very accommodating.",
                "date": "2024-11-15", 
                "author": "Jennifer L."
            }
        ]
        
        await asyncio.sleep(0.7)  # Simulate API delay
        
        response = f"""⭐ Latest Reviews for {restaurant_name}:\n\n"""
        
        for i, review in enumerate(mock_reviews[:limit], 1):
            stars = "⭐" * int(review["rating"])
            response += f"""**{i}. {review['platform']} - {stars} ({review['rating']}/5)**
👤 {review['author']} • 📅 {review['date']}
💬 "{review['text']}"

"""
        
        # Add summary
        avg_rating = sum(r["rating"] for r in mock_reviews[:limit]) / min(limit, len(mock_reviews))
        response += f"📊 **Average Rating:** {avg_rating:.1f}/5 ⭐"
        
        return [TextContent(type="text", text=response)]
    
    async def _check_menu_prices(self, args: dict) -> List[TextContent]:
        """Get menu and pricing - Mock implementation"""
        restaurant_name = args["restaurant_name"]
        
        # Mock menu data
        mock_menu = {
            "Appetizers": [
                {"item": "Escargot de Bourgogne", "price": "€12"},
                {"item": "French Onion Soup", "price": "€8"},
                {"item": "Pâté de Campagne", "price": "€10"}
            ],
            "Main Courses": [
                {"item": "Coq au Vin", "price": "€28"},
                {"item": "Beef Bourguignon", "price": "€32"},
                {"item": "Duck Confit", "price": "€30"},
                {"item": "Ratatouille Provençale", "price": "€22"}
            ],
            "Desserts": [
                {"item": "Crème Brûlée", "price": "€9"},
                {"item": "Tarte Tatin", "price": "€8"},
                {"item": "Chocolate Soufflé", "price": "€12"}
            ]
        }
        
        await asyncio.sleep(0.8)
        
        response = f"""🍽️ {restaurant_name} - Current Menu & Prices:\n\n"""
        
        for category, items in mock_menu.items():
            response += f"**{category}:**\n"
            for item in items:
                response += f"• {item['item']} - {item['price']}\n"
            response += "\n"
        
        response += "💰 **Average Cost:** €35-45 per person\n"
        response += "🍷 **Wine:** Starting from €6/glass, €28/bottle"
        
        return [TextContent(type="text", text=response)]
    
    async def _check_reservations(self, args: dict) -> List[TextContent]:
        """Check availability - Mock implementation"""
        restaurant_name = args["restaurant_name"]
        date = args["date"]
        time = args["time"] 
        party_size = args["party_size"]
        
        await asyncio.sleep(1.0)  # Simulate reservation system check
        
        # Mock availability
        available_times = ["19:00", "19:30", "21:00", "21:30"]
        requested_time = args["time"]
        
        response = f"""🍽️ Reservation Check for {restaurant_name}:

📅 **Date:** {date}
⏰ **Requested Time:** {time}
👥 **Party Size:** {party_size}

"""
        
        if requested_time in available_times:
            response += f"✅ **AVAILABLE** at {requested_time}!\n\n"
            response += "📞 **Reservation Options:**\n"
            response += "• Call directly: +33 1 42 XX XX XX\n"
            response += "• Online: restaurant-website.com/booking\n"
            response += "• OpenTable: opentable.com\n"
        else:
            response += f"❌ **NOT AVAILABLE** at {requested_time}\n\n"
            response += "⏰ **Alternative Times:**\n"
            for alt_time in available_times:
                response += f"• {alt_time} - Available\n"
        
        response += f"\n💡 **Tip:** Book 2-3 days ahead for weekend dinner reservations"
        
        return [TextContent(type="text", text=response)]

# MCP Server Entry Point
async def main():
    """Run the MCP server"""
    server = RestaurantMCPServer()
    
    async with stdio_server() as (read_stream, write_stream):
        await server.server.run(
            read_stream,
            write_stream,
            InitializeResult(
                protocolVersion="2024-11-05",
                capabilities=server.server.get_capabilities(
                    notification_options=NotificationOptions(),
                    experimental_capabilities={}
                )
            )
        )

if __name__ == "__main__":
    asyncio.run(main())
```

#### **3. Test MCP Server (10 min)**
```bash
# Terminal 1: Start MCP server
cd app/mcp
python restaurant_server.py

# Terminal 2: Test with MCP client
# Install MCP inspector (optional)
npm install -g @modelcontextprotocol/inspector

# Test tool listing
echo '{"jsonrpc": "2.0", "id": 1, "method": "tools/list"}' | python restaurant_server.py

# Test tool execution
echo '{"jsonrpc": "2.0", "id": 2, "method": "tools/call", "params": {"name": "get_restaurant_hours", "arguments": {"restaurant_name": "Café de Flore"}}}' | python restaurant_server.py
```

---

## 🔗 **Exercise 3: Integrate RAG + MCP into Agent**

### **🎯 Objective**
Combine RAG local knowledge with real-time MCP data.

#### **1. Create MCP Tool Wrapper (20 min)**
```python
# File: app/agents/tools/mcp_integration.py

import asyncio
import subprocess
import json
from langchain.tools import BaseTool

class MCPRestaurantTool(BaseTool):
    name = "restaurant_realtime_data"
    description = """
    Get comprehensive real-time restaurant information including:
    - Current opening hours and status
    - Latest customer reviews and ratings
    - Current menu items and pricing
    - Table availability and reservations
    
    Use this when users need up-to-date, actionable restaurant information.
    Input: restaurant_name (required), data_type (hours|reviews|menu|reservations), additional params as needed
    """
    
    def __init__(self):
        super().__init__()
        self.mcp_server_path = "app/mcp/restaurant_server.py"
    
    def _run(self, restaurant_name: str, data_type: str = "hours", **kwargs) -> str:
        """Execute MCP tool via subprocess"""
        return asyncio.run(self._arun(restaurant_name, data_type, **kwargs))
    
    async def _arun(self, restaurant_name: str, data_type: str = "hours", **kwargs) -> str:
        """Async execution of MCP tool"""
        try:
            # Map data types to MCP tool names
            tool_mapping = {
                "hours": "get_restaurant_hours",
                "reviews": "get_current_reviews", 
                "menu": "check_menu_prices",
                "reservations": "check_reservations"
            }
            
            tool_name = tool_mapping.get(data_type, "get_restaurant_hours")
            
            # Prepare arguments
            args = {"restaurant_name": restaurant_name}
            args.update(kwargs)
            
            # Call MCP server
            result = await self._call_mcp_tool(tool_name, args)
            return result
            
        except Exception as e:
            return f"❌ Error getting real-time data: {str(e)}"
    
    async def _call_mcp_tool(self, tool_name: str, arguments: dict) -> str:
        """Call MCP server tool"""
        request = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {
                "name": tool_name,
                "arguments": arguments
            }
        }
        
        # Execute MCP server
        process = await asyncio.create_subprocess_exec(
            "python", self.mcp_server_path,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        
        # Send request
        request_json = json.dumps(request) + "\n"
        stdout, stderr = await process.communicate(request_json.encode())
        
        if process.returncode == 0:
            response = json.loads(stdout.decode())
            if "result" in response and "content" in response["result"]:
                return response["result"]["content"][0]["text"]
            else:
                return "No data received from MCP server"
        else:
            return f"MCP server error: {stderr.decode()}"

class EnhancedLocalKnowledgeTool(BaseTool):
    name = "local_knowledge_enhanced"
    description = """
    Query local Paris dining knowledge base for cultural insights, etiquette, 
    neighborhood guides, and timing tips. Combines with real-time data for 
    comprehensive recommendations.
    """
    
    def __init__(self, rag_service):
        super().__init__()
        self.rag_service = rag_service
    
    def _run(self, query: str) -> str:
        """Query RAG knowledge base"""
        try:
            knowledge_results = self.rag_service.query_knowledge(query, k=3)
            
            if not knowledge_results:
                return "No relevant local knowledge found."
            
            response = "🧠 **Local Knowledge & Insights:**\n\n"
            
            for i, result in enumerate(knowledge_results, 1):
                response += f"{i}. {result['text']}\n"
                if i < len(knowledge_results):
                    response += "\n"
            
            return response
            
        except Exception as e:
            return f"Error accessing local knowledge: {str(e)}"
```

#### **2. Update Discovery Agent (25 min)**
```python
# Update app/agents/discovery_agent.py

from app.agents.tools.mcp_integration import MCPRestaurantTool, EnhancedLocalKnowledgeTool

class EnhancedDiscoveryAgent(DiscoveryAgent):
    def __init__(self, ollama_service, memory_type="buffer"):
        # Initialize RAG service
        from app.services.rag_service import ParisRAGService
        self.rag_service = ParisRAGService()
        
        super().__init__(ollama_service, memory_type)
    
    def _init_tools(self):
        """Initialize enhanced tools with RAG + MCP"""
        traditional_tools = [
            SearchTool(),
            GeocodingTool()
        ]
        
        # Enhanced tools with RAG + MCP
        enhanced_tools = [
            EnhancedLocalKnowledgeTool(self.rag_service),
            MCPRestaurantTool()
        ]
        
        return traditional_tools + enhanced_tools
    
    def _create_enhanced_agent_prompt(self):
        """Enhanced prompt for RAG + MCP capabilities"""
        return """
You are an expert Paris restaurant discovery agent with access to:

🔍 **DISCOVERY TOOLS:**
- search: Find restaurants via Google Local search
- geocoding: Get precise locations and coordinates

🧠 **KNOWLEDGE TOOLS:**
- local_knowledge_enhanced: Cultural insights, etiquette, neighborhood guides
- restaurant_realtime_data: Real-time hours, reviews, menu, reservations

🎯 **OPTIMAL WORKFLOW:**

For restaurant queries:
1. **Discovery**: Use 'search' to find restaurants matching the request
2. **Context**: Use 'local_knowledge_enhanced' for cultural insights and tips
3. **Real-time**: Use 'restaurant_realtime_data' for current information:
   - hours: Get opening hours and current status
   - reviews: Get latest customer feedback  
   - menu: Get current pricing and menu items
   - reservations: Check availability (needs date/time/party_size)

🌟 **EXAMPLE WORKFLOW:**
User: "Romantic dinner tonight near Notre Dame for 2 people"

1. search("romantic restaurants Notre Dame") 
2. local_knowledge_enhanced("romantic dining Paris tips")
3. restaurant_realtime_data(restaurant_name="[top result]", data_type="hours")
4. restaurant_realtime_data(restaurant_name="[top result]", data_type="reviews", limit=3)
5. Synthesize comprehensive recommendation

Always combine multiple sources for rich, actionable recommendations!
"""

    async def enhanced_restaurant_query(self, query: str, location: str) -> str:
        """Enhanced query processing with RAG + MCP"""
        
        # Step 1: Initial search
        search_results = self.search_tool._run(query, location)
        
        # Step 2: Get local knowledge
        cultural_context = self.enhanced_knowledge_tool._run(f"{query} Paris dining culture")
        
        # Step 3: Extract restaurant names from search (simplified)
        restaurant_names = self._extract_restaurant_names(search_results)
        
        # Step 4: Get real-time data for top restaurants
        enhanced_results = []
        for restaurant in restaurant_names[:2]:  # Process top 2
            hours = await self.mcp_tool._arun(restaurant, "hours")
            reviews = await self.mcp_tool._arun(restaurant, "reviews", limit=2)
            
            enhanced_results.append({
                "name": restaurant,
                "hours": hours,
                "reviews": reviews
            })
        
        # Step 5: Synthesize comprehensive response
        return self._synthesize_enhanced_response(
            query, search_results, cultural_context, enhanced_results
        )
```

---

## 🧪 **Exercise 4: Complete Integration Test**

### **🎯 Objective**
Test the complete RAG + MCP system with complex queries.

#### **Test Complex Query (15 min)**
```python
# File: test_integration.py

import asyncio
from app.agents.discovery_agent import EnhancedDiscoveryAgent
from app.services.ollama_service import OllamaService

async def test_complex_query():
    """Test complete RAG + MCP integration"""
    
    # Initialize enhanced agent
    ollama_service = OllamaService()
    agent = EnhancedDiscoveryAgent(ollama_service)
    
    # Test query
    test_query = """
    I'm celebrating my anniversary tonight and need a romantic restaurant 
    near Notre Dame. We prefer French cuisine, want to know if it's open 
    now, recent reviews, and if we can get a table for 2 at 8 PM today.
    Also give me cultural tips for a perfect French dining experience.
    """
    
    print("🎯 Testing Complex Query:")
    print(f"Query: {test_query}")
    print("\n" + "="*60 + "\n")
    
    # Execute query
    result = await agent.enhanced_restaurant_query(test_query, "Paris")
    
    print("📋 **Complete Response:**")
    print(result)
    
    print("\n" + "="*60 + "\n")
    print("✅ Integration test completed!")

if __name__ == "__main__":
    asyncio.run(test_complex_query())
```

#### **Performance Benchmark (10 min)**
```python
# File: benchmark_performance.py

import time
import asyncio
from app.agents.discovery_agent import EnhancedDiscoveryAgent, DiscoveryAgent
from app.services.ollama_service import OllamaService

async def benchmark_performance():
    """Compare performance: Traditional vs Enhanced Agent"""
    
    ollama_service = OllamaService()
    
    # Initialize both agents
    traditional_agent = DiscoveryAgent(ollama_service)
    enhanced_agent = EnhancedDiscoveryAgent(ollama_service)
    
    test_query = "romantic restaurant near Eiffel Tower"
    location = "Paris"
    
    # Benchmark traditional agent
    print("⏱️ Benchmarking Traditional Agent...")
    start_time = time.time()
    traditional_result = traditional_agent.run(test_query, location)
    traditional_time = time.time() - start_time
    
    # Benchmark enhanced agent  
    print("⏱️ Benchmarking Enhanced Agent (RAG + MCP)...")
    start_time = time.time()
    enhanced_result = await enhanced_agent.enhanced_restaurant_query(test_query, location)
    enhanced_time = time.time() - start_time
    
    # Results
    print(f"\n📊 **Performance Comparison:**")
    print(f"Traditional Agent: {traditional_time:.2f}s")
    print(f"Enhanced Agent: {enhanced_time:.2f}s")
    print(f"Overhead: +{enhanced_time - traditional_time:.2f}s ({((enhanced_time/traditional_time - 1) * 100):.1f}%)")
    
    # Quality comparison
    print(f"\n📋 **Quality Comparison:**")
    print(f"Traditional response length: {len(traditional_result)} characters")
    print(f"Enhanced response length: {len(enhanced_result)} characters")
    print(f"Enhanced includes: Real-time data ✅, Cultural insights ✅, Multiple sources ✅")

if __name__ == "__main__":
    asyncio.run(benchmark_performance())
```

---

## 🎯 **Phase 2 Mastery Checklist**

After completing these exercises, you should be able to:

### **RAG Integration ✅**
- [ ] Set up vector database with local knowledge
- [ ] Query knowledge base for relevant insights
- [ ] Combine search results with cultural context
- [ ] Measure knowledge relevance and quality

### **MCP Development ✅**
- [ ] Build MCP server with multiple tools
- [ ] Handle real-time data requests
- [ ] Implement error handling and fallbacks
- [ ] Test MCP tools independently

### **Integration Mastery ✅**
- [ ] Combine RAG + MCP in intelligent workflows
- [ ] Chain multiple data sources effectively
- [ ] Handle parallel tool execution
- [ ] Optimize performance and response times

### **Production Readiness ✅**
- [ ] Handle errors gracefully
- [ ] Monitor performance metrics
- [ ] Provide fallback strategies
- [ ] Deliver comprehensive user experience

---

## 🚀 **Ready for Phase 3: Performance & Caching!**

Your agent now combines:
- **Traditional search** (SerpAPI, Mapbox)
- **Local knowledge** (RAG with cultural insights) 
- **Real-time data** (MCP for hours, reviews, availability)

Next phase will add **intelligent caching** and **performance optimization** to make it production-ready! 🎯