# 🔧 Phase 2: Tool Architecture Mastery
*From Using Tools to Building Tools*

## 🎯 Learning Objectives

By the end of Phase 2, you'll master:
- **Creating custom tools** that extend your agent's capabilities
- **Tool chaining strategies** for complex multi-step operations
- **Error handling patterns** that make your system bulletproof
- **Performance optimization** techniques for production scale

---

## 🧠 Understanding Tool Architecture

### **Simple Explanation**
Think of tools as specialized workers in a factory:
- Each worker has **one specific job** (search, geocode, calculate)
- They all speak the **same language** (input/output format)
- The **supervisor (agent)** decides which worker to use
- Workers can **pass results** to each other

### **Technical Deep Dive**
```python
# Tool Architecture Pattern
class BaseTool:
    name: str           # Tool identifier
    description: str    # What it does (for LLM)
    input_schema: dict  # Expected inputs
    output_format: str  # Standardized output
    
    def _run(self, **kwargs) -> str:
        # Core tool logic
        pass
    
    def _handle_error(self, error) -> str:
        # Error handling
        pass
```

---

## 🛠️ Module 1: Creating Custom Tools

### **Exercise 1: Restaurant Hours Tool**

#### **Simple Explanation**
We'll create a tool that finds restaurant opening hours - something your current system can't do.

#### **Step-by-Step Implementation**

1. **Create the tool file:**
```python
# app/agents/tools/restaurant_hours_tool.py

from langchain.tools import BaseTool
from typing import Optional
import requests
from datetime import datetime

class RestaurantHoursTool(BaseTool):
    name = "restaurant_hours"
    description = """
    Get opening hours and current status for restaurants.
    Use this when users ask about opening hours, closing times, 
    or if a place is currently open.
    Input should be restaurant name and location.
    """
    
    def _run(self, restaurant_name: str, location: str = "") -> str:
        """Get restaurant hours"""
        try:
            # Method 1: Use Google Places API (if available)
            hours_data = self._get_hours_from_places_api(restaurant_name, location)
            
            if not hours_data:
                # Method 2: Fallback to web scraping
                hours_data = self._get_hours_from_web(restaurant_name, location)
            
            if not hours_data:
                return f"Sorry, I couldn't find opening hours for {restaurant_name} in {location}"
            
            return self._format_hours_response(hours_data, restaurant_name)
            
        except Exception as e:
            return f"Error getting hours for {restaurant_name}: {str(e)}"
    
    def _get_hours_from_places_api(self, name: str, location: str) -> dict:
        """Get hours from Google Places API"""
        # Implementation depends on your API keys
        # For now, return mock data
        return {
            "monday": "9:00 AM - 10:00 PM",
            "tuesday": "9:00 AM - 10:00 PM", 
            "wednesday": "9:00 AM - 10:00 PM",
            "thursday": "9:00 AM - 10:00 PM",
            "friday": "9:00 AM - 11:00 PM",
            "saturday": "9:00 AM - 11:00 PM",
            "sunday": "10:00 AM - 9:00 PM",
            "is_open_now": True
        }
    
    def _get_hours_from_web(self, name: str, location: str) -> dict:
        """Fallback web scraping method"""
        # Basic implementation - you can expand this
        return None
    
    def _format_hours_response(self, hours_data: dict, restaurant_name: str) -> str:
        """Format hours in a user-friendly way"""
        current_day = datetime.now().strftime("%A").lower()
        
        response = f"📍 {restaurant_name} Hours:\n\n"
        
        days = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
        for day in days:
            if day in hours_data:
                status = " ← TODAY" if day == current_day else ""
                response += f"• {day.title()}: {hours_data[day]}{status}\n"
        
        if hours_data.get("is_open_now"):
            response += f"\n✅ Currently OPEN"
        else:
            response += f"\n❌ Currently CLOSED"
            
        return response
```

2. **Register the tool in your agent:**
```python
# app/agents/discovery_agent.py

from .tools.restaurant_hours_tool import RestaurantHoursTool

class DiscoveryAgent:
    def _init_tools(self):
        tools = [
            SearchTool(),
            GeocodingTool(),
            RestaurantHoursTool()  # Add your new tool
        ]
        return tools
```

#### **Test Your New Tool**
```json
POST http://localhost:8000/search
{
  "query": "Is Café de Flore open right now?",
  "location": "Paris"
}
```

### **Exercise 2: Price Range Tool**

Create a tool that estimates price ranges for restaurants:

```python
class PriceRangeTool(BaseTool):
    name = "price_range"
    description = """
    Get price range information for restaurants and venues.
    Use when users ask about cost, pricing, budget, or affordability.
    """
    
    def _run(self, venue_name: str, location: str = "") -> str:
        # Your implementation here
        price_indicators = {
            "$": "Budget-friendly (Under 20€ per person)",
            "$$": "Moderate (20-40€ per person)", 
            "$$$": "Upscale (40-80€ per person)",
            "$$$$": "Fine dining (80€+ per person)"
        }
        
        # Logic to determine price range
        estimated_range = self._estimate_price_range(venue_name, location)
        
        return f"💰 {venue_name} - {price_indicators[estimated_range]}"
```

---

## 🔗 Module 2: Tool Chaining Strategies

### **Simple Explanation**
Tool chaining is like a relay race - each tool passes its result to the next tool to build a complete answer.

### **Exercise 3: Complex Query Handler**

**Scenario**: User asks *"Find me a romantic restaurant near Notre Dame that's open until midnight and costs under 50€"*

**Chain Strategy**:
1. **SearchTool**: Find romantic restaurants near Notre Dame
2. **RestaurantHoursTool**: Check which ones are open until midnight
3. **PriceRangeTool**: Filter by budget under 50€
4. **GeocodingTool**: Get exact locations for final results

#### **Implementation Pattern**

```python
class ChainedQueryHandler:
    def __init__(self, agent):
        self.agent = agent
        self.search_tool = SearchTool()
        self.hours_tool = RestaurantHoursTool()
        self.price_tool = PriceRangeTool()
        self.geo_tool = GeocodingTool()
    
    def handle_complex_query(self, query: str, location: str) -> str:
        """Handle multi-constraint queries"""
        
        # Step 1: Extract requirements
        requirements = self._parse_requirements(query)
        
        # Step 2: Initial search
        initial_results = self.search_tool._run(
            query=requirements['cuisine_type'], 
            location=location
        )
        
        # Step 3: Filter by hours if needed
        if requirements.get('hours_requirement'):
            filtered_by_hours = self._filter_by_hours(
                initial_results, 
                requirements['hours_requirement']
            )
        else:
            filtered_by_hours = initial_results
        
        # Step 4: Filter by price if needed  
        if requirements.get('budget'):
            final_results = self._filter_by_price(
                filtered_by_hours,
                requirements['budget']
            )
        else:
            final_results = filtered_by_hours
        
        # Step 5: Get precise locations
        enriched_results = self._enrich_with_locations(final_results)
        
        return self._format_final_response(enriched_results, requirements)
    
    def _parse_requirements(self, query: str) -> dict:
        """Extract search requirements from natural language"""
        requirements = {}
        
        # Simple keyword matching (you can use NLP libraries for better parsing)
        if "romantic" in query.lower():
            requirements['ambiance'] = 'romantic'
        if "midnight" in query.lower():
            requirements['hours_requirement'] = 'late_night'
        if "under" in query.lower() and "€" in query:
            # Extract budget amount
            import re
            budget_match = re.search(r'under (\d+)€', query.lower())
            if budget_match:
                requirements['budget'] = int(budget_match.group(1))
        
        return requirements
```

### **Exercise 4: Smart Fallback Chains**

**Scenario**: Primary tool fails, gracefully degrade

```python
class SmartFallbackChain:
    def __init__(self):
        self.primary_search = SearchTool()
        self.fallback_search = BasicWebSearchTool()
        self.local_cache = LocalCacheTool()
    
    def search_with_fallbacks(self, query: str, location: str) -> str:
        """Try multiple search strategies"""
        
        # Try 1: Primary search (SerpAPI)
        try:
            result = self.primary_search._run(query, location)
            if self._is_good_result(result):
                return result
        except Exception as e:
            self._log_error("Primary search failed", e)
        
        # Try 2: Fallback search
        try:
            result = self.fallback_search._run(query, location)
            if self._is_good_result(result):
                return f"⚠️ Using backup search:\n{result}"
        except Exception as e:
            self._log_error("Fallback search failed", e)
        
        # Try 3: Local cache
        cached_result = self.local_cache.get_similar_query(query, location)
        if cached_result:
            return f"📁 From cache (may be outdated):\n{cached_result}"
        
        # Last resort
        return f"❌ Unable to search right now. Try: '{query}' on Google Maps"
```

---

## 🛡️ Module 3: Error Handling Patterns

### **Exercise 5: Bulletproof Error Handling**

#### **The 4-Layer Defense Pattern**

```python
class RobustTool(BaseTool):
    name = "robust_search"
    description = "A bulletproof search tool with comprehensive error handling"
    
    def _run(self, query: str, location: str = "") -> str:
        """Main execution with 4-layer error handling"""
        
        # Layer 1: Input Validation
        validation_error = self._validate_inputs(query, location)
        if validation_error:
            return validation_error
        
        # Layer 2: Rate Limiting & Circuit Breaker
        if not self._check_rate_limits():
            return self._get_rate_limit_message()
        
        # Layer 3: Execution with Timeout
        try:
            with timeout(30):  # 30-second timeout
                result = self._execute_search(query, location)
                return self._validate_output(result)
                
        except TimeoutError:
            return self._handle_timeout(query, location)
        except APIError as e:
            return self._handle_api_error(e, query, location)
        except Exception as e:
            return self._handle_unexpected_error(e, query, location)
    
    def _validate_inputs(self, query: str, location: str) -> Optional[str]:
        """Layer 1: Input validation"""
        if not query or len(query.strip()) == 0:
            return "❌ Please provide a search query"
        
        if len(query) > 200:
            return "❌ Query too long. Please be more specific"
        
        # Check for potentially harmful inputs
        dangerous_patterns = ['<script>', 'javascript:', 'data:']
        if any(pattern in query.lower() for pattern in dangerous_patterns):
            return "❌ Invalid query format"
        
        return None
    
    def _check_rate_limits(self) -> bool:
        """Layer 2: Rate limiting"""
        # Implement rate limiting logic
        current_requests = self._get_current_request_count()
        max_requests_per_minute = 60
        
        return current_requests < max_requests_per_minute
    
    def _handle_timeout(self, query: str, location: str) -> str:
        """Layer 3: Timeout handling"""
        return f"""⏱️ Search is taking longer than expected.
        
This might be due to:
• High search volume
• API slowness
• Complex query processing

🔄 Try: Simplify your query to "{query.split()[0]}" or try again in a few minutes"""
    
    def _handle_api_error(self, error: APIError, query: str, location: str) -> str:
        """Layer 3: API error handling"""
        if error.status_code == 429:  # Rate limited
            return f"🚦 Too many requests. Please wait 60 seconds and try again"
        elif error.status_code == 401:  # Unauthorized
            return f"🔑 API access issue. Please contact support"
        elif error.status_code >= 500:  # Server error
            return f"🛠️ Search service temporarily unavailable. Try again in 5 minutes"
        else:
            return f"🔍 Search temporarily unavailable. Try Google Maps for '{query} {location}'"
    
    def _handle_unexpected_error(self, error: Exception, query: str, location: str) -> str:
        """Layer 4: Catch-all error handling"""
        error_id = self._log_error(error, query, location)
        
        return f"""🚨 Unexpected error occurred (ID: {error_id})
        
Meanwhile, try:
• Google Maps: "{query} {location}"
• TripAdvisor: "{query}"
• Local directory services

The error has been logged and will be fixed soon."""
```

### **Exercise 6: Graceful Degradation**

```python
class GracefulSearchTool(BaseTool):
    def _run(self, query: str, location: str) -> str:
        """Search with graceful degradation levels"""
        
        # Level 1: Full-featured search
        try:
            return self._premium_search(query, location)
        except Exception:
            pass
        
        # Level 2: Basic search
        try:
            return self._basic_search(query, location) 
        except Exception:
            pass
        
        # Level 3: Cached results
        try:
            cached = self._get_cached_results(query, location)
            return f"📁 Cached results:\n{cached}"
        except Exception:
            pass
        
        # Level 4: Helpful suggestions
        return self._get_helpful_fallback(query, location)
    
    def _get_helpful_fallback(self, query: str, location: str) -> str:
        """Last resort - provide helpful alternatives"""
        return f"""🔍 Search temporarily unavailable. Here are alternatives:

📱 **Mobile Apps:**
• Google Maps: Search "{query}"
• TripAdvisor: Browse {location}
• Yelp: Find "{query}" nearby

🌐 **Websites:**
• {location} tourism website
• Local directory services
• Restaurant booking platforms

💡 **Pro tip:** Try a simpler search term like "{query.split()[0]}" when service resumes"""
```

---

## ⚡ Module 4: Performance Optimization

### **Exercise 7: Caching Strategy**

#### **Simple Explanation**
Caching is like having a notebook where you write down answers to questions you've already solved, so you don't have to solve them again.

```python
class CachedSearchTool(BaseTool):
    def __init__(self):
        self.cache = {}
        self.cache_ttl = 3600  # 1 hour
    
    def _run(self, query: str, location: str) -> str:
        # Create cache key
        cache_key = self._create_cache_key(query, location)
        
        # Check cache first
        cached_result = self._get_from_cache(cache_key)
        if cached_result:
            return f"⚡ {cached_result}"
        
        # Execute search
        result = self._execute_search(query, location)
        
        # Store in cache
        self._store_in_cache(cache_key, result)
        
        return result
    
    def _create_cache_key(self, query: str, location: str) -> str:
        """Create normalized cache key"""
        # Normalize for better cache hits
        normalized_query = query.lower().strip()
        normalized_location = location.lower().strip()
        
        return f"{normalized_query}|{normalized_location}"
    
    def _get_from_cache(self, key: str) -> Optional[str]:
        """Get from cache if not expired"""
        if key in self.cache:
            cached_data, timestamp = self.cache[key]
            if time.time() - timestamp < self.cache_ttl:
                return cached_data
            else:
                # Remove expired entry
                del self.cache[key]
        return None
```

### **Exercise 8: Parallel Tool Execution**

```python
import asyncio
from concurrent.futures import ThreadPoolExecutor

class ParallelToolExecutor:
    def __init__(self):
        self.executor = ThreadPoolExecutor(max_workers=4)
    
    async def execute_parallel_search(self, query: str, location: str) -> str:
        """Execute multiple tools in parallel"""
        
        # Define parallel tasks
        tasks = [
            self._async_search(query, location),
            self._async_geocoding(location),
            self._async_hours_check(query, location)
        ]
        
        # Execute in parallel
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Combine results
        search_result, geo_result, hours_result = results
        
        return self._combine_parallel_results(
            search_result, geo_result, hours_result
        )
    
    async def _async_search(self, query: str, location: str) -> dict:
        """Async wrapper for search tool"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self.executor, 
            self.search_tool._run, 
            query, 
            location
        )
```

### **Exercise 9: Smart Request Batching**

```python
class BatchProcessor:
    def __init__(self):
        self.pending_requests = []
        self.batch_size = 5
        self.batch_timeout = 2  # seconds
    
    async def add_request(self, query: str, location: str) -> str:
        """Add request to batch"""
        request = {
            'query': query,
            'location': location,
            'future': asyncio.Future()
        }
        
        self.pending_requests.append(request)
        
        # Trigger batch if full
        if len(self.pending_requests) >= self.batch_size:
            await self._process_batch()
        
        # Wait for result
        return await request['future']
    
    async def _process_batch(self):
        """Process batch of requests efficiently"""
        if not self.pending_requests:
            return
        
        batch = self.pending_requests.copy()
        self.pending_requests.clear()
        
        # Batch API call (more efficient than individual calls)
        try:
            results = await self._batch_api_call(batch)
            
            # Distribute results
            for request, result in zip(batch, results):
                request['future'].set_result(result)
                
        except Exception as e:
            # Handle batch failure
            for request in batch:
                request['future'].set_exception(e)
```

---

## 🎯 Phase 2 Mastery Checklist

After completing these exercises, you should be able to:

### **Tool Creation**
- [ ] Build custom tools that extend agent capabilities
- [ ] Design proper input/output schemas
- [ ] Implement error handling in tools
- [ ] Test tools independently

### **Tool Chaining** 
- [ ] Chain multiple tools for complex queries
- [ ] Implement fallback strategies
- [ ] Parse natural language requirements
- [ ] Build smart routing logic

### **Error Handling**
- [ ] Implement 4-layer defense patterns
- [ ] Design graceful degradation
- [ ] Provide helpful error messages
- [ ] Log errors for debugging

### **Performance**
- [ ] Implement caching strategies
- [ ] Execute tools in parallel
- [ ] Batch API requests efficiently
- [ ] Monitor performance metrics

---

## 🚀 Ready for Phase 3?

**Phase 3: LLM Integration & Prompt Mastery** will cover:
- Advanced prompt engineering
- Local LLM optimization  
- Context management strategies
- Multi-model architectures

You're building real expertise! 🎓