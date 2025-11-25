# 🎓 AI Agent Mastery Lecture Plan
*From Basic Local Agent to Production Multi-Agent System*

## 📅 **Lecture Schedule Overview**

### **Wednesday**: Foundation Understanding (3 hours)
### **Day 2-3**: Phase 1 - Reflexion Implementation (2 days)
### **Day 4-5**: Phase 2 - RAG Integration (2 days) 
### **Day 6-7**: Phase 3 - Caching & Performance (2 days)
### **Day 8-9**: Phase 4 - LangChain/LangGraph Mastery (2 days)
### **Day 10-11**: Phase 5 - Production Single Agent (2 days)
### **Day 12-14**: Phase 6 - Multi-Agent Collaboration (3 days)

---

## 🎯 **Wednesday: Foundation Day**
*Understanding What You Already Have*

### **Hour 1: System Architecture Deep Dive** (60 min)

#### **Live Code Walkthrough** (40 min)
```bash
# Start with your actual system
cd backend && python main.py
cd frontend && npm run dev
```

**Walk through each component:**

1. **Backend Structure** (10 min)
   - `app/agents/discovery_agent.py` - The brain
   - `app/agents/tools.py` - The hands  
   - `app/api/routes.py` - The interface
   - `main.py` - The entry point

2. **ReAct Pattern in Action** (15 min)
   ```python
   # Show actual code in discovery_agent.py lines 45-60
   def _create_agent(self):
       return initialize_agent(
           tools=self.tools,
           llm=self.llm,
           agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
           memory=self.memory,
           verbose=True
       )
   ```

3. **Tools Deep Dive** (15 min)
   ```python
   # Show tools.py - SearchTool and GeocodingTool
   # Demonstrate how each tool works
   # Test individual tools in isolation
   ```

#### **Interactive Demo** (20 min)
- **Simple Query**: "Find pizza near Eiffel Tower"
- **Complex Query**: "Romantic restaurant with outdoor seating near Notre Dame under 50€"
- **Memory Test**: Follow-up questions
- **Show Logs**: Actual ReAct thinking process

### **Hour 2: Component Analysis** (60 min)

#### **Local LLM Integration** (20 min)
```python
# Show app/services/ollama_service.py
# Explain Ollama vs OpenAI choice
# Test different models: llama3.2 vs mixtral
```

#### **Memory Systems** (20 min)
```python
# Test different memory types
curl -X POST "localhost:8000/agent/memory/switch" \
  -d '{"memory_type": "window", "window_size": 5}'

# Show the difference in conversation handling
```

#### **API Design** (20 min)
```python
# Show app/api/routes.py
# Explain FastAPI patterns
# Test all endpoints live
```

### **Hour 3: Hands-On Exploration** (60 min)

#### **Students Implement Simple Tool** (40 min)
```python
# Guide them to create WeatherTool
class WeatherTool(BaseTool):
    name = "weather"
    description = "Get weather for a location"
    
    def _run(self, location: str) -> str:
        # Simple implementation using OpenWeatherMap
        return f"Weather in {location}: Sunny, 22°C"
```

#### **Q&A and Troubleshooting** (20 min)

---

## 🔄 **Phase 1: Reflexion Implementation** (Days 2-3)

### **Day 2: Failure Detection & Reflection Logic**

#### **Morning: Understanding Reflexion** (3 hours)

**What is Reflexion?**
- Self-improving agent that learns from failures
- When agent fails → reflect on why → try again with insights
- Like a student reviewing wrong answers

**Live Implementation:**

1. **Failure Detection** (45 min)
   ```python
   # Add to discovery_agent.py
   def _detect_failure(self, result: str, query: str) -> bool:
       """Detect if agent response indicates failure"""
       failure_indicators = [
           "no results found",
           "couldn't find",
           "sorry",
           "unable to locate",
           "0 results"
       ]
       
       return any(indicator in result.lower() for indicator in failure_indicators)
   ```

2. **Reflection Prompt** (45 min)
   ```python
   def _create_reflection_prompt(self, failed_query: str, failed_result: str) -> str:
       return f"""
   I failed to find good results for: "{failed_query}"
   My result was: "{failed_result}"
   
   Reflection Questions:
   1. Why did this search fail?
   2. What was wrong with my search strategy?
   3. What should I try differently?
   4. How can I improve my search terms?
   
   Provide a brief analysis and improved search strategy:
   """
   ```

3. **Second Attempt Logic** (90 min)
   ```python
   def _run_with_reflexion(self, query: str, location: str) -> str:
       # First attempt
       first_result = self._run_single_attempt(query, location)
       
       if not self._detect_failure(first_result, query):
           return first_result
       
       # Reflexion
       reflection = self._reflect_on_failure(query, first_result)
       
       # Second attempt with reflection
       improved_query = self._improve_query_from_reflection(query, reflection)
       second_result = self._run_single_attempt(improved_query, location)
       
       return second_result
   ```

#### **Afternoon: Integration & Testing** (2 hours)

**Integration into existing system:**
- Modify main agent flow
- Add reflexion toggle in settings
- Test with deliberately failing queries

### **Day 3: Reflection Memory & Optimization**

#### **Morning: Reflection Memory** (3 hours)

1. **Memory Storage** (90 min)
   ```python
   class ReflectionMemory:
       def __init__(self):
           self.reflections = []
           self.max_reflections = 10
       
       def add_reflection(self, query_type: str, failure_reason: str, improvement: str):
           reflection = {
               'timestamp': datetime.now(),
               'query_type': query_type,
               'failure_reason': failure_reason, 
               'improvement': improvement
           }
           self.reflections.append(reflection)
           
           if len(self.reflections) > self.max_reflections:
               self.reflections.pop(0)
   ```

2. **Learning from Past Reflections** (90 min)
   ```python
   def _get_relevant_reflections(self, current_query: str) -> List[dict]:
       """Find similar past failures and learnings"""
       relevant = []
       for reflection in self.reflection_memory.reflections:
           if self._is_similar_query(current_query, reflection['query_type']):
               relevant.append(reflection)
       return relevant
   ```

#### **Afternoon: Complete Integration** (2 hours)

**Full reflexion system working end-to-end**

---

## 📚 **Phase 2: RAG Integration** (Days 4-5)

### **Day 4: RAG Foundation**

#### **Morning: Understanding RAG** (3 hours)

**What is RAG?**
- Retrieval-Augmented Generation
- Agent can query knowledge base before answering
- Like giving the agent a library to reference

**Components:**
1. **Vector Database** (ChromaDB)
2. **Embeddings** (local embeddings via Ollama)
3. **Retrieval Logic**
4. **Generation with Context**

#### **Implementation:**
```python
# New file: app/services/rag_service.py
class RAGService:
    def __init__(self):
        self.vectorstore = Chroma()
        self.embeddings = OllamaEmbeddings(model="nomic-embed-text")
    
    def add_documents(self, documents: List[str]):
        """Add documents to knowledge base"""
        self.vectorstore.add_documents(documents, embeddings=self.embeddings)
    
    def retrieve_context(self, query: str, k: int = 3) -> List[str]:
        """Retrieve relevant context for query"""
        docs = self.vectorstore.similarity_search(query, k=k)
        return [doc.page_content for doc in docs]
```

### **Day 5: RAG Integration with Agent**

#### **RAG Tool Creation**
```python
class RAGTool(BaseTool):
    name = "knowledge_base"
    description = "Query internal knowledge base for local insights and tips"
    
    def _run(self, query: str) -> str:
        context = self.rag_service.retrieve_context(query)
        return f"Knowledge base insights: {context}"
```

---

## ⚡ **Phase 3: Caching & Performance** (Days 6-7)

### **Day 6: Caching Implementation**

#### **Multi-Level Caching:**
1. **Memory Cache** (Redis-like)
2. **Database Cache** (SQLite)
3. **API Response Cache**
4. **Vector Cache** (for RAG)

### **Day 7: Performance Optimization**

#### **Parallel Execution:**
```python
# Execute tools in parallel
async def parallel_search(self, query: str, location: str):
    tasks = [
        self.search_tool.arun(query, location),
        self.geocoding_tool.arun(location),
        self.rag_tool.arun(query)
    ]
    
    results = await asyncio.gather(*tasks)
    return self._combine_results(results)
```

---

## 🔗 **Phase 4: LangChain/LangGraph Mastery** (Days 8-9)

### **Day 8: LangGraph Introduction**

#### **Graph-Based Agent Flow:**
```python
from langgraph import StateGraph

# Define agent workflow as graph
workflow = StateGraph(AgentState)
workflow.add_node("search", search_node)
workflow.add_node("reflect", reflection_node)
workflow.add_node("generate", generation_node)

# Add conditional edges
workflow.add_conditional_edges(
    "search",
    should_reflect,
    {"reflect": "reflect", "generate": "generate"}
)
```

### **Day 9: Advanced LangChain Patterns**

#### **Custom Chains & Runnables**
#### **LangSmith Integration for Monitoring**

---

## 🏭 **Phase 5: Production Single Agent** (Days 10-11)

### **Day 10: Production Readiness**

#### **Features to Add:**
- **Health Monitoring**
- **Metrics & Analytics**  
- **Rate Limiting**
- **Security Hardening**
- **Deployment Automation**

### **Day 11: Scalability**

#### **Horizontal Scaling:**
- **Load Balancing**
- **Database Sharding**
- **Microservice Architecture**

---

## 👥 **Phase 6: Multi-Agent System** (Days 12-14)

### **Day 12: Multi-Agent Architecture**

#### **Agent Roles:**
```python
class PlannerAgent:
    """Breaks down complex queries into subtasks"""
    pass

class SearchWorkerAgent:
    """Specializes in place search"""
    pass

class LocationWorkerAgent:
    """Specializes in geocoding/mapping"""
    pass

class CriticAgent:
    """Reviews and improves results"""
    pass
```

### **Day 13: Agent Communication**

#### **Message Passing System:**
```python
class AgentCommunicator:
    def __init__(self):
        self.message_queue = asyncio.Queue()
    
    async def send_message(self, from_agent: str, to_agent: str, message: dict):
        await self.message_queue.put({
            'from': from_agent,
            'to': to_agent,
            'content': message,
            'timestamp': datetime.now()
        })
```

### **Day 14: Complete Multi-Agent System**

#### **Full Integration & Testing**

---

## ✅ **Validation Checkpoints**

### **After Each Phase:**

1. **Demo Working Feature**
2. **Code Review Session**
3. **Performance Benchmarks**
4. **Student Q&A**
5. **Next Phase Preview**

### **Final Assessment:**

**Students must build from scratch:**
- A complete multi-agent system
- For a different domain (e.g., recipe finder, job search)
- With all 6 phases implemented
- Deployed and working

---

## 🎯 **Success Metrics**

### **By End of Course, Students Can:**

- [ ] **Explain** ReAct, RAG, Reflexion patterns clearly
- [ ] **Build** custom agents with any tool combination
- [ ] **Deploy** production-ready AI systems
- [ ] **Debug** complex multi-agent interactions
- [ ] **Optimize** performance for scale
- [ ] **Teach** others these concepts

---

## 📋 **Daily Schedule Template**

### **Every Day Structure:**

#### **Morning (3 hours):**
- **Theory** (45 min): Concept explanation
- **Live Coding** (90 min): Implementation together
- **Break** (15 min)
- **Hands-On** (30 min): Student practice

#### **Afternoon (2 hours):**
- **Integration** (60 min): Add to main system
- **Testing** (45 min): Validate everything works
- **Q&A** (15 min): Troubleshooting & questions

---

**This plan takes students from understanding your current system to building production-grade multi-agent systems in 2 weeks!** 🚀

**Ready to validate this plan?** 🎓