# 🎯 Phase 1: Hands-On Mastery Exercises
*Understanding Your AI System Through Practice*

## 🧪 Exercise 1: Observing the ReAct Pattern

### **Simple Explanation**
Your AI agent follows a specific thinking pattern: **Thought → Action → Observation → Thought → Action...** until it finds the answer. It's like a detective gathering clues step by step.

### **Exercise: Watch Your Agent Think**

1. **Start your backend:**
   ```bash
   cd backend
   python main.py
   ```

2. **Test these queries and observe the pattern:**

   **Query A: Simple Search**
   ```json
   POST http://localhost:8000/search
   {
     "query": "Best pizza near Eiffel Tower",
     "location": "Paris"
   }
   ```

   **Query B: Complex Request**
   ```json
   POST http://localhost:8000/search
   {
     "query": "I need a romantic restaurant with outdoor seating near Notre Dame that's open until midnight and has good wine",
     "location": "Paris"
   }
   ```

3. **What to Look For:**
   - **Thought**: "I need to search for restaurants..."
   - **Action**: Uses search tool
   - **Observation**: Gets results
   - **Thought**: "Let me get the location..."
   - **Action**: Uses geocoding tool
   - **Final Answer**: Combines everything

### **🎓 Learning Questions:**
- How many steps did your agent take for each query?
- What tools did it choose to use?
- How did it combine the information?

---

## 🔧 Exercise 2: Understanding Your Tools

### **Simple Explanation**
Your agent has a toolbox with 3 main tools:
- **Search Tool**: Finds places on Google (like a super-powered search)
- **Geocoding Tool**: Finds exact addresses and coordinates
- **Memory**: Remembers your conversation

### **Exercise: Test Each Tool Individually**

1. **Test Search Tool Only:**
   ```json
   {
     "query": "coffee shops",
     "location": "Paris"
   }
   ```
   → Watch what data comes back

2. **Test Geocoding Tool:**
   ```json
   {
     "query": "Where exactly is 1 Rue de Rivoli?",
     "location": "Paris"
   }
   ```
   → See how it finds precise coordinates

3. **Test Memory:**
   ```json
   # First request
   {
     "query": "Find me a good bakery in Montmartre",
     "location": "Paris"
   }
   
   # Second request (in same conversation)
   {
     "query": "Is that bakery open on Sundays?",
     "location": "Paris"
   }
   ```
   → Notice how it remembers "that bakery"

### **🔍 Technical Deep Dive:**

**Check your tools configuration:**
```python
# Look at backend/app/agents/tools.py
class SearchTool:
    def __init__(self):
        self.serp_api = SerpAPIService()
    
    def _run(self, query: str, location: str) -> str:
        # This is where the magic happens
```

---

## 🧠 Exercise 3: Testing Memory Types

### **Simple Explanation**
Your AI can remember conversations in 3 different ways:
- **Buffer**: Remembers everything (like a perfect diary)
- **Window**: Remembers only recent messages (like short-term memory)
- **Summary**: Creates summaries (like taking notes)

### **Exercise: Switch Memory Types**

1. **Test with Buffer Memory (default):**
   ```bash
   # Have a long conversation (10+ messages)
   # Ask: "What did we talk about first?"
   # It should remember everything
   ```

2. **Switch to Window Memory:**
   ```json
   POST http://localhost:8000/agent/memory/switch
   {
     "memory_type": "window",
     "window_size": 3
   }
   ```

3. **Test Window Memory:**
   ```bash
   # Have the same long conversation
   # Ask: "What did we talk about first?"
   # It should only remember last 3 messages
   ```

### **🔬 Technical Analysis:**
```python
# Check backend/app/agents/discovery_agent.py line ~50
if memory_type == "buffer":
    self.memory = ConversationBufferMemory()
elif memory_type == "window":
    self.memory = ConversationBufferWindowMemory(k=window_size)
```

---

## 🎯 Exercise 4: Prompt Engineering Experiment

### **Simple Explanation**
The way you ask questions affects how smart your AI appears. It's like the difference between asking "food?" vs "Can you recommend a romantic Italian restaurant for a special anniversary dinner?"

### **Exercise: Test Different Query Styles**

1. **Vague Query:**
   ```json
   {"query": "food", "location": "Paris"}
   ```

2. **Specific Query:**
   ```json
   {"query": "Traditional French bistro with outdoor seating, serves escargot, budget under 50€ per person", "location": "Paris"}
   ```

3. **Context-Rich Query:**
   ```json
   {"query": "I'm celebrating my graduation with my parents who don't speak French. We need a restaurant near the Louvre with English menus and classic French dishes they'd recognize", "location": "Paris"}
   ```

### **📊 Compare Results:**
- **Quality of recommendations**
- **Number of results**
- **Relevance to needs**
- **Agent reasoning depth**

---

## 🧪 Exercise 5: Error Handling & Edge Cases

### **Simple Explanation**
Testing what happens when things go wrong helps you understand how robust your system is.

### **Exercise: Break Things Safely**

1. **Invalid Location:**
   ```json
   {"query": "pizza", "location": "Atlantis"}
   ```

2. **Empty Query:**
   ```json
   {"query": "", "location": "Paris"}
   ```

3. **Impossible Request:**
   ```json
   {"query": "24-hour ice cream shop that also sells cars and offers skydiving lessons", "location": "Paris"}
   ```

### **🛠️ What to Observe:**
- How does the agent handle errors?
- Does it give helpful feedback?
- Does it gracefully degrade or crash?

---

## 📈 Exercise 6: Performance Analysis

### **Exercise: Measure Your System**

1. **Response Time Test:**
   ```bash
   # Time simple vs complex queries
   time curl -X POST "http://localhost:8000/search" \
     -H "Content-Type: application/json" \
     -d '{"query": "pizza", "location": "Paris"}'
   ```

2. **Memory Usage:**
   ```bash
   # Check memory info
   curl http://localhost:8000/agent/memory/info
   ```

3. **Available Models:**
   ```bash
   # See what LLMs you can use
   curl http://localhost:8000/models/available
   ```

---

## 🎯 Phase 1 Mastery Checklist

After completing these exercises, you should be able to explain:

### **To Non-Technical People:**
- [ ] "Our AI thinks step-by-step like a detective"
- [ ] "It has three main tools to find information"
- [ ] "It remembers conversations in different ways"
- [ ] "The way you ask questions affects the quality of answers"

### **To Technical People:**
- [ ] "We use the ReAct pattern for agent reasoning"
- [ ] "Our tools are SerpAPI, Mapbox, and LangChain memory"
- [ ] "We have configurable memory types with different trade-offs"
- [ ] "Prompt engineering significantly impacts performance"

### **To Yourself:**
- [ ] I can predict how my agent will behave
- [ ] I know which memory type to use when
- [ ] I can write effective queries for different scenarios
- [ ] I understand the limitations and strengths of my system

---

## 🚀 Ready for Phase 2?

Once you can confidently explain these concepts and demonstrate the exercises, you're ready to dive deeper into **Phase 2: Tool Architecture Mastery**.

The next phase will cover:
- Creating custom tools
- Tool chaining strategies
- Error handling patterns
- Performance optimization