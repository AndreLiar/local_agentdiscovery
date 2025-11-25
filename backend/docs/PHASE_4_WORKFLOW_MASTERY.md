# 🔗 Phase 4: Workflow Automation Mastery
*LangGraph + N8N: From Agent Workflows to Business Automation*

## 🎯 **Learning Objectives**

Master two complementary workflow systems:
- **LangGraph**: Internal agent decision workflows
- **N8N**: External business process automation

## 📋 **Phase 4 Structure (4 days)**

### **Days 1-2: LangGraph (Agent Workflows)**
### **Days 3-4: N8N (Business Workflows)**

---

## 🧠 **Part A: LangGraph Agent Workflows (Days 1-2)**

### **Day 1: LangGraph Fundamentals**

#### **🎯 What is LangGraph?**
LangGraph creates **decision trees** for your AI agent - like a flowchart that the agent follows based on results and conditions.

**Simple Example:**
```
User Query → Agent analyzes → Decision:
├── Simple query? → Direct search → Done
├── Complex query? → Multi-tool workflow → Synthesize → Done  
└── Unclear query? → Ask clarification → Retry
```

#### **Implementation: Restaurant Discovery Graph**
```python
# File: app/agents/langgraph_workflows.py

from langgraph import StateGraph, END
from typing import TypedDict, List

class AgentState(TypedDict):
    user_query: str
    location: str
    search_results: Optional[str]
    local_knowledge: Optional[str]
    mcp_data: Optional[str]
    final_response: str
    confidence_score: float
    errors: List[str]

class RestaurantDiscoveryGraph:
    def __init__(self, agent):
        self.agent = agent
        self.workflow = self._create_workflow()
        self.app = self.workflow.compile()
    
    def _create_workflow(self) -> StateGraph:
        """Create restaurant discovery workflow graph"""
        
        workflow = StateGraph(AgentState)
        
        # Add nodes (workflow steps)
        workflow.add_node("analyze_query", self.analyze_query_node)
        workflow.add_node("search_places", self.search_places_node)
        workflow.add_node("get_knowledge", self.get_knowledge_node)
        workflow.add_node("get_realtime_data", self.get_realtime_data_node)
        workflow.add_node("synthesize_response", self.synthesize_response_node)
        workflow.add_node("handle_failure", self.handle_failure_node)
        
        # Set entry point
        workflow.set_entry_point("analyze_query")
        
        # Add conditional routing (the magic!)
        workflow.add_conditional_edges(
            "analyze_query",
            self.should_continue_after_analysis,
            {
                "search": "search_places",
                "clarify": END,  # Ask user for clarification
                "error": "handle_failure"
            }
        )
        
        workflow.add_conditional_edges(
            "search_places", 
            self.should_enhance_results,
            {
                "enhance": "get_knowledge",
                "sufficient": "synthesize_response",
                "failed": "handle_failure"
            }
        )
        
        workflow.add_edge("get_knowledge", "get_realtime_data")
        workflow.add_edge("get_realtime_data", "synthesize_response")
        workflow.add_edge("synthesize_response", END)
        workflow.add_edge("handle_failure", END)
        
        return workflow
    
    def analyze_query_node(self, state: AgentState) -> AgentState:
        """Analyze and categorize the user query"""
        query = state["user_query"]
        
        # Simple query classification
        if len(query.split()) < 3:
            state["errors"].append("Query too short - needs clarification")
            return state
        
        # Check for location
        if not state["location"]:
            state["errors"].append("No location provided")
            return state
        
        # Query is valid
        return state
    
    def should_continue_after_analysis(self, state: AgentState) -> str:
        """Decide next step after query analysis"""
        if state["errors"]:
            return "error"
        
        # Check if query needs clarification
        vague_terms = ["restaurant", "food", "place", "somewhere"]
        query_words = state["user_query"].lower().split()
        
        if len(query_words) < 4 and any(term in query_words for term in vague_terms):
            return "clarify"
        
        return "search"
    
    def search_places_node(self, state: AgentState) -> AgentState:
        """Execute place search"""
        try:
            search_tool = self.agent.get_tool("search")
            results = search_tool._run(state["user_query"], state["location"])
            state["search_results"] = results
            
            # Basic quality check
            if "no results" in results.lower() or len(results) < 100:
                state["errors"].append("Poor search results")
                
        except Exception as e:
            state["errors"].append(f"Search failed: {str(e)}")
        
        return state
    
    def should_enhance_results(self, state: AgentState) -> str:
        """Decide if results need enhancement"""
        if state["errors"]:
            return "failed"
        
        # If query mentions specific needs, enhance with knowledge/real-time data
        enhancement_triggers = [
            "romantic", "anniversary", "business", "family", 
            "vegetarian", "gluten-free", "budget", "expensive",
            "tonight", "reservation", "open now"
        ]
        
        query_lower = state["user_query"].lower()
        if any(trigger in query_lower for trigger in enhancement_triggers):
            return "enhance"
        
        # Basic results are sufficient
        return "sufficient"
```

### **Day 2: Advanced LangGraph Patterns**

#### **Complex Decision Trees**
```python
def create_advanced_workflow(self) -> StateGraph:
    """Advanced workflow with parallel processing and loops"""
    
    workflow = StateGraph(AgentState)
    
    # Parallel data gathering
    workflow.add_node("gather_basic_data", self.parallel_basic_search)
    workflow.add_node("gather_enhanced_data", self.parallel_enhanced_search)
    workflow.add_node("quality_check", self.quality_assessment_node)
    workflow.add_node("improve_results", self.improvement_node)
    
    # Add loop for iterative improvement
    workflow.add_conditional_edges(
        "quality_check",
        self.quality_gate,
        {
            "good": "synthesize_response",
            "improve": "improve_results",  # Loop back
            "failed": "handle_failure"
        }
    )
    
    # Loop back to quality check after improvement
    workflow.add_edge("improve_results", "quality_check")
    
    return workflow

async def parallel_basic_search(self, state: AgentState) -> AgentState:
    """Execute multiple searches in parallel"""
    
    tasks = [
        self._async_search(state["user_query"], state["location"]),
        self._async_geocoding(state["location"]),
        self._async_knowledge_query(state["user_query"])
    ]
    
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    state["search_results"] = results[0] if not isinstance(results[0], Exception) else None
    state["geocoding_data"] = results[1] if not isinstance(results[1], Exception) else None  
    state["knowledge_data"] = results[2] if not isinstance(results[2], Exception) else None
    
    return state
```

---

## 🔧 **Part B: N8N Business Workflow Automation (Days 3-4)**

### **Day 3: N8N Setup & Basic Workflows**

#### **🎯 What is N8N?**
N8N automates **business processes** around your AI agent - connecting it to external systems and creating complete customer journeys.

**Business Example:**
```
Customer uses AI agent → N8N workflow:
├── Log customer preferences to CRM
├── Check if restaurant has availability  
├── Make actual reservation via API
├── Send confirmation email
├── Add to customer's calendar
├── Set SMS reminder
└── Schedule follow-up survey
```

#### **Installation & Setup (30 min)**
```bash
# Install N8N self-hosted
npm install n8n -g

# Start N8N  
npx n8n start

# Access at http://localhost:5678
```

#### **Connect to Your AI Agent (45 min)**
```javascript
// N8N HTTP Request Node - Call your agent
{
  "url": "http://localhost:8000/search",
  "method": "POST", 
  "headers": {
    "Content-Type": "application/json"
  },
  "body": {
    "query": "{{$json.user_query}}",
    "location": "{{$json.location}}"
  }
}
```

#### **Workflow 1: Smart Recommendation Logger (90 min)**

**Visual Flow:**
```
Webhook → Parse Request → Call AI Agent → Parse Response → 
├── Log to Airtable (customer database)
├── Send Email (confirmation)  
├── Slack Notification (team alert)
└── Google Sheets (analytics log)
```

**N8N Implementation:**
```javascript
// 1. Webhook Trigger
{
  "httpMethod": "POST",
  "path": "ai-recommendation",
  "responseMode": "responseNode"
}

// 2. Call AI Agent  
{
  "url": "http://localhost:8000/search",
  "method": "POST",
  "body": {
    "query": "{{$json.query}}",
    "location": "{{$json.location}}"
  }
}

// 3. Log to Airtable
{
  "base": "your-base-id",
  "table": "Customer Interactions",
  "fields": {
    "Query": "{{$node['Webhook'].json.query}}",
    "Result": "{{$node['AI Agent'].json.result}}",
    "Timestamp": "{{new Date().toISOString()}}",
    "Location": "{{$node['Webhook'].json.location}}"
  }
}

// 4. Send Confirmation Email
{
  "to": "{{$node['Webhook'].json.email}}",
  "subject": "Your Restaurant Recommendations",
  "html": `
    <h2>Here are your personalized recommendations:</h2>
    <p>{{$node['AI Agent'].json.result}}</p>
    <p>Enjoy your dining experience!</p>
  `
}
```

### **Day 4: Advanced Business Integrations**

#### **Workflow 2: Complete Reservation System (3 hours)**

**Business Process:**
```
AI Recommendation → Check Availability → Make Reservation → 
Confirm with Customer → Add to Calendar → Set Reminders
```

**N8N Advanced Workflow:**
```javascript
// 1. Parse AI Agent Response for Restaurant Names
{
  "code": `
    const result = $input.first().json.result;
    const restaurantRegex = /(\w+ \w+(?:\s+\w+)*)\s*⭐/g;
    const restaurants = [];
    let match;
    
    while ((match = restaurantRegex.exec(result)) !== null) {
      restaurants.push(match[1]);
    }
    
    return restaurants.map(name => ({ restaurant_name: name }));
  `
}

// 2. Check Availability (OpenTable API)
{
  "url": "https://api.opentable.com/availability",
  "method": "GET",
  "headers": {
    "Authorization": "Bearer {{$env.OPENTABLE_API_KEY}}"
  },
  "qs": {
    "restaurant": "{{$json.restaurant_name}}",
    "date": "{{$json.date}}",
    "time": "{{$json.time}}",
    "party_size": "{{$json.party_size}}"
  }
}

// 3. Make Reservation (if available)
{
  "url": "https://api.opentable.com/reservations", 
  "method": "POST",
  "body": {
    "restaurant_id": "{{$node['Check Availability'].json.restaurant_id}}",
    "date": "{{$json.date}}",
    "time": "{{$json.time}}",
    "party_size": "{{$json.party_size}}",
    "customer": {
      "name": "{{$json.customer_name}}",
      "email": "{{$json.customer_email}}",
      "phone": "{{$json.customer_phone}}"
    }
  }
}

// 4. Add to Google Calendar
{
  "calendarId": "primary",
  "summary": "Dinner at {{$node['Parse Restaurants'].json.restaurant_name}}",
  "start": {
    "dateTime": "{{$json.date}}T{{$json.time}}:00",
    "timeZone": "Europe/Paris"
  },
  "end": {
    "dateTime": "{{$json.date}}T{{moment($json.time, 'HH:mm').add(2, 'hours').format('HH:mm')}}:00",
    "timeZone": "Europe/Paris" 
  },
  "description": "Reservation confirmed via AI Assistant\nDetails: {{$node['Make Reservation'].json.confirmation_details}}"
}

// 5. Send Confirmation SMS
{
  "to": "{{$json.customer_phone}}",
  "body": "🍽️ Reservation confirmed at {{$node['Parse Restaurants'].json.restaurant_name}} for {{$json.date}} at {{$json.time}}. Confirmation: {{$node['Make Reservation'].json.confirmation_code}}"
}
```

#### **Workflow 3: Customer Feedback & Learning Loop (2 hours)**

**Automated Learning System:**
```javascript
// Daily Cron Trigger → Get Recent Reservations → 
// Send Survey → Collect Feedback → Update AI Training Data

// 1. Cron Trigger (daily at 10 AM)
{
  "triggerTimes": "0 10 * * *"
}

// 2. Get Yesterday's Reservations
{
  "code": `
    const yesterday = new Date();
    yesterday.setDate(yesterday.getDate() - 1);
    return [{ date: yesterday.toISOString().split('T')[0] }];
  `
}

// 3. Query Database for Reservations
{
  "query": "SELECT * FROM reservations WHERE date = '{{$json.date}}' AND status = 'completed'"
}

// 4. Send Feedback Survey Email
{
  "to": "{{$json.customer_email}}",
  "subject": "How was your dinner at {{$json.restaurant_name}}?",
  "html": `
    <h2>We'd love your feedback!</h2>
    <p>How was your experience at {{$json.restaurant_name}}?</p>
    <a href="{{$env.SURVEY_URL}}?reservation_id={{$json.id}}">Take 2-minute survey</a>
  `
}

// 5. Collect Responses (webhook endpoint)
// 6. Update AI Training Data
{
  "url": "http://localhost:8000/feedback/update-training",
  "method": "POST",
  "body": {
    "restaurant_name": "{{$json.restaurant_name}}",
    "rating": "{{$json.rating}}",
    "feedback": "{{$json.feedback}}",
    "query_context": "{{$json.original_query}}"
  }
}
```

---

## 🧪 **Integration Testing**

### **End-to-End Workflow Test**
```python
# Test complete LangGraph + N8N integration

async def test_complete_workflow():
    """Test full customer journey"""
    
    # 1. User query through LangGraph
    langgraph_result = await langgraph_agent.process_query(
        "romantic dinner tonight for anniversary near Notre Dame"
    )
    
    # 2. Trigger N8N workflow via webhook
    n8n_response = requests.post("http://localhost:5678/webhook/ai-recommendation", json={
        "query": "romantic dinner tonight for anniversary near Notre Dame",
        "location": "Paris",
        "customer_email": "test@example.com",
        "customer_phone": "+33123456789",
        "date": "2024-11-24",
        "time": "20:00",
        "party_size": 2
    })
    
    # 3. Verify complete automation
    assert n8n_response.status_code == 200
    print("✅ End-to-end workflow completed successfully!")
```

---

## 🎯 **Phase 4 Mastery Checklist**

### **LangGraph Agent Workflows ✅**
- [ ] Build decision trees for agent behavior
- [ ] Implement conditional routing logic
- [ ] Create parallel processing workflows
- [ ] Handle iterative improvement loops

### **N8N Business Automation ✅**
- [ ] Set up N8N self-hosted environment
- [ ] Connect AI agent to external systems
- [ ] Build complete customer journey workflows
- [ ] Implement feedback and learning loops

### **Integration Mastery ✅**
- [ ] Combine agent intelligence with business automation
- [ ] Handle real-world external API integrations
- [ ] Create scalable workflow templates
- [ ] Monitor and optimize workflow performance

---

## 🚀 **Business Value Created:**

### **Before Phase 4:**
- AI agent provides recommendations

### **After Phase 4:**
- **Complete business solution** with:
  - Intelligent agent decision making
  - Automated customer journey
  - External system integrations  
  - Feedback loops and learning
  - CRM integration and analytics

**Students build production-ready business automation around their AI agent!** 🎯