"""
Local Discovery Agent implementation
"""

import logging
import re
import json
import time
from typing import Optional, List, Dict, Any
from datetime import datetime

from langchain_ollama import ChatOllama
from langchain.agents import AgentExecutor, create_react_agent
from langchain.memory import ConversationBufferMemory, ConversationBufferWindowMemory, ConversationSummaryMemory
from langchain.prompts import PromptTemplate

from app.config import settings
from app.models.schemas import PlaceResult, MemoryType
from .tools import AGENT_TOOLS
from .prompts import REACT_PROMPT_TEMPLATE

logger = logging.getLogger(__name__)

class LocalDiscoveryAgent:
    """Production-grade Local Discovery Agent using local LLMs"""
    
    def __init__(self, model_name: Optional[str] = None, memory_type: str = "buffer"):
        """
        Initialize the Local Discovery Agent
        
        Args:
            model_name: Ollama model to use (defaults to settings.ollama_model)
            memory_type: Type of memory to use ("buffer", "window", "summary")
        """
        self.model_name = model_name or settings.ollama_model
        self.memory_type = memory_type
        
        # Initialize LLM
        self.llm = self._initialize_llm()
        
        # Initialize memory
        self.memory = self._initialize_memory(memory_type)
        
        # Initialize agent
        self.agent_executor = self._initialize_agent()
        
        logger.info(f"LocalDiscoveryAgent initialized with model: {self.model_name}, memory: {memory_type}")
    
    def _initialize_llm(self) -> ChatOllama:
        """Initialize the Ollama LLM"""
        return ChatOllama(
            model=self.model_name,
            base_url=settings.ollama_base_url,
            temperature=settings.agent_temperature,
            timeout=settings.agent_timeout
        )
    
    def _initialize_memory(self, memory_type: str):
        """Initialize conversation memory"""
        if memory_type == "window":
            return ConversationBufferWindowMemory(
                k=settings.max_memory_messages,
                memory_key="chat_history",
                return_messages=True
            )
        elif memory_type == "summary":
            return ConversationSummaryMemory(
                llm=self.llm,
                memory_key="chat_history",
                return_messages=True
            )
        else:  # Default to buffer
            return ConversationBufferMemory(
                memory_key="chat_history",
                return_messages=True
            )
    
    def _initialize_agent(self) -> AgentExecutor:
        """Initialize the ReAct agent with tools"""
        # Create prompt template
        prompt = PromptTemplate(
            template=REACT_PROMPT_TEMPLATE,
            input_variables=["input", "agent_scratchpad"],
            partial_variables={
                "tools": "\n".join([f"{tool.name}: {tool.description}" for tool in AGENT_TOOLS]),
                "tool_names": ", ".join([tool.name for tool in AGENT_TOOLS])
            }
        )
        
        # Create ReAct agent
        agent = create_react_agent(
            llm=self.llm,
            tools=AGENT_TOOLS,
            prompt=prompt
        )
        
        # Create agent executor
        return AgentExecutor.from_agent_and_tools(
            agent=agent,
            tools=AGENT_TOOLS,
            memory=self.memory,
            verbose=settings.debug,
            max_iterations=4,  # Allow complex workflows: 1) First tool 2) Second tool 3) Analysis 4) Response
            max_execution_time=settings.agent_timeout,
            handle_parsing_errors=True
        )
    
    def search_places(self, query: str, location: Optional[str] = None, fast_mode: bool = False) -> Dict[str, Any]:
        """
        Search for places using the agent
        
        Args:
            query: Search query
            location: Optional location context
            fast_mode: If True, bypass agent and use direct tool calls for faster response
            
        Returns:
            Dictionary with success status, response, places, and metadata
        """
        start_time = time.time()
        
        try:
            # Prepare the input
            if location:
                input_text = f"Find {query} in {location}"
            else:
                input_text = query
            
            logger.info(f"Processing search request: {input_text}")
            
            # Add timing checkpoints
            agent_start_time = time.time()
            logger.info(f"[TIMING] Starting agent processing at {agent_start_time - start_time:.3f}s")
            
            # Normal mode: Use the agent for rich natural language response
            result = self.agent_executor.invoke({"input": input_text})
            agent_end_time = time.time()
            logger.info(f"[TIMING] Agent processing completed in {agent_end_time - agent_start_time:.3f}s")
            
            response_text = result.get("output", "")
            
            # Parse results timing
            parse_start_time = time.time()
            places = self._extract_places_from_response(response_text)
            parse_end_time = time.time()
            logger.info(f"[TIMING] Response parsing completed in {parse_end_time - parse_start_time:.3f}s")
            
            # If no structured data found, try direct tool approach as fallback
            if not places:
                fallback_start_time = time.time()
                logger.info(f"[TIMING] Starting fallback tool call at {fallback_start_time - start_time:.3f}s")
                try:
                    from .tools import search_places as search_tool
                    tool_result = search_tool.invoke({"query": f"{query} in {location or 'Paris'}"})
                    
                    if tool_result:
                        try:
                            import json
                            data = json.loads(tool_result)
                            if "places" in data:
                                for place_data in data["places"]:
                                    try:
                                        place = PlaceResult(**place_data)
                                        places.append(place)
                                    except Exception as e:
                                        logger.warning(f"Failed to parse place data: {e}")
                                        continue
                        except json.JSONDecodeError:
                            logger.warning("Fallback tool returned non-JSON data")
                except Exception as e:
                    logger.warning(f"Fallback tool call failed: {e}")
                
                fallback_end_time = time.time()
                logger.info(f"[TIMING] Fallback tool call completed in {fallback_end_time - fallback_start_time:.3f}s")
            
            processing_time = time.time() - start_time
            logger.info(f"[TIMING] Total processing time: {processing_time:.3f}s")
            
            return {
                "success": True,
                "response": response_text,
                "places": places,
                "query": input_text,
                "processing_time": processing_time
            }
            
        except Exception as e:
            logger.error(f"Error in search_places: {e}")
            processing_time = time.time() - start_time
            
            return {
                "success": False,
                "error": str(e),
                "places": [],
                "query": query,
                "processing_time": processing_time
            }
    
    def _extract_places_from_response(self, response_text: str) -> List[PlaceResult]:
        """Extract structured place data from agent response"""
        places = []
        
        try:
            # Look for JSON blocks in the response
            json_pattern = r'\{[^{}]*"places"[^{}]*\[[^\]]*\][^{}]*\}'
            json_matches = re.findall(json_pattern, response_text, re.DOTALL)
            
            for match in json_matches:
                try:
                    data = json.loads(match)
                    if "places" in data:
                        for place_data in data["places"]:
                            try:
                                place = PlaceResult(**place_data)
                                places.append(place)
                            except Exception as e:
                                logger.warning(f"Failed to parse place data: {e}, data: {place_data}")
                                continue
                        break  # Use first valid JSON block
                except json.JSONDecodeError:
                    continue
            
            # If no JSON found, try to parse from text patterns
            if not places:
                places = self._parse_places_from_text(response_text)
            
        except Exception as e:
            logger.warning(f"Error extracting places from response: {e}")
        
        return places
    
    def _parse_places_from_text(self, text: str) -> List[PlaceResult]:
        """Parse places from plain text response (fallback method)"""
        places = []
        
        # Simple regex patterns to extract place information
        name_pattern = r'(?:Name|Title):\s*([^\n]+)'
        rating_pattern = r'(?:Rating|Stars?):\s*([0-9.]+)'
        address_pattern = r'(?:Address|Location):\s*([^\n]+)'
        
        names = re.findall(name_pattern, text, re.IGNORECASE)
        ratings = re.findall(rating_pattern, text, re.IGNORECASE)
        addresses = re.findall(address_pattern, text, re.IGNORECASE)
        
        # Create place objects from extracted data
        max_places = max(len(names), len(ratings), len(addresses))
        for i in range(max_places):
            place_data = {}
            
            if i < len(names):
                place_data["name"] = names[i].strip()
            if i < len(ratings):
                try:
                    place_data["rating"] = float(ratings[i])
                except ValueError:
                    pass
            if i < len(addresses):
                place_data["address"] = addresses[i].strip()
            
            if place_data:
                try:
                    places.append(PlaceResult(**place_data))
                except Exception as e:
                    logger.warning(f"Failed to create PlaceResult from text parsing: {e}, data: {place_data}")
                    continue
        
        return places
    
    def get_conversation_history(self) -> List[Dict[str, str]]:
        """Get conversation history"""
        try:
            # Extract messages from memory
            messages = []
            if hasattr(self.memory, 'chat_memory') and hasattr(self.memory.chat_memory, 'messages'):
                for msg in self.memory.chat_memory.messages:
                    messages.append({
                        "role": msg.__class__.__name__.lower().replace("message", ""),
                        "content": msg.content,
                        "timestamp": datetime.now().isoformat()
                    })
            return messages
        except Exception as e:
            logger.error(f"Error getting conversation history: {e}")
            return []
    
    def clear_memory(self):
        """Clear conversation memory"""
        try:
            self.memory.clear()
            logger.info("Conversation memory cleared")
        except Exception as e:
            logger.error(f"Error clearing memory: {e}")
    
    def switch_memory_type(self, memory_type: str):
        """Switch to a different memory type"""
        try:
            self.memory_type = memory_type
            self.memory = self._initialize_memory(memory_type)
            self.agent_executor = self._initialize_agent()
            logger.info(f"Switched to {memory_type} memory type")
        except Exception as e:
            logger.error(f"Error switching memory type: {e}")
            raise
    
    def get_memory_info(self) -> Dict[str, Any]:
        """Get information about current memory"""
        try:
            message_count = 0
            
            # Handle different memory types
            if self.memory_type == "summary":
                # For summary memory, count both the summary and recent messages
                if hasattr(self.memory, 'chat_memory') and hasattr(self.memory.chat_memory, 'messages'):
                    message_count = len(self.memory.chat_memory.messages)
                # Add 1 for the summary if it exists
                if hasattr(self.memory, 'moving_summary_buffer') and self.memory.moving_summary_buffer:
                    message_count += 1
            else:
                # For buffer and window memory
                if hasattr(self.memory, 'chat_memory') and hasattr(self.memory.chat_memory, 'messages'):
                    message_count = len(self.memory.chat_memory.messages)
            
            return {
                "type": self.memory_type,
                "description": f"Using {self.memory_type} memory for conversation context",
                "message_count": message_count
            }
        except Exception as e:
            logger.error(f"Error getting memory info: {e}")
            return {
                "type": "unknown",
                "description": "Memory information unavailable",
                "message_count": 0
            }