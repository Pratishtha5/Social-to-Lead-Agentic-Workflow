"""
State definition for the AutoStream conversational agent.
Shared across all LangGraph nodes throughout the conversation lifecycle.
"""

from typing import TypedDict, Optional, List, Dict, Any


class AgentState(TypedDict):
    """
    Persistent state object passed through all LangGraph nodes.
    Maintains full context across multi-turn conversations.
    """
    messages: List[Dict[str, str]]      # Full conversation history [{role, content}]
    intent: Optional[str]               # Detected intent: greeting | product_query | high_intent
    name: Optional[str]                 # Collected lead name
    email: Optional[str]                # Collected lead email
    platform: Optional[str]            # Collected lead platform/source
    lead_captured: bool    
    lead_mode=bool                    # Whether the lead tool has been called
    current_user_message: str          # The latest user message being processed
