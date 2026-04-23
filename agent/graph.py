"""
LangGraph workflow definition for the AutoStream Social-to-Lead agent.
Defines the graph structure: nodes, edges, and conditional routing.
"""

from langgraph.graph import StateGraph, END

from agent.state import AgentState
from agent.nodes import make_intent_node, make_greeting_node, make_rag_node, make_lead_node
from agent.rag import RAGSystem


def route_by_intent(state: AgentState) -> str:
    """
    Routing function: directs flow based on detected intent.
    
    Returns:
        Node name to route to next.
    """
    intent = state.get("intent", "product_query")

    if intent == "greeting":
        return "greeting_node"
    elif intent == "high_intent":
        return "lead_node"
    else:
        # Default to RAG for product queries and unknown intents
        return "rag_node"


def build_graph(llm, rag_system: RAGSystem) -> StateGraph:
    """
    Construct and compile the LangGraph workflow.
    
    Graph flow:
        START → intent_node → [router] → greeting_node / rag_node / lead_node → END
    
    Args:
        llm: Initialized ChatOpenAI LLM instance.
        rag_system: Initialized RAGSystem with FAISS vector store.
    
    Returns:
        Compiled LangGraph StateGraph ready to invoke.
    """
    # Initialize graph with our shared state schema
    graph = StateGraph(AgentState)

    # Register all nodes using factory functions
    graph.add_node("intent_node", make_intent_node(llm))
    graph.add_node("greeting_node", make_greeting_node(llm))
    graph.add_node("rag_node", make_rag_node(rag_system, llm))
    graph.add_node("lead_node", make_lead_node(llm))

    # Entry point: always detect intent first
    graph.set_entry_point("intent_node")

    # Conditional routing from intent node based on detected intent
    graph.add_conditional_edges(
        "intent_node",
        route_by_intent,
        {
            "greeting_node": "greeting_node",
            "rag_node": "rag_node",
            "lead_node": "lead_node"
        }
    )

    # All terminal nodes end the graph turn
    graph.add_edge("greeting_node", END)
    graph.add_edge("rag_node", END)
    graph.add_edge("lead_node", END)

    return graph.compile()
