from dotenv import load_dotenv
import os

from langchain_google_genai import ChatGoogleGenerativeAI

from agent.rag import RAGSystem
from agent.graph import build_graph
from agent.state import AgentState

load_dotenv()

KNOWLEDGE_PATH = os.path.join(os.path.dirname(__file__), "data", "knowledge.json")


def initialize_state() -> AgentState:
    return {
        "messages": [],
        "intent": None,
        "name": "",
        "email": "",
        "platform": "",
        "lead_mode":False,
        "lead_captured": False,
        "current_user_message": ""
    }


def run_chat():
    api_key = os.getenv("GOOGLE_API_KEY")

    if not api_key:
        print("❌ GOOGLE_API_KEY not found in .env")
        return

    print("\n🎬 AutoStream AI Assistant (Gemini + LangGraph + RAG)\n")

    llm = ChatGoogleGenerativeAI(
        model="models/gemini-robotics-er-1.5-preview",
        temperature=0.7
    )

    print("⚙️ Loading knowledge base...")
    rag_system = RAGSystem(knowledge_path=KNOWLEDGE_PATH)
    print("✅ Ready!\n")

    graph = build_graph(llm, rag_system)
    state = initialize_state()

    while True:
        user_input = input("You: ")

        if user_input.lower() in ["exit", "quit"]:
            print("Assistant: Goodbye! 👋")
            break

        state["current_user_message"] = user_input
        state = graph.invoke(state)

        if state["messages"]:
            print("Assistant:", state["messages"][-1]["content"], "\n")


if __name__ == "__main__":
    run_chat()