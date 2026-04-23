# 🎬 AutoStream Social-to-Lead Agent

A production-quality conversational AI agent for AutoStream (a SaaS video streaming platform) that detects user intent, answers questions using RAG, identifies high-intent buyers, and collects lead data — all orchestrated through a LangGraph workflow.

---

## 🚀 Setup Instructions

### 1. Clone and Navigate
```bash
git clone <repo-url>
cd social-to-lead-agent
```

### 2. Create a Virtual Environment
```bash
python -m venv venv
source venv/bin/activate       # macOS/Linux
venv\Scripts\activate          # Windows
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure API Key
Create a `.env` file in the root directory:
```
GEMINI_API_KEY=your_gemini_api_key_here
```

### 5. Run the Agent
```bash
python app.py
```

---

## 💬 How to Run

Once running, the CLI will prompt you to chat with the AutoStream assistant:

- Say **"Hi"** → triggers a greeting response
- Ask **"What's the pricing?"** → triggers RAG-based product answer
- Say **"I want to sign up"** → triggers the lead capture flow (name → email → platform)

**Example Session:**
```

```

---

## 🏗️ Architecture Explanation

The AutoStream agent is built as a modular, graph-based system with clear separation of concerns across six Python modules.

**Intent Detection** (`agent/intent.py`) uses GPT-4o-mini to classify every incoming message into one of three intents: `greeting`, `product_query`, or `high_intent`. This is purely LLM-based — no keyword matching.

**RAG System** (`agent/rag.py`) loads `data/knowledge.json`, converts each entry into an OpenAI embedding, and stores them in a FAISS vector index. On each product query, it retrieves the top-3 most semantically relevant chunks and injects them as context into the LLM prompt, ensuring answers are grounded in actual AutoStream knowledge.

**LangGraph Workflow** (`agent/graph.py`) orchestrates the entire flow as a directed graph: every user message enters at `intent_node`, then routes via conditional edges to one of `greeting_node`, `rag_node`, or `lead_node` based on the detected intent. Each node processes the shared `AgentState` and returns an updated version.

**Lead Capture** (`agent/nodes.py`, `agent/tools.py`) progressively collects name, email, and platform — one field per turn — using LLM extraction. The `mock_lead_capture()` function is called only after all three fields are confirmed, never prematurely.

---

## ❓ Why LangGraph?

LangGraph was chosen because:
1. **Stateful graphs**: It natively manages persistent state (`AgentState`) across multiple conversation turns without manual wiring.
2. **Conditional routing**: The `add_conditional_edges` API cleanly maps intent → node without messy if/else chains in the main loop.
3. **Composability**: Each node is an isolated function, making the system easy to test, extend, and debug.
4. **Production-ready**: LangGraph is designed for agentic workflows where execution paths vary per message — exactly what this use case requires.

---

## 🧠 How Memory Works

Memory is implemented via the `AgentState` TypedDict which persists across the entire session. The `messages` list accumulates every user message and assistant response in `{"role": ..., "content": ...}` format. Before each LLM call, the last 4–8 turns of history are injected into the prompt, giving the model full conversational context. Lead data fields (`name`, `email`, `platform`) also persist in state so partial collection carries forward between turns.

---

## 📚 How RAG Works

1. **Ingestion**: `data/knowledge.json` contains 10 curated chunks covering pricing, policies, features, and general info.
2. **Embedding**: Each chunk is converted to a vector using OpenAI's `text-embedding-3-small` model.
3. **Storage**: Vectors are stored in an in-memory FAISS index for fast similarity search.
4. **Retrieval**: On each `product_query`, the user's message is embedded and the top-3 most semantically similar chunks are fetched.
5. **Generation**: The retrieved chunks are injected into the LLM system prompt, grounding the response in factual knowledge and reducing hallucination.

---

## 📱 WhatsApp Integration via Webhooks

To deploy this agent on WhatsApp, you would use the **Meta WhatsApp Business API** with webhooks:

### Overview
1. **Webhook Setup**: Deploy the agent as a web server (FastAPI or Flask). Register your endpoint URL with Meta's WhatsApp Business Platform as a webhook receiver.
2. **Incoming Messages**: When a WhatsApp user sends a message, Meta sends a POST request to your webhook containing the message payload (sender ID, message text, timestamp).
3. **Agent Invocation**: Your webhook handler extracts the message, looks up or creates the user's persistent `AgentState` (keyed by their WhatsApp phone number, stored in Redis or a database), and invokes the LangGraph agent.
4. **Response Delivery**: The agent's reply is sent back to the user via the Meta Send Message API using `requests.post()` with the sender's phone number and the generated text.
5. **State Persistence**: For multi-turn WhatsApp conversations, `AgentState` must be stored externally (Redis/PostgreSQL) since each webhook call is stateless.

### Minimal FastAPI Webhook Example
```python
from fastapi import FastAPI, Request
import httpx, json, redis

app = FastAPI()
r = redis.Redis()

@app.post("/webhook")
async def whatsapp_webhook(request: Request):
    payload = await request.json()
    message = payload["entry"][0]["changes"][0]["value"]["messages"][0]
    sender = message["from"]
    text = message["text"]["body"]

    # Load persistent state for this user
    raw_state = r.get(f"state:{sender}")
    state = json.loads(raw_state) if raw_state else initialize_state()
    state["current_user_message"] = text

    # Run agent
    state = agent_graph.invoke(state)
    r.set(f"state:{sender}", json.dumps(state))

    # Send reply via WhatsApp API
    reply = state["messages"][-1]["content"]
    await send_whatsapp_message(sender, reply)
    return {"status": "ok"}
```

This architecture makes the agent fully production-ready for WhatsApp deployment with minimal changes to the core agent logic.

---

## 📁 Project Structure

```
social-to-lead-agent/
├── app.py                  # CLI entry point and chat loop
├── agent/
│   ├── __init__.py
│   ├── graph.py            # LangGraph workflow definition
│   ├── state.py            # AgentState TypedDict
│   ├── nodes.py            # Node functions (greeting, rag, lead)
│   ├── intent.py           # LLM-based intent classification
│   ├── rag.py              # FAISS vector store + RAG generation
│   └── tools.py            # Lead capture tool + LLM extraction
├── data/
│   └── knowledge.json      # AutoStream knowledge base (10 chunks)
├── requirements.txt
└── README.md
```

---

## 🔧 Tech Stack

| Component | Technology |
|-----------|-----------|
| Language | Python 3.9+ |
| LLM | gemini-robotics-er-1.5-preview |
| Workflow | LangGraph |
| RAG Framework | LangChain |
| Vector DB | FAISS (faiss-cpu) |
| Embeddings | gemini-embedding-001 |
| Config | python-dotenv |
