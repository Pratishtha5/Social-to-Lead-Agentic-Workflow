from langchain.schema import HumanMessage, SystemMessage

from agent.intent import detect_intent
from agent.tools import mock_lead_capture


def make_intent_node(llm):
    def node(state):
        intent = detect_intent(
            state["current_user_message"],
            state["messages"],
            llm
        )
        state["intent"] = intent
        return state
    return node


def make_greeting_node(llm):
    def node(state):
        reply = "Hey! 👋 I can help you with AutoStream pricing, features, or getting started!"

        state["messages"].append({"role": "user", "content": state["current_user_message"]})
        state["messages"].append({"role": "assistant", "content": reply})

        return state
    return node


def make_rag_node(rag, llm):
    def node(state):
        answer = rag.answer(
            state["current_user_message"],
            state["messages"],
            llm
        )

        state["messages"].append({"role": "user", "content": state["current_user_message"]})
        state["messages"].append({"role": "assistant", "content": answer})

        return state
    return node


def _agent_already_asked(state, keyword):
    """Check if the assistant has already asked for a specific field."""
    return any(
        keyword in msg.get("content", "").lower()
        for msg in state["messages"]
        if msg["role"] == "assistant"
    )


def make_lead_node(llm):
    def node(state):
        msg = state["current_user_message"]

        if not state["name"]:
            if _agent_already_asked(state, "name"):
                # Agent already asked "what's your name?" — capture the answer
                state["name"] = msg
                reply = f"Nice to meet you, {msg}! What's your email address?"
            else:
                # First time in lead node — ask for name first
                reply = "Awesome, let's get you set up! What's your name?"

        elif not state["email"]:
            if _agent_already_asked(state, "email"):
                state["email"] = msg
                reply = "Got it! Which platform do you mainly create content on? (e.g. YouTube, Instagram, TikTok)"
            else:
                reply = f"Nice to meet you, {state['name']}! What's your email address?"

        elif not state["platform"]:
            if _agent_already_asked(state, "platform") or _agent_already_asked(state, "create content"):
                state["platform"] = msg
                mock_lead_capture(state["name"], state["email"], state["platform"])
                reply = f"You're all set, {state['name']}! 🚀 Our team will reach out to {state['email']} shortly. Welcome to AutoStream Pro!"
            else:
                reply = f"Got it! Which platform do you mainly create content on? (e.g. YouTube, Instagram, TikTok)"

        else:
            reply = "We've already got your details — our team will be in touch soon! 😊"

        state["messages"].append({"role": "user", "content": msg})
        state["messages"].append({"role": "assistant", "content": reply})

        return state

    return node