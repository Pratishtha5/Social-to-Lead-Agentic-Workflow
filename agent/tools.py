"""
Tool definitions for the AutoStream agent.
Gemini-compatible version (no OpenAI dependency).
"""

from langchain.schema import HumanMessage, SystemMessage


def mock_lead_capture(name: str, email: str, platform: str) -> dict:
    print(f"\n{'='*50}")
    print(f"✅ Lead captured successfully: {name}, {email}, {platform}")
    print(f"{'='*50}\n")

    return {
        "status": "success",
        "lead": {
            "name": name,
            "email": email,
            "platform": platform
        }
    }


LEAD_COLLECTION_PROMPT = """You are a friendly sales assistant for AutoStream.

Current state:
- Name: {name}
- Email: {email}
- Platform: {platform}

Conversation:
{history}

User: {user_message}

Instructions:
- Acknowledge user naturally
- Ask ONLY next missing field
- Be human, warm, not robotic
"""


def generate_lead_response(
    user_message,
    conversation_history,
    name,
    email,
    platform,
    llm
) -> str:

    history_text = ""
    for turn in conversation_history[-6:]:
        role = "User" if turn["role"] == "user" else "Assistant"
        history_text += f"{role}: {turn['content']}\n"

    prompt = LEAD_COLLECTION_PROMPT.format(
        name=name or "None",
        email=email or "None",
        platform=platform or "None",
        history=history_text,
        user_message=user_message
    )

    response = llm.invoke([HumanMessage(content=prompt)])
    return response.content.strip()


def extract_lead_field(user_message, field, llm) -> str:

    prompts = {
        "name": "Extract ONLY the person's name. If none, return NONE.",
        "email": "Extract ONLY the email. If none, return NONE.",
        "platform": "Extract ONLY platform/source (YouTube, Instagram, etc). If none, return NONE."
    }

    response = llm.invoke([
        SystemMessage(content=prompts[field]),
        HumanMessage(content=user_message)
    ])

    result = response.content.strip()

    return "" if result.upper() == "NONE" else result