from langchain.schema import HumanMessage, SystemMessage

PROMPT = """Classify intent:

greeting
product_query
high_intent

Return ONLY one word.
"""


def detect_intent(user_message, history, llm):

    messages = [
        SystemMessage(content=PROMPT),
        HumanMessage(content=user_message)
    ]

    res = llm.invoke(messages)
    intent = res.content.strip().lower()

    if intent not in ["greeting", "product_query", "high_intent"]:
        return "product_query"

    return intent