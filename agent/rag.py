import json
from typing import List

from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.schema import Document, HumanMessage, SystemMessage, AIMessage


RAG_PROMPT = """You are a helpful assistant for AutoStream.

Use the context below to answer:

{context}
"""


class RAGSystem:

    def __init__(self, knowledge_path: str):
        self.embeddings = GoogleGenerativeAIEmbeddings(
            model="models/gemini-embedding-001"
        )
        self.vector_store = self._build_vector_store(knowledge_path)

    def _build_vector_store(self, path: str):
        with open(path, "r") as f:
            data = json.load(f)

        docs = []

        for item in data:
            docs.append(Document(page_content=item["content"]))

        return FAISS.from_documents(docs, self.embeddings)

    def retrieve(self, query: str, k: int = 3):
        return self.vector_store.similarity_search(query, k=k)

    def answer(self, user_message, history, llm):
        docs = self.retrieve(user_message)

        context = "\n".join([d.page_content for d in docs])

        messages = [SystemMessage(content=RAG_PROMPT.format(context=context))]

        for turn in history[-6:]:
            if turn["role"] == "user":
                messages.append(HumanMessage(content=turn["content"]))
            else:
                messages.append(AIMessage(content=turn["content"]))

        messages.append(HumanMessage(content=user_message))

        res = llm.invoke(messages)
        return res.content.strip()