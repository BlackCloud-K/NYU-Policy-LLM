from openai import OpenAI
from ragflow_sdk import RAGFlow
from dotenv import load_dotenv
import os

load_dotenv()
API_KEY = os.getenv("API_KEY")
API_URL = "http://127.0.0.1"
CHAT_ID = os.getenv("CHAT_ID")

rag_object = RAGFlow(api_key=API_KEY, base_url=API_URL)
chat = rag_object.list_chats(id = CHAT_ID)[0]
chat.update({"name": "Gemini 2"})

print(chat)
