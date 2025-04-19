from openai import OpenAI
import os
from dotenv import load_dotenv


load_dotenv()
API_KEY = os.getenv("API_KEY")
CHAT_ID = os.getenv("CHAT_ID")
API_URL = "http://127.0.0.1"

model = "model"
client = OpenAI(api_key=API_KEY, base_url=f"{API_URL}/api/v1/chats_openai/{CHAT_ID}")
prompt = ""

with open("prompt.txt", 'r', encoding='utf-8') as file:
    prompt = file.read()
    file.close()

completion = client.chat.completions.create(
    model=model,
    messages=[
        {"role": "system", "content": prompt},
        {"role": "user", "content": "What is the minimum GPA required for a student to be eligible for Departmental Honors in the College of Arts and Science?"},
    ],
    stream=False
)

stream = False
if stream:
    for chunk in completion:
        print(chunk)
else:
    print(completion.choices[0].message.content)