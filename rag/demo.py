from openai import OpenAI

model = "model"
client = OpenAI(api_key="ragflow-gxZDExNjZjMWQ0ZDExZjBiN2RiNWFlZT", base_url=f"http://127.0.0.1//api/v1/chats_openai/5050af121d3e11f0a2935aee9372892d")
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