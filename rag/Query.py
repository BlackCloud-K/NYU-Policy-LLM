from openai import OpenAI
import requests
import json
import os
from dotenv import load_dotenv
from ragflow_sdk import RAGFlow

load_dotenv()
API_KEY = os.getenv("API_KEY")
API_URL = "http://127.0.0.1"
CHAT_ID = os.getenv("CHAT_ID")


def get_dataset_id_by_name(name):
    """find dataset ID"""
    headers = {"Authorization": f"Bearer {API_KEY}"}
    url = f"{API_URL}/api/v1/datasets"
    params = {'name': name}
    
    response = requests.get(url, headers=headers, params=params)
    
    if response.status_code != 200:
        print(f"Request Failed: {response.status_code} - {response.text}")
        return None
    
    data = response.json()
    if data.get('code') != 0:
        print(f"API ERROR: {data.get('message')}")
        return None
    
    datasets = data.get('data', [])
    if not datasets:
        print(f"DATASET NOT FOUND: {name}")
        return None
    
    return datasets[0]['id']


def ask_one_question(question, prompt, stream=False):
    model = "model"
    client = OpenAI(api_key=API_KEY, base_url=f"{API_URL}/api/v1/chats_openai/{CHAT_ID}")
    # print(prompt)
    # print(CHAT_ID)

    completion = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": question},
        ],
        stream=stream
    )

    if stream:
        for chunk in completion:
            return chunk
    else:
        if completion.choices:
            return completion.choices[0].message.content
        else:
            print(f"NO RESULT: {question}")
            return None
    

def ask_questions(questions):
    result = []
    prompt = ""
    # chat_id = get_dataset_id_by_name(dataset_name)

    with open("prompt.txt", 'r', encoding='utf-8') as file:
        prompt = file.read()
        file.close()

    for question in questions:
        answer = ask_one_question(question, prompt)
        result.append(answer)

    return result


def update_dataset(dataset_names):
    dataset_ids = []

    for dataset_name in dataset_names:
        dataset_id = get_dataset_id_by_name(dataset_name)
        if not dataset_id:
            return None
        dataset_ids.append(dataset_id)
    
    # print(dataset_ids)
    rag_object = RAGFlow(api_key=API_KEY, base_url=API_URL)
    chat_list = rag_object.list_chats(id = CHAT_ID)
    if not chat_list:
        print("CAN NOT FIND CURRENT CHAT")
        return None
    
    chat = chat_list[0]
    chat.update({"dataset_ids": dataset_ids})
    print(f"Updated successfully: {chat.dataset_ids}")


def parse_input():
    questions = []
    answers = []
    input = []

    with open("input.json", 'r', encoding='utf-8') as f:
        input = json.load(f)
    
    questions = [item["question"] for item in input]
    answers = [item["answer"] for item in input]

    return (questions, answers)


def main():
    # dataset_names = ["NYU CAS Policy", "NYU CAS Admission"]
    # update_dataset(["NYU CAS Policy"])

    questions, answers = parse_input()
    res = ask_questions(questions)
    print(res)


if __name__ == '__main__':
    main()
