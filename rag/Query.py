from openai import OpenAI
import requests
import json
import os
from dotenv import load_dotenv
from ragflow_sdk import RAGFlow
from tqdm import tqdm
from time import sleep

load_dotenv()
API_KEY = os.getenv("API_KEY")
API_URL = "http://127.0.0.1"
CHAT_ID = os.getenv("CHAT_ID_GM3")
SLEEP_TIME = 5
file_names = ["admission_questions", "degree_questions", "honors_questions", "policy_data_questions", "standing_committee_questions", "transfer_questions"]
model_name = "gemma"


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
            {"role": "user", "content": prompt + question},
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

    for question in tqdm(questions):
        answer = ask_one_question(question, prompt)
        result.append(answer)
        # print(answer)
        sleep(SLEEP_TIME)

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


def parse_input(test_set_name):
    questions = []
    answers = []
    input = []

    with open(f"test_sets/{test_set_name}.json", 'r', encoding='utf-8') as f:
        input = json.load(f)
    
    questions = [item["question"] for item in input]
    answers = [item["answer"] for item in input]

    return (questions, answers)


def save_result(res, test_set_name):
    with open(f"test_sets/{test_set_name}.json", 'r', encoding='utf-8') as f:
        questions = json.load(f)

    merged = []
    n_answered, n_correct = 0, 0

    for q, model_ans in zip(questions, res):
        if model_ans is None:
            print(f"Warning: model_ans is None for question {q}")
            correction = -1
        elif model_ans.strip()[0].upper() == q["answer"].strip()[0].upper():
            correction = 1
            n_answered += 1
            n_correct += 1
        elif model_ans.startswith("**ERROR**"):
            correction = -1
        else:
            correction = 0
            n_answered += 1

        merged.append({
            "question": q["question"],
            "correct_answer": q["answer"],
            "model_answer": model_ans,
            "is_correct": correction
        })

    acc = n_correct * 100 / n_answered
    print("Result saved")
    print(f"{test_set_name} Acc: {acc: .2f}")

    with open(f"answers/{model_name}/{test_set_name}.json", 'w', encoding='utf-8') as f:
        json.dump(merged, f, indent=2, ensure_ascii=False)


def main():
    # dataset_names = ["NYU CAS Policy", "NYU CAS Admission"]
    # update_dataset(["NYU CAS Policy"])
    for test_set_name in file_names:
        questions, answers = parse_input(test_set_name)
        res = ask_questions(questions)
        # print(res)
        save_result(res, test_set_name)


if __name__ == '__main__':
    main()
