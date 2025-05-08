import json
import requests
from pathlib import Path
import os
from dotenv import load_dotenv
from typing import Dict, Tuple
import re

load_dotenv()
model_name = "gemma3"

def run_mcq_inference_batch(
    data_folder: str,
    mcq_path: str,
    output_path: str = "gemma_mcq_answers.json",
    model: str = "google/gemma-3-27b-it-fast",
    base_url: str = "https://router.huggingface.co/nebius/v1",
    max_tokens: int = 1024,
) -> Tuple[Dict[str, Dict[str, str]], float]:

    api_key = os.getenv("HF_TOKEN")
    if not api_key:
        raise ValueError("Please set the HF_TOKEN environment variable.")

    # Load context from all JSON files
    context_parts = []
    for json_file in Path(data_folder).glob("*.json"):
        with open(json_file, "r", encoding="utf-8") as f:
            content = json.load(f)
            context_parts.append(json.dumps(content, ensure_ascii=False))
    full_context = "\n\n".join(context_parts)

    # Load MCQ test set with answers
    with open(mcq_path, "r", encoding="utf-8") as f:
        question_items = json.load(f)

    questions = [item["question"] for item in question_items]

    # Combine all questions into one block
    all_questions_text = "\n\n".join([f"Q{i+1}: {q}" for i, q in enumerate(questions)])

    prompt = f"""You are a helpful assistant. Based on the following context, answer all the questions below. Only return the selected options in the format: Q1: A, Q2: B, Q3: C, etc.

        Context:
        {full_context}

        Questions:
        {all_questions_text}

        Answers:"""

    headers = {"Authorization": f"Bearer {api_key}"}
    url = f"{base_url}/chat/completions"

    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
    }

    response = requests.post(url, headers=headers, json=payload)

    evaluation = {}
    correct_count = 0
    if response.status_code == 200:
        try:
            answer_block = response.json()["choices"][0]["message"]["content"].strip()
            model_answers = dict(re.findall(r"Q(\d+):\s*([A-D])", answer_block))

            for i, item in enumerate(question_items):
                qid = f"Q{i+1}"
                correct = item["answer"].strip().upper()
                predicted = model_answers.get(str(i + 1), "N/A").upper()
                is_correct = predicted == correct
                if is_correct:
                    correct_count += 1
                evaluation[qid] = {
                    "question": item["question"],
                    "predicted": predicted,
                    "ground_truth": correct,
                    "is_correct": is_correct
                }
        except Exception as e:
            evaluation["error"] = {"message": f"Error parsing response: {e}"}
    else:
        evaluation["error"] = {"message": f"Request failed: {response.status_code} - {response.text}"}

    # Save results to JSON file
    with open(output_path, "w", encoding="utf-8") as out_file:
        json.dump(evaluation, out_file, indent=2, ensure_ascii=False)

    total = len(question_items)
    accuracy = correct_count / total if total > 0 else 0.0
    print(f"Accuracy: {accuracy:.2%} ({correct_count}/{total})")

    return evaluation, accuracy


if __name__ == "__main__":

    file_names = ["admission_questions", "degree_questions", "honors_questions", "policy_data_questions", "standing_committee_questions", "transfer_questions"]
    data_folder = "../data/policy_files"
    for file in file_names:
        mcq_path = f"./test_sets/{file}.json"
        output_path = f"./answers/{model_name}/{file}_answers.json"

        results, acc = run_mcq_inference_batch(data_folder, mcq_path, output_path)
