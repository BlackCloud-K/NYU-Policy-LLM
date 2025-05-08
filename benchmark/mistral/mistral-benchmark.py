import json
import subprocess
import os

def ask_ollama(prompt, model_name="mistral"):
    result = subprocess.run(
        ["ollama", "run", model_name],
        input=prompt.encode(),
        stdout=subprocess.PIPE
    )
    return result.stdout.decode()

def run_mcq_inference_batch(mcq_path, output_path):
    with open(mcq_path, "r") as f:
        dataset = json.load(f)

    results = []
    correct = 0
    total = len(dataset)

    for i, item in enumerate(dataset):
        prompt = f"Answer the following multiple choice question by selecting A, B, C, or D only.\n\n{item['question']}\n\nYour answer:"
        output = ask_ollama(prompt).strip()

        # 尝试提取模型的答案（如 C、B、D）
        extracted_answer = "?"
        for option in ["A", "B", "C", "D"]:
            if option in output:
                extracted_answer = option
                break

        result = {
            "question": item["question"],
            "expected": item["answer"],
            "model_output": output,
            "model_choice": extracted_answer
        }

        if extracted_answer == item["answer"]:
            correct += 1

        results.append(result)

    accuracy = correct / total
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    return results, accuracy

# -------------------------------
if __name__ == "__main__":

    file_names = [
        "admission_questions",
        "degree_questions",
        "honors_questions",
        "policy_data_questions",
        "standing_committee_questions",
        "transfer_questions"
    ]

    for file in file_names:
        mcq_path = f"./test_sets/{file}.json"
        output_path = f"./answers/mistral_benchmark/{file}_answers.json"

        print(f"\n📂 Running benchmark on {file} ...")
        results, acc = run_mcq_inference_batch(mcq_path, output_path)
        print(f"✅ Accuracy for {file}: {acc:.2%}")
