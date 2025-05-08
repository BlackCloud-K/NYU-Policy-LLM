import json
import subprocess
import os
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import torch

# Load the fine-tuned model and tokenizer once
base_model_id = "google/gemma-7b-it"
adapter_path = "./lora-gemma-adapter"

tokenizer = AutoTokenizer.from_pretrained(base_model_id)
tokenizer.pad_token = tokenizer.eos_token

base_model = AutoModelForCausalLM.from_pretrained(
    base_model_id, torch_dtype=torch.float16, device_map="auto"
)
model = PeftModel.from_pretrained(base_model, adapter_path)
model.eval()

def ask_finetuned_model(prompt):
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=64,
            temperature=0.7,
            top_p=0.9,
            do_sample=True
        )
    full_output = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # 去掉前面的 prompt
    if full_output.startswith(prompt):
        return full_output[len(prompt):].strip()
    else:
        return full_output.strip()


def ask_model(prompt):
    return ask_finetuned_model(prompt).strip()

import re

def extract_choice(output: str) -> str:
    output = output.strip()

    # 优先匹配格式化明确答案
    patterns = [
        r"\*\*?Answer\*\*?:?\s*([A-D])\b",     # "**Answer:** B"
        r"Answer\s*[:：]?\s*([A-D])\b",         # "Answer: C"
        r"The correct answer is\s*([A-D])\b",   # "The correct answer is D"
        r"\b([A-D])\b"                          # standalone A/B/C/D
    ]
    for pattern in patterns:
        matches = re.findall(pattern, output, flags=re.IGNORECASE)
        if matches:
            return matches[-1].upper()  # 取最后一个，避免前面题目中的干扰

    return "UNKNOWN"


def run_mcq_inference_batch(mcq_path, output_path):
    with open(mcq_path, "r") as f:
        dataset = json.load(f)

    results = []
    correct = 0
    total = len(dataset)

    for i, item in enumerate(dataset):
        prompt = f"Answer the following multiple choice question by selecting A, B, C, or D only, no need to explain. IMPORTANT: Output ONLY the letter (A, B, C, or D). Do NOT include any explanation. \n\n{item['question']}\n\nYour Answer:"
        output = ask_model(prompt)
        choice = extract_choice(output)
        result = {
            "question": item["question"],
            "expected": item["answer"],
            "model_output": output,
            "model_choice": choice
        }

        if choice == item["answer"]:
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
        mcq_path = f"./test_files/{file}.json"
        output_path = f"./answers/gemma_benchmark/{file}_answers.json"

        print(f"\n📂 Running benchmark on {file} ...")
        results, acc = run_mcq_inference_batch(mcq_path, output_path)
        print(f"✅ Accuracy for {file}: {acc:.2%}")
