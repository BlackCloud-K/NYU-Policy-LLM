import torch
import json
from transformers import AutoTokenizer, AutoModelForCausalLM
from bert_score import score

# === 加载原始 Mistral 模型（未加 LoRA）===
model_id = "google/gemma-7b-it"

tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    model_id, torch_dtype=torch.float16, device_map="auto"
)
model.eval()

# === 加载测试集 ===
with open("./test_files/nyu_qa_testset_full.json") as f:
    dataset = json.load(f)

# === 生成 baseline 回答 ===
results = []
candidates = []
references = []

print("🧠 Generating baseline model answers...")
for item in dataset:
    prompt = f"{item['instruction']}\n\n{item['input']}\n\nAnswer:"
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=64,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id
        )

    answer = tokenizer.decode(output_ids[0], skip_special_tokens=True).split("Answer:")[-1].strip()
    gt = item["output"]

    candidates.append(answer)
    references.append(gt)

    results.append({
        "input": item["input"],
        "expected_output": gt,
        "model_output": answer
    })

# === 计算 BERTScore ===
print("🔍 Computing BERTScore...")
P, R, F1 = score(candidates, references, lang="en", model_type="microsoft/deberta-base", verbose=True)

# === 保存结果 ===
for i, r in enumerate(results):
    r["bertscore_f1"] = round(F1[i].item(), 4)

avg_f1 = round(sum(f.item() for f in F1) / len(F1), 4)
print(f"\n📊 Baseline Average BERTScore F1: {avg_f1}")

with open("baseline_mistral_scored.json", "w") as f:
    json.dump(results, f, indent=2)

print("✅ Saved baseline model results to baseline_mistral_scored.json")
