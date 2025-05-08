import torch
import json
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from bert_score import score

# === 加载 LoRA 模型 ===
model_id = "google/gemma-7b-it"
adapter_path = "./lora-gemma-adapter"

tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token

base_model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16, device_map="auto")
model = PeftModel.from_pretrained(base_model, adapter_path)
model.eval()

# === 加载测试集 ===
with open("./test_files/nyu_qa_testset_full.json") as f:
    dataset = json.load(f)

# === 生成回答并计算 BERTScore ===
candidates = []
references = []
results = []

for item in dataset:
    prompt = f"{item['instruction']}\n\n{item['input']}\n\nAnswer:"
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=64,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id
        )

    answer = tokenizer.decode(outputs[0], skip_special_tokens=True).split("Answer:")[-1].strip()

    candidates.append(answer)
    references.append(item["output"])

    results.append({
        "instruction": item["instruction"],
        "input": item["input"],
        "expected_output": item["output"],
        "model_output": answer
    })

# === BERTScore 相似度打分 ===
print("🔍 Computing BERTScore similarity...")
P, R, F1 = score(
    candidates,
    references,
    lang="en",
    model_type="microsoft/deberta-base",  # 替代 roberta-base，更好用
    verbose=True
)


# 添加分数到结果中
for i in range(len(results)):
    results[i]["bertscore_f1"] = round(F1[i].item(), 4)

# 打印平均分
avg_f1 = round(sum(F1[i].item() for i in range(len(F1))) / len(F1), 4)
print(f"📊 Average BERTScore F1: {avg_f1:.4f}")

# 保存结果
with open("nyu_qa_results_with_similarity.json", "w") as f:
    json.dump(results, f, indent=2)

print("✅ Results saved to lora_gemma_similarity.json")
