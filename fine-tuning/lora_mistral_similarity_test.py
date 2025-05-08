# generate_and_score.py
import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Load model + tokenizer
base_model = "mistralai/Mistral-7B-Instruct-v0.2"
adapter_path = "./lora-llama3-adapter"
tokenizer = AutoTokenizer.from_pretrained(base_model)
tokenizer.pad_token = tokenizer.eos_token
base = AutoModelForCausalLM.from_pretrained(base_model, torch_dtype=torch.float16, device_map="auto")
model = PeftModel.from_pretrained(base, adapter_path)
model.eval()

# Load test set
with open("nyu_qa_testset_full.json") as f:
    dataset = json.load(f)

# Sentence similarity model
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# Run predictions + similarity
results = []
for item in dataset:
    prompt = f"{item['instruction']}\n\n{item['input']}\n\nAnswer:"
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=64,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.pad_token_id
        )
    generated = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    answer = generated.split("Answer:")[-1].strip()

    emb = embedder.encode([answer, item["output"]])
    similarity = float(cosine_similarity([emb[0]], [emb[1]])[0][0])

    results.append({
        "question": item["input"],
        "expected": item["output"],
        "generated": answer,
        "similarity": round(similarity, 4)
    })

# Save
with open("qa_results_scored.json", "w") as f:
    json.dump(results, f, indent=2)

avg_sim = sum(r["similarity"] for r in results) / len(results)
print(f"✅ Results saved to qa_results_scored.json")
print(f"📊 Average similarity: {avg_sim:.4f}")