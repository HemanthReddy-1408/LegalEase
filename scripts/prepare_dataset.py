from datasets import load_dataset, concatenate_datasets
import json
import os

# Step 1: Load datasets
ipc = load_dataset("Techmaestro369/indian-legal-texts-finetuning", data_files="ipc_qa.json", split="train")
crpc = load_dataset("Techmaestro369/indian-legal-texts-finetuning", data_files="crpc_qa.json", split="train")
const = load_dataset("Techmaestro369/indian-legal-texts-finetuning", data_files="constitution_qa.json", split="train")

# Step 2: Merge
full_dataset = concatenate_datasets([ipc, crpc, const])

# Step 3: Convert to SFT format
sft_data = []
for item in full_dataset:
    if item["question"] and item["answer"]:
        sft_data.append({
            "instruction": item["question"].strip(),
            "input": "",
            "output": item["answer"].strip()
        })

# Step 4: Save
os.makedirs("data", exist_ok=True)
with open("data/legal_qa.json", "w", encoding="utf-8") as f:
    json.dump(sft_data, f, indent=2, ensure_ascii=False)

print(f"Saved {len(sft_data)} examples to data/legal_qa.json")
