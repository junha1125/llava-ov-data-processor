#!/usr/bin/env python3
# analyze_similarity.py

# 1) Load the JSON with similarity scores
input_json = "dpo_mmpr_llava_with_similarity.json"
# 2) Output histogram image filename
hist_image = "similarity_histogram.png"
# 3) Subfolder for sub JSONs
subfolder = "output_json"

import json
import os
from pathlib import Path
import matplotlib.pyplot as plt


def load_data(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_json(data, path):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def make_histogram(similarities, output_path):
    plt.figure(figsize=(8, 6))
    plt.hist(similarities, bins=50)
    plt.title('Similarity Distribution')
    plt.xlabel('Cosine Similarity')
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


# 1) Load data
data = load_data(input_json)
# Extract similarity values
sims = [item.get('similarity', 0.0) for item in data]
os.makedirs(subfolder, exist_ok=True)

# 2) Plot and save histogram
hist_image = os.path.join(subfolder, hist_image)
make_histogram(sims, hist_image)
print(f"Saved histogram to {hist_image}")

# 3) Create subfolder

# 4) Sort data by similarity
sorted_data = sorted(data, key=lambda x: x.get('similarity', 0.0))
total = len(sorted_data)

# Bottom 20
bottom20 = sorted_data[:40]
save_json(bottom20, os.path.join(subfolder, 'bottom20.json'))

# Middle 20
mid = total // 2
start = max(0, mid - 10)
end = min(total, start + 40)
mid20 = sorted_data[start:end]
save_json(mid20, os.path.join(subfolder, 'mid20.json'))

# Top 20
top20 = sorted_data[-40:]
save_json(top20, os.path.join(subfolder, 'top20.json'))

print(f"Saved bottom20.json, mid20.json, top20.json in {subfolder}/")
