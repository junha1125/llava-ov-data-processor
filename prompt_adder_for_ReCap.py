import json
import random
from tqdm import tqdm

RANDOM_PROMPTS = [
    "What is in the photo?",
    "Share a interpretation of the image provided.",
    "Describe the image concisely.",
    "Give a description of the image.",
    "Give a explanation of the image.",
    "Provide a description of the given image.",
]

def append_prompt_to_json(input_path, output_path):
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    for item in tqdm(data, desc="Processing items"):
        for conv in item.get("conversations", []):
            if conv.get("from") == "human" and conv.get("value") == "<image>":
                if random.random() < 0.8:
                    prompt = "Please generate detailed descriptions of the given image."
                else:
                    prompt = random.choice(RANDOM_PROMPTS)
                conv["value"] = "<image>\n" + prompt

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 3:
        print("Usage: python append_prompt.py input.json output.json")
    else:
        append_prompt_to_json(sys.argv[1], sys.argv[2])