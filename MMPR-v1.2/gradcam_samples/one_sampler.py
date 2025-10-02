# make_jsons.py
import json
import os
import sys


input_path = "dpo_mmpr_gradient.json"
dir_name = "one-sample"
with open(input_path, "r", encoding="utf-8") as f:
    datas = json.load(f)

for data in datas:
    image_base = os.path.basename(data["image"]).split(".")[0]

    # 1) dpo-{image_base}.json : 원본 data를 20번 반복
    dpo_list = [data.copy() for _ in range(20)]
    json_dpo_path = os.path.join(dir_name, f"dpo-{image_base}.json")
    with open(json_dpo_path, "w", encoding="utf-8") as f:
        json.dump(dpo_list, f, ensure_ascii=False, indent=2)

    # 2) sft-{image_base}.json : 변환 포맷으로 20번 반복
    sft_item = {
        "id": data["id"],
        "conversations": [
            {"from": "human", "value": data["prompt"]},
            {"from": "gpt", "value": data["chosen"]},
        ],
        "image": data["image"],
    }
    sft_list = [sft_item.copy() for _ in range(20)]
    json_sft_path = os.path.join(dir_name, f"sft-{image_base}.json")
    with open(json_sft_path, "w", encoding="utf-8") as f:
        json.dump(sft_list, f, ensure_ascii=False, indent=2)
