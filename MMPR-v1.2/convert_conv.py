import random
import json, os

# tokenizer 준비 (예시: transformers의 AutoTokenizer 사용)
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")

# meta.json 과 같은 폴더에 두고 실행
base_dir = os.path.dirname(__file__)
meta_path = os.path.join(base_dir, "meta.json")
meta = json.load(open(meta_path, encoding="utf-8"))

results = []   

for info in meta.values():
    root = info["root"]
    ann = info["annotation"]
    prefix = os.path.join('MMPR-v1.2', root)

    print(f"Processing {ann} ...")

    with open(ann, encoding="utf-8") as f:
        for line in f:
            item = json.loads(line) # what is different btw line and item? line is a string, item is a dict 
            if "image" not in item.keys():
                conv = [
                    {"from": "human", "value": item.get("question", "")},
                    {"from": "gpt",   "value": item.get("chosen", "")}
                ]
                img_path = None
                _id = str(random.randint(10**7, 10**8 - 1))
                print(f"{_id} has no image, using random ID")
            else:   
                rel_img = item["image"]

                if isinstance(rel_img, list):
                    print(f"Skipping entire file {ann} ...")
                    break
                
                if rel_img.endswith('.gif'):
                    rel_img = rel_img[:-4] + '.jpg'
                assert rel_img.endswith(('.jpg', '.jpeg', '.png')), f"Invalid image format: {rel_img}"
                img_path = os.path.join(prefix, rel_img)

                parent = os.path.basename(os.path.dirname(img_path))  
                fname = os.path.splitext(os.path.basename(img_path))[0]
                ramdom_num = str(random.randint(10**4, 10**5 - 1))
                _id = f"{parent}-{fname}-{ramdom_num}" if parent else fname 

                # check human value
                human_value = item.get("question", "")
                if "<image>" not in human_value:
                    human_value = f"<image>\n{human_value}"

                # check gpt value
                gpt_value = item.get("chosen", "")
                if "<iamge>" in gpt_value:
                    continue 

                conv = [
                    {"from": "human", "value": human_value},
                    {"from": "gpt",   "value": gpt_value}
                ]

            # --- 토큰 개수 검사 추가 ---
            human = conv[0]["value"]
            gpt = conv[1]["value"]
            merged = f"{human} {gpt}".strip()
            tokens = tokenizer.encode(merged, add_special_tokens=False)
            if len(tokens) > 1400:
                continue  # 1400개 초과면 append하지 않음
            # ------------------------

            if img_path is None:
                out = {
                    "id": _id,
                    "conversations": conv,
                    "data_source": root
                }
            else:
                out = {
                    "id": _id,
                    "conversations": conv,
                    "data_source": root,
                    "image": img_path
                }
            results.append(out)    # <-- append

# 결과를 한 번에 파일로 저장
out_path = os.path.join(base_dir, "sft_mmpr_llava_format.json")
with open(out_path, "w", encoding="utf-8") as wf:
    json.dump(results, wf, ensure_ascii=False, indent=2)

print(f"Saved {len(results)} items to {out_path}")