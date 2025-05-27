import random
import os, json

# transformers의 AutoTokenizer 사용 (필요시 모델명 교체)
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")

# meta.json과 같은 폴더에 두고 실행
base_dir = os.path.dirname(__file__)
meta_path = os.path.join(base_dir, "meta.json")
meta = json.load(open(meta_path, encoding="utf-8"))

results = []

for info in meta.values():
    root = info["root"]
    ann  = info["annotation"]
    prefix = os.path.join("MMPR-v1.2", root)

    print(f"Processing {ann} ...")

    with open(os.path.join(base_dir, ann), encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            if "image" not in item.keys():
                prompt = item.get("question", "")
                assert "<image>" not in item.get("chosen", ""), f"<image> found in {item.get('chosen', '')}"
                assert "<image>" not in item.get("rejected", ""), f"<image> found in {item.get('rejected', '')}"
                chosen = item.get("chosen", "") 
                rejected = item.get("rejected", "")
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
                
                if "<image>" not in item.get("question", ""):
                    prompt = f"<image>\n{item.get('question', '')}"
                else:
                    prompt = item.get('question', '')
                if "<image>" in item.get("chosen", ""):
                    continue
                if "<image>" in item.get("rejected", ""):
                    r = item.get("rejected", "")
                    print(f"<image> found in rejected {r}, skipping...")
                    continue
                
                chosen = item.get("chosen", "")
                rejected = item.get("rejected", "")

            # --- 토큰 개수 검사 ---
            merged = f"{prompt} {chosen}".strip()
            tokens = tokenizer.encode(merged, add_special_tokens=False)
            if len(tokens) > 1400:
                continue
            # ------------------

            if img_path is None:
                out = {
                    "id": _id,
                    "prompt": prompt,
                    "chosen": chosen,
                    "rejected": rejected,
                }
            else:
                out = {
                    "id": _id,
                    "prompt": prompt,
                    "chosen": chosen,
                    "rejected": rejected,
                    "image": img_path
                }
            results.append(out)

# 한 번에 파일로 저장
out_path = os.path.join(base_dir, "dpo_mmpr_llava_format.json")
with open(out_path, "w", encoding="utf-8") as wf:
    json.dump(results, wf, ensure_ascii=False, indent=2)

print(f"Saved {len(results)} items to {out_path}")