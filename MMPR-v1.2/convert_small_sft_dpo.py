import random
import os
import json
from transformers import AutoTokenizer
import random
from tqdm import tqdm

SEED = 95
DATA_SIZE = 10_000

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")

base_dir  = os.path.dirname(__file__)
meta_path = os.path.join(base_dir, "meta.json")
meta      = json.load(open(meta_path, encoding="utf-8"))

# 1) meta.json 에 담긴 모든 (annotation 경로, 이미지 prefix, root 이름) 정보를 미리 수집
file_infos = []
for info in meta.values():
    ann_rel = info["annotation"]                          # ex) "foo.jsonl"
    root    = info["root"]                                # ex) "bar"
    prefix  = os.path.join("MMPR-v1.2", root)             # 이미지가 실제 저장된 디렉토리
    ann_path = os.path.join(base_dir, ann_rel)            # 실제 파일 경로
    file_infos.append((ann_path, prefix, root))

# 2) 모든 JSONL 항목을 메모리에 한 번에 읽어서 raw_items 에 누적
raw_items = []
for ann_path, prefix, root in file_infos:
    print(f"Loading {ann_path} ...")
    with open(ann_path, encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            # skip multiple images
            if "image" in item.keys():
                rel_img = item["image"]
                if isinstance(rel_img, list):
                    print(f"Skipping entire file {ann_path} due to multiple images...")
                    break
            raw_items.append((item, prefix, root))
    # break

# 샘플링: raw_items에서 10,000개 항목을 랜덤하게 선택
print(f"Total items loaded: {len(raw_items)}")
random.seed(SEED)
indices = list(range(len(raw_items)))
random.shuffle(indices)
sampled_indices = indices[:DATA_SIZE*2]
raw_items = [raw_items[i] for i in sampled_indices]

# 3) 이제 raw_items 리스트를 한 번만 순회하며 SFT/DPO 로직을 적용
sft_results = []
dpo_results = []
iter = 0

for item, prefix, root in tqdm(raw_items, desc="Processing items"):
    ###### SFT ############################################################
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
        
        if rel_img.endswith('.gif'):
            rel_img = rel_img[:-4] + '.jpg'
        assert rel_img.endswith(('.jpg', '.jpeg', '.png')), f"Invalid image format: {rel_img}"
        img_path = os.path.join(prefix, rel_img)
        
        if not os.path.exists(os.path.join('/mnt/ssd/junha/dataset', img_path)):
            print(f"Image file does not exist, skipping...")
            print(f"Path: {os.path.join('/mnt/ssd/junha/dataset', img_path)}")
            continue

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
    
    ###### DPO ############################################################
    if "image" not in item.keys():
        prompt = item.get("question", "")
        assert "<image>" not in item.get("chosen", ""), f"<image> found in {item.get('chosen', '')}"
        assert "<image>" not in item.get("rejected", ""), f"<image> found in {item.get('rejected', '')}"
        chosen = item.get("chosen", "") 
        rejected = item.get("rejected", "")
        img_path = None
    else:   
        rel_img = item["image"]
        
        if rel_img.endswith('.gif'):
            rel_img = rel_img[:-4] + '.jpg'
        assert rel_img.endswith(('.jpg', '.jpeg', '.png')), f"Invalid image format: {rel_img}"
        img_path = os.path.join(prefix, rel_img)    
        
        if "<image>" not in item.get("question", ""):
            prompt = f"<image>\n{item.get('question', '')}"
        else:
            prompt = item.get('question', '')
        if "<image>" in item.get("chosen", ""):
            continue
        if "<image>" in item.get("rejected", ""):
            r = item.get("rejected", "")
            print(f"<image> found in rejected, skipping...")
            continue
        
        chosen = item.get("chosen", "")
        rejected = item.get("rejected", "")
    
    # --- 토큰 개수 검사 ---
    merged = f"{prompt} {chosen}".strip()
    tokens = tokenizer.encode(merged, add_special_tokens=False)
    if len(tokens) > 1400:
        print("Skipping due to token limit exceeded")
        continue
    # ------------------

    if img_path is None:
        sft_out = {
            "id": _id,
            "conversations": conv,
            "data_source": root
        }
        dpo_out = {
            "id": _id,
            "prompt": prompt,
            "chosen": chosen,
            "rejected": rejected,
        }
    else:
        sft_out = {
            "id": _id,
            "conversations": conv,
            "data_source": root,
            "image": img_path
        }
        dpo_out = {
            "id": _id,
            "prompt": prompt,
            "chosen": chosen,
            "rejected": rejected,
            "image": img_path
        }
    sft_results.append(sft_out)
    dpo_results.append(dpo_out)
    iter += 1
    if iter >= DATA_SIZE:
        break

sft_out_path = os.path.join(base_dir, f"sft_mmpr_{SEED}.json")
with open(sft_out_path, "w", encoding="utf-8") as wf:
    json.dump(sft_results, wf, ensure_ascii=False, indent=2)

print(f"Total SFT samples: {len(sft_results)}, File saved to {sft_out_path}")

dpo_out_path = os.path.join(base_dir, f"dpo_mmpr_{SEED}.json")
with open(dpo_out_path, "w", encoding="utf-8") as wf:
    json.dump(dpo_results, wf, ensure_ascii=False, indent=2)

print(f"Total DPO samples: {len(dpo_results)}, File saved to {dpo_out_path}")