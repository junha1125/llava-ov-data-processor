import os, json

# meta.json과 같은 폴더에 두고 실행
base_dir = os.path.dirname(__file__)
meta_path = os.path.join(base_dir, "meta.json")
meta = json.load(open(meta_path, encoding="utf-8"))

results = []

for info in meta.values():
    root = info["root"]
    ann  = info["annotation"]
    prefix = os.path.join("MMPR-v1.2", root)

    print(f"Processing {root} ...")

    with open(os.path.join(base_dir, ann), encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            rel_img = item["image"]

            if isinstance(rel_img, list):
                print(f"Skipping entire file {ann}, entry: {json.dumps(item, ensure_ascii=False)}")
                break

            img_path = os.path.join(prefix, rel_img)

            parent = os.path.basename(os.path.dirname(img_path))
            fname  = os.path.splitext(os.path.basename(img_path))[0]
            _id    = f"{parent}-{fname}" if parent else fname

            out = {
                "id": _id,
                "prompt": f"<image>\n{item.get('question', '')}",
                "chosen": item.get("chosen", ""),
                "rejected": item.get("rejected", ""),
                "image": img_path
            }
            results.append(out)

# 한 번에 파일로 저장
out_path = os.path.join(base_dir, "dpo_mmpr_llava_format.json")
with open(out_path, "w", encoding="utf-8") as wf:
    json.dump(results, wf, ensure_ascii=False, indent=2)

print(f"Saved {len(results)} items to {out_path}")