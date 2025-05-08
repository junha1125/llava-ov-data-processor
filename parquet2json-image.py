#!/usr/bin/env python
# parquet2json-image.py
"""
Convert a local parquet-based dataset folder (e.g. ReCap-118K) to the
JSON + image-directory format LLaVA-NeXT uses.

Example:
    python parquet2json-image.py /mnt/image-full/junha/dataset/ReCap-118K recap118k.json
    python parquet2json-image.py /mnt/backbone-nfs/junha/dataset/ReCap-118K recap118k.json
"""

import argparse, glob, json, os, csv
from io import BytesIO
from datasets import load_dataset, Image as HFImage
from tqdm import tqdm
from PIL import Image, UnidentifiedImageError

def convert_dataset(source_root: str, json_name: str) -> None:
    source_root = os.path.abspath(source_root)
    base_name  = os.path.basename(source_root)          # e.g. ReCap-118K

    target_root = os.path.join("/mnt/backbone-nfs/junha/dataset", base_name)
    image_dir   = os.path.join(target_root, "image")
    os.makedirs(image_dir, exist_ok=True)

    parquet_files = sorted(glob.glob(os.path.join(source_root, "data", "*.parquet")))
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files in {os.path.join(source_root,'data')}")

    # ── 1) 자동 디코딩 끄기 ─────────────────────────────────────────────
    ds = load_dataset(
        "parquet",
        data_files={"train": parquet_files},
        split="train",
        cache_dir="/mnt/image-net-full/junha/.cache/huggingface",
    ).cast_column("image", HFImage(decode=False))   # ← decode=False!

    converted, bad_samples = [], []                 # bad_samples → CSV로 남김

    for row in tqdm(ds, desc=f"Converting {base_name}"):
        rec = {"id": row["id"], "conversations": row["conversations"]}

        bytes_ = row["image"]["bytes"] if row.get("image") else None
        if bytes_:
            try:
                img = Image.open(BytesIO(bytes_))
                img.load()                          # 실제 디코딩
                if getattr(img, "mode", "RGB") != "RGB":
                    img = img.convert("RGB")
                img_filename = f"{row['id']}.jpg"
                sub_dir = os.path.dirname(img_filename)
                if sub_dir:
                    os.makedirs(os.path.join(image_dir, sub_dir), exist_ok=True)
                rel_path     = os.path.join(base_name, "image", img_filename)
                rec["image"] = rel_path
                img.save(os.path.join(image_dir, img_filename))

            # ── 2) 깨진 이미지 건너뛰기 ───────────────────────────────
            except (UnidentifiedImageError, OSError) as e:
                bad_samples.append({"id": row["id"], "reason": str(e)})
                continue

        converted.append(rec)

    # ── 3) JSON 저장 ─────────────────────────────────────────────────
    if not json_name.endswith(".json"):
        json_name += ".json"
    json_path = os.path.join(target_root, json_name)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(converted, f, indent=4, ensure_ascii=False)

    # ── 4) 깨진 샘플 로그 저장 (ReCap‑CC3M 폴더) ───────────────────────
    if bad_samples:
        bed_root = os.path.join("/mnt/backbone-nfs/junha/dataset", base_name)
        log_path = os.path.join(bed_root, f"{base_name}_bad_samples.csv")
        with open(log_path, "w", newline="", encoding="utf-8") as csvfile:
            fieldnames = ["id", "reason"]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(bad_samples)
        print(f"||   {len(bad_samples):,} bad samples logged to {log_path}")

    print(f"||   Done. {len(converted):,} good samples saved to {json_path}")
    print(f"||   Images written to {image_dir}")

# ── main 그대로 ──────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Convert parquet dataset folder to LLaVA-NeXT json+image format.")
    parser.add_argument("dataset_folder", help="Source dataset folder path (e.g. /mnt/image-full/...)")
    parser.add_argument("json_name",      help="Output JSON filename (e.g. recap118k.json)")
    args = parser.parse_args()
    convert_dataset(args.dataset_folder, args.json_name)

if __name__ == "__main__":
    main()