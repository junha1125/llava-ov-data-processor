#!/usr/bin/env python3
# ov_data_imagenet2backbone.py

# chmod +x ov_data_imagenet2backbone.py
# ./ov_data_imagenet2backbone.py \
#   --source-root /mnt/image-net-full/junha/dataset/OneVisionData \
#   --target-root /mnt/backbone-nfs/junha/dataset/OneVisionData \
#   --cache-dir /mnt/image-net-full/junha/.cache/huggingface

import os
import glob
import argparse
import json
import csv
from io import BytesIO

from datasets import load_dataset, Image as HFImage
from tqdm import tqdm
from PIL import Image, UnidentifiedImageError

def convert_dataset(source_root: str, target_base: str, cache_dir: str):
    """
    source_root: 원본 parquet 폴더 (e.g. /mnt/image-net-full/.../OneVisionData/ai2d(cauldron,llava_format))
    target_base: 변환된 결과가 저장될 최상위 디렉토리 
                 (e.g. /mnt/backbone-nfs/junha/dataset/OneVisionData)
    cache_dir:   HuggingFace Dataset 캐시 디렉토리
    """
    base_name   = os.path.basename(source_root.rstrip("/"))
    target_root = os.path.join(target_base, base_name)
    image_dir   = os.path.join(target_root, "image")
    os.makedirs(image_dir, exist_ok=True)

    # parquet 파일 검색 (폴더 직속 또는 재귀)
    parquet_files = sorted(glob.glob(os.path.join(source_root, "*.parquet")))
    if not parquet_files:
        parquet_files = sorted(glob.glob(os.path.join(source_root, "**", "*.parquet"), recursive=True))
    if not parquet_files:
        print(f"[SKIP] '{base_name}'에 parquet 파일이 없습니다.")
        return

    print(f"[START] Converting '{base_name}', {len(parquet_files)} files…")

    # ── 1) 자동 디코딩 끄기 ─────────────────────────────────────────────
    ds = (
        load_dataset(
            "parquet",
            data_files={"train": parquet_files},
            split="train",
            cache_dir=cache_dir,
        )
        .cast_column("image", HFImage(decode=False))  # decode=False!  🔑
    )

    converted, bad_samples = [], []

    for row in tqdm(ds, desc=f"  → {base_name}"):
        rec = {
            "id": row["id"],
            "conversations": row.get("conversations", []),
        }

        # 이미지 처리 (lazy‑decode & 오류 처리)
        bytes_ = row["image"].get("bytes") if row.get("image") else None
        if bytes_:
            try:
                img = Image.open(BytesIO(bytes_))
                img.load()  # 실제 디코딩 ➜ 오류가 여기서 발생하면 except 로

                # JPEG에 쓸 수 없는 모드(RGBA 등)는 RGB 변환
                if getattr(img, "mode", "RGB") != "RGB":
                    img = img.convert("RGB")

                img_fname = f"{row['id']}.jpg"
                rec["image"] = os.path.join("OneVisionData", base_name, "image", img_fname)
                img.save(os.path.join(image_dir, img_fname))

            except (UnidentifiedImageError, OSError) as e:
                bad_samples.append({"id": row["id"], "reason": str(e)})
                continue  # 해당 샘플은 건너뜀

        converted.append(rec)

    # ── 2) JSON 파일 저장 ─────────────────────────────────────────────
    # CSV에 정의된 폴더→JSON 이름 매핑 로드
    json_mapping = {}
    mapping_csv = os.path.join(os.path.dirname(__file__), "OneVisionData", "JSON_Mapping.csv")
    if os.path.exists(mapping_csv):
        with open(mapping_csv, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                json_mapping[row["folder_name"]] = row["json_file"]

    json_name = json_mapping.get(base_name, f"{base_name}.json")
    json_path = os.path.join(target_root, json_name)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(converted, f, ensure_ascii=False, indent=2)

    # ── 3) 깨진 샘플 로그 저장 ────────────────────────────────────────
    if bad_samples:
        log_path = os.path.join(target_root, f"{base_name}_bad_samples.csv")
        with open(log_path, "w", newline="", encoding="utf-8") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=["id", "reason"])
            writer.writeheader()
            writer.writerows(bad_samples)
        print(f"[WARN] {len(bad_samples):,} bad samples logged to {log_path}")

    print(f"[DONE] '{base_name}': {len(converted):,} good samples → {json_path}")


def main():
    parser = argparse.ArgumentParser(
        description="OneVisionData 전체 서브폴더를 json+image 포맷으로 일괄 변환합니다."
    )
    parser.add_argument(
        "--source-root",
        default="/mnt/image-net-full/junha/dataset/OneVisionData",
        help="원본 OneVisionData 폴더 경로",
    )
    parser.add_argument(
        "--target-root",
        default="/mnt/backbone-nfs/junha/dataset/OneVisionData",
        help="변환 결과를 저장할 폴더 경로",
    )
    parser.add_argument(
        "--cache-dir",
        default="/mnt/image-net-full/junha/.cache/huggingface",
        help="HuggingFace Dataset 캐시 디렉토리",
    )
    args = parser.parse_args()

    # 타겟 루트 만들기
    os.makedirs(args.target_root, exist_ok=True)

    for entry in sorted(os.listdir(args.source_root)):
        src_path = os.path.join(args.source_root, entry)
        # .cache 같은 폴더는 스킵
        if entry.startswith(".") or not os.path.isdir(src_path):
            continue
        # 이미 변환된 폴더 스킵
        target_folder = os.path.join(args.target_root, entry)
        if os.path.exists(target_folder):
            print(f"[SKIP] '{entry}' already exists in target directory")
            continue

        convert_dataset(src_path, args.target_root, args.cache_dir)


if __name__ == "__main__":
    main()
