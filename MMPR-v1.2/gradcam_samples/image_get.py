# resize_images.py
# 요구사항:
# 1) 글로벌 변수(JSON_PATH)에 지정된 JSON 파일을 로드
# 2) 각 항목의 "image" 경로 이미지를 PIL로 열어 384x384로 리사이즈
# 3) 리사이즈된 이미지만 새 폴더(images)에 저장

import json
import os
from PIL import Image

# >>> 이 값을 당신의 JSON 파일 경로로 바꿔주세요.
JSON_PATH = "dpo_mmpr_gradient.json"

IMAGES_DIR = "images"
TARGET_SIZE = (384, 384)
ROOT = "/mnt/ssd/junha/dataset"


with open(JSON_PATH, "r", encoding="utf-8") as f:
    items = json.load(f)

# 저장 폴더 생성
os.makedirs(IMAGES_DIR, exist_ok=True)

# 각 항목 처리
for item in items:
    img_path = item.get("image")
    img_path = os.path.join(ROOT, img_path)
    if not img_path:
        continue

    try:
        with Image.open(img_path) as img:
            # 384x384로 리사이즈 (고품질 보간)
            resized = img.resize(TARGET_SIZE, Image.LANCZOS)

            # 저장 파일명: id가 있으면 id를 사용, 없으면 원본 파일명 사용
            base = os.path.basename(img_path)
            name, ext = os.path.splitext(base)
            out_name = f"{item.get('id', name)}{ext or '.png'}"
            out_path = os.path.join(IMAGES_DIR, out_name)

            # 저장
            resized.save(out_path)
            print(f"Saved: {out_path}")

    except Exception as e:
        print(f"Skip '{img_path}': {e}")

