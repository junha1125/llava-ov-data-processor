# make_mdpo.py
import os
import sys
import json
from pathlib import Path
from tqdm import tqdm

from torchvision.io import read_image, write_jpeg, write_png
from torchvision.transforms import v2

INPUT_JSON = "jsons/dpo_mmpr_95.json"
OUTPUT_JSON = "jsons/dpo_mmpr_95_mdpo.json"
IMAGES_PREFIX = "MMPR-v1.2/images/"
REJECTED_PREFIX = "MMPR-v1.2/images/rejected/"
DATA_ROOT = "/mnt/ssd/junha/dataset"

def crop_images(images):
    new_images = []
    for image in images:
        # image: torch.Tensor [C, H, W] (uint8 OK)
        resize_cropper = v2.RandomResizedCrop(size=image.size()[-2:], scale=(0.01, 0.2))
        image = resize_cropper(image.squeeze(0)).unsqueeze(0)
        new_images.append(image)
    return new_images

def to_rejected_path(image_path: str) -> str:
    if IMAGES_PREFIX not in image_path:
        # 그래도 최대한 맞춰서 뒤에 붙여 저장
        return os.path.join(REJECTED_PREFIX, image_path)
    return image_path.replace(IMAGES_PREFIX, REJECTED_PREFIX, 1)

def save_tensor_image(tensor_img, out_path: str):
    # tensor_img: [1, C, H, W] uint8
    img = tensor_img.squeeze(0)  # [C, H, W]
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ext = out_path.suffix.lower()
    if ext in [".jpg", ".jpeg"]:
        write_jpeg(img, str(out_path))
    elif ext == ".png":
        write_png(img, str(out_path))
    else:
        # 기본은 jpg로 저장
        out_jpg = out_path.with_suffix(".jpg")
        write_jpeg(img, str(out_jpg))

def main():
    # 입력 JSON 로드
    with open(INPUT_JSON, "r", encoding="utf-8") as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError as e:
            print(f"JSON 파싱 오류: {e}")
            sys.exit(1)

    out_items = []

    for idx, item in tqdm(enumerate(data), total=len(data), desc="처리 중"):
        # 필수 키 점검
        for k in ("id", "prompt", "chosen", "image"):
            if k not in item:
                # 최소한의 방어: 스킵
                continue

        img_path = item["image"]
        rejected_path = to_rejected_path(img_path)

        real_img_path = os.path.join(DATA_ROOT, img_path)
        real_rejected_path = os.path.join(DATA_ROOT, rejected_path)
        # 이미지 로드 & 랜덤 크롭 저장 (0~20%)
        try:
            img_tensor = read_image(real_img_path)  # [C,H,W], uint8
            cropped = crop_images([img_tensor])[0]  # [1,C,H,W]
            save_tensor_image(cropped, real_rejected_path)
        except Exception as e:
            # 이미지 처리 실패 시에도 JSON은 만들되, 파일 생성은 건너뜀
            print(f"[경고] 이미지 처리 실패 ({img_path}): {e}")

        # 출력 항목 구성
        out_items.append({
            "id": item["id"],
            "prompt": item["prompt"],
            # 규칙: chosen, rejected 모두 입력의 chosen 으로
            "chosen": item["chosen"],
            "rejected": item["chosen"],
            # 규칙: image_chosen = 입력 image, image_rejected = 크롭 저장 경로
            "image": img_path,
            "image_rejected": rejected_path,
        })

    # 출력 JSON 저장
    Path(OUTPUT_JSON).parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(out_items, f, ensure_ascii=False, indent=2)

    print(f"완료! {len(out_items)}개 항목을 '{OUTPUT_JSON}' 에 저장했습니다.")

if __name__ == "__main__":
    main()