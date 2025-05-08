import os
import shutil
import tarfile
from tqdm import tqdm
from pathlib import Path
import json

src_base = "/mnt/image-net-full/junha/dataset/OneVisionMidData"
dst_base = "/mnt/backbone-nfs/junha/dataset/OneVisionMidData"

os.makedirs(dst_base, exist_ok=True)

# OneVisionMidData 내 폴더 목록 가져오기
folders = [f for f in os.listdir(src_base) if os.path.isdir(os.path.join(src_base, f))]

for folder in folders:
    src_folder = os.path.join(src_base, folder)
    dst_folder = os.path.join(dst_base, folder)
    
    # 대상 폴더 생성
    os.makedirs(dst_folder, exist_ok=True)
    
    # 폴더 내 파일 처리
    for file in os.listdir(src_folder):
        src_path = os.path.join(src_folder, file)
        
        if file.endswith('.json'):
            # JSON 파일 처리 - 이미지 경로 수정 후 저장
            with open(src_path, 'r') as f:
                data = json.load(f)
                
            # JSON 데이터가 리스트 형태인지 확인
            if isinstance(data, list):
                print(f"Processing {file} with {len(data)} items...")
                for item in tqdm(data, desc=f"Updating image paths in {file}"):
                    if 'image' in item:
                        original_path = item['image']
                        root = os.path.basename(dst_base)
                        new_path = f"{root}/{folder}/image/{original_path}"
                        item['image'] = new_path
            else:
                print(f"Warning: {file} is not a list. Skipping...")
            
            # 수정된 JSON 저장
            dst_json_path = os.path.join(dst_base, folder, file)
            with open(dst_json_path, 'w') as f:
                json.dump(data, f, indent=2)
            print(f"Processed and saved {file} to {dst_base}")
            
        elif 'images_' in file and file.endswith('.tar.gz'):
            # 이미지 압축 파일은 image 폴더에 압축 해제
            image_dir = os.path.join(dst_folder, "image")
            os.makedirs(image_dir, exist_ok=True)
            
            with tarfile.open(src_path, 'r:gz') as tar:
                tar.extractall(path=image_dir)
            print(f"Extracted {file} to {image_dir}")

print("All files processed successfully!")