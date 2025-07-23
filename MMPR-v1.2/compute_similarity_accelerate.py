#!/usr/bin/env python3
# compute_similarity_accelerate_mod.py

# ---------------------------------------
import os
os.environ["NCCL_TIMEOUT"] = "600"  # 10 minutes
os.environ["NCCL_BLOCKING_WAIT"] = "1"
os.environ["NCCL_DEBUG"] = "INFO"
os.environ["NCCL_SOCKET_IFNAME"] = "eth0"  # 시스템에 맞게 조정

# 1) 입력 JSON 파일
input_json = "dpo_mmpr_llava_format.json"
# 2) 출력 JSON 파일 (파일명 템플릿)
output_dir = "output_json"
output_json = "dpo_mmpr_llava_with_similarity.json"
# 3) 배치 사이즈 (GPU 메모리에 맞게 조절)
batch_size = 32

import json
import math
import torch
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer
from accelerate import Accelerator
from tqdm import tqdm
import json
import random


def load_data(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_data(data, path):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def main():
    # Accelerator 초기화
    accelerator = Accelerator()
    device = accelerator.device

    # 전체 데이터 로드
    data = load_data(input_json)
    total = len(data)
    world_size = accelerator.num_processes
    rank = accelerator.process_index
    per_rank = math.ceil(total / world_size)

    # 이 프로세스가 담당할 인덱스 범위 계산
    start = rank * per_rank
    end = min(start + per_rank, total)
    subset = data[start:end]

    # 모델 로드
    model = SentenceTransformer("Qwen/Qwen3-Embedding-4B")
    model.max_seq_length = 2048
    model.to(device)
    model.eval()

    # 이 프로세스의 유사도 계산
    local_sims = []
    for i in tqdm(range(0, len(subset), batch_size),
                  desc=f"Rank {rank} computing", disable=not accelerator.is_local_main_process):
        batch = subset[i : i + batch_size]
        queries = [item["chosen"] for item in batch]
        docs    = [item["rejected"] for item in batch]

        with torch.no_grad():
            q_emb = model.encode(
                queries,
                prompt_name="query",
                batch_size=len(queries),
                convert_to_tensor=True,
                device=device,
            )
            d_emb = model.encode(
                docs,
                batch_size=len(docs),
                convert_to_tensor=True,
                device=device,
            )
            sims = F.cosine_similarity(q_emb, d_emb)
        local_sims.extend(sims.tolist())

    # 유사도 값을 subset 항목에 추가
    for idx, sim in enumerate(local_sims):
        subset[idx]["similarity"] = float(f"{sim:.4f}")

    # 각 GPU(rank)별로 JSON 파일 저장
    base, ext = os.path.splitext(output_json)
    out_path = f"{base}_rank{rank}{ext}"
    out_path = os.path.join(output_dir, out_path)
    save_data(subset, out_path)
    print(f"Rank {rank} saved results to {out_path}")



def merge():
    import json
    import glob
    import os

    # 전역변수: 패턴과 출력 파일명
    SAVED_DIR = "output_json"
    PATTERN = "dpo_mmpr_llava_with_similarity_rank*.json"
    OUTPUT_PATH = "dpo_mmpr_llava_with_similarity.json"
    
    files_path = os.path.join(SAVED_DIR, PATTERN)
    files = sorted(glob.glob(files_path))
    merged = []

    for fp in files:
        with open(fp, 'r', encoding='utf-8') as f:
            data = json.load(f)
        merged.extend(data)
        print(f"Loaded {len(data)} records from {os.path.basename(fp)}")

    with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
        json.dump(merged, f, ensure_ascii=False, indent=2)

    print(f"Merged total {len(merged)} records into {OUTPUT_PATH}")



def make():

    INPUT_FILE = "dpo_mmpr_llava_with_similarity.json"
    LOW_OUT = "sample_similarity_le0.7_10k.json"
    HIGH_OUT = "sample_similarity_ge0.9_10k.json"
    output_dir = "output_json"
    LOW_OUT = os.path.join(output_dir, LOW_OUT)
    HIGH_OUT = os.path.join(output_dir, HIGH_OUT)

    data = load_data(INPUT_FILE)

    # Filter low similarity (<= 0.7)
    low_sim = [item for item in data if item.get('similarity', 0) <= 0.7]
    print(f"Low similarity (<=0.7) count: {len(low_sim)}")
    if len(low_sim) < 10000:
        raise ValueError(f"Not enough items with similarity <= 0.7: found {len(low_sim)}, need at least 10000.")

    # Sample and save
    low_sample = random.sample(low_sim, 10000)
    save_data(low_sample, LOW_OUT)
    print(f"Saved 10000 low-similarity samples to {LOW_OUT}")

    # Filter high similarity (>= 0.9)
    high_sim = [item for item in data if item.get('similarity', 0) >= 0.9]
    print(f"High similarity (>=0.9) count: {len(high_sim)}")
    if len(high_sim) < 10000:
        raise ValueError(f"Not enough items with similarity >= 0.9: found {len(high_sim)}, need at least 10000.")

    # Sample and save
    high_sample = random.sample(high_sim, 10000)
    save_data(high_sample, HIGH_OUT)
    print(f"Saved 10000 high-similarity samples to {HIGH_OUT}")


def sft():
    OUTPUT_DIR = "output_json"
    INPUT_FILE = "sample_similarity_ge0.9_10k.json"
    OUTPUT_FILE = "sft_ge0.9_10k.json"
    INPUT_FILE = os.path.join(OUTPUT_DIR, INPUT_FILE)
    OUTPUT_FILE = os.path.join(OUTPUT_DIR, OUTPUT_FILE)

    def build_conversations(records):
        conv_data = []
        for item in records:
            # Ensure required keys exist
            if not all(k in item for k in ("id", "prompt", "chosen", "image")):
                continue
            conv_data.append({
                "id": item["id"],
                "conversations": [
                    {"from": "human", "value": item["prompt"]},
                    {"from": "gpt", "value": item["chosen"]}
                ],
                "image": item["image"]
            })
        return conv_data

    if not os.path.exists(INPUT_FILE):
        raise FileNotFoundError(f"Input file not found: {INPUT_FILE}")

    data = load_data(INPUT_FILE)
    conversations = build_conversations(data)

    print(f"Built {len(conversations)} conversation entries.")
    save_data(conversations, OUTPUT_FILE)
    print(f"Saved conversations to {OUTPUT_FILE}")


if __name__ == "__main__":
    os.makedirs(output_dir, exist_ok=True)
    # main()
    # merge()
    # make()
    sft()