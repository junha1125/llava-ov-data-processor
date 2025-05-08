# llava-ov-data-processor

## Dataset path

### ✔ scripts/[data](https://github.com/LLaVA-VL/LLaVA-NeXT/tree/main/scripts/train#about-the-llava-onevision-data)

| 단계          | 설명                                | 관련 링크                                                    |
| ------------- | ----------------------------------- | ------------------------------------------------------------ |
| **Stage 1**   | Pretraining 데이터셋 (BLIP558K)     | [BLIP558K](https://huggingface.co/datasets/liuhaotian/LLaVA-Pretrain/tree/main) |
| **Stage 1.5** | Mid-stage 학습 데이터셋             | [data yaml](https://github.com/LLaVA-VL/LLaVA-NeXT/blob/main/scripts/train/mid_stage.yaml) <br> [ReCap-118K](https://huggingface.co/datasets/lmms-lab/LLaVA-ReCap-118K) <br> [ReCap-558K](https://huggingface.co/datasets/lmms-lab/LLaVA-ReCap-558K) <br> [ReCap-CC3M](https://huggingface.co/datasets/lmms-lab/LLaVA-ReCap-CC3M) <br> [Remaining mid-data](https://huggingface.co/datasets/lmms-lab/LLaVA-OneVision-Mid-Data) |
| **Stage 2-1** | Single image 기반 학습              | [data yaml](https://github.com/LLaVA-VL/LLaVA-NeXT/blob/main/scripts/train/single_image.yaml) <br> [OneVisionData](https://huggingface.co/datasets/lmms-lab/LLaVA-OneVision-Data) <br> [upload script](https://github.com/LLaVA-VL/LLaVA-NeXT/blob/0070d0ae4931c9b19d9cc57c38e16a87c270a61c/playground/upload_data.py) |
| **Stage 2-2** | OneVision 종합 학습                 | [data yaml](https://github.com/LLaVA-VL/LLaVA-NeXT/blob/main/scripts/train/onevision.yaml) <br> [M4-Instruct](https://huggingface.co/datasets/lmms-lab/M4-Instruct-Data) <br> [LLaVA-Video-178K](https://huggingface.co/datasets/lmms-lab/LLaVA-Video-178K) |
| *(참고)*      | Stage 2-2는 일부 경우에는 생략 가능 | -                                                            |



## Download dataset 

```bash
pip install --upgrade huggingface_hub
pip install datasets
from huggingface_hub import snapshot_download

# Stage 1 
snapshot_download(repo_id="liuhaotian/LLaVA-Pretrain", repo_type="dataset", local_dir="/mnt/backbone-nfs/junha/dataset/BLIP558K") # (A) BLIP558K

# Stage 1.5
snapshot_download(repo_id="lmms-lab/LLaVA-ReCap-118K", repo_type="dataset", local_dir="/mnt/backbone-nfs/junha/dataset/ReCap-118K") # (B) ReCap-118K
snapshot_download(repo_id="lmms-lab/LLaVA-ReCap-558K", repo_type="dataset", local_dir="/mnt/backbone-nfs/junha/dataset/ReCap-558K") # (C) ReCap-558K
snapshot_download(repo_id="lmms-lab/LLaVA-ReCap-CC3M", repo_type="dataset", local_dir="/mnt/backbone-nfs/junha/dataset/ReCap-CC3M") # (D) ReCap-CC3M
snapshot_download(repo_id="lmms-lab/LLaVA-OneVision-Mid-Data", repo_type="dataset", local_dir="/mnt/backbone-nfs/junha/dataset/OneVisionMidData") # (E) OneVisionMidData

# Stage 2-1
snapshot_download(repo_id="lmms-lab/LLaVA-OneVision-Data", repo_type="dataset", local_dir="/mnt/backbone-nfs/junha/dataset/OneVisionData") # (F) OneVisionData
snapshot_download(repo_id="lmms-lab/LLaVA-NeXT-Data", repo_type="dataset", local_dir="/mnt/backbone-nfs/junha/dataset/LLaVA-Next") # (G) LLaVA-Next
```

## data parsing

```bash

cd /mnt/image-net-full/junha/dataset
# (A) BLIP558K
unzip /mnt/image-net-full/junha/dataset/BLIP558K/images.zip -d /mnt/backbone-nfs/junha/dataset/BLIP558K/image
rsync -av /mnt/image-net-full/junha/dataset/BLIP558K/blip_laion_cc_sbu_558k.json /mnt/backbone-nfs/junha/dataset/BLIP558K/

# (B) ReCap-118K
python parquet2json-image.py ReCap-118K coco118k_stage1.5_finetune_w_prompt.json

# (C) ReCap-558K
python parquet2json-image.py ReCap-558K blip558k_stage1.5_finetune_w_prompt.json

# (D) ReCap-CC3M
python parquet2json-image.py ReCap-CC3M cc3m_recap_data_prompt_v2.json # RAM 메모리 충분해야함 적어도 128G 이상

# (G) LLaVA-Next
python parquet2json-image.py LLaVA-Next llava_next_fit_mix_filtered_text_wild_738590.json # tqdm(ds, desc=f"Converting {base_name}") 이거 중엣 PIL.UnidentifiedImageError 에러가 나는 부분 있음

# (E) OneVisionMidData
python ov_mid_data_imagenet2backbone.py # json 파일의 image path 확인필요

# (F) OneVisionData
python ov_data_imagenet2backbone.py # CSV 파일 필요, json 파일의 image path 확인필요
```