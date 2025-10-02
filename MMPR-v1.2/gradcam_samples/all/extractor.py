import json

# 전역 변수 (필터링할 키워드)
TARGET_KEYWORD = "docvqa/train/documents"

# 입력/출력 파일 경로
INPUT_FILE = "dpo_mmpr_llava_format.json"
OUTPUT_FILE = "dpo_mmpr_document.json"

def main():
    try:
        # JSON 파일 읽기
        with open(INPUT_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)

        # 필터링
        filtered_data = [item for item in data if TARGET_KEYWORD in item.get("image", "")]

        # 결과 저장
        with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
            json.dump(filtered_data, f, indent=2, ensure_ascii=False)

        print(f"총 {len(filtered_data)}개의 데이터가 '{OUTPUT_FILE}'에 저장되었습니다.")

    except FileNotFoundError:
        print(f"파일을 찾을 수 없습니다: {INPUT_FILE}")
    except json.JSONDecodeError:
        print("JSON 파일 형식이 잘못되었습니다.")
    except Exception as e:
        print(f"에러 발생: {e}")

if __name__ == "__main__":
    main()