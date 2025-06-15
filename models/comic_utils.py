import json
from typing import List, Dict

def parse_ocr_file(json_path: str) -> List[Dict]:
    """OCR JSON 파일을 파싱하여 텍스트 영역 목록을 반환합니다."""
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"오류: JSON 파일을 찾을 수 없습니다 - {json_path}")
        return []
    except json.JSONDecodeError:
        print(f"오류: JSON 파일 파싱 실패 - {json_path}")
        return []

    regions = []
    # 'results' 키가 없는 경우를 대비하여 안전하게 접근
    for detection in data.get('results', []):
        if detection.get('score', 0) >= 0.7:
            box = detection.get('box', [[0], [0], [0], [0]])

            # --- TypeError 해결을 위한 수정된 부분 시작 ---
            # box가 중첩 리스트일 가능성에 대비하여 안전하게 값을 추출합니다.
            try:
                # 1. 언패킹 시도
                x_val, y_val, w_val, h_val = box

                # 2. 각 값이 리스트인지 확인하고, 리스트이면 첫 번째 요소를 사용
                x = x_val[0] if isinstance(x_val, list) else x_val
                y = y_val[0] if isinstance(y_val, list) else y_val
                w = w_val[0] if isinstance(w_val, list) else w_val
                h = h_val[0] if isinstance(h_val, list) else h_val
            except (ValueError, TypeError) as e:
                print(f"경고: box 데이터 형식이 예상과 다릅니다. 건너뜁니다. (box: {box}, 오류: {e})")
                continue
            # --- 수정된 부분 끝 ---

            text = detection.get('text', '').strip()
            # OCR 결과에서 번호 제거
            if ':' in text and text.split(':')[0].strip().isdigit():
                text = text.split(':', 1)[1].strip()

            regions.append({
                'text': text,
                'bbox': [int(x), int(y), int(x + w), int(y + h)]
            })
    return regions
