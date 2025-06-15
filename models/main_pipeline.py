import cv2
from typing import List, Tuple
import os
from datetime import datetime
from .gpt_translator import GPTTranslator
from .comic_renderer import ComicTextRenderer
import numpy as np
from .paddleocr_clustering import process_images_paddleocr
from PIL import Image

class ComicTranslationPipeline:

    def __init__(self, api_key: str, font_path: str):
        self.translator = GPTTranslator(api_key=api_key)
        self.renderer = ComicTextRenderer(font_path=font_path)
        print("만화 번역 파이프라인 준비 완료.")

    def translate_comic_with_ocr_pil(self, image: Image.Image, image_name: str):
        # PIL 이미지를 OpenCV 형식으로 변환
        image_cv = cv2.cvtColor(np.array(image.convert("RGB")), cv2.COLOR_RGB2BGR)

        # 임시 파일 저장 없이 OCR 처리
        # PaddleOCR는 파일 경로 기반이므로, 메모리에서 바로 처리하려면 추가 구현 필요
        # 여기서는 임시 파일 저장 방식 예시 (메모리 처리 지원 시 해당 부분 수정)
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = os.path.join(tmpdir, image_name)
            cv2.imwrite(tmp_path, image_cv)
            ocr_results = process_images_paddleocr(
                image_dir=tmpdir,
                output_image_dir=None,
                json_output_path=None
            )
            ocr_regions = ocr_results[image_name]["results"]

        # 번역 및 렌더링
        regions = []
        for detection in ocr_regions:
            box = detection.get('box', [0, 0, 0, 0])
            text = detection.get('text', '').strip()
            if ':' in text and text.split(':')[0].strip().isdigit():
                text = text.split(':', 1)[1].strip()
            x, y, w, h = box
            regions.append({'text': text, 'bbox': [x, y, x + w, y + h]})

        if not regions:
            return image, []

        original_texts = [region['text'] for region in regions]
        translations = self.translator.translate_batch(original_texts)
        cleaned_image = self.renderer.remove_text(image_cv, regions)
        final_image = self.renderer.render_text(cleaned_image, regions, translations)

        # OpenCV 이미지를 PIL.Image로 변환하여 반환
        result_pil = Image.fromarray(cv2.cvtColor(final_image, cv2.COLOR_BGR2RGB))
        translation_pairs = list(zip(original_texts, translations))
        return result_pil, translation_pairs

    def _generate_timestamped_path(self, base_path: str) -> str:
        dir_name = os.path.dirname(base_path)
        base_name = os.path.basename(base_path)
        file_name, ext = os.path.splitext(base_name)
        timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
        new_filename = f"{file_name}_{timestamp}{ext}"
        os.makedirs(dir_name, exist_ok=True)
        return os.path.join(dir_name, new_filename)

    def translate_comic_from_ocr_result(self, image_path: str, ocr_regions: list, output_path: str) \
            -> Tuple[np.ndarray, List[Tuple[str, str]]]:
        print(f"'{image_path}' 번역 시작...")

        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"이미지 파일을 찾을 수 없습니다: {image_path}")

        regions = []
        for detection in ocr_regions:
            box = detection.get('box', [0, 0, 0, 0])
            text = detection.get('text', '').strip()
            if ':' in text and text.split(':')[0].strip().isdigit():
                text = text.split(':', 1)[1].strip()
            x, y, w, h = box
            regions.append({'text': text, 'bbox': [x, y, x + w, y + h]})

        if not regions:
            print("번역할 텍스트가 없습니다.")
            return image, []

        original_texts = [region['text'] for region in regions]
        print(f"{len(original_texts)}개의 텍스트를 번역합니다...")

        translations = self.translator.translate_batch(original_texts)
        cleaned_image = self.renderer.remove_text(image, regions)
        final_image = self.renderer.render_text(cleaned_image, regions, translations)

        timestamped_output_path = self._generate_timestamped_path(output_path)
        cv2.imwrite(timestamped_output_path, final_image)
        print(f"번역 완료! 결과 저장: {timestamped_output_path}")

        translation_pairs = list(zip(original_texts, translations))
        return final_image, translation_pairs

    def translate_comic_with_ocr(self, image_dir: str, image_name: str, output_path: str, output_image_dir: str = None):
        # PaddleOCR + 클러스터링 결과 딕셔너리 획득
        ocr_results = process_images_paddleocr(
            image_dir=image_dir,
            output_image_dir=output_image_dir,
            json_output_path=None  # 파일 저장 안함
        )
        ocr_regions = ocr_results[image_name]["results"]
        return self.translate_comic_from_ocr_result(
            image_path=os.path.join(image_dir, image_name),
            ocr_regions=ocr_regions,
            output_path=output_path
        )
