import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


class ComicTextRenderer:
    """만화 텍스트 렌더링 클래스"""

    def __init__(self, font_path: str):
        self.font_path = font_path

    def remove_text(self, image: np.ndarray, regions: list) -> np.ndarray:
        """원본 텍스트 영역을 제거합니다."""
        result = image.copy()
        for region in regions:
            x1, y1, x2, y2 = region['bbox']
            cv2.rectangle(result, (x1 - 3, y1 - 3), (x2 + 3, y2 + 3), (255, 255, 255), -1)
        return result

    def _get_optimal_font(self, text: str, width: int, height: int) -> ImageFont.FreeTypeFont:
        """박스에 맞는 최적의 폰트 크기를 찾습니다."""
        for size in range(48, 10, -2):
            font = ImageFont.truetype(self.font_path, size)
            wrapped_text = self._wrap_text(text, font, width * 0.95)
            bbox = ImageDraw.Draw(Image.new('RGB', (1, 1))).multiline_textbbox((0, 0), wrapped_text, font=font)
            text_width, text_height = bbox[2] - bbox[0], bbox[3] - bbox[1]
            if text_width <= width * 0.95 and text_height <= height * 0.95:
                return font
        return ImageFont.truetype(self.font_path, 12)

    def _wrap_text(self, text: str, font: ImageFont.FreeTypeFont, max_width: int) -> str:
        """의미 단위로 텍스트를 줄바꿈합니다."""
        words = text.split()
        lines = []
        current_line = ""
        for word in words:
            draw = ImageDraw.Draw(Image.new('RGB', (1, 1)))
            if draw.textlength(current_line + " " + word, font=font) <= max_width:
                current_line += " " + word
            else:
                lines.append(current_line.strip())
                current_line = word
        lines.append(current_line.strip())
        return "\n".join(lines).strip()

    def render_text(self, image: np.ndarray, regions: list, translations: list) -> np.ndarray:
        """번역된 텍스트를 이미지에 렌더링합니다."""
        pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil_image)

        for region, text in zip(regions, translations):
            x1, y1, x2, y2 = region['bbox']
            width, height = x2 - x1, y2 - y1

            font = self._get_optimal_font(text, width, height)
            wrapped_text = self._wrap_text(text, font, width * 0.95)

            bbox = draw.multiline_textbbox((0, 0), wrapped_text, font, align="center")
            text_width, text_height = bbox[2] - bbox[0], bbox[3] - bbox[1]

            pos_x = x1 + (width - text_width) / 2
            pos_y = y1 + (height - text_height) / 2

            outline_thickness = 2
            for x_offset in range(-outline_thickness, outline_thickness + 1):
                for y_offset in range(-outline_thickness, outline_thickness + 1):
                    if x_offset != 0 or y_offset != 0:
                        draw.multiline_text((pos_x + x_offset, pos_y + y_offset), wrapped_text, font=font,
                                            fill=(255, 255, 255), align="center")

            draw.multiline_text((pos_x, pos_y), wrapped_text, font=font, fill=(0, 0, 0), align="center")

        return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
