import openai
import time
from typing import List

class GPTTranslator:
    """GPT API 전용 번역기 (openai 최신 버전 호환)"""

    def __init__(self, api_key, model="gpt-4"):
        if not api_key:
            raise ValueError("OpenAI API 키가 설정되지 않았습니다.")
        self.api_key = api_key
        self.model = model
        self.max_retries = 3
        self.retry_delay = 1.0

        self.client = openai.Client(api_key=self.api_key)
        print(f"GPT 번역기 초기화 완료 (모델: {self.model}, 최신 openai 패키지 호환)")

    def get_single_config(self):
        return {
            "temperature": 0.2,
            "max_tokens": 256,
            "top_p": 1,
            "frequency_penalty": 0,
            "presence_penalty": 0,
            "timeout": 30,
        }

    def get_batch_config(self):
        return {
            "temperature": 0.2,
            "max_tokens": 1024,
            "top_p": 1,
            "frequency_penalty": 0,
            "presence_penalty": 0,
            "timeout": 60,
        }

    def translate_single_text(self, text: str, source_lang: str = "English", target_lang: str = "Korean") -> str:
        try:
            prompt = self._create_prompt(text, source_lang, target_lang)
            api_config = self.get_single_config()
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system",
                     "content": f"You are a professional comic translator specializing in {source_lang} to {target_lang} translation."},
                    {"role": "user", "content": prompt}
                ],
                temperature=api_config["temperature"],
                max_tokens=api_config["max_tokens"],
                top_p=api_config["top_p"],
                frequency_penalty=api_config["frequency_penalty"],
                presence_penalty=api_config["presence_penalty"],
                timeout=api_config["timeout"],
            )
            translation = response.choices[0].message.content.strip()
            return self.clean_translation_result(translation)
        except Exception as e:
            print(f"GPT API 호출 오류 (단일): {e}")
            if self.max_retries > 0:
                return self._retry_translation(self.translate_single_text, text, source_lang, target_lang)
            return f"[번역 실패] {text}"

    def translate_batch(self, texts: List[str], source_lang: str = "English", target_lang: str = "Korean") -> List[str]:
        if not texts:
            return []
        try:
            batch_prompt = self.create_ocr_aware_batch_prompt(texts, source_lang, target_lang)
            api_config = self.get_batch_config()
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a professional comic translator..."},
                    {"role": "user", "content": batch_prompt}
                ],
                temperature=api_config["temperature"],
                max_tokens=api_config["max_tokens"],
                top_p=api_config["top_p"],
                frequency_penalty=api_config["frequency_penalty"],
                presence_penalty=api_config["presence_penalty"],
                timeout=api_config["timeout"],
            )
            translation_result = response.choices[0].message.content.strip()
            return self.parse_batch_translation_result(translation_result, len(texts))
        except Exception as e:
            print(f"GPT API 호출 오류 (배치): {e}")
            return self.translate_texts_individually(texts, source_lang, target_lang)

    def _retry_translation(self, func, *args, **kwargs):
        for attempt in range(self.max_retries):
            print(f"재시도 {attempt + 1}/{self.max_retries}...")
            try:
                time.sleep(self.retry_delay * (attempt + 1))
                return func(*args, **kwargs)
            except Exception as e:
                if attempt == self.max_retries - 1:
                    print(f"최종 재시도 실패: {e}")
                    return f"[번역 실패] {args[0] if args else ''}"
                continue

    def _create_prompt(self, text, source_lang, target_lang):
        return f"""You are an expert comic book localizer, specializing in translating English comics into natural, modern, and engaging Korean. Your task is to not just translate, but to adapt the text for a Korean audience while preserving the original intent and emotion.

## Primary Goal:
Translate the following English text into Korean, making it sound as if it were originally written by a native Korean comic artist.

## CRITICAL INSTRUCTIONS:

1.  **Pre-computation (Mental Correction):** Before translating, you MUST first mentally identify and correct any potential OCR (Optical Character Recognition) errors. The most common error is words being concatenated without spaces (e.g., "OFYOUWILL" must be processed as "OF YOU WILL").

2.  **Tone and Style Adaptation:**
    *   **Dialogue:** Convert stiff or literal English into natural, conversational Korean dialogue (구어체). Consider the character's personality and the situation.
    *   **Narration:** Translate narration boxes with a slightly more formal but still engaging tone.
    *   **Emotional Nuance:** Pay close attention to interjections (e.g., "Ugh," "Phew," "Wow") and translate the underlying emotion, not just the word itself.

3.  **Cultural Localization:**
    *   **Onomatopoeia & Sound Effects (SFX):** Do NOT literally translate sound effects. Instead, replace them with their standard Korean comic equivalents (e.g., "BANG!" might become "쾅!" or "탕!").
    *   **Colloquialisms:** Use common Korean slang or informal expressions where they fit the character and context to enhance realism.

4.  **Output Formatting:**
    *   Provide ONLY the final, translated Korean text.
    *   Do NOT include any of your reasoning, explanations, or the original English text in the response.

## Text to Translate:
"{text}"
"""

    def create_ocr_aware_batch_prompt(self, texts: List[str], source_lang: str, target_lang: str) -> str:
        prompt = f"""
        You are an expert comic book localizer... (same persona and instructions as above)

## Primary Goal:
Translate the following numbered list of English text snippets from a single comic page. Maintain context and consistency across all lines.

## CRITICAL INSTRUCTIONS: (same as above)
...

## Output Formatting:
*   Provide a numbered list of translations that corresponds EXACTLY to the input numbers.
*   Provide ONLY the final, translated Korean text for each line.

## Your Response Format (Example):
1. 안녕!
2. 누구시죠?
3. 흠...

## Texts to Translate:
"""
        for i, text in enumerate(texts, 1):
            prompt += f"{i}. {text}\n"
        prompt += f"\nProvide translations in this exact format:\n1. [corrected and translated text]\n2. [corrected and translated text]\n..."
        return prompt

    def parse_batch_translation_result(self, result: str, expected_count: int) -> List[str]:
        lines = result.strip().split('\n')
        translations = {}
        for line in lines:
            line = line.strip()
            if not line:
                continue
            if line and line[0].isdigit() and '. ' in line:
                try:
                    num_part, trans_part = line.split('. ', 1)
                    translations[int(num_part)] = trans_part
                except IndexError:
                    continue
        return [translations.get(i, "[번역 실패]") for i in range(1, expected_count + 1)]

    def translate_texts_individually(self, texts: List[str], source_lang: str = "English",
                                     target_lang: str = "Korean") -> List[str]:
        print("배치 번역 실패. 개별 번역으로 전환합니다.")
        translations = []
        for text in texts:
            translation = self.translate_single_text(text, source_lang, target_lang)
            translations.append(translation)
            time.sleep(self.retry_delay)
        return translations

    def clean_translation_result(self, translation: str) -> str:
        if translation.startswith('"') and translation.endswith('"'):
            translation = translation[1:-1]
        prefixes_to_remove = ["번역:", "Translation:", "Korean:", "한국어:", "Corrected:", "Fixed:"]
        for prefix in prefixes_to_remove:
            if translation.startswith(prefix):
                translation = translation[len(prefix):].strip()
        return translation.strip()

    def translate_text(self, english_text: str) -> str:
        return self.translate_single_text(english_text, "English", "Korean")
