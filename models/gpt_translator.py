import openai
import time
import re
from typing import List, Dict, Any, Tuple


class GPTTranslator:
    """GPT API 번역기"""

    def __init__(self, api_key, model="gpt-4"):
        if not api_key:
            raise ValueError("OpenAI API 키가 설정되지 않았습니다.")
        self.api_key = api_key
        self.model = model
        self.max_retries = 3
        self.retry_delay = 1.0

        self.client = openai.Client(api_key=self.api_key)
        print(f"GPT 번역기 초기화 완료 (모델: {self.model})")

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

    def analyze_context(self, clusters: List[List[str]]) -> Dict[str, Any]:
        """Analyze the context relationships between text clusters"""
        if not clusters:
            return {"context_type": "empty", "analysis": None}

        # Prepare clusters for analysis
        clusters_text = "\n\n".join([
            f"Cluster {i + 1}:\n" + "\n".join([f"- {text}" for text in cluster])
            for i, cluster in enumerate(clusters)
        ])

        context_prompt = f"""You are a comic narrative analysis expert examining text clusters from a comic page.

## CONTEXT ANALYSIS TASK:
Analyze these text clusters and identify:

1. **Speaker Identification**: Who is speaking in each cluster
2. **Dialogue Flow**: How conversations connect between clusters
3. **Scene Setting**: The likely environment/situation
4. **Emotional Context**: The emotional tone of each cluster
5. **Relationship Dynamics**: How characters relate to each other

## Text Clusters:
{clusters_text}

## Output Requirements:
Provide a JSON with:
- context_type: dialogue/narration/mixed
- speakers: array of likely speakers
- tone: dominant emotional tone
- scene_summary: brief description of the scene
- relationships: connections between clusters

Be concise and factual in your analysis.
"""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a comic narrative analysis expert."},
                    {"role": "user", "content": context_prompt}
                ],
                temperature=0.3,
                max_tokens=600,
            )
            result = response.choices[0].message.content.strip()
            return {"context_type": "analyzed", "analysis": result}
        except Exception as e:
            print(f"문맥 분석 중 오류: {e}")
            return {"context_type": "error", "analysis": None}

    def translate_single_text(self, text: str, source_lang: str = "English", target_lang: str = "Korean") -> str:
        try:
            prompt = self._create_enhanced_prompt(text, source_lang, target_lang)
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

    def _create_enhanced_prompt(self, text, source_lang, target_lang):
        return f"""You are an expert comic book localizer with 15+ years of experience translating English comics for Korean publishers. You understand both Western and Korean comic culture deeply.

## STEP 1: CONTEXT ANALYSIS
Before translating, analyze:
1. **Text Type**: Is this dialogue, narration, sound effect, or thought bubble?
2. **Character Voice**: Age, personality, social status indicators
3. **Emotional Tone**: Excitement, anger, sadness, surprise, etc.
4. **OCR Errors**: Identify concatenated words (e.g., "WHEREAREYOU" → "WHERE ARE YOU")

## STEP 2: CULTURAL ADAPTATION STRATEGY
- **Dialogue**: Use appropriate Korean speech levels (존댓말/반말/친근한말)
- **Sound Effects**: Convert to Korean comic conventions (쾅!/휘익!/두근두근)
- **Cultural References**: Adapt for Korean readers while preserving intent

## STEP 3: TRANSLATION EXECUTION
Create natural Korean that sounds like it was originally written by a Korean comic artist.

## QUALITY CHECKLIST:
- ✓ OCR errors corrected?
- ✓ Appropriate speech level used?
- ✓ Natural Korean flow?
- ✓ Emotional tone preserved?

## Text to Translate:
"{text}"

## Output:
[Provide ONLY the final Korean translation]"""

    def translate_batch(self, texts: List[str], source_lang: str = "English", target_lang: str = "Korean") -> List[str]:
        if not texts:
            return []
        try:
            batch_prompt = self.create_enhanced_batch_prompt(texts, source_lang, target_lang)
            api_config = self.get_batch_config()
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system",
                     "content": "You are a professional comic translator specializing in contextual translation."},
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

    def create_enhanced_batch_prompt(self, texts: List[str], source_lang: str, target_lang: str) -> str:
        prompt = f"""You are a senior comic localization director at a major Korean publisher. You're translating a complete comic page to maintain narrative flow and character consistency.

## PAGE-LEVEL CONTEXT ANALYSIS:
1. **Scene Setting**: Analyze the overall mood and situation
2. **Character Dynamics**: Identify speaker relationships and personalities
3. **Narrative Flow**: Ensure translations connect naturally
4. **Consistency Requirements**: Maintain character voices throughout

## TRANSLATION PROTOCOLS:

### Pre-Processing:
- Fix OCR concatenation errors (WHEREARE → WHERE ARE)
- Identify speaker age/status for appropriate Korean speech levels
- Note sound effects vs dialogue vs narration

### Cultural Adaptation Matrix:
- **Young Characters**: 반말, 줄임말 사용 (야, 뭐야, 헐)
- **Formal Situations**: 존댓말 (습니다/습니까 체)
- **Emotional Outbursts**: Korean emotional expressions (아이고, 어머나, 헉)
- **Action Sounds**: Korean onomatopoeia (쾅, 휘익, 쿵)

### Quality Standards:
- Each line must sound naturally Korean
- Maintain emotional intensity across the page
- Character personalities must remain consistent
- Dialogue flow should feel conversational

## Texts to Translate:
"""

        # 텍스트 추가
        for i, text in enumerate(texts, 1):
            prompt += f"{i}. {text}\n"

        prompt += """
## STRICT OUTPUT FORMAT REQUIREMENTS:
1. [Korean translation]
2. [Korean translation]
...

CRITICAL INSTRUCTIONS:
- Provide ONLY numbered Korean translations
- NO explanations, notes, or metadata after translations
- NO markdown formatting in translations
- NO "Final Check" or "Note" sections
- EXACTLY one translation per number, maintaining original numbering

Your response will be directly inserted into speech bubbles without processing, so it must contain ONLY translation text.
"""
        return prompt

    def translate_cluster(self, cluster: List[str], context_info: Dict[str, Any] = None) -> List[str]:
        """Translate a cluster of related texts with context awareness"""
        if not cluster:
            return []

        context_analysis = ""
        if context_info and context_info.get("analysis"):
            context_analysis = f"\n## Context Information:\n{context_info['analysis']}"

        cluster_prompt = f"""You are translating a cluster of related comic text elements that form a coherent narrative unit.

## CLUSTER TRANSLATION REQUIREMENTS:
- Maintain consistent tone and voice throughout the cluster
- Ensure natural dialogue flow between elements
- Preserve character personality traits across translations
- Adapt speech levels appropriately for each speaker

## Texts to Translate as a Cohesive Unit:{context_analysis}
"""

        for i, text in enumerate(cluster, 1):
            cluster_prompt += f"{i}. {text}\n"

        cluster_prompt += """
## Output Format:
Provide translations as a numbered list with exactly the same numbers as the input.
1. [Korean translation]
2. [Korean translation]
...

Translations must be in Korean only. No explanations or English text.
"""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a comic cluster translation specialist."},
                    {"role": "user", "content": cluster_prompt}
                ],
                temperature=0.2,
                max_tokens=800,
            )
            result = response.choices[0].message.content.strip()
            return self.parse_batch_translation_result(result, len(cluster))
        except Exception as e:
            print(f"클러스터 번역 오류: {e}")
            return self.translate_texts_individually(cluster)

    def translate_with_refinement(self, texts: List[str], source_lang: str = "English", target_lang: str = "Korean") -> \
    List[str]:
        """Two-stage translation: initial translation + quality refinement"""

        # Stage 1: Basic translation
        initial_translations = self.translate_batch(texts, source_lang, target_lang)

        # Stage 2: Quality refinement (optional)
        if len(texts) > 5:  # Only for complex pages
            refined_translations = self._refine_translations(texts, initial_translations)
            return refined_translations

        return initial_translations

    def translate_clusters(self, clusters: List[List[str]]) -> List[str]:
        """Translate multiple clusters with context awareness and ensure consistency"""
        if not clusters:
            return []

        # Get context analysis
        context_info = self.analyze_context(clusters)

        # Translate each cluster
        all_translations = []
        for i, cluster in enumerate(clusters):
            print(f"클러스터 {i + 1}/{len(clusters)} 번역 중...")
            cluster_translations = self.translate_cluster(cluster, context_info)
            all_translations.extend(cluster_translations)

        # Ensure consistency across all translations
        if len(all_translations) > 10:
            all_translations = self._ensure_consistency(clusters, all_translations)

        return all_translations

    def _refine_translations(self, originals: List[str], translations: List[str]) -> List[str]:
        """Refine translations to improve quality and consistency"""
        refinement_prompt = f"""Review and improve these comic translations:

## Quality Review Criteria:
- Natural Korean conversation flow
- Consistent character voices
- Appropriate formality levels
- Korean comic reader expectations

## Original → Current Translation pairs:
"""

        for i, (orig, trans) in enumerate(zip(originals, translations), 1):
            refinement_prompt += f"{i}. \"{orig}\" → \"{trans}\"\n"

        refinement_prompt += """
## Instructions:
If any translation sounds unnatural or inconsistent, provide an improved version.
Otherwise, keep the original translation.

## Output Format:
1. [Refined Korean translation]
2. [Refined Korean translation]
...

Provide only the final translations, no explanations.
"""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a Korean comic translation editor."},
                    {"role": "user", "content": refinement_prompt}
                ],
                temperature=0.2,
                max_tokens=1024,
            )
            result = response.choices[0].message.content.strip()
            return self.parse_batch_translation_result(result, len(originals))
        except Exception as e:
            print(f"번역 개선 중 오류: {e}")
            return translations

    def _ensure_consistency(self, original_clusters: List[List[str]], translations: List[str]) -> List[str]:
        """Ensure consistency across all translations"""
        # Prepare flat list of originals
        flat_originals = []
        for cluster in original_clusters:
            flat_originals.extend(cluster)

        consistency_prompt = f"""Review these translations for consistency across a complete comic page:

## Consistency Requirements:
- Character names and terms are consistent throughout
- Character speech patterns remain consistent
- Emotional tone flows naturally
- Korean speech levels are appropriate and consistent

## Original → Translation:
"""

        for i, (orig, trans) in enumerate(zip(flat_originals, translations), 1):
            consistency_prompt += f"{i}. \"{orig}\" → \"{trans}\"\n"

        consistency_prompt += """
## Output:
Provide the final consistent translations in numbered format:
1. [Final translation]
2. [Final translation]
...

Focus on making all translations work together as a cohesive page.
"""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a comic page consistency specialist."},
                    {"role": "user", "content": consistency_prompt}
                ],
                temperature=0.2,
                max_tokens=1200,
            )
            result = response.choices[0].message.content.strip()
            return self.parse_batch_translation_result(result, len(translations))
        except Exception as e:
            print(f"일관성 확보 중 오류: {e}")
            return translations

    def parse_batch_translation_result(self, result: str, expected_count: int) -> List[str]:
        """배치 번역 결과 파싱 (부가 정보 필터링 기능 추가)"""
        lines = result.strip().split('\n')
        translations = {}

        # 메타 정보 제거 (추가 설명 등)
        filtered_lines = []
        recording = True

        for line in lines:
            line = line.strip()
            if not line:
                continue

            # 번역 영역이 끝나고 부가 설명이 시작되는 패턴 감지
            if re.match(r'^(Note|참고|설명|번역자|Translator|Additional|추가|메모):.*', line, re.IGNORECASE):
                recording = False
                continue

            if recording:
                filtered_lines.append(line)

        # 다양한 번호 형식 인식을 위한 정규식 패턴
        patterns = [
            r'^(\d+)[\.\:\)\]]\s+(.+)$',  # 1. 텍스트, 1: 텍스트, 1) 텍스트, 1] 텍스트
            r'^(\d+)[\.]\s*(.+)$',  # 1.텍스트 (공백 없는 경우)
            r'^(\d+)[\s]+(.+)$',  # 1 텍스트 (숫자와 공백만 있는 경우)
            r'^[^\d]*(\d+)[^\d]+(.+)$'  # 기타 형식 (숫자가 포함된 경우)
        ]

        current_index = None

        for line in filtered_lines:
            # 여러 패턴 시도
            matched = False
            for pattern in patterns:
                match = re.match(pattern, line)
                if match:
                    try:
                        num = int(match.group(1))
                        trans_text = match.group(2).strip()
                        if 1 <= num <= expected_count:
                            # 각 번역 결과도 clean_translation_result 적용
                            trans_text = self.clean_translation_result(trans_text)
                            translations[num] = trans_text
                            current_index = num
                            matched = True
                            break
                    except (ValueError, IndexError):
                        continue

            # 번호 없이 이전 번역의 연속으로 판단되는 경우
            if not matched and current_index is not None and line:
                translations[current_index] += " " + line

        # 결과 목록 구성
        result_list = []
        missing_count = 0
        for i in range(1, expected_count + 1):
            if i in translations:
                result_list.append(translations[i])
            else:
                result_list.append(f"[번역 실패 {i}]")
                missing_count += 1
                print(f"경고: {i}번째 번역 결과 누락")

        if missing_count > 0:
            print(f"총 {missing_count}개 번역 결과 누락됨")

        return result_list

    # 기존 메소드들...
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

    def translate_texts_individually(self, texts: List[str], source_lang: str = "English",
                                     target_lang: str = "Korean") -> List[str]:
        print("배치 번역 실패. 개별 번역으로 전환합니다.")
        translations = []
        for i, text in enumerate(texts):
            print(f"개별 번역 진행: {i + 1}/{len(texts)}")
            translation = self.translate_single_text(text, source_lang, target_lang)
            translations.append(translation)
            time.sleep(self.retry_delay)
        return translations

    def clean_translation_result(self, translation: str) -> str:
        """번역 결과에서 부가 정보와 마크다운 형식을 제거합니다"""
        # 따옴표 제거
        if translation.startswith('"') and translation.endswith('"'):
            translation = translation[1:-1]

        # 접두어 제거
        prefixes_to_remove = ["번역:", "Translation:", "Korean:", "한국어:", "Corrected:", "Fixed:"]
        for prefix in prefixes_to_remove:
            if translation.startswith(prefix):
                translation = translation[len(prefix):].strip()

        # 마크다운 헤더나 부가 정보가 포함된 경우 제거
        markdown_patterns = [
            r"##\s+Final\s+Check.*$",
            r"##\s+출력.*$",
            r"##\s+Output.*$",
            r"##\s+[Nn]ote.*$",
            r"##\s+참고.*$",
            r"^-\s+.*$"  # 체크리스트 항목
        ]

        for pattern in markdown_patterns:
            match = re.search(pattern, translation, re.IGNORECASE | re.DOTALL)
            if match:
                translation = translation[:match.start()].strip()

        return translation.strip()
