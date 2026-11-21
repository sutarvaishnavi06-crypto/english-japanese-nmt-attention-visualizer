import re
from typing import List, Dict, Any
from .translator import Translator

class SRTEngine:
    def __init__(self, translator: Translator = None):
        self.translator = translator

    def parse_srt(self, srt_content: str) -> List[Dict[str, Any]]:
        content = srt_content.replace('\r\n', '\n').replace('\r', '\n')
        blocks = re.split(r'\n\s*\n', content.strip())
        parsed_subtitles = []

        for block in blocks:
            lines = [line.strip() for line in block.split('\n') if line.strip()]
            if len(lines) < 2:
                continue
            if re.match(r'^\d+$', lines[0]):
                sub_id = int(lines[0])
                time_line = lines[1]
                text_lines = lines[2:]
            else:
                sub_id = len(parsed_subtitles) + 1
                time_line = lines[0]
                text_lines = lines[1:]

            time_match = re.search(r'(\d{2}:\d{2}:\d{2}[,\.]\d{3})\s*-->\s*(\d{2}:\d{2}:\d{2}[,\.]\d{3})', time_line)
            if not time_match:
                continue

            parsed_subtitles.append({
                "id": sub_id,
                "start": time_match.group(1).replace('.', ','),
                "end": time_match.group(2).replace('.', ','),
                "en": " ".join(text_lines),
                "ja": ""
            })
        return parsed_subtitles

    def translate_subtitles(self, subtitles: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if self.translator is None:
            raise ValueError("No trained model loaded for subtitle translation.")
        translated_subs = []
        for item in subtitles:
            res = self.translator.translate_sentence(item["en"])
            translated_subs.append({
                "id": item["id"],
                "start": item["start"],
                "end": item["end"],
                "en": item["en"],
                "ja": res["translation"]
            })
        return translated_subs

    def export_srt(self, subtitles: List[Dict[str, Any]]) -> str:
        blocks = []
        for idx, sub in enumerate(subtitles, 1):
            blocks.append(f"{idx}\n{sub['start']} --> {sub['end']}\n{sub.get('ja', sub.get('en', ''))}\n")
        return "\n".join(blocks)
