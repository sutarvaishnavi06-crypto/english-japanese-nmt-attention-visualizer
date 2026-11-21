"""Japanese and English tokenization module.
Uses SudachiPy for morphological analysis of Japanese text and regularized tokenization for English.
Preserves Japanese Kanji, Hiragana, Katakana, and punctuation.
"""
import re
import unicodedata
from typing import List

try:
    from sudachipy import tokenizer as su_tok, dictionary as su_dict
    _sudachi_dict = su_dict.Dictionary()
    _sudachi_tokenizer = _sudachi_dict.create()
    _SUDACHI_AVAILABLE = True
except Exception:
    _SUDACHI_AVAILABLE = False
    _sudachi_tokenizer = None


def normalize_japanese_text(text: str) -> str:
    """Normalize Japanese Unicode characters (NFKC) while preserving full-width punctuation."""
    if not text:
        return ""
    text = unicodedata.normalize('NFKC', text)
    text = re.sub(r'[\r\n\t]+', ' ', text)
    return text.strip()


def normalize_english_text(text: str) -> str:
    """Normalize and lowercase English text, standardizing punctuation."""
    if not text:
        return ""
    text = text.lower().strip()
    text = re.sub(r"([?.!,¿'\"()-])", r" \1 ", text)
    text = re.sub(r'[" "]+', " ", text)
    return text.strip()


def tokenize_english(text: str) -> List[str]:
    """Tokenize English text into list of lowercase tokens."""
    norm = normalize_english_text(text)
    if not norm:
        return []
    return norm.split()


def tokenize_japanese(text: str) -> List[str]:
    """Tokenize Japanese text using SudachiPy morphological segmentation (Mode C)."""
    text = normalize_japanese_text(text)
    if not text:
        return []

    if _SUDACHI_AVAILABLE and _sudachi_tokenizer is not None:
        try:
            morphemes = _sudachi_tokenizer.tokenize(text, su_tok.Tokenizer.SplitMode.C)
            tokens = [m.surface() for m in morphemes if m.surface().strip()]
            if tokens:
                return tokens
        except Exception:
            pass

    tokens = re.findall(r'[\u4e00-\u9faf]+|[\u3040-\u309f]+|[\u30a0-\u30ff]+|[a-zA-Z0-9]+|[^\s]', text)
    return [t for t in tokens if t.strip()]


def detokenize_japanese(tokens: List[str]) -> str:
    """Reconstruct natural Japanese text from a token list without spaces."""
    clean_tokens = [t for t in tokens if t not in ('<pad>', '<unk>', '<sos>', '<eos>')]
    return "".join(clean_tokens)


def detokenize_english(tokens: List[str]) -> str:
    """Reconstruct English sentence with proper punctuation spacing."""
    clean_tokens = [t for t in tokens if t not in ('<pad>', '<unk>', '<sos>', '<eos>')]
    text = " ".join(clean_tokens)
    text = re.sub(r'\s+([?.!,¿\'"])', r'\1', text)
    text = re.sub(r'(\()\s+', r'\1', text)
    return text.strip().capitalize()
