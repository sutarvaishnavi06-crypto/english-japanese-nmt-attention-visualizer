import json
import pickle
from collections import Counter
from typing import List, Dict, Optional

PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"
SOS_TOKEN = "<sos>"
EOS_TOKEN = "<eos>"

PAD_IDX = 0
UNK_IDX = 1
SOS_IDX = 2
EOS_IDX = 3

SPECIAL_TOKENS = [PAD_TOKEN, UNK_TOKEN, SOS_TOKEN, EOS_TOKEN]


class Vocabulary:
    def __init__(self, name: str = "vocab", min_freq: int = 2, max_size: Optional[int] = 10000):
        self.name = name
        self.min_freq = min_freq
        self.max_size = max_size

        self.w2i: Dict[str, int] = {}
        self.i2w: Dict[int, str] = {}
        self.counts: Counter = Counter()

        self._initialize_special_tokens()

    def _initialize_special_tokens(self):
        for idx, token in enumerate(SPECIAL_TOKENS):
            self.w2i[token] = idx
            self.i2w[idx] = token

    def build_vocab(self, tokenized_sentences: List[List[str]]):
        for sent in tokenized_sentences:
            self.counts.update(sent)

        sorted_tokens = [tok for tok, count in self.counts.most_common() if count >= self.min_freq]

        if self.max_size is not None:
            allowed_words = self.max_size - len(SPECIAL_TOKENS)
            sorted_tokens = sorted_tokens[:allowed_words]

        idx = len(SPECIAL_TOKENS)
        for tok in sorted_tokens:
            if tok not in self.w2i:
                self.w2i[tok] = idx
                self.i2w[idx] = tok
                idx += 1

    def __len__(self) -> int:
        return len(self.w2i)

    def encode(self, tokens: List[str], add_sos: bool = True, add_eos: bool = True) -> List[int]:
        indices = []
        if add_sos:
            indices.append(SOS_IDX)
        for tok in tokens:
            indices.append(self.w2i.get(tok, UNK_IDX))
        if add_eos:
            indices.append(EOS_IDX)
        return indices

    def decode(self, indices: List[int], remove_special: bool = True) -> List[str]:
        tokens = []
        for idx in indices:
            if idx == EOS_IDX and remove_special:
                break
            tok = self.i2w.get(idx, UNK_TOKEN)
            if remove_special and tok in (PAD_TOKEN, SOS_TOKEN):
                continue
            tokens.append(tok)
        return tokens

    def save(self, filepath: str):
        data = {
            "name": self.name,
            "min_freq": self.min_freq,
            "max_size": self.max_size,
            "w2i": self.w2i,
            "i2w": {str(k): v for k, v in self.i2w.items()},
            "counts": dict(self.counts)
        }
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)

    @classmethod
    def load(cls, filepath: str) -> 'Vocabulary':
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        vocab = cls(name=data["name"], min_freq=data["min_freq"], max_size=data["max_size"])
        vocab.w2i = data["w2i"]
        vocab.i2w = {int(k): v for k, v in data["i2w"].items()}
        vocab.counts = Counter(data.get("counts", {}))
        return vocab
