import torch
from torch.utils.data import Dataset
from typing import List, Dict, Tuple
from app.utils.vocab import Vocabulary, PAD_IDX

class TranslationDataset(Dataset):
    def __init__(self, data_list: List[Dict[str, str]], src_vocab: Vocabulary, trg_vocab: Vocabulary):
        self.data = data_list
        self.src_vocab = src_vocab
        self.trg_vocab = trg_vocab

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, any]:
        item = self.data[idx]
        en_tokens = item["en_tokens"]
        ja_tokens = item["ja_tokens"]

        src_indices = self.src_vocab.encode(en_tokens, add_sos=False, add_eos=True)
        trg_indices = self.trg_vocab.encode(ja_tokens, add_sos=True, add_eos=True)

        return {
            "src": torch.tensor(src_indices, dtype=torch.long),
            "trg": torch.tensor(trg_indices, dtype=torch.long),
            "en_tokens": en_tokens,
            "ja_tokens": ja_tokens
        }

def collate_fn(batch: List[Dict[str, any]]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    src_list = [item["src"] for item in batch]
    trg_list = [item["trg"] for item in batch]

    src_lens = torch.tensor([len(s) for s in src_list], dtype=torch.long)
    src_padded = torch.nn.utils.rnn.pad_sequence(src_list, batch_first=True, padding_value=PAD_IDX)
    trg_padded = torch.nn.utils.rnn.pad_sequence(trg_list, batch_first=True, padding_value=PAD_IDX)

    return src_padded, trg_padded, src_lens
