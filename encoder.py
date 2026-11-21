import torch
import torch.nn as nn
from typing import Tuple

class Encoder(nn.Module):
    def __init__(self, src_vocab_size: int, emb_dim: int, enc_hidden_dim: int, dec_hidden_dim: int, dropout: float = 0.2, pad_idx: int = 0):
        super().__init__()
        self.embedding = nn.Embedding(src_vocab_size, emb_dim, padding_idx=pad_idx)
        self.gru = nn.GRU(emb_dim, enc_hidden_dim, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(enc_hidden_dim * 2, dec_hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src: torch.Tensor, src_lens: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor]:
        embedded = self.dropout(self.embedding(src))
        if src_lens is not None:
            packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, src_lens.to('cpu'), batch_first=True, enforce_sorted=False)
            packed_outputs, hidden = self.gru(packed_embedded)
            outputs, _ = nn.utils.rnn.pad_packed_sequence(packed_outputs, batch_first=True)
        else:
            outputs, hidden = self.gru(embedded)

        concat_hidden = torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)
        dec_init_hidden = torch.tanh(self.fc(concat_hidden))
        return outputs, dec_init_hidden
