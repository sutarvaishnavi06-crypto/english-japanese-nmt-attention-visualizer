import torch
import torch.nn as nn
from typing import Tuple
from .attention import BahdanauAttention

class Decoder(nn.Module):
    def __init__(self, trg_vocab_size: int, emb_dim: int, enc_hidden_dim: int, dec_hidden_dim: int, dropout: float = 0.2, pad_idx: int = 0, attention: BahdanauAttention = None):
        super().__init__()
        self.trg_vocab_size = trg_vocab_size
        self.attention = attention or BahdanauAttention(enc_hidden_dim, dec_hidden_dim)
        self.embedding = nn.Embedding(trg_vocab_size, emb_dim, padding_idx=pad_idx)

        enc_dim = enc_hidden_dim * 2
        self.gru = nn.GRU(emb_dim + enc_dim, dec_hidden_dim, batch_first=True)
        self.fc_out = nn.Linear(dec_hidden_dim + enc_dim + emb_dim, trg_vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward_step(self, input_token: torch.Tensor, decoder_hidden: torch.Tensor, encoder_outputs: torch.Tensor, mask: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        input_token = input_token.unsqueeze(1)
        embedded = self.dropout(self.embedding(input_token))

        attn_weights = self.attention(decoder_hidden, encoder_outputs, mask)
        context = torch.bmm(attn_weights.unsqueeze(1), encoder_outputs)

        gru_input = torch.cat((embedded, context), dim=2)
        gru_output, next_hidden = self.gru(gru_input, decoder_hidden.unsqueeze(0))
        next_hidden = next_hidden.squeeze(0)

        concat_features = torch.cat((next_hidden, context.squeeze(1), embedded.squeeze(1)), dim=1)
        prediction = self.fc_out(concat_features)

        return prediction, next_hidden, attn_weights
