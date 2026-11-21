import torch
import torch.nn as nn
import torch.nn.functional as F

class BahdanauAttention(nn.Module):
    def __init__(self, enc_hidden_dim: int, dec_hidden_dim: int, attn_dim: int = 128):
        super().__init__()
        self.W_encoder = nn.Linear(enc_hidden_dim * 2, attn_dim, bias=False)
        self.W_decoder = nn.Linear(dec_hidden_dim, attn_dim, bias=False)
        self.v = nn.Linear(attn_dim, 1, bias=False)

    def forward(self, decoder_hidden: torch.Tensor, encoder_outputs: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        src_len = encoder_outputs.shape[1]
        dec_proj = self.W_decoder(decoder_hidden).unsqueeze(1).repeat(1, src_len, 1)
        enc_proj = self.W_encoder(encoder_outputs)
        energy = torch.tanh(enc_proj + dec_proj)
        scores = self.v(energy).squeeze(2)

        if mask is not None:
            scores = scores.masked_fill(~mask, -1e9)

        return F.softmax(scores, dim=1)
