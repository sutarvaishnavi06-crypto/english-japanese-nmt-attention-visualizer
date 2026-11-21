import random
import torch
import torch.nn as nn
from typing import Tuple
from .encoder import Encoder
from .decoder import Decoder

class Seq2Seq(nn.Module):
    def __init__(self, encoder: Encoder, decoder: Decoder, pad_idx: int = 0, sos_idx: int = 2, eos_idx: int = 3, device: torch.device = torch.device('cpu')):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.pad_idx = pad_idx
        self.sos_idx = sos_idx
        self.eos_idx = eos_idx
        self.device = device

    def create_mask(self, src: torch.Tensor) -> torch.Tensor:
        return src != self.pad_idx

    def forward(self, src: torch.Tensor, trg: torch.Tensor, teacher_forcing_ratio: float = 0.5, src_lens: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = src.shape[0]
        trg_len = trg.shape[1]
        trg_vocab_size = self.decoder.trg_vocab_size

        outputs = torch.zeros(batch_size, trg_len, trg_vocab_size, device=self.device)
        attentions = torch.zeros(batch_size, trg_len, src.shape[1], device=self.device)

        encoder_outputs, hidden = self.encoder(src, src_lens)
        mask = self.create_mask(src)
        input_token = trg[:, 0]

        for t in range(1, trg_len):
            prediction, hidden, attn_weights = self.decoder.forward_step(input_token, hidden, encoder_outputs, mask)
            outputs[:, t, :] = prediction
            attentions[:, t, :] = attn_weights

            teacher_force = random.random() < teacher_forcing_ratio
            top1 = prediction.argmax(1)
            input_token = trg[:, t] if teacher_force else top1

        return outputs, attentions
