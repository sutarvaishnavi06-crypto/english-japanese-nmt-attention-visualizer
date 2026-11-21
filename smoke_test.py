"""Small forward/backward smoke test for the GRU + Bahdanau model."""
import torch
from app.model.encoder import Encoder
from app.model.attention import BahdanauAttention
from app.model.decoder import Decoder
from app.model.seq2seq import Seq2Seq


def main():
    device = torch.device("cpu")
    src_vocab, trg_vocab = 32, 36
    encoder = Encoder(src_vocab, emb_dim=16, enc_hidden_dim=12, dec_hidden_dim=16, dropout=0.0, pad_idx=0)
    attention = BahdanauAttention(enc_hidden_dim=12, dec_hidden_dim=16, attn_dim=8)
    decoder = Decoder(trg_vocab, emb_dim=16, enc_hidden_dim=12, dec_hidden_dim=16, dropout=0.0, pad_idx=0, attention=attention)
    model = Seq2Seq(encoder, decoder, pad_idx=0, device=device).to(device)
    src = torch.tensor([[4, 5, 6, 3], [7, 8, 3, 0]], dtype=torch.long)
    trg = torch.tensor([[2, 9, 10, 3], [2, 11, 3, 0]], dtype=torch.long)
    lengths = torch.tensor([4, 3])
    outputs, weights = model(src, trg, teacher_forcing_ratio=1.0, src_lens=lengths)
    loss = outputs[:, 1:, :].reshape(-1, trg_vocab).sum() + weights.sum()
    loss.backward()
    assert outputs.shape == (2, 4, trg_vocab)
    assert weights.shape == (2, 4, 4)
    assert torch.isfinite(outputs).all() and torch.isfinite(weights).all()
    print("SMOKE TEST PASSED: forward, attention output shapes, finite values, and backward pass.")

if __name__ == "__main__":
    main()
