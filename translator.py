import torch
from typing import List, Dict, Any
from app.utils.tokenizer import tokenize_english, detokenize_japanese
from app.utils.vocab import Vocabulary, PAD_IDX, SOS_IDX, EOS_IDX, UNK_IDX
from app.model.seq2seq import Seq2Seq

class Translator:
    def __init__(self, model: Seq2Seq, src_vocab: Vocabulary, trg_vocab: Vocabulary, device: torch.device = None):
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model.to(self.device)
        self.model.eval()
        self.src_vocab = src_vocab
        self.trg_vocab = trg_vocab

    def translate_sentence(self, sentence: str, max_len: int = 50, repetition_penalty: float = 1.2) -> Dict[str, Any]:
        if not sentence or not sentence.strip():
            return {"translation": "", "src_tokens": [], "trg_tokens": [], "attention_matrix": []}

        src_tokens = tokenize_english(sentence)
        if not src_tokens:
            return {"translation": "", "src_tokens": [], "trg_tokens": [], "attention_matrix": []}

        src_indices = self.src_vocab.encode(src_tokens, add_sos=False, add_eos=True)
        src_tensor = torch.tensor(src_indices, dtype=torch.long, device=self.device).unsqueeze(0)

        with torch.no_grad():
            encoder_outputs, hidden = self.model.encoder(src_tensor)
            mask = self.model.create_mask(src_tensor)

            input_token = torch.tensor([SOS_IDX], dtype=torch.long, device=self.device)
            trg_indices: List[int] = []
            attention_steps: List[List[float]] = []
            generated_set = set()

            for step in range(max_len):
                prediction, hidden, attn_weights = self.model.decoder.forward_step(input_token, hidden, encoder_outputs, mask)
                logits = prediction.squeeze(0).clone()

                for past_idx in generated_set:
                    if logits[past_idx] > 0:
                        logits[past_idx] /= repetition_penalty
                    else:
                        logits[past_idx] *= repetition_penalty

                logits[UNK_IDX] -= 2.0
                logits[PAD_IDX] = -1e9
                logits[SOS_IDX] = -1e9

                best_idx = logits.argmax().item()
                if best_idx == EOS_IDX:
                    break

                trg_indices.append(best_idx)
                generated_set.add(best_idx)
                attention_steps.append(attn_weights.squeeze(0).tolist())
                input_token = torch.tensor([best_idx], dtype=torch.long, device=self.device)

        trg_tokens = [self.trg_vocab.i2w.get(idx, "<unk>") for idx in trg_indices]
        detok_ja = detokenize_japanese(trg_tokens)
        display_src_tokens = src_tokens + ["<eos>"]
        trimmed_attn = [step[:len(display_src_tokens)] for step in attention_steps]

        return {
            "translation": detok_ja,
            "src_tokens": display_src_tokens,
            "trg_tokens": trg_tokens,
            "attention_matrix": trimmed_attn
        }
