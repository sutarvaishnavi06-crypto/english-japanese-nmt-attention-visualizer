import math
from collections import Counter
from typing import List, Dict, Any
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from app.utils.vocab import PAD_IDX
from app.model.seq2seq import Seq2Seq
from app.inference.translator import Translator
from training.dataset import TranslationDataset, collate_fn

def calculate_sentence_bleu(candidate: List[str], references: List[List[str]], max_n: int = 4) -> float:
    """Compute BLEU score with standard Chen & Cherry smoothing method 1."""
    if not candidate:
        return 0.0
    cand_len = len(candidate)
    ref_lens = [len(ref) for ref in references]
    closest_ref_len = min(ref_lens, key=lambda ref_l: (abs(ref_l - cand_len), ref_l))
    bp = 1.0 if cand_len > closest_ref_len else (math.exp(1 - closest_ref_len / cand_len) if cand_len > 0 else 0.0)

    p_ns = []
    for n in range(1, max_n + 1):
        cand_ngrams = [tuple(candidate[i:i+n]) for i in range(len(candidate) - n + 1)]
        if not cand_ngrams:
            p_ns.append(1e-5)
            continue
        cand_counts = Counter(cand_ngrams)
        max_ref_counts = Counter()
        for ref in references:
            ref_ngrams = Counter([tuple(ref[i:i+n]) for i in range(len(ref) - n + 1)])
            for ng in cand_counts:
                max_ref_counts[ng] = max(max_ref_counts[ng], ref_ngrams[ng])

        clipped = sum(min(count, max_ref_counts[ng]) for ng, count in cand_counts.items())
        p_n = (clipped + 0.1) / (len(cand_ngrams) + 0.1)
        p_ns.append(max(p_n, 1e-7))

    log_sum = sum(math.log(p) for p in p_ns) / max_n
    return bp * math.exp(log_sum) * 100.0

def evaluate_test_set(model: Seq2Seq, test_data: List[Dict[str, Any]], src_vocab, trg_vocab, device: torch.device, sample_bleu_count: int = 200) -> Dict[str, Any]:
    model.eval()
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)
    dataset = TranslationDataset(test_data, src_vocab, trg_vocab)
    loader = DataLoader(dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)

    total_loss, total_tokens = 0.0, 0
    with torch.no_grad():
        for batch_src, batch_trg, src_lens in loader:
            batch_src = batch_src.to(device)
            batch_trg = batch_trg.to(device)
            src_lens = src_lens.to(device)
            outputs, _ = model(batch_src, batch_trg, teacher_forcing_ratio=0.0, src_lens=src_lens)
            output_dim = outputs.shape[-1]
            loss = criterion(outputs[:, 1:, :].reshape(-1, output_dim), batch_trg[:, 1:].reshape(-1))
            non_pad = (batch_trg[:, 1:] != PAD_IDX).sum().item()
            total_loss += loss.item() * non_pad
            total_tokens += non_pad

    avg_test_loss = total_loss / max(total_tokens, 1)
    translator = Translator(model, src_vocab, trg_vocab, device=device)
    bleu_scores = []
    for item in test_data[:sample_bleu_count]:
        pred = translator.translate_sentence(item["en"])
        bleu_scores.append(calculate_sentence_bleu(pred["trg_tokens"], [item["ja_tokens"]]))

    avg_bleu = sum(bleu_scores) / max(len(bleu_scores), 1)
    return {
        "test_loss": round(avg_test_loss, 4),
        "perplexity": round(math.exp(min(avg_test_loss, 100.0)), 2),
        "bleu": round(avg_bleu, 2),
        "total_test_samples": len(test_data),
        "bleu_evaluated_samples": min(sample_bleu_count, len(test_data))
    }
