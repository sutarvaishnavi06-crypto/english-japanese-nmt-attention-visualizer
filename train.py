import os, json, time, math, random, argparse
from typing import Dict, Any, List
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from app.utils.tokenizer import tokenize_english, tokenize_japanese
from app.utils.vocab import Vocabulary, PAD_IDX
from app.model.encoder import Encoder
from app.model.attention import BahdanauAttention
from app.model.decoder import Decoder
from app.model.seq2seq import Seq2Seq
from training.dataset import TranslationDataset, collate_fn
from evaluation.evaluate import evaluate_test_set

def set_seed(seed: int = 42):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def load_and_tokenize_split(split_path: str, max_samples: int = None) -> List[Dict[str, Any]]:
    with open(split_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if max_samples:
        data = data[:max_samples]
    for item in data:
        if "en_tokens" not in item:
            item["en_tokens"] = tokenize_english(item["en"])
        if "ja_tokens" not in item:
            item["ja_tokens"] = tokenize_japanese(item["ja"])
    return data

def train_one_epoch(model, loader, optimizer, criterion, clip, teacher_forcing_ratio, device):
    model.train()
    epoch_loss, total_tokens = 0.0, 0
    for i, (src, trg, src_lens) in enumerate(loader):
        src, trg, src_lens = src.to(device), trg.to(device), src_lens.to(device)
        optimizer.zero_grad()
        outputs, _ = model(src, trg, teacher_forcing_ratio=teacher_forcing_ratio, src_lens=src_lens)
        output_dim = outputs.shape[-1]
        loss = criterion(outputs[:, 1:, :].reshape(-1, output_dim), trg[:, 1:].reshape(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        non_pad = (trg[:, 1:] != PAD_IDX).sum().item()
        epoch_loss += loss.item() * non_pad
        total_tokens += non_pad
    return epoch_loss / max(total_tokens, 1)

def evaluate_epoch(model, loader, criterion, device):
    model.eval()
    epoch_loss, total_tokens = 0.0, 0
    with torch.no_grad():
        for src, trg, src_lens in loader:
            src, trg, src_lens = src.to(device), trg.to(device), src_lens.to(device)
            outputs, _ = model(src, trg, teacher_forcing_ratio=0.0, src_lens=src_lens)
            output_dim = outputs.shape[-1]
            loss = criterion(outputs[:, 1:, :].reshape(-1, output_dim), trg[:, 1:].reshape(-1))
            non_pad = (trg[:, 1:] != PAD_IDX).sum().item()
            epoch_loss += loss.item() * non_pad
            total_tokens += non_pad
    return epoch_loss / max(total_tokens, 1)

def train_pipeline(train_size=10000, val_size=1200, test_size=1200, emb_dim=128, enc_hidden_dim=256, dec_hidden_dim=256, batch_size=64, epochs=4, lr=0.001, clip=1.0):
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    checkpoint_dir = os.path.join(base, "models", "checkpoints")
    splits_dir = os.path.join(base, "data", "splits")
    os.makedirs(checkpoint_dir, exist_ok=True)

    print(f"[*] Training device: {device}")
    print("[*] Loading splits...")
    train_data = load_and_tokenize_split(os.path.join(splits_dir, "train.json"), max_samples=train_size)
    val_data = load_and_tokenize_split(os.path.join(splits_dir, "val.json"), max_samples=val_size)
    test_data = load_and_tokenize_split(os.path.join(splits_dir, "test.json"), max_samples=test_size)

    print(f"  [+] Loaded pairs: Train={len(train_data)}, Val={len(val_data)}, Test={len(test_data)}")
    src_vocab = Vocabulary("en_vocab", min_freq=2, max_size=7500)
    src_vocab.build_vocab([item["en_tokens"] for item in train_data])
    trg_vocab = Vocabulary("ja_vocab", min_freq=2, max_size=7500)
    trg_vocab.build_vocab([item["ja_tokens"] for item in train_data])

    src_vocab.save(os.path.join(checkpoint_dir, "src_vocab.pkl"))
    trg_vocab.save(os.path.join(checkpoint_dir, "trg_vocab.pkl"))

    train_loader = DataLoader(TranslationDataset(train_data, src_vocab, trg_vocab), batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(TranslationDataset(val_data, src_vocab, trg_vocab), batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    encoder = Encoder(src_vocab_size=len(src_vocab), emb_dim=emb_dim, enc_hidden_dim=enc_hidden_dim, dec_hidden_dim=dec_hidden_dim, pad_idx=PAD_IDX, dropout=0.2)
    attention = BahdanauAttention(enc_hidden_dim=enc_hidden_dim, dec_hidden_dim=dec_hidden_dim, attn_dim=128)
    decoder = Decoder(trg_vocab_size=len(trg_vocab), emb_dim=emb_dim, enc_hidden_dim=enc_hidden_dim, dec_hidden_dim=dec_hidden_dim, pad_idx=PAD_IDX, dropout=0.2, attention=attention)
    model = Seq2Seq(encoder, decoder, pad_idx=PAD_IDX, device=device).to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  [+] Trainable parameters: {total_params:,}")

    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=1)
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)

    best_val_loss = float("inf")
    history = {"epochs": [], "train_loss": [], "val_loss": [], "learning_rates": []}
    best_checkpoint_path = os.path.join(checkpoint_dir, "best_model.pt")

    print(f"STARTING TRAINING: {epochs} Epochs | Batch Size: {batch_size}")
    for epoch in range(1, epochs + 1):
        t0 = time.time()
        tf_ratio = max(0.40, 0.85 - (0.45 * (epoch - 1) / max(epochs - 1, 1)))
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, clip, tf_ratio, device)
        val_loss = evaluate_epoch(model, val_loader, criterion, device)
        scheduler.step(val_loss)

        cur_lr = optimizer.param_groups[0]['lr']
        history["epochs"].append(epoch)
        history["train_loss"].append(round(train_loss, 4))
        history["val_loss"].append(round(val_loss, 4))
        history["learning_rates"].append(cur_lr)

        print(f"Epoch {epoch:02d}/{epochs:02d} [{time.time()-t0:.1f}s] | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | LR: {cur_lr:.6f}")

        # Always ensure directory exists right before saving
        os.makedirs(checkpoint_dir, exist_ok=True)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "src_vocab": src_vocab,
                "trg_vocab": trg_vocab,
                "val_loss": val_loss,
                "train_loss": train_loss,
                "config": {"emb_dim": emb_dim, "enc_hidden_dim": enc_hidden_dim, "dec_hidden_dim": dec_hidden_dim, "total_params": total_params, "train_pairs": len(train_data)}
            }, best_checkpoint_path)
            print(f"  --> [BEST] Saved new best checkpoint to {best_checkpoint_path}")

    # Final Evaluation
    best_ckpt = torch.load(best_checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(best_ckpt["model_state_dict"])
    test_metrics = evaluate_test_set(model, test_data, src_vocab, trg_vocab, device=device)

    metrics_path = os.path.join(checkpoint_dir, "metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump({
            "test_metrics": test_metrics,
            "training_history": history,
            "model_config": {"emb_dim": emb_dim, "enc_hidden_dim": enc_hidden_dim, "dec_hidden_dim": dec_hidden_dim, "trainable_parameters": total_params, "train_size": len(train_data), "val_size": len(val_data), "test_size": len(test_data), "src_vocab_size": len(src_vocab), "trg_vocab_size": len(trg_vocab)}
        }, f, indent=2)
    print(f"[+] Saved metrics: BLEU={test_metrics['bleu']}, Test Loss={test_metrics['test_loss']}")

if __name__ == "__main__":
    train_pipeline()
