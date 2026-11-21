import os
import time
import json
import torch
from flask import Blueprint, request, jsonify, render_template, send_file
import io

from app.model.encoder import Encoder
from app.model.attention import BahdanauAttention
from app.model.decoder import Decoder
from app.model.seq2seq import Seq2Seq
from app.utils.vocab import Vocabulary, PAD_IDX
from app.inference.translator import Translator
from app.inference.srt_engine import SRTEngine

api_bp = Blueprint('api', __name__)
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

_translator = None
_srt_engine = None
_model_metadata = {
    "status": "unloaded",
    "checkpoint_path": None,
    "metrics": {}
}

def load_model_pipeline(checkpoint_path=None, metrics_path=None):
    global _translator, _srt_engine, _model_metadata
    checkpoint_path = checkpoint_path or os.path.join(PROJECT_ROOT, "models", "checkpoints", "best_model.pt")
    metrics_path = metrics_path or os.path.join(PROJECT_ROOT, "models", "checkpoints", "metrics.json")

    if not os.path.exists(checkpoint_path):
        _model_metadata["status"] = "checkpoint_not_found"
        return False

    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

        src_vocab = ckpt["src_vocab"]
        trg_vocab = ckpt["trg_vocab"]
        config = ckpt.get("config", {})

        emb_dim = config.get("emb_dim", 128)
        enc_hidden_dim = config.get("enc_hidden_dim", 256)
        dec_hidden_dim = config.get("dec_hidden_dim", 256)

        encoder = Encoder(
            src_vocab_size=len(src_vocab),
            emb_dim=emb_dim,
            enc_hidden_dim=enc_hidden_dim,
            dec_hidden_dim=dec_hidden_dim,
            pad_idx=PAD_IDX
        )
        attention = BahdanauAttention(
            enc_hidden_dim=enc_hidden_dim,
            dec_hidden_dim=dec_hidden_dim,
            attn_dim=128
        )
        decoder = Decoder(
            trg_vocab_size=len(trg_vocab),
            emb_dim=emb_dim,
            enc_hidden_dim=enc_hidden_dim,
            dec_hidden_dim=dec_hidden_dim,
            pad_idx=PAD_IDX,
            attention=attention
        )
        seq2seq = Seq2Seq(encoder, decoder, pad_idx=PAD_IDX, device=device)
        seq2seq.load_state_dict(ckpt["model_state_dict"])
        seq2seq.to(device)
        seq2seq.eval()

        _translator = Translator(seq2seq, src_vocab, trg_vocab, device=device)
        _srt_engine = SRTEngine(_translator)

        _model_metadata["status"] = "ready"
        _model_metadata["checkpoint_path"] = os.path.relpath(checkpoint_path, PROJECT_ROOT)
        _model_metadata["val_loss"] = ckpt.get("val_loss", None)
        _model_metadata["train_loss"] = ckpt.get("train_loss", None)
        _model_metadata["config"] = config
        _model_metadata["src_vocab_size"] = len(src_vocab)
        _model_metadata["trg_vocab_size"] = len(trg_vocab)

        if os.path.exists(metrics_path):
            with open(metrics_path, "r", encoding="utf-8") as f:
                _model_metadata["metrics"] = json.load(f)

        return True
    except Exception as e:
        _model_metadata["status"] = f"error: {str(e)}"
        return False


@api_bp.route("/api/health", methods=["GET"])
def health_check():
    return jsonify({
        "status": "healthy",
        "model_loaded": _translator is not None,
        "metadata": _model_metadata
    })


@api_bp.route("/api/translate", methods=["POST"])
def translate():
    global _translator
    if _translator is None:
        load_model_pipeline()
        if _translator is None:
            return jsonify({
                "error": "Model checkpoint not loaded.",
                "status": _model_metadata["status"]
            }), 503

    data = request.get_json(silent=True) or {}
    text = data.get("text", "").strip()

    if not text:
        return jsonify({"error": "Empty input text provided"}), 400

    if len(text) > 300:
        return jsonify({"error": "Input text too long (max 300 characters)"}), 400

    start_time = time.time()
    try:
        result = _translator.translate_sentence(text)
        duration_ms = round((time.time() - start_time) * 1000, 2)

        return jsonify({
            "success": True,
            "source_text": text,
            "translation": result["translation"],
            "src_tokens": result["src_tokens"],
            "trg_tokens": result["trg_tokens"],
            "attention_matrix": result["attention_matrix"],
            "latency_ms": duration_ms
        })
    except Exception as e:
        return jsonify({"error": f"Translation failed: {str(e)}"}), 500


@api_bp.route("/api/subtitles/upload", methods=["POST"])
def parse_subtitles():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    if not file.filename.lower().endswith(".srt"):
        return jsonify({"error": "Only .srt files are supported"}), 400

    try:
        content = file.read().decode("utf-8-sig", errors="replace")
        parser = SRTEngine()
        subtitles = parser.parse_srt(content)

        if not subtitles:
            return jsonify({"error": "Could not find valid SRT subtitle timing blocks in file"}), 400

        return jsonify({
            "success": True,
            "filename": file.filename,
            "count": len(subtitles),
            "subtitles": subtitles
        })
    except Exception as e:
        return jsonify({"error": f"Failed to parse SRT file: {str(e)}"}), 500


@api_bp.route("/api/subtitles/translate", methods=["POST"])
def translate_subtitles_endpoint():
    global _srt_engine, _translator
    if _srt_engine is None or _translator is None:
        load_model_pipeline()
        if _srt_engine is None:
            return jsonify({"error": "Trained model checkpoint required for subtitle translation"}), 503

    data = request.get_json(silent=True) or {}
    subtitles = data.get("subtitles", [])

    if not subtitles:
        return jsonify({"error": "No subtitles provided"}), 400

    try:
        translated = _srt_engine.translate_subtitles(subtitles)
        return jsonify({
            "success": True,
            "subtitles": translated
        })
    except Exception as e:
        return jsonify({"error": f"Failed translating subtitles: {str(e)}"}), 500


@api_bp.route("/api/subtitles/export", methods=["POST"])
def export_subtitles():
    data = request.get_json(silent=True) or {}
    subtitles = data.get("subtitles", [])

    if not subtitles:
        return jsonify({"error": "No subtitle content to export"}), 400

    try:
        engine = SRTEngine()
        srt_text = engine.export_srt(subtitles)

        mem = io.BytesIO()
        mem.write(srt_text.encode("utf-8"))
        mem.seek(0)

        return send_file(
            mem,
            mimetype="text/plain",
            as_attachment=True,
            download_name="translated_japanese.srt"
        )
    except Exception as e:
        return jsonify({"error": f"Failed to export SRT: {str(e)}"}), 500


@api_bp.route("/api/metrics", methods=["GET"])
def get_metrics():
    metrics_path = os.path.join(PROJECT_ROOT, "models", "checkpoints", "metrics.json")
    if os.path.exists(metrics_path):
        with open(metrics_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return jsonify({"success": True, "data": data})
    else:
        return jsonify({"success": False, "message": "Metrics not yet available"}), 404
