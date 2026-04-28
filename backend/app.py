import json
import math
import tempfile
from pathlib import Path

import librosa
import numpy as np
import torch
from flask import Flask, jsonify, request
from flask_cors import CORS
from transformers import AutoModel, Wav2Vec2FeatureExtractor

MERT_MODEL_ID = "m-a-p/MERT-v1-330M"
MERT_SR = 24000
SEGMENT_SECONDS = 10

PROTOTYPE_FILE = Path(__file__).parent / "prototypes.json"
SUPPORTED_MOODS = ["happy", "sad", "angry", "calm", "neutral"]

app = Flask(__name__)
CORS(app)

# Apple Silicon (MPS) > CUDA > CPU
if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

print(f"[mert] loading {MERT_MODEL_ID} on {device}…")
mert_processor = Wav2Vec2FeatureExtractor.from_pretrained(MERT_MODEL_ID, trust_remote_code=True)
mert_model = AutoModel.from_pretrained(MERT_MODEL_ID, trust_remote_code=True).to(device)
mert_model.eval()
print("[mert] ready")


# ---------- prototype storage ----------

def load_prototypes() -> dict:
    if PROTOTYPE_FILE.exists():
        with open(PROTOTYPE_FILE) as f:
            data = json.load(f)
        return {m: data.get(m, []) for m in SUPPORTED_MOODS}
    return {m: [] for m in SUPPORTED_MOODS}


def save_prototypes(d: dict) -> None:
    with open(PROTOTYPE_FILE, "w") as f:
        json.dump(d, f)


prototypes = load_prototypes()
print("[prototypes]", {m: len(v) for m, v in prototypes.items()})


# ---------- MERT embedding ----------

def embed_audio(y_24k: np.ndarray) -> np.ndarray:
    """Run MERT and return a single 1024-d vector (mean over layers and time)."""
    inputs = mert_processor(y_24k, sampling_rate=MERT_SR, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        out = mert_model(**inputs, output_hidden_states=True)
    # hidden_states: tuple of (L+1) tensors of shape (1, T, 1024)
    layers = torch.stack(out.hidden_states, dim=0)  # (L+1, 1, T, 1024)
    pooled = layers.mean(dim=0).mean(dim=1).squeeze(0)  # (1024,)
    return pooled.cpu().numpy().astype(np.float32)


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


def classify_by_prototype(seg_emb: np.ndarray) -> tuple[str, float, dict]:
    """Compare segment embedding to each mood's mean prototype embedding."""
    sims: dict[str, float] = {}
    for mood, embs in prototypes.items():
        if not embs:
            continue
        proto_mean = np.mean(np.array(embs, dtype=np.float32), axis=0)
        sims[mood] = cosine(seg_emb, proto_mean)
    if not sims:
        return "neutral", 0.0, {}
    best = max(sims, key=sims.get)
    return best, sims[best], sims


# ---------- librosa per-second timeline ----------

def dominant_band(y_window: np.ndarray, sr: int) -> str:
    if y_window.size == 0 or not np.any(y_window):
        return "sub-bass"
    n_fft = min(2048, len(y_window))
    spectrum = np.abs(np.fft.rfft(y_window, n=n_fft))
    freqs = np.fft.rfftfreq(n_fft, d=1.0 / sr)
    bands = {
        "sub-bass": (0, 60),
        "bass": (60, 250),
        "mid": (250, 4000),
        "high": (4000, sr / 2),
    }
    energies = {
        name: float(spectrum[(freqs >= lo) & (freqs < hi)].sum())
        for name, (lo, hi) in bands.items()
    }
    return max(energies, key=energies.get)


def build_timeline(y: np.ndarray, sr: int) -> list[dict]:
    duration = librosa.get_duration(y=y, sr=sr)
    n_seconds = max(1, int(math.ceil(duration)))

    rms = librosa.feature.rms(y=y, frame_length=sr, hop_length=sr)[0]
    centroid = librosa.feature.spectral_centroid(
        y=y, sr=sr, n_fft=2048, hop_length=sr
    )[0]
    onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=sr)
    tempo_raw = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr, hop_length=sr)[0]
    tempo = float(np.atleast_1d(tempo_raw)[0])

    rms_max = float(rms.max()) if rms.size and rms.max() > 0 else 1.0
    nyquist = sr / 2.0
    onset_max = float(onset_env.max()) if onset_env.size and onset_env.max() > 0 else 1.0

    timeline = []
    for i in range(n_seconds):
        start = i * sr
        end = min(start + sr, len(y))
        window = y[start:end]
        timeline.append({
            "second": i,
            "rms_energy": float(rms[i] / rms_max) if i < len(rms) else 0.0,
            "spectral_centroid": float(min(centroid[i] / nyquist, 1.0)) if i < len(centroid) else 0.0,
            "dominant_band": dominant_band(window, sr),
            "tempo": tempo,
            "onset_strength": float(onset_env[i] / onset_max) if i < len(onset_env) else 0.0,
        })
    return timeline


# ---------- librosa fallback heuristic (used when no prototypes registered yet) ----------

def heuristic_mood(rms: float, centroid: float, tempo: float) -> str:
    if rms < 0.13:
        return "calm"
    if tempo >= 130 and rms > 0.5:
        return "happy" if centroid > 0.18 else "angry"
    if tempo <= 75 and rms < 0.3 and centroid < 0.12:
        return "sad"
    return "neutral"


# ---------- mood segments ----------

def build_mood_segments(y: np.ndarray, sr: int, timeline: list[dict]) -> list[dict]:
    if sr != MERT_SR:
        y_24k = librosa.resample(y, orig_sr=sr, target_sr=MERT_SR)
    else:
        y_24k = y

    has_prototypes = any(prototypes[m] for m in SUPPORTED_MOODS)

    seg_len = SEGMENT_SECONDS * MERT_SR
    n_segments = max(1, math.ceil(len(y_24k) / seg_len))
    segments = []
    for i in range(n_segments):
        start = i * seg_len
        end = min(start + seg_len, len(y_24k))
        chunk = y_24k[start:end]
        if chunk.size < MERT_SR // 2:
            continue

        start_s = i * SEGMENT_SECONDS
        end_s = start_s + SEGMENT_SECONDS
        seg_entries = [t for t in timeline if start_s <= t["second"] < end_s]
        avg_rms = float(np.mean([t["rms_energy"] for t in seg_entries])) if seg_entries else 0.0
        avg_centroid = float(np.mean([t["spectral_centroid"] for t in seg_entries])) if seg_entries else 0.5
        seg_tempo = float(seg_entries[0]["tempo"]) if seg_entries else 90.0

        if has_prototypes:
            emb = embed_audio(chunk)
            mood, confidence, all_sims = classify_by_prototype(emb)
            method = "prototype"
        else:
            mood = heuristic_mood(avg_rms, avg_centroid, seg_tempo)
            confidence = 0.0
            all_sims = {}
            method = "heuristic"

        segments.append({
            "start": start_s,
            "end": end_s,
            "mood": mood,
            "confidence": confidence,
            "method": method,
            "similarities": {k: round(v, 4) for k, v in all_sims.items()},
            "rms_avg": round(avg_rms, 3),
            "centroid_avg": round(avg_centroid, 3),
            "tempo": round(seg_tempo, 1),
        })
    return segments


# ---------- routes ----------

@app.route("/api/analyze", methods=["POST"])
def analyze():
    if "file" not in request.files:
        return jsonify({"error": "missing file field"}), 400
    audio = request.files["file"]

    with tempfile.NamedTemporaryFile(suffix=".audio", delete=True) as tmp:
        audio.save(tmp.name)
        y, sr = librosa.load(tmp.name, sr=None, mono=True)

    if y.size == 0:
        return jsonify({"error": "empty audio"}), 400

    timeline = build_timeline(y, sr)
    mood_segments = build_mood_segments(y, sr, timeline)

    return jsonify({
        "duration": float(librosa.get_duration(y=y, sr=sr)),
        "sample_rate": int(sr),
        "prototype_counts": {m: len(prototypes[m]) for m in SUPPORTED_MOODS},
        "timeline": timeline,
        "mood_segments": mood_segments,
    })


@app.route("/api/prototype", methods=["POST"])
def register_prototype():
    if "file" not in request.files or "mood" not in request.form:
        return jsonify({"error": "need 'file' and 'mood' fields"}), 400
    mood = request.form["mood"].lower().strip()
    if mood not in SUPPORTED_MOODS:
        return jsonify({"error": f"mood must be one of {SUPPORTED_MOODS}"}), 400

    audio = request.files["file"]
    with tempfile.NamedTemporaryFile(suffix=".audio", delete=True) as tmp:
        audio.save(tmp.name)
        y, sr = librosa.load(tmp.name, sr=MERT_SR, mono=True)

    if y.size < MERT_SR:
        return jsonify({"error": "prototype audio must be at least 1 s"}), 400

    # Cap prototype length to 30 s — only need a representative chunk
    max_len = MERT_SR * 30
    if len(y) > max_len:
        y = y[:max_len]

    emb = embed_audio(y).tolist()
    prototypes[mood].append(emb)
    save_prototypes(prototypes)

    return jsonify({
        "mood": mood,
        "filename": audio.filename,
        "added": True,
        "counts": {m: len(prototypes[m]) for m in SUPPORTED_MOODS},
    })


@app.route("/api/prototypes", methods=["GET"])
def list_prototypes():
    return jsonify({
        "counts": {m: len(prototypes[m]) for m in SUPPORTED_MOODS},
        "supported_moods": SUPPORTED_MOODS,
    })


@app.route("/api/prototypes", methods=["DELETE"])
def clear_prototypes():
    mood = request.args.get("mood")
    if mood:
        if mood not in SUPPORTED_MOODS:
            return jsonify({"error": "unknown mood"}), 400
        prototypes[mood] = []
    else:
        for m in SUPPORTED_MOODS:
            prototypes[m] = []
    save_prototypes(prototypes)
    return jsonify({"counts": {m: len(prototypes[m]) for m in SUPPORTED_MOODS}})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5050, debug=False)
