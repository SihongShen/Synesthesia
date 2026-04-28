# Music Vis — Audio-Driven Real-Time 3D Visualization

A web app that turns an uploaded audio file into a real-time 3D scene in the browser. The backend runs offline analysis with Flask + librosa + MERT (mood classification is a cosine-similarity lookup against user-uploaded prototype clips). The frontend uses Tone.js for live FFT and Three.js (`@react-three/fiber`) for rendering — the whole atmosphere reacts to the music's energy, spectrum, and mood.

---

## Stack

**Backend** (Python)
- Flask 3 + flask-cors
- librosa 0.10 — time/frequency-domain analysis
- transformers + torch — MERT (`m-a-p/MERT-v1-330M`) for 1024-dim audio embeddings
- soundfile — audio decoding
- Device priority: Apple Silicon MPS > CUDA > CPU

**Frontend** (Vite + React 19)
- @react-three/fiber 9 + three.js — 3D rendering
- @react-three/postprocessing — post effects (Bloom / ChromaticAberration / Vignette)
- tone.js 15 — Web Audio playback + FFT(256) + Meter
- Plain CSS for the glassmorphism overlay

---

## Features

### 1. Backend audio analysis API

Four endpoints:

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/api/analyze` | Upload audio, get back timeline + mood segments |
| `POST` | `/api/prototype` | Upload a reference clip, register it as a prototype for one mood |
| `GET`  | `/api/prototypes` | Show how many prototypes are registered per mood |
| `DELETE` | `/api/prototypes` | Clear all prototypes, or `?mood=happy` to clear just one |

#### `POST /api/analyze`

Accepts a multipart upload (`form field = file`) and returns JSON shaped like:

```json
{
  "duration": 123.45,
  "sample_rate": 44100,
  "prototype_counts": { "happy": 2, "sad": 1, "angry": 0, "calm": 0, "neutral": 0 },
  "timeline": [
    {
      "second": 0,
      "rms_energy": 0.42,         // 0-1, normalized to the track's max
      "spectral_centroid": 0.31,  // 0-1, normalized to Nyquist
      "dominant_band": "bass",    // sub-bass / bass / mid / high
      "tempo": 128.0,             // global BPM
      "onset_strength": 0.18      // 0-1, normalized to the track's max
    }
  ],
  "mood_segments": [
    {
      "start": 0,
      "end": 10,
      "mood": "calm",             // happy / sad / angry / calm / neutral
      "confidence": 0.83,         // best-match cosine similarity in prototype mode; always 0 in heuristic mode
      "method": "prototype",      // "prototype" or "heuristic"
      "similarities": { "happy": 0.41, "sad": 0.27, "calm": 0.83 },
      "rms_avg": 0.18,
      "centroid_avg": 0.11,
      "tempo": 92.0
    }
  ]
}
```

Implementation notes:
- **One timeline entry per second**: `librosa.feature.rms`, `spectral_centroid`, and `onset_strength` are all called with `hop_length=sr` so each frame lines up with exactly one second.
- **dominant_band** is computed from a per-second FFT, summing energy in four bands (<60 Hz / 60–250 / 250–4000 / >4000) and picking the largest.
- **tempo** is the whole-track estimate from `librosa.beat.beat_track` — not a per-second value.
- **mood_segments — prototype mode**: audio is resampled to 24 kHz (MERT's training rate), split into 10-second chunks, and each chunk passes through MERT to produce a 1024-dim embedding (averaged across all hidden layers and time steps). The embedding is compared via cosine similarity to each mood's mean prototype embedding; the highest-scoring mood wins, and the similarity itself is the `confidence`.
- **mood_segments — heuristic fallback**: if `prototypes.json` has no prototypes registered yet (cold start), the backend falls back to a coarse rule-based classifier on `(rms, centroid, tempo)`. This keeps the UI from being stuck on `neutral`, but quality is limited — register at least 1–2 reference clips per mood for anything decent.
- MERT is loaded once at startup onto GPU/MPS/CPU.
- CORS is wide-open so the frontend can `fetch` directly.

#### `POST /api/prototype`

Multipart upload, fields:
- `file` — an audio clip (5–30 s recommended; anything past 30 s is truncated)
- `mood` — one of `happy` / `sad` / `angry` / `calm` / `neutral`

The backend computes the MERT embedding and appends it to that mood's list in `prototypes.json`. Multiple clips per mood are averaged together; more clips → more stable.

### 2. Frontend audio pipeline — `useAudio` hook

Wraps Tone.js playback and analysis. Exposes:

| Field | Type | Notes |
|---|---|---|
| `play()` | `() => Promise<void>` | Starts the AudioContext and plays |
| `pause()` | `() => void` | Stops after recording the offset; next `play()` resumes from there |
| `isPlaying` | `boolean` | |
| `isLoaded` | `boolean` | Whether the Player has finished decoding |
| `currentTime` | `number` | Seconds, updated every frame |
| `fftData` | `Float32Array(256)` | Same reference, mutated in place each frame (dB values) |
| `volume` | `number` | dB, from `Tone.Meter` |

Internals:
- `Tone.Player.fan(fft, meter).toDestination()` — one source feeds FFT, Meter, and the speakers in parallel.
- A single `requestAnimationFrame` loop: FFT values are `set()` into the same Float32Array (consumers read it in `useFrame` without triggering React re-renders), and `currentTime` / `volume` state is updated.
- Pause/resume is implemented by computing `Tone.now() - startTime` as the offset, then `start(undefined, offset)`.
- The FFT buffer is initialized with `.fill(-100)` so silent pre-playback frames don't get misread as "all bands at full volume."

### 3. The 3D scene

#### Center object — Icosahedron Breather
- `IcosahedronGeometry(1.6, detail=4)`, ~960 vertices.
- Each frame, every vertex is pushed along its radial normal by `amp * 0.9`, where `amp` comes from that vertex's mapped FFT bin.
- Bin mapping uses spherical coordinates: `(⌊u·16⌋·16 + ⌊v·16⌋) % 256`, ensuring shared vertices on adjacent triangles get the same bin (no tearing).
- `computeVertexNormals()` runs every frame so lighting follows the deformation.
- Material is `MeshPhysicalMaterial`:
  - `color` / `emissive` lerp to the current mood's target color (`λ=1.5`, ~2 s transition)
  - `emissiveIntensity` follows `rms_energy`
  - `roughness = max(0.04, 1 - spectral_centroid)` (brighter sound → smoother surface)
  - `transmission: 0.3`, plus `ior 1.4 + clearcoat 0.5` for a glassy look
- Rotation speed = `(tempo / 120) * 0.18` — higher BPM spins faster.

#### Three surrounding decoration layers (rendered selectively per preset)

| Layer | Implementation | Driven by |
|---|---|---|
| **FrequencyRing** | `<instancedMesh>` × 128, thin upright bars on a circle of radius 3 | Height = corresponding FFT bin; each bar has a fixed HSL color (red→green→blue, `hue = i/128 * 0.67`) written via `setColorAt` into instanceColor |
| **ParticleField** | 500 `<points>` with a custom ShaderMaterial (additive blending, circular sprite) | Orbit radius `baseR + rms*0.9`; sprite size from a mapped FFT bin; alpha is solid at high energy, sparse when quiet |
| **WaveformRibbon** | 256-segment triangle strip forming a 2.5-turn spiral around the center | Radius modulated by FFT; the gap between the upper and lower edges (the ribbon's thickness) is driven by `volume` (Meter dB) |
| **CubeGrid** | `<instancedMesh>` 16×16 = 256 cubes as a floor | Each cube maps to one FFT bin, height `0.05 + amp*3.5`; color is a radial gradient from the center |

#### Atmosphere & post

The `MoodAtmosphere` subcomponent owns:

- **Background** — an inside-out giant sphere shell (radius 80) with a custom ShaderMaterial doing a 3-stop vertical gradient `#01010a → #06080f → #0c1430`. `renderOrder=-1`, doesn't write depth. The shader excludes the fog chunk so the background isn't dimmed by fog.
- **Fog** — `fogExp2`, with `density` lerped each frame toward `0.02 + (1-rms)*0.06` (clearer at high energy, dreamier when quiet); fog color is mixed slightly (35%) toward the mood color.
- **Lights**:
  - Center point light: color = mood color, intensity = `0.4 + rms*4.5`, `distance=22, decay=2`
  - Two low-intensity ambient lights with cool/warm tints (0.18 / 0.12) for color bias
- **Post effects** (`<EffectComposer>`):
  - `Bloom` — `mipmapBlur`, intensity = `0.35 + rms*1.4`
  - `ChromaticAberration` — resting at 0.0006; per-second `onset_strength` pushes the offset to `+0.004` then decays quickly
  - `Vignette` — static `offset=0.32, darkness=0.75`

#### Mood transitions (~2 s smoothing)
All "color/atmosphere" parameters share the same exponential damping `1 - exp(-1.5 * dt)`, reaching ~95% in 2 s. Affects:
- Center object's `color` & `emissive`
- Center point light color
- Fog color
- The mood pill in the UI (matched with CSS `transition: 0.6s`)

Parameters tied to "energy/rhythm" (emissiveIntensity, roughness, light intensity, fog density, bloom intensity) use a stronger damping coefficient (`λ=4–8`) so they remain visibly reactive on a beat-by-beat scale.

### 4. UI overlay

A glassmorphism panel pinned to the bottom-left, containing:

- **Drag-and-drop upload zone** — a `<label>` wrapping a hidden `<input type=file>`, supporting both click and drag (`onDragOver/onDrop`).
- **Play/pause button** — a circular button, disabled until audio is loaded and analyzed.
- **Progress bar** — current time / total duration with a gradient fill.
- **Mood pill** — the current 10-second segment's mood, tinted per mood (happy=orange, sad=blue, angry=red, calm=cyan, neutral=gray).
- **Preset switcher** — a 3-way toggle:
  - **Organic**: icosahedron + particles
  - **Geometric**: cube grid + frequency ring
  - **Minimal**: just the spiral ribbon
  - Switching is instant; the atmosphere (lights/fog/post) carries over.
- **Live metrics** — Energy / Centroid bars, updated by the internal rAF loop directly via `transform: scaleX()` on the DOM nodes (no React state, no re-render cost).

Styling: `backdrop-filter: blur(14px) saturate(1.2)` + `rgba(15,17,26,0.55)` background + 1 px translucent white border + a large shadow.

---

## Project layout

```
Synesthesia/
├── README.md                       # this file
├── .gitignore
├── backend/
│   ├── app.py                      # Flask + librosa + MERT analysis service
│   ├── requirements.txt
│   ├── prototypes.json             # user-uploaded mood prototype embeddings (gitignored)
│   └── venv/                       # Python 3.10 virtualenv
└── frontend/
    ├── package.json                # React 19, r3f 9, three, tone, postprocessing
    ├── vite.config.js
    ├── index.html
    └── src/
        ├── main.jsx
        ├── App.jsx                 # top-level component + UI overlay + MetricsBars
        ├── App.css                 # glassmorphism styles
        ├── index.css               # reset + global background
        ├── hooks/
        │   └── useAudio.js         # Tone.Player + FFT(256) + Meter
        └── components/
            ├── Scene.jsx           # scene composition, center ico, MoodAtmosphere
            ├── FrequencyRing.jsx   # 128 instanced frequency bars
            ├── ParticleField.jsx   # 500 particles (ShaderMaterial)
            ├── WaveformRibbon.jsx  # spiral waveform ribbon
            └── CubeGrid.jsx        # 16×16 cube floor (Geometric preset)
```

---

## Install & run

### Backend

```bash
cd backend
# Activate the venv:
#   macOS/Linux:    source venv/bin/activate
#   Git Bash:       source venv/Scripts/activate
#   PowerShell:     venv\Scripts\Activate.ps1
#   CMD:            venv\Scripts\activate.bat
pip install -r requirements.txt    # CPU build is enough to run; see GPU section below
python app.py                      # listens on :5050
```

**GPU (optional, strongly recommended)**: `requirements.txt` installs the CPU build of torch by default. To use CUDA, install a torch wheel matching your driver per https://pytorch.org first — e.g. for RTX 40-series + driver ≥ 552:

```bash
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt   # the rest installs as usual; pip skips torch since it's already satisfied
```

If you see `[mert] loading m-a-p/MERT-v1-330M on cuda…` at startup, GPU is in use. Apple Silicon picks MPS automatically.

The first startup downloads MERT weights (~1.3 GB) into `~/.cache/huggingface/`. If your network is slow, set `export HF_ENDPOINT=https://hf-mirror.com` (PowerShell: `$env:HF_ENDPOINT="https://hf-mirror.com"`) before running. Subsequent startups read from the local cache and are near-instant.

### Frontend

```bash
cd frontend
npm install
npm run dev                        # Vite default port :5173
```

Open http://localhost:5173, drag in an audio file, wait for "Analyzing on server…" to disappear, then hit ▶.

---

## Data flow

```
Audio File
    │
    ├── (multipart upload) ──► Flask /api/analyze
    │                              │
    │                              ├── librosa: per-second timeline
    │                              └── MERT embedding → cosine vs prototypes (per 10 s)
    │                              │
    │                          JSON ◄── analysis state
    │
    └── (URL.createObjectURL) ──► Tone.Player ──► fan(FFT, Meter, Destination)
                                                   │       │
                                                   ▼       ▼
                                          fftData (Float32Array, RAF) │ volume (state)
                                                   │
                                                   ▼
                                  Scene (r3f) — useFrame reads fftData
                                                   │
                                          icosahedron / ring / particles / ribbon / cubes
                                                   │
                                          MoodAtmosphere — fog / lights / post-FX
```

---

## Known limitations & trade-offs

- **`tempo` is a single global value** — every row of the timeline repeats the same number. It's the whole-track estimate from `librosa.beat.beat_track`, not a per-second value.
- **Mood quality depends on prototypes**: MERT produces general-purpose audio embeddings with no built-in emotion labels. With no prototypes registered (cold start), the backend falls back to a heuristic on rms/centroid/tempo only — limited results. Register at least 1–2 representative clips per mood for decent classification.
- **Prototypes are simply averaged**: multiple clips per mood are averaged in embedding space before cosine comparison. If a single mood spans drastically different styles (e.g. "happy" with both folk and EDM clips), the mean vector blurs. For more reliable results, keep prototypes within a mood stylistically consistent.
- **`onset_strength` is per-second**, so the ChromaticAberration "punch" is a second-level spike, not a true beat-by-beat reaction. Real per-beat response would need a live onset detector on the frontend.
- **`transmission: 0.3`** doesn't refract very visibly without an environment map — you mostly see slight internal translucency. For a more obvious glass look, add `<Environment preset="city" />` (drei).
- **First request is slow on the backend**: MERT loads into memory at import time, and the first `librosa.load` still has to decode the full clip.
- **The progress bar isn't draggable yet**: useAudio doesn't expose `seek()` — left as a follow-up.
