# Music Vis — 音频驱动的 3D 实时可视化

一个把上传的音频在浏览器里跑成实时 3D 场景的 web app。后端用 Flask + librosa + HuBERT 做离线分析，前端用 Tone.js 做实时 FFT，Three.js (`@react-three/fiber`) 渲染场景，整体氛围会跟着音乐的能量、频谱、情绪变化。

---

## 技术栈

**后端**（Python）
- Flask 3 + flask-cors
- librosa 0.10 — 时域/频域分析
- transformers + torch — HuBERT 情绪分类模型
- soundfile — 音频解码

**前端**（Vite + React 19）
- @react-three/fiber 9 + three.js — 3D 渲染
- @react-three/postprocessing — 后期效果（Bloom / ChromaticAberration / Vignette）
- tone.js 15 — Web Audio 播放 + FFT(256) + Meter
- 纯 CSS 实现 glassmorphism 覆盖层

---

## 已实现的功能

### 1. 后端音频分析 API

`POST /api/analyze`，接受 multipart 音频上传（`form field = file`），返回如下结构的 JSON：

```json
{
  "duration": 123.45,
  "sample_rate": 44100,
  "timeline": [
    {
      "second": 0,
      "rms_energy": 0.42,         // 0-1，按整段最大值归一化
      "spectral_centroid": 0.31,  // 0-1，按 Nyquist 归一化
      "dominant_band": "bass",    // sub-bass / bass / mid / high
      "tempo": 128.0,             // 全局 BPM
      "onset_strength": 0.18      // 0-1，按整段最大值归一化
    }
  ],
  "mood_segments": [
    {
      "start": 0,
      "end": 10,
      "mood": "calm",             // happy / sad / angry / calm / neutral
      "raw_label": "neu",         // 模型原始标签 (neu/hap/ang/sad)
      "confidence": 0.74
    }
  ]
}
```

实现要点：
- **每秒一个 timeline entry**：`librosa.feature.rms`、`spectral_centroid`、`onset_strength` 的 `hop_length=sr` 让每帧正好对应 1 秒。
- **dominant_band** 通过 FFT 累加四个频段（<60 Hz / 60–250 / 250–4000 / >4000）的能量后取最大。
- **tempo** 是全曲的 `librosa.beat.beat_track`（不是逐秒动态）。
- **mood_segments**：把 16 kHz 重采样后的音频切成 10 秒块，喂给 `superb/hubert-large-superb-er` 分类。模型只输出 `neu/hap/ang/sad`，所以 `neu + 平均 RMS < 0.2` 会被映射成 `calm`，其它按字面映射到中文要求的五分类。
- 启动时模型加载一次，CUDA 可用时走 GPU。
- CORS 全开，前端可以直接 `fetch`。

### 2. 前端音频管线 — `useAudio` hook

封装 Tone.js 的播放与分析，对外暴露：

| 字段 | 类型 | 说明 |
|---|---|---|
| `play()` | `() => Promise<void>` | 启动 AudioContext 后播放 |
| `pause()` | `() => void` | 记录 offset 后停止，下次 `play` 从 offset 继续 |
| `isPlaying` | `boolean` | |
| `isLoaded` | `boolean` | Player 是否完成解码 |
| `currentTime` | `number` | 秒，每帧更新 |
| `fftData` | `Float32Array(256)` | 同一个引用，每帧原地更新（dB 值） |
| `volume` | `number` | dB，由 `Tone.Meter` 给出 |

内部结构：
- `Tone.Player.fan(fft, meter).toDestination()` —— 一个源同时连到 FFT、Meter 和扬声器。
- 单一 `requestAnimationFrame` 循环：把 FFT 值 `set()` 到同一个 Float32Array（消费者可以在 `useFrame` 里读，不引发 React re-render），同时更新 `currentTime`、`volume` 状态。
- 暂停/恢复用 `Tone.now() - startTime` 计算的 offset，再 `start(undefined, offset)`。
- 初始 FFT buffer `.fill(-100)`，避免播放前的零值被误读成"全频段满音量"。

### 3. 三维场景

#### 中心物体 — Icosahedron Breather
- `IcosahedronGeometry(1.6, detail=4)`，约 960 个顶点。
- 每帧把每个顶点沿其径向法线推出去 `amp * 0.9`，其中 `amp` 来自该顶点对应的 FFT bin。
- bin 映射通过球面坐标：`(⌊u·16⌋·16 + ⌊v·16⌋) % 256`，保证三角面共享顶点拿到同一个 bin、不撕裂。
- 每帧重算 `computeVertexNormals()`，让光照跟随形变。
- 材质 `MeshPhysicalMaterial`：
  - `color` / `emissive` 跟随当前 mood lerp 到目标色（`λ=1.5` 大约 2 秒过渡）
  - `emissiveIntensity` 跟随 `rms_energy`
  - `roughness = max(0.04, 1 - spectral_centroid)`（声音越亮越光滑）
  - `transmission: 0.3`，加 `ior 1.4 + clearcoat 0.5` 的玻璃感
- 旋转速度 = `(tempo / 120) * 0.18`，BPM 越高转得越快。

#### 三层环绕装饰（按 preset 选择性渲染）

| 层 | 实现 | 驱动 |
|---|---|---|
| **FrequencyRing** | `<instancedMesh>` × 128，半径 3 的圆周上垂直排列细方柱 | 高度=对应 FFT bin；每根固定 HSL（红→绿→蓝，`hue = i/128 * 0.67`）通过 `setColorAt` 写入 instanceColor |
| **ParticleField** | 500 个 `<points>`，自定义 ShaderMaterial（加法混合、圆形 sprite） | 轨道半径 `baseR + rms*0.9`；尺寸来自映射的 FFT bin；alpha 在高能量时实，安静时稀疏 |
| **WaveformRibbon** | 256 段三角带，绕中心绕 2.5 圈的螺旋 | 半径受 FFT 调制；上下两条边的距离（厚度）由 `volume` (Meter dB) 控制 |
| **CubeGrid** | `<instancedMesh>` 16×16 = 256 个立方体地板 | 每个 cube 直接对应一个 FFT bin，高度 `0.05 + amp*3.5`；颜色按到中心距离做径向渐变 |

#### 大气与后期

`MoodAtmosphere` 子组件统一管理：

- **背景** — 一个反向渲染的大球壳（半径 80），自定义 ShaderMaterial 做三段式垂直渐变 `#01010a → #06080f → #0c1430`。`renderOrder=-1` 且不写深度。Shader 不包含 fog chunk，所以背景不被雾影响。
- **雾** — `fogExp2`，`density` 每帧 lerp 到 `0.02 + (1-rms)*0.06`（高能量更清晰，安静更梦幻）；雾色按 mood 颜色微弱混入（35% 强度）。
- **灯光**：
  - 中心点光源：颜色 = mood 颜色，强度 = `0.4 + rms*4.5`，`distance=22, decay=2`
  - 两个偏冷/偏暖的低强度 ambient light（0.18 / 0.12）做色偏
- **后期效果**（`<EffectComposer>`）：
  - `Bloom` — `mipmapBlur`，强度 = `0.35 + rms*1.4`
  - `ChromaticAberration` — 静止 0.0006，每秒的 `onset_strength` 把 offset 推到 `+0.004`，再快速衰减
  - `Vignette` — 静态 `offset=0.32, darkness=0.75`

#### 心情过渡（2 秒平滑）
所有"颜色/氛围"类参数都用统一的指数阻尼 `1 - exp(-1.5 * dt)`，2 秒到 95%。统一影响：
- 中心物体的 `color` & `emissive`
- 中心点光源颜色
- 雾色
- UI 上的 mood pill（CSS `transition: 0.6s` 配合）

跟"能量/节奏"耦合的参数（emissiveIntensity, roughness, light intensity, fog density, bloom intensity）阻尼系数更大（`λ=4–8`），保持随节拍可见地起伏。

### 4. UI 覆盖层

固定在左下角的玻璃拟态面板，包含：

- **拖拽上传区** — `<label>` 包裹隐藏的 `<input type=file>`，同时支持点击和拖拽（`onDragOver/onDrop`）。
- **播放/暂停按钮** — 圆形按钮，未加载或未分析时禁用。
- **进度条** — 当前时间 / 总时长 + 渐变填充条。
- **Mood pill** — 当前 10 秒段的情绪，按 mood 切换不同色调（happy=橙、sad=蓝、angry=红、calm=青、neutral=灰）。
- **Preset 切换** — 三段式按钮组：
  - **Organic**：icosahedron + 粒子
  - **Geometric**：立方体网格 + 频率环
  - **Minimal**：只有螺旋波形带
  - 切换瞬时生效，但氛围（灯光/雾/后期）一直保留。
- **实时 metrics 面板** — Energy / Centroid 双横条，由内部 rAF 循环直接 `transform: scaleX()` DOM 节点（不走 React state，无 re-render 开销）。

样式：`backdrop-filter: blur(14px) saturate(1.2)` + `rgba(15,17,26,0.55)` 背景 + 1px 半透白边 + 大阴影。

---

## 项目结构

```
Synesthesia/
├── README.md                       # 本文件
├── .gitignore
├── backend/
│   ├── app.py                      # Flask + librosa + MERT 分析服务
│   ├── requirements.txt
│   ├── prototypes.json             # 用户训练的 mood prototype embeddings (gitignored)
│   └── venv/                       # Python 3.12 虚拟环境
└── frontend/
    ├── package.json                # React 19, r3f 9, three, tone, postprocessing
    ├── vite.config.js
    ├── index.html
    └── src/
        ├── main.jsx
        ├── App.jsx                 # 顶层组件 + UI 覆盖层 + MetricsBars
        ├── App.css                 # Glassmorphism 样式
        ├── index.css               # Reset 与全局背景
        ├── hooks/
        │   └── useAudio.js         # Tone.Player + FFT(256) + Meter
        └── components/
            ├── Scene.jsx           # 场景组合、中心 ico、MoodAtmosphere
            ├── FrequencyRing.jsx   # 128 instanced 频率柱
            ├── ParticleField.jsx   # 500 粒子（ShaderMaterial）
            ├── WaveformRibbon.jsx  # 螺旋波形带
            └── CubeGrid.jsx        # 16×16 立方体地板（Geometric preset）
```

---

## 安装与运行

### 后端

```bash
cd backend
source venv/bin/activate           # 已有 Python 3.12 venv
pip install -r requirements.txt    # 第一次 ~2GB（torch + transformers + HuBERT 模型）
python app.py                      # 监听 :5050
```

第一次启动会下载 `superb/hubert-large-superb-er` 权重（~1.2 GB）到 `~/.cache/huggingface/`。后续启动只走本地缓存，秒开。

### 前端

```bash
cd frontend
npm install
npm run dev                        # Vite 默认 :5173
```

打开 http://localhost:5173，拖入音频文件，等"Analyzing on server…"消失后点 ▶。

---

## 数据流向（一图）

```
Audio File
    │
    ├── (multipart upload) ──► Flask /api/analyze
    │                              │
    │                              ├── librosa: per-second timeline
    │                              └── HuBERT: per-10s mood
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

## 已知边界与权衡

- **`tempo` 是全局值**，timeline 里每行都重复同一个数；这是 `librosa.beat.beat_track` 给的全曲估计，不是逐秒动态。
- **"calm" 是启发式**：HuBERT 模型本身只有 4 类情绪，calm 由 `neutral + 平均 RMS < 0.2` 推断。
- **`onset_strength` 是每秒一个**，所以 ChromaticAberration 的"打击感"是秒级的尖峰，不是真正逐节拍。要真节拍级响应需要在前端跑实时 onset 检测。
- **`transmission: 0.3`** 在没有 environment map 的场景里折射不太明显，更多是体现在内部一点透光感。如果需要更明显的玻璃质感，可以加 `<Environment preset="city" />`（drei）。
- **后端首请求慢**：HuBERT 模型导入即加载到内存，第一次还要等 `librosa.load` 解码长音频。
- **进度条目前不可拖动**：useAudio 没暴露 `seek()`，留给后续。
```
