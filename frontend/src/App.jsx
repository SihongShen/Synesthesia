import { useEffect, useRef, useState } from 'react'
import { Canvas } from '@react-three/fiber'
import { OrbitControls } from '@react-three/drei'
import { Scene } from './components/Scene'
import { useAudio } from './hooks/useAudio'
import './App.css'

const API_BASE = 'http://localhost:5050/api'
const ANALYZE_URL = `${API_BASE}/analyze`
const PROTOTYPE_URL = `${API_BASE}/prototype`
const PROTOTYPES_URL = `${API_BASE}/prototypes`

const TRAIN_MOODS = ['happy', 'sad', 'angry', 'calm', 'neutral']

const PRESETS = [
  { id: 'organic', label: 'Organic' },
  { id: 'geometric', label: 'Geometric' },
  { id: 'minimal', label: 'Minimal' },
]

const MOOD_LABEL = {
  happy: 'Happy',
  sad: 'Sad',
  angry: 'Angry',
  calm: 'Calm',
  neutral: 'Neutral',
}

function currentMood(analysis, time) {
  if (!analysis?.mood_segments?.length) return null
  const seg = analysis.mood_segments.find((s) => time >= s.start && time < s.end)
  return seg?.mood ?? analysis.mood_segments[analysis.mood_segments.length - 1].mood
}

function fmtTime(s) {
  if (!Number.isFinite(s) || s < 0) s = 0
  const m = Math.floor(s / 60)
  const ss = Math.floor(s % 60).toString().padStart(2, '0')
  return `${m}:${ss}`
}

function MetricsBars({ analysis, currentTime }) {
  const energyRef = useRef(null)
  const centroidRef = useRef(null)
  const targetRef = useRef({ rms: 0, centroid: 0 })
  const stateRef = useRef({ rms: 0, centroid: 0 })

  useEffect(() => {
    const tlen = analysis?.timeline?.length ?? 0
    if (!tlen) {
      targetRef.current = { rms: 0, centroid: 0 }
      return
    }
    const idx = Math.min(tlen - 1, Math.max(0, Math.floor(currentTime)))
    const e = analysis.timeline[idx]
    targetRef.current = {
      rms: e.rms_energy ?? 0,
      centroid: e.spectral_centroid ?? 0,
    }
  }, [analysis, currentTime])

  useEffect(() => {
    let prev = performance.now()
    let rafId = 0
    const tick = (now) => {
      const dt = Math.min(0.05, (now - prev) / 1000)
      prev = now
      const k = 1 - Math.exp(-6 * dt)
      stateRef.current.rms +=
        (targetRef.current.rms - stateRef.current.rms) * k
      stateRef.current.centroid +=
        (targetRef.current.centroid - stateRef.current.centroid) * k
      if (energyRef.current)
        energyRef.current.style.transform = `scaleX(${stateRef.current.rms.toFixed(3)})`
      if (centroidRef.current)
        centroidRef.current.style.transform = `scaleX(${stateRef.current.centroid.toFixed(3)})`
      rafId = requestAnimationFrame(tick)
    }
    rafId = requestAnimationFrame(tick)
    return () => cancelAnimationFrame(rafId)
  }, [])

  return (
    <div className="metrics">
      <div className="metric">
        <span className="metric-label">Energy</span>
        <div className="metric-track">
          <div className="metric-fill" ref={energyRef} />
        </div>
      </div>
      <div className="metric">
        <span className="metric-label">Centroid</span>
        <div className="metric-track">
          <div className="metric-fill metric-fill-cool" ref={centroidRef} />
        </div>
      </div>
    </div>
  )
}

export default function App() {
  const [audioUrl, setAudioUrl] = useState(null)
  const [analysis, setAnalysis] = useState(null)
  const [status, setStatus] = useState('idle')
  const [error, setError] = useState(null)
  const [preset, setPreset] = useState('organic')
  const [dragOver, setDragOver] = useState(false)
  const [fileName, setFileName] = useState(null)
  const [protoMood, setProtoMood] = useState('happy')
  const [protoCounts, setProtoCounts] = useState({})
  const [protoStatus, setProtoStatus] = useState('idle')

  useEffect(() => {
    fetch(PROTOTYPES_URL)
      .then((r) => r.json())
      .then((d) => setProtoCounts(d.counts ?? {}))
      .catch(() => {})
  }, [])

  const handleProtoUpload = async (file) => {
    if (!file) return
    setProtoStatus('uploading')
    try {
      const form = new FormData()
      form.append('file', file)
      form.append('mood', protoMood)
      const res = await fetch(PROTOTYPE_URL, { method: 'POST', body: form })
      if (!res.ok) throw new Error(`server ${res.status}`)
      const data = await res.json()
      setProtoCounts(data.counts ?? {})
      setProtoStatus('done')
      setTimeout(() => setProtoStatus('idle'), 1500)
    } catch (err) {
      console.error(err)
      setProtoStatus('error')
    }
  }

  const handleProtoClear = async () => {
    if (!confirm(`Clear all prototypes for "${protoMood}"?`)) return
    const res = await fetch(`${PROTOTYPES_URL}?mood=${protoMood}`, { method: 'DELETE' })
    const data = await res.json()
    setProtoCounts(data.counts ?? {})
  }

  const { play, pause, isPlaying, isLoaded, currentTime, fftData, volume } =
    useAudio(audioUrl)

  const handleFile = async (file) => {
    if (!file) return
    setError(null)
    setStatus('uploading')
    setFileName(file.name)
    try {
      const form = new FormData()
      form.append('file', file)
      const res = await fetch(ANALYZE_URL, { method: 'POST', body: form })
      if (!res.ok) throw new Error(`server ${res.status}`)
      const data = await res.json()
      console.group(`[analyze] ${file.name}`)
      console.log('full response:', data)
      console.log('duration:', data.duration, 's,  timeline length:', data.timeline?.length)
      console.log('tempo (BPM):', data.timeline?.[0]?.tempo)
      console.table(
        data.mood_segments?.map((s) => ({
          start: s.start,
          end: s.end,
          mood: s.mood,
          raw_label: s.raw_label,
          confidence: +s.confidence.toFixed(3),
          rms_avg: s.rms_avg,
          centroid_avg: s.centroid_avg,
          tempo: s.tempo,
        })),
      )
      console.groupEnd()
      setAnalysis(data)
      if (audioUrl) URL.revokeObjectURL(audioUrl)
      setAudioUrl(URL.createObjectURL(file))
      setStatus('ready')
    } catch (err) {
      setError(err.message ?? String(err))
      setStatus('error')
    }
  }

  const duration = analysis?.duration ?? 0
  const progress = duration ? Math.min(1, currentTime / duration) : 0
  const mood = analysis ? currentMood(analysis, currentTime) : null

  const dropMessage = (() => {
    if (status === 'uploading') return 'Analyzing on server…'
    if (status === 'error') return `Error: ${error}`
    if (fileName) return fileName
    return 'Drop audio here or click to upload'
  })()

  return (
    <>
      <Canvas
        className="canvas"
        camera={{ position: [0, 6, 18], fov: 55 }}
        dpr={[1, 2]}
      >
        <Scene
          fftData={fftData}
          analysis={analysis}
          currentTime={currentTime}
          volume={volume}
          preset={preset}
        />
        <OrbitControls
          enablePan={false}
          enableDamping
          dampingFactor={0.08}
          rotateSpeed={0.6}
          zoomSpeed={0.7}
          minDistance={5}
          maxDistance={50}
        />
      </Canvas>

      <div className="overlay">
        <div className="row top-row">
          <span className={`mood-pill mood-${mood ?? 'idle'}`}>
            {mood ? MOOD_LABEL[mood] : 'No audio'}
          </span>
          <div className="presets" role="tablist">
            {PRESETS.map((p) => (
              <button
                key={p.id}
                role="tab"
                aria-selected={preset === p.id}
                className={`preset-btn ${preset === p.id ? 'active' : ''}`}
                onClick={() => setPreset(p.id)}
              >
                {p.label}
              </button>
            ))}
          </div>
        </div>

        <label
          className={`drop ${dragOver ? 'over' : ''} ${
            status === 'error' ? 'err' : ''
          }`}
          onDragOver={(e) => {
            e.preventDefault()
            setDragOver(true)
          }}
          onDragLeave={() => setDragOver(false)}
          onDrop={(e) => {
            e.preventDefault()
            setDragOver(false)
            handleFile(e.dataTransfer.files?.[0])
          }}
        >
          <input
            type="file"
            accept="audio/*"
            onChange={(e) => handleFile(e.target.files?.[0])}
          />
          <span className="drop-label">{dropMessage}</span>
        </label>

        <div className="row playback">
          <button
            className="play"
            disabled={!isLoaded || !analysis}
            onClick={isPlaying ? pause : play}
            aria-label={isPlaying ? 'Pause' : 'Play'}
          >
            {isPlaying ? '❚❚' : '▶'}
          </button>
          <div className="progress">
            <div className="progress-track">
              <div
                className="progress-fill"
                style={{ width: `${progress * 100}%` }}
              />
            </div>
            <span className="time">
              {fmtTime(currentTime)} / {fmtTime(duration)}
            </span>
          </div>
        </div>

        <MetricsBars analysis={analysis} currentTime={currentTime} />

        <div className="proto-panel">
          <div className="proto-header">
            <span className="proto-title">Train mood</span>
            <button
              className="proto-clear"
              onClick={handleProtoClear}
              disabled={!protoCounts[protoMood]}
              title={`Clear all prototypes for "${protoMood}"`}
            >
              Clear
            </button>
          </div>
          <div className="proto-row">
            <select
              className="proto-select"
              value={protoMood}
              onChange={(e) => setProtoMood(e.target.value)}
            >
              {TRAIN_MOODS.map((m) => (
                <option key={m} value={m}>
                  {m} ({protoCounts[m] ?? 0})
                </option>
              ))}
            </select>
            <label className="proto-drop">
              <input
                type="file"
                accept="audio/*"
                onChange={(e) => handleProtoUpload(e.target.files?.[0])}
              />
              <span>
                {protoStatus === 'uploading' && 'Embedding…'}
                {protoStatus === 'done' && 'Added ✓'}
                {protoStatus === 'error' && 'Error'}
                {protoStatus === 'idle' && `+ Add ${protoMood} example`}
              </span>
            </label>
          </div>
        </div>
      </div>
    </>
  )
}
