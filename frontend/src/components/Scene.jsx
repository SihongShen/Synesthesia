import { useCallback, useEffect, useMemo, useRef } from 'react'
import { useFrame, useThree } from '@react-three/fiber'
import {
  EffectComposer,
  Bloom,
  ChromaticAberration,
  Vignette,
} from '@react-three/postprocessing'
import * as THREE from 'three'
import { FrequencyRing } from './FrequencyRing'
import { ParticleField } from './ParticleField'
import { WaveformRibbon } from './WaveformRibbon'
import { CubeGrid } from './CubeGrid'

const FFT_SIZE = 256
const MIN_DB = -100
const MAX_DB = 0
const RADIUS = 1.6
const DISPLACE_AMOUNT = 0.9

// Mood transition: ~2s settle (1 - exp(-1.5 * 2) ≈ 0.95)
const MOOD_LAMBDA = 1.5

const MOOD_COLOR = {
  happy: '#ff8c1a',
  sad: '#1a6dff',
  angry: '#ff2a3a',
  calm: '#22d3ee',
  neutral: '#ffffff',
}

const dbToNorm = (db) =>
  Math.max(0, Math.min(1, (db - MIN_DB) / (MAX_DB - MIN_DB)))

const damp = (current, target, lambda, dt) =>
  current + (target - current) * (1 - Math.exp(-lambda * dt))

function currentMood(analysis, time) {
  if (!analysis?.mood_segments?.length) return 'neutral'
  const seg = analysis.mood_segments.find((s) => time >= s.start && time < s.end)
  return seg?.mood ?? analysis.mood_segments[analysis.mood_segments.length - 1].mood
}

function currentTimelineEntry(analysis, time) {
  if (!analysis?.timeline?.length) return null
  const idx = Math.min(
    analysis.timeline.length - 1,
    Math.max(0, Math.floor(time)),
  )
  return analysis.timeline[idx]
}

function GradientBackground() {
  const material = useMemo(
    () =>
      new THREE.ShaderMaterial({
        side: THREE.BackSide,
        depthWrite: false,
        depthTest: false,
        uniforms: {
          uTop: { value: new THREE.Color('#0c1430') },
          uMid: { value: new THREE.Color('#06080f') },
          uBottom: { value: new THREE.Color('#01010a') },
        },
        vertexShader: /* glsl */ `
          varying vec3 vWorldPos;
          void main() {
            vec4 wp = modelMatrix * vec4(position, 1.0);
            vWorldPos = wp.xyz;
            gl_Position = projectionMatrix * viewMatrix * wp;
          }
        `,
        fragmentShader: /* glsl */ `
          uniform vec3 uTop;
          uniform vec3 uMid;
          uniform vec3 uBottom;
          varying vec3 vWorldPos;
          void main() {
            float t = clamp((vWorldPos.y + 40.0) / 80.0, 0.0, 1.0);
            vec3 col = t < 0.5
              ? mix(uBottom, uMid, t * 2.0)
              : mix(uMid, uTop, (t - 0.5) * 2.0);
            gl_FragColor = vec4(col, 1.0);
          }
        `,
      }),
    [],
  )
  useEffect(() => () => material.dispose(), [material])
  return (
    <mesh frustumCulled={false} renderOrder={-1}>
      <sphereGeometry args={[80, 32, 16]} />
      <primitive object={material} attach="material" />
    </mesh>
  )
}

function MoodAtmosphere({ analysis, currentTime }) {
  const scene = useThree((s) => s.scene)
  const lightRef = useRef(null)
  const bloomRef = useRef(null)
  // Callback ref: function refs are silently dropped by JSON.stringify,
  // which the @react-three/postprocessing wrapper does on its props for memoization.
  // Passing a normal { current } ref makes it walk into BloomEffect's circular scene graph.
  const setBloomRef = useCallback((eff) => {
    bloomRef.current = eff
  }, [])
  // Stable Vector2 — gets stored as the ChromaticAberration uniform value,
  // so mutating it via .set() each frame updates the GPU directly.
  const chromaOffset = useMemo(() => new THREE.Vector2(0.0006, 0.0006), [])

  const moodColor = useRef(new THREE.Color(MOOD_COLOR.neutral))
  const targetColor = useRef(new THREE.Color(MOOD_COLOR.neutral))

  useFrame((_, delta) => {
    const moodK = 1 - Math.exp(-MOOD_LAMBDA * delta)

    const mood = currentMood(analysis, currentTime)
    targetColor.current.set(MOOD_COLOR[mood] ?? MOOD_COLOR.neutral)
    moodColor.current.lerp(targetColor.current, moodK)

    const entry = currentTimelineEntry(analysis, currentTime)
    const rms = entry?.rms_energy ?? 0
    const onset = entry?.onset_strength ?? 0

    if (lightRef.current) {
      lightRef.current.color.copy(moodColor.current)
      lightRef.current.intensity = damp(
        lightRef.current.intensity,
        0.4 + rms * 4.5,
        6,
        delta,
      )
    }

    if (scene.fog) {
      // High RMS → low density (clearer); low RMS → dreamy fog
      const targetDensity = 0.02 + (1 - rms) * 0.06
      scene.fog.density = damp(scene.fog.density, targetDensity, 4, delta)
      // Fog color follows mood subtly with the same 2s envelope
      scene.fog.color.lerp(moodColor.current, moodK * 0.35)
    }

    if (bloomRef.current) {
      bloomRef.current.intensity = damp(
        bloomRef.current.intensity,
        0.35 + rms * 1.4,
        5,
        delta,
      )
    }

    const chromaTarget = 0.0006 + onset * 0.004
    const chromaNext = damp(chromaOffset.x, chromaTarget, 7, delta)
    chromaOffset.set(chromaNext, chromaNext)
  })

  return (
    <>
      <ambientLight intensity={0.18} color="#5a6a8a" />
      <ambientLight intensity={0.12} color="#8a5a4a" />
      <pointLight
        ref={lightRef}
        position={[0, 0, 0]}
        intensity={0.6}
        color={MOOD_COLOR.neutral}
        distance={22}
        decay={2}
      />
      <EffectComposer>
        <Bloom
          ref={setBloomRef}
          intensity={0.5}
          luminanceThreshold={0.15}
          luminanceSmoothing={0.7}
          mipmapBlur
        />
        <ChromaticAberration offset={chromaOffset} />
        <Vignette eskil={false} offset={0.32} darkness={0.75} />
      </EffectComposer>
    </>
  )
}

export function Scene({ fftData, analysis, currentTime, volume, preset = 'organic' }) {
  const showIco = preset === 'organic'
  const showParticles = preset === 'organic'
  const showCubeGrid = preset === 'geometric'
  const showRing = preset === 'geometric'
  const showRibbon = preset === 'minimal'
  const meshRef = useRef(null)

  const { geometry, originals, bins } = useMemo(() => {
    const geo = new THREE.IcosahedronGeometry(RADIUS, 4)
    const posAttr = geo.attributes.position
    const originals = new Float32Array(posAttr.array)
    const bins = new Uint16Array(posAttr.count)
    for (let i = 0; i < posAttr.count; i++) {
      const ox = originals[i * 3]
      const oy = originals[i * 3 + 1]
      const oz = originals[i * 3 + 2]
      const len = Math.sqrt(ox * ox + oy * oy + oz * oz) || 1
      const u = (Math.atan2(oz, ox) + Math.PI) / (2 * Math.PI)
      const v = Math.acos(Math.min(1, Math.max(-1, oy / len))) / Math.PI
      bins[i] = (Math.floor(u * 16) * 16 + Math.floor(v * 16)) % FFT_SIZE
    }
    return { geometry: geo, originals, bins }
  }, [])

  const material = useMemo(
    () =>
      new THREE.MeshPhysicalMaterial({
        color: MOOD_COLOR.neutral,
        emissive: MOOD_COLOR.neutral,
        emissiveIntensity: 0,
        roughness: 0.5,
        metalness: 0,
        transmission: 0.3,
        thickness: 1.2,
        ior: 1.4,
        clearcoat: 0.5,
        clearcoatRoughness: 0.2,
      }),
    [],
  )

  useEffect(() => {
    return () => {
      geometry.dispose()
      material.dispose()
    }
  }, [geometry, material])

  const targetColor = useMemo(() => new THREE.Color(MOOD_COLOR.neutral), [])

  useFrame((_, delta) => {
    const mesh = meshRef.current
    if (!mesh) return

    const posAttr = geometry.attributes.position
    const arr = posAttr.array

    for (let i = 0; i < posAttr.count; i++) {
      const ox = originals[i * 3]
      const oy = originals[i * 3 + 1]
      const oz = originals[i * 3 + 2]
      const len = Math.sqrt(ox * ox + oy * oy + oz * oz) || 1
      const nx = ox / len
      const ny = oy / len
      const nz = oz / len
      const amp = dbToNorm(fftData[bins[i]] ?? MIN_DB)
      const d = amp * DISPLACE_AMOUNT
      arr[i * 3] = ox + nx * d
      arr[i * 3 + 1] = oy + ny * d
      arr[i * 3 + 2] = oz + nz * d
    }
    posAttr.needsUpdate = true
    geometry.computeVertexNormals()

    const entry = currentTimelineEntry(analysis, currentTime)
    const rms = entry?.rms_energy ?? 0
    const centroid = entry?.spectral_centroid ?? 0.3

    material.emissiveIntensity = damp(
      material.emissiveIntensity,
      rms * 1.6,
      8,
      delta,
    )
    material.roughness = damp(
      material.roughness,
      Math.max(0.04, 1 - centroid),
      5,
      delta,
    )

    const moodK = 1 - Math.exp(-MOOD_LAMBDA * delta)
    const mood = currentMood(analysis, currentTime)
    targetColor.set(MOOD_COLOR[mood] ?? MOOD_COLOR.neutral)
    material.color.lerp(targetColor, moodK)
    material.emissive.lerp(targetColor, moodK)

    const tempo = entry?.tempo ?? analysis?.timeline?.[0]?.tempo ?? 90
    const rotSpeed = (tempo / 120) * 0.18
    mesh.rotation.y += rotSpeed * delta
    mesh.rotation.x += rotSpeed * 0.4 * delta
  })

  return (
    <>
      <GradientBackground />
      <fogExp2 attach="fog" args={['#0a1024', 0.05]} />

      {showIco && <mesh ref={meshRef} geometry={geometry} material={material} />}
      {showRing && <FrequencyRing fftData={fftData} />}
      {showParticles && (
        <ParticleField
          fftData={fftData}
          analysis={analysis}
          currentTime={currentTime}
        />
      )}
      {showCubeGrid && <CubeGrid fftData={fftData} />}
      {showRibbon && <WaveformRibbon fftData={fftData} volume={volume} />}

      <MoodAtmosphere analysis={analysis} currentTime={currentTime} />
    </>
  )
}
