import { useEffect, useMemo, useRef } from 'react'
import { useFrame } from '@react-three/fiber'
import * as THREE from 'three'

const COUNT = 500
const FFT_SIZE = 256
const MIN_DB = -100
const MAX_DB = 0

const vertexShader = /* glsl */ `
  attribute float aSize;
  attribute float aAlpha;
  varying float vAlpha;
  void main() {
    vAlpha = aAlpha;
    vec4 mvPosition = modelViewMatrix * vec4(position, 1.0);
    gl_PointSize = aSize * (320.0 / -mvPosition.z);
    gl_Position = projectionMatrix * mvPosition;
  }
`

const fragmentShader = /* glsl */ `
  uniform vec3 uColor;
  varying float vAlpha;
  void main() {
    vec2 c = gl_PointCoord - vec2(0.5);
    float d = length(c);
    if (d > 0.5) discard;
    float falloff = smoothstep(0.5, 0.0, d);
    gl_FragColor = vec4(uColor, falloff * vAlpha);
  }
`

export function ParticleField({ fftData, analysis, currentTime }) {
  const pointsRef = useRef(null)

  const { geometry, base } = useMemo(() => {
    const positions = new Float32Array(COUNT * 3)
    const sizes = new Float32Array(COUNT)
    const alphas = new Float32Array(COUNT)
    const baseAngles = new Float32Array(COUNT)
    const baseRadii = new Float32Array(COUNT)
    const baseY = new Float32Array(COUNT)
    const speeds = new Float32Array(COUNT)
    const fftIndices = new Uint16Array(COUNT)

    for (let i = 0; i < COUNT; i++) {
      baseAngles[i] = Math.random() * Math.PI * 2
      baseRadii[i] = 2.6 + Math.random() * 2.4
      baseY[i] = (Math.random() - 0.5) * 4.5
      speeds[i] = 0.15 + Math.random() * 0.5
      fftIndices[i] = Math.floor(Math.random() * FFT_SIZE)
      sizes[i] = 0.1
      alphas[i] = 0
    }

    const geo = new THREE.BufferGeometry()
    geo.setAttribute('position', new THREE.BufferAttribute(positions, 3))
    geo.setAttribute('aSize', new THREE.BufferAttribute(sizes, 1))
    geo.setAttribute('aAlpha', new THREE.BufferAttribute(alphas, 1))

    return {
      geometry: geo,
      base: {
        positions,
        sizes,
        alphas,
        baseAngles,
        baseRadii,
        baseY,
        speeds,
        fftIndices,
      },
    }
  }, [])

  const material = useMemo(
    () =>
      new THREE.ShaderMaterial({
        uniforms: { uColor: { value: new THREE.Color('#ffffff') } },
        vertexShader,
        fragmentShader,
        transparent: true,
        depthWrite: false,
        blending: THREE.AdditiveBlending,
      }),
    [],
  )

  useEffect(() => {
    return () => {
      geometry.dispose()
      material.dispose()
    }
  }, [geometry, material])

  useFrame((state) => {
    const t = state.clock.getElapsedTime()
    const positions = geometry.attributes.position.array
    const sizes = geometry.attributes.aSize.array
    const alphas = geometry.attributes.aAlpha.array

    const tlen = analysis?.timeline?.length ?? 0
    const idx = tlen ? Math.min(tlen - 1, Math.max(0, Math.floor(currentTime))) : -1
    const rms = idx >= 0 ? analysis.timeline[idx].rms_energy ?? 0 : 0

    const radiusPulse = rms * 0.9
    const baseAlpha = 0.15 + rms * 0.85

    for (let i = 0; i < COUNT; i++) {
      const angle = base.baseAngles[i] + t * base.speeds[i]
      const r = base.baseRadii[i] + radiusPulse
      positions[i * 3] = Math.cos(angle) * r
      positions[i * 3 + 1] = base.baseY[i]
      positions[i * 3 + 2] = Math.sin(angle) * r

      const db = fftData[base.fftIndices[i]] ?? MIN_DB
      const amp = Math.max(0, Math.min(1, (db - MIN_DB) / (MAX_DB - MIN_DB)))
      sizes[i] = 0.04 + amp * 0.55
      alphas[i] = Math.min(1, baseAlpha * (0.5 + amp))
    }

    geometry.attributes.position.needsUpdate = true
    geometry.attributes.aSize.needsUpdate = true
    geometry.attributes.aAlpha.needsUpdate = true
  })

  return <points ref={pointsRef} geometry={geometry} material={material} />
}
