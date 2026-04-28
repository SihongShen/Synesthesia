import { useEffect, useMemo } from 'react'
import { useFrame } from '@react-three/fiber'
import * as THREE from 'three'

const SAMPLES = 256
const TURNS = 2.5
const BASE_R = 5
const HEIGHT = 6
const FFT_SIZE = 256
const MIN_DB = -100
const MAX_DB = 0
const VOL_MIN_DB = -60
const VOL_MAX_DB = 0

export function WaveformRibbon({ fftData, volume }) {
  const geometry = useMemo(() => {
    const geo = new THREE.BufferGeometry()
    const positions = new Float32Array(SAMPLES * 2 * 3)
    const indices = []
    for (let i = 0; i < SAMPLES - 1; i++) {
      const a = i * 2
      const b = i * 2 + 1
      const c = (i + 1) * 2
      const d = (i + 1) * 2 + 1
      indices.push(a, c, b, b, c, d)
    }
    geo.setIndex(indices)
    geo.setAttribute('position', new THREE.BufferAttribute(positions, 3))
    return geo
  }, [])

  const material = useMemo(
    () =>
      new THREE.MeshStandardMaterial({
        color: '#dde7ff',
        emissive: '#3a5e8a',
        emissiveIntensity: 0.6,
        roughness: 0.35,
        metalness: 0.2,
        side: THREE.DoubleSide,
        transparent: true,
        opacity: 0.85,
      }),
    [],
  )

  useEffect(() => {
    return () => {
      geometry.dispose()
      material.dispose()
    }
  }, [geometry, material])

  useFrame(() => {
    const arr = geometry.attributes.position.array
    const volNorm = Math.max(
      0,
      Math.min(1, (volume - VOL_MIN_DB) / (VOL_MAX_DB - VOL_MIN_DB)),
    )
    const halfThickness = 0.05 + volNorm * 0.55

    for (let i = 0; i < SAMPLES; i++) {
      const t = i / (SAMPLES - 1)
      const angle = t * TURNS * Math.PI * 2
      const fftIdx = Math.floor(t * FFT_SIZE)
      const db = fftData[fftIdx] ?? MIN_DB
      const amp = Math.max(0, Math.min(1, (db - MIN_DB) / (MAX_DB - MIN_DB)))
      const r = BASE_R + amp * 0.9
      const yCenter = (t - 0.5) * HEIGHT

      const cx = Math.cos(angle) * r
      const cz = Math.sin(angle) * r

      arr[i * 6] = cx
      arr[i * 6 + 1] = yCenter + halfThickness
      arr[i * 6 + 2] = cz
      arr[i * 6 + 3] = cx
      arr[i * 6 + 4] = yCenter - halfThickness
      arr[i * 6 + 5] = cz
    }
    geometry.attributes.position.needsUpdate = true
    geometry.computeVertexNormals()
  })

  return <mesh geometry={geometry} material={material} />
}
