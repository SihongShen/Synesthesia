import { useEffect, useMemo, useRef } from 'react'
import { useFrame } from '@react-three/fiber'
import * as THREE from 'three'

const COUNT = 128
const RADIUS = 3
const FFT_SIZE = 256
const MIN_DB = -100
const MAX_DB = 0
const MAX_HEIGHT = 4

export function FrequencyRing({ fftData }) {
  const meshRef = useRef(null)
  const dummy = useMemo(() => new THREE.Object3D(), [])

  const angles = useMemo(() => {
    const a = new Float32Array(COUNT)
    for (let i = 0; i < COUNT; i++) a[i] = (i / COUNT) * Math.PI * 2
    return a
  }, [])

  useEffect(() => {
    const inst = meshRef.current
    if (!inst) return
    const c = new THREE.Color()
    for (let i = 0; i < COUNT; i++) {
      const t = i / (COUNT - 1)
      // 0.0=red, 0.33=green, 0.67=blue
      c.setHSL(t * 0.67, 0.9, 0.55)
      inst.setColorAt(i, c)

      dummy.position.set(
        Math.cos(angles[i]) * RADIUS,
        0,
        Math.sin(angles[i]) * RADIUS,
      )
      dummy.scale.set(1, 0.05, 1)
      dummy.updateMatrix()
      inst.setMatrixAt(i, dummy.matrix)
    }
    inst.instanceMatrix.needsUpdate = true
    if (inst.instanceColor) inst.instanceColor.needsUpdate = true
  }, [angles, dummy])

  useFrame(() => {
    const inst = meshRef.current
    if (!inst) return
    for (let i = 0; i < COUNT; i++) {
      const fftIdx = Math.floor((i / COUNT) * FFT_SIZE)
      const db = fftData[fftIdx] ?? MIN_DB
      const amp = Math.max(0, Math.min(1, (db - MIN_DB) / (MAX_DB - MIN_DB)))
      const h = 0.05 + amp * MAX_HEIGHT
      dummy.position.set(
        Math.cos(angles[i]) * RADIUS,
        h / 2,
        Math.sin(angles[i]) * RADIUS,
      )
      dummy.scale.set(1, h, 1)
      dummy.updateMatrix()
      inst.setMatrixAt(i, dummy.matrix)
    }
    inst.instanceMatrix.needsUpdate = true
  })

  return (
    <instancedMesh ref={meshRef} args={[undefined, undefined, COUNT]}>
      <boxGeometry args={[0.06, 1, 0.06]} />
      <meshStandardMaterial
        color="#ffffff"
        emissive="#222222"
        roughness={0.4}
        metalness={0.1}
      />
    </instancedMesh>
  )
}
