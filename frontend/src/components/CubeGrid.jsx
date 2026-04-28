import { useEffect, useMemo, useRef } from 'react'
import { useFrame } from '@react-three/fiber'
import * as THREE from 'three'

const COUNT_X = 16
const COUNT_Z = 16
const COUNT = COUNT_X * COUNT_Z
const SPACING = 0.42
const CUBE_SIZE = 0.22
const FFT_SIZE = 256
const MIN_DB = -100
const MAX_DB = 0
const MAX_HEIGHT = 3.5
const FLOOR_Y = -2.4

export function CubeGrid({ fftData }) {
  const meshRef = useRef(null)
  const dummy = useMemo(() => new THREE.Object3D(), [])

  const positions = useMemo(() => {
    const arr = new Float32Array(COUNT * 2)
    for (let i = 0; i < COUNT_X; i++) {
      for (let j = 0; j < COUNT_Z; j++) {
        const idx = i * COUNT_Z + j
        arr[idx * 2] = (i - (COUNT_X - 1) / 2) * SPACING
        arr[idx * 2 + 1] = (j - (COUNT_Z - 1) / 2) * SPACING
      }
    }
    return arr
  }, [])

  useEffect(() => {
    const inst = meshRef.current
    if (!inst) return
    const c = new THREE.Color()
    const halfExtent = (COUNT_X - 1) * SPACING * 0.5
    for (let n = 0; n < COUNT; n++) {
      const x = positions[n * 2]
      const z = positions[n * 2 + 1]
      const dist = Math.min(1, Math.sqrt(x * x + z * z) / halfExtent)
      // inner = warm violet, outer = cool cyan
      c.setHSL(0.72 - dist * 0.26, 0.7, 0.55)
      inst.setColorAt(n, c)

      dummy.position.set(x, FLOOR_Y, z)
      dummy.scale.set(1, 0.05, 1)
      dummy.updateMatrix()
      inst.setMatrixAt(n, dummy.matrix)
    }
    inst.instanceMatrix.needsUpdate = true
    if (inst.instanceColor) inst.instanceColor.needsUpdate = true
  }, [positions, dummy])

  useFrame(() => {
    const inst = meshRef.current
    if (!inst) return
    for (let n = 0; n < COUNT; n++) {
      const fftIdx = n % FFT_SIZE
      const db = fftData[fftIdx] ?? MIN_DB
      const amp = Math.max(0, Math.min(1, (db - MIN_DB) / (MAX_DB - MIN_DB)))
      const h = 0.05 + amp * MAX_HEIGHT
      dummy.position.set(positions[n * 2], FLOOR_Y + h / 2, positions[n * 2 + 1])
      dummy.scale.set(1, h, 1)
      dummy.updateMatrix()
      inst.setMatrixAt(n, dummy.matrix)
    }
    inst.instanceMatrix.needsUpdate = true
  })

  return (
    <instancedMesh ref={meshRef} args={[undefined, undefined, COUNT]}>
      <boxGeometry args={[CUBE_SIZE, 1, CUBE_SIZE]} />
      <meshStandardMaterial
        color="#ffffff"
        emissive="#1a1f2e"
        roughness={0.35}
        metalness={0.25}
      />
    </instancedMesh>
  )
}
