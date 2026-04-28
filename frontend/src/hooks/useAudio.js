import { useCallback, useEffect, useRef, useState } from 'react'
import * as Tone from 'tone'

const FFT_SIZE = 256

export function useAudio(audioUrl) {
  const [isPlaying, setIsPlaying] = useState(false)
  const [currentTime, setCurrentTime] = useState(0)
  const [volume, setVolume] = useState(-Infinity)
  const [isLoaded, setIsLoaded] = useState(false)

  const playerRef = useRef(null)
  const fftRef = useRef(null)
  const meterRef = useRef(null)
  const fftDataRef = useRef(new Float32Array(FFT_SIZE).fill(-100))
  const rafRef = useRef(0)
  const startTimeRef = useRef(0)
  const offsetRef = useRef(0)
  const playingRef = useRef(false)

  useEffect(() => {
    if (!audioUrl) return

    setIsLoaded(false)
    const fft = new Tone.FFT(FFT_SIZE)
    const meter = new Tone.Meter()
    const player = new Tone.Player({
      url: audioUrl,
      onload: () => setIsLoaded(true),
      onstop: () => {
        if (playingRef.current) {
          playingRef.current = false
          offsetRef.current = 0
          setIsPlaying(false)
          setCurrentTime(0)
        }
      },
    })
    player.fan(fft, meter).toDestination()

    playerRef.current = player
    fftRef.current = fft
    meterRef.current = meter
    offsetRef.current = 0
    startTimeRef.current = 0

    return () => {
      playingRef.current = false
      try { player.stop() } catch (_) {}
      player.dispose()
      fft.dispose()
      meter.dispose()
    }
  }, [audioUrl])

  useEffect(() => {
    const tick = () => {
      const fft = fftRef.current
      const meter = meterRef.current
      if (fft) {
        const values = fft.getValue()
        fftDataRef.current.set(values)
      }
      if (meter) {
        const v = meter.getValue()
        setVolume(typeof v === 'number' ? v : v[0])
      }
      if (playingRef.current) {
        setCurrentTime(Tone.now() - startTimeRef.current)
      }
      rafRef.current = requestAnimationFrame(tick)
    }
    rafRef.current = requestAnimationFrame(tick)
    return () => cancelAnimationFrame(rafRef.current)
  }, [])

  const play = useCallback(async () => {
    const player = playerRef.current
    if (!player || !player.loaded) return
    await Tone.start()
    const offset = offsetRef.current
    player.start(undefined, offset)
    startTimeRef.current = Tone.now() - offset
    playingRef.current = true
    setIsPlaying(true)
  }, [])

  const pause = useCallback(() => {
    const player = playerRef.current
    if (!player) return
    const elapsed = Tone.now() - startTimeRef.current
    offsetRef.current = elapsed
    playingRef.current = false
    player.stop()
    setIsPlaying(false)
  }, [])

  return {
    play,
    pause,
    isPlaying,
    isLoaded,
    currentTime,
    fftData: fftDataRef.current,
    volume,
  }
}
