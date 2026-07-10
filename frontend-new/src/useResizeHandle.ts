import { useCallback, useRef } from 'react'

interface Options {
  width: number
  setWidth: (w: number) => void
  min: number
  max: number
  defaultWidth: number
  /** +1 if dragging right grows the panel (handle on its right edge), -1 if dragging right shrinks it (handle on its left edge) */
  direction: 1 | -1
}

/** Pointer-driven width drag: pointerdown on the handle, pointermove/pointerup on window so the drag survives the cursor leaving the handle. */
export function useResizeHandle({ width, setWidth, min, max, defaultWidth, direction }: Options) {
  const startX = useRef(0)
  const startWidth = useRef(0)

  const handlePointerDown = useCallback(
    (e: React.PointerEvent) => {
      e.preventDefault()
      startX.current = e.clientX
      startWidth.current = width
      document.body.style.cursor = 'col-resize'
      document.body.style.userSelect = 'none'

      const handleMove = (ev: PointerEvent) => {
        const delta = (ev.clientX - startX.current) * direction
        const next = Math.min(max, Math.max(min, startWidth.current + delta))
        setWidth(next)
      }

      const handleUp = () => {
        document.body.style.cursor = ''
        document.body.style.userSelect = ''
        window.removeEventListener('pointermove', handleMove)
        window.removeEventListener('pointerup', handleUp)
      }

      window.addEventListener('pointermove', handleMove)
      window.addEventListener('pointerup', handleUp)
    },
    [width, min, max, direction, setWidth],
  )

  const handleDoubleClick = useCallback(() => {
    setWidth(defaultWidth)
  }, [defaultWidth, setWidth])

  return { handlePointerDown, handleDoubleClick }
}
