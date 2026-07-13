import { useEffect, useMemo, useCallback, useRef, useState } from 'react'
import {
  ReactFlow,
  Background,
  Controls,
  useNodesState,
  Handle,
  Position,
  BackgroundVariant,
} from '@xyflow/react'
import type { Node, Edge, NodeProps, ReactFlowInstance } from '@xyflow/react'
import '@xyflow/react/dist/style.css'

import { RISK_COLOR, type RiskLabel } from '../risk'
import type { MergedComponent } from '../api'

// ── Layout constants ─────────────────────────────────────────────────────────

const NODE_W = 188
const NODE_H = 34   // compact — 50 short nodes fit; 50 tall ones don't
const V_GAP = 6
// tier 3 (raw/logistics) leftmost → tier 1 (critical) rightmost
const TIER_X: Record<number, number> = { 3: 0, 2: NODE_W + 220, 1: (NODE_W + 220) * 2 }
const EDGE_FADE_MS = 150

// ── Custom node (module-level for stable reference) ───────────────────────────

function ComponentNode({ data, selected }: NodeProps) {
  const c = data.component as MergedComponent
  return (
    <>
      <Handle
        type="target"
        position={Position.Left}
        style={{ opacity: 0, width: 4, height: 4 }}
      />
      <div
        className={`flex items-center gap-1.5 px-2 bg-zinc-900 border rounded-sm ${
          selected ? 'border-zinc-600' : 'border-zinc-800 hover:border-zinc-600'
        }`}
        style={{
          width: NODE_W,
          height: NODE_H,
          opacity: data.dimmed ? 0.25 : 1,
          transition: 'border-color 100ms, opacity 150ms',
        }}
      >
        <div
          className="w-2 h-2 rounded-full flex-shrink-0"
          style={{ backgroundColor: RISK_COLOR[c.risk_label as RiskLabel] }}
        />
        <div className="min-w-0 flex-1">
          <div className="text-[11px] text-zinc-100 truncate leading-none mb-0.5">{c.name}</div>
          <div className="text-[10px] text-zinc-400 truncate">{c.country}</div>
        </div>
      </div>
      <Handle
        type="source"
        position={Position.Right}
        style={{ opacity: 0, width: 4, height: 4 }}
      />
    </>
  )
}

const nodeTypes = { componentNode: ComponentNode }

// ── Layout computation ────────────────────────────────────────────────────────

interface GraphAssets {
  baseNodes: Node[]
  baseEdges: Edge[]
  /** adjacency: componentId → set of all directly connected ids */
  adjacency: Map<string, Set<string>>
  /** depMap: componentId → its direct dependency ids (upstream suppliers only) */
  depMap: Map<string, string[]>
}

function buildGraph(components: MergedComponent[]): GraphAssets {
  // Deduplicate
  const seen = new Map<string, MergedComponent>()
  for (const c of components) {
    if (!seen.has(c.id)) seen.set(c.id, c)
  }
  const uniq = [...seen.values()]

  // Group by tier, sort by risk_score descending within each tier
  const byTier = new Map<number, MergedComponent[]>()
  for (const c of uniq) {
    const bucket = byTier.get(c.tier) ?? []
    bucket.push(c)
    byTier.set(c.tier, bucket)
  }

  const baseNodes: Node[] = []
  for (const [tier, comps] of byTier.entries()) {
    const x = TIER_X[tier] ?? tier * (NODE_W + 220)
    const sorted = [...comps].sort((a, b) => b.risk_score - a.risk_score)
    const totalH = sorted.length * (NODE_H + V_GAP) - V_GAP
    sorted.forEach((c, i) => {
      baseNodes.push({
        id: c.id,
        type: 'componentNode',
        position: { x, y: i * (NODE_H + V_GAP) - totalH / 2 },
        data: { component: c, dimmed: false },
        width: NODE_W,
        height: NODE_H,
      })
    })
  }

  const idSet = new Set(baseNodes.map((n) => n.id))
  const edgeSet = new Set<string>()
  const baseEdges: Edge[] = []
  const adjacency = new Map<string, Set<string>>()

  for (const c of uniq) {
    for (const depId of c.dependencies ?? []) {
      const edgeId = `${depId}=>${c.id}`
      if (edgeSet.has(edgeId)) continue
      edgeSet.add(edgeId)
      if (!idSet.has(depId)) {
        console.warn(`Graph: dependency ${depId} referenced by ${c.id} not found`)
        continue
      }
      // source = supplier (lower tier, left); target = consumer (higher tier, right)
      baseEdges.push({
        id: edgeId,
        source: depId,
        target: c.id,
        type: 'smoothstep',
        style: { stroke: '#3f3f46', strokeWidth: 1 }, // zinc-700 default
        animated: false,
      })
      // Record adjacency both ways for highlight lookup
      const fromSet = adjacency.get(depId) ?? new Set()
      fromSet.add(c.id)
      adjacency.set(depId, fromSet)
      const toSet = adjacency.get(c.id) ?? new Set()
      toSet.add(depId)
      adjacency.set(c.id, toSet)
    }
  }

  // Upstream-only dependency map (each node → the parts it depends on), used to
  // trace the multi-hop supply chain the same way the globe does.
  const depMap = new Map<string, string[]>()
  for (const c of uniq) {
    depMap.set(c.id, (c.dependencies ?? []).filter((id) => idSet.has(id)))
  }

  return { baseNodes, baseEdges, adjacency, depMap }
}

// ── Component ─────────────────────────────────────────────────────────────────

interface Props {
  components: MergedComponent[]
  selectedId: string | null
  onSelect: (id: string) => void
  highlightedIds: Set<string>
}

export default function Graph({ components, selectedId, onSelect, highlightedIds }: Props) {
  const [nodes, setNodes, onNodesChange] = useNodesState<Node>([])

  const { baseNodes, baseEdges, depMap } = useMemo(
    () => buildGraph(components),
    [components],
  )

  // Upstream supply chain from the selected component, two hops deep — mirrors
  // the globe (direct dependencies, then THEIR dependencies). `edges` maps each
  // edge id to its hop: 1 = direct supply, 2 = upstream supply.
  const chain = useMemo(() => {
    if (!selectedId) return null
    const nodes = new Set<string>([selectedId])
    const edges = new Map<string, number>()
    let frontier = [selectedId]
    for (let hop = 1; hop <= 2; hop++) {
      const next: string[] = []
      for (const id of frontier) {
        for (const dep of depMap.get(id) ?? []) {
          edges.set(`${dep}=>${id}`, hop)
          if (!nodes.has(dep)) {
            nodes.add(dep)
            next.push(dep)
          }
        }
      }
      frontier = next
    }
    return { nodes, edges }
  }, [selectedId, depMap])

  // Initialize on first load
  useEffect(() => {
    if (baseNodes.length === 0) return
    setNodes(baseNodes)
  }, [baseNodes, setNodes])

  // Apply dimming: selection wins over highlight; highlight wins over nothing
  useEffect(() => {
    if (baseNodes.length === 0) return

    if (selectedId && chain) {
      setNodes(
        baseNodes.map((n) => ({
          ...n,
          selected: n.id === selectedId,
          data: { ...n.data, dimmed: !chain.nodes.has(n.id) },
        })),
      )
      return
    }

    if (highlightedIds.size > 0) {
      setNodes(
        baseNodes.map((n) => ({
          ...n,
          selected: false,
          data: { ...n.data, dimmed: !highlightedIds.has(n.id) },
        })),
      )
      return
    }

    // Nothing selected, nothing highlighted: full reset
    setNodes(baseNodes)
  }, [selectedId, highlightedIds, baseNodes, chain, setNodes])

  // Only edges touching the selected node are ever rendered. They fade in on
  // selection and fade out on deselect, so a just-hidden edge stays mounted
  // (opacity animating to 0) for EDGE_FADE_MS before it's actually removed.
  const [edges, setEdges] = useState<Edge[]>([])
  const fadeOutTimer = useRef<ReturnType<typeof setTimeout>>()

  useEffect(() => {
    if (fadeOutTimer.current) {
      clearTimeout(fadeOutTimer.current)
      fadeOutTimer.current = undefined
    }

    if (!selectedId) {
      // Fade whatever is currently shown out, then unmount it
      setEdges((prev) => prev.map((e) => ({ ...e, style: { ...e.style, opacity: 0 } })))
      fadeOutTimer.current = setTimeout(() => setEdges([]), EDGE_FADE_MS)
      return
    }

    // Render every edge in the two-hop upstream chain. Hop 1 (direct supply)
    // is solid and bright; hop 2 (upstream supply) is dashed and faint — the
    // same visual hierarchy as the globe's two arc layers.
    const chainEdges = chain?.edges ?? new Map<string, number>()
    const touching = baseEdges
      .filter((e) => chainEdges.has(e.id))
      .map((e) => {
        const direct = chainEdges.get(e.id) === 1
        return {
          ...e,
          style: {
            stroke: '#a1a1aa', // zinc-400
            strokeWidth: direct ? 2 : 1.5,
            strokeDasharray: direct ? undefined : '4 3',
            opacity: 0,
            transition: `opacity ${EDGE_FADE_MS}ms`,
          },
        }
      })

    // Mount at opacity 0, then flip to the target on the next frame so the
    // transition animates instead of snapping in.
    setEdges(touching)
    const raf = requestAnimationFrame(() => {
      setEdges((prev) =>
        prev.map((e) => ({
          ...e,
          style: { ...e.style, opacity: chainEdges.get(e.id) === 1 ? 1 : 0.55 },
        })),
      )
    })
    return () => cancelAnimationFrame(raf)
  }, [selectedId, baseEdges, chain])

  useEffect(() => {
    return () => {
      if (fadeOutTimer.current) clearTimeout(fadeOutTimer.current)
    }
  }, [])

  const handleNodeClick = useCallback(
    (_: React.MouseEvent, node: Node) => onSelect(node.id),
    [onSelect],
  )

  // Re-fit whenever the graph's own container is resized (e.g. a side panel
  // being dragged), debounced so a drag doesn't spam fitView on every pixel.
  const wrapperRef = useRef<HTMLDivElement>(null)
  const rfInstance = useRef<ReactFlowInstance | null>(null)
  const resizeSettleTimer = useRef<ReturnType<typeof setTimeout>>()

  useEffect(() => {
    const el = wrapperRef.current
    if (!el) return
    const observer = new ResizeObserver(() => {
      if (resizeSettleTimer.current) clearTimeout(resizeSettleTimer.current)
      resizeSettleTimer.current = setTimeout(() => {
        rfInstance.current?.fitView({ padding: 0.1 })
      }, 150)
    })
    observer.observe(el)
    return () => {
      observer.disconnect()
      if (resizeSettleTimer.current) clearTimeout(resizeSettleTimer.current)
    }
  }, [])

  return (
    <div
      ref={wrapperRef}
      className="flex-1 h-full w-full"
      style={{ minHeight: '40vh', background: '#09090b' }}
    >
      <ReactFlow
        nodes={nodes}
        edges={edges}
        onNodesChange={onNodesChange}
        nodeTypes={nodeTypes}
        onNodeClick={handleNodeClick}
        onInit={(instance) => {
          rfInstance.current = instance
        }}
        fitView
        fitViewOptions={{ padding: 0.1 }}
        minZoom={0.15}
        maxZoom={2}
        proOptions={{ hideAttribution: true }}
        nodesDraggable={true}
        nodesConnectable={false}
        elementsSelectable={true}
      >
        <Background
          variant={BackgroundVariant.Dots}
          gap={24}
          size={1}
          color="#27272a"
        />
        <Controls
          showInteractive={false}
          style={{ background: '#18181b', borderColor: '#27272a', color: '#71717a' }}
        />
      </ReactFlow>
    </div>
  )
}
