import { useEffect, useMemo, useCallback } from 'react'
import {
  ReactFlow,
  Background,
  Controls,
  useNodesState,
  useEdgesState,
  Handle,
  Position,
  BackgroundVariant,
} from '@xyflow/react'
import type { Node, Edge, NodeProps } from '@xyflow/react'
import '@xyflow/react/dist/style.css'

import { RISK_COLOR, type RiskLabel } from '../risk'
import type { MergedComponent } from '../api'

// ── Layout constants ─────────────────────────────────────────────────────────

const NODE_W = 188
const NODE_H = 34   // compact — 50 short nodes fit; 50 tall ones don't
const V_GAP = 6
// tier 3 (raw/logistics) leftmost → tier 1 (critical) rightmost
const TIER_X: Record<number, number> = { 3: 0, 2: NODE_W + 120, 1: (NODE_W + 120) * 2 }

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
        className="flex items-center gap-1.5 px-2 bg-zinc-900 border border-zinc-800 rounded-sm"
        style={{
          width: NODE_W,
          height: NODE_H,
          opacity: data.dimmed ? 0.25 : 1,
          borderColor: selected ? '#52525b' : '#27272a', // zinc-600 when selected
        }}
      >
        <div
          className="w-2 h-2 rounded-full flex-shrink-0"
          style={{ backgroundColor: RISK_COLOR[c.risk_label as RiskLabel] }}
        />
        <div className="min-w-0 flex-1">
          <div className="text-[11px] text-zinc-300 truncate leading-none mb-0.5">{c.name}</div>
          <div className="text-[9px] text-zinc-500 truncate">{c.country}</div>
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
    const x = TIER_X[tier] ?? tier * (NODE_W + 120)
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

  return { baseNodes, baseEdges, adjacency }
}

// ── Component ─────────────────────────────────────────────────────────────────

interface Props {
  components: MergedComponent[]
  selectedId: string | null
  onSelect: (id: string) => void
}

export default function Graph({ components, selectedId, onSelect }: Props) {
  const [nodes, setNodes, onNodesChange] = useNodesState<Node>([])
  const [edges, setEdges, onEdgesChange] = useEdgesState<Edge>([])

  const { baseNodes, baseEdges, adjacency } = useMemo(
    () => buildGraph(components),
    [components],
  )

  // Initialize on first load
  useEffect(() => {
    if (baseNodes.length === 0) return
    setNodes(baseNodes)
    setEdges(baseEdges)
  }, [baseNodes, baseEdges, setNodes, setEdges])

  // Apply dimming when selection changes
  useEffect(() => {
    if (baseNodes.length === 0) return
    if (!selectedId) {
      // No selection: reset all to full opacity, all edges to default
      setNodes(baseNodes)
      setEdges(baseEdges)
      return
    }
    const connected = adjacency.get(selectedId) ?? new Set<string>()
    const relevant = new Set([selectedId, ...connected])

    setNodes(
      baseNodes.map((n) => ({
        ...n,
        selected: n.id === selectedId,
        data: { ...n.data, dimmed: !relevant.has(n.id) },
      })),
    )
    setEdges(
      baseEdges.map((e) => {
        const touches = e.source === selectedId || e.target === selectedId
        return {
          ...e,
          style: {
            stroke: touches ? '#71717a' : '#18181b', // zinc-500 highlight, zinc-900 dim
            strokeWidth: touches ? 1.5 : 1,
          },
        }
      }),
    )
  }, [selectedId, baseNodes, baseEdges, adjacency, setNodes, setEdges])

  const handleNodeClick = useCallback(
    (_: React.MouseEvent, node: Node) => onSelect(node.id),
    [onSelect],
  )

  return (
    <div className="flex-1 h-full w-full" style={{ minHeight: '40vh', background: '#09090b' }}>
      <ReactFlow
        nodes={nodes}
        edges={edges}
        onNodesChange={onNodesChange}
        onEdgesChange={onEdgesChange}
        nodeTypes={nodeTypes}
        onNodeClick={handleNodeClick}
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
