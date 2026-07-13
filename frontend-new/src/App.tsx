import { useState, useEffect, useCallback } from 'react'
import {
  fetchComponents,
  fetchHealth,
  fetchGraph,
  fetchRiskDetail,
} from './api'
import type {
  Component,
  HealthStats,
  ComponentRiskDetail,
  MergedComponent,
} from './api'

import Sidebar from './components/Sidebar'
import RiskStrip from './components/RiskStrip'
import Graph from './components/Graph'
import QAPanel from './components/QAPanel'
import { GlobeMap } from './GlobeMap'
import { useResizeHandle } from './useResizeHandle'

const SIDEBAR_DEFAULT = 260
const SIDEBAR_MIN = 200
const SIDEBAR_MAX = 420
const QA_DEFAULT = 340
const QA_MIN = 280
const QA_MAX = 560

type ViewMode = 'graph' | 'map'

export default function App() {
  const [components, setComponents] = useState<Component[]>([])
  const [mergedComponents, setMergedComponents] = useState<MergedComponent[]>([])
  const [health, setHealth] = useState<HealthStats | null>(null)
  const [selectedId, setSelectedId] = useState<string | null>(null)
  const [riskDetail, setRiskDetail] = useState<ComponentRiskDetail | null>(null)
  const [riskLoading, setRiskLoading] = useState(false)
  const [highlightedIds, setHighlightedIds] = useState<Set<string>>(new Set())
  const [sidebarWidth, setSidebarWidth] = useState(SIDEBAR_DEFAULT)
  const [qaWidth, setQaWidth] = useState(QA_DEFAULT)
  const [viewMode, setViewMode] = useState<ViewMode>('graph')

  const sidebarHandle = useResizeHandle({
    width: sidebarWidth,
    setWidth: setSidebarWidth,
    min: SIDEBAR_MIN,
    max: SIDEBAR_MAX,
    defaultWidth: SIDEBAR_DEFAULT,
    direction: 1, // handle on the right edge: dragging right grows it
  })
  const qaHandle = useResizeHandle({
    width: qaWidth,
    setWidth: setQaWidth,
    min: QA_MIN,
    max: QA_MAX,
    defaultWidth: QA_DEFAULT,
    direction: -1, // handle on the left edge: dragging right shrinks it
  })

  // Load components, graph, and health in parallel on mount
  useEffect(() => {
    Promise.all([fetchComponents(), fetchGraph(), fetchHealth()])
      .then(([comps, graph, h]) => {
        setComponents(comps)
        setHealth(h)

        // Merge component list (has risk_label) with graph (has dependencies)
        const graphById = new Map(graph.components.map((g) => [g.id, g]))
        const merged: MergedComponent[] = comps.map((c) => ({
          ...c,
          dependencies: graphById.get(c.id)?.dependencies ?? [],
        }))
        setMergedComponents(merged)
      })
      .catch((err) => console.error('Failed to load initial data:', err))
  }, [])

  const handleSelect = useCallback(async (id: string) => {
    setSelectedId(id)
    setHighlightedIds(new Set())
    setRiskLoading(true)
    setRiskDetail(null)
    try {
      const detail = await fetchRiskDetail(id)
      setRiskDetail(detail)
    } catch (err) {
      console.error('Failed to fetch risk detail:', err)
    } finally {
      setRiskLoading(false)
    }
  }, [])

  // Submitting a query is a deliberate mode switch to filtering: a non-empty
  // match takes over the graph from whatever was selected. An empty match
  // (nothing to show) leaves the current selection untouched.
  const handleQueryMatch = useCallback((ids: Set<string>) => {
    setHighlightedIds(ids)
    if (ids.size > 0) {
      setSelectedId(null)
      setRiskDetail(null)
    }
  }, [])

  const ds = health?.data_summary

  return (
    <div className="flex flex-col h-screen overflow-hidden bg-zinc-950 text-zinc-100">
      {/* Main area */}
      <div className="flex flex-1 overflow-hidden">
        {/* Sidebar — resizable, default 260px */}
        <Sidebar
          components={components}
          selectedId={selectedId}
          onSelect={handleSelect}
          width={sidebarWidth}
          onHandlePointerDown={sidebarHandle.handlePointerDown}
          onHandleDoubleClick={sidebarHandle.handleDoubleClick}
        />

        {/* Center — flex-1 */}
        <main className="flex flex-1 flex-col min-w-0 overflow-hidden">
          {/* Risk strip — top of center, only when component selected */}
          <RiskStrip detail={riskDetail} loading={riskLoading} />

          {/* View toggle — Graph / Globe */}
          <div className="flex items-center gap-1 px-3 py-1.5 border-b border-zinc-800 flex-shrink-0">
            <button
              onClick={() => setViewMode('graph')}
              className={`text-[10px] uppercase tracking-wide px-2 py-1 rounded ${
                viewMode === 'graph'
                  ? 'bg-zinc-800 text-zinc-100'
                  : 'text-zinc-500 hover:text-zinc-300'
              }`}
            >
              Graph
            </button>
            <button
              onClick={() => setViewMode('map')}
              className={`text-[10px] uppercase tracking-wide px-2 py-1 rounded ${
                viewMode === 'map'
                  ? 'bg-zinc-800 text-zinc-100'
                  : 'text-zinc-500 hover:text-zinc-300'
              }`}
            >
              Globe
            </button>
          </div>

          {/* Graph or Globe — fills remaining center height */}
          <div className="relative flex flex-1 min-h-0">
            {viewMode === 'graph' ? (
              <Graph
                components={mergedComponents}
                selectedId={selectedId}
                onSelect={handleSelect}
                highlightedIds={highlightedIds}
              />
            ) : riskDetail ? (
              <GlobeMap detail={riskDetail} components={mergedComponents} />
            ) : (
              <div className="flex-1 flex items-center justify-center text-zinc-500 text-sm uppercase tracking-wide">
                Select a component to view its supply routes
              </div>
            )}

            {/* Loading overlay — shown while a component's risk brief is fetched */}
            {riskLoading && (
              <div className="absolute inset-0 z-30 flex flex-col items-center justify-center gap-3 bg-zinc-950/70 backdrop-blur-sm">
                <div className="h-8 w-8 rounded-full border-2 border-zinc-700 border-t-zinc-100 animate-spin" />
                <div className="text-[10px] uppercase tracking-wide text-zinc-400">
                  Analyzing supply-chain risk…
                </div>
              </div>
            )}
          </div>
        </main>

        {/* Q&A panel — resizable, default 340px */}
        <QAPanel
          selectedId={selectedId}
          components={components}
          riskDetail={riskDetail}
          onSelect={handleSelect}
          highlightedIds={highlightedIds}
          onMatch={handleQueryMatch}
          width={qaWidth}
          onHandlePointerDown={qaHandle.handlePointerDown}
          onHandleDoubleClick={qaHandle.handleDoubleClick}
        />
      </div>

      {/* Footer */}
      <footer className="h-10 flex-shrink-0 border-t border-zinc-800 flex items-center px-4 gap-6">
        <div className="flex items-baseline gap-1.5">
          <span className="text-[13px] tabular-nums text-zinc-100">
            {ds?.components_count ?? '—'}
          </span>
          <span className="text-[10px] uppercase text-zinc-400">Components</span>
        </div>
        <div className="flex items-baseline gap-1.5">
          <span className="text-[13px] tabular-nums text-zinc-100">
            {ds?.single_source_count ?? '—'}
          </span>
          <span className="text-[10px] uppercase text-zinc-400">Single-Source</span>
        </div>
        <div className="flex items-baseline gap-1.5">
          <span className="text-[13px] tabular-nums text-zinc-100">
            {ds?.export_controlled_count ?? '—'}
          </span>
          <span className="text-[10px] uppercase text-zinc-400">Export Ctrl</span>
        </div>
        <div className="flex items-baseline gap-1.5">
          <span className="text-[13px] tabular-nums text-zinc-100">
            {ds ? `${ds.median_lead_time}d` : '—'}
          </span>
          <span className="text-[10px] uppercase text-zinc-400">Median Lead</span>
        </div>

        <div className="flex-1" />

        {/* GPU indicator — right side */}
        {health && (
          <span
            className={`text-[10px] tabular-nums ${
              health.gpu_available ? 'text-zinc-400' : 'text-amber-600/70'
            }`}
          >
            {health.gpu || 'no gpu'}
          </span>
        )}
      </footer>
    </div>
  )
}