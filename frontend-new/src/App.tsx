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

export default function App() {
  const [components, setComponents] = useState<Component[]>([])
  const [mergedComponents, setMergedComponents] = useState<MergedComponent[]>([])
  const [health, setHealth] = useState<HealthStats | null>(null)
  const [selectedId, setSelectedId] = useState<string | null>(null)
  const [riskDetail, setRiskDetail] = useState<ComponentRiskDetail | null>(null)
  const [riskLoading, setRiskLoading] = useState(false)

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

  const ds = health?.data_summary

  return (
    <div className="flex flex-col h-screen overflow-hidden bg-zinc-950 text-zinc-300">
      {/* Main area */}
      <div className="flex flex-1 overflow-hidden">
        {/* Sidebar — 260px */}
        <Sidebar
          components={components}
          selectedId={selectedId}
          onSelect={handleSelect}
        />

        {/* Center — flex-1 */}
        <main className="flex flex-1 flex-col min-w-0 overflow-hidden">
          {/* Risk strip — top of center, only when component selected */}
          <RiskStrip detail={riskDetail} loading={riskLoading} />

          {/* Graph — fills remaining center height */}
          <Graph
            components={mergedComponents}
            selectedId={selectedId}
            onSelect={handleSelect}
          />
        </main>

        {/* Q&A panel — 340px */}
        <QAPanel selectedId={selectedId} />
      </div>

      {/* Footer */}
      <footer className="h-10 flex-shrink-0 border-t border-zinc-800 flex items-center px-4 gap-6">
        <div className="flex items-baseline gap-1.5">
          <span className="text-[13px] tabular-nums text-zinc-300">
            {ds?.components_count ?? '—'}
          </span>
          <span className="text-[10px] uppercase text-zinc-500">Components</span>
        </div>
        <div className="flex items-baseline gap-1.5">
          <span className="text-[13px] tabular-nums text-zinc-300">
            {ds?.single_source_count ?? '—'}
          </span>
          <span className="text-[10px] uppercase text-zinc-500">Single-Source</span>
        </div>
        <div className="flex items-baseline gap-1.5">
          <span className="text-[13px] tabular-nums text-zinc-300">
            {ds?.export_controlled_count ?? '—'}
          </span>
          <span className="text-[10px] uppercase text-zinc-500">Export Ctrl</span>
        </div>
        <div className="flex items-baseline gap-1.5">
          <span className="text-[13px] tabular-nums text-zinc-300">
            {ds ? `${ds.median_lead_time}d` : '—'}
          </span>
          <span className="text-[10px] uppercase text-zinc-500">Median Lead</span>
        </div>

        <div className="flex-1" />

        {/* GPU indicator — right side */}
        {health && (
          <span
            className={`text-[10px] tabular-nums ${
              health.gpu_available ? 'text-zinc-600' : 'text-amber-600/70'
            }`}
          >
            {health.gpu || 'no gpu'}
          </span>
        )}
      </footer>
    </div>
  )
}
