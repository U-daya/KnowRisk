import { useMemo, useState, useRef, useEffect } from 'react'
import { submitQuery } from '../api'
import type { Component, ComponentRiskDetail, QueryResponse } from '../api'
import { RISK_COLOR, type RiskLabel } from '../risk'

// ── Types ─────────────────────────────────────────────────────────────────────

type Turn =
  | { role: 'user'; text: string }
  | { role: 'system'; response: QueryResponse }
  | { role: 'pending' }

// ── Source label ──────────────────────────────────────────────────────────────

function SourceLine({ response }: { response: QueryResponse }) {
  const { source, latency_ms, news_grounded } = response

  if (source === 'synthetic') {
    return (
      <div className="text-[10px] uppercase tracking-wide text-amber-600/70 mb-1">
        SYNTHETIC — LLM OFFLINE
      </div>
    )
  }

  if (source === 'cache') {
    return (
      <div className="text-[10px] uppercase tracking-wide text-zinc-600 mb-1">
        CACHE
      </div>
    )
  }

  // mi300x
  const noNews =
    news_grounded === false ? (
      <span className="text-zinc-700"> · NO NEWS</span>
    ) : null

  return (
    <div className="text-[10px] uppercase tracking-wide text-zinc-600 mb-1">
      MI300X · <span className="tabular-nums">{latency_ms.toFixed(0)}ms</span>
      {noNews}
    </div>
  )
}

// ── Component ─────────────────────────────────────────────────────────────────

interface Props {
  selectedId: string | null
  components: Component[]
  riskDetail: ComponentRiskDetail | null
  onSelect: (id: string) => void
}

export default function QAPanel({ selectedId, components, riskDetail, onSelect }: Props) {
  const [turns, setTurns] = useState<Turn[]>([])
  const [input, setInput] = useState('')
  const [busy, setBusy] = useState(false)
  const bottomRef = useRef<HTMLDivElement>(null)

  // Lookup for the risk band of any component by id, for dependency-row dots
  const labelById = useMemo(
    () => new Map(components.map((c) => [c.id, c.risk_label as RiskLabel])),
    [components],
  )

  const sortedDeps = useMemo(
    () => [...(riskDetail?.dependency_risks ?? [])].sort((a, b) => b.risk_score - a.risk_score),
    [riskDetail],
  )

  // Scroll to bottom on new turns
  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'instant' })
  }, [turns])

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault()
    const query = input.trim()
    if (!query || busy) return

    setInput('')
    setBusy(true)
    setTurns((prev) => [...prev, { role: 'user', text: query }, { role: 'pending' }])

    try {
      const response = await submitQuery(query, selectedId)
      setTurns((prev) => [
        ...prev.slice(0, -1), // remove pending
        { role: 'system', response },
      ])
    } catch (err) {
      setTurns((prev) => [
        ...prev.slice(0, -1),
        {
          role: 'system',
          response: {
            text: 'Request failed. Is the backend running?',
            source: 'synthetic',
            latency_ms: 0,
            news_grounded: null,
          },
        },
      ])
    } finally {
      setBusy(false)
    }
  }

  return (
    <aside className="w-[340px] flex-shrink-0 flex flex-col border-l border-zinc-800 bg-zinc-950">
      {/* Header */}
      <div className="px-3 pt-3 pb-2 border-b border-zinc-800 flex items-baseline gap-2">
        <span className="text-[10px] uppercase tracking-wide text-zinc-500">Q&amp;A</span>
        {selectedId && (
          <span className="text-[10px] text-zinc-600 tabular-nums">{selectedId}</span>
        )}
      </div>

      {/* Turn stream */}
      <div className="flex-1 overflow-y-auto px-3 py-3 space-y-4">
        {turns.length === 0 && (
          <p className="text-[10px] uppercase tracking-wide text-zinc-600">
            {selectedId ? `${selectedId} selected` : 'No component selected'}
          </p>
        )}

        {selectedId && riskDetail && (
          <div>
            <div className="text-[9px] uppercase tracking-wide text-zinc-600 border-b border-zinc-800 pb-1 mb-2">
              Dependencies
            </div>
            {sortedDeps.map((dep) => (
              <button
                key={dep.id}
                onClick={() => onSelect(dep.id)}
                className="w-full flex items-center gap-1.5 py-1 text-left hover:bg-zinc-900 transition-colors duration-100"
              >
                <span
                  className="w-2 h-2 rounded-full flex-shrink-0"
                  style={{ backgroundColor: RISK_COLOR[labelById.get(dep.id) ?? 'LOW'] }}
                />
                <span className="text-[11px] text-zinc-400 truncate flex-1 min-w-0">
                  {dep.name}
                </span>
                <span className="text-[9px] text-zinc-600 flex-shrink-0">{dep.country}</span>
                {dep.single_source && (
                  <span className="text-[8px] uppercase text-amber-600/70 flex-shrink-0">
                    SINGLE-SOURCE
                  </span>
                )}
                <span className="text-[10px] tabular-nums text-zinc-600 flex-shrink-0 text-right">
                  {dep.risk_score.toFixed(4)}
                </span>
              </button>
            ))}
          </div>
        )}

        {turns.map((turn, i) => {
          if (turn.role === 'user') {
            return (
              <div
                key={i}
                className="pl-3 border-l border-zinc-700 text-[12px] text-zinc-400"
              >
                {turn.text}
              </div>
            )
          }

          if (turn.role === 'pending') {
            return (
              <div key={i} className="text-[13px] text-zinc-600">
                …
              </div>
            )
          }

          // system turn
          return (
            <div key={i}>
              <SourceLine response={turn.response} />
              <p className="text-[13px] leading-relaxed text-zinc-200">
                {turn.response.text}
              </p>
            </div>
          )
        })}
        <div ref={bottomRef} />
      </div>

      {/* Input */}
      <form
        onSubmit={handleSubmit}
        className="border-t border-zinc-800 flex gap-0"
      >
        <input
          type="text"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder="Ask about supply-chain risk…"
          disabled={busy}
          className="flex-1 bg-zinc-900 border-0 px-3 py-2.5 text-[13px] text-zinc-300 placeholder:text-zinc-600 focus:outline-none disabled:opacity-50"
        />
        <button
          type="submit"
          disabled={busy || !input.trim()}
          className="px-3 py-2.5 bg-zinc-800 text-[12px] text-zinc-400 hover:bg-zinc-700 disabled:opacity-40 border-l border-zinc-800 flex-shrink-0"
        >
          Ask
        </button>
      </form>
    </aside>
  )
}
