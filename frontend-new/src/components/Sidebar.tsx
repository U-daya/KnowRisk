import { RISK_COLOR, type RiskLabel } from '../risk'
import type { Component } from '../api'

interface Props {
  components: Component[]
  selectedId: string | null
  onSelect: (id: string) => void
}

export default function Sidebar({ components, selectedId, onSelect }: Props) {
  const sorted = [...components].sort((a, b) => b.risk_score - a.risk_score)

  return (
    <aside className="w-[260px] flex-shrink-0 flex flex-col border-r border-zinc-800 bg-zinc-950">
      <div className="px-3 pt-3 pb-2 flex items-baseline gap-2">
        <span className="text-[10px] uppercase tracking-wide text-zinc-500">Components</span>
        <span className="text-[10px] tabular-nums text-zinc-600">{components.length}</span>
      </div>
      <div className="px-3 pb-2 text-[9px] uppercase text-zinc-600 border-b border-zinc-800">
        TOPOLOGICAL RISK · TOP DECILE = CRITICAL
      </div>

      <div className="flex-1 overflow-y-auto">
        {sorted.map((c) => {
          const selected = c.id === selectedId
          return (
            <button
              key={c.id}
              onClick={() => onSelect(c.id)}
              className={`w-full text-left px-3 py-2 flex items-center gap-2 border-b border-zinc-800/50 transition-colors duration-100 ${
                selected ? 'bg-zinc-900' : 'bg-zinc-950 hover:bg-zinc-900'
              }`}
            >
              {/* 8px status dot */}
              <div
                className="w-2 h-2 rounded-full flex-shrink-0"
                style={{ backgroundColor: RISK_COLOR[c.risk_label as RiskLabel] }}
              />
              <div className="flex-1 min-w-0">
                <div className="flex items-baseline gap-1.5">
                  <span
                    className="text-[9px] uppercase tracking-wide flex-shrink-0"
                    style={{ color: RISK_COLOR[c.risk_label as RiskLabel] }}
                  >
                    {c.risk_label}
                  </span>
                  <span className="text-[12px] text-zinc-300 truncate">{c.name}</span>
                </div>
                <div className="text-[10px] text-zinc-500 truncate">
                  {c.country} · T{c.tier} · {c.lead_time_days}d
                </div>
              </div>
              <div className="text-[11px] tabular-nums text-zinc-600 flex-shrink-0">
                {c.risk_score.toFixed(4)}
              </div>
            </button>
          )
        })}
      </div>
    </aside>
  )
}
