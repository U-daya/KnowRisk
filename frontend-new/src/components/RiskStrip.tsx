import type { ComponentRiskDetail } from '../api'

interface Props {
  detail: ComponentRiskDetail | null
  loading: boolean
}

const BLOCKS = [
  { key: 'risk_factor' as const, label: 'Risk Factor', borderColor: '#dc2626' }, // red-600
  { key: 'scenario'    as const, label: 'Scenario',    borderColor: '#d97706' }, // amber-600
  { key: 'mitigation' as const, label: 'Mitigation',  borderColor: '#52525b' }, // zinc-600
]

export default function RiskStrip({ detail, loading }: Props) {
  // Before any component is selected, hold the space with a single hint line
  if (!detail && !loading) {
    return (
      <div className="flex items-center justify-center border-b border-zinc-800 px-4 py-8">
        <span className="text-[10px] uppercase tracking-wide text-zinc-500">
          SELECT A COMPONENT
        </span>
      </div>
    )
  }

  if (!loading && detail?.llm_explanation.parse_failed) {
    return (
      <div className="flex border-b border-zinc-800 px-4 py-3">
        <div className="flex-1 pl-3" style={{ borderLeft: `2px solid ${BLOCKS[0].borderColor}` }}>
          <div className="text-[10px] uppercase tracking-wide text-zinc-400 mb-1">
            Risk Factor
          </div>
          <p className="text-[12px] leading-snug text-zinc-100">
            {detail?.llm_explanation.risk_factor ?? ''}
          </p>
        </div>
      </div>
    )
  }

  return (
    <div className="flex gap-6 border-b border-zinc-800 px-4 py-3">
      {BLOCKS.map((block) => (
        <div
          key={block.key}
          className="flex-1 pl-3"
          style={{ borderLeft: `2px solid ${block.borderColor}` }}
        >
          <div className="text-[10px] uppercase tracking-wide text-zinc-400 mb-1">
            {block.label}
          </div>
          {loading ? (
            <div className="space-y-1.5 py-0.5" aria-label="Loading">
              <div className="h-2 w-full rounded bg-zinc-800 animate-pulse" />
              <div className="h-2 w-4/5 rounded bg-zinc-800 animate-pulse" />
            </div>
          ) : (
            <p className="text-[12px] leading-snug text-zinc-100">
              {detail?.llm_explanation[block.key] ?? ''}
            </p>
          )}
        </div>
      ))}
    </div>
  )
}
