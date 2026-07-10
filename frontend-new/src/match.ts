import type { Component } from './api'

// Words that appear in component names but carry no distinguishing signal
const GENERIC_NAME_WORDS = new Set([
  'system', 'module', 'equipment', 'component', 'die', 'memory',
  'license', 'service', 'supply', 'controller',
])

// Normalized alias/phrase -> canonical country string as it appears on Component.country
const COUNTRY_ALIASES: Array<[string, string]> = [
  ['taiwan', 'Taiwan'],
  ['south korea', 'South Korea'],
  ['korea', 'South Korea'],
  ['netherlands', 'Netherlands'],
  ['dutch', 'Netherlands'],
  ['usa', 'USA'],
  ['us', 'USA'],
  ['united states', 'USA'],
  ['japan', 'Japan'],
  ['germany', 'Germany'],
  ['china', 'China'],
  ['malaysia', 'Malaysia'],
  ['israel', 'Israel'],
]

function normalize(s: string): string {
  return s
    .toLowerCase()
    .replace(/[^a-z0-9\s]/g, ' ')
    .replace(/\s+/g, ' ')
    .trim()
}

// Whole-phrase match against a question already padded with boundary spaces
function hasPhrase(paddedQuestion: string, phrase: string): boolean {
  return phrase.length > 0 && paddedQuestion.includes(` ${phrase} `)
}

/**
 * Rule-based match of a free-text question against the component list.
 * No LLM, no fuzzy/edit-distance matching — exact normalized phrase/token
 * containment only. Returns the empty set when nothing matches.
 */
export function matchComponents(query: string, components: Component[]): Set<string> {
  const padded = ` ${normalize(query)} `
  const matched = new Set<string>()

  // 1. Country
  const matchedCountries = new Set<string>()
  for (const [alias, country] of COUNTRY_ALIASES) {
    if (hasPhrase(padded, alias)) matchedCountries.add(country)
  }
  if (matchedCountries.size > 0) {
    for (const c of components) {
      if (matchedCountries.has(c.country)) matched.add(c.id)
    }
  }

  // 2. Component name — distinctive tokens only
  for (const c of components) {
    const tokens = c.name
      .split(/[\s()]+/)
      .map(normalize)
      .filter((t) => t && !GENERIC_NAME_WORDS.has(t))
    if (tokens.some((t) => hasPhrase(padded, t))) {
      matched.add(c.id)
    }
  }

  // 3. Attribute
  if (hasPhrase(padded, 'single source')) {
    for (const c of components) if (c.single_source) matched.add(c.id)
  }
  if (hasPhrase(padded, 'export control') || hasPhrase(padded, 'export controlled')) {
    for (const c of components) if (c.export_controlled) matched.add(c.id)
  }
  if (hasPhrase(padded, 'critical')) {
    for (const c of components) if (c.risk_label === 'CRITICAL') matched.add(c.id)
  }

  return matched
}
