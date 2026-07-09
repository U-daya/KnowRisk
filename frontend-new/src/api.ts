export interface Component {
  id: string;
  name: string;
  category: string;
  tier: number;
  country: string;
  single_source: boolean;
  export_controlled: boolean;
  lead_time_days: number;
  risk_score: number;
  risk_label: string;
  n_dependencies: number;
}

export interface DependencyRisk {
  id: string;
  name: string;
  country: string;
  risk_score: number;
  single_source: boolean;
}

// Shape returned by /api/risk/{id}'s llm_explanation field.
// Three named fields produced by the structured LLM prompt.
export interface LLMExplanation {
  risk_factor: string;
  scenario: string;
  mitigation: string;
  parse_failed: boolean;
  source: 'cache' | 'mi300x' | 'synthetic';
  latency_ms: number;
  news_grounded: boolean | null;
}

export interface ComponentRiskDetail {
  component_id: string;
  component_name: string;
  category: string;
  tier: number;
  country: string;
  single_source: boolean;
  export_controlled: boolean;
  lead_time_days: number;
  risk_score: number;
  risk_label: string;
  llm_explanation: LLMExplanation;
  dependency_risks: DependencyRisk[];
}

export interface HealthStats {
  status: string;
  llm_model: string;
  gpu: string;
  gpu_available: boolean;
  cache_hit_rate: number;
  p50_latency_ms: number;
  news_search_failures: number;
  news_empty_results: number;
  llm_loaded: boolean;
  data_summary?: {
    components_count: number;
    single_source_count: number;
    export_controlled_count: number;
    median_lead_time: number;
  };
}

// POST /api/query returns plain text Q&A — separate from the structured LLMExplanation.
// source distinguishes 'cache' | 'mi300x' | 'synthetic'.
// news_grounded is null when no component context provided and no search attempted.
export interface QueryResponse {
  text: string;
  source: 'cache' | 'mi300x' | 'synthetic';
  latency_ms: number;
  news_grounded: boolean | null;
}

// /api/graph returns the raw suppliers.json structure including dependency arrays.
export interface GraphNode {
  id: string;
  name: string;
  category: string;
  tier: number;
  country: string;
  single_source: boolean;
  export_controlled: boolean;
  lead_time_days: number;
  risk_score: number;
  dependencies: string[];
}

export interface GraphData {
  schema_version: string;
  generated_at: string;
  component_count: number;
  components: GraphNode[];
}

// Component merged with graph dependency data.
export type MergedComponent = Component & { dependencies: string[] };

const API_BASE = import.meta.env.DEV ? 'http://localhost:8000' : '';

export async function fetchComponents(): Promise<Component[]> {
  const response = await fetch(`${API_BASE}/api/components`);
  if (!response.ok) throw new Error(`Failed to fetch components: ${response.statusText}`);
  return response.json();
}

export async function fetchRiskDetail(componentId: string): Promise<ComponentRiskDetail> {
  const response = await fetch(`${API_BASE}/api/risk/${componentId}`);
  if (!response.ok) throw new Error(`Failed to fetch risk detail for ${componentId}: ${response.statusText}`);
  return response.json();
}

export async function submitQuery(query: string, componentId: string | null): Promise<QueryResponse> {
  const response = await fetch(`${API_BASE}/api/query`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ query, component_id: componentId }),
  });
  if (!response.ok) throw new Error(`Query failed: ${response.statusText}`);
  return response.json();
}

export async function fetchHealth(): Promise<HealthStats> {
  const response = await fetch(`${API_BASE}/api/health`);
  if (!response.ok) throw new Error(`Health check failed: ${response.statusText}`);
  return response.json();
}

export async function fetchGraph(): Promise<GraphData> {
  const response = await fetch(`${API_BASE}/api/graph`);
  if (!response.ok) throw new Error(`Failed to fetch graph: ${response.statusText}`);
  return response.json();
}
