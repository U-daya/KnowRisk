export const RISK_COLOR = {
  CRITICAL: '#dc2626',  // red-600
  HIGH:     '#d97706',  // amber-600
  MEDIUM:   '#ca8a04',  // yellow-600
  LOW:      '#3f3f46',  // zinc-700 — low risk is not green, it is quiet
} as const;

export type RiskLabel = keyof typeof RISK_COLOR;
