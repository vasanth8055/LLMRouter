
const names: Record<string, string> = {
  "deepseek-v3-0324": "DeepSeek V3",
  "gemini-2.5-flash": "Gemini 2.5 Flash",
  "gemini-2.5-pro": "Gemini 2.5 Pro",
};

export function displayModel(id: string | null | undefined): string {
  return id ? names[id] ?? id : "Unknown model";
}

export function formatCost(value: number | null | undefined): string {
  return value === null || value === undefined ? "Unavailable" : `$${value.toFixed(6)}`;
}

export function formatLatency(value: number | null | undefined): string {
  return value === null || value === undefined ? "Unavailable" : `${value.toLocaleString(undefined, { maximumFractionDigits: 1 })} ms`;
}

export function formatQuality(value: number | null | undefined): string {
  return value === null || value === undefined ? "Unavailable" : `${(value * 100).toFixed(2)}%`;
}

export function formatTimestamp(value: string): string {
  return new Intl.DateTimeFormat(undefined, { dateStyle: "medium", timeStyle: "short" }).format(new Date(value));
}

export function modelColor(id: string): string {
  if (id === "deepseek-v3-0324") return "#5177e8";
  if (id === "gemini-2.5-flash") return "#12a886";
  if (id === "gemini-2.5-pro") return "#a171dc";
  return "#8a98ad";
}

