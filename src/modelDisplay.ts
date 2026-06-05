/**
 * Helpers for turning raw OpenAI `/v1/models` entries into the shape that the
 * VS Code Copilot Chat model picker renders nicely.
 *
 * Kept as its own module so it can be unit-tested without VS Code.
 */

import { OpenAIModel } from './types';

/**
 * Produce a display-friendly short name for a model ID.
 *
 * Hugging-Face-style IDs contain an org prefix (`Qwen/Qwen3-8B`) that the
 * Copilot model picker renders verbatim with a slash. Strip the prefix so the
 * picker shows `Qwen3-8B` (the owner is still available on the tooltip).
 */
export function friendlyModelName(id: string): string {
  const slash = id.lastIndexOf('/');
  if (slash >= 0 && slash < id.length - 1) {
    return id.slice(slash + 1);
  }
  return id;
}

const FAMILY_KEYWORDS: Array<{ match: RegExp; family: string }> = [
  { match: /qwen/i, family: 'qwen' },
  { match: /llama/i, family: 'llama' },
  { match: /mistral/i, family: 'mistral' },
  { match: /mixtral/i, family: 'mixtral' },
  { match: /deepseek/i, family: 'deepseek' },
  { match: /phi/i, family: 'phi' },
  { match: /gemma/i, family: 'gemma' },
  { match: /gpt-?oss/i, family: 'gpt-oss' },
  { match: /yi[-_]/i, family: 'yi' },
  { match: /command[-_]?r/i, family: 'command-r' },
];

/**
 * Infer a VS Code `family` value from the model ID. Falls back to
 * `'llm-gateway'` so the picker still groups all gateway models together when
 * the family can't be determined.
 */
export function inferModelFamily(id: string): string {
  for (const { match, family } of FAMILY_KEYWORDS) {
    if (match.test(id)) { return family; }
  }
  return 'llm-gateway';
}

/**
 * Build a short description string highlighting context size and owner.
 * Used as the `detail` shown under the model name in the picker.
 */
export function describeModel(model: OpenAIModel): string {
  const context = model.max_input_tokens ?? model.max_model_len ?? model.context_length ?? model.context_window;
  const parts: string[] = [];
  if (context) {
    parts.push(`${formatTokens(context)} ctx`);
  }
  if (model.owned_by && model.owned_by !== 'organization-owner') {
    parts.push(model.owned_by);
  }
  return parts.join(' • ');
}

function formatTokens(n: number): string {
  if (n >= 1_000_000) { return `${Math.round(n / 1_000_000)}M`; }
  if (n >= 1_000) { return `${Math.round(n / 1_000)}K`; }
  return String(n);
}

/**
 * Deduplicate a list of models by `id`.
 *
 * Servers sometimes return the same id multiple times (for example, a single
 * gateway aggregating several upstream vendors that each advertise the same
 * model id but with slightly different capability metadata). Naively keeping
 * the first occurrence loses information: in particular, an upstream that
 * reports a much smaller `max_input_tokens` than the model truly supports
 * would silently cap users to the wrong context window.
 *
 * Strategy: keep the entry with the largest `max_input_tokens` for each id
 * (assuming the largest reporter is the most accurate about the model's
 * real capacity), while preserving the first-seen relative order of ids in
 * the output so the picker UI stays stable across refreshes.
 *
 * Ties (or missing `max_input_tokens` on every entry) fall back to first-seen,
 * matching the previous behaviour.
 */
export function dedupeModels(models: readonly OpenAIModel[]): OpenAIModel[] {
  const bestById = new Map<string, OpenAIModel>();
  const firstSeenOrder: string[] = [];

  for (const model of models) {
    const existing = bestById.get(model.id);
    if (!existing) {
      bestById.set(model.id, model);
      firstSeenOrder.push(model.id);
      continue;
    }
    // Prefer the entry advertising the larger context window. Missing values
    // are treated as 0 so any reported value beats no value.
    const existingCtx = existing.max_input_tokens ?? 0;
    const candidateCtx = model.max_input_tokens ?? 0;
    if (candidateCtx > existingCtx) {
      bestById.set(model.id, model);
    }
  }

  return firstSeenOrder.map((id) => bestById.get(id)!);
}
