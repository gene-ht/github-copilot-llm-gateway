/**
 * Static capability table distilled from the GitHub Copilot upstream
 * `/models` raw response (the `model_capabilities` shape).
 *
 * Why this exists: our gateway's upstream (`github-copilot-api-vscode`) only
 * re-exposes the slim `vscode.LanguageModelChat` projection over `/v1/models`
 * — i.e. `id`, `name`, `family`, `version`, `max_input_tokens`, plus a custom
 * `capabilities` object that does NOT carry `max_output_tokens`, tool-calling,
 * vision, or reasoning-effort information. To rebuild a complete
 * `vscode.LanguageModelChatInformation` for the model picker we look up the
 * model's `version` in this table and fill in the missing fields.
 *
 * The connecting key is `version` — the upstream version string points at the
 * REAL underlying model (e.g. `copilot-utility` has version `gpt-5.3-codex`,
 * `gpt-4o-mini` has version `gpt-4o-mini-2024-07-18`), whereas `family` is
 * often a logical alias/route name. Keying by `version` means routing aliases
 * automatically inherit the capabilities of the concrete model they resolve
 * to, without maintaining duplicate alias rows.
 *
 * Lookup is by `version` only — the single dimension this table is keyed on.
 * When the version is unknown (e.g. a self-hosted / custom model with no known
 * Copilot version), an empty capability object is returned and the caller
 * fills the gaps from the server-reported values and configured defaults — we
 * never guess.
 */

/** Reasoning-effort levels recognized by the VS Code "Thinking Effort" picker. */
export type ReasoningEffortLevel = 'none' | 'minimal' | 'low' | 'medium' | 'high' | 'xhigh' | 'max';

/**
 * The subset of upstream `model_capabilities` we need to rebuild a complete
 * `LanguageModelChatInformation`. All fields are optional so callers can layer
 * user config and defaults on top.
 */
export interface CopilotModelCapabilities {
  /** Full context window (`limits.max_context_window_tokens`). */
  readonly maxContextWindowTokens?: number;
  /** Usable prompt budget (`limits.max_prompt_tokens`). */
  readonly maxPromptTokens?: number;
  /** Max output tokens (`limits.max_output_tokens`). */
  readonly maxOutputTokens?: number;
  /** Whether the model supports tool/function calling (`supports.tool_calls`). */
  readonly toolCalling?: boolean;
  /** Whether the model supports image input (`supports.vision`). */
  readonly vision?: boolean;
  /** Reasoning-effort levels (`supports.reasoning_effort`), if the model is a thinking model. */
  readonly reasoningEffort?: readonly ReasoningEffortLevel[];
}

/**
 * Capability table keyed by `version` (the concrete underlying model string
 * the gateway reports). This is a faithful, 1:1 transcription of the upstream
 * Copilot `/models` raw response (`capabilities.limits` + `capabilities.supports`),
 * grouped by `version` — NOT a heuristic approximation. Each row mirrors the
 * exact upstream `max_context_window_tokens` / `max_prompt_tokens` /
 * `max_output_tokens` / `tool_calls` / `vision` / `reasoning_effort` values.
 *
 * Keep this in sync with upstream by re-transcribing the raw `/models`
 * response. Unknown versions (self-hosted / custom models) resolve to an empty
 * capability object so the caller falls back to server-reported values and
 * configured defaults.
 *
 * Dedupe note: where the raw response lists the same `version` twice with
 * differing capabilities, the more capable record is kept (union of vision /
 * larger output budget): `gpt-4o-2024-11-20` and `gpt-4o-2024-05-13`.
 *
 * Routing aliases (`auto`, `copilot-utility`, `copilot-utility-small`) are
 * intentionally absent — their gateway `version` points at a concrete model
 * already in this table, so they inherit those capabilities automatically.
 */
export const COPILOT_MODEL_CAPABILITIES: Readonly<Record<string, CopilotModelCapabilities>> = {
  // ---- Anthropic Claude ----
  'claude-opus-4.6-1m': { maxContextWindowTokens: 1_000_000, maxPromptTokens: 936_000, maxOutputTokens: 64_000, toolCalling: true, vision: true, reasoningEffort: ['low', 'medium', 'high', 'max'] },
  'claude-opus-4.6': { maxContextWindowTokens: 1_000_000, maxPromptTokens: 936_000, maxOutputTokens: 64_000, toolCalling: true, vision: true, reasoningEffort: ['low', 'medium', 'high', 'max'] },
  'claude-opus-4.7-1m-internal': { maxContextWindowTokens: 1_000_000, maxPromptTokens: 936_000, maxOutputTokens: 64_000, toolCalling: true, vision: true, reasoningEffort: ['low', 'medium', 'high', 'xhigh', 'max'] },
  'claude-opus-4.7': { maxContextWindowTokens: 1_000_000, maxPromptTokens: 936_000, maxOutputTokens: 64_000, toolCalling: true, vision: true, reasoningEffort: ['low', 'medium', 'high', 'xhigh', 'max'] },
  'claude-opus-4.8': { maxContextWindowTokens: 1_000_000, maxPromptTokens: 936_000, maxOutputTokens: 64_000, toolCalling: true, vision: true, reasoningEffort: ['low', 'medium', 'high', 'xhigh', 'max'] },
  'claude-sonnet-4.6': { maxContextWindowTokens: 1_000_000, maxPromptTokens: 936_000, maxOutputTokens: 64_000, toolCalling: true, vision: true, reasoningEffort: ['low', 'medium', 'high', 'max'] },
  'claude-sonnet-4.5': { maxContextWindowTokens: 200_000, maxPromptTokens: 168_000, maxOutputTokens: 32_000, toolCalling: true, vision: true },
  'claude-opus-4.5': { maxContextWindowTokens: 200_000, maxPromptTokens: 168_000, maxOutputTokens: 32_000, toolCalling: true, vision: true },
  'claude-haiku-4.5': { maxContextWindowTokens: 200_000, maxPromptTokens: 136_000, maxOutputTokens: 64_000, toolCalling: true, vision: true },

  // ---- Google Gemini ----
  'gemini-3.1-pro-preview': { maxContextWindowTokens: 1_000_000, maxPromptTokens: 936_000, maxOutputTokens: 64_000, toolCalling: true, vision: true, reasoningEffort: ['low', 'medium', 'high'] },
  'gemini-3.5-flash': { maxContextWindowTokens: 1_000_000, maxPromptTokens: 936_000, maxOutputTokens: 64_000, toolCalling: true, vision: true, reasoningEffort: ['minimal', 'low', 'medium', 'high'] },
  'gemini-3-flash-preview': { maxContextWindowTokens: 128_000, maxPromptTokens: 128_000, maxOutputTokens: 64_000, toolCalling: true, vision: true, reasoningEffort: ['low', 'medium', 'high'] },
  'gemini-2.5-pro': { maxContextWindowTokens: 128_000, maxPromptTokens: 128_000, maxOutputTokens: 64_000, toolCalling: true, vision: true },

  // ---- OpenAI GPT-5.x ----
  'gpt-5.3-codex': { maxContextWindowTokens: 400_000, maxPromptTokens: 272_000, maxOutputTokens: 128_000, toolCalling: true, vision: true, reasoningEffort: ['low', 'medium', 'high', 'xhigh'] },
  'gpt-5.4-mini': { maxContextWindowTokens: 400_000, maxPromptTokens: 272_000, maxOutputTokens: 128_000, toolCalling: true, vision: true, reasoningEffort: ['none', 'low', 'medium', 'high', 'xhigh'] },
  'gpt-5.4': { maxContextWindowTokens: 1_050_000, maxPromptTokens: 922_000, maxOutputTokens: 128_000, toolCalling: true, vision: true, reasoningEffort: ['none', 'low', 'medium', 'high', 'xhigh'] },
  'gpt-5.5': { maxContextWindowTokens: 1_050_000, maxPromptTokens: 922_000, maxOutputTokens: 128_000, toolCalling: true, vision: true, reasoningEffort: ['none', 'low', 'medium', 'high', 'xhigh'] },
  'gpt-5-mini': { maxContextWindowTokens: 264_000, maxPromptTokens: 128_000, maxOutputTokens: 64_000, toolCalling: true, vision: true, reasoningEffort: ['low', 'medium', 'high'] },

  // ---- OpenAI GPT-4 family (keyed by concrete version string) ----
  'gpt-4.1-2025-04-14': { maxContextWindowTokens: 128_000, maxPromptTokens: 128_000, maxOutputTokens: 16_384, toolCalling: true, vision: true },
  'gpt-4o-2024-11-20': { maxContextWindowTokens: 128_000, maxPromptTokens: 64_000, maxOutputTokens: 16_384, toolCalling: true, vision: true },
  'gpt-4o-2024-08-06': { maxContextWindowTokens: 128_000, maxPromptTokens: 64_000, maxOutputTokens: 16_384, toolCalling: true, vision: false },
  'gpt-4o-2024-05-13': { maxContextWindowTokens: 128_000, maxPromptTokens: 64_000, maxOutputTokens: 16_384, toolCalling: true, vision: true },
  'gpt-4o-mini-2024-07-18': { maxContextWindowTokens: 128_000, maxPromptTokens: 12_288, maxOutputTokens: 4_096, toolCalling: true, vision: false },
  'gpt-4-0125-preview': { maxContextWindowTokens: 128_000, maxPromptTokens: 64_000, maxOutputTokens: 4_096, toolCalling: true, vision: false },
  'gpt-4-0613': { maxContextWindowTokens: 32_768, maxPromptTokens: 32_768, maxOutputTokens: 4_096, toolCalling: true, vision: false },
  'gpt-3.5-turbo-0613': { maxContextWindowTokens: 16_384, maxPromptTokens: 12_288, maxOutputTokens: 4_096, toolCalling: true, vision: false },

  // ---- Microsoft / Experimental (internal) ----
  'mai-code-1-flash-internal': { maxContextWindowTokens: 256_000, maxPromptTokens: 128_000, maxOutputTokens: 128_000, toolCalling: true, vision: false, reasoningEffort: ['low', 'medium', 'high'] },
  'trajectory-compaction': { maxContextWindowTokens: 262_144, maxPromptTokens: 245_760, maxOutputTokens: 16_384, toolCalling: true, vision: false },
};

/**
 * Resolve capabilities by `version` — the single dimension this table is keyed
 * on. Returns an empty object when the version is unknown (or absent); the
 * caller (`buildModelInfo`) then fills missing fields from the server-reported
 * values and configured defaults rather than guessing.
 */
export function resolveCopilotCapabilities(version: string | undefined): CopilotModelCapabilities {
  return (version && COPILOT_MODEL_CAPABILITIES[version]) || {};
}
