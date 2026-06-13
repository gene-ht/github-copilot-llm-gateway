/**
 * Build the `LanguageModelChatInformation` object that VS Code's model picker
 * renders. Kept as its own module so the picker-facing shape (especially the
 * first-party-style `detail`/`multiplierNumeric` fields) can be unit-tested
 * without standing up the full provider.
 *
 * The gateway's upstream `/v1/models` only re-exposes the slim
 * `vscode.LanguageModelChat` projection (`id`, `name`, `family`, `version`,
 * `max_input_tokens`). The richer fields a complete
 * `LanguageModelChatInformation` needs — `maxOutputTokens`, tool-calling,
 * vision, and reasoning-effort (Thinking Effort) — are missing. We rebuild
 * them by looking the model's `family` up in a static capability table
 * distilled from the Copilot upstream `model_capabilities` response, layering
 * caller-provided defaults on top. See {@link ./copilotModelCapabilities}.
 */

import { OpenAIModel } from './types';
import { describeModel, friendlyModelName, inferModelFamily } from './modelDisplay';
import { TOKEN_CONSTANTS } from './tokenBudget';
import {
  CopilotModelCapabilities,
  ReasoningEffortLevel,
  resolveCopilotCapabilities,
} from './copilotModelCapabilities';

/**
 * Grey right-hand label rendered in the VS Code chat model picker. Matches the
 * shape native Copilot Chat BYOK providers use (e.g. `detail: 'Anthropic'`),
 * which is what visually groups all of our models under the provider.
 */
export const PROVIDER_DETAIL_LABEL = 'LLM Gateway';

/**
 * Cost-tier multiplier surfaced to Copilot Chat. Set to 0 so BYOK / self-hosted
 * models don't appear to consume Copilot premium request quota.
 */
export const PROVIDER_MULTIPLIER_NUMERIC = 0;

/** The configuration-schema property key for the Thinking Effort picker. */
export const REASONING_EFFORT_PROPERTY = 'reasoningEffort';

export interface ModelCapabilities {
  readonly imageInput?: boolean;
  readonly toolCalling?: boolean | number;
}

export interface BuildModelInfoInput {
  readonly model: OpenAIModel;
  readonly defaultMaxTokens: number;
  readonly defaultMaxOutputTokens: number;
  /**
   * Caller-provided capability defaults (from gateway config). These act as the
   * lowest-priority fallback: family-table values win, then these defaults.
   */
  readonly capabilities: ModelCapabilities;
}

/**
 * A minimal JSON-schema-like configuration descriptor for the picker. Mirrors
 * the subset of `vscode.LanguageModelConfigurationSchema` we populate.
 */
export interface ModelConfigurationSchema {
  readonly properties?: {
    readonly [key: string]: Record<string, unknown> & {
      readonly enumItemLabels?: string[];
      readonly group?: string;
    };
  };
}

/**
 * Picker-facing fields plus the resolved total context size. Total context is
 * returned separately because VS Code's `maxInputTokens` means usable prompt
 * budget, not the model's full context window.
 */
export interface BuildModelInfoResult {
  readonly info: {
    readonly id: string;
    readonly name: string;
    readonly family: string;
    readonly version: string;
    readonly maxInputTokens: number;
    readonly maxOutputTokens: number;
    readonly capabilities: ModelCapabilities;
    readonly detail: string;
    readonly tooltip: string;
    readonly description?: string;
    readonly isUserSelectable: true;
    readonly multiplierNumeric: number;
    readonly configurationSchema?: ModelConfigurationSchema;
  };
  readonly totalContext: number;
  readonly hasServerReportedContext: boolean;
}

const REASONING_EFFORT_DESCRIPTIONS: Record<ReasoningEffortLevel, string> = {
  none: 'No reasoning applied',
  minimal: 'Minimal reasoning for fastest responses',
  low: 'Faster responses with less reasoning',
  medium: 'Balanced reasoning and speed',
  high: 'Greater reasoning depth but slower',
  xhigh: 'Highest reasoning depth but slowest',
  max: 'Absolute maximum capability with no constraints',
};

/**
 * Pick a sensible default reasoning-effort level: prefer `high` when offered,
 * else the median of the available levels.
 */
function pickDefaultReasoningEffort(levels: readonly ReasoningEffortLevel[]): ReasoningEffortLevel | undefined {
  if (levels.length === 0) {
    return undefined;
  }
  if (levels.includes('high')) {
    return 'high';
  }
  return levels[Math.floor((levels.length - 1) / 2)];
}

/**
 * Build a `configurationSchema` exposing a "Thinking Effort" dropdown when the
 * model supports reasoning-effort selection. The `group: 'navigation'` hint
 * surfaces it as a primary action in the model picker (same shape native
 * Copilot BYOK providers use).
 */
export function buildReasoningEffortSchema(levels: readonly ReasoningEffortLevel[]): ModelConfigurationSchema | undefined {
  if (!levels || levels.length === 0) {
    return undefined;
  }
  return {
    properties: {
      [REASONING_EFFORT_PROPERTY]: {
        type: 'string',
        title: 'Thinking Effort',
        enum: [...levels],
        enumItemLabels: levels.map((l) => l.charAt(0).toUpperCase() + l.slice(1)),
        enumDescriptions: levels.map((l) => REASONING_EFFORT_DESCRIPTIONS[l] ?? l),
        default: pickDefaultReasoningEffort(levels),
        group: 'navigation',
      },
    },
  };
}

/**
 * Translate a raw `/v1/models` entry into the picker-facing model info plus
 * the resolved total context.
 *
 * Field resolution priority is: **family capability table → caller defaults →
 * conservative constants**.
 *
 * `maxInputTokens` is reported to the picker as the usable prompt budget. The
 * gateway's `max_input_tokens` (when present) is already the upstream
 * `max_prompt_tokens` — a usable budget, NOT the full context window — so we
 * pass it straight through rather than subtracting the output budget again.
 * The full context window is returned separately as `totalContext` for
 * request-time budgeting and status display.
 */
export function buildModelInfo({
  model,
  defaultMaxTokens,
  defaultMaxOutputTokens,
  capabilities,
}: BuildModelInfoInput): BuildModelInfoResult {
  const caps: CopilotModelCapabilities = resolveCopilotCapabilities(model.version);

  // Server-reported usable prompt budget (gateway `max_input_tokens` ==
  // upstream `max_prompt_tokens`). Other self-hosted backends may instead
  // report a full window via max_model_len/context_length/context_window.
  const serverPromptBudget = model.max_input_tokens;
  const serverFullWindow = model.max_model_len ?? model.context_length ?? model.context_window;
  const hasServerReportedContext = serverPromptBudget !== undefined || serverFullWindow !== undefined;

  // Usable prompt budget exposed to the picker as `maxInputTokens`.
  const maxInputTokens =
    serverPromptBudget ??
    caps.maxPromptTokens ??
    serverFullWindow ??
    caps.maxContextWindowTokens ??
    defaultMaxTokens;

  // Max output tokens: family table wins, then caller default constant.
  const maxOutputTokens = Math.max(
    TOKEN_CONSTANTS.MIN_OUTPUT_TOKENS,
    caps.maxOutputTokens ?? defaultMaxOutputTokens,
  );

  // Full context window for request-time budgeting / status display.
  const totalContext =
    caps.maxContextWindowTokens ??
    serverFullWindow ??
    (serverPromptBudget !== undefined ? serverPromptBudget + maxOutputTokens : defaultMaxTokens);

  // Capabilities: family table wins; fall back to caller-provided config defaults.
  const resolvedCapabilities: ModelCapabilities = {
    toolCalling: caps.toolCalling ?? capabilities.toolCalling,
    imageInput: caps.vision ?? capabilities.imageInput,
  };

  const description = describeModel(model);
  const tooltip = description ? `${model.id} — ${description}` : model.id;
  // Prefer the server-provided display name/family/version, fall back to id-derived values.
  const friendlyName = model.name || friendlyModelName(model.id);
  const family = model.family || inferModelFamily(model.id);
  const version = model.version || friendlyName;

  const configurationSchema = caps.reasoningEffort
    ? buildReasoningEffortSchema(caps.reasoningEffort)
    : undefined;

  const info: BuildModelInfoResult['info'] = {
    id: model.id,
    name: friendlyName,
    family,
    version,
    maxInputTokens,
    maxOutputTokens,
    capabilities: resolvedCapabilities,
    detail: PROVIDER_DETAIL_LABEL,
    tooltip,
    isUserSelectable: true,
    multiplierNumeric: PROVIDER_MULTIPLIER_NUMERIC,
    ...(description ? { description } : {}),
    ...(configurationSchema ? { configurationSchema } : {}),
  };

  return {
    info,
    totalContext,
    hasServerReportedContext,
  };
}
