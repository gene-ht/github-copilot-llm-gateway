/**
 * Build the `LanguageModelChatInformation` object that VS Code's model picker
 * renders. Kept as its own module so the picker-facing shape (especially the
 * first-party-style `detail`/`multiplierNumeric` fields) can be unit-tested
 * without standing up the full provider.
 */

import { OpenAIModel } from './types';
import { describeModel, friendlyModelName, inferModelFamily } from './modelDisplay';
import { TOKEN_CONSTANTS } from './tokenBudget';

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

export interface ModelCapabilities {
  readonly imageInput?: boolean;
  readonly toolCalling?: boolean | number;
}

export interface BuildModelInfoInput {
  readonly model: OpenAIModel;
  readonly defaultMaxTokens: number;
  readonly defaultMaxOutputTokens: number;
  readonly capabilities: ModelCapabilities;
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
  };
  readonly totalContext: number;
  readonly hasServerReportedContext: boolean;
}

/**
 * Translate a raw `/v1/models` entry into the picker-facing model info plus
 * the resolved total context.
 *
 * `maxInputTokens` is the usable prompt budget after reserving the advertised
 * output budget and a small safety buffer. The full context window is returned
 * separately as `totalContext` for request-time budgeting and status display.
 */
export function buildModelInfo({
  model,
  defaultMaxTokens,
  defaultMaxOutputTokens,
  capabilities,
}: BuildModelInfoInput): BuildModelInfoResult {
  const serverContext = model.max_input_tokens ?? model.max_model_len ?? model.context_length ?? model.context_window;
  const totalContext = serverContext ?? defaultMaxTokens;
  const maxOutputTokens = Math.min(
    defaultMaxOutputTokens,
    Math.max(
      TOKEN_CONSTANTS.MIN_OUTPUT_TOKENS,
      totalContext - TOKEN_CONSTANTS.ADJUST_TOKEN_BUFFER
    )
  );
  const maxInputTokens = Math.max(
    TOKEN_CONSTANTS.MIN_OUTPUT_TOKENS,
    totalContext - maxOutputTokens - TOKEN_CONSTANTS.ADJUST_TOKEN_BUFFER
  );

  const description = describeModel(model);
  const tooltip = description ? `${model.id} — ${description}` : model.id;
  const friendlyName = friendlyModelName(model.id);

  const info: BuildModelInfoResult['info'] = {
    id: model.id,
    name: friendlyName,
    family: inferModelFamily(model.id),
    version: friendlyName,
    maxInputTokens,
    maxOutputTokens,
    capabilities,
    detail: PROVIDER_DETAIL_LABEL,
    tooltip,
    isUserSelectable: true,
    multiplierNumeric: PROVIDER_MULTIPLIER_NUMERIC,
    ...(description ? { description } : {}),
  };

  return {
    info,
    totalContext,
    hasServerReportedContext: serverContext !== undefined,
  };
}
