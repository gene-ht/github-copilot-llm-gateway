import * as vscode from 'vscode';
import { GatewayClient } from './client';
import { GatewayConfig, OpenAIChatCompletionRequest, OpenAIMessage } from './types';
import {
  convertMessage,
  NormalizedMessage,
  NormalizedPart,
  NormalizedRole,
  stripFakeToolCallText,
  containsFakeToolCallText,
} from './messageConverter';
import { parseFakeToolCalls } from './fakeToolCallParser';
import {
  TOKEN_CONSTANTS,
  buildInputText,
  calculateMaxInputTokens,
  calculateSafeMaxOutputTokens,
  estimateTextTokens,
  mergeConsecutiveSameRoleMessages,
  repairToolCallPairing,
  truncateMessagesToFit,
} from './tokenBudget';
import { tryRepairJson } from './jsonRepair';
import { fillMissingRequiredProperties } from './toolSchema';
import { buildChatRequest, OpenAIToolDefinition, ToolChoice } from './requestBuilder';
import {
  StreamChunk,
  StreamReporter,
  isEmptyStreamResult,
  streamResponse,
} from './responseStreamer';
import { dedupeModels, friendlyModelName } from './modelDisplay';
import { buildModelInfo } from './modelInfoBuilder';
import { TokenUsage, extractHost } from './statusBarController';
import {
  SessionStats,
  accumulateUsage,
  emptySessionStats,
  recordRequest,
} from './sessionStats';
import {
  ConnectionState,
  ModelSummary,
  StatusSnapshot,
  formatCapabilityLabels,
  formatContextLabel,
} from './statusSnapshot';
import {
  FrameworkConfigOverride,
  readFrameworkConfiguration,
  resolveApiKey,
} from './frameworkConfig';
import { diagnoseModelFetchError } from './errorDiagnostics';
import {
  ConfigurationTarget as SecretConfigurationTarget,
  LegacyConfigAccessor,
  SECRET_KEYS,
  formatMigrationToast,
  migrateLegacySecrets,
  parseCustomHeadersJson,
} from './secretMigration';

const DEFAULT_REQUEST_TIMEOUT_MS = 60000;
const DEFAULT_TEMPERATURE = 0.7;
const DEBUG_REQUEST_MAX_LOG_LENGTH = 2000;
const MAX_TOOL_ARGS_LOG_LENGTH = 1000;
const MAX_TOOL_DESCRIPTION_LOG_LENGTH = 100;

/**
 * MIME type VS Code 1.120 watches for on `LanguageModelDataPart`s to extract
 * BYOK / language-model-provider token usage and feed it into the chat
 * context-window widget. See microsoft/vscode#315394.
 */
const USAGE_DATA_PART_MIME_TYPE = 'usage';

/**
 * Lifecycle event the status bar (and any other listener) consumes to render
 * live request state. Exactly one terminal event (`complete` or `error`)
 * follows every `start` event for the same request.
 */
export type RequestStateEvent =
  | { readonly kind: 'start'; readonly modelId: string; readonly modelName: string }
  | {
      readonly kind: 'complete';
      readonly modelId: string;
      readonly modelName: string;
      readonly usage?: TokenUsage;
    }
  | {
      readonly kind: 'error';
      readonly modelId: string;
      readonly modelName: string;
      readonly errorMessage: string;
    };

/**
 * Format a tool's description for the output channel: trim, truncate at
 * MAX_TOOL_DESCRIPTION_LOG_LENGTH characters, and only append `...` when an
 * actual truncation happened. Returns `'(none)'` when the tool didn't supply
 * a description at all.
 */
function formatToolDescription(description: string | undefined): string {
  if (!description) { return '(none)'; }
  if (description.length <= MAX_TOOL_DESCRIPTION_LOG_LENGTH) { return description; }
  return `${description.substring(0, MAX_TOOL_DESCRIPTION_LOG_LENGTH)}...`;
}

/**
 * Map a `LanguageModelChatToolMode` enum value to a human-readable label for
 * the output channel. The enum is numeric at runtime, so the raw `${toolMode}`
 * was rendering as `0` / `1` and looked like a stray index.
 */
function describeToolMode(toolMode: vscode.LanguageModelChatToolMode | undefined): string {
  if (toolMode === undefined) { return 'unset'; }
  if (toolMode === vscode.LanguageModelChatToolMode.Required) { return 'required'; }
  if (toolMode === vscode.LanguageModelChatToolMode.Auto) { return 'auto'; }
  return String(toolMode);
}

/**
 * Setting keys that change the shape of the model list returned from the
 * server (and so require VS Code to re-request it). Other keys like
 * `agentTemperature` don't, so we don't want to fire the change event for
 * them — otherwise every keystroke in the settings UI triggers a re-fetch.
 *
 * `apiKey` and `customHeaders` are intentionally listed here for the
 * backward-compat path: if a user re-adds a legacy value to settings.json,
 * the config-change handler triggers a re-migration into SecretStorage and a
 * model refresh (issue #28). Those settings are deprecated for direct use.
 */
const MODEL_AFFECTING_KEYS: readonly string[] = [
  'github.copilot.llm-gateway.serverUrl',
  'github.copilot.llm-gateway.apiKey',
  'github.copilot.llm-gateway.requestTimeout',
  'github.copilot.llm-gateway.defaultMaxTokens',
  'github.copilot.llm-gateway.defaultMaxOutputTokens',
  'github.copilot.llm-gateway.enableImageInput',
  'github.copilot.llm-gateway.enableToolCalling',
  'github.copilot.llm-gateway.customHeaders',
];

/**
 * Legacy plain-text settings that we still watch on the config-change event
 * so a user manually re-adding them in `settings.json` gets re-migrated into
 * SecretStorage instead of silently sitting in plain text (issue #28).
 */
const LEGACY_SECRET_KEYS: readonly string[] = [
  'github.copilot.llm-gateway.apiKey',
  'github.copilot.llm-gateway.customHeaders',
];

/**
 * Language model provider for OpenAI-compatible inference servers.
 *
 * This class is the VS Code surface area; most of the logic lives in focused
 * pure modules (messageConverter, tokenBudget, responseStreamer, etc.) which
 * are unit-tested independently.
 */
export class GatewayProvider implements vscode.LanguageModelChatProvider {
  private readonly client: GatewayClient;
  private config: GatewayConfig;
  private readonly outputChannel: vscode.OutputChannel;
  private readonly secrets: vscode.SecretStorage;
  /**
   * Snapshot of secret values read from `vscode.ExtensionContext.secrets`.
   * `loadConfig` is synchronous (called from the constructor and every
   * config-change event), so we cache the secret values here and refresh the
   * cache via `loadSecrets` / `setApiKey` / `setCustomHeaders` instead of
   * hitting SecretStorage on every read.
   */
  private secretCache: { apiKey: string; customHeaders: Record<string, string> } = {
    apiKey: '',
    customHeaders: {},
  };
  /**
   * Latest API-key override supplied by VS Code's framework-managed
   * `configuration` schema (the `chatProvider@4` proposed API used by native
   * BYOK providers). Wins over the SecretStorage cache when set so users can
   * manage credentials from the native model-picker UI without going through
   * our bespoke `Configure Server` command. Empty values are meaningful —
   * a user clearing the field in the native UI should override any stale
   * SecretStorage entry.
   */
  private frameworkOverride: FrameworkConfigOverride = {};
  /**
   * Real server-reported context per model id (`max_input_tokens` / etc.).
   * Needed because the picker-facing `maxInputTokens` is only the usable input
   * budget after reserving output tokens, not the full context window.
   */
  private readonly contextByModelId: Map<string, number> = new Map();
  /**
   * In-flight model-fetch promise + its completion timestamp. Shared between
   * `provideLanguageModelChatInformation` (called by VS Code's picker) and
   * the status-bar probe, so rapid-fire calls don't stack HTTP requests
   * against the inference server.
   */
  private modelFetchInFlight?: Promise<vscode.LanguageModelChatInformation[]>;
  private modelFetchLast?: { at: number; result: vscode.LanguageModelChatInformation[] };
  /** Tracks the last values we warned about, to avoid notification spam on each keystroke in the settings UI. */
  private lastInvalidUrlNotified?: string;
  private lastOutputTokenAdjustmentNotified?: { output: number; total: number };

  private readonly _onDidChangeLanguageModelChatInformation = new vscode.EventEmitter<void>();
  readonly onDidChangeLanguageModelChatInformation = this._onDidChangeLanguageModelChatInformation.event;

  /**
   * Fired around each chat request so the status bar (or other listeners) can
   * surface live request state. `complete` may carry the usage frame the
   * inference server reported; `error` carries the message that failed the
   * request — both kinds always fire eventually, never both for the same call.
   */
  private readonly _onDidChangeRequestState = new vscode.EventEmitter<RequestStateEvent>();
  readonly onDidChangeRequestState = this._onDidChangeRequestState.event;

  /**
   * Fired whenever the data behind `getStatusSnapshot()` changes (model
   * fetch outcome, request completion, session totals). The status dialog
   * subscribes to this to live-refresh while it's open.
   */
  private readonly _onDidChangeStatusSnapshot = new vscode.EventEmitter<void>();
  readonly onDidChangeStatusSnapshot = this._onDidChangeStatusSnapshot.event;

  private sessionStats: SessionStats = emptySessionStats();
  private lastRequest?: {
    modelId: string;
    modelName: string;
    completedAt: number;
    usage?: TokenUsage;
  };
  private lastSuccessfulFetchAt?: number;
  private lastConnectionError?: string;

  constructor(context: vscode.ExtensionContext) {
    this.outputChannel = vscode.window.createOutputChannel('GitHub Copilot LLM Gateway');
    this.secrets = context.secrets;
    this.config = this.loadConfig();
    this.client = new GatewayClient(this.config, (msg) => this.outputChannel.appendLine(msg));

    context.subscriptions.push(
      this.outputChannel,
      this._onDidChangeLanguageModelChatInformation,
      this._onDidChangeRequestState,
      this._onDidChangeStatusSnapshot,
      vscode.workspace.onDidChangeConfiguration((e: vscode.ConfigurationChangeEvent) => {
        if (!e.affectsConfiguration('github.copilot.llm-gateway')) {
          return;
        }
        this.outputChannel.appendLine('Configuration changed, reloading...');
        // If a deprecated legacy secret setting just gained a value (manually
        // typed into settings.json or pasted via the settings UI), pull it
        // back into SecretStorage and clear the plain-text copy. Errors are
        // logged but never thrown — the config-change listener can't be async.
        if (LEGACY_SECRET_KEYS.some((key) => e.affectsConfiguration(key))) {
          void this.reMigrateLegacySecrets();
        }
        this.reloadConfig();
        // Only nudge VS Code to refetch models when a setting that actually
        // affects the model list has changed.
        const affectsModels = MODEL_AFFECTING_KEYS.some((key) => e.affectsConfiguration(key));
        if (affectsModels) {
          this._onDidChangeLanguageModelChatInformation.fire();
        }
      }),
      this.secrets.onDidChange((e: vscode.SecretStorageChangeEvent) => {
        if (e.key !== SECRET_KEYS.apiKey && e.key !== SECRET_KEYS.customHeaders) {
          return;
        }
        // Another VS Code window (or our own setApiKey) updated a secret —
        // refresh the cache + config so subsequent requests use the new
        // values. Errors here would silently produce stale credentials, so
        // we surface them in the output channel.
        void this.refreshSecretCache().catch((err: unknown) => {
          this.outputChannel.appendLine(
            `Failed to refresh secret cache after change: ${err instanceof Error ? err.message : String(err)}`
          );
        });
      })
    );
  }

  /**
   * Called from `extension.activate` after construction so the first chat
   * request uses the right credentials. Performs one-time migration of legacy
   * plain-text settings into SecretStorage and surfaces a single toast if
   * anything was actually moved.
   */
  public async loadSecrets(): Promise<void> {
    const config = this.legacyConfigAccessor();
    try {
      const result = await migrateLegacySecrets(config, this.secrets, (m) =>
        this.outputChannel.appendLine(m)
      );
      const toast = formatMigrationToast(result);
      if (toast) {
        vscode.window.showInformationMessage(toast);
      }
    } catch (error) {
      this.outputChannel.appendLine(
        `Failed to migrate legacy secrets: ${error instanceof Error ? error.message : String(error)}`
      );
    }
    await this.refreshSecretCache();
  }

  /**
   * Persist a new API key in SecretStorage and refresh the cache. Pass `''`
   * to clear the stored key. Called from the Configure Server command.
   */
  public async setApiKey(apiKey: string): Promise<void> {
    const trimmed = apiKey.trim();
    if (trimmed.length === 0) {
      await this.secrets.delete(SECRET_KEYS.apiKey);
    } else {
      await this.secrets.store(SECRET_KEYS.apiKey, trimmed);
    }
    // `onDidChange` will repopulate the cache, but we also refresh
    // synchronously so callers can immediately use the new value.
    await this.refreshSecretCache();
  }

  /**
   * Persist a new customHeaders map in SecretStorage and refresh the cache.
   * Pass `{}` to clear the stored headers. Called from the Edit Custom
   * Headers command.
   */
  public async setCustomHeaders(headers: Record<string, string>): Promise<void> {
    if (Object.keys(headers).length === 0) {
      await this.secrets.delete(SECRET_KEYS.customHeaders);
    } else {
      await this.secrets.store(SECRET_KEYS.customHeaders, JSON.stringify(headers));
    }
    await this.refreshSecretCache();
  }

  /** Snapshot of the cached custom headers — used by the Edit flow. */
  public getCustomHeadersSnapshot(): Record<string, string> {
    return { ...this.secretCache.customHeaders };
  }

  private async refreshSecretCache(): Promise<void> {
    const apiKey = await this.secrets.get(SECRET_KEYS.apiKey);
    const headersJson = await this.secrets.get(SECRET_KEYS.customHeaders);
    this.secretCache = {
      apiKey: apiKey ?? '',
      customHeaders: parseCustomHeadersJson(headersJson, (m) =>
        this.outputChannel.appendLine(m)
      ),
    };
    this.reloadConfig();
  }

  private async reMigrateLegacySecrets(): Promise<void> {
    try {
      const result = await migrateLegacySecrets(
        this.legacyConfigAccessor(),
        this.secrets,
        (m) => this.outputChannel.appendLine(m)
      );
      if (result.apiKeyMigrated || result.customHeadersMigrated) {
        // No toast on the re-migration path — the user is actively editing
        // settings and a popup mid-keystroke is jarring. The output channel
        // line is enough for diagnostics.
        await this.refreshSecretCache();
      }
    } catch (error) {
      this.outputChannel.appendLine(
        `Failed to re-migrate legacy secret setting: ${error instanceof Error ? error.message : String(error)}`
      );
    }
  }

  /**
   * Adapter from `vscode.WorkspaceConfiguration` to the
   * `LegacyConfigAccessor` interface the migration helpers expect. Done as a
   * small wrapper so the migration logic can be unit-tested without `vscode`.
   */
  private legacyConfigAccessor(): LegacyConfigAccessor {
    const config = vscode.workspace.getConfiguration('github.copilot.llm-gateway');
    return {
      get: <T>(section: string, defaultValue: T): T => config.get<T>(section, defaultValue),
      inspect: <T>(section: string) => {
        const inspection = config.inspect<T>(section);
        if (!inspection) { return undefined; }
        return {
          workspaceValue: inspection.workspaceValue,
          globalValue: inspection.globalValue,
        };
      },
      update: async (section: string, value: unknown, target: SecretConfigurationTarget) => {
        const vsTarget =
          target === SecretConfigurationTarget.Workspace
            ? vscode.ConfigurationTarget.Workspace
            : vscode.ConfigurationTarget.Global;
        await config.update(section, value, vsTarget);
      },
    };
  }

  /**
   * Force a refresh of the language model list. Called from the
   * `Refresh Models` command so users can re-probe the server without
   * editing settings.
   */
  public refreshModels(): void {
    this._onDidChangeLanguageModelChatInformation.fire();
  }

  /**
   * Invalidate the in-memory model-fetch cache so the next call re-probes
   * the server. Called from the `Refresh Models` command.
   */
  public invalidateModelCache(): void {
    this.modelFetchLast = undefined;
  }

  /**
   * Provide language model information - fetches available models from
   * inference server. Multiple concurrent callers share a single HTTP
   * request; successful results are cached for a short window so the picker
   * and the status bar don't double-probe.
   */
  async provideLanguageModelChatInformation(
    options: { silent: boolean; configuration?: { readonly [key: string]: unknown } },
    token: vscode.CancellationToken
  ): Promise<vscode.LanguageModelChatInformation[]> {
    // Pick up any framework-managed configuration (e.g. an apiKey entered via
    // VS Code's native model-picker UI). Only mutates state when the
    // configuration actually changed so we don't churn the cache on every
    // picker open.
    this.applyFrameworkConfiguration(options.configuration);

    const outcome = await this.getOrFetchModels(token);
    if (!options.silent && outcome.error) {
      this.promptOpenSettings(
        `GitHub Copilot LLM Gateway: Failed to fetch models. ${diagnoseModelFetchError(outcome.error)}`
      );
    }
    return outcome.models;
  }

  /**
   * Merge a framework-supplied configuration into the in-memory override.
   * Only forwards `apiKey` for now — `serverUrl` stays in workspace settings
   * so the per-window scope picker (issue #23) keeps working. An explicit
   * empty string is preserved as a "no key" override; a missing/non-string
   * `apiKey` leaves the previous override untouched (defensive — the framework
   * may simply not pass `configuration` on every call).
   */
  private applyFrameworkConfiguration(
    configuration: { readonly [key: string]: unknown } | undefined
  ): void {
    const next = readFrameworkConfiguration(configuration);
    if (next.apiKey === undefined) {
      return;
    }
    if (next.apiKey === this.frameworkOverride.apiKey) {
      return;
    }
    this.frameworkOverride.apiKey = next.apiKey;
    this.outputChannel.appendLine(
      'API key updated from VS Code framework configuration; reloading.'
    );
    this.invalidateModelCache();
    this.reloadConfig();
  }

  /**
   * Underlying model-fetch with cache + single-flight dedup. Never shows any
   * UI itself — that decision belongs to the caller based on its `silent` flag.
   */
  private async getOrFetchModels(
    token: vscode.CancellationToken
  ): Promise<{ models: vscode.LanguageModelChatInformation[]; error?: string }> {
    const now = Date.now();
    const cacheTtlMs = 1000;
    if (this.modelFetchLast && now - this.modelFetchLast.at < cacheTtlMs) {
      return { models: this.modelFetchLast.result };
    }
    if (this.modelFetchInFlight) {
      try {
        return { models: await this.modelFetchInFlight };
      } catch (error) {
        return { models: [], error: error instanceof Error ? error.message : String(error) };
      }
    }

    const inFlight = this.doFetchModels(token);
    this.modelFetchInFlight = inFlight;
    try {
      const result = await inFlight;
      // Don't poison the cache with cancelled-empty results — the next caller
      // should re-probe instead of seeing a stale empty list.
      if (!token.isCancellationRequested) {
        this.modelFetchLast = { at: Date.now(), result };
        this.lastSuccessfulFetchAt = Date.now();
        this.lastConnectionError = undefined;
        this._onDidChangeStatusSnapshot.fire();
      }
      return { models: result };
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      this.lastConnectionError = message;
      this._onDidChangeStatusSnapshot.fire();
      return { models: [], error: message };
    } finally {
      if (this.modelFetchInFlight === inFlight) {
        this.modelFetchInFlight = undefined;
      }
    }
  }

  private async doFetchModels(
    token: vscode.CancellationToken
  ): Promise<vscode.LanguageModelChatInformation[]> {
    this.outputChannel.appendLine('Fetching models from inference server...');
    let response;
    try {
      response = await this.client.fetchModels(token);
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : String(error);
      this.outputChannel.appendLine(`ERROR: Failed to fetch models: ${errorMessage}`);
      throw error;
    }

    if (token.isCancellationRequested) {
      return [];
    }

    const uniqueModels = dedupeModels(response.data);
    if (uniqueModels.length !== response.data.length) {
      this.outputChannel.appendLine(
        `Server returned ${response.data.length} models, ${uniqueModels.length} unique after dedupe`
      );
    }

    // Rebuild the per-id context map from the latest fetch. If the server
    // removed a model, drop its entry so stale data can't leak into future
    // chat requests.
    this.contextByModelId.clear();

    const models = uniqueModels.map((model) => {
      const { info, totalContext, hasServerReportedContext } = buildModelInfo({
        model,
        defaultMaxTokens: this.config.defaultMaxTokens,
        defaultMaxOutputTokens: this.config.defaultMaxOutputTokens,
        capabilities: {
          imageInput: this.config.enableImageInput,
          toolCalling: this.config.enableToolCalling,
        },
      });
      this.contextByModelId.set(model.id, totalContext);

      if (hasServerReportedContext) {
        this.outputChannel.appendLine(
          `  Model ${model.id}: server-reported context ${totalContext} tokens (exposed as input=${info.maxInputTokens}, output=${info.maxOutputTokens})`
        );
      }

      return info;
    });

    this.outputChannel.appendLine(
      `Found ${models.length} models: ${models.map((m) => m.id).join(', ')}`
    );
    return models;
  }

  /**
   * Provide language model chat response - streams responses from inference server
   */
  async provideLanguageModelChatResponse(
    model: vscode.LanguageModelChatInformation,
    messages: readonly vscode.LanguageModelChatMessage[],
    options: vscode.ProvideLanguageModelChatResponseOptions,
    progress: vscode.Progress<vscode.LanguageModelResponsePart>,
    token: vscode.CancellationToken
  ): Promise<void> {
    if (this.config.verboseLogging) {
      this.outputChannel.appendLine(`Sending chat request to model: ${model.id}`);
      this.outputChannel.appendLine(
        `Tool mode: ${describeToolMode(options.toolMode)}, Tools: ${options.tools?.length ?? 0}`
      );
      this.outputChannel.appendLine(`Message count: ${messages.length}`);
    }

    const modelName = friendlyModelName(model.id);
    this._onDidChangeRequestState.fire({ kind: 'start', modelId: model.id, modelName });

    let openAIMessages = this.convertAllMessages(messages);

    // Always-on detection: scan for Copilot Chat's cross-session injection
    // markers. Logs a one-line summary when triggered so users can correlate
    // "陌生 TODO 出现"等现象 with the actual injected content. Independent of
    // verboseLogging so it captures real-world incidents without setup.
    this.detectCrossSessionInjection(openAIMessages, messages.length);

    if (this.config.stripFakeToolCallText) {
      openAIMessages = stripFakeToolCallText(openAIMessages, (msg) =>
        this.outputChannel.appendLine(msg)
      );
    }
    if (this.config.verboseLogging) {
      this.outputChannel.appendLine(`Converted to ${openAIMessages.length} OpenAI messages`);
      this.logMessageStructure(openAIMessages);
    }

    const modelMaxContext = this.resolveModelMaxContext(model);
    const configuredMaxOutput =
      model.maxOutputTokens || TOKEN_CONSTANTS.DEFAULT_OUTPUT_TOKENS;
    const toolsSerializedLength = options.tools ? JSON.stringify(options.tools).length : 0;

    const maxInputTokens = calculateMaxInputTokens({
      modelMaxContext,
      configuredMaxOutput,
      toolsSerializedLength,
    });

    const verboseLog = this.config.verboseLogging
      ? (msg: string) => this.outputChannel.appendLine(msg)
      : () => { /* no-op */ };
    const fittedMessages = truncateMessagesToFit(openAIMessages, maxInputTokens, verboseLog);
    if (fittedMessages.length < openAIMessages.length) {
      this.outputChannel.appendLine(
        `WARNING: Truncated conversation from ${openAIMessages.length} to ${fittedMessages.length} messages to fit context limit`
      );
    }
    // Truncation slices messages by token budget alone and can leave orphan
    // `role: tool` blocks (or dangling assistant tool_calls) if the matching
    // counterpart fell off the start of the window. Anthropic-backed gateways
    // (e.g. Vertex AI Claude) reject such payloads with a 400
    // "unexpected tool_use_id found in tool_result blocks" error. Strip those
    // before sending so the same conversation that survives an OpenAI backend
    // also survives an Anthropic one.
    // Log first few messages' structure for debugging tool pairing issues
    if (this.config.verboseLogging) {
      const preRepairSummary = fittedMessages.slice(0, 8).map((m, i) => {
        const r = (m as Record<string, unknown>).role;
        const tcId = (m as Record<string, unknown>).tool_call_id;
        const hasTc = Array.isArray((m as Record<string, unknown>).tool_calls);
        return `[${i}] role=${r}${tcId ? ` tool_call_id=${tcId}` : ''}${hasTc ? ' has_tool_calls' : ''}`;
      });
      this.outputChannel.appendLine(`Pre-repair first 8: ${preRepairSummary.join(', ')}`);
    }

    const truncatedMessages = repairToolCallPairing(fittedMessages, verboseLog);
    if (truncatedMessages.length !== fittedMessages.length) {
      this.outputChannel.appendLine(
        `Tool pairing: ${fittedMessages.length} → ${truncatedMessages.length} messages after repair`
      );
    } else if (this.config.verboseLogging) {
      this.outputChannel.appendLine(
        `Tool pairing: all ${fittedMessages.length} messages passed (no orphans detected)`
      );
    }

    // Merge consecutive same-role user messages so the Anthropic gateway's
    // OpenAI→Anthropic converter doesn't shift message boundaries and
    // misalign tool_use/tool_result pairing.
    const finalMessages = mergeConsecutiveSameRoleMessages(truncatedMessages, verboseLog);

    const inputText = buildInputText(finalMessages);
    const toolsOverhead = Math.ceil(toolsSerializedLength / TOKEN_CONSTANTS.CHARS_PER_TOKEN);
    const estimatedInputTokens = await this.provideTokenCount(model, inputText, token);
    const safeMaxOutputTokens = calculateSafeMaxOutputTokens({
      estimatedInputTokens,
      toolsOverhead,
      modelMaxContext,
      configuredMaxOutput,
    });

    if (this.config.verboseLogging) {
      this.outputChannel.appendLine(
        `Token estimate: input=${estimatedInputTokens}, tools=${toolsOverhead}, model_context=${modelMaxContext}, chosen_max_tokens=${safeMaxOutputTokens}`
      );
    }

    const { tools, schemas: toolSchemas } = this.buildToolsConfig(options);
    const hasTools = tools !== undefined && tools.length > 0;
    const temperature = hasTools ? this.config.agentTemperature : DEFAULT_TEMPERATURE;

    // Always-on log: dump the effective per-model settings + final max_tokens
    // so users can verify their `perModelSettings.<modelId>.max_tokens` override
    // is actually being applied at request time. Independent of verboseLogging
    // because this is critical for "my setting didn't take effect" debugging.
    // VS Code returns configuration values as Proxy/getter-backed objects, not
    // plain objects. Spreading them via `{...src}` enumerates own keys and
    // invokes each getter; if any getter throws (e.g. a key like
    // `reasoning_effort` whose backing config is undefined), the spread
    // crashes the entire request. Use a safe shallow-clone that:
    //   1) accepts any object-like input,
    //   2) iterates own enumerable keys,
    //   3) skips keys whose getters throw,
    //   4) drops `undefined` values to avoid `JSON.stringify` clutter.
    const safeShallowClone = (src: unknown, label: string): Record<string, unknown> => {
      const out: Record<string, unknown> = {};
      if (!src || typeof src !== 'object') { return out; }
      let keys: string[];
      try {
        keys = Object.keys(src as Record<string, unknown>);
      } catch (err) {
        this.outputChannel.appendLine(
          `[Settings] WARNING: Object.keys(${label}) threw: ${err instanceof Error ? err.message : String(err)}`
        );
        return out;
      }
      for (const key of keys) {
        try {
          const value = (src as Record<string, unknown>)[key];
          if (value !== undefined) {
            out[key] = value;
          }
        } catch (err) {
          this.outputChannel.appendLine(
            `[Settings] WARNING: skipped ${label}.${key} — getter threw: ${err instanceof Error ? err.message : String(err)}`
          );
        }
      }
      return out;
    };

    const perModelOverride = safeShallowClone(
      this.config.perModelSettings[model.id],
      `perModelSettings[${model.id}]`
    );
    const callerOverride = safeShallowClone(options.modelOptions, 'options.modelOptions');
    const extraModelOptions = safeShallowClone(this.config.extraModelOptions, 'extraModelOptions');
    this.outputChannel.appendLine(
      `[Settings] model=${model.id} ` +
      `safeMaxOutputTokens=${safeMaxOutputTokens} ` +
      `perModel.keys=[${Object.keys(perModelOverride).join(',') || '(none)'}] ` +
      `caller.keys=[${Object.keys(callerOverride).join(',') || '(none)'}]`
    );
    if (Object.keys(perModelOverride).length > 0) {
      this.outputChannel.appendLine(
        `[Settings] perModelSettings[${model.id}] = ${JSON.stringify(perModelOverride)}`
      );
    }

    const requestOptions = buildChatRequest({
      model: model.id,
      messages: finalMessages,
      maxTokens: safeMaxOutputTokens,
      temperature,
      tools,
      toolChoice: hasTools ? this.mapToolChoice(options.toolMode) : undefined,
      parallelToolCalls: hasTools ? this.config.parallelToolCalling : undefined,
      extraOptions: {
        ...extraModelOptions,
        ...perModelOverride,
        ...callerOverride,
      },
    });

    // Confirm the final values that will be sent on the wire.
    this.outputChannel.appendLine(
      `[Settings] final request: max_tokens=${requestOptions.max_tokens} ` +
      `temperature=${requestOptions.temperature}` +
      ((requestOptions as Record<string, unknown>).reasoning_effort !== undefined
        ? ` reasoning_effort=${(requestOptions as Record<string, unknown>).reasoning_effort}`
        : '')
    );

    if (hasTools && this.config.verboseLogging) {
      this.outputChannel.appendLine(
        `Sending ${tools.length} tools to model (parallel: ${this.config.parallelToolCalling})`
      );
    }

    if (this.config.verboseLogging) {
      this.logRequest(requestOptions);
    }

    let capturedUsage: TokenUsage | undefined;
    try {
      const reporter = this.createStreamReporter(progress, (usage) => {
        capturedUsage = usage;
      });
      const chunks = this.client.streamChatCompletion(requestOptions, token);
      const stats = await streamResponse({
        chunks: chunks as AsyncIterable<StreamChunk>,
        reporter,
        isCancelled: () => token.isCancellationRequested,
        resolveToolCallArgs: (toolCall) => this.resolveToolCallArgs(toolCall, toolSchemas),
        captureContent: hasTools && this.config.retryFakeToolCalls,
      });

      if (this.config.verboseLogging) {
        this.outputChannel.appendLine(
          `Completed chat request, received ${stats.totalContentLength} chars, ${stats.totalTextParts} text parts, ${stats.totalToolCalls} tool calls`
        );
      }

      // Detect self-poisoning: model wrote tool calls as plain text instead of
      // using the structured tool_calls mechanism. Recovery strategy:
      //   1. PARSE the text — if we can extract structured tool calls from
      //      the "Completed tool calls:" block, emit them directly. This is
      //      a synthetic reconstruction; it works even when the upstream
      //      rejects `tool_choice="required"` (e.g. GitHub Copilot's backend
      //      caps Required mode to a single tool).
      //   2. If parsing fails (malformed JSON, unrecognized format), fall
      //      back to a retry with a strong correction prompt. We don't use
      //      `tool_choice="required"` because it's not portable across
      //      OpenAI-compatible backends.
      if (
        this.config.retryFakeToolCalls &&
        hasTools &&
        stats.totalToolCalls === 0 &&
        stats.capturedContent &&
        containsFakeToolCallText(stats.capturedContent) &&
        !token.isCancellationRequested
      ) {
        // -------- Step 1: try to reconstruct tool_calls from text --------
        const parsed = parseFakeToolCalls(stats.capturedContent);
        if (parsed.length > 0) {
          this.outputChannel.appendLine(
            `WARNING: Model wrote tool calls as plain text. ` +
            `Parsed ${parsed.length} tool call(s) from text; emitting as structured tool_calls.`
          );
          for (const tc of parsed) {
            const args = this.resolveToolCallArgs(
              { id: tc.id, name: tc.name, arguments: tc.arguments },
              toolSchemas
            );
            if (this.config.verboseLogging) {
              this.outputChannel.appendLine(
                `  Synthesized tool call: ${tc.name} (id=${tc.id})`
              );
            }
            reporter.reportToolCall(tc.id, tc.name, args);
            stats.totalToolCalls++;
          }
          this.outputChannel.appendLine(
            `Synthesized ${parsed.length} tool call(s) from text-mode output ✓`
          );
        } else {
          // -------- Step 2: parsing failed → retry with correction --------
          this.outputChannel.appendLine(
            `WARNING: Model wrote tool calls as plain text (${stats.totalContentLength} chars, 0 real tool_calls) ` +
            `but parsing failed. Falling back to retry with correction prompt.`
          );

          const correctionSystem: OpenAIMessage = {
            role: 'system',
            content:
              'CRITICAL: You must use the function-calling (tool_calls) mechanism to invoke tools. ' +
              'NEVER write tool calls as plain text, markdown, or any human-readable format. ' +
              'The user CANNOT execute text descriptions of tool calls — only structured tool_calls ' +
              'are actually executed. If you need to call a tool, emit a real tool_call; do not narrate it.',
          };
          const retryMessages: OpenAIMessage[] = [
            correctionSystem,
            ...requestOptions.messages,
            { role: 'assistant', content: stats.capturedContent },
            {
              role: 'user',
              content:
                'STOP. You just described tool calls as plain text. Those were NOT executed. ' +
                'Re-invoke the exact same tools you just described, but this time use the function ' +
                'calling mechanism so they actually run. Do not output any text — just call the tools.',
            },
          ];

          // Note: NOT setting tool_choice="required" — GitHub Copilot's
          // backend rejects it when >1 tool is in the request, and most
          // OpenAI-compatible servers vary in support. The prompt + system
          // message is the only portable lever.
          const retryRequest: OpenAIChatCompletionRequest = {
            ...requestOptions,
            messages: retryMessages,
          };

          if (this.config.verboseLogging) {
            this.outputChannel.appendLine(
              `Retry request: ${retryMessages.length} messages (was ${requestOptions.messages.length})`
            );
          }

          const retryChunks = this.client.streamChatCompletion(retryRequest, token);
          const retryStats = await streamResponse({
            chunks: retryChunks as AsyncIterable<StreamChunk>,
            reporter,
            isCancelled: () => token.isCancellationRequested,
            resolveToolCallArgs: (toolCall) => this.resolveToolCallArgs(toolCall, toolSchemas),
          });

          if (retryStats.totalToolCalls > 0) {
            this.outputChannel.appendLine(
              `Retry recovered ${retryStats.totalToolCalls} real tool call(s) ✓`
            );
          } else {
            this.outputChannel.appendLine(
              `WARNING: Retry did not recover tool calls — model may be stuck in text mode. ` +
              `Likely cause: context too large (${requestOptions.messages.length} messages) for reliable tool calling. ` +
              `Consider starting a new chat session.`
            );
          }
        }
      }

      if (isEmptyStreamResult(stats)) {
        const toolCount = tools?.length ?? 0;
        await this.handleEmptyResponse(model, inputText, openAIMessages.length, toolCount, token, progress);
      }
      this.recordCompletedRequest(model.id, modelName, capturedUsage);
      this._onDidChangeRequestState.fire({
        kind: 'complete',
        modelId: model.id,
        modelName,
        usage: capturedUsage,
      });
    } catch (error) {
      this._onDidChangeRequestState.fire({
        kind: 'error',
        modelId: model.id,
        modelName,
        errorMessage: error instanceof Error ? error.message : String(error),
      });
      this.handleChatError(error);
    }
  }

  /**
   * Provide token count estimation (rough char/4 approximation).
   *
   * Non-text parts contribute too: tool calls / tool results are serialized
   * and counted, and each image contributes a conservative fixed overhead so
   * we don't undercount multimodal conversations (otherwise the output-token
   * budget overshoots the real context window).
   */
  async provideTokenCount(
    _model: vscode.LanguageModelChatInformation,
    text: string | vscode.LanguageModelChatMessage,
    _token: vscode.CancellationToken
  ): Promise<number> {
    if (typeof text === 'string') {
      return estimateTextTokens(text);
    }
    let tokens = 0;
    for (const part of text.content) {
      if (part instanceof vscode.LanguageModelTextPart) {
        tokens += estimateTextTokens(part.value);
      } else if (part instanceof vscode.LanguageModelToolCallPart) {
        tokens += estimateTextTokens(part.name + JSON.stringify(part.input ?? {}));
      } else if (part instanceof vscode.LanguageModelToolResultPart) {
        const body = typeof part.content === 'string' ? part.content : JSON.stringify(part.content);
        tokens += estimateTextTokens(body);
      } else if (part instanceof vscode.LanguageModelDataPart) {
        // Images don't map cleanly to tokens — reserve a conservative fixed
        // overhead so multimodal requests aren't massively undercounted.
        tokens += 800;
      }
    }
    return tokens;
  }

  /**
   * Capture a successful request in the running session totals and the
   * `lastRequest` slot the status dialog renders. Failed requests are not
   * counted — the connection-state row already reflects them and a single
   * failure shouldn't pad the request count.
   */
  private recordCompletedRequest(
    modelId: string,
    modelName: string,
    usage: TokenUsage | undefined
  ): void {
    let next = recordRequest(this.sessionStats);
    if (usage) {
      next = accumulateUsage(next, usage);
    }
    this.sessionStats = next;
    this.lastRequest = {
      modelId,
      modelName,
      completedAt: Date.now(),
      ...(usage ? { usage } : {}),
    };
    this._onDidChangeStatusSnapshot.fire();
  }

  /**
   * Open the extension's output channel. Exposed so the status dialog's
   * "Open output log" button can show the panel without the controller
   * having to reach into the provider's internals.
   */
  public showOutput(): void {
    this.outputChannel.show();
  }

  /** Expose the output channel so the proxy server can log alongside the provider. */
  public getOutputChannel(): vscode.OutputChannel {
    return this.outputChannel;
  }

  /**
   * Return the effective API key after resolving framework overrides and
   * SecretStorage cache. Used by the Copilot proxy to authenticate with
   * the upstream server without duplicating the resolution logic.
   */
  public getResolvedApiKey(): string {
    return this.config.apiKey ?? '';
  }

  /**
   * Snapshot of everything the status dialog renders: connection state, the
   * cached model list, running session totals, last request, feature flags.
   * Re-built fresh on every call so relative-time fields ("2m ago") move
   * forward whenever the dialog re-renders.
   */
  public getStatusSnapshot(): StatusSnapshot {
    const cachedModels = this.modelFetchLast?.result ?? [];
    const models: ModelSummary[] = cachedModels.map((m) => {
      const totalContext = this.contextByModelId.get(m.id);
      return {
        id: m.id,
        name: m.name,
        contextLabel: formatContextLabel(totalContext),
        ...(totalContext !== undefined ? { totalContext } : {}),
        capabilityLabels: formatCapabilityLabels(m.capabilities ?? {}),
      };
    });

    let connection: { state: ConnectionState; errorMessage?: string };
    if (this.lastConnectionError) {
      connection = { state: 'error', errorMessage: this.lastConnectionError };
    } else if (this.lastSuccessfulFetchAt === undefined) {
      connection = { state: 'unknown' };
    } else if (cachedModels.length === 0) {
      connection = { state: 'noModels' };
    } else {
      connection = { state: 'ok' };
    }

    return {
      host: extractHost(this.config.serverUrl),
      connection,
      ...(this.lastSuccessfulFetchAt !== undefined
        ? { lastSuccessfulFetchAt: this.lastSuccessfulFetchAt }
        : {}),
      models,
      sessionStats: this.sessionStats,
      ...(this.lastRequest ? { lastRequest: this.lastRequest } : {}),
      features: {
        toolCalling: this.config.enableToolCalling,
        imageInput: this.config.enableImageInput,
        parallelToolCalling: this.config.parallelToolCalling,
        agentTemperature: this.config.agentTemperature,
      },
      now: Date.now(),
    };
  }

  /**
   * Resolve the real server-reported context size for a model. Prefer the
   * per-id context map captured from `/v1/models`; if VS Code routes a cached
   * model before a fresh fetch, reconstruct an approximate total from the
   * picker-facing input/output budgets.
   */
  private resolveModelMaxContext(model: vscode.LanguageModelChatInformation): number {
    const cached = this.contextByModelId.get(model.id);
    if (cached && cached > 0) {
      return cached;
    }
    // Fallback path: the model list hasn't been fetched yet in this session
    // (e.g. VS Code routed a cached chat directly to the provider). Rebuild
    // the approximate total context from the picker-facing input/output budget.
    if (model.maxInputTokens && model.maxInputTokens > 0) {
      return model.maxInputTokens + (model.maxOutputTokens || 0) + TOKEN_CONSTANTS.ADJUST_TOKEN_BUFFER;
    }
    return TOKEN_CONSTANTS.DEFAULT_CONTEXT_TOKENS;
  }

  // ---------- message classification ----------

  private convertAllMessages(messages: readonly vscode.LanguageModelChatMessage[]): OpenAIMessage[] {
    const result: OpenAIMessage[] = [];
    // Per-part `Found tool call/result:` lines are firehose-level detail —
    // gate them behind verboseLogging so the Output channel stays readable
    // unless the user explicitly opts in.
    const log: (msg: string) => void = this.config.verboseLogging
      ? (msg) => this.outputChannel.appendLine(msg)
      : () => { /* no-op */ };
    for (const msg of messages) {
      const normalized: NormalizedMessage = {
        role: this.mapRole(msg.role),
        parts: msg.content.map((part) => this.classifyPart(part)),
      };
      result.push(
        ...convertMessage(normalized, { enableImageInput: this.config.enableImageInput }, log)
      );
    }
    return result;
  }

  private mapRole(role: vscode.LanguageModelChatMessageRole): NormalizedRole {
    if (role === vscode.LanguageModelChatMessageRole.Assistant) {
      return 'assistant';
    }
    return 'user';
  }

  /**
   * Translate a vscode LanguageModel*Part into the plain data shape used by
   * messageConverter. Falls back to duck typing for older VS Code versions
   * where the constructors may not match.
   */
  private classifyPart(part: unknown): NormalizedPart {
    if (part instanceof vscode.LanguageModelTextPart) {
      return { kind: 'text', value: part.value };
    }
    if (part instanceof vscode.LanguageModelToolResultPart) {
      return {
        kind: 'toolResult',
        callId: part.callId,
        content: typeof part.content === 'string' ? part.content : JSON.stringify(part.content),
      };
    }
    if (part instanceof vscode.LanguageModelToolCallPart) {
      return {
        kind: 'toolCall',
        callId: part.callId,
        name: part.name,
        input: part.input,
      };
    }
    if (part instanceof vscode.LanguageModelDataPart) {
      return { kind: 'image', mimeType: part.mimeType, data: part.data };
    }
    return this.classifyPartDuckTyped(part);
  }

  private classifyPartDuckTyped(part: unknown): NormalizedPart {
    if (typeof part !== 'object' || part === null) {
      return { kind: 'unknown' };
    }
    const anyPart = part as Record<string, unknown>;

    if ('callId' in anyPart && 'content' in anyPart && !('name' in anyPart)) {
      if (this.config.verboseLogging) {
        this.outputChannel.appendLine(`  Found tool result (duck-typed): callId=${anyPart.callId}`);
      }
      return {
        kind: 'toolResult',
        callId: String(anyPart.callId),
        content:
          typeof anyPart.content === 'string' ? anyPart.content : JSON.stringify(anyPart.content),
      };
    }
    if ('callId' in anyPart && 'name' in anyPart && 'input' in anyPart) {
      if (this.config.verboseLogging) {
        this.outputChannel.appendLine(
          `  Found tool call (duck-typed): callId=${anyPart.callId}, name=${anyPart.name}`
        );
      }
      return {
        kind: 'toolCall',
        callId: String(anyPart.callId),
        name: String(anyPart.name),
        input: anyPart.input,
      };
    }
    return { kind: 'unknown' };
  }

  // ---------- tool config + stream adapters ----------

  private mapToolChoice(toolMode: vscode.LanguageModelChatToolMode | undefined): ToolChoice | undefined {
    switch (toolMode) {
      case vscode.LanguageModelChatToolMode.Required:
        return 'required';
      case vscode.LanguageModelChatToolMode.Auto:
        return 'auto';
      default:
        return undefined;
    }
  }

  private buildToolsConfig(
    options: vscode.ProvideLanguageModelChatResponseOptions
  ): {
    tools: OpenAIToolDefinition[] | undefined;
    schemas: Map<string, Record<string, unknown> | undefined>;
  } {
    const schemas = new Map<string, Record<string, unknown> | undefined>();
    if (!this.config.enableToolCalling || !options.tools || options.tools.length === 0) {
      return { tools: undefined, schemas };
    }

    const verbose = this.config.verboseLogging;
    const tools: OpenAIToolDefinition[] = options.tools.map((tool) => {
      if (verbose) {
        this.outputChannel.appendLine(`Tool: ${tool.name}`);
        this.outputChannel.appendLine(
          `  Description: ${formatToolDescription(tool.description)}`
        );
      }

      const schema = tool.inputSchema as Record<string, unknown> | undefined;
      schemas.set(tool.name, schema);

      if (verbose && schema?.required && Array.isArray(schema.required)) {
        this.outputChannel.appendLine(
          `  Required properties: ${(schema.required as string[]).join(', ')}`
        );
      }

      return {
        type: 'function',
        function: {
          name: tool.name,
          description: tool.description,
          parameters: tool.inputSchema,
        },
      };
    });

    return { tools, schemas };
  }

  /**
   * Parse and patch tool call arguments before reporting them upstream.
   * The schemas map is per-request so concurrent `provideLanguageModelChatResponse`
   * calls can't clobber each other's tool definitions.
   */
  private resolveToolCallArgs(
    toolCall: { id: string; name: string; arguments: string },
    toolSchemas: Map<string, Record<string, unknown> | undefined>
  ): Record<string, unknown> {
    const verbose = this.config.verboseLogging;
    if (verbose) {
      this.outputChannel.appendLine(`\n=== TOOL CALL RECEIVED ===`);
      this.outputChannel.appendLine(`  ID: ${toolCall.id}`);
      this.outputChannel.appendLine(`  Name: ${toolCall.name}`);
      this.outputChannel.appendLine(
        `  Raw arguments: ${toolCall.arguments.substring(0, MAX_TOOL_ARGS_LOG_LENGTH)}${
          toolCall.arguments.length > MAX_TOOL_ARGS_LOG_LENGTH ? '...' : ''
        }`
      );
    }

    const log = verbose
      ? (msg: string): void => this.outputChannel.appendLine(msg)
      : () => { /* no-op */ };
    let args = tryRepairJson(toolCall.arguments, log) as Record<string, unknown> | null;

    if (args === null) {
      this.outputChannel.appendLine(`  ERROR: Failed to parse tool call arguments for ${toolCall.name}`);
      this.outputChannel.appendLine(`  Full arguments: ${toolCall.arguments}`);
      args = {};
    } else if (verbose) {
      const argKeys = Object.keys(args);
      this.outputChannel.appendLine(
        `  Parsed argument keys: ${argKeys.length > 0 ? argKeys.join(', ') : '(none)'}`
      );
    }

    const toolSchema = toolSchemas.get(toolCall.name);
    if (toolSchema) {
      args = fillMissingRequiredProperties(args, toolSchema, log);
    }

    if (verbose) {
      this.outputChannel.appendLine(`=== END TOOL CALL ===\n`);
    }
    return args;
  }

  private createStreamReporter(
    progress: vscode.Progress<vscode.LanguageModelResponsePart>,
    onUsage?: (usage: TokenUsage) => void
  ): StreamReporter {
    return {
      reportText: (text) => progress.report(new vscode.LanguageModelTextPart(text)),
      reportThinking: (text) => progress.report(new vscode.LanguageModelThinkingPart(text)),
      reportThinkingDone: () =>
        progress.report(new vscode.LanguageModelThinkingPart('', '', { vscode_reasoning_done: true })),
      reportToolCall: (id, name, args) =>
        progress.report(new vscode.LanguageModelToolCallPart(id, name, args)),
      reportUsage: (usage) => {
        // VS Code 1.120 picks up token usage emitted as a LanguageModelDataPart
        // with the literal mime type `usage` (see microsoft/vscode#315394).
        // The shape mirrors OpenAI's `usage` object. Surfacing it here makes
        // the chat view's context-window widget render real numbers instead
        // of `0%` for gateway models (issue #24).
        if (this.config.verboseLogging) {
          this.outputChannel.appendLine(
            `Usage: prompt=${usage.prompt_tokens}, completion=${usage.completion_tokens}, total=${usage.total_tokens}`
          );
        }
        onUsage?.({
          prompt: usage.prompt_tokens,
          completion: usage.completion_tokens,
          total: usage.total_tokens,
        });
        const payload = new TextEncoder().encode(JSON.stringify(usage));
        progress.report(new vscode.LanguageModelDataPart(payload, USAGE_DATA_PART_MIME_TYPE));
      },
    };
  }

  // ---------- logging helpers ----------

  /**
   * Detector for self-poisoning patterns in assistant message content.
   *
   * Specifically: when an assistant message includes a textual representation
   * of "Completed tool calls: - foo (call_xxx) { ... }" (as some Copilot Chat
   * clients do for human-readable replay), it can act as a few-shot example
   * teaching the model to *write* tool calls as text instead of emitting a
   * real tool_calls array. We log when this is detected so users can correlate
   * "model stopped calling tools" with the actual injected text.
   *
   * Runs regardless of `verboseLogging` — detection needs to capture incidents
   * in normal use, not just during debug sessions. Conversation-summary blocks
   * are NOT detected here (they're a normal in-session context compression
   * mechanism, not pollution).
   */
  private detectCrossSessionInjection(
    openAIMessages: readonly OpenAIMessage[],
    vsCodeMessageCount: number
  ): void {
    const markers: Array<{ name: string; regex: RegExp; count: number; firstHit?: { msgIndex: number; snippet: string } }> = [
      { name: 'Completed tool calls (fake-toolcall self-poisoning)', regex: /Completed tool calls:\s*\n[ \t]*-[ \t]+\S+[ \t]*\(call_[0-9a-zA-Z]+\)/g, count: 0 },
    ];

    for (let i = 0; i < openAIMessages.length; i++) {
      const msg = openAIMessages[i];
      const content = (msg as Record<string, unknown>).content;
      const text =
        typeof content === 'string'
          ? content
          : Array.isArray(content)
            ? content
                .map((p) =>
                  typeof p === 'object' && p !== null && typeof (p as Record<string, unknown>).text === 'string'
                    ? (p as Record<string, unknown>).text
                    : ''
                )
                .join('\n')
            : '';
      if (typeof text !== 'string' || text.length === 0) { continue; }

      for (const m of markers) {
        const matches = text.match(m.regex);
        if (matches && matches.length > 0) {
          m.count += matches.length;
          if (!m.firstHit) {
            // Capture a 200-char window around the first occurrence to help diagnose
            const idx = text.search(m.regex);
            const start = Math.max(0, idx - 50);
            const end = Math.min(text.length, idx + 150);
            m.firstHit = {
              msgIndex: i,
              snippet: text.slice(start, end).replace(/\n/g, ' '),
            };
          }
        }
      }
    }

    const triggered = markers.filter((m) => m.count > 0);
    if (triggered.length === 0) { return; }

    this.outputChannel.appendLine(
      `[Self-poisoning detector] Detected fake-toolcall markers in request (${vsCodeMessageCount} messages):`
    );
    for (const m of triggered) {
      this.outputChannel.appendLine(
        `  ${m.name}: ${m.count} occurrence(s), first at message[${m.firstHit?.msgIndex}]`
      );
      if (m.firstHit) {
        this.outputChannel.appendLine(`    snippet: ...${m.firstHit.snippet}...`);
      }
    }
  }

  private logMessageStructure(openAIMessages: readonly OpenAIMessage[]): void {
    for (let i = 0; i < openAIMessages.length; i++) {
      const msg = openAIMessages[i];
      const toolCallId = typeof msg.tool_call_id === 'string' ? msg.tool_call_id : 'none';
      let hasContent: boolean;
      if (typeof msg.content === 'string') {
        hasContent = msg.content.length > 0;
      } else if (Array.isArray(msg.content)) {
        hasContent = msg.content.length > 0;
      } else {
        hasContent = msg.content !== null && msg.content !== undefined;
      }
      const hasToolCalls = Array.isArray(msg.tool_calls) && msg.tool_calls.length > 0;
      this.outputChannel.appendLine(
        `  Message ${i + 1}: role=${msg.role}, hasContent=${hasContent}, hasToolCalls=${hasToolCalls}, toolCallId=${toolCallId}`
      );
    }
  }

  private logRequest(request: OpenAIChatCompletionRequest): void {
    if (!this.config.verboseLogging) {
      // By default log only the non-content envelope so user conversation
      // data (file contents, tool args, credentials pasted into chat) is
      // not captured in logs they may share for support.
      const toolCount = Array.isArray(request.tools) ? request.tools.length : 0;
      this.outputChannel.appendLine(
        `Request: model=${request.model}, messages=${request.messages.length}, tools=${toolCount}, max_tokens=${request.max_tokens}, temperature=${request.temperature}`
      );
      return;
    }
    const debugRequest = JSON.stringify(request, null, 2);
    this.outputChannel.appendLine(
      debugRequest.length > DEBUG_REQUEST_MAX_LOG_LENGTH
        ? `Request (truncated): ${debugRequest.substring(0, DEBUG_REQUEST_MAX_LOG_LENGTH)}...`
        : `Request: ${debugRequest}`
    );
  }

  // ---------- error / UI helpers ----------

  private async handleEmptyResponse(
    model: vscode.LanguageModelChatInformation,
    inputText: string,
    messageCount: number,
    toolCount: number,
    token: vscode.CancellationToken,
    progress: vscode.Progress<vscode.LanguageModelResponsePart>
  ): Promise<void> {
    const inputTokenCount = await this.provideTokenCount(model, inputText, token);
    const modelMaxContext = this.resolveModelMaxContext(model);

    this.outputChannel.appendLine(`WARNING: Model returned empty response with no tool calls.`);
    this.outputChannel.appendLine(`  Input tokens estimated: ${inputTokenCount}`);
    this.outputChannel.appendLine(`  Messages in conversation: ${messageCount}`);
    this.outputChannel.appendLine(`  Tools provided: ${toolCount}`);

    const errorHint =
      toolCount > 0
        ? `The model returned an empty response. This typically indicates the model failed to generate valid output with tool calling enabled. Check the inference server logs for errors.`
        : `The model returned an empty response. Check the inference server logs for details.`;

    this.outputChannel.appendLine(`  Issue: ${errorHint}`);

    const errorMessage =
      `I was unable to generate a response. ${errorHint}\n\n` +
      `Diagnostic info:\n- Model: ${model.id}\n- Tools provided: ${toolCount}\n` +
      `- Estimated input tokens: ${inputTokenCount}\n- Context limit: ${modelMaxContext}\n\n` +
      `Check the "GitHub Copilot LLM Gateway" output panel for detailed logs.`;

    progress.report(new vscode.LanguageModelTextPart(errorMessage));
  }

  private handleChatError(error: unknown): never {
    const errorMessage = error instanceof Error ? error.message : String(error);
    const errorStack = error instanceof Error ? error.stack : '';

    this.outputChannel.appendLine(`ERROR: Chat request failed: ${errorMessage}`);
    if (errorStack) {
      this.outputChannel.appendLine(`Stack trace: ${errorStack}`);
    }

    // Be conservative — only treat the error as a tool-calling format error
    // when the message contains a known tool-parser signal. The previous
    // heuristic also matched on `unexpected tokens`, which appears in many
    // unrelated errors and was triggering the "may not support tool calling"
    // prompt incorrectly.
    const isToolError =
      errorMessage.includes('HarmonyError') ||
      /tool[_ -]?call.*parse/i.test(errorMessage);

    if (isToolError) {
      this.outputChannel.appendLine('HINT: This appears to be a tool calling format error.');
      this.outputChannel.appendLine('The model may not support function calling properly.');
      this.outputChannel.appendLine(
        'Try: 1) Using a different model, 2) Disabling tool calling in settings, or 3) Checking inference server logs'
      );
      this.promptToolCallingError();
    } else {
      vscode.window.showErrorMessage(
        `GitHub Copilot LLM Gateway: Chat request failed. ${errorMessage}`
      );
    }

    throw error;
  }

  private promptOpenSettings(message: string): void {
    vscode.window.showErrorMessage(message, 'Open Settings').then(
      (selection: string | undefined) => {
        if (selection === 'Open Settings') {
          vscode.commands.executeCommand(
            'workbench.action.openSettings',
            'github.copilot.llm-gateway'
          );
        }
      },
      (err: unknown) => {
        this.outputChannel.appendLine(
          `Failed to show settings prompt: ${err instanceof Error ? err.message : String(err)}`
        );
      }
    );
  }

  private promptToolCallingError(): void {
    vscode.window
      .showErrorMessage(
        `GitHub Copilot LLM Gateway: Model failed to generate valid tool calls. This model may not support function calling. Check Output panel for details.`,
        'Open Output',
        'Disable Tool Calling'
      )
      .then(
        (selection: string | undefined) => {
          if (selection === 'Open Output') {
            this.outputChannel.show();
          } else if (selection === 'Disable Tool Calling') {
            vscode.workspace
              .getConfiguration('github.copilot.llm-gateway')
              .update('enableToolCalling', false, vscode.ConfigurationTarget.Global);
          }
        },
        (err: unknown) => {
          this.outputChannel.appendLine(
            `Failed to show tool calling error prompt: ${err instanceof Error ? err.message : String(err)}`
          );
        }
      );
  }

  // ---------- config ----------

  private loadConfig(): GatewayConfig {
    const config = vscode.workspace.getConfiguration('github.copilot.llm-gateway');

    // `apiKey` and `customHeaders` come from the in-memory secret cache
    // populated by `loadSecrets` / `refreshSecretCache`. The legacy
    // plain-text settings of the same name are still read by the migration
    // path, but are cleared once their values are safely in SecretStorage
    // (issue #28). Until `loadSecrets` runs, the cache holds empty values —
    // an early model fetch would just send unauthenticated requests.
    const cfg: GatewayConfig = {
      serverUrl: config.get<string>('serverUrl', 'http://localhost:8000'),
      // Framework-managed API key (from VS Code's native model-picker UI) wins
      // over the SecretStorage cache. Falls back to SecretStorage when nothing
      // has come in via the configuration arg yet, which preserves the
      // existing Configure Server flow for users on builds without the
      // framework UI.
      apiKey: resolveApiKey(this.frameworkOverride, this.secretCache.apiKey),
      requestTimeout: config.get<number>('requestTimeout', DEFAULT_REQUEST_TIMEOUT_MS),
      defaultMaxTokens: config.get<number>('defaultMaxTokens', TOKEN_CONSTANTS.DEFAULT_CONTEXT_TOKENS),
      defaultMaxOutputTokens: config.get<number>(
        'defaultMaxOutputTokens',
        TOKEN_CONSTANTS.FALLBACK_OUTPUT_TOKENS
      ),
      enableImageInput: config.get<boolean>('enableImageInput', false),
      enableToolCalling: config.get<boolean>('enableToolCalling', true),
      parallelToolCalling: config.get<boolean>('parallelToolCalling', true),
      agentTemperature: config.get<number>('agentTemperature', 0),
      verboseLogging: config.get<boolean>('verboseLogging', false),
      customHeaders: { ...this.secretCache.customHeaders },
      extraModelOptions: config.get<Record<string, unknown>>('extraModelOptions', {}) ?? {},
      perModelSettings: config.get<Record<string, Record<string, unknown>>>('perModelSettings', {}) ?? {},
      stripFakeToolCallText: config.get<boolean>('stripFakeToolCallText', true),
      retryFakeToolCalls: config.get<boolean>('retryFakeToolCalls', true),
    };

    const MAX_INT32 = 2147483647; // Maximum value for setTimeout (signed 32-bit integer)
    if (cfg.requestTimeout <= 0) {
      this.outputChannel.appendLine(
        `ERROR: requestTimeout must be > 0; using default ${DEFAULT_REQUEST_TIMEOUT_MS}`
      );
      cfg.requestTimeout = DEFAULT_REQUEST_TIMEOUT_MS;
    } else if (cfg.requestTimeout > MAX_INT32) {
      this.outputChannel.appendLine(
        `WARNING: requestTimeout (${cfg.requestTimeout}) exceeds the maximum value of 2147483647 ms (signed 32-bit integer). Setting to ${MAX_INT32}.`
      );
      cfg.requestTimeout = MAX_INT32;
    }

    try {
      new URL(cfg.serverUrl);
      // URL became valid — reset the dedupe key so future invalid values are
      // re-surfaced.
      this.lastInvalidUrlNotified = undefined;
    } catch {
      const fallback = 'http://localhost:8000';
      this.outputChannel.appendLine(
        `ERROR: Invalid server URL ${JSON.stringify(cfg.serverUrl)}. Falling back to ${fallback}; fix this in settings.`
      );
      // Only surface the UI prompt if we haven't already warned about this
      // exact value — otherwise the user gets a new modal for every keystroke
      // while they're typing a URL in settings.
      if (this.lastInvalidUrlNotified !== cfg.serverUrl) {
        this.lastInvalidUrlNotified = cfg.serverUrl;
        setImmediate(() => {
          this.promptOpenSettings(
            `GitHub Copilot LLM Gateway: Invalid Server URL ${JSON.stringify(cfg.serverUrl)}. Open Settings to fix.`
          );
        });
      }
      cfg.serverUrl = fallback;
    }

    if (cfg.defaultMaxOutputTokens >= cfg.defaultMaxTokens) {
      const adjusted = Math.max(
        TOKEN_CONSTANTS.MIN_OUTPUT_TOKENS,
        cfg.defaultMaxTokens - TOKEN_CONSTANTS.ADJUST_TOKEN_BUFFER
      );
      this.outputChannel.appendLine(
        `WARNING: github.copilot.llm-gateway.defaultMaxOutputTokens (${cfg.defaultMaxOutputTokens}) >= defaultMaxTokens (${cfg.defaultMaxTokens}). Adjusting to ${adjusted}.`
      );
      // Only pop a toast when the values the user is typing actually change,
      // otherwise every keystroke during settings editing produces a warning.
      const last = this.lastOutputTokenAdjustmentNotified;
      if (last?.output !== cfg.defaultMaxOutputTokens || last?.total !== cfg.defaultMaxTokens) {
        this.lastOutputTokenAdjustmentNotified = {
          output: cfg.defaultMaxOutputTokens,
          total: cfg.defaultMaxTokens,
        };
        vscode.window.showWarningMessage(
          `GitHub Copilot LLM Gateway: 'defaultMaxOutputTokens' was >= 'defaultMaxTokens'. Adjusted to ${adjusted} to avoid request errors.`
        );
      }
      cfg.defaultMaxOutputTokens = adjusted;
    } else {
      // Valid configuration — reset the dedupe key.
      this.lastOutputTokenAdjustmentNotified = undefined;
    }

    return cfg;
  }

  private reloadConfig(): void {
    this.config = this.loadConfig();
    this.client.updateConfig(this.config);
    this.outputChannel.appendLine('Configuration reloaded');
  }
}
