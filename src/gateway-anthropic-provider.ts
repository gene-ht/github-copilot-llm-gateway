/**
 * Anthropic Messages API support for the Gateway extension.
 *
 * Uses `@anthropic-ai/sdk` (same as Copilot's built-in AnthropicLMProvider)
 * to call Claude models via native Anthropic Messages API (`/v1/messages`).
 *
 * When the model is detected as Claude-family, the Gateway routes requests
 * through this module instead of the OpenAI `/v1/chat/completions` path.
 * This eliminates the lossy OpenAI→Anthropic translation on the upstream
 * side that causes `<invoke>` XML self-poisoning, tool-call pairing errors,
 * and dropped `tool_calls` turns.
 *
 * Integrates with Gateway infrastructure:
 *   - `StreamReporter` / `StreamStats` for status bar / session stats
 *   - `resolveToolCallArgs` for JSON repair + schema fill
 *   - `OpenAIUsage` mapping for usage reporting
 *
 * Reference: Copilot's AnthropicLMProvider
 *   (src/extension/byok/vscode-node/anthropicProvider.ts)
 */

import Anthropic from '@anthropic-ai/sdk';
import type {
  ContentBlockParam,
  MessageParam,
  TextBlockParam,
  Tool,
  MessageCreateParamsStreaming,
  RawMessageStreamEvent,
} from '@anthropic-ai/sdk/resources';
import * as vscode from 'vscode';
import { StreamReporter, StreamStats } from './responseStreamer';
import { OpenAIUsage } from './types';
import { stripPoisonText } from './messageConverter';

// ============================================================================
// Model detection
// ============================================================================

const CLAUDE_MODEL_RE = /^(anthropic\/)?claude/i;

/**
 * Returns true if the model id looks like a Claude model.
 * Matches: `claude-opus-4`, `anthropic/claude-3.5-sonnet`, `claude-3-5-sonnet@20240620`
 */
export function isClaudeModel(modelId: string): boolean {
  return CLAUDE_MODEL_RE.test(modelId);
}

// ============================================================================
// Message conversion: VS Code → Anthropic SDK types
// ============================================================================

export interface AnthropicConversion {
  messages: MessageParam[];
  system: TextBlockParam;
}

/**
 * Convert VS Code LanguageModelChatMessage[] to Anthropic Messages API format.
 *
 * - System messages → extracted into top-level `system` field
 * - Tool calls → `tool_use` content blocks
 * - Tool results → `tool_result` content blocks
 * - Thinking → `thinking` / `redacted_thinking` blocks
 * - Adjacent same-role messages → merged (Anthropic requires strict alternation)
 */
export function convertMessagesToAnthropic(
  messages: readonly vscode.LanguageModelChatMessage[],
  enableImageInput: boolean = true
): AnthropicConversion {
  const unmergedMessages: MessageParam[] = [];
  const systemMessage: TextBlockParam = { type: 'text', text: '' };

  for (const message of messages) {
    if (message.role === vscode.LanguageModelChatMessageRole.Assistant) {
      // Strip fake tool-call text from assistant messages to prevent
      // self-poisoning. Historical assistant text may contain "Completed
      // tool calls:" bullets or <invoke> XML from the OpenAI path.
      const content = convertContentParts(message.content, enableImageInput, true);
      if (content.length > 0) {
        unmergedMessages.push({ role: 'assistant', content });
      }
    } else if (message.role === vscode.LanguageModelChatMessageRole.User) {
      const content = convertContentParts(message.content, enableImageInput, false);
      if (content.length > 0) {
        unmergedMessages.push({ role: 'user', content });
      }
    } else {
      // System message — extract and merge text
      for (const part of message.content) {
        if (part instanceof vscode.LanguageModelTextPart) {
          if (systemMessage.text.length > 0) {
            systemMessage.text += '\n\n';
          }
          systemMessage.text += part.value;
        }
      }
    }
  }

  // Merge adjacent messages of the same role (Anthropic strict requirement)
  const mergedMessages: MessageParam[] = [];
  for (const message of unmergedMessages) {
    if (
      mergedMessages.length === 0 ||
      mergedMessages[mergedMessages.length - 1].role !== message.role
    ) {
      mergedMessages.push(message);
    } else {
      const prev = mergedMessages[mergedMessages.length - 1];
      if (Array.isArray(prev.content) && Array.isArray(message.content)) {
        (prev.content as ContentBlockParam[]).push(
          ...(message.content as ContentBlockParam[])
        );
      }
    }
  }

  // Anthropic API strictly validates tool_use/tool_result pairing: every
  // tool_result must have a matching tool_use in the immediately preceding
  // assistant message. VS Code can truncate conversation history mid-turn,
  // leaving orphaned tool_result blocks at the start. Strip them.
  const validatedMessages = stripOrphanedToolResults(mergedMessages);

  return { messages: validatedMessages, system: systemMessage };
}

/**
 * Remove `tool_result` content blocks whose `tool_use_id` has no matching
 * `tool_use` block in the immediately preceding `assistant` message.
 *
 * This handles the case where VS Code truncates conversation history and
 * resumes mid-tool-call-sequence, leaving the first user message with
 * tool_result blocks whose tool_use was in a truncated assistant message.
 */
function stripOrphanedToolResults(messages: MessageParam[]): MessageParam[] {
  const result: MessageParam[] = [];

  for (const msg of messages) {
    if (msg.role === 'user' && Array.isArray(msg.content)) {
      // Collect tool_use IDs from the preceding assistant message
      const prev = result.length > 0 ? result[result.length - 1] : undefined;
      const validIds = new Set<string>();
      if (prev?.role === 'assistant' && Array.isArray(prev.content)) {
        for (const block of prev.content) {
          const b = block as unknown as Record<string, unknown>;
          if (b.type === 'tool_use' && typeof b.id === 'string') {
            validIds.add(b.id);
          }
        }
      }

      // Keep non-tool_result blocks + tool_result blocks with valid IDs
      const filtered = (msg.content as ContentBlockParam[]).filter((block) => {
        const b = block as unknown as Record<string, unknown>;
        if (b.type === 'tool_result') {
          return validIds.has(b.tool_use_id as string);
        }
        return true;
      });

      if (filtered.length > 0) {
        result.push({ role: 'user', content: filtered });
      }
    } else {
      result.push(msg);
    }
  }

  // Ensure conversation starts with user message
  while (result.length > 0 && result[0].role !== 'user') {
    result.shift();
  }

  return result;
}

function convertContentParts(
  content: ReadonlyArray<
    | vscode.LanguageModelTextPart
    | vscode.LanguageModelToolCallPart
    | vscode.LanguageModelToolResultPart
    | vscode.LanguageModelDataPart
    | vscode.LanguageModelThinkingPart
  >,
  enableImageInput: boolean,
  stripPoison: boolean
): ContentBlockParam[] {
  const result: ContentBlockParam[] = [];

  for (const part of content) {
    if (part instanceof vscode.LanguageModelThinkingPart) {
      const meta = (part as unknown as Record<string, unknown>).metadata as
        | Record<string, unknown>
        | undefined;
      if (meta?.redactedData) {
        result.push({
          type: 'redacted_thinking',
          data: meta.redactedData as string,
        });
      } else if (meta?._completeThinking) {
        result.push({
          type: 'thinking',
          thinking: meta._completeThinking as string,
          signature: (meta.signature as string) || '',
        });
      }
    } else if (part instanceof vscode.LanguageModelToolCallPart) {
      result.push({
        type: 'tool_use',
        id: part.callId,
        name: part.name,
        input: part.input,
      });
    } else if (part instanceof vscode.LanguageModelToolResultPart) {
      const textParts = part.content
        .filter(
          (c): c is vscode.LanguageModelTextPart =>
            c instanceof vscode.LanguageModelTextPart
        )
        .map((c) => ({ type: 'text' as const, text: c.value || '(empty)' }));
      // Anthropic 400s on empty tool result content
      if (textParts.length === 0) {
        textParts.push({ type: 'text', text: '(empty)' });
      }
      result.push({
        type: 'tool_result',
        tool_use_id: part.callId,
        content: textParts,
      });
    } else if (part instanceof vscode.LanguageModelDataPart) {
      if (enableImageInput && part.mimeType.startsWith('image/')) {
        result.push({
          type: 'image',
          source: {
            type: 'base64',
            data: Buffer.from(part.data).toString('base64'),
            media_type: part.mimeType as
              | 'image/jpeg'
              | 'image/png'
              | 'image/gif'
              | 'image/webp',
          },
        });
      }
    } else if (part instanceof vscode.LanguageModelTextPart) {
      // Anthropic 400s on empty text blocks
      let text = part.value;
      if (stripPoison && text) {
        text = stripPoisonText(text);
      }
      if (text !== '') {
        result.push({ type: 'text', text });
      }
    }
  }

  return result;
}

// ============================================================================
// Tool definition conversion
// ============================================================================

/**
 * Convert VS Code tool definitions to Anthropic SDK `Tool` format.
 */
export function convertToolsToAnthropic(
  tools: readonly vscode.LanguageModelChatTool[] | undefined
): Tool[] | undefined {
  if (!tools || tools.length === 0) {
    return undefined;
  }

  return tools.map((tool) => ({
    name: tool.name,
    description: tool.description || '',
    input_schema: buildToolInputSchema(
      tool.inputSchema as Record<string, unknown> | undefined
    ),
  }));
}

function buildToolInputSchema(
  schema: Record<string, unknown> | undefined
): Record<string, unknown> & { type: 'object' } {
  if (!schema) {
    return { type: 'object', properties: {}, required: [] };
  }
  const { $schema: _, ...rest } = schema;
  return { type: 'object', properties: {}, ...rest };
}

// ============================================================================
// Tool choice mapping
// ============================================================================

/**
 * Map VS Code tool mode to Anthropic tool_choice.
 */
export function mapToolChoice(
  toolMode: vscode.LanguageModelChatToolMode | undefined,
  parallelToolCalling: boolean
): MessageCreateParamsStreaming['tool_choice'] | undefined {
  if (!toolMode) {
    return undefined;
  }
  const disableParallel = !parallelToolCalling
    ? { disable_parallel_tool_use: true as const }
    : {};
  if (toolMode === vscode.LanguageModelChatToolMode.Required) {
    return { type: 'any' as const, ...disableParallel };
  }
  return { type: 'auto' as const, ...disableParallel };
}

// ============================================================================
// Request building with perModelSettings + reasoning_effort mapping
// ============================================================================

/**
 * OpenAI fields that should NOT be forwarded to Anthropic.
 */
const OPENAI_ONLY_FIELDS = new Set([
  'reasoning_effort',
  'frequency_penalty',
  'presence_penalty',
  'logit_bias',
  'logprobs',
  'top_logprobs',
  'response_format',
  'function_call',
  'functions',
  'seed',
  'n',
  'stop',
  'stream_options',
  '_capturingTokenCorrelationId',
  '_otelTraceContext',
]);

/**
 * Map `reasoning_effort` (OpenAI concept) to Anthropic `thinking` configuration.
 */
function mapReasoningEffort(
  effort: unknown,
  existingThinking: unknown
): MessageCreateParamsStreaming['thinking'] | undefined {
  if (existingThinking !== undefined) {
    return existingThinking as MessageCreateParamsStreaming['thinking'];
  }
  if (typeof effort !== 'string') {
    return undefined;
  }
  switch (effort) {
    case 'low':
      return { type: 'enabled', budget_tokens: 1024 };
    case 'medium':
      return { type: 'enabled', budget_tokens: 8192 };
    case 'high':
      return { type: 'enabled', budget_tokens: 32768 };
    default:
      return undefined;
  }
}

export interface BuildAnthropicRequestOptions {
  model: string;
  conversion: AnthropicConversion;
  maxTokens: number;
  temperature?: number;
  tools?: Tool[];
  toolChoice?: MessageCreateParamsStreaming['tool_choice'];
  /** Merged extraModelOptions + perModelSettings + callerOverride */
  extraOptions?: Record<string, unknown>;
}

/**
 * Build the Anthropic Messages API request params for the SDK.
 */
export function buildAnthropicRequest(
  opts: BuildAnthropicRequestOptions
): MessageCreateParamsStreaming {
  const params: MessageCreateParamsStreaming = {
    model: opts.model,
    messages: opts.conversion.messages,
    max_tokens: opts.maxTokens,
    stream: true as const,
  };

  if (opts.conversion.system.text) {
    params.system = [opts.conversion.system];
  }

  if (opts.temperature !== undefined) {
    params.temperature = opts.temperature;
  }

  if (opts.tools && opts.tools.length > 0) {
    params.tools = opts.tools;
    if (opts.toolChoice) {
      params.tool_choice = opts.toolChoice;
    }
  }

  // Extra options: filter OpenAI-only fields, map reasoning_effort
  if (opts.extraOptions) {
    const thinking = mapReasoningEffort(
      opts.extraOptions.reasoning_effort,
      opts.extraOptions.thinking
    );
    if (thinking) {
      params.thinking = thinking;
    }

    for (const [key, value] of Object.entries(opts.extraOptions)) {
      if (OPENAI_ONLY_FIELDS.has(key) || key === 'thinking' || key.startsWith('_')) {
        continue;
      }
      if (value !== undefined) {
        (params as unknown as Record<string, unknown>)[key] = value;
      }
    }
  }

  // Anthropic API constraint: when extended thinking is enabled, temperature
  // must be exactly 1.0 or be omitted entirely. If the user/perModelSettings
  // set a different temperature alongside reasoning_effort, we override it.
  if (params.thinking && typeof params.thinking === 'object' && 'type' in params.thinking && params.thinking.type === 'enabled') {
    if (params.temperature !== undefined && params.temperature !== 1) {
      params.temperature = 1;
    }
  }

  return params;
}

// ============================================================================
// Anthropic SDK client factory
// ============================================================================

/**
 * Create an Anthropic SDK client pointing at the Gateway's upstream server.
 *
 * The SDK handles:
 *   - HTTP connection management
 *   - SSE stream parsing
 *   - Typed event emission
 *   - Retry logic (disabled by default for streaming)
 */
export function createAnthropicClient(
  serverUrl: string,
  apiKey: string | undefined,
  customHeaders: Record<string, string>
): Anthropic {
  // Merge custom headers — the SDK sets its own x-api-key and anthropic-version,
  // but customHeaders can override anything.
  const defaultHeaders: Record<string, string> = {};
  for (const [key, value] of Object.entries(customHeaders)) {
    if (value !== undefined && value !== '') {
      defaultHeaders[key] = value;
    }
  }

  // The SDK appends `/v1/messages` to the baseURL automatically.
  // Strip any trailing `/v1` from the serverUrl to avoid double-prefix
  // (e.g. `https://api.example.com/v1` → `https://api.example.com`).
  const baseURL = serverUrl
    .replace(/\/$/, '')
    .replace(/\/v1$/i, '');

  return new Anthropic({
    apiKey: apiKey || '',
    baseURL,
    defaultHeaders,
    // Disable SDK retry for streaming — we manage our own timeout/cancel
    maxRetries: 0,
  });
}

// ============================================================================
// SSE stream processing → StreamReporter + StreamStats
// ============================================================================

const CAPTURE_LIMIT = 20 * 1024; // 20 KB, same as OpenAI path

export interface AnthropicStreamParams {
  /** SDK stream from `client.messages.create()` */
  stream: AsyncIterable<RawMessageStreamEvent>;
  /** Reporter that drives progress.report() + status bar */
  reporter: StreamReporter;
  /** Called before reading each event; return true to stop early */
  isCancelled: () => boolean;
  /** Capture content for post-stream inspection */
  captureContent?: boolean;
  /**
   * Called with each finished tool call. The callback is responsible for
   * JSON-repairing the arguments and filling any missing required properties
   * from the tool's schema.
   */
  resolveToolCallArgs: (toolCall: {
    id: string;
    name: string;
    arguments: string;
  }) => Record<string, unknown>;
  /** Verbose logging callback */
  log?: (message: string) => void;
}

interface AnthropicUsage {
  input_tokens: number;
  output_tokens: number;
  cache_creation_input_tokens?: number;
  cache_read_input_tokens?: number;
}

/**
 * Process an Anthropic SDK event stream, reporting parts through `StreamReporter`.
 *
 * This is the Anthropic counterpart of `responseStreamer.streamResponse()`.
 * It returns the same `StreamStats` shape so the provider can use the same
 * post-stream logic (empty-response handling, session stats, etc.).
 *
 * Directly adapted from Copilot's `AnthropicLMProvider._makeRequest()`.
 */
export async function streamAnthropicResponse(
  params: AnthropicStreamParams
): Promise<StreamStats> {
  const {
    stream,
    reporter,
    isCancelled,
    captureContent,
    resolveToolCallArgs,
    log,
  } = params;

  const stats: StreamStats = {
    totalContentLength: 0,
    totalToolCalls: 0,
    totalTextParts: 0,
    hadThinking: false,
    thinkingForceClosed: false,
    ...(captureContent ? { capturedContent: '' } : {}),
  };

  let pendingToolCall:
    | { id: string; name: string; argsBuffer: string }
    | undefined;
  let inThinking = false;
  let usage: AnthropicUsage | undefined;

  for await (const chunk of stream) {
    if (isCancelled()) {
      break;
    }

    // ── content_block_start ─────────────────────────────────────
    if (chunk.type === 'content_block_start') {
      if ('content_block' in chunk) {
        const block = chunk.content_block;
        if (block.type === 'tool_use') {
          pendingToolCall = {
            id: block.id,
            name: block.name,
            argsBuffer: '',
          };
          log?.(
            `[Anthropic] tool_use start: ${block.name} (${block.id})`
          );
        } else if (block.type === 'thinking') {
          inThinking = true;
          stats.hadThinking = true;
        }
      }
      continue;
    }

    // ── content_block_delta ─────────────────────────────────────
    if (chunk.type === 'content_block_delta') {
      const delta = chunk.delta;

      if (delta.type === 'text_delta') {
        const text = delta.text || '';
        reporter.reportText(text);
        stats.totalContentLength += text.length;
        stats.totalTextParts++;
        if (
          captureContent &&
          stats.capturedContent !== undefined &&
          stats.capturedContent.length < CAPTURE_LIMIT
        ) {
          stats.capturedContent += text.slice(
            0,
            CAPTURE_LIMIT - stats.capturedContent.length
          );
        }
      } else if (delta.type === 'input_json_delta' && pendingToolCall) {
        pendingToolCall.argsBuffer += delta.partial_json || '';
      } else if (delta.type === 'thinking_delta') {
        const text = (delta as unknown as { thinking: string }).thinking || '';
        reporter.reportThinking(text);
        stats.hadThinking = true;
      }
      // signature_delta: not reported (v1 limitation)
      continue;
    }

    // ── content_block_stop ──────────────────────────────────────
    if (chunk.type === 'content_block_stop') {
      if (pendingToolCall) {
        // Flush through resolveToolCallArgs (JSON repair + schema fill)
        const args = resolveToolCallArgs({
          id: pendingToolCall.id,
          name: pendingToolCall.name,
          arguments: pendingToolCall.argsBuffer || '{}',
        });
        reporter.reportToolCall(pendingToolCall.id, pendingToolCall.name, args);
        stats.totalToolCalls++;
        log?.(
          `[Anthropic] tool_use complete: ${pendingToolCall.name} (${pendingToolCall.id})`
        );
        pendingToolCall = undefined;
      }

      if (inThinking) {
        reporter.reportThinkingDone();
        inThinking = false;
      }
      continue;
    }

    // ── message_start (input usage) ─────────────────────────────
    if (chunk.type === 'message_start') {
      const msg = (chunk as unknown as Record<string, unknown>)
        .message as Record<string, unknown> | undefined;
      const msgUsage = msg?.usage as AnthropicUsage | undefined;
      if (msgUsage) {
        usage = {
          input_tokens: msgUsage.input_tokens ?? 0,
          output_tokens: 0,
          cache_creation_input_tokens: msgUsage.cache_creation_input_tokens,
          cache_read_input_tokens: msgUsage.cache_read_input_tokens,
        };
      }
      continue;
    }

    // ── message_delta (output usage) ────────────────────────────
    if (chunk.type === 'message_delta') {
      const deltaUsage = (chunk as unknown as Record<string, unknown>)
        .usage as Partial<AnthropicUsage> | undefined;
      if (usage && deltaUsage?.output_tokens) {
        usage.output_tokens = deltaUsage.output_tokens;
      }
      continue;
    }

    // ── message_stop → report final usage ───────────────────────
    if (chunk.type === 'message_stop') {
      if (usage && !stats.reportedUsage) {
        const openAIUsage: OpenAIUsage = {
          prompt_tokens: usage.input_tokens,
          completion_tokens: usage.output_tokens,
          total_tokens: usage.input_tokens + usage.output_tokens,
          ...(usage.cache_read_input_tokens !== undefined
            ? {
                prompt_tokens_details: {
                  cached_tokens: usage.cache_read_input_tokens,
                },
              }
            : {}),
        };
        reporter.reportUsage(openAIUsage);
        stats.reportedUsage = true;
        log?.(
          `[Anthropic] Usage: input=${usage.input_tokens} output=${usage.output_tokens}` +
            (usage.cache_read_input_tokens
              ? ` cached=${usage.cache_read_input_tokens}`
              : '') +
            (usage.cache_creation_input_tokens
              ? ` cache_created=${usage.cache_creation_input_tokens}`
              : '')
        );
      }
      continue;
    }

    // ping, other events: ignore (SDK throws on error events automatically)
  }

  // Force-close thinking if stream ended mid-think
  if (inThinking) {
    reporter.reportThinkingDone();
    stats.thinkingForceClosed = true;
  }

  return stats;
}
