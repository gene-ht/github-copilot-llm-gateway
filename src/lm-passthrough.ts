/**
 * LM Passthrough transport for the Gateway extension.
 *
 * Serializes VS Code's `LanguageModelChatMessage[]` directly to JSON and
 * sends them to an upstream Gateway via `POST /lm/chat`. The upstream
 * handles all model-specific format conversion (OpenAI / Anthropic / etc.),
 * so this module performs **no format translation** — it's a transparent
 * pipe between Copilot's tool-calling loop and the upstream gateway.
 *
 * Response comes back as SSE events with a simple typed envelope:
 *   data: {"type":"text","value":"..."}
 *   data: {"type":"tool_call","callId":"...","name":"...","input":{...}}
 *   data: {"type":"thinking","value":"...","id":"...","metadata":{...}}
 *   data: {"type":"error","message":"..."}
 *   data: {"type":"done"}
 */

import * as vscode from 'vscode';
import { buildHeaders, normalizeBaseUrl } from './client';

/**
 * MIME type VS Code 1.120 watches for on `LanguageModelDataPart`s to extract
 * token usage and feed it into the chat context-window widget. Must match the
 * constant used by the OpenAI/Anthropic paths (see microsoft/vscode#315394).
 */
const USAGE_DATA_PART_MIME_TYPE = 'usage';

// ============================================================================
// Serialization: VS Code types → JSON wire format
// ============================================================================

/**
 * Serialized message part on the wire.
 */
export type LmPartPayload =
  | { type: 'text'; value: string }
  | { type: 'tool_call'; callId: string; name: string; input: unknown }
  | { type: 'tool_result'; callId: string; content: LmPartPayload[] }
  | { type: 'thinking'; value: string; id?: string; metadata?: Record<string, unknown> }
  | { type: 'data'; mimeType: string; data: string };

/**
 * Serialized message on the wire.
 *
 * `role` uses string values matching the VS Code enum names:
 *   - `'user'`      (LanguageModelChatMessageRole.User = 1)
 *   - `'assistant'`  (LanguageModelChatMessageRole.Assistant = 2)
 */
export interface LmMessagePayload {
  role: string;
  content: LmPartPayload[];
}

/**
 * Serialized tool definition on the wire.
 */
export interface LmToolPayload {
  name: string;
  description: string;
  inputSchema: unknown;
}

/**
 * Request body for `POST /lm/chat`.
 *
 * Wire protocol is point-to-point between this extension and the upstream
 * gateway. Field names are fixed (no aliases) — both sides must agree.
 */
export interface LmChatRequest {
  model: string;
  messages: LmMessagePayload[];
  tools?: LmToolPayload[];
  /**
   * Tool-selecting mode. Maps from VS Code's `LanguageModelChatToolMode`:
   *   - `'auto'`     -> Auto
   *   - `'required'` -> Required
   * Upstream maps it back to the same enum on `vscode.lm.sendRequest`.
   */
  tool_mode?: 'auto' | 'required';
  /**
   * Public `modelOptions` from `vscode.LanguageModelChatRequestOptions`.
   * Transparently forwarded to upstream `sendRequest(...).options.modelOptions`.
   * The receiving provider decides which keys it honors (e.g. official Copilot
   * applies a whitelist of `stop` / `temperature` / `max_tokens` /
   * `frequency_penalty` / `presence_penalty`; other keys are dropped). This
   * layer does no filtering.
   */
  model_options?: Record<string, unknown>;
  /**
   * Picker "Thinking Effort" selection. Upstream injects this into the
   * official Copilot provider's `configuration.reasoningEffort`.
   */
  reasoning_effort?: string;
  /**
   * Prompt/input budget. Upstream injects this into the official Copilot
   * provider's `configuration.contextSize` to override the endpoint's
   * prompt budget.
   */
  context_size?: number;
}

/**
 * Serialize a single `LanguageModelChatMessage` content part to the wire
 * format. Returns `undefined` for unrecognized part types so the caller
 * can filter them out.
 */
function serializePart(part: unknown): LmPartPayload | undefined {
  if (part instanceof vscode.LanguageModelTextPart) {
    return { type: 'text', value: part.value };
  }
  if (part instanceof vscode.LanguageModelToolCallPart) {
    return { type: 'tool_call', callId: part.callId, name: part.name, input: part.input };
  }
  if (part instanceof vscode.LanguageModelToolResultPart) {
    const content: LmPartPayload[] = [];
    for (const c of part.content) {
      if (c instanceof vscode.LanguageModelTextPart) {
        content.push({ type: 'text', value: c.value });
      } else {
        // LanguageModelDataPart or similar — encode as base64
        const dp = c as { mimeType?: string; data?: Uint8Array };
        if (dp.data) {
          content.push({
            type: 'data',
            mimeType: dp.mimeType ?? 'application/octet-stream',
            data: Buffer.from(dp.data).toString('base64'),
          });
        }
      }
    }
    return { type: 'tool_result', callId: part.callId, content };
  }
  if (part instanceof vscode.LanguageModelThinkingPart) {
    return {
      type: 'thinking',
      value: part.value,
      ...(part.id !== undefined ? { id: part.id } : {}),
      ...((part as { metadata?: Record<string, unknown> }).metadata !== undefined
        ? { metadata: (part as { metadata?: Record<string, unknown> }).metadata }
        : {}),
    };
  }
  if (part instanceof vscode.LanguageModelDataPart) {
    return {
      type: 'data',
      mimeType: part.mimeType,
      data: Buffer.from(part.data).toString('base64'),
    };
  }
  return undefined;
}

/**
 * Map a VS Code LanguageModelChatMessageRole enum value to a wire-format
 * string. VS Code enums are numeric at runtime:
 *   - User      = 1
 *   - Assistant  = 2
 *
 * Any other value (e.g. future additions) falls through to `'user'`.
 */
function roleToString(role: vscode.LanguageModelChatMessageRole): string {
  if (role === vscode.LanguageModelChatMessageRole.Assistant) { return 'assistant'; }
  return 'user';
}

/**
 * Serialize `LanguageModelChatMessage[]` to the LM passthrough JSON wire
 * format. Roles are mapped to `'user'` / `'assistant'` strings.
 */
export function serializeMessages(
  messages: readonly vscode.LanguageModelChatMessage[]
): LmMessagePayload[] {
  return messages.map((msg) => ({
    role: roleToString(msg.role),
    content: msg.content
      .map((part) => serializePart(part))
      .filter((p): p is LmPartPayload => p !== undefined),
  }));
}

/**
 * Serialize `LanguageModelChatTool[]` to the flat wire format.
 * Returns `undefined` when the tools array is empty or absent.
 */
export function serializeTools(
  tools: readonly vscode.LanguageModelChatTool[] | undefined
): LmToolPayload[] | undefined {
  if (!tools || tools.length === 0) { return undefined; }
  return tools.map((t) => ({
    name: t.name,
    description: t.description,
    inputSchema: t.inputSchema,
  }));
}

// ============================================================================
// Tool result pairing repair
// ============================================================================

/**
 * Strip orphaned `tool_result` parts whose matching `tool_call` (by `callId`)
 * doesn't appear **anywhere** in earlier messages.
 *
 * VS Code / Copilot truncates conversation history, which can leave
 * `tool_result` parts at the start of the window whose corresponding
 * `tool_call` was dropped. Anthropic rejects these with:
 *   "unexpected tool_use_id found in tool_result blocks"
 *
 * This is a conservative repair: it only strips `tool_result` parts whose
 * `callId` has no matching `tool_call` in the *entire preceding history*.
 * It does NOT try to enforce Anthropic's strict "immediately preceding
 * message" rule — that's the upstream Gateway's responsibility when it
 * converts to Anthropic format.
 *
 * Also strips orphaned `tool_call` parts (at the end of the window)
 * whose `callId` has no matching `tool_result` in any *later* message,
 * to avoid the symmetric error.
 *
 * Empty messages (all parts stripped) are removed.
 */
export function repairToolResultPairing(
  messages: LmMessagePayload[],
  log: (message: string) => void
): LmMessagePayload[] {
  // Pass 1: collect ALL tool_call and tool_result callIds across all messages
  const allToolCallIds = new Set<string>();
  const allToolResultIds = new Set<string>();
  for (const msg of messages) {
    for (const part of msg.content) {
      if (part.type === 'tool_call') {
        allToolCallIds.add(part.callId);
      } else if (part.type === 'tool_result') {
        allToolResultIds.add(part.callId);
      }
    }
  }

  // Pass 2: filter — strip tool_results with no matching tool_call,
  // and tool_calls with no matching tool_result
  const result: LmMessagePayload[] = [];
  let strippedResults = 0;
  let strippedCalls = 0;

  for (const msg of messages) {
    const filteredContent = msg.content.filter((part) => {
      if (part.type === 'tool_result') {
        if (!allToolCallIds.has(part.callId)) {
          strippedResults++;
          return false;
        }
      }
      if (part.type === 'tool_call') {
        if (!allToolResultIds.has(part.callId)) {
          strippedCalls++;
          return false;
        }
      }
      return true;
    });

    if (filteredContent.length > 0) {
      result.push({ ...msg, content: filteredContent });
    }
  }

  if (strippedResults > 0 || strippedCalls > 0) {
    log(
      `[lm-passthrough] Stripped ${strippedResults} orphaned tool_result + ` +
      `${strippedCalls} orphaned tool_call part(s) ` +
      `(${messages.length} → ${result.length} messages)`
    );
  }

  return result;
}

// ============================================================================
// SSE response events
// ============================================================================

/**
 * Typed union of all SSE event payloads the upstream can send.
 */
export type LmSseEvent =
  | { type: 'text'; value: string }
  | { type: 'tool_call'; callId: string; name: string; input: unknown }
  | { type: 'thinking'; value: string; id?: string; metadata?: Record<string, unknown> }
  | { type: 'usage'; inputTokens?: number; outputTokens?: number }
  | { type: 'error'; message: string }
  | { type: 'done' };

// ============================================================================
// HTTP + SSE streaming
// ============================================================================

export interface LmPassthroughParams {
  serverUrl: string;
  apiKey: string | undefined;
  customHeaders: Record<string, string> | undefined;
  requestTimeout: number;
  model: string;
  messages: readonly vscode.LanguageModelChatMessage[];
  tools: readonly vscode.LanguageModelChatTool[] | undefined;
  progress: vscode.Progress<vscode.LanguageModelResponsePart>;
  token: vscode.CancellationToken;
  log: (message: string) => void;
  verbose: boolean;
  /** Per-request tool-selecting mode to forward upstream. */
  toolMode?: vscode.LanguageModelChatToolMode;
  /** Public modelOptions from `LanguageModelChatRequestOptions` to forward upstream verbatim. */
  modelOptions?: Record<string, unknown>;
  /** Per-request reasoning effort (picker "Thinking Effort") to forward upstream. */
  reasoningEffort?: string;
  /** Per-request context window (prompt budget) to forward upstream. */
  contextSize?: number;
}

export interface LmPassthroughStats {
  totalTextParts: number;
  totalToolCalls: number;
  totalThinkingParts: number;
  /** Token usage reported by the upstream `usage` SSE event, if any. */
  usage?: { prompt: number; completion: number; total: number };
}

export interface LmPassthroughResult {
  stats: LmPassthroughStats;
  /** The serialized request body, captured for error-time curl logging. */
  requestBody: LmChatRequest;
}

/**
 * Send a chat request via the LM passthrough protocol and stream responses
 * back through the VS Code progress API.
 *
 * 1. Serialize messages + tools to JSON
 * 2. POST to `{serverUrl}/lm/chat`
 * 3. Parse SSE lines and report parts
 */
export async function streamLmPassthrough(
  params: LmPassthroughParams
): Promise<LmPassthroughResult> {
  const {
    serverUrl, apiKey, customHeaders, requestTimeout,
    model, messages, tools, progress, token, log, verbose,
    toolMode, modelOptions, reasoningEffort, contextSize,
  } = params;

  const stats: LmPassthroughStats = {
    totalTextParts: 0,
    totalToolCalls: 0,
    totalThinkingParts: 0,
  };

  // 1. Build request body
  const serialized = serializeMessages(messages);
  log(`[lm-passthrough] Serialized ${serialized.length} messages, running tool_result repair...`);
  const repairedMessages = repairToolResultPairing(serialized, log);
  log(`[lm-passthrough] After repair: ${repairedMessages.length} messages`);

  // Always-on diagnostic: report first message structure when it starts
  // with tool_result (the most common cause of upstream 400 errors).
  if (repairedMessages.length > 0) {
    const firstMsg = repairedMessages[0];
    const firstPartTypes = firstMsg.content.map((p) => p.type).join(', ');
    if (firstMsg.content.some((p) => p.type === 'tool_result')) {
      log(
        `[lm-passthrough] WARNING: msg[0] still has tool_result after repair. ` +
        `role=${firstMsg.role}, parts=[${firstPartTypes}]`
      );
      // Dump the orphaned tool_result callIds for debugging
      const orphanIds = firstMsg.content
        .filter((p) => p.type === 'tool_result')
        .map((p) => (p as { callId: string }).callId);
      log(`[lm-passthrough] Orphaned tool_result callIds: ${orphanIds.join(', ')}`);
    }
  }

  // Wire is a transparent pipe: forward whatever Copilot supplied. We only
  // suppress fields Copilot left `undefined`, and we map the only field that
  // requires type translation (`toolMode` enum -> wire string).
  const wireToolMode: 'auto' | 'required' | undefined =
    toolMode === undefined
      ? undefined
      : (toolMode === vscode.LanguageModelChatToolMode.Required ? 'required' : 'auto');

  const body: LmChatRequest = {
    model,
    messages: repairedMessages,
    tools: serializeTools(tools),
    ...(wireToolMode !== undefined ? { tool_mode: wireToolMode } : {}),
    ...(modelOptions !== undefined ? { model_options: modelOptions } : {}),
    ...(reasoningEffort !== undefined ? { reasoning_effort: reasoningEffort } : {}),
    ...(contextSize !== undefined ? { context_size: contextSize } : {}),
  };

  const baseUrl = normalizeBaseUrl(serverUrl);
  const url = `${baseUrl}/v1/lm/chat`;
  const headers = {
    'Content-Type': 'application/json',
    ...buildHeaders(apiKey, customHeaders),
  };

  if (verbose) {
    log(`[lm-passthrough] POST ${url}`);
    log(`[lm-passthrough] ${messages.length} messages, ${tools?.length ?? 0} tools`);
  }

  // 2. Send request
  const controller = new AbortController();
  const cancelListener = token.onCancellationRequested(() => controller.abort());
  const timeoutId = setTimeout(() => controller.abort(), requestTimeout);

  try {
    const response = await fetch(url, {
      method: 'POST',
      headers,
      body: JSON.stringify(body),
      signal: controller.signal,
    });

    if (!response.ok) {
      const errorText = await response.text().catch(() => '(unreadable body)');
      throw new Error(`Upstream error: ${response.status} ${errorText}`);
    }

    if (!response.body) {
      throw new Error('Upstream returned no response body');
    }

    // 3. Parse SSE stream
    const reader = (response.body as ReadableStream<Uint8Array>).getReader();
    const decoder = new TextDecoder();
    let buffer = '';

    while (true) {
      if (token.isCancellationRequested) { break; }

      const { done, value } = await reader.read();
      if (done) { break; }

      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split('\n');
      buffer = lines.pop() || '';

      for (const line of lines) {
        if (!line.startsWith('data: ')) { continue; }
        const jsonStr = line.slice(6).trim();
        if (!jsonStr) { continue; }

        let event: LmSseEvent;
        try {
          event = JSON.parse(jsonStr) as LmSseEvent;
        } catch {
          log(`[lm-passthrough] WARNING: Failed to parse SSE event: ${jsonStr}`);
          continue;
        }

        switch (event.type) {
          case 'text':
            progress.report(new vscode.LanguageModelTextPart(event.value));
            stats.totalTextParts++;
            break;

          case 'tool_call':
            progress.report(
              new vscode.LanguageModelToolCallPart(
                event.callId,
                event.name,
                event.input as Record<string, unknown>
              )
            );
            stats.totalToolCalls++;
            break;

          case 'thinking': {
            const tp = new vscode.LanguageModelThinkingPart(
              event.value,
              event.id,
              event.metadata
            );
            progress.report(tp);
            stats.totalThinkingParts++;
            break;
          }

          case 'usage': {
            // Forward token usage to VS Code's context-window widget via a
            // `usage`-mime LanguageModelDataPart (microsoft/vscode#315394),
            // mirroring the OpenAI/Anthropic paths' reportUsage. Upstream
            // sends inputTokens/outputTokens; normalize to the OpenAI shape.
            const prompt = event.inputTokens ?? 0;
            const completion = event.outputTokens ?? 0;
            const usagePayload = {
              prompt_tokens: prompt,
              completion_tokens: completion,
              total_tokens: prompt + completion,
            };
            stats.usage = { prompt, completion, total: prompt + completion };
            progress.report(
              new vscode.LanguageModelDataPart(
                new TextEncoder().encode(JSON.stringify(usagePayload)),
                USAGE_DATA_PART_MIME_TYPE
              )
            );
            break;
          }

          case 'error':
            throw new Error(event.message);

          case 'done':
            if (verbose) {
              log(
                `[lm-passthrough] Done: ${stats.totalTextParts} text, ` +
                `${stats.totalToolCalls} tool calls, ${stats.totalThinkingParts} thinking`
              );
            }
            return { stats, requestBody: body };
        }
      }
    }

    // Stream ended without explicit 'done' event — still valid
    if (verbose) {
      log(
        `[lm-passthrough] Stream ended: ${stats.totalTextParts} text, ` +
        `${stats.totalToolCalls} tool calls, ${stats.totalThinkingParts} thinking`
      );
    }
    return { stats, requestBody: body };
  } finally {
    clearTimeout(timeoutId);
    cancelListener.dispose();
  }
}
