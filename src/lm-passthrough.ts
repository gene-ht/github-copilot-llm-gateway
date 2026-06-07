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
 */
export interface LmChatRequest {
  model: string;
  messages: LmMessagePayload[];
  tools?: LmToolPayload[];
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
// SSE response events
// ============================================================================

/**
 * Typed union of all SSE event payloads the upstream can send.
 */
export type LmSseEvent =
  | { type: 'text'; value: string }
  | { type: 'tool_call'; callId: string; name: string; input: unknown }
  | { type: 'thinking'; value: string; id?: string; metadata?: Record<string, unknown> }
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
}

export interface LmPassthroughStats {
  totalTextParts: number;
  totalToolCalls: number;
  totalThinkingParts: number;
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
): Promise<LmPassthroughStats> {
  const {
    serverUrl, apiKey, customHeaders, requestTimeout,
    model, messages, tools, progress, token, log, verbose,
  } = params;

  const stats: LmPassthroughStats = {
    totalTextParts: 0,
    totalToolCalls: 0,
    totalThinkingParts: 0,
  };

  // 1. Build request body
  const body: LmChatRequest = {
    model,
    messages: serializeMessages(messages),
    tools: serializeTools(tools),
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

          case 'error':
            throw new Error(event.message);

          case 'done':
            if (verbose) {
              log(
                `[lm-passthrough] Done: ${stats.totalTextParts} text, ` +
                `${stats.totalToolCalls} tool calls, ${stats.totalThinkingParts} thinking`
              );
            }
            return stats;
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
    return stats;
  } finally {
    clearTimeout(timeoutId);
    cancelListener.dispose();
  }
}
