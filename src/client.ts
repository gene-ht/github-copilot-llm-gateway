import * as vscode from 'vscode';
import {
  OpenAIChatCompletionRequest,
  OpenAIModelsResponse,
  OpenAIUsage,
  GatewayConfig,
} from './types';
import {
  AccumulatedToolCall,
  LegacyFunctionCall,
  ToolCallAccumulator,
  ToolCallDelta,
} from './toolCallAccumulator';

/**
 * Trim trailing slashes and a trailing `/v1` (or `/openai/v1`) segment so the
 * client can safely append `/v1/models` / `/v1/chat/completions` regardless of
 * how the user typed their Server URL in settings.
 */
export function normalizeBaseUrl(rawUrl: string): string {
  let url = rawUrl.trim();
  while (url.endsWith('/')) { url = url.slice(0, -1); }
  url = url.replace(/\/(openai\/)?v1$/i, '');
  return url;
}

/**
 * Strip a leading `Bearer ` (case-insensitive) from the configured API key
 * and trim whitespace. The client always prepends `Bearer`, so users who
 * paste their full `Authorization: Bearer …` header would otherwise send
 * `Bearer Bearer …` and get 401s.
 */
export function normalizeApiKey(rawKey: string | undefined): string {
  if (!rawKey) { return ''; }
  return rawKey.trim().replace(/^Bearer\s+/i, '');
}

/**
 * Build the request header set for the inference server. Authorization is
 * applied first so user-configured `customHeaders` can override it for
 * backends that need a non-Bearer auth scheme (e.g. Azure's `api-key`).
 * Empty/non-string values and empty header names are dropped.
 */
export function buildHeaders(
  apiKey: string | undefined,
  customHeaders: Record<string, string> | undefined
): Record<string, string> {
  const headers: Record<string, string> = {};
  const key = normalizeApiKey(apiKey);
  if (key) {
    headers['Authorization'] = `Bearer ${key}`;
  }
  if (customHeaders) {
    for (const [name, value] of Object.entries(customHeaders)) {
      if (typeof value === 'string' && name.length > 0) {
        headers[name] = value;
      }
    }
  }
  return headers;
}

/**
 * Wire-format chat-completion chunk that downstream consumers see.
 *
 * `usage` is set only on the final chunk of a stream (OpenAI's convention
 * when the request was sent with `stream_options.include_usage: true`).
 * Older or stripped servers may omit it entirely — we surface it when
 * present so VS Code's chat context-window widget can render running
 * token counts (issue #24).
 */
export interface GatewayStreamChunk {
  content: string;
  reasoning_content: string;
  tool_calls: AccumulatedToolCall[];
  finished_tool_calls: AccumulatedToolCall[];
  usage?: OpenAIUsage;
}

/**
 * Re-export so existing imports of `StreamingToolCall` from this module keep
 * working without churn.
 */
export type StreamingToolCall = AccumulatedToolCall;

/**
 * Shape of an OpenAI streaming/non-streaming choice payload that we know
 * how to read. Kept loose; servers vary.
 */
interface ParsedChunk {
  delta?: {
    content?: string;
    reasoning_content?: string;
    tool_calls?: ToolCallDelta[];
    function_call?: LegacyFunctionCall;
  };
  message?: {
    content?: string;
    reasoning_content?: string;
    text?: string;
    tool_calls?: ToolCallDelta[];
    function_call?: LegacyFunctionCall;
  };
  finishReason?: string;
  id?: string;
}

interface ServerErrorPayload {
  error: { message?: string } | string;
}

export type GatewayLogger = (message: string) => void;

const SSE_DATA_PREFIX = 'data: ';
const SSE_DONE_LINE = 'data: [DONE]';
const ERROR_PREFIX = 'Inference server reported an error mid-stream: ';

/**
 * Lifecycle handles for the AbortController + the two timers used by a
 * streaming chat-completion request. Returned by `createStreamTimers` so the
 * main streaming function doesn't have to track them inline.
 */
/**
 * Why the AbortController fired. `null` means it hasn't fired yet. Captured by
 * the closures inside `createStreamTimers` so `streamChatCompletion` can turn
 * the otherwise-opaque "This operation was aborted" error into a descriptive
 * message identifying which timer / cancellation source tripped.
 */
type AbortReason =
  | { kind: 'cancellation' }
  | { kind: 'header-timeout'; timeoutMs: number }
  | { kind: 'inactivity-timeout'; timeoutMs: number; phase: 'pre-headers' | 'streaming' };

interface StreamTimers {
  readonly controller: AbortController;
  readonly resetInactivity: () => void;
  /** Called once response headers arrive — switches to the inactivity timer. */
  readonly onHeadersReceived: () => void;
  /** Clears every outstanding timer + cancellation subscription. */
  readonly dispose: () => void;
  /** What caused `controller.abort()` to fire (null = not aborted). */
  readonly getAbortReason: () => AbortReason | null;
}

/** Human-readable explanation of an AbortReason, used in error messages and logs. */
function describeAbortReason(reason: AbortReason): string {
  switch (reason.kind) {
    case 'cancellation':
      return 'cancelled by VS Code (Copilot Chat stopped or switched the request)';
    case 'header-timeout':
      return `no HTTP headers received within ${reason.timeoutMs}ms (requestTimeout). Increase 'github.copilot.llm-gateway.requestTimeout' or check the server is reachable`;
    case 'inactivity-timeout':
      return reason.phase === 'streaming'
        ? `stream went idle for ${reason.timeoutMs}ms after headers arrived (requestTimeout). The server stopped sending SSE chunks mid-response — increase requestTimeout or check the upstream model`
        : `no chunks received within ${reason.timeoutMs}ms of headers (requestTimeout)`;
  }
}

/**
 * Throw a descriptive error for a failed chat-completion response, including
 * the response body when the server provided one. Pulled out of
 * `streamChatCompletion` so the main function stays under the
 * cognitive-complexity budget.
 */
async function assertChatStreamResponseOk(response: Response): Promise<void> {
  if (response.ok && response.body) { return; }
  if (!response.ok) {
    const errorText = await response.text();
    throw new Error(`Chat completion failed: ${response.status} ${response.statusText} - ${errorText}`);
  }
  throw new Error('Response body is null');
}

/**
 * Minimum undici headersTimeout/bodyTimeout (5 minutes). Node's built-in
 * `fetch` uses an undici `Agent` whose default headersTimeout is 300s. For
 * thinking models with very long first-token latency, requestTimeout can be
 * configured to be larger; we mirror that into undici so undici doesn't kill
 * the connection before our own timer fires. We never go *below* 5 minutes
 * even if requestTimeout is shorter, so we don't make undici more aggressive
 * than its safe default.
 */
const UNDICI_TIMEOUT_FLOOR_MS = 300_000;

export class GatewayClient {
  private config: GatewayConfig;
  private readonly log: GatewayLogger;
  // Cached undici Agent; rebuilt when requestTimeout changes so headersTimeout
  // tracks user configuration. Typed loosely because undici types are only
  // available via the optional `undici` peer; we use the runtime require.
  private dispatcher: unknown | undefined;
  private dispatcherTimeoutMs: number | undefined;

  constructor(config: GatewayConfig, logger?: GatewayLogger) {
    this.config = config;
    this.log = logger ?? (() => { /* no-op */ });
  }

  public updateConfig(config: GatewayConfig): void {
    this.config = config;
  }

  /**
   * Lazily build (or rebuild) an undici Agent whose headersTimeout and
   * bodyTimeout are at least UNDICI_TIMEOUT_FLOOR_MS (5min) and grow to
   * match requestTimeout when the user configures a longer value.
   *
   * Returns undefined if the undici module can't be loaded (e.g. browser
   * environment, unsupported Node version), in which case fetch falls back
   * to the default global dispatcher.
   */
  private getDispatcher(): unknown | undefined {
    const desired = Math.max(UNDICI_TIMEOUT_FLOOR_MS, this.config.requestTimeout);
    if (this.dispatcher && this.dispatcherTimeoutMs === desired) {
      return this.dispatcher;
    }
    try {
      // `undici` is bundled into Node's runtime; require it dynamically so
      // we don't add a hard dependency and so esbuild leaves it external.
      // eslint-disable-next-line @typescript-eslint/no-require-imports, @typescript-eslint/no-var-requires
      const undici = require('undici') as { Agent: new (opts: Record<string, unknown>) => unknown };
      this.dispatcher = new undici.Agent({
        headersTimeout: desired,
        bodyTimeout: desired,
        // Leave connect/keepalive at undici defaults; we only care about the
        // two timeouts that were biting us (UND_ERR_HEADERS_TIMEOUT).
      });
      this.dispatcherTimeoutMs = desired;
      this.log(`undici Agent (re)created with headersTimeout=${desired}ms bodyTimeout=${desired}ms`);
      return this.dispatcher;
    } catch (err) {
      this.log(`failed to load undici, falling back to default fetch dispatcher: ${err instanceof Error ? err.message : String(err)}`);
      return undefined;
    }
  }

  /**
   * Fetch available models from the server's models endpoint.
   *
   * Tries `/v1/models` first and falls back to `/models` so the client works
   * against servers that mount the OpenAI API at the root.
   */
  public async fetchModels(cancellationToken?: vscode.CancellationToken): Promise<OpenAIModelsResponse> {
    const base = normalizeBaseUrl(this.config.serverUrl);
    const candidates = [`${base}/v1/models`, `${base}/models`];
    let lastError: Error | undefined;

    for (let i = 0; i < candidates.length; i++) {
      const url = candidates[i];
      const isLast = i === candidates.length - 1;
      try {
        const result = await this.tryFetchModels(url, isLast, cancellationToken);
        if (result) { return result; }
      } catch (error) {
        lastError = error instanceof Error ? error : new Error(String(error));
        if (isLast) { break; }
      }
    }

    const message = lastError?.message ?? 'unknown error';
    throw new Error(`Failed to connect to inference server at ${base}: ${message}`);
  }

  /**
   * Attempt a single model-fetch against `url`. Returns the parsed response
   * on success, `undefined` if the endpoint returned 404 and `allowFallback`
   * is true, or throws on any other failure.
   */
  private async tryFetchModels(
    url: string,
    isLast: boolean,
    cancellationToken?: vscode.CancellationToken
  ): Promise<OpenAIModelsResponse | undefined> {
    const response = await this.fetchWithTimeout(url, {
      method: 'GET',
      headers: this.getHeaders(),
    }, cancellationToken);

    if (response.ok) {
      return await response.json();
    }

    if (response.status === 404 && !isLast) {
      this.log(`Models endpoint not found at ${url}, trying fallback...`);
      return undefined;
    }

    const bodyText = await response.text().catch(() => '');
    const truncated = bodyText.length > 200 ? bodyText.slice(0, 200) + '...' : bodyText;
    const suffix = truncated ? ' — ' + truncated : '';
    throw new Error(
      `Failed to fetch models from ${url}: ${response.status} ${response.statusText}${suffix}`
    );
  }

  /**
   * Stream chat completions from `/v1/chat/completions`. Tool calls are
   * accumulated by index across chunks (their `id` may arrive later than
   * their name/arguments). Manages two timers explicitly:
   *   - the configured `requestTimeout` applies until headers arrive,
   *   - then a per-read inactivity timer of the same duration is reset on
   *     each chunk so long generations aren't aborted mid-stream.
   */
  public async *streamChatCompletion(
    request: OpenAIChatCompletionRequest,
    cancellationToken: vscode.CancellationToken
  ): AsyncGenerator<GatewayStreamChunk, void, unknown> {
    const url = `${normalizeBaseUrl(this.config.serverUrl)}/v1/chat/completions`;
    const accumulator = new ToolCallAccumulator();
    const timers = this.createStreamTimers(cancellationToken);

    // Reproducible curl + body dump is logged only on error (see catch block).
    // Logging on every request floods the output channel and rewrites the
    // workspace-local tmp/llm-gateway-debug-request.json file each turn, which
    // is noisy and risks leaking conversation content in normal sessions.

    try {
      const dispatcher = this.getDispatcher();
      const response = await fetch(url, {
        method: 'POST',
        headers: { ...this.getHeaders(), 'Content-Type': 'application/json' },
        // `stream_options.include_usage` tells OpenAI-compatible servers to
        // emit a final SSE chunk containing `usage` totals once the model
        // finishes. We forward that to VS Code's chat context-window widget
        // (issue #24). Servers that don't recognise the option simply
        // ignore it; servers behind aggressive proxies may strip it.
        body: JSON.stringify({
          ...request,
          stream: true,
          stream_options: { ...(request.stream_options as object | undefined), include_usage: true },
        }),
        signal: timers.controller.signal,
        // Node-only: pass an undici Agent so headersTimeout/bodyTimeout track
        // the user's requestTimeout (instead of undici's hard-coded 5min).
        ...(dispatcher ? { dispatcher } : {}),
      } as RequestInit);

      // Headers received — switch from the request-deadline timer to the
      // per-chunk inactivity timer so long generations aren't aborted.
      timers.onHeadersReceived();

      await assertChatStreamResponseOk(response);

      yield* this.readChatStreamChunks(response.body!, accumulator, cancellationToken, timers.resetInactivity);

      const remaining = accumulator.drain(true);
      if (remaining.length > 0) {
        yield { content: '', reasoning_content: '', tool_calls: [], finished_tool_calls: remaining };
      }
    } catch (error) {
      if (error instanceof Error) {
        const reason = timers.getAbortReason();
        let detail = reason ? ` [${describeAbortReason(reason)}]` : '';
        // `fetch failed` from undici is opaque — surface error.cause (low-level
        // SystemError with code like ECONNRESET / UND_ERR_SOCKET / EPIPE) so
        // users can diagnose vs guess.
        if (!reason && error.message === 'fetch failed') {
          const cause = (error as Error & { cause?: unknown }).cause;
          if (cause instanceof Error) {
            const code = (cause as Error & { code?: string }).code;
            detail = ` [cause: ${cause.name}: ${cause.message}${code ? ` (code=${code})` : ''}]`;
          } else if (cause !== undefined) {
            detail = ` [cause: ${String(cause)}]`;
          }
        }
        if (reason) {
          this.log(`Chat stream aborted: ${describeAbortReason(reason)} (url=${url})`);
        } else if (detail) {
          this.log(`Chat stream error${detail} (url=${url})`);
        }
        // Log the reproducible curl + dump body now, on error, so the user
        // can replay the failing request manually. We skip user-cancellation
        // aborts because they aren't actionable bugs.
        if (!reason || reason.kind !== 'cancellation') {
          this.logReproducibleCurl(url, request);
        }
        throw new Error(`Chat completion request failed: ${error.message}${detail}`);
      }
      throw error;
    } finally {
      timers.dispose();
    }
  }

  /**
   * Read SSE chunks off the response body until done or cancelled, parsing
   * each line through {@link processSSELine}. Split out of
   * `streamChatCompletion` so the parent function stays under SonarCloud's
   * cognitive-complexity budget.
   */
  private async *readChatStreamChunks(
    body: ReadableStream<Uint8Array>,
    accumulator: ToolCallAccumulator,
    cancellationToken: vscode.CancellationToken,
    resetInactivity: () => void
  ): AsyncGenerator<GatewayStreamChunk, void, unknown> {
    const reader = body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';

    while (true) {
      if (cancellationToken.isCancellationRequested) {
        reader.cancel();
        return;
      }

      const { done, value } = await reader.read();
      if (done) { return; }

      resetInactivity();

      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split('\n');
      buffer = lines.pop() ?? '';

      for (const line of lines) {
        const result = this.processSSELine(line, accumulator);
        if (result) { yield result; }
      }
    }
  }

  /**
   * Wire up the AbortController, request-deadline timer, and per-chunk
   * inactivity timer used by the streaming request. The two timers run
   * sequentially: the request timer fires until headers arrive, then the
   * inactivity timer takes over and is reset on each chunk.
   */
  private createStreamTimers(cancellationToken: vscode.CancellationToken): StreamTimers {
    const controller = new AbortController();
    const timeoutMs = this.config.requestTimeout;
    let abortReason: AbortReason | null = null;
    let headersReceived = false;

    const abortWith = (reason: AbortReason): void => {
      if (abortReason) { return; } // First reason wins
      abortReason = reason;
      controller.abort();
    };

    const cancelSub = cancellationToken.onCancellationRequested(() =>
      abortWith({ kind: 'cancellation' })
    );
    const headerTimeoutId = setTimeout(
      () => abortWith({ kind: 'header-timeout', timeoutMs }),
      timeoutMs
    );
    let inactivityTimeoutId: ReturnType<typeof setTimeout> | undefined;
    const resetInactivity = (): void => {
      if (inactivityTimeoutId) { clearTimeout(inactivityTimeoutId); }
      inactivityTimeoutId = setTimeout(
        () => abortWith({
          kind: 'inactivity-timeout',
          timeoutMs,
          phase: headersReceived ? 'streaming' : 'pre-headers',
        }),
        timeoutMs
      );
    };
    return {
      controller,
      resetInactivity,
      onHeadersReceived: () => {
        headersReceived = true;
        clearTimeout(headerTimeoutId);
        resetInactivity();
      },
      dispose: () => {
        clearTimeout(headerTimeoutId);
        if (inactivityTimeoutId) { clearTimeout(inactivityTimeoutId); }
        cancelSub.dispose();
      },
      getAbortReason: () => abortReason,
    };
  }

  /**
   * Process one SSE line. Returns a chunk to yield, or null if there's
   * nothing to emit. Throws if the server sent an inline error payload —
   * the caller can then surface a real error instead of an empty stream.
   */
  private processSSELine(line: string, accumulator: ToolCallAccumulator): GatewayStreamChunk | null {
    const trimmed = line.trim();
    if (trimmed === '' || trimmed === SSE_DONE_LINE) { return null; }
    if (trimmed.startsWith('event:')) { return null; }
    if (!trimmed.startsWith(SSE_DATA_PREFIX)) { return null; }

    const data = trimmed.slice(SSE_DATA_PREFIX.length);

    let parsed: unknown;
    try {
      parsed = JSON.parse(data);
    } catch {
      this.log(`Failed to parse SSE chunk: ${data}`);
      return null;
    }
    if (!parsed || typeof parsed !== 'object') { return null; }

    const obj = parsed as Record<string, unknown>;

    // Inline error payload: `{ "error": { "message": "..." } }`. Distinguished
    // from a normal chunk (which has `choices`).
    if ('error' in obj && !('choices' in obj)) {
      const message = extractServerErrorMessage(obj as unknown as ServerErrorPayload);
      throw new Error(`${ERROR_PREFIX}${message}`);
    }

    return this.dispatchParsedChunk(obj, accumulator);
  }

  private dispatchParsedChunk(
    obj: Record<string, unknown>,
    accumulator: ToolCallAccumulator
  ): GatewayStreamChunk | null {
    const usage = extractUsage(obj.usage);
    const choices = Array.isArray(obj.choices) ? obj.choices : undefined;
    const choice = choices?.[0] as Record<string, unknown> | undefined;

    // OpenAI's stream-with-include_usage convention puts the totals on a
    // trailing chunk with an empty `choices` array — surface it as a
    // usage-only stream chunk so the provider can forward it to the chat
    // context-window widget (issue #24).
    if (!choice) {
      if (!usage) { return null; }
      return {
        content: '',
        reasoning_content: '',
        tool_calls: [],
        finished_tool_calls: [],
        usage,
      };
    }

    const chunk: ParsedChunk = {
      delta: choice.delta as ParsedChunk['delta'],
      message: choice.message as ParsedChunk['message'],
      finishReason: choice.finish_reason as string | undefined,
      id: typeof obj.id === 'string' ? obj.id : undefined,
    };

    if (chunk.delta) {
      const { content, reasoningContent, finishedToolCalls } = this.applyDeltaChoice(chunk, accumulator);
      return {
        content,
        reasoning_content: reasoningContent,
        tool_calls: [],
        finished_tool_calls: finishedToolCalls,
        ...(usage ? { usage } : {}),
      };
    }
    if (chunk.message) {
      const { content, reasoningContent, finishedToolCalls } = this.applyMessageChoice(chunk, accumulator);
      return {
        content,
        reasoning_content: reasoningContent,
        tool_calls: [],
        finished_tool_calls: finishedToolCalls,
        ...(usage ? { usage } : {}),
      };
    }
    return null;
  }

  private applyDeltaChoice(
    parsed: ParsedChunk,
    accumulator: ToolCallAccumulator
  ): { content: string; reasoningContent: string; finishedToolCalls: AccumulatedToolCall[] } {
    const delta = parsed.delta!;
    const finishedToolCalls: AccumulatedToolCall[] = [];

    if (Array.isArray(delta.tool_calls)) {
      for (const tc of delta.tool_calls) {
        accumulator.applyDelta(tc);
      }
    }

    if (delta.function_call) {
      accumulator.applyLegacy(delta.function_call, parsed.id ?? '');
    }

    if (parsed.finishReason === 'tool_calls' || parsed.finishReason === 'function_call') {
      finishedToolCalls.push(...accumulator.drain());
    }

    return {
      content: delta.content ?? '',
      reasoningContent: delta.reasoning_content ?? '',
      finishedToolCalls,
    };
  }

  private applyMessageChoice(
    parsed: ParsedChunk,
    accumulator: ToolCallAccumulator
  ): { content: string; reasoningContent: string; finishedToolCalls: AccumulatedToolCall[] } {
    const message = parsed.message!;
    const finishedToolCalls: AccumulatedToolCall[] = [];

    if (Array.isArray(message.tool_calls)) {
      finishedToolCalls.push(...accumulator.applyComplete(message.tool_calls));
    }

    if (message.function_call) {
      const completed = accumulator.applyComplete([
        { index: 0, id: parsed.id, function: message.function_call },
      ]);
      finishedToolCalls.push(...completed);
    }

    return {
      content: message.content ?? message.text ?? '',
      reasoningContent: message.reasoning_content ?? '',
      finishedToolCalls,
    };
  }

  private getHeaders(): Record<string, string> {
    return buildHeaders(this.config.apiKey, this.config.customHeaders);
  }

  /**
   * Log a curl command that reproduces the chat-completion request.
   * Sensitive header values (Authorization, Bearer tokens, API keys) are
   * masked so the output is safe to share in bug reports. The request body
   * is written to a workspace-local tmp file to avoid shell quoting issues
   * with large JSON payloads.
   */
  public logReproducibleCurl(url: string, request: Record<string, unknown>): void {
    try {
      const headers = { ...this.getHeaders(), 'Content-Type': 'application/json' };
      const maskedHeaders = Object.entries(headers).map(([k, v]) => {
        const lower = k.toLowerCase();
        if (lower === 'authorization' || lower === 'x-api-key' || lower === 'api-key') {
          return `-H '${k}: ***REDACTED***'`;
        }
        return `-H '${k}: ${v}'`;
      });
      // Only inject stream_options for OpenAI-style requests. Anthropic uses
      // a different mechanism for usage reporting and rejects unknown fields.
      const isAnthropicEndpoint = url.includes('/messages');
      const body = JSON.stringify({
        ...request,
        stream: true,
        ...(isAnthropicEndpoint ? {} : { stream_options: { ...(request.stream_options as object | undefined), include_usage: true } }),
      });

      // For large bodies, write to the extension's own tmp/ directory;
      // for smaller ones, inline. Uses __dirname (out/) to resolve the
      // extension root so we never write into the host window's workspace.
      const MAX_INLINE_BODY = 50000;
      let bodyArg: string;
      if (body.length <= MAX_INLINE_BODY) {
        bodyArg = `-d '${body.replace(/'/g, "'\\''")}'`;
      } else {
        const path = require('path');
        const fs = require('fs');
        // __dirname is <extensionRoot>/out at runtime
        const extensionRoot = path.resolve(__dirname, '..');
        const debugDir = path.join(extensionRoot, 'tmp');
        const debugPath = path.join(debugDir, 'llm-gateway-debug-request.json');
        try {
          fs.mkdirSync(debugDir, { recursive: true });
          fs.writeFileSync(debugPath, body, 'utf8');
          bodyArg = `-d @${debugPath}`;
        } catch {
          bodyArg = `-d @${debugPath}  # Failed to write (${body.length} chars)`;
        }
      }

      const curl = [
        `curl -X POST '${url}'`,
        ...maskedHeaders,
        bodyArg,
      ].join(' \\\n  ');

      this.log(`\n=== Reproducible curl (headers redacted) ===\n${curl}\n=== End curl ===`);

      // Also log the first few messages of the body for quick inspection
      // without needing to reconstruct the full payload.
      const msgs = request.messages;
      if (Array.isArray(msgs) && msgs.length > 0) {
        const preview = msgs.slice(0, 3).map((m, i) => {
          const role = typeof m === 'object' && m !== null ? (m as Record<string, unknown>).role : '?';
          const toolCallId = typeof m === 'object' && m !== null ? (m as Record<string, unknown>).tool_call_id : undefined;
          const hasToolCalls = typeof m === 'object' && m !== null && Array.isArray((m as Record<string, unknown>).tool_calls);
          return `  [${i}] role=${role}${toolCallId ? `, tool_call_id=${toolCallId}` : ''}${hasToolCalls ? ', has_tool_calls' : ''}`;
        });
        this.log(`First ${Math.min(3, msgs.length)} of ${msgs.length} messages:\n${preview.join('\n')}`);
      }
    } catch {
      // Best-effort — never let curl logging itself cause a failure.
    }
  }

  /**
   * Fetch wrapper with a fixed total-request timeout and optional
   * cancellation-token wiring. Used for non-streaming requests like the
   * model list. Streaming requests manage their own timers in
   * `streamChatCompletion`.
   */
  private async fetchWithTimeout(
    url: string,
    options: RequestInit,
    cancellationToken?: vscode.CancellationToken
  ): Promise<Response> {
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), this.config.requestTimeout);
    const cancelSub = cancellationToken?.onCancellationRequested(() => controller.abort());

    try {
      const dispatcher = this.getDispatcher();
      return await fetch(url, {
        ...options,
        signal: controller.signal,
        ...(dispatcher ? { dispatcher } : {}),
      } as RequestInit);
    } finally {
      clearTimeout(timeoutId);
      cancelSub?.dispose();
    }
  }
}

/**
 * Validate and shape a raw `usage` payload from the inference server. Coerces
 * NaN/missing fields to 0 and clamps negative sentinel values (some servers
 * emit -1 when totals aren't yet available) so VS Code's chat context-window
 * widget doesn't render nonsensical numbers (issue #24).
 */
export function extractUsage(raw: unknown): OpenAIUsage | undefined {
  if (!raw || typeof raw !== 'object') { return undefined; }
  const obj = raw as Record<string, unknown>;
  const prompt = toNonNegativeNumber(obj.prompt_tokens);
  const completion = toNonNegativeNumber(obj.completion_tokens);
  const total = toNonNegativeNumber(obj.total_tokens, prompt + completion);

  // Some servers omit `prompt_tokens` and `completion_tokens` entirely.
  // Require at least one signal so we don't emit an all-zero usage frame
  // that would briefly reset the context-window widget to 0% mid-stream.
  if (obj.prompt_tokens === undefined && obj.completion_tokens === undefined && obj.total_tokens === undefined) {
    return undefined;
  }

  const detailsRaw = obj.prompt_tokens_details;
  const cached = detailsRaw && typeof detailsRaw === 'object'
    ? toNonNegativeNumber((detailsRaw as Record<string, unknown>).cached_tokens)
    : 0;

  return {
    prompt_tokens: prompt,
    completion_tokens: completion,
    total_tokens: total,
    prompt_tokens_details: { cached_tokens: cached },
  };
}

function toNonNegativeNumber(value: unknown, fallback = 0): number {
  if (typeof value !== 'number' || !Number.isFinite(value)) { return fallback; }
  return value < 0 ? 0 : value;
}

function extractServerErrorMessage(payload: ServerErrorPayload): string {
  const err = payload.error;
  if (typeof err === 'string') { return err; }
  if (err && typeof err === 'object' && 'message' in err && typeof err.message === 'string') {
    return err.message;
  }
  return JSON.stringify(err);
}
