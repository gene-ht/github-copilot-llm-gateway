/**
 * Local HTTP proxy server for Copilot background services.
 *
 * Copilot's background services (title generation, summarization, intent
 * classification, commit messages, etc.) use `getChatEndpoint('copilot-fast')`
 * which bypasses the VS Code `LanguageModelChat` API and hits
 * `api.githubcopilot.com` directly. When Copilot quota is exhausted these
 * services all fail.
 *
 * This module starts a local HTTP server that Copilot can be redirected to
 * via `github.copilot.advanced.debug.overrideProxyUrl`. Incoming requests
 * are model-mapped and forwarded to the user's upstream inference server
 * using a plain `fetch` + SSE pipe — no dependency on `GatewayClient` or
 * any VS Code API, so it is fully unit-testable.
 */

import * as http from 'node:http';
import { normalizeBaseUrl, buildHeaders } from './client';

/** Logger callback — wired to the output channel by the caller. */
export type ProxyLogger = (message: string) => void;

/** Configuration the proxy needs from the outside world. */
export interface ProxyUpstreamConfig {
  serverUrl: string;
  apiKey?: string;
  customHeaders?: Record<string, string>;
  requestTimeout: number;
}

/** Copilot proxy–specific settings. */
export interface CopilotProxyConfig {
  enabled: boolean;
  modelMapping: Record<string, string>;
}


/**
 * Build fake quota response headers that tell Copilot "quota is available".
 * Copilot's ChatQuotaService.processQuotaHeaders() reads
 * `x-quota-snapshot-chat` and parses it as URL search params. When
 * `rem > 0` and `ovPerm=false`, it clears the quota-exceeded state.
 */
function quotaHeaders(): Record<string, string> {
  return {
    'x-quota-snapshot-chat': 'ent=9999&rem=99.0&ovPerm=false&rst=2099-12-31T00:00:00Z',
  };
}

/**
 * Read the full request body from an `IncomingMessage`. Rejects if the body
 * exceeds `maxBytes` (default 10 MB) to prevent OOM on malformed requests.
 */
function readBody(req: http.IncomingMessage, maxBytes = 10 * 1024 * 1024): Promise<string> {
  return new Promise((resolve, reject) => {
    const chunks: Buffer[] = [];
    let total = 0;
    req.on('data', (chunk: Buffer) => {
      total += chunk.length;
      if (total > maxBytes) {
        req.destroy();
        reject(new Error('Request body too large'));
        return;
      }
      chunks.push(chunk);
    });
    req.on('end', () => resolve(Buffer.concat(chunks).toString('utf8')));
    req.on('error', reject);
  });
}

/**
 * Write a JSON error response matching the OpenAI error format.
 */
function writeJsonError(res: http.ServerResponse, status: number, message: string): void {
  const body = JSON.stringify({ error: { message, type: 'proxy_error', code: status } });
  res.writeHead(status, {
    'Content-Type': 'application/json',
    'Content-Length': Buffer.byteLength(body),
  });
  res.end(body);
}

/**
 * Resolve the model name Copilot requests to the upstream model the user
 * configured. Tries exact match first, then prefix match (e.g.
 * `gpt-4o-mini-2024-07-18` matches a `gpt-4o-mini` mapping key).
 * Falls back to `requestedModel` unchanged when no mapping exists.
 */
export function mapModel(
  requestedModel: string,
  mapping: Record<string, string>,
  fallbackModel?: string
): string {
  // Exact match
  const exact = mapping[requestedModel];
  if (exact) { return exact; }

  // Prefix match: Copilot appends date suffixes like `-2024-07-18`
  for (const [key, value] of Object.entries(mapping)) {
    if (value && requestedModel.startsWith(key)) {
      return value;
    }
  }

  return fallbackModel ?? requestedModel;
}

export class CopilotProxyServer {
  private server: http.Server | undefined;
  private _port: number | undefined;
  private upstreamConfig: ProxyUpstreamConfig;
  private proxyConfig: CopilotProxyConfig;
  private readonly log: ProxyLogger;
  /** Fallback model when a mapping yields '' — set externally by the caller. */
  public fallbackModel: string | undefined;

  constructor(
    upstreamConfig: ProxyUpstreamConfig,
    proxyConfig: CopilotProxyConfig,
    logger?: ProxyLogger
  ) {
    this.upstreamConfig = upstreamConfig;
    this.proxyConfig = proxyConfig;
    this.log = logger ?? (() => { /* no-op */ });
  }

  get port(): number | undefined { return this._port; }
  get isRunning(): boolean { return this.server?.listening === true; }

  updateConfig(upstreamConfig: ProxyUpstreamConfig, proxyConfig: CopilotProxyConfig): void {
    this.upstreamConfig = upstreamConfig;
    this.proxyConfig = proxyConfig;
  }

  /**
   * Start the HTTP server on an auto-assigned port. Resolves with the
   * assigned port number once the server is listening.
   */
  start(): Promise<number> {
    return new Promise((resolve, reject) => {
      const server = http.createServer((req, res) => {
        void this.handleRequest(req, res);
      });
      server.on('error', reject);
      server.listen(0, '127.0.0.1', () => {
        const addr = server.address();
        if (!addr || typeof addr === 'string') {
          reject(new Error('Failed to get server address'));
          return;
        }
        this._port = addr.port;
        this.server = server;
        this.log(`Copilot proxy server listening on 127.0.0.1:${this._port}`);
        resolve(this._port);
      });
    });
  }

  /**
   * Gracefully stop the HTTP server.
   */
  stop(): Promise<void> {
    return new Promise((resolve) => {
      if (!this.server) { resolve(); return; }
      this.server.close(() => {
        this.log('Copilot proxy server stopped');
        this.server = undefined;
        this._port = undefined;
        resolve();
      });
    });
  }

  dispose(): void {
    void this.stop();
  }

  // ---------- request routing ----------

  private async handleRequest(req: http.IncomingMessage, res: http.ServerResponse): Promise<void> {
    const url = req.url ?? '';
    this.log(`← ${req.method} ${url}`);

    // CORS preflight
    if (req.method === 'OPTIONS') {
      res.writeHead(204, {
        'Access-Control-Allow-Origin': '*',
        'Access-Control-Allow-Methods': 'GET, POST, OPTIONS',
        'Access-Control-Allow-Headers': '*',
      });
      res.end();
      return;
    }

    // ---- Selective proxy ----
    // 1. POST /chat/completions → intercept & forward to upstream /v1/chat/completions (OpenAI)
    // 2. POST /v1/messages → intercept & forward to upstream /v1/messages (Anthropic)
    // 3. Everything else → transparent pass-through
    //
    // /models is NOT intercepted — Copilot's own model metadata stays intact.
    // To make sub-agents use upstream models, configure VS Code settings
    // like `chat.exploreAgent.defaultModel` to select models from the
    // Gateway's LanguageModelChatProvider (vendor=copilot-llm-gateway).

    if (req.method === 'POST' && url.includes('/chat/completions')) {
      await this.handleChatCompletions(req, res);
      return;
    }

    // Anthropic Messages API (used by claude-* models, e.g. background tasks)
    if (req.method === 'POST' && url.includes('/v1/messages')) {
      await this.handleAnthropicMessages(req, res);
      return;
    }

    // All other requests → transparent pass-through to Copilot's real API.
    // If the pass-through fails (network error, HMAC mismatch from CLI),
    // fall back to a safe empty response so Copilot doesn't break.
    await this.passThrough(req, res);
  }

  // ---------- POST /chat/completions ----------

  private async handleChatCompletions(
    req: http.IncomingMessage,
    res: http.ServerResponse
  ): Promise<void> {
    let body: Record<string, unknown>;
    try {
      const raw = await readBody(req);
      body = JSON.parse(raw);
    } catch {
      writeJsonError(res, 400, 'Invalid JSON body');
      return;
    }

    // Map model name
    const requestedModel = typeof body.model === 'string' ? body.model : '';
    const model = mapModel(requestedModel, this.proxyConfig.modelMapping, this.fallbackModel);
    this.log(`Proxy: ${requestedModel} → ${model}`);

    // Build upstream request
    const upstreamUrl = `${normalizeBaseUrl(this.upstreamConfig.serverUrl)}/v1/chat/completions`;
    const upstreamHeaders = {
      ...buildHeaders(this.upstreamConfig.apiKey, this.upstreamConfig.customHeaders),
      'Content-Type': 'application/json',
    };

    const upstreamBody = JSON.stringify({
      ...body,
      model,
      stream: true,
      stream_options: { include_usage: true },
    });

    // Abort upstream request if the client disconnects
    const controller = new AbortController();
    req.on('close', () => controller.abort());

    const timeout = setTimeout(() => controller.abort(), this.upstreamConfig.requestTimeout);

    try {
      const upstream = await fetch(upstreamUrl, {
        method: 'POST',
        headers: upstreamHeaders,
        body: upstreamBody,
        signal: controller.signal,
      });

      clearTimeout(timeout);

      if (!upstream.ok) {
        const errText = await upstream.text().catch(() => '');
        this.log(`Upstream error: ${upstream.status} ${upstream.statusText} - ${errText.slice(0, 500)}`);
        writeJsonError(res, upstream.status, `Upstream: ${upstream.status} ${upstream.statusText}`);
        return;
      }

      if (!upstream.body) {
        writeJsonError(res, 502, 'Upstream returned no body');
        return;
      }

      // Stream SSE back to Copilot
      res.writeHead(200, {
        'Content-Type': 'text/event-stream',
        'Cache-Control': 'no-cache',
        'Connection': 'keep-alive',
        'Access-Control-Allow-Origin': '*',
        ...quotaHeaders(),
      });

      await this.pipeSSE(upstream.body, res, controller.signal);
    } catch (err) {
      clearTimeout(timeout);
      if (controller.signal.aborted) {
        // Client disconnected or timeout — don't log as error
        if (!res.headersSent) {
          writeJsonError(res, 499, 'Client disconnected');
        }
        return;
      }
      const message = err instanceof Error ? err.message : String(err);
      this.log(`Proxy fetch error: ${message}`);
      if (!res.headersSent) {
        writeJsonError(res, 502, `Upstream error: ${message}`);
      }
    }
  }

  // ---------- POST /v1/messages (Anthropic Messages API) ----------

  /**
   * Handle Anthropic Messages API requests (used by claude-* models including
   * sub-agents). Maps the model name and forwards to upstream /v1/messages.
   * The upstream must support Anthropic Messages API format.
   */
  private async handleAnthropicMessages(
    req: http.IncomingMessage,
    res: http.ServerResponse
  ): Promise<void> {
    let body: Record<string, unknown>;
    try {
      const raw = await readBody(req);
      body = JSON.parse(raw);
    } catch {
      writeJsonError(res, 400, 'Invalid JSON body');
      return;
    }

    // Map model name (same logic as /chat/completions)
    const requestedModel = typeof body.model === 'string' ? body.model : '';
    const model = mapModel(requestedModel, this.proxyConfig.modelMapping, this.fallbackModel);
    this.log(`Anthropic: ${requestedModel} → ${model}`);

    // Build upstream request to /v1/messages
    const upstreamUrl = `${normalizeBaseUrl(this.upstreamConfig.serverUrl)}/v1/messages`;
    const upstreamHeaders = {
      ...buildHeaders(this.upstreamConfig.apiKey, this.upstreamConfig.customHeaders),
      'Content-Type': 'application/json',
    };

    // Preserve the original Anthropic format, just rewrite model
    const upstreamBody = JSON.stringify({
      ...body,
      model,
    });

    // Abort upstream if client disconnects
    const controller = new AbortController();
    req.on('close', () => controller.abort());

    const timeout = setTimeout(() => controller.abort(), this.upstreamConfig.requestTimeout);

    try {
      const upstream = await fetch(upstreamUrl, {
        method: 'POST',
        headers: upstreamHeaders,
        body: upstreamBody,
        signal: controller.signal,
      });

      clearTimeout(timeout);

      if (!upstream.ok) {
        const errText = await upstream.text().catch(() => '');
        this.log(`Anthropic upstream error: ${upstream.status} ${upstream.statusText} - ${errText.slice(0, 500)}`);
        writeJsonError(res, upstream.status, `Upstream: ${upstream.status} ${upstream.statusText}`);
        return;
      }

      // Forward upstream response headers (preserve content-type for SSE vs JSON)
      const responseHeaders: Record<string, string> = { ...quotaHeaders() };
      upstream.headers.forEach((value, key) => {
        const lower = key.toLowerCase();
        if (lower !== 'transfer-encoding' && lower !== 'connection') {
          responseHeaders[key] = value;
        }
      });

      const isStreaming = body.stream === true ||
        (upstream.headers.get('content-type') ?? '').includes('event-stream');

      if (isStreaming && upstream.body) {
        // Stream SSE through unchanged — upstream already returns Anthropic SSE format
        res.writeHead(upstream.status, responseHeaders);
        const reader = upstream.body.getReader();
        try {
          while (!controller.signal.aborted) {
            const { done, value } = await reader.read();
            if (done) { break; }
            res.write(value);
          }
        } finally {
          reader.releaseLock();
          res.end();
        }
      } else {
        // Non-streaming: just return JSON
        const text = await upstream.text();
        res.writeHead(upstream.status, responseHeaders);
        res.end(text);
      }
    } catch (err) {
      clearTimeout(timeout);
      if (controller.signal.aborted) {
        if (!res.headersSent) {
          writeJsonError(res, 499, 'Client disconnected');
        }
        return;
      }
      const message = err instanceof Error ? err.message : String(err);
      this.log(`Anthropic proxy error: ${message}`);
      if (!res.headersSent) {
        writeJsonError(res, 502, `Upstream error: ${message}`);
      }
    }
  }

  /**
   * Pipe the upstream SSE response body to the client response, line by line.
   * Transparently forwards all `data:` lines including `[DONE]`.
   */
  private async pipeSSE(
    body: ReadableStream<Uint8Array>,
    res: http.ServerResponse,
    signal: AbortSignal
  ): Promise<void> {
    const reader = body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';

    try {
      while (!signal.aborted) {
        const { done, value } = await reader.read();
        if (done) { break; }

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split('\n');
        buffer = lines.pop() ?? '';

        for (const line of lines) {
          const trimmed = line.trim();
          if (trimmed === '') { continue; }
          // Forward every SSE line as-is — preserves the upstream format
          // exactly so Copilot gets the response it expects.
          res.write(trimmed + '\n\n');
        }
      }

      // Flush remaining buffer
      if (buffer.trim()) {
        res.write(buffer.trim() + '\n\n');
      }
    } catch {
      // Reader error (abort, network) — just end the response
    } finally {
      reader.releaseLock();
      res.end();
    }
  }

  // ---------- Pass-through to api.githubcopilot.com ----------

  /** Copilot's default CAPI host. All non-chat-completions requests are
   *  forwarded here with their original headers intact so subscription,
   *  token validation, model metadata, agents, etc. work normally. */
  private static readonly COPILOT_API_HOST = 'https://api.githubcopilot.com';

  /**
   * Transparently forward a request to `api.githubcopilot.com`, preserving
   * the original path, method, headers, and body. The response (status,
   * headers, body) is piped back to the client unchanged.
   */
  private async passThrough(
    req: http.IncomingMessage,
    res: http.ServerResponse
  ): Promise<void> {
    const targetUrl = `${CopilotProxyServer.COPILOT_API_HOST}${req.url ?? '/'}`;
    this.log(`  → pass-through to ${targetUrl}`);

    // Collect original headers. Replace `host` with the real target so
    // HMAC signature verification against the Host header can pass.
    // Also strip hop-by-hop headers that shouldn't be forwarded.
    const forwardHeaders: Record<string, string> = {
      host: 'api.githubcopilot.com',
    };
    for (const [key, value] of Object.entries(req.headers)) {
      const lower = key.toLowerCase();
      if (lower === 'host' || lower === 'connection' || lower === 'transfer-encoding') {
        continue;
      }
      if (typeof value === 'string') {
        forwardHeaders[key] = value;
      } else if (Array.isArray(value)) {
        forwardHeaders[key] = value.join(', ');
      }
    }

    try {
      // Read body for non-GET requests
      let body: string | undefined;
      if (req.method !== 'GET' && req.method !== 'HEAD') {
        body = await readBody(req);
      }

      const upstream = await fetch(targetUrl, {
        method: req.method ?? 'GET',
        headers: forwardHeaders,
        ...(body ? { body } : {}),
      });

      // Forward response headers
      const responseHeaders: Record<string, string> = {};
      upstream.headers.forEach((value, key) => {
        const lower = key.toLowerCase();
        if (lower !== 'transfer-encoding' && lower !== 'connection') {
          responseHeaders[key] = value;
        }
      });

      if (upstream.body) {
        res.writeHead(upstream.status, responseHeaders);
        // Pipe the body through
        const reader = upstream.body.getReader();
        try {
          while (true) {
            const { done, value } = await reader.read();
            if (done) { break; }
            res.write(value);
          }
        } finally {
          reader.releaseLock();
          res.end();
        }
      } else {
        const text = await upstream.text();
        res.writeHead(upstream.status, responseHeaders);
        res.end(text);
      }
    } catch (err) {
      let detail = err instanceof Error ? err.message : String(err);
      if (err instanceof Error && detail === 'fetch failed') {
        const cause = (err as Error & { cause?: unknown }).cause;
        if (cause instanceof Error) {
          const code = (cause as Error & { code?: string }).code;
          detail = `${cause.message}${code ? ` (${code})` : ''}`;
        }
      }
      this.log(`  → pass-through failed: ${detail} — returning safe fallback`);
      if (!res.headersSent) {
        // Return a safe 200 fallback instead of 502 so Copilot doesn't
        // treat the proxy as broken. The request path tells us what
        // shape Copilot expects.
        this.writeFallbackResponse(req.url ?? '', res);
      }
    }
  }

  /**
   * Return a safe fallback response when pass-through fails. For most
   * endpoints we return 200 with an appropriate empty body. For `/models`
   * we return 304 Not Modified so Copilot keeps its cached model list
   * instead of replacing it with an empty one.
   */
  private writeFallbackResponse(url: string, res: http.ServerResponse): void {
    const headers = { 'Content-Type': 'application/json', ...quotaHeaders() };

    if (url.includes('/models')) {
      // 304 tells Copilot "nothing changed" — it keeps its cached list.
      // Returning 200 + empty list would wipe the model picker.
      res.writeHead(304, quotaHeaders());
      res.end();
      return;
    }

    if (url.includes('/agents')) {
      res.writeHead(200, headers);
      res.end(JSON.stringify({ agents: [] }));
      return;
    }

    // Generic fallback
    res.writeHead(200, headers);
    res.end('{}');
  }
}
