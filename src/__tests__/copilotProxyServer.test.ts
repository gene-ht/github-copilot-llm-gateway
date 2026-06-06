import { test, describe, afterEach } from 'node:test';
import assert from 'node:assert/strict';
import * as http from 'node:http';
import { mapModel, CopilotProxyServer } from '../copilotProxyServer';
import type { ProxyUpstreamConfig, CopilotProxyConfig } from '../copilotProxyServer';

// ---------- mapModel (pure) ----------

describe('mapModel', () => {
  test('returns mapped model when mapping exists', () => {
    const mapping = { 'gpt-4o-mini': 'qwen2.5-7b', 'gpt-4.1-2025-04-14': 'qwen2.5-72b' };
    assert.equal(mapModel('gpt-4o-mini', mapping), 'qwen2.5-7b');
    assert.equal(mapModel('gpt-4.1-2025-04-14', mapping), 'qwen2.5-72b');
  });

  test('returns requestedModel unchanged when no mapping exists and no fallback', () => {
    assert.equal(mapModel('unknown-model', {}), 'unknown-model');
  });

  test('returns fallbackModel when no mapping exists and fallback is provided', () => {
    assert.equal(mapModel('unknown-model', {}, 'default-model'), 'default-model');
  });

  test('prefers explicit mapping over fallback', () => {
    const mapping = { 'gpt-4o-mini': 'qwen2.5-7b' };
    assert.equal(mapModel('gpt-4o-mini', mapping, 'fallback'), 'qwen2.5-7b');
  });

  test('empty string mapping value is falsy — uses fallback', () => {
    const mapping = { 'gpt-4o-mini': '' };
    assert.equal(mapModel('gpt-4o-mini', mapping, 'fallback'), 'fallback');
  });

  test('prefix match: gpt-4o-mini-2024-07-18 matches gpt-4o-mini mapping', () => {
    const mapping = { 'gpt-4o-mini': 'my-fast-model' };
    assert.equal(mapModel('gpt-4o-mini-2024-07-18', mapping), 'my-fast-model');
  });

  test('exact match takes priority over prefix match', () => {
    const mapping = { 'gpt-4o-mini': 'prefix-match', 'gpt-4o-mini-2024-07-18': 'exact-match' };
    assert.equal(mapModel('gpt-4o-mini-2024-07-18', mapping), 'exact-match');
  });
});

// ---------- CopilotProxyServer lifecycle ----------

describe('CopilotProxyServer', () => {
  const upstream: ProxyUpstreamConfig = {
    serverUrl: 'http://localhost:9999',
    apiKey: 'test-key',
    requestTimeout: 5000,
  };
  const proxy: CopilotProxyConfig = {
    enabled: true,
    modelMapping: { 'gpt-4o-mini': 'test-model' },
  };

  let server: CopilotProxyServer | undefined;

  afterEach(async () => {
    if (server) {
      await server.stop();
      server = undefined;
    }
  });

  test('starts and reports a port', async () => {
    server = new CopilotProxyServer(upstream, proxy);
    const port = await server.start();
    assert.ok(port > 0, `port should be > 0, got ${port}`);
    assert.equal(server.port, port);
    assert.equal(server.isRunning, true);
  });

  test('stop shuts down cleanly', async () => {
    server = new CopilotProxyServer(upstream, proxy);
    await server.start();
    assert.equal(server.isRunning, true);
    await server.stop();
    assert.equal(server.isRunning, false);
    assert.equal(server.port, undefined);
    server = undefined; // already stopped
  });

  test('GET / passes through (does not crash)', async () => {
    server = new CopilotProxyServer(upstream, proxy);
    const port = await server.start();
    // GET / is forwarded to api.githubcopilot.com — we just verify the
    // proxy responds without crashing (status depends on the remote).
    const res = await fetch(`http://127.0.0.1:${port}/`);
    assert.ok(res.status > 0, 'should get some response status');
  });

  test('POST /chat/completions with invalid JSON returns 400', async () => {
    server = new CopilotProxyServer(upstream, proxy);
    const port = await server.start();
    const res = await fetch(`http://127.0.0.1:${port}/chat/completions`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: 'not json',
    });
    assert.equal(res.status, 400);
    const body = await res.json();
    assert.ok(body.error.message.includes('Invalid JSON'));
  });

  test('POST /chat/completions forwards to upstream and returns 502 when upstream is down', async () => {
    server = new CopilotProxyServer(upstream, proxy);
    const port = await server.start();
    const res = await fetch(`http://127.0.0.1:${port}/v1/chat/completions`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        model: 'gpt-4o-mini',
        messages: [{ role: 'user', content: 'hello' }],
        stream: true,
      }),
    });
    // Upstream at localhost:9999 is not running, should get 502
    assert.equal(res.status, 502);
  });

  test('updateConfig updates settings without restart', async () => {
    server = new CopilotProxyServer(upstream, proxy);
    await server.start();
    const newProxy: CopilotProxyConfig = {
      enabled: true,
      modelMapping: { 'gpt-4o-mini': 'new-model' },
    };
    server.updateConfig({ ...upstream, serverUrl: 'http://localhost:9998' }, newProxy);
    assert.equal(server.isRunning, true); // still running
  });

  test('handles SSE streaming from a mock upstream', async () => {
    // Create a mock upstream that returns SSE chunks
    const mockUpstream = http.createServer((req, res) => {
      if (req.method === 'POST' && req.url?.includes('/chat/completions')) {
        res.writeHead(200, { 'Content-Type': 'text/event-stream' });
        res.write('data: {"id":"1","object":"chat.completion.chunk","model":"test","choices":[{"index":0,"delta":{"content":"Hello"},"finish_reason":null}]}\n\n');
        res.write('data: {"id":"1","object":"chat.completion.chunk","model":"test","choices":[{"index":0,"delta":{"content":" World"},"finish_reason":"stop"}]}\n\n');
        res.write('data: [DONE]\n\n');
        res.end();
        return;
      }
      res.writeHead(404);
      res.end();
    });

    await new Promise<void>((resolve) => mockUpstream.listen(0, '127.0.0.1', resolve));
    const mockPort = (mockUpstream.address() as { port: number }).port;

    try {
      const upstreamWithMock: ProxyUpstreamConfig = {
        ...upstream,
        serverUrl: `http://127.0.0.1:${mockPort}`,
      };
      server = new CopilotProxyServer(upstreamWithMock, proxy);
      const proxyPort = await server.start();

      const res = await fetch(`http://127.0.0.1:${proxyPort}/chat/completions`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          model: 'gpt-4o-mini',
          messages: [{ role: 'user', content: 'test' }],
          stream: true,
        }),
      });

      assert.equal(res.status, 200);
      assert.equal(res.headers.get('content-type'), 'text/event-stream');

      const text = await res.text();
      assert.ok(text.includes('"content":"Hello"'), 'should contain Hello chunk');
      assert.ok(text.includes('"content":" World"'), 'should contain World chunk');
      assert.ok(text.includes('[DONE]'), 'should contain DONE marker');
    } finally {
      mockUpstream.close();
    }
  });

  test('model mapping is applied in proxied request', async () => {
    // Mock upstream that captures the model from the request body
    let capturedModel = '';
    const mockUpstream = http.createServer(async (req, res) => {
      if (req.method === 'POST') {
        const chunks: Buffer[] = [];
        for await (const chunk of req) { chunks.push(chunk); }
        const body = JSON.parse(Buffer.concat(chunks).toString());
        capturedModel = body.model;
        res.writeHead(200, { 'Content-Type': 'text/event-stream' });
        res.write('data: [DONE]\n\n');
        res.end();
        return;
      }
      res.writeHead(404);
      res.end();
    });

    await new Promise<void>((resolve) => mockUpstream.listen(0, '127.0.0.1', resolve));
    const mockPort = (mockUpstream.address() as { port: number }).port;

    try {
      const proxyWithMapping: CopilotProxyConfig = {
        enabled: true,
        modelMapping: { 'gpt-4o-mini': 'my-custom-model' },
      };
      server = new CopilotProxyServer(
        { ...upstream, serverUrl: `http://127.0.0.1:${mockPort}` },
        proxyWithMapping
      );
      const proxyPort = await server.start();

      await fetch(`http://127.0.0.1:${proxyPort}/chat/completions`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          model: 'gpt-4o-mini',
          messages: [{ role: 'user', content: 'test' }],
        }),
      });

      assert.equal(capturedModel, 'my-custom-model', 'upstream should receive mapped model name');
    } finally {
      mockUpstream.close();
    }
  });
});
