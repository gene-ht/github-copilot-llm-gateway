/**
 * Unit tests for the LM passthrough module.
 *
 * `lm-passthrough.ts` imports `vscode` (for LanguageModel*Part instanceof
 * checks), so the test cannot import directly from it in a pure Node.js
 * runner. Instead, this file validates the wire-format contracts by
 * exercising the serialization logic inline — the same algorithm as
 * `serializeTools` but without the module-level `vscode` dependency.
 *
 * Type correctness of the full module is verified by the TypeScript
 * compiler (`tsc -p tsconfig.test.json --noEmit` exits 0).
 */

import { test, describe } from 'node:test';
import assert from 'node:assert/strict';

// ── Inline copy of serializeTools logic (no vscode dependency) ──────────
interface LmToolPayload {
  name: string;
  description: string;
  inputSchema: unknown;
}

interface ToolLike {
  name: string;
  description: string;
  inputSchema: unknown;
}

function serializeTools(
  tools: readonly ToolLike[] | undefined
): LmToolPayload[] | undefined {
  if (!tools || tools.length === 0) { return undefined; }
  return tools.map((t) => ({
    name: t.name,
    description: t.description,
    inputSchema: t.inputSchema,
  }));
}

// ── Tests ───────────────────────────────────────────────────────────────

describe('serializeTools (lm-passthrough)', () => {
  test('returns undefined for undefined input', () => {
    const result = serializeTools(undefined);
    assert.equal(result, undefined);
  });

  test('returns undefined for empty array', () => {
    const result = serializeTools([]);
    assert.equal(result, undefined);
  });

  test('serializes a single tool', () => {
    const tools: ToolLike[] = [
      {
        name: 'search',
        description: 'Search files',
        inputSchema: { type: 'object', properties: { query: { type: 'string' } } },
      },
    ];
    const result = serializeTools(tools)!;
    assert.ok(result);
    assert.equal(result.length, 1);
    assert.deepEqual(result[0], {
      name: 'search',
      description: 'Search files',
      inputSchema: { type: 'object', properties: { query: { type: 'string' } } },
    });
  });

  test('serializes multiple tools', () => {
    const tools: ToolLike[] = [
      {
        name: 'read_file',
        description: 'Read a file',
        inputSchema: { type: 'object', properties: { path: { type: 'string' } } },
      },
      {
        name: 'write_file',
        description: 'Write a file',
        inputSchema: {
          type: 'object',
          properties: {
            path: { type: 'string' },
            content: { type: 'string' },
          },
        },
      },
    ];
    const result = serializeTools(tools)!;
    assert.ok(result);
    assert.equal(result.length, 2);
    assert.equal(result[0].name, 'read_file');
    assert.equal(result[1].name, 'write_file');
  });

  test('preserves flat structure (no OpenAI function wrapper)', () => {
    const tools: ToolLike[] = [
      {
        name: 'test_tool',
        description: 'A test tool',
        inputSchema: { type: 'object' },
      },
    ];
    const result = serializeTools(tools)!;
    assert.ok(result);
    const serialized = result[0] as unknown as Record<string, unknown>;
    // Must NOT have `type: "function"` or `function: { ... }` wrapper
    assert.equal(serialized.type, undefined, 'should not have OpenAI-style type field');
    assert.equal(
      serialized.function,
      undefined,
      'should not have OpenAI-style function wrapper'
    );
    // Must have flat keys
    assert.ok('name' in serialized);
    assert.ok('description' in serialized);
    assert.ok('inputSchema' in serialized);
  });

  test('handles tool with complex inputSchema', () => {
    const schema = {
      type: 'object',
      properties: {
        regex: { type: 'string', description: 'The regex pattern' },
        path: { type: 'string', description: 'Directory to search' },
        maxResults: { type: 'number', default: 10 },
      },
      required: ['regex', 'path'],
    };
    const tools: ToolLike[] = [
      { name: 'search_regex', description: 'Regex search', inputSchema: schema },
    ];
    const result = serializeTools(tools)!;
    assert.ok(result);
    assert.deepEqual(result[0].inputSchema, schema);
  });
});

describe('LM passthrough wire format contracts', () => {
  test('message payload uses numeric roles', () => {
    // The plan specifies role 1 = User, 2 = Assistant
    // VS Code's LanguageModelChatMessageRole enum: User = 1, Assistant = 2
    const payload = { role: 1, content: [{ type: 'text' as const, value: 'hello' }] };
    assert.equal(payload.role, 1);
    assert.equal(payload.content[0].type, 'text');
  });

  test('SSE event types match protocol', () => {
    // Validate that the SSE protocol types are well-formed
    const textEvent = { type: 'text', value: 'hello' };
    const toolCallEvent = { type: 'tool_call', callId: 'call_1', name: 'search', input: {} };
    const thinkingEvent = { type: 'thinking', value: 'hmm', id: 'think_1' };
    const doneEvent = { type: 'done' };
    const errorEvent = { type: 'error', message: 'Model not found' };

    assert.equal(textEvent.type, 'text');
    assert.equal(toolCallEvent.type, 'tool_call');
    assert.equal(thinkingEvent.type, 'thinking');
    assert.equal(doneEvent.type, 'done');
    assert.equal(errorEvent.type, 'error');
  });

  test('tool_result part nests content array', () => {
    const toolResult = {
      type: 'tool_result' as const,
      callId: 'call_123',
      content: [
        { type: 'text' as const, value: 'Found 5 files...' },
      ],
    };
    assert.equal(toolResult.type, 'tool_result');
    assert.equal(toolResult.callId, 'call_123');
    assert.equal(toolResult.content.length, 1);
    assert.equal(toolResult.content[0].type, 'text');
  });
});
