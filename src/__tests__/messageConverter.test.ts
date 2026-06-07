import { test, describe } from 'node:test';
import assert from 'node:assert/strict';
import {
  convertMessage,
  convertMessages,
  encodeImageAsDataUrl,
  NormalizedMessage,
  NormalizedPart,
  stripFakeToolCallText,
  containsFakeToolCallText,
} from '../messageConverter';

const WITH_IMAGES = { enableImageInput: true };
const WITHOUT_IMAGES = { enableImageInput: false };

const textMsg = (role: 'user' | 'assistant', value: string): NormalizedMessage => ({
  role,
  parts: [{ kind: 'text', value }],
});

describe('encodeImageAsDataUrl', () => {
  test('wraps byte data as a base64 data URL with the given mime type', () => {
    const data = new Uint8Array([1, 2, 3, 4]);
    const url = encodeImageAsDataUrl({ mimeType: 'image/png', data });
    assert.ok(url.startsWith('data:image/png;base64,'));
    const encoded = url.slice('data:image/png;base64,'.length);
    // Round-trip via atob to confirm the bytes.
    const decoded = atob(encoded);
    assert.equal(decoded.length, 4);
    assert.equal(decoded.codePointAt(0), 1);
    assert.equal(decoded.codePointAt(3), 4);
  });
});

describe('convertMessage', () => {
  test('converts a plain user text message to string content', () => {
    const result = convertMessage(textMsg('user', 'hello'), WITHOUT_IMAGES);
    assert.equal(result.length, 1);
    assert.equal(result[0].role, 'user');
    assert.equal(result[0].content, 'hello');
  });

  test('converts an assistant text message to string content', () => {
    const result = convertMessage(textMsg('assistant', 'sure'), WITHOUT_IMAGES);
    assert.equal(result[0].role, 'assistant');
    assert.equal(result[0].content, 'sure');
  });

  test('emits array content when images are enabled and included', () => {
    const data = new Uint8Array([10, 20, 30]);
    const msg: NormalizedMessage = {
      role: 'user',
      parts: [
        { kind: 'text', value: 'look' },
        { kind: 'image', mimeType: 'image/jpeg', data },
      ],
    };
    const result = convertMessage(msg, WITH_IMAGES);
    assert.equal(result.length, 1);
    const parts = result[0].content as Array<Record<string, unknown>>;
    assert.equal(parts.length, 2);
    assert.equal(parts[0].type, 'text');
    assert.equal(parts[1].type, 'image_url');
    const imageUrl = (parts[1].image_url as { url: string }).url;
    assert.ok(imageUrl.startsWith('data:image/jpeg;base64,'));
  });

  test('skips image parts and logs when enableImageInput is false', () => {
    const logs: string[] = [];
    const msg: NormalizedMessage = {
      role: 'user',
      parts: [
        { kind: 'text', value: 'hi' },
        { kind: 'image', mimeType: 'image/png', data: new Uint8Array([1]) },
      ],
    };
    const result = convertMessage(msg, WITHOUT_IMAGES, (m) => logs.push(m));
    // With image skipped, pure-text user message becomes string content
    assert.equal(result[0].content, 'hi');
    assert.ok(logs.some((m) => m.includes('Skipping data part')));
  });

  test('skips non-image data parts even when enableImageInput is true', () => {
    const msg: NormalizedMessage = {
      role: 'user',
      parts: [{ kind: 'image', mimeType: 'application/pdf', data: new Uint8Array([1]) }],
    };
    const result = convertMessage(msg, WITH_IMAGES);
    assert.equal(result.length, 0);
  });

  test('converts assistant tool call part into assistant message with tool_calls', () => {
    const msg: NormalizedMessage = {
      role: 'assistant',
      parts: [
        { kind: 'text', value: 'calling tool' },
        { kind: 'toolCall', callId: 'c1', name: 'search', input: { q: 'x' } },
      ],
    };
    const result = convertMessage(msg, WITHOUT_IMAGES);
    assert.equal(result.length, 1);
    assert.equal(result[0].role, 'assistant');
    assert.equal(result[0].content, 'calling tool');
    const toolCalls = result[0].tool_calls as Array<Record<string, unknown>>;
    assert.equal(toolCalls.length, 1);
    assert.equal(toolCalls[0].id, 'c1');
    const fn = toolCalls[0].function as Record<string, unknown>;
    assert.equal(fn.name, 'search');
    assert.equal(fn.arguments, JSON.stringify({ q: 'x' }));
  });

  test('assistant with only tool calls has empty string content', () => {
    const msg: NormalizedMessage = {
      role: 'assistant',
      parts: [{ kind: 'toolCall', callId: 'c1', name: 'f', input: {} }],
    };
    const result = convertMessage(msg, WITHOUT_IMAGES);
    assert.equal(result[0].content, '');
  });

  test('tool result part produces a role:tool message', () => {
    const msg: NormalizedMessage = {
      role: 'user',
      parts: [{ kind: 'toolResult', callId: 'c1', content: 'result data' }],
    };
    const result = convertMessage(msg, WITHOUT_IMAGES);
    assert.equal(result.length, 1);
    assert.equal(result[0].role, 'tool');
    assert.equal(result[0].tool_call_id, 'c1');
    assert.equal(result[0].content, 'result data');
  });

  test('multiple tool results flatten into multiple tool messages', () => {
    const msg: NormalizedMessage = {
      role: 'user',
      parts: [
        { kind: 'toolResult', callId: 'a', content: 'x' },
        { kind: 'toolResult', callId: 'b', content: 'y' },
      ],
    };
    const result = convertMessage(msg, WITHOUT_IMAGES);
    assert.equal(result.length, 2);
    assert.equal(result[0].tool_call_id, 'a');
    assert.equal(result[1].tool_call_id, 'b');
  });

  test('tool calls take precedence over tool results in the same message', () => {
    const msg: NormalizedMessage = {
      role: 'assistant',
      parts: [
        { kind: 'toolCall', callId: 'c1', name: 'f', input: {} },
        { kind: 'toolResult', callId: 'c0', content: 'prev' },
      ],
    };
    const result = convertMessage(msg, WITHOUT_IMAGES);
    assert.equal(result.length, 1);
    assert.equal(result[0].role, 'assistant');
    assert.ok(Array.isArray(result[0].tool_calls));
  });

  test('empty message produces no output', () => {
    const result = convertMessage({ role: 'user', parts: [] }, WITHOUT_IMAGES);
    assert.equal(result.length, 0);
  });

  test('unknown parts are silently dropped', () => {
    const msg: NormalizedMessage = {
      role: 'user',
      parts: [{ kind: 'text', value: 'hi' }, { kind: 'unknown' }],
    };
    const result = convertMessage(msg, WITHOUT_IMAGES);
    assert.equal(result.length, 1);
    // Pure-text user message becomes string content
    assert.equal(result[0].content, 'hi');
  });

  test('logs image addition with URL length', () => {
    const logs: string[] = [];
    const msg: NormalizedMessage = {
      role: 'user',
      parts: [{ kind: 'image', mimeType: 'image/png', data: new Uint8Array([1, 2]) }],
    };
    convertMessage(msg, WITH_IMAGES, (m) => logs.push(m));
    assert.ok(logs.some((m) => m.includes('Added image data part')));
  });

  test('skips image parts on assistant messages', () => {
    const logs: string[] = [];
    const msg: NormalizedMessage = {
      role: 'assistant',
      parts: [
        { kind: 'text', value: 'text only' },
        { kind: 'image', mimeType: 'image/png', data: new Uint8Array([1]) },
      ],
    };
    const result = convertMessage(msg, WITH_IMAGES, (m) => logs.push(m));
    assert.equal(result[0].content, 'text only');
    assert.ok(logs.some((m) => m.includes('Skipping image data part on non-user message')));
  });
});

describe('convertMessages', () => {
  test('flattens a list of normalized messages through convertMessage', () => {
    const messages: NormalizedMessage[] = [
      textMsg('user', 'q'),
      textMsg('assistant', 'a'),
    ];
    const result = convertMessages(messages, WITHOUT_IMAGES);
    assert.equal(result.length, 2);
    assert.equal(result[0].role, 'user');
    assert.equal(result[1].role, 'assistant');
  });

  test('returns empty array for empty input', () => {
    assert.deepEqual(convertMessages([], WITHOUT_IMAGES), []);
  });

  test('tool result messages get flattened into multiple entries', () => {
    const messages: NormalizedMessage[] = [
      textMsg('user', 'question'),
      {
        role: 'user',
        parts: [
          { kind: 'toolResult', callId: 'a', content: 'x' },
          { kind: 'toolResult', callId: 'b', content: 'y' },
        ],
      },
    ];
    const result = convertMessages(messages, WITHOUT_IMAGES);
    assert.equal(result.length, 3);
  });

  const usedPartKinds: ReadonlyArray<NormalizedPart['kind']> = [
    'text',
    'toolResult',
    'toolCall',
    'image',
    'unknown',
  ];

  test('exhausts the NormalizedPart discriminant (typesafety guard)', () => {
    // If a new kind is added, this assertion will break, nudging maintainers
    // to handle it in convertMessage().
    assert.equal(usedPartKinds.length, 5);
  });
});

describe('stripFakeToolCallText', () => {
  test('passes through messages without the marker untouched', () => {
    const messages = [
      { role: 'user', content: 'hello' },
      { role: 'assistant', content: 'sure, doing it' },
      { role: 'tool', tool_call_id: 'call_abc', content: 'result' },
    ];
    const out = stripFakeToolCallText(messages);
    assert.deepEqual(out, messages);
  });

  test('strips the "Completed tool calls" block from assistant text (idx-535 form)', () => {
    // Pure-text "fake" tool call: the seed of the self-poisoning loop.
    // New behavior: synthesize structured tool_call + synthetic tool result.
    const polluted =
      '`KnowledgePanel.tsx` deep uses currentStage…\n\nCompleted tool calls:\n- read_file (call_91e3b9d70b8c4be0a4d3c1f7) {"endLine":205,"filePath":"/x/y.tsx","startLine":170}\n';
    const out = stripFakeToolCallText([{ role: 'assistant', content: polluted }]);
    // 1 assistant message + 1 synthetic tool result
    assert.equal(out.length, 2);
    const assistantMsg = out[0] as { content: string; tool_calls: Array<{ id: string }> };
    assert.ok(!/Completed tool calls/.test(assistantMsg.content), 'header still present');
    assert.ok(!/call_91e3b9d70b8c4be0a4d3c1f7/.test(assistantMsg.content), 'call id still present');
    assert.ok(/KnowledgePanel\.tsx/.test(assistantMsg.content), 'human-readable prose was stripped');
    assert.equal(assistantMsg.tool_calls.length, 1, 'synthesized tool_call present');
    assert.equal(assistantMsg.tool_calls[0].id, 'call_91e3b9d70b8c4be0a4d3c1f7');
    assert.equal((out[1] as { role: string }).role, 'tool');
  });

  test('preserves real tool_calls when stripping coexisting text block (idx-537 form)', () => {
    // Hybrid: model emitted real tool_calls AND echoed them as text. We
    // strip the text and dedupe by id — synthesizing the same id again would
    // produce a duplicate tool_call, and the real tool result will follow in
    // history (we don't double-emit a synthetic result).
    const realToolCall = {
      id: 'call_a6f7e54b2eaa4b9c812a2cae',
      type: 'function',
      function: { name: 'replace_string_in_file', arguments: '{"filePath":"/x/y.tsx"}' },
    };
    const msg = {
      role: 'assistant',
      content:
        '读到了 FewshotSection 顶部。\n\nCompleted tool calls:\n- replace_string_in_file (call_a6f7e54b2eaa4b9c812a2cae) {"filePath":"/x/y.tsx"}\n',
      tool_calls: [realToolCall],
    };
    const out = stripFakeToolCallText([msg]);
    // Since parsed id matches existing real tool_call id, we dedupe and don't
    // insert a synthetic tool result. Only 1 assistant message in output.
    assert.equal(out.length, 1);
    const cleaned = out[0] as { content: string; tool_calls: unknown[] };
    assert.ok(!/Completed tool calls/.test(cleaned.content));
    assert.deepEqual(cleaned.tool_calls, [realToolCall], 'real tool_calls must survive without duplication');
    assert.ok(/FewshotSection/.test(cleaned.content));
  });

  test('keeps message when content becomes empty but tool_calls remain (idx-537 hybrid)', () => {
    // Edge case: the assistant only wrote the fake summary and nothing else,
    // but the structured tool_calls field is still there. Dropping the
    // message would break the tool_call_id <-> next tool message linkage.
    // The parsed id matches the real one → dedupe → no synthetic tool result.
    const realToolCall = {
      id: 'call_deadbeef00000000000000',
      type: 'function',
      function: { name: 'read_file', arguments: '{}' },
    };
    const out = stripFakeToolCallText([
      {
        role: 'assistant',
        content: 'Completed tool calls:\n- read_file (call_deadbeef00000000000000) {}\n',
        tool_calls: [realToolCall],
      },
    ]);
    assert.equal(out.length, 1, 'message must be retained for tool_call_id linkage');
    assert.equal((out[0] as { content: string }).content, '');
  });

  test('transforms pure-pollution assistant message into structured tool_calls + tool result', () => {
    // Even when the assistant content is *only* fake-toolcall text, we no
    // longer drop the message — we synthesize structured tool_calls + a
    // synthetic tool result so the LLM sees a completed call instead of
    // an unanswered one (which would trigger a retry loop).
    const out = stripFakeToolCallText([
      {
        role: 'assistant',
        content: 'Completed tool calls:\n- foo (call_aaaaaaaaaaaaaaaaaaaaaaaa) {}\n',
      },
    ]);
    assert.equal(out.length, 2);
    const assistantMsg = out[0] as Record<string, unknown>;
    assert.equal(assistantMsg.role, 'assistant');
    assert.equal(assistantMsg.content, '');
    const toolCalls = assistantMsg.tool_calls as Array<Record<string, unknown>>;
    assert.equal(toolCalls.length, 1);
    assert.equal(toolCalls[0].id, 'call_aaaaaaaaaaaaaaaaaaaaaaaa');
    const toolMsg = out[1] as Record<string, unknown>;
    assert.equal(toolMsg.role, 'tool');
    assert.equal(toolMsg.tool_call_id, 'call_aaaaaaaaaaaaaaaaaaaaaaaa');
  });

  test('does not touch user or tool messages even if they contain the marker', () => {
    // Conservative scope: only assistant.content is the poisoning vector;
    // user/tool messages with that string (e.g., a user pasting a transcript)
    // should pass through verbatim.
    const messages = [
      { role: 'user', content: 'Completed tool calls:\n- foo (call_xxxxxxxxxxxxxxxxxxxxxxxx) {}' },
      { role: 'tool', tool_call_id: 'call_xxxxxxxxxxxxxxxxxxxxxxxx', content: 'result' },
    ];
    const out = stripFakeToolCallText(messages);
    assert.deepEqual(out, messages);
  });

  test('handles array-content assistant messages by leaving them alone', () => {
    // Array content (multimodal) isn't the observed pollution shape — keep
    // the function focused; future work can extend to array text parts if
    // the failure mode shows up there too.
    const messages = [
      {
        role: 'assistant',
        content: [{ type: 'text', text: 'Completed tool calls:\n- foo (call_abc) {}' }],
      },
    ];
    const out = stripFakeToolCallText(messages);
    assert.deepEqual(out, messages);
  });

  test('strips multiple bullets in a single block', () => {
    // The Copilot client sometimes flushes several executed calls together.
    const content =
      'doing stuff\n\nCompleted tool calls:\n- read_file (call_aaaaaaaaaaaaaaaaaaaaaaaa) {"path":"/a"}\n- read_file (call_bbbbbbbbbbbbbbbbbbbbbbbb) {"path":"/b"}\n- grep_search (call_cccccccccccccccccccccccc) {"q":"foo"}\n';
    const out = stripFakeToolCallText([{ role: 'assistant', content }]);
    const cleaned = (out[0] as { content: string }).content;
    assert.ok(!/call_/.test(cleaned));
    assert.equal(cleaned.trim(), 'doing stuff');
  });

  test('logs summary lines when blocks are stripped and synthesized', () => {
    const logs: string[] = [];
    stripFakeToolCallText(
      [
        {
          role: 'assistant',
          content: 'x\n\nCompleted tool calls:\n- a (call_xxxxxxxxxxxxxxxxxxxxxxxx) {}\n',
        },
        {
          role: 'assistant',
          content: 'y\n\nCompleted tool calls:\n- b (call_yyyyyyyyyyyyyyyyyyyyyyyy) {}\n',
        },
      ],
      (m) => logs.push(m)
    );
    // Two log lines now: strip summary + synthesized summary.
    assert.equal(logs.length, 2);
    assert.match(logs[0], /Stripped "Completed tool calls" text from 2 assistant message/);
    assert.match(logs[1], /Synthesized 2 structured tool_call/);
  });

  test('synthesizes structured tool_calls + tool results from bullet text', () => {
    const out = stripFakeToolCallText([
      {
        role: 'assistant',
        content:
          'analysis\n\nCompleted tool calls:\n' +
          '- read_file (call_abcdef0123456789abcdef01) {"filePath":"/x.ts"}\n',
      },
    ]);
    // assistant message kept + structured tool_calls added + synthetic tool result inserted
    assert.equal(out.length, 2);
    const assistantMsg = out[0] as Record<string, unknown>;
    assert.equal(assistantMsg.role, 'assistant');
    assert.equal(assistantMsg.content, 'analysis');
    const toolCalls = assistantMsg.tool_calls as Array<Record<string, unknown>>;
    assert.equal(toolCalls.length, 1);
    assert.equal(toolCalls[0].id, 'call_abcdef0123456789abcdef01');
    assert.equal((toolCalls[0].function as Record<string, unknown>).name, 'read_file');
    const toolMsg = out[1] as Record<string, unknown>;
    assert.equal(toolMsg.role, 'tool');
    assert.equal(toolMsg.tool_call_id, 'call_abcdef0123456789abcdef01');
    assert.match(toolMsg.content as string, /Synthesized by LLM Gateway/);
  });

  test('synthesizes structured tool_calls + tool results from <invoke> XML', () => {
    const out = stripFakeToolCallText([
      {
        role: 'assistant',
        content:
          'analysis text\n' +
          '<invoke name="read_file">\n' +
          '<parameter name="filePath">/x.ts</parameter>\n' +
          '<parameter name="startLine">10</parameter>\n' +
          '</invoke>',
      },
    ]);
    assert.equal(out.length, 2);
    const assistantMsg = out[0] as Record<string, unknown>;
    assert.equal(assistantMsg.role, 'assistant');
    assert.equal(assistantMsg.content, 'analysis text');
    const toolCalls = assistantMsg.tool_calls as Array<Record<string, unknown>>;
    assert.equal(toolCalls.length, 1);
    assert.equal((toolCalls[0].function as Record<string, unknown>).name, 'read_file');
    const args = JSON.parse((toolCalls[0].function as Record<string, unknown>).arguments as string);
    assert.equal(args.filePath, '/x.ts');
    assert.equal(args.startLine, 10);
    const toolMsg = out[1] as Record<string, unknown>;
    assert.equal(toolMsg.role, 'tool');
    assert.match(toolMsg.content as string, /Synthesized by LLM Gateway/);
  });
});

describe('containsFakeToolCallText', () => {
  test('returns true for text containing a Completed tool calls block', () => {
    const text =
      'doing stuff\n\nCompleted tool calls:\n- read_file (call_91e3b9d70b8c4be0a4d3c1f7) {"endLine":205}\n';
    assert.equal(containsFakeToolCallText(text), true);
  });

  test('returns false for normal text without the marker', () => {
    assert.equal(containsFakeToolCallText('just some analysis text'), false);
  });

  test('returns false for text mentioning tool calls without the exact format', () => {
    assert.equal(containsFakeToolCallText('I will call read_file next'), false);
  });

  test('returns true for multi-bullet blocks', () => {
    const text =
      'ok\n\nCompleted tool calls:\n- a (call_aaaaaaaaaaaaaaaaaaaaaaaa) {}\n- b (call_bbbbbbbbbbbbbbbbbbbbbbbb) {}\n';
    assert.equal(containsFakeToolCallText(text), true);
  });

  test('can be called multiple times without sticky regex state', () => {
    const text =
      'x\n\nCompleted tool calls:\n- a (call_aaaaaaaaaaaaaaaaaaaaaaaa) {}\n';
    // Call 3 times to test regex lastIndex reset
    assert.equal(containsFakeToolCallText(text), true);
    assert.equal(containsFakeToolCallText(text), true);
    assert.equal(containsFakeToolCallText(text), true);
  });
});
