import { test, describe } from 'node:test';
import assert from 'node:assert/strict';
import { parseFakeToolCalls } from '../fakeToolCallParser';

describe('parseFakeToolCalls', () => {
  test('returns empty array for text with no block', () => {
    assert.deepEqual(parseFakeToolCalls('just normal text'), []);
  });

  test('returns empty array for empty input', () => {
    assert.deepEqual(parseFakeToolCalls(''), []);
  });

  test('parses a single-bullet block (the idx-535 form)', () => {
    const text =
      'thinking about it…\n\nCompleted tool calls:\n' +
      '- read_file (call_91e3b9d70b8c4be0a4d3c1f7) {"endLine":205,"filePath":"/x/y.tsx","startLine":170}\n';
    const out = parseFakeToolCalls(text);
    assert.equal(out.length, 1);
    assert.equal(out[0].id, 'call_91e3b9d70b8c4be0a4d3c1f7');
    assert.equal(out[0].name, 'read_file');
    // Round-trip the JSON to confirm it's exactly what was in the text.
    const args = JSON.parse(out[0].arguments);
    assert.equal(args.endLine, 205);
    assert.equal(args.filePath, '/x/y.tsx');
    assert.equal(args.startLine, 170);
  });

  test('parses multiple bullets in one block', () => {
    const text =
      'Completed tool calls:\n' +
      '- read_file (call_aaaaaaaaaaaaaaaaaaaaaaaa) {"path":"/a"}\n' +
      '- read_file (call_bbbbbbbbbbbbbbbbbbbbbbbb) {"path":"/b"}\n' +
      '- grep_search (call_cccccccccccccccccccccccc) {"q":"foo"}\n';
    const out = parseFakeToolCalls(text);
    assert.equal(out.length, 3);
    assert.deepEqual(out.map((c) => c.name), ['read_file', 'read_file', 'grep_search']);
    assert.deepEqual(
      out.map((c) => JSON.parse(c.arguments)),
      [{ path: '/a' }, { path: '/b' }, { q: 'foo' }]
    );
  });

  test('parses multiple separate blocks in the same text (the user-observed double-output case)', () => {
    // Real-world: the agent emits one block, the retry/echo emits another.
    const text =
      'first analysis\n\nCompleted tool calls:\n' +
      '- read_file (call_111111111111111111111111) {"path":"/a"}\n' +
      '\nthen retry analysis\n\nCompleted tool calls:\n' +
      '- read_file (call_222222222222222222222222) {"path":"/b"}\n';
    const out = parseFakeToolCalls(text);
    assert.equal(out.length, 2);
    assert.deepEqual(out.map((c) => c.id), [
      'call_111111111111111111111111',
      'call_222222222222222222222222',
    ]);
  });

  test('handles JSON with nested braces inside string values (paths, code snippets)', () => {
    // Replace_string_in_file commonly has `{` / `}` inside the JSON string
    // values. Naive regex would terminate at the first `}`.
    const text =
      'Completed tool calls:\n' +
      '- replace_string_in_file (call_abc123def456789012345678) ' +
      '{"filePath":"/x.ts","oldString":"if (x) {\\n  return 1;\\n}","newString":"if (x) {\\n  return 2;\\n}"}\n';
    const out = parseFakeToolCalls(text);
    assert.equal(out.length, 1);
    const args = JSON.parse(out[0].arguments);
    assert.equal(args.filePath, '/x.ts');
    assert.ok(args.oldString.includes('{'));
    assert.ok(args.newString.includes('}'));
  });

  test('handles escaped quotes inside JSON strings', () => {
    const text =
      'Completed tool calls:\n' +
      '- run (call_abc123def456789012345678) {"cmd":"echo \\"hi\\" > /tmp/a"}\n';
    const out = parseFakeToolCalls(text);
    assert.equal(out.length, 1);
    const args = JSON.parse(out[0].arguments);
    assert.equal(args.cmd, 'echo "hi" > /tmp/a');
  });

  test('preserves raw arguments string for downstream json-repair', () => {
    // Caller runs tryRepairJson on .arguments, so we must return the raw
    // JSON string (not a parsed object). This matches the toolCall shape
    // used by the streaming pipeline.
    const text =
      'Completed tool calls:\n' +
      '- foo (call_abc123def456789012345678) {"x":1}\n';
    const out = parseFakeToolCalls(text);
    assert.equal(typeof out[0].arguments, 'string');
    assert.equal(out[0].arguments, '{"x":1}');
  });

  test('stops parsing a block when JSON is unbalanced (malformed model output)', () => {
    const text =
      'Completed tool calls:\n' +
      '- foo (call_abc123def456789012345678) {"x":1\n' + // missing closing }
      '- bar (call_def456abc123789012345678) {"y":2}\n';
    const out = parseFakeToolCalls(text);
    // First bullet's JSON swallows everything until EOF since there's no
    // closing brace anywhere. So we get 0 valid bullets.
    assert.equal(out.length, 0);
  });

  test('parses across multi-line JSON values', () => {
    const text =
      'Completed tool calls:\n' +
      '- write (call_abc123def456789012345678) {\n  "path": "/a",\n  "content": "line1\\nline2"\n}\n';
    const out = parseFakeToolCalls(text);
    assert.equal(out.length, 1);
    const args = JSON.parse(out[0].arguments);
    assert.equal(args.path, '/a');
    assert.equal(args.content, 'line1\nline2');
  });

  test('ignores text after the block boundary (non-bullet line stops the block)', () => {
    const text =
      'Completed tool calls:\n' +
      '- foo (call_abc123def456789012345678) {"x":1}\n' +
      '\nSome other commentary that is not a bullet.\n' +
      '- bar (call_def456abc123789012345678) {"y":2}\n';
    // The `\nSome other commentary` breaks the block; only foo is parsed.
    const out = parseFakeToolCalls(text);
    assert.equal(out.length, 1);
    assert.equal(out[0].name, 'foo');
  });

  test('prepends call_ prefix to the id', () => {
    // Format in the text is `(call_xxx)`; we re-attach the prefix to
    // produce the same id shape the OpenAI tool_call protocol uses.
    const text =
      'Completed tool calls:\n' +
      '- foo (call_91e3b9d70b8c4be0a4d3c1f7) {}\n';
    const out = parseFakeToolCalls(text);
    assert.equal(out[0].id, 'call_91e3b9d70b8c4be0a4d3c1f7');
  });

  test('tolerates extra whitespace around bullet components', () => {
    const text =
      'Completed tool calls:\n' +
      '-   read_file   (call_abc123def456789012345678)   {"x":1}\n';
    const out = parseFakeToolCalls(text);
    assert.equal(out.length, 1);
    assert.equal(out[0].name, 'read_file');
  });
});
