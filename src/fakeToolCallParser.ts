/**
 * Parse Copilot Chat's "Completed tool calls" textual summary blocks back
 * into structured tool calls.
 *
 * Background: Under long contexts, Claude (and other models) occasionally
 * degrade and write tool calls as plain text in this format:
 *
 *     Completed tool calls:
 *     - read_file (call_91e3b9d70b8c4be0a4d3c1f7) {"endLine":205,"filePath":"/x/y.tsx","startLine":170}
 *     - replace_string_in_file (call_a6f7e54b2eaa4b9c812a2cae) {"filePath":"/x"}
 *
 * Instead of emitting a real `tool_calls` field, the model "narrates" the
 * call. The user's chat client cannot execute narrations — only structured
 * `tool_calls`. This module rescues that case by parsing the text back into
 * the structured form so the caller can re-emit it as a real tool call.
 *
 * Pure / no VS Code imports / no side effects so it's trivially unit-testable.
 */

export interface ParsedFakeToolCall {
  /** Tool call id, e.g. "call_91e3b9d70b8c4be0a4d3c1f7" */
  id: string;
  /** Tool name, e.g. "read_file" */
  name: string;
  /** Raw JSON arguments string (not yet parsed — caller runs json-repair). */
  arguments: string;
}

/**
 * Regex matching a single bullet line:
 *   `- toolname (call_xxx) {…json…}`
 *
 * The argument JSON is non-greedy to handle multiple bullets in the same
 * block. We require at least the opening `{`; the closing `}` is matched
 * by a balanced-brace scan in the parser because models occasionally embed
 * `{` / `}` inside the JSON strings (paths with braces, code snippets, etc).
 */
const BULLET_HEAD_RE =
  /^[ \t]*-[ \t]+(\S+?)[ \t]*\(call_([0-9a-zA-Z]+)\)[ \t]*(\{)/;

/**
 * Find the index immediately after the matching closing brace for the
 * opening brace at `openIdx`. Uses a simple state machine that tracks
 * string literals and escape characters so `"\"}"` inside the JSON doesn't
 * mis-terminate the scan. Returns -1 if unbalanced.
 */
function findMatchingBraceEnd(text: string, openIdx: number): number {
  let depth = 0;
  let inString = false;
  let escape = false;
  for (let i = openIdx; i < text.length; i++) {
    const ch = text[i];
    if (escape) {
      escape = false;
      continue;
    }
    if (inString) {
      if (ch === '\\') {
        escape = true;
      } else if (ch === '"') {
        inString = false;
      }
      continue;
    }
    if (ch === '"') {
      inString = true;
    } else if (ch === '{') {
      depth++;
    } else if (ch === '}') {
      depth--;
      if (depth === 0) {
        return i + 1;
      }
    }
  }
  return -1;
}

/**
 * Parse all `Completed tool calls:` blocks from a text and return the
 * structured tool calls found inside them.
 *
 * - Multiple `Completed tool calls:` blocks are aggregated.
 * - Bullets with malformed JSON (no matching closing brace) are skipped.
 * - The order of the returned array preserves the order in the source text.
 * - Returns `[]` when no recognizable block is found.
 */
export function parseFakeToolCalls(text: string): ParsedFakeToolCall[] {
  const results: ParsedFakeToolCall[] = [];

  // Find each "Completed tool calls:" header, then scan forward bullet by
  // bullet until we hit a non-matching line.
  const HEADER = /Completed tool calls:[ \t]*\n/g;
  let headerMatch: RegExpExecArray | null;
  while ((headerMatch = HEADER.exec(text)) !== null) {
    let cursor = headerMatch.index + headerMatch[0].length;

    while (cursor < text.length) {
      // Slice the next line worth of text (up to and including the first
      // `{`) to test the bullet pattern. We don't need a full line — the
      // regex is anchored at the start of the bullet.
      const remaining = text.slice(cursor);
      const bulletMatch = BULLET_HEAD_RE.exec(remaining);
      if (!bulletMatch || bulletMatch.index !== 0) {
        // No more bullets in this block.
        break;
      }

      const name = bulletMatch[1];
      const id = `call_${bulletMatch[2]}`;
      // Position of the opening `{` in the original text.
      const braceOpenIdx = cursor + bulletMatch[0].length - 1;
      const braceEndIdx = findMatchingBraceEnd(text, braceOpenIdx);

      if (braceEndIdx < 0) {
        // Unbalanced JSON — skip this bullet and stop the block (the model's
        // output is malformed, no reliable way to find the next bullet start).
        break;
      }

      const argsJson = text.slice(braceOpenIdx, braceEndIdx);
      results.push({ id, name, arguments: argsJson });

      cursor = braceEndIdx;
      // Skip trailing whitespace / newline up to the next non-blank char.
      while (cursor < text.length && (text[cursor] === '\n' || text[cursor] === ' ' || text[cursor] === '\t')) {
        cursor++;
      }
    }
  }

  return results;
}

// ---------- Anthropic-style <invoke> / <parameter> format ----------

/**
 * Regex that locates the start of an `<invoke name="...">` element.
 *
 * Some models (notably Claude under long contexts) degrade and write tool
 * calls using Anthropic's XML invocation syntax — even when the upstream
 * server doesn't actually support this format. Example seen in the wild:
 *
 *     <invoke name="read_file">
 *     <parameter name="endLine">155</parameter>
 *     <parameter name="filePath">/path/to/file.ts</parameter>
 *     <parameter name="startLine">118</parameter>
 *     </invoke>
 *
 * This module parses those blocks back into structured tool calls so the
 * caller can re-emit them as real `tool_calls` and the user's chat client
 * can actually execute them.
 */
const INVOKE_OPEN_RE = /<invoke\s+name=["']([^"']+)["']\s*>/g;

/**
 * Regex matching a single `<parameter name="...">value</parameter>` element.
 * Non-greedy on the value so multiple parameters within one invoke parse
 * independently. The value can be a primitive (string/number/bool) or a
 * JSON object/array; we capture it raw and let the caller decide how to
 * coerce it.\n */
const PARAM_RE =
  /<parameter\s+name=["']([^"']+)["']\s*>([\s\S]*?)<\/parameter>/g;

/**
 * Returns true when the given text contains at least one `<invoke name="..."`
 * tag — fast check before running the full parser.
 */
export function containsAnthropicXmlToolCall(text: string): boolean {
  INVOKE_OPEN_RE.lastIndex = 0;
  const found = INVOKE_OPEN_RE.test(text);
  INVOKE_OPEN_RE.lastIndex = 0;
  return found;
}

/**
 * Coerce a raw parameter value to a JSON-serializable primitive.
 *
 * Tries (in order):
 *   1. Empty string → empty string
 *   2. `true` / `false` / `null` → booleans / null\n *   3. Pure-numeric string → number\n *   4. Looks like JSON object/array → JSON.parse\n *   5. Otherwise → raw trimmed string\n */
function coerceParameterValue(raw: string): unknown {
  const trimmed = raw.trim();
  if (trimmed === '') { return ''; }
  if (trimmed === 'true') { return true; }
  if (trimmed === 'false') { return false; }
  if (trimmed === 'null') { return null; }

  // Pure numeric: 123, 12.5, -3, 1e5
  if (/^-?\d+(\.\d+)?([eE][+-]?\d+)?$/.test(trimmed)) {
    const n = Number(trimmed);
    if (Number.isFinite(n)) { return n; }
  }

  // JSON object / array — attempt parse, fall back to string on failure
  if (
    (trimmed.startsWith('{') && trimmed.endsWith('}')) ||
    (trimmed.startsWith('[') && trimmed.endsWith(']'))
  ) {
    try {
      return JSON.parse(trimmed);
    } catch {
      /* fall through to string */
    }
  }

  return trimmed;
}

/**
 * Generate a synthetic tool-call id for an invoke block that doesn't carry
 * one. Format mirrors OpenAI's `call_<hex>` convention so downstream code
 * doesn't need to special-case the rescue path.
 */
function synthesizeCallId(): string {
  const rand = Math.random().toString(36).slice(2, 10);
  const ts = Date.now().toString(36);
  return `call_xml${ts}${rand}`;
}

/**
 * Parse all `<invoke name="...">...</invoke>` blocks from a text and return
 * the structured tool calls found inside them.
 *
 * - Multiple `<invoke>` blocks are aggregated in source order.
 * - Each `<parameter>` becomes a key in the arguments JSON object.
 * - Unterminated `<invoke>` blocks (missing `</invoke>`) are skipped.
 * - Parameter values are coerced via {@link coerceParameterValue}.
 * - Returns `[]` when no recognizable block is found.
 */
export function parseAnthropicXmlToolCalls(text: string): ParsedFakeToolCall[] {
  const results: ParsedFakeToolCall[] = [];

  INVOKE_OPEN_RE.lastIndex = 0;
  let openMatch: RegExpExecArray | null;
  while ((openMatch = INVOKE_OPEN_RE.exec(text)) !== null) {
    const name = openMatch[1];
    const blockStart = openMatch.index + openMatch[0].length;

    // Find matching </invoke>
    const closeIdx = text.indexOf('</invoke>', blockStart);
    if (closeIdx < 0) { break; }

    const body = text.slice(blockStart, closeIdx);

    // Extract parameters
    const args: Record<string, unknown> = {};
    PARAM_RE.lastIndex = 0;
    let paramMatch: RegExpExecArray | null;
    while ((paramMatch = PARAM_RE.exec(body)) !== null) {
      const paramName = paramMatch[1];
      const paramRaw = paramMatch[2];
      args[paramName] = coerceParameterValue(paramRaw);
    }

    results.push({
      id: synthesizeCallId(),
      name,
      arguments: JSON.stringify(args),
    });

    // Advance regex cursor past </invoke> for the next iteration
    INVOKE_OPEN_RE.lastIndex = closeIdx + '</invoke>'.length;
  }

  INVOKE_OPEN_RE.lastIndex = 0;
  return results;
}
