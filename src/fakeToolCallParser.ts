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
