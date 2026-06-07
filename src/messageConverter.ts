/**
 * Convert normalized chat messages into the OpenAI wire format.
 *
 * This module is intentionally free of any VS Code imports — callers pass in
 * {@link NormalizedMessage}s, which are plain-data descriptions of each part
 * (text, tool call, tool result, image). The provider is responsible for
 * translating `vscode.LanguageModel*Part` instances into these descriptors;
 * that's where the `instanceof` checks and duck-typed fallbacks live.
 *
 * This split makes the converter trivially unit-testable and eliminates the
 * God-object shape that provider.ts used to have.
 */

import { OpenAIMessage } from './types';
import {
  parseFakeToolCalls,
  parseAnthropicXmlToolCalls,
  ParsedFakeToolCall,
} from './fakeToolCallParser';

export type NormalizedRole = 'user' | 'assistant' | 'system' | 'tool';

export type NormalizedPart =
  | { kind: 'text'; value: string }
  | { kind: 'toolResult'; callId: string; content: string }
  | { kind: 'toolCall'; callId: string; name: string; input: unknown }
  | { kind: 'image'; mimeType: string; data: Uint8Array }
  | { kind: 'unknown' };

export interface NormalizedMessage {
  role: NormalizedRole;
  parts: NormalizedPart[];
}

export type ConverterLogger = (message: string) => void;

export interface MessageConverterOptions {
  enableImageInput: boolean;
}

const NOOP_LOGGER: ConverterLogger = () => {
  /* no-op */
};

type UserContentPart =
  | { type: 'text'; text: string }
  | { type: 'image_url'; image_url: { url: string } };

/**
 * Encode an image data part as a `data:` URL suitable for the OpenAI
 * multimodal `image_url` message shape.
 *
 * Uses Node's Buffer for base64 encoding — the previous `btoa(String.fromCodePoint(...data))`
 * approach spread the whole byte array onto the JS call stack and threw
 * `RangeError: Maximum call stack size exceeded` on images larger than ~65 KB.
 */
export function encodeImageAsDataUrl(part: { mimeType: string; data: Uint8Array }): string {
  const base64Data = Buffer.from(part.data).toString('base64');
  return `data:${part.mimeType};base64,${base64Data}`;
}

/**
 * Convert a normalized message into zero or more OpenAI wire messages.
 *
 * The conversion is lossy-but-deliberate:
 *  - pure-text user/assistant messages collapse into `{ role, content: text }`
 *  - messages that carry image parts become `{ role, content: UserContentPart[] }`
 *  - assistant messages with tool calls become `{ role: 'assistant', tool_calls }`
 *  - tool result parts are flattened into their own `{ role: 'tool' }` messages
 *
 * When `enableImageInput` is false, image parts are dropped with a log line.
 */
export function convertMessage(
  message: NormalizedMessage,
  options: MessageConverterOptions,
  log: ConverterLogger = NOOP_LOGGER
): OpenAIMessage[] {
  const toolResults: OpenAIMessage[] = [];
  const toolCalls: OpenAIMessage[] = [];
  const userContent: UserContentPart[] = [];
  let textContent = '';

  for (const part of message.parts) {
    switch (part.kind) {
      case 'text':
        if (message.role === 'user') {
          userContent.push({ type: 'text', text: part.value });
        }
        textContent += part.value;
        break;

      case 'toolResult':
        log(`  Found tool result: callId=${part.callId}`);
        toolResults.push({
          tool_call_id: part.callId,
          role: 'tool',
          content: part.content,
        });
        break;

      case 'toolCall':
        log(`  Found tool call: callId=${part.callId}, name=${part.name}`);
        toolCalls.push({
          id: part.callId,
          type: 'function',
          function: {
            name: part.name,
            arguments: JSON.stringify(part.input),
          },
        });
        break;

      case 'image':
        if (message.role !== 'user') {
          log(`  Skipping image data part on non-user message: role=${message.role}`);
          break;
        }
        if (!options.enableImageInput) {
          log(
            `  Skipping data part: mimeType=${part.mimeType}, size=${part.data.length} bytes. (Please enable github.copilot.llm-gateway.enableImageInput in settings)`
          );
          break;
        }
        if (part.mimeType.startsWith('image/')) {
          const url = encodeImageAsDataUrl(part);
          userContent.push({ type: 'image_url', image_url: { url } });
          log(
            `  Added image data part as base64 URL: mimeType=${part.mimeType}, size=${part.data.length} bytes, urlLength=${url.length}`
          );
        }
        break;

      case 'unknown':
        // Unknown parts are silently dropped; the classifier has already logged.
        break;

      default: {
        const _never: never = part;
        throw new Error(`Unexpected part kind: ${String(_never)}`);
      }
    }
  }

  const result: OpenAIMessage[] = [];
  if (toolCalls.length > 0) {
    // Some OpenAI-compatible Anthropic gateways drop assistant tool-call turns
    // whose content is JSON null, which makes the following role:tool message
    // look like an orphan tool_result. Use an empty string instead.
    result.push({ role: 'assistant', content: textContent || '', tool_calls: toolCalls });
  } else if (toolResults.length > 0) {
    result.push(...toolResults);
  } else if (message.role === 'user' && userContent.length > 0) {
    // Use string content for pure-text user messages (no images).
    // Some Anthropic-compatible gateways treat array-content user messages
    // differently (e.g. extracting as system prompt), which can shift message
    // boundaries and break tool_use/tool_result pairing.
    const hasNonText = userContent.some((p) => p.type !== 'text');
    if (hasNonText) {
      result.push({ role: message.role, content: userContent });
    } else {
      result.push({ role: message.role, content: textContent });
    }
  } else if (textContent) {
    result.push({ role: message.role, content: textContent });
  }
  return result;
}

/**
 * Convert a list of normalized messages into the flat OpenAI message stream
 * sent to the server.
 */
export function convertMessages(
  messages: readonly NormalizedMessage[],
  options: MessageConverterOptions,
  log: ConverterLogger = NOOP_LOGGER
): OpenAIMessage[] {
  const result: OpenAIMessage[] = [];
  for (const msg of messages) {
    result.push(...convertMessage(msg, options, log));
  }
  return result;
}


/**
 * Regex that matches Copilot Chat's "Completed tool calls" textual summary
 * blocks inside assistant content. Format observed in the wild:
 *
 *     Completed tool calls:
 *     - read_file (call_91e3b9d70b8c4be0a4d3c1f7) {"endLine":205,...}
 *     - replace_string_in_file (call_a6f7e54b2eaa4b9c812a2cae) {"filePath":...}
 *
 * The block is rendered for human-readable history but, when it round-trips
 * back into the next request's `messages[]`, it acts as a few-shot example
 * teaching the model to *write* tool calls as text instead of emitting a
 * real `tool_calls` array. The resulting self-poisoning loop is the failure
 * mode this regex breaks.
 *
 * Conservative on purpose: requires the literal header and at least one
 * `- name (call_<hex>) {...}` bullet. Won't touch normal markdown lists or
 * legitimate code blocks.
 */
const FAKE_TOOL_CALL_TEXT_RE =
  /\n*Completed tool calls:\s*\n(?:[ \t]*-[ \t]+\S+[ \t]*\(call_[0-9a-zA-Z]+\)[ \t]*\{[\s\S]*?\}[ \t]*\n?)+/g;

/**
 * Regex that matches Anthropic-style `<invoke name="...">...</invoke>` blocks
 * — the second known fake-toolcall self-poisoning format. Example:
 *
 *     <invoke name="read_file">
 *     <parameter name="endLine">155</parameter>
 *     <parameter name="filePath">/path/to/file.ts</parameter>
 *     </invoke>
 *
 * Same self-poisoning concern as the "Completed tool calls:" format — once
 * an assistant message contains this text, future turns will mimic it.
 *
 * Also matches an optional leading `call` keyword and trailing whitespace so
 * the surrounding "call\n<invoke>...</invoke>" preamble seen in real
 * captures gets cleaned up.
 */
const ANTHROPIC_XML_TOOL_CALL_RE =
  /(?:^|\n)[ \t]*(?:call[ \t]*\n[ \t]*)?<invoke\s+name=["'][^"']+["']\s*>[\s\S]*?<\/invoke>[ \t]*\n?/g;

/**
 * Strip fake-toolcall text patterns from a string. Used by both the OpenAI
 * message-level stripper and the Anthropic converter to clean assistant text
 * before sending to the model.
 */
export function stripPoisonText(text: string): string {
  FAKE_TOOL_CALL_TEXT_RE.lastIndex = 0;
  ANTHROPIC_XML_TOOL_CALL_RE.lastIndex = 0;
  let cleaned = text
    .replace(FAKE_TOOL_CALL_TEXT_RE, '\n')
    .replace(ANTHROPIC_XML_TOOL_CALL_RE, '\n');
  FAKE_TOOL_CALL_TEXT_RE.lastIndex = 0;
  ANTHROPIC_XML_TOOL_CALL_RE.lastIndex = 0;
  return cleaned.trim();
}

/**
 * Returns true when the given text contains either known fake-toolcall
 * format (Copilot's "Completed tool calls:" block or Anthropic-style
 * `<invoke>` XML). Useful for detecting self-poisoning in streaming output
 * *before* the text is fed back into history.
 */
export function containsFakeToolCallText(text: string): boolean {
  FAKE_TOOL_CALL_TEXT_RE.lastIndex = 0;
  if (FAKE_TOOL_CALL_TEXT_RE.test(text)) {
    FAKE_TOOL_CALL_TEXT_RE.lastIndex = 0;
    return true;
  }
  ANTHROPIC_XML_TOOL_CALL_RE.lastIndex = 0;
  const xmlMatch = ANTHROPIC_XML_TOOL_CALL_RE.test(text);
  ANTHROPIC_XML_TOOL_CALL_RE.lastIndex = 0;
  return xmlMatch;
}

/**
 * Transform fake-toolcall text in historical assistant messages into
 * structured `tool_calls` + synthetic `tool` result messages.
 *
 * ## Background
 *
 * Some Copilot Chat clients/agents embed tool calls as plain text in the
 * assistant's `content` field instead of (or in addition to) the structured
 * `tool_calls` field. Two formats observed:
 *
 *   1. Copilot bullet:  `Completed tool calls:\n- foo (call_xxx) {...}`
 *   2. Anthropic XML:   `<invoke name="foo"><parameter ...>...</invoke>`
 *
 * When those texts are fed back as history, they become a few-shot example
 * teaching the model to keep writing tool calls as text — and worse, since
 * no structured `tool_calls` exists, the LLM thinks its previous turn never
 * actually invoked anything, prompting it to retry. This causes a
 * **retry loop**: every turn the LLM re-writes the same fake call, the
 * client never executes it (because it's text), and Copilot keeps retrying.
 *
 * ## Fix
 *
 * For each affected assistant message we:
 *
 *   1. Parse the fake-toolcall text back into structured calls (`{id, name,
 *      arguments}`) using the existing parsers.
 *   2. Strip the text from `content`.
 *   3. Set `tool_calls` on the assistant message to the parsed calls (or
 *      merge with existing `tool_calls` if the assistant already had some).
 *   4. Insert synthetic `tool` role messages right after the assistant
 *      message — one per parsed call — with a placeholder content. This is
 *      required by the OpenAI/Anthropic protocol (every tool_call must have
 *      a matching tool result in history) and signals to the LLM that the
 *      call already completed, so it won't retry.
 *
 * ## Placeholder content
 *
 * Since the original tool result is gone (it was textual and never executed),
 * we inject a short marker. The LLM treats this as the result and moves on.
 */
export function stripFakeToolCallText(
  messages: readonly OpenAIMessage[],
  log: ConverterLogger = NOOP_LOGGER
): OpenAIMessage[] {
  let strippedBulletCount = 0;
  let strippedXmlCount = 0;
  let synthesizedCallCount = 0;
  const result: OpenAIMessage[] = [];

  for (const msg of messages) {
    const role = (msg as Record<string, unknown>).role;
    const content = (msg as Record<string, unknown>).content;

    if (role !== 'assistant' || typeof content !== 'string') {
      result.push(msg);
      continue;
    }

    FAKE_TOOL_CALL_TEXT_RE.lastIndex = 0;
    ANTHROPIC_XML_TOOL_CALL_RE.lastIndex = 0;
    const hasBullet = FAKE_TOOL_CALL_TEXT_RE.test(content);
    const hasXml = ANTHROPIC_XML_TOOL_CALL_RE.test(content);

    if (!hasBullet && !hasXml) {
      result.push(msg);
      continue;
    }

    // Parse BEFORE stripping so we can transform text → structured calls.
    const parsedCalls: ParsedFakeToolCall[] = [];
    if (hasBullet) {
      parsedCalls.push(...parseFakeToolCalls(content));
      strippedBulletCount++;
    }
    if (hasXml) {
      parsedCalls.push(...parseAnthropicXmlToolCalls(content));
      strippedXmlCount++;
    }

    // Strip text from content.
    FAKE_TOOL_CALL_TEXT_RE.lastIndex = 0;
    ANTHROPIC_XML_TOOL_CALL_RE.lastIndex = 0;
    let cleaned = content;
    if (hasBullet) {
      cleaned = cleaned.replace(FAKE_TOOL_CALL_TEXT_RE, '\n');
    }
    if (hasXml) {
      cleaned = cleaned.replace(ANTHROPIC_XML_TOOL_CALL_RE, '\n');
    }
    cleaned = cleaned.trim();

    // Merge parsed calls with existing tool_calls. Dedupe by id — if the
    // assistant already has a real tool_call with the same id as a parsed
    // text bullet (idx-537 hybrid form), the real one wins and we skip the
    // synthetic copy + result (the real tool result will follow naturally).
    const existingToolCalls = Array.isArray((msg as Record<string, unknown>).tool_calls)
      ? ((msg as Record<string, unknown>).tool_calls as Array<Record<string, unknown>>)
      : [];
    const existingIds = new Set(
      existingToolCalls
        .map((tc) => (typeof tc.id === 'string' ? tc.id : ''))
        .filter((id) => id.length > 0)
    );

    const newSyntheticCalls = parsedCalls
      .filter((p) => !existingIds.has(p.id))
      .map((p) => ({
        id: p.id,
        type: 'function' as const,
        function: { name: p.name, arguments: p.arguments },
      }));

    // Emit the assistant message with merged tool_calls.
    const allToolCalls = [...existingToolCalls, ...newSyntheticCalls];
    const newMsg: Record<string, unknown> = { ...msg, content: cleaned };
    if (allToolCalls.length > 0) {
      newMsg.tool_calls = allToolCalls;
    }
    // Drop the assistant message entirely only if content is empty AND no
    // tool_calls.
    if (cleaned.length > 0 || allToolCalls.length > 0) {
      result.push(newMsg as OpenAIMessage);
    }

    // Inject synthetic tool result messages — only for the *new* synthetic
    // calls. Existing real tool_calls have their own tool result messages
    // following in history (we don't double-emit).
    //
    // The chat-completions protocol requires every tool_call in an assistant
    // message to have a matching tool message following it. Without these
    // synthetic results, the LLM sees an unanswered tool call and retries.
    for (const call of newSyntheticCalls) {
      result.push({
        role: 'tool',
        tool_call_id: call.id,
        content:
          '[Synthesized by LLM Gateway: the model wrote this tool call as plain text ' +
          'in a previous turn, so it was not actually executed. Treat this call as a ' +
          'no-op and continue.]',
      } as OpenAIMessage);
      synthesizedCallCount++;
    }
  }

  if (strippedBulletCount > 0) {
    log(`Stripped "Completed tool calls" text from ${strippedBulletCount} assistant message(s)`);
  }
  if (strippedXmlCount > 0) {
    log(`Stripped Anthropic <invoke> XML from ${strippedXmlCount} assistant message(s)`);
  }
  if (synthesizedCallCount > 0) {
    log(
      `Synthesized ${synthesizedCallCount} structured tool_call(s) + tool result(s) ` +
      'from fake-toolcall text to break self-poisoning retry loop'
    );
  }
  return result;
}
