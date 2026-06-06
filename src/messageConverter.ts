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
 * Strip "Completed tool calls: …" textual summaries from assistant content.
 *
 * Some Copilot Chat clients/agents embed a human-readable summary of executed
 * tool calls into the assistant message's `content` *in addition to* (or
 * sometimes instead of) the structured `tool_calls` field. When that text is
 * fed back in subsequent turns, models — especially under long contexts —
 * start mimicking the format and emit tool calls as plain text, skipping
 * actual execution.
 *
 * Removing the textual block leaves the structured `tool_calls` field intact
 * (this function never touches it), so real tool-call semantics are preserved
 * while the few-shot poisoning trigger is removed.
 *
 * Only affects assistant messages with string content. Messages that become
 * empty after stripping are dropped entirely (rare — usually the assistant
 * also has its own analysis text).
 */
export function stripFakeToolCallText(
  messages: readonly OpenAIMessage[],
  log: ConverterLogger = NOOP_LOGGER
): OpenAIMessage[] {
  let strippedBulletCount = 0;
  let strippedXmlCount = 0;
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

    FAKE_TOOL_CALL_TEXT_RE.lastIndex = 0;
    ANTHROPIC_XML_TOOL_CALL_RE.lastIndex = 0;
    let cleaned = content;
    if (hasBullet) {
      cleaned = cleaned.replace(FAKE_TOOL_CALL_TEXT_RE, '\n');
      strippedBulletCount++;
    }
    if (hasXml) {
      cleaned = cleaned.replace(ANTHROPIC_XML_TOOL_CALL_RE, '\n');
      strippedXmlCount++;
    }
    cleaned = cleaned.trim();

    // If the assistant message had a real tool_calls field, keep it even
    // when the human-readable part became empty — the structured call is
    // what actually drives the next-turn tool/role message.
    const hasToolCalls = Array.isArray((msg as Record<string, unknown>).tool_calls)
      && ((msg as Record<string, unknown>).tool_calls as unknown[]).length > 0;

    if (cleaned.length > 0 || hasToolCalls) {
      result.push({ ...msg, content: cleaned });
    }
  }

  if (strippedBulletCount > 0) {
    log(`Stripped "Completed tool calls" text from ${strippedBulletCount} assistant message(s)`);
  }
  if (strippedXmlCount > 0) {
    log(`Stripped Anthropic <invoke> XML from ${strippedXmlCount} assistant message(s)`);
  }
  return result;
}
