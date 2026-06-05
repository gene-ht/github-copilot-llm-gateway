/**
 * Token accounting utilities.
 *
 * Token estimates use UTF-8 byte length divided by BYTES_PER_TOKEN. This is
 * more accurate than char-based estimation for code/JSON-heavy content where
 * structural characters (`{`, `}`, `:`, etc.) typically tokenize 1:1 in BPE
 * tokenizers, and for multi-byte content (CJK) where one char ≠ one token.
 *
 * Empirical calibration against real Claude tokenizer output on tool-call
 * heavy conversations shows bytes/token ≈ 2.08, so BYTES_PER_TOKEN = 2 gives
 * a slight safety margin (~3-5% overestimate vs real).
 */

export const TOKEN_CONSTANTS = {
  DEFAULT_CONTEXT_TOKENS: 262144,
  DEFAULT_OUTPUT_TOKENS: 2048,
  FALLBACK_OUTPUT_TOKENS: 4096,
  MIN_OUTPUT_TOKENS: 64,
  CONTEXT_BUFFER_TOKENS: 256,
  ADJUST_TOKEN_BUFFER: 256,
  INPUT_OVERHEAD_RATIO: 1.2,
  /** UTF-8 bytes per token. Calibrated against real Claude tokenizer:
   *  - Pure English prose: ~4.5 bytes/token
   *  - Code/JSON-heavy tool conversations: ~2.1 bytes/token
   *  - CJK content: ~2-3 bytes/token (each CJK char ≈ 1 token, 3 bytes UTF-8)
   *  We use 2 as the worst-case bound so truncate fires before the server rejects. */
  BYTES_PER_TOKEN: 2,
  /** Legacy alias kept for backward-compat with code reading `CHARS_PER_TOKEN`.
   *  Now points to BYTES_PER_TOKEN; consumers should switch to byte-based
   *  estimation in their own loops. */
  CHARS_PER_TOKEN: 4,
} as const;

export type TokenLogger = (message: string) => void;

const NOOP_LOGGER: TokenLogger = () => {
  /* no-op */
};

/**
 * Minimal message shape needed for token estimation. Intentionally structural
 * so callers can pass OpenAI wire-format messages without a type cast.
 */
export interface TokenEstimableMessage {
  content?: string | object | null;
  tool_calls?: unknown;
}

/**
 * Estimate token count for a text string using UTF-8 byte length divided by
 * BYTES_PER_TOKEN. Falls back to char length × 1.0 (≈ 1 byte/char) when
 * Buffer is unavailable (e.g. in non-Node environments) — but in Node/VS Code
 * extension host we always have Buffer.
 */
export function estimateTextTokens(text: string): number {
  if (typeof text !== 'string' || text.length === 0) { return 0; }
  // Use UTF-8 byte length: more accurate than char length for CJK and JSON.
  // Buffer is always available in the VS Code extension host (Node runtime).
  const byteLen = typeof Buffer !== 'undefined'
    ? Buffer.byteLength(text, 'utf8')
    : text.length;
  return Math.ceil(byteLen / TOKEN_CONSTANTS.BYTES_PER_TOKEN);
}

/**
 * Estimate tokens for an OpenAI-format message, including tool_calls if present.
 */
export function estimateMessageTokens(message: TokenEstimableMessage): number {
  let text = '';
  if (typeof message.content === 'string') {
    text = message.content;
  } else if (message.content) {
    text = JSON.stringify(message.content);
  }
  if (message.tool_calls) {
    text += JSON.stringify(message.tool_calls);
  }
  return estimateTextTokens(text);
}

/**
 * Concatenate all message text into a single string, mirroring what we'd send
 * on the wire. Used as input to {@link estimateTextTokens}.
 */
export function buildInputText(messages: readonly TokenEstimableMessage[]): string {
  return messages
    .map((m) => {
      let text = typeof m.content === 'string' ? m.content : JSON.stringify(m.content ?? '');
      if (m.tool_calls) {
        text += JSON.stringify(m.tool_calls);
      }
      return text;
    })
    .join('\n');
}

/**
 * Truncate messages to fit within `maxTokens`.
 *
 * Strategy: always keep the first message (typically the system prompt) and
 * as many trailing messages as will fit, working backwards from the end.
 * Mid-conversation messages are dropped first.
 */
export function truncateMessagesToFit<T extends TokenEstimableMessage>(
  messages: readonly T[],
  maxTokens: number,
  log: TokenLogger = NOOP_LOGGER
): T[] {
  if (messages.length === 0) {
    return [];
  }

  let totalTokens = 0;
  const messageTokens: number[] = [];
  for (const msg of messages) {
    const tokens = estimateMessageTokens(msg);
    messageTokens.push(tokens);
    totalTokens += tokens;
  }

  if (totalTokens <= maxTokens) {
    return [...messages];
  }

  log(`Context overflow: ${totalTokens} tokens > ${maxTokens} limit. Truncating...`);

  const result: T[] = [messages[0]];
  let usedTokens = messageTokens[0];

  const recentMessages: T[] = [];
  for (let i = messages.length - 1; i > 0; i--) {
    const msgTokens = messageTokens[i];
    if (usedTokens + msgTokens <= maxTokens) {
      recentMessages.unshift(messages[i]);
      usedTokens += msgTokens;
    } else {
      break;
    }
  }

  result.push(...recentMessages);
  log(`Truncated: kept ${result.length}/${messages.length} messages, ~${usedTokens} tokens`);
  return result;
}

/**
 * Internal: read a string field from an unknown message-like object without
 * relying on a stricter shape. Returns undefined if the field is missing or
 * not a string. Kept as a tiny helper so `repairToolCallPairing` stays readable.
 */
function getStringField(msg: unknown, field: string): string | undefined {
  if (typeof msg !== 'object' || msg === null) { return undefined; }
  const value = (msg as Record<string, unknown>)[field];
  return typeof value === 'string' ? value : undefined;
}

/**
 * Read an `assistant.tool_calls[]` array off a message and return the set of
 * `tool_call.id` values it advertises. Returns an empty Set for messages that
 * don't carry tool_calls (so the caller can do `has()` lookups uniformly).
 */
function getAdvertisedToolCallIds(msg: unknown): Set<string> {
  const ids = new Set<string>();
  if (typeof msg !== 'object' || msg === null) { return ids; }
  const toolCalls = (msg as Record<string, unknown>).tool_calls;
  if (!Array.isArray(toolCalls)) { return ids; }
  for (const call of toolCalls) {
    const id = getStringField(call, 'id');
    if (id) { ids.add(id); }
  }
  return ids;
}

/**
 * Strip messages that would violate the Anthropic tool_use ↔ tool_result
 * pairing rule when the conversation gets relayed to an Anthropic-backed
 * gateway. This guards against two failure modes introduced by mid-context
 * truncation:
 *
 *   1. Orphan tool result — a `role: tool` message whose `tool_call_id` is not
 *      advertised by the immediately preceding assistant message (because the
 *      assistant call was truncated away). Anthropic returns:
 *        "unexpected tool_use_id found in tool_result blocks: <id>".
 *      We drop these orphan tool messages.
 *   2. Orphan assistant tool_call — an assistant message that advertises
 *      `tool_calls` but is not followed by the matching `tool` results
 *      (because they were truncated away or never produced). Anthropic will
 *      reject this with a similar mismatch error. We drop the `tool_calls`
 *      array from the assistant message; if that leaves the message with no
 *      content either, we drop the whole assistant turn.
 *
 * The function is conservative: it only removes messages or fields that would
 * cause a 400 from a strict backend. Well-formed conversations pass through
 * unchanged. Idempotent: calling it on already-clean input is a no-op.
 *
 * This is a post-pass over `truncateMessagesToFit` rather than a change to the
 * truncation algorithm itself, so the greedy "fit as many recent turns as
 * possible" behaviour is preserved.
 */
export function repairToolCallPairing<T extends TokenEstimableMessage>(
  messages: readonly T[],
  log: TokenLogger = NOOP_LOGGER
): T[] {
  if (messages.length === 0) { return []; }

  // Pass 1: find ids that DO have a matching follow-up `tool` message in the
  // truncated transcript, so we know which assistant tool_calls are "answered"
  // vs. dangling.
  const answeredIds = new Set<string>();
  for (const msg of messages) {
    if (getStringField(msg, 'role') === 'tool') {
      const id = getStringField(msg, 'tool_call_id');
      if (id) { answeredIds.add(id); }
    }
  }

  const result: T[] = [];
  let droppedOrphanResults = 0;
  let droppedOrphanAssistants = 0;
  let strippedToolCallArrays = 0;

  // Pass 2: walk forward, dropping orphan tool results and unanchored
  // assistant tool_calls. We track the most-recently-emitted assistant's
  // advertised tool_call ids so we can validate the very next batch of
  // `role: tool` messages against it.
  let lastAssistantToolIds = new Set<string>();
  for (const msg of messages) {
    const role = getStringField(msg, 'role');

    if (role === 'tool') {
      const id = getStringField(msg, 'tool_call_id');
      if (id && lastAssistantToolIds.has(id)) {
        result.push(msg);
        lastAssistantToolIds.delete(id);
      } else {
        droppedOrphanResults++;
      }
      continue;
    }

    if (role === 'assistant') {
      const advertised = getAdvertisedToolCallIds(msg);
      if (advertised.size === 0) {
        result.push(msg);
        lastAssistantToolIds = new Set<string>();
        continue;
      }
      // Filter to only those tool_calls whose responses survived truncation,
      // so we don't leave a dangling tool_use the backend will reject.
      const survivingCalls = (msg as unknown as { tool_calls: Array<{ id?: unknown }> })
        .tool_calls.filter((c) => {
          const id = getStringField(c, 'id');
          return id !== undefined && answeredIds.has(id);
        });

      if (survivingCalls.length === advertised.size) {
        result.push(msg);
        lastAssistantToolIds = new Set(survivingCalls.map((c) => String(c.id)));
        continue;
      }
      strippedToolCallArrays++;
      const hasContent =
        getStringField(msg, 'content') !== undefined ||
        Array.isArray((msg as Record<string, unknown>).content);
      if (survivingCalls.length === 0 && !hasContent) {
        // Nothing useful left in this assistant turn — drop it entirely.
        droppedOrphanAssistants++;
        lastAssistantToolIds = new Set<string>();
        continue;
      }
      const cloned = { ...(msg as Record<string, unknown>) };
      if (survivingCalls.length === 0) {
        delete cloned.tool_calls;
      } else {
        cloned.tool_calls = survivingCalls;
      }
      result.push(cloned as unknown as T);
      lastAssistantToolIds = new Set(survivingCalls.map((c) => String(c.id)));
      continue;
    }

    // user / system / anything else — pass through and reset pending tool
    // expectations, since a tool result block must directly follow its
    // assistant turn per Anthropic protocol.
    result.push(msg);
    lastAssistantToolIds = new Set<string>();
  }

  if (droppedOrphanResults || droppedOrphanAssistants || strippedToolCallArrays) {
    log(
      `Tool pairing repair: dropped ${droppedOrphanResults} orphan tool result(s), ` +
        `${droppedOrphanAssistants} orphan assistant turn(s), ` +
        `stripped tool_calls from ${strippedToolCallArrays} assistant turn(s)`
    );
  }
  return result;
}

export interface SafeOutputTokensParams {
  estimatedInputTokens: number;
  toolsOverhead: number;
  modelMaxContext: number;
  configuredMaxOutput: number;
}

/**
 * Given an input-token estimate, decide how many output tokens we can safely
 * request without tripping context-length errors. Adds INPUT_OVERHEAD_RATIO
 * slack to account for tokenizer drift and a fixed CONTEXT_BUFFER_TOKENS.
 */
export function calculateSafeMaxOutputTokens(params: SafeOutputTokensParams): number {
  const totalEstimatedTokens = params.estimatedInputTokens + params.toolsOverhead;
  const conservativeInputEstimate = Math.ceil(
    totalEstimatedTokens * TOKEN_CONSTANTS.INPUT_OVERHEAD_RATIO
  );

  const safeMaxOutputTokens = Math.min(
    params.configuredMaxOutput,
    Math.floor(params.modelMaxContext - conservativeInputEstimate - TOKEN_CONSTANTS.CONTEXT_BUFFER_TOKENS)
  );

  return Math.max(TOKEN_CONSTANTS.MIN_OUTPUT_TOKENS, safeMaxOutputTokens);
}

export interface MaxInputTokensParams {
  modelMaxContext: number;
  configuredMaxOutput: number;
  toolsSerializedLength: number;
}

/**
 * Compute the ceiling on input tokens for a request so there's still room
 * for output + tools in the context window.
 */
export function calculateMaxInputTokens(params: MaxInputTokensParams): number {
  const desiredOutputTokens = Math.min(
    params.configuredMaxOutput,
    Math.floor(params.modelMaxContext / 2)
  );
  const toolsTokenEstimate = Math.ceil(
    (params.toolsSerializedLength / TOKEN_CONSTANTS.CHARS_PER_TOKEN) *
      TOKEN_CONSTANTS.INPUT_OVERHEAD_RATIO
  );
  return params.modelMaxContext - desiredOutputTokens - toolsTokenEstimate - TOKEN_CONSTANTS.CONTEXT_BUFFER_TOKENS;
}

/**
 * Merge consecutive messages that share the same role into a single message.
 *
 * Anthropic-backed OpenAI-compatible gateways convert the OpenAI message list
 * into Anthropic's strict user/assistant alternation format. When consecutive
 * user messages appear (e.g. system-injected instructions followed by the
 * actual user turn), the gateway merges them into one Anthropic "user" message.
 * This can shift message boundaries and cause tool_result blocks to land in the
 * wrong Anthropic message, producing:
 *   "unexpected tool_use_id found in tool_result blocks"
 *
 * By pre-merging consecutive same-role messages on our side, we ensure our
 * message boundaries match what the gateway will produce, so tool_use/tool_result
 * pairing stays intact.
 *
 * Only merges user and assistant messages with string/array content.
 * Does NOT merge role:tool messages (each must keep its own tool_call_id).
 */
export function mergeConsecutiveSameRoleMessages<T extends TokenEstimableMessage>(
  messages: readonly T[],
  log: TokenLogger = NOOP_LOGGER
): T[] {
  if (messages.length <= 1) { return [...messages]; }

  const result: T[] = [];
  let mergedCount = 0;

  for (const msg of messages) {
    const role = getStringField(msg, 'role');
    const prev = result.length > 0 ? result[result.length - 1] : undefined;
    const prevRole = prev ? getStringField(prev, 'role') : undefined;

    // Only merge consecutive user messages (not tool, not assistant with tool_calls)
    if (
      role === 'user' &&
      prevRole === 'user' &&
      prev &&
      !getStringField(msg, 'tool_call_id') &&
      !getStringField(prev, 'tool_call_id')
    ) {
      // Merge content
      const prevContent = (prev as Record<string, unknown>).content;
      const currContent = (msg as Record<string, unknown>).content;
      const merged = { ...(prev as Record<string, unknown>) };

      if (typeof prevContent === 'string' && typeof currContent === 'string') {
        merged.content = prevContent + '\n' + currContent;
      } else {
        // Convert both to array form and concatenate
        const toArray = (c: unknown): unknown[] => {
          if (Array.isArray(c)) { return c; }
          if (typeof c === 'string') { return [{ type: 'text', text: c }]; }
          return [];
        };
        merged.content = [...toArray(prevContent), ...toArray(currContent)];
      }

      result[result.length - 1] = merged as unknown as T;
      mergedCount++;
      continue;
    }

    result.push(msg);
  }

  if (mergedCount > 0) {
    log(`Merged ${mergedCount} consecutive same-role user message(s) (${messages.length} → ${result.length})`);
  }
  return result;
}
