/**
 * Context compaction for the LM passthrough path.
 *
 * When the upstream returns "prompt is too long", this module:
 *   1. Splits messages into [early history] + [recent rounds]
 *   2. Summarizes the early history using the same model
 *   3. Rebuilds the message list as [summary] + [recent rounds]
 *
 * The summary preserves key context (goals, decisions, file paths, code)
 * while dramatically reducing token count.
 */

import * as vscode from 'vscode';

// ============================================================================
// Error detection
// ============================================================================

/**
 * Check if an error indicates the prompt exceeded the model's context window.
 * Matches error strings from Anthropic, OpenAI, and common gateway formats.
 */
export function isContextTooLongError(error: unknown): boolean {
  const msg = error instanceof Error ? error.message : String(error);
  return (
    msg.includes('prompt is too long') ||
    msg.includes('context_length_exceeded') ||
    msg.includes('model_context_window_exceeded') ||
    msg.includes('too many tokens') ||
    /max.*tokens?\s*exceeded/i.test(msg) ||
    /tokens?\s*>\s*\d+\s*maximum/i.test(msg)
  );
}

// ============================================================================
// Message splitting
// ============================================================================

/**
 * Split messages into [early history to summarize] + [recent rounds to keep].
 *
 * Strategy: keep the last `keepRounds` complete tool-call rounds + trailing
 * messages. A "round" is an assistant message with tool_call parts followed
 * by a user message with tool_result parts.
 *
 * @param keepRounds Number of recent complete rounds to preserve verbatim
 */
export function splitForCompaction(
  messages: readonly vscode.LanguageModelChatMessage[],
  keepRounds: number = 3,
): { early: vscode.LanguageModelChatMessage[]; recent: vscode.LanguageModelChatMessage[] } {
  const mutableMessages = [...messages];

  if (mutableMessages.length <= keepRounds * 2 + 2) {
    // Too few messages to compact — keep everything
    return { early: [], recent: mutableMessages };
  }

  // Walk backward to find `keepRounds` complete tool-call rounds.
  let splitIndex = mutableMessages.length;
  let roundsFound = 0;

  for (let i = mutableMessages.length - 1; i >= 0; i--) {
    const msg = mutableMessages[i];
    // A complete round boundary: user message containing tool_result
    const hasToolResult =
      msg.role === vscode.LanguageModelChatMessageRole.User &&
      msg.content.some((p) => p instanceof vscode.LanguageModelToolResultPart);

    if (hasToolResult) {
      roundsFound++;
      if (roundsFound >= keepRounds) {
        // Find the corresponding assistant message before this user message
        let j = i - 1;
        while (
          j >= 0 &&
          mutableMessages[j].role !== vscode.LanguageModelChatMessageRole.Assistant
        ) {
          j--;
        }
        splitIndex = j >= 0 ? j : i;
        break;
      }
    }
  }

  // Ensure we keep at least the last few messages
  if (splitIndex <= 1) {
    splitIndex = Math.max(1, Math.floor(mutableMessages.length / 2));
  }

  return {
    early: mutableMessages.slice(0, splitIndex),
    recent: mutableMessages.slice(splitIndex),
  };
}

// ============================================================================
// Summarization
// ============================================================================

const SUMMARIZATION_SYSTEM_PROMPT = `You are a helpful AI assistant tasked with summarizing conversations.
Given the following conversation history between a user and an assistant, create a concise but comprehensive summary that captures:
- The user's goals and requirements
- Key decisions made
- Important context and constraints
- Current state of the task
- Any files, code, or tools that were discussed

Format the summary as a clear narrative. Preserve technical details, file paths, and code references.
Do NOT include tool call/result details unless they produced critical insights.`;

/**
 * Convert a LanguageModelChatMessage to a plain-text representation for
 * summarization. Tool calls/results are condensed to short markers.
 */
function messageToText(msg: vscode.LanguageModelChatMessage): string {
  const role =
    msg.role === vscode.LanguageModelChatMessageRole.Assistant
      ? 'Assistant'
      : 'User';

  const text = msg.content
    .map((part) => {
      if (part instanceof vscode.LanguageModelTextPart) {
        return part.value;
      }
      if (part instanceof vscode.LanguageModelToolCallPart) {
        return `[Tool call: ${part.name}]`;
      }
      if (part instanceof vscode.LanguageModelToolResultPart) {
        const resultText = part.content
          .map((c) =>
            c instanceof vscode.LanguageModelTextPart ? c.value : '[data]'
          )
          .join('');
        // Truncate long tool results to avoid blowing up the summary prompt
        return `[Tool result: ${resultText.slice(0, 500)}${resultText.length > 500 ? '...' : ''}]`;
      }
      return '';
    })
    .join('');

  return `${role}: ${text}`;
}

/**
 * Maximum characters of conversation text to feed into the summarization
 * prompt. If the early history exceeds this, only the first + last portions
 * are included (with a gap marker) to keep the summary request itself from
 * hitting context limits.
 */
const MAX_SUMMARY_INPUT_CHARS = 200_000;

/**
 * Summarize early conversation history using the VS Code language model API.
 *
 * Uses the same model as the current request so the summary style matches.
 * If the model isn't available via `vscode.lm`, falls back to a simple
 * truncation (first + last messages only).
 */
export async function summarizeHistory(
  earlyMessages: readonly vscode.LanguageModelChatMessage[],
  modelId: string,
  token: vscode.CancellationToken,
  log: (msg: string) => void,
): Promise<string> {
  // Build conversation text for summarization
  let conversationText = earlyMessages
    .map((msg) => messageToText(msg))
    .join('\n\n');

  // Truncate if the conversation text itself is too long
  if (conversationText.length > MAX_SUMMARY_INPUT_CHARS) {
    const half = Math.floor(MAX_SUMMARY_INPUT_CHARS / 2);
    conversationText =
      conversationText.slice(0, half) +
      '\n\n[... middle portion omitted for brevity ...]\n\n' +
      conversationText.slice(-half);
    log(
      `[compaction] Conversation text truncated to ${MAX_SUMMARY_INPUT_CHARS} chars for summarization`
    );
  }

  const summaryPrompt = `${SUMMARIZATION_SYSTEM_PROMPT}\n\n---\n\nConversation to summarize:\n\n${conversationText}\n\n---\n\nProvide a concise summary:`;

  log(
    `[compaction] Summarizing ${earlyMessages.length} early messages ` +
    `(${conversationText.length} chars) using ${modelId}...`
  );

  // Try to get the model via vscode.lm API
  try {
    const models = await vscode.lm.selectChatModels({ id: modelId });
    if (models.length === 0) {
      log(`[compaction] Model ${modelId} not available via vscode.lm, using simple truncation`);
      return buildFallbackSummary(earlyMessages);
    }

    const model = models[0];
    const summaryMessages = [
      vscode.LanguageModelChatMessage.User(summaryPrompt),
    ];

    const response = await model.sendRequest(summaryMessages, {}, token);
    let summary = '';
    for await (const part of response.stream) {
      if (part instanceof vscode.LanguageModelTextPart) {
        summary += part.value;
      }
    }

    log(`[compaction] Summary generated: ${summary.length} chars`);
    return summary.trim();
  } catch (err) {
    log(
      `[compaction] Summarization failed: ${err instanceof Error ? err.message : String(err)}. ` +
      `Falling back to simple truncation.`
    );
    return buildFallbackSummary(earlyMessages);
  }
}

/**
 * Fallback when LLM summarization isn't available: extract the first user
 * message (likely the original task description) and a brief outline.
 */
function buildFallbackSummary(
  messages: readonly vscode.LanguageModelChatMessage[]
): string {
  const parts: string[] = ['[Conversation summary - auto-generated fallback]'];

  // Find the first substantive user message
  for (const msg of messages) {
    if (msg.role === vscode.LanguageModelChatMessageRole.User) {
      const text = msg.content
        .filter((p) => p instanceof vscode.LanguageModelTextPart)
        .map((p) => (p as vscode.LanguageModelTextPart).value)
        .join('');
      if (text.length > 50) {
        parts.push(`Original request: ${text.slice(0, 2000)}`);
        break;
      }
    }
  }

  // Count tool calls for context
  let toolCallCount = 0;
  for (const msg of messages) {
    toolCallCount += msg.content.filter(
      (p) => p instanceof vscode.LanguageModelToolCallPart
    ).length;
  }
  if (toolCallCount > 0) {
    parts.push(`(${toolCallCount} tool calls were made in the summarized portion)`);
  }

  parts.push(`(${messages.length} messages were summarized)`);

  return parts.join('\n\n');
}

// ============================================================================
// Compacted message builder
// ============================================================================

/**
 * Build a compacted message list by replacing early history with a summary.
 *
 * @returns The compacted messages ready to be re-sent, or `null` if compaction
 * is not possible (too few messages to split).
 */
export async function compactMessages(
  messages: readonly vscode.LanguageModelChatMessage[],
  modelId: string,
  token: vscode.CancellationToken,
  log: (msg: string) => void,
  keepRounds: number = 3,
): Promise<vscode.LanguageModelChatMessage[] | null> {
  const { early, recent } = splitForCompaction(messages, keepRounds);

  if (early.length === 0) {
    log('[compaction] Cannot compact — too few messages to split');
    return null;
  }

  log(
    `[compaction] Split: ${early.length} early + ${recent.length} recent ` +
    `(keeping last ${keepRounds} tool-call rounds)`
  );

  const summary = await summarizeHistory(early, modelId, token, log);

  const compactedMessages = [
    vscode.LanguageModelChatMessage.User(
      `<conversation-summary>\n${summary}\n</conversation-summary>`
    ),
    ...recent,
  ];

  log(
    `[compaction] Compacted: ${messages.length} → ${compactedMessages.length} messages`
  );

  return compactedMessages;
}
