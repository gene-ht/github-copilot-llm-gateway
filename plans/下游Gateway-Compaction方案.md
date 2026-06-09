# 下游 Gateway：Context Too Long Compaction 方案

## 背景

当上游返回 `prompt is too long: 1005104 tokens > 1000000 maximum` 时，下游 Gateway 需要在内部做 compaction（对话摘要压缩），然后用压缩后的消息重试请求。

Copilot 原生的 `renderWithSummarization` 做的事：
1. 对对话历史生成摘要
2. 用摘要替换早期对话轮次
3. 重新渲染 prompt 后发送

下游 Gateway 要在 `provideLanguageModelChatResponse()` 内部实现同样的逻辑。

---

## 核心思路

```
provideLanguageModelChatResponse(model, messages, options, progress, token)
  │
  ▼
  序列化 messages → POST /lm/chat → 上游
  │
  ├─ 成功 → 正常解析 SSE → progress.report()
  │
  └─ 失败: "prompt is too long" 或 "context too long"
       │
       ▼
       执行 compaction:
       1. 将 messages 分为 [早期历史] + [最近 N 轮]
       2. 用当前模型对 [早期历史] 生成摘要
       3. 构建压缩后的 messages = [摘要消息] + [最近 N 轮]
       4. 用压缩后的 messages 重新发送 POST /lm/chat
```

---

## 实现方案

### 1. Compaction 函数

```typescript
import * as vscode from 'vscode';

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
 * Summarize early conversation history using the same model as the current request.
 * Returns a single text summary that replaces the early messages.
 *
 * @param model The current request's model — reuse it for summarization so the
 *              summary style matches the model's capabilities.
 */
async function summarizeHistory(
  earlyMessages: vscode.LanguageModelChatMessage[],
  model: vscode.LanguageModelChat,
  token: vscode.CancellationToken,
  log: (msg: string) => void,
): Promise<string> {

  // 构建摘要 prompt
  const conversationText = earlyMessages.map(msg => {
    const role = msg.role === vscode.LanguageModelChatMessageRole.Assistant ? 'Assistant' : 'User';
    const text = msg.content
      .map(part => {
        if (part instanceof vscode.LanguageModelTextPart) return part.value;
        if (part instanceof vscode.LanguageModelToolCallPart) return `[Tool call: ${part.name}]`;
        if (part instanceof vscode.LanguageModelToolResultPart) {
          const resultText = part.content
            .map(c => c instanceof vscode.LanguageModelTextPart ? c.value : '[data]')
            .join('');
          // 截断过长的工具结果
          return `[Tool result: ${resultText.slice(0, 500)}${resultText.length > 500 ? '...' : ''}]`;
        }
        return '';
      })
      .join('');
    return `${role}: ${text}`;
  }).join('\n\n');

  const summaryMessages = [
    vscode.LanguageModelChatMessage.User(
      `${SUMMARIZATION_SYSTEM_PROMPT}\n\n---\n\nConversation to summarize:\n\n${conversationText}\n\n---\n\nProvide a concise summary:`
    ),
  ];

  log(`[compaction] Summarizing ${earlyMessages.length} early messages using ${model.id}...`);

  const response = await model.sendRequest(summaryMessages, {}, token);
  let summary = '';
  for await (const part of response.stream) {
    if (part instanceof vscode.LanguageModelTextPart) {
      summary += part.value;
    }
  }

  log(`[compaction] Summary generated: ${summary.length} chars`);
  return summary.trim();
}
```

### 2. 分割消息：早期历史 vs 最近轮次

```typescript
/**
 * Split messages into [early history to summarize] + [recent rounds to keep].
 *
 * Strategy: keep the last K complete tool-call rounds + the last user message.
 * A "round" is: assistant(tool_calls) + user(tool_results).
 *
 * @param keepRounds Number of recent complete rounds to preserve verbatim
 */
function splitForCompaction(
  messages: vscode.LanguageModelChatMessage[],
  keepRounds: number = 3,
): { early: vscode.LanguageModelChatMessage[]; recent: vscode.LanguageModelChatMessage[] } {
  if (messages.length <= keepRounds * 2 + 2) {
    // Too few messages to compact — keep everything
    return { early: [], recent: [...messages] };
  }

  // Find the split point: walk backward to find `keepRounds` complete
  // assistant+user pairs (tool call rounds).
  let splitIndex = messages.length;
  let roundsFound = 0;

  for (let i = messages.length - 1; i >= 0; i--) {
    const msg = messages[i];
    // A complete round ends with a user message containing tool_result
    const hasToolResult = msg.role === vscode.LanguageModelChatMessageRole.User &&
      msg.content.some(p => p instanceof vscode.LanguageModelToolResultPart);

    if (hasToolResult) {
      roundsFound++;
      if (roundsFound >= keepRounds) {
        // Find the corresponding assistant message before this user message
        // (it should be the previous message with tool_call parts)
        let j = i - 1;
        while (j >= 0 && messages[j].role !== vscode.LanguageModelChatMessageRole.Assistant) {
          j--;
        }
        splitIndex = j >= 0 ? j : i;
        break;
      }
    }
  }

  // Ensure we keep at least the last few messages
  if (splitIndex <= 1) {
    splitIndex = Math.max(1, Math.floor(messages.length / 2));
  }

  return {
    early: messages.slice(0, splitIndex),
    recent: messages.slice(splitIndex),
  };
}
```

### 3. 在 provideLanguageModelChatResponse 中集成

```typescript
async provideLanguageModelChatResponse(
  model: vscode.LanguageModelChatInformation,
  messages: vscode.LanguageModelChatMessage[],
  options: vscode.ProvideLanguageModelChatResponseOptions,
  progress: vscode.Progress<vscode.LanguageModelResponsePart>,
  token: vscode.CancellationToken,
): Promise<void> {

  try {
    // 正常发送
    await streamLmPassthrough({
      model: model.id,
      messages,
      tools: options.tools,
      progress,
      token,
      // ... other params
    });
  } catch (e) {
    if (isContextTooLongError(e)) {
      this.log(`[compaction] Context too long, triggering compaction...`);

      // 1. 分割消息
      const { early, recent } = splitForCompaction(messages);

      if (early.length === 0) {
        // 无法再压缩 — 直接报错
        throw e;
      }

      // 2. 用当前模型生成摘要
      const currentModel = (await vscode.lm.selectChatModels({ id: model.id }))[0];
      if (!currentModel) { throw e; }
      const summary = await summarizeHistory(early, currentModel, token, this.log);

      // 3. 构建压缩后的消息
      const compactedMessages = [
        vscode.LanguageModelChatMessage.User(
          `<conversation-summary>\n${summary}\n</conversation-summary>`
        ),
        ...recent,
      ];

      this.log(`[compaction] Compacted: ${messages.length} → ${compactedMessages.length} messages`);

      // 4. 用压缩后的消息重新发送
      await streamLmPassthrough({
        model: model.id,
        messages: compactedMessages,
        tools: options.tools,
        progress,
        token,
        // ... other params
      });
    } else {
      throw e;
    }
  }
}

function isContextTooLongError(e: unknown): boolean {
  const msg = e instanceof Error ? e.message : String(e);
  return msg.includes('prompt is too long') ||
         msg.includes('context_length_exceeded') ||
         msg.includes('model_context_window_exceeded') ||
         msg.includes('too many tokens');
}
```

---

## 与 Copilot 原生 compaction 的对比

| 维度 | Copilot 原生 | 下游 Gateway |
|---|---|---|
| 触发时机 | prompt-tsx 渲染时 BudgetExceeded | 上游返回 400 错误时 |
| 摘要模型 | `copilot-fast` (gpt-4o-mini) | 当前请求的同款模型 |
| 摘要格式 | `<conversation-summary>` 标签 | 同上 |
| 保留策略 | 最后一轮 tool-call round | 最后 3 轮 |
| 重试 | 重新渲染 prompt-tsx | 重新序列化 + POST |
| 对用户透明 | 是 | 是（用户不感知） |

---

## 注意事项

1. **摘要需要额外的 LLM 调用** — 使用当前模型，会增加延迟和 token 消耗
2. **摘要后 tool_call/tool_result 配对** — 只有 recent 部分保留原生 tool 配对，early 部分被摘要替代
3. **递归 compaction** — 如果压缩后仍然超限，可以进一步减少 `keepRounds`
4. **超时** — summarizeHistory 的 LLM 调用需要在 requestTimeout 内完成
