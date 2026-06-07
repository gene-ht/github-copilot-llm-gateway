# 下游 Gateway：LanguageModelChatMessage 透传方案

## 背景

### 问题

Copilot 扩展对第三方 Provider（`vendor !== 'copilot'`）不发 HTTP 请求，只通过 VS Code `lm` API 传递 `LanguageModelChatMessage[]`。下游 Gateway 需要将这些数据发送给上游 Gateway 处理。

### 解决方案

下游 Gateway 将 `LanguageModelChatMessage[]` 直接序列化为 JSON，通过 `POST /lm/chat` 发给上游。上游反序列化后交给 Copilot vendor 模型处理，响应以 SSE 流返回。

**不需要关心 Anthropic/OpenAI 数据格式** — 格式转换完全由上游的 Copilot Provider 内部完成。所有模型（Claude/GPT/Gemini）统一走同一条链路。

---

## 交互链路

```
Copilot ToolCallingLoop
  → vendor !== 'copilot'
  → ExtensionContributedChatEndpoint
  → vscode.lm.sendRequest(messages, options, token)
  │
  ▼
下游 Gateway（你的 LanguageModelChatProvider）
  → 序列化 LanguageModelChatMessage[] 为 JSON
  → 序列化 tools 为 JSON
  → POST /lm/chat → 上游 Gateway
  → 解析上游返回的 SSE 流
  → progress.report(TextPart / ToolCallPart / ThinkingPart)
  │
  ▼
Copilot 消费响应 → 执行工具 → 继续对话循环
```

---

## 请求协议

### 请求

```
POST /lm/chat HTTP/1.1
Content-Type: application/json
x-api-key: <认证密钥>
```

```json
{
  "model": "claude-opus-4.8",
  "messages": [
    {
      "role": 1,
      "content": [
        { "type": "text", "value": "帮我搜索文件" }
      ]
    },
    {
      "role": 2,
      "content": [
        { "type": "text", "value": "我来搜索" },
        { "type": "tool_call", "callId": "call_123", "name": "search", "input": {"query": "*.ts"} }
      ]
    },
    {
      "role": 1,
      "content": [
        { "type": "tool_result", "callId": "call_123", "content": [
          { "type": "text", "value": "找到 5 个文件..." }
        ]}
      ]
    }
  ],
  "tools": [
    { "name": "search", "description": "搜索文件", "inputSchema": {"type":"object",...} }
  ]
}
```

其中 `role`: 1 = User, 2 = Assistant

`content` 中的 part 类型：
- `text` — 文本内容
- `tool_call` — 工具调用（Assistant 侧）
- `tool_result` — 工具结果（User 侧）
- `thinking` — 思考过程
- `data` — 二进制数据（图片等），base64 编码

### 响应（SSE）

```
data: {"type":"text","value":"我来帮你"}
data: {"type":"text","value":"搜索文件"}
data: {"type":"tool_call","callId":"call_456","name":"read_file","input":{"path":"src/app.ts"}}
data: {"type":"thinking","value":"思考中...","id":"think_1","metadata":{...}}
data: {"type":"done"}
```

错误时：
```
data: {"type":"error","message":"Model not found"}
```

---

## 下游处理代码

### 序列化

```typescript
function serializeMessages(messages: vscode.LanguageModelChatMessage[]): any[] {
    return messages.map(msg => ({
        role: msg.role,
        content: msg.content.map(part => {
            if (part instanceof vscode.LanguageModelTextPart) {
                return { type: 'text', value: part.value };
            }
            if (part instanceof vscode.LanguageModelToolCallPart) {
                return { type: 'tool_call', callId: part.callId, name: part.name, input: part.input };
            }
            if (part instanceof vscode.LanguageModelToolResultPart) {
                return {
                    type: 'tool_result',
                    callId: part.callId,
                    content: part.content.map(c =>
                        c instanceof vscode.LanguageModelTextPart
                            ? { type: 'text', value: c.value }
                            : { type: 'data', mimeType: (c as any).mimeType, data: Buffer.from((c as any).data).toString('base64') }
                    )
                };
            }
            if (part instanceof vscode.LanguageModelThinkingPart) {
                return { type: 'thinking', value: part.value, id: part.id, metadata: (part as any).metadata };
            }
            if (part instanceof vscode.LanguageModelDataPart) {
                return { type: 'data', mimeType: part.mimeType, data: Buffer.from(part.data).toString('base64') };
            }
        }).filter(Boolean)
    }));
}

function serializeTools(tools?: vscode.LanguageModelChatTool[]): any[] | undefined {
    if (!tools?.length) return undefined;
    return tools.map(t => ({ name: t.name, description: t.description, inputSchema: t.inputSchema }));
}
```

### Provider 入口

```typescript
async provideLanguageModelChatResponse(
    model: vscode.LanguageModelChatInformation,
    messages: vscode.LanguageModelChatMessage[],
    options: vscode.ProvideLanguageModelChatResponseOptions,
    progress: vscode.Progress<vscode.LanguageModelTextPart | vscode.LanguageModelToolCallPart | vscode.LanguageModelThinkingPart>,
    token: vscode.CancellationToken,
): Promise<void> {

    const body = JSON.stringify({
        model: model.id,
        messages: serializeMessages(messages),
        tools: serializeTools(options.tools),
    });

    const response = await fetch(`${this.upstreamUrl}/lm/chat`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', 'x-api-key': this.apiKey },
        body,
    });

    if (!response.ok) {
        throw new Error(`Upstream error: ${response.status} ${await response.text()}`);
    }

    const reader = response.body!.getReader();
    const decoder = new TextDecoder();
    let buffer = '';

    while (true) {
        const { done, value } = await reader.read();
        if (done || token.isCancellationRequested) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split('\n');
        buffer = lines.pop() || '';

        for (const line of lines) {
            if (!line.startsWith('data: ')) continue;
            const event = JSON.parse(line.slice(6));

            switch (event.type) {
                case 'text':
                    progress.report(new vscode.LanguageModelTextPart(event.value));
                    break;
                case 'tool_call':
                    progress.report(new vscode.LanguageModelToolCallPart(event.callId, event.name, event.input));
                    break;
                case 'thinking': {
                    const tp = new vscode.LanguageModelThinkingPart(event.value);
                    if (event.metadata) (tp as any).metadata = event.metadata;
                    progress.report(tp);
                    break;
                }
                case 'error':
                    throw new Error(event.message);
                case 'done':
                    return;
            }
        }
    }
}
```
