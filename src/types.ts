/**
 * Type definitions for OpenAI-compatible API responses
 */

export interface OpenAIModel {
  id: string;
  object: string;
  created: number;
  owned_by: string;
  /** GitHub Copilot model registry */
  max_input_tokens?: number;
  /** vLLM, LiteLLM */
  max_model_len?: number;
  /** Ollama, LocalAI, LM Studio */
  context_length?: number;
  /** llama.cpp */
  context_window?: number;
}

export interface OpenAIModelsResponse {
  object: string;
  data: OpenAIModel[];
}

/**
 * Wire-format message sent to an OpenAI-compatible chat endpoint.
 *
 * Typed loosely because we pass through a handful of provider-specific
 * variants (content can be string OR an array of text/image parts, tool
 * results appear as `role: 'tool'` messages, etc.). Treat it as the shape
 * that JSON.stringify will be called on.
 */
export type OpenAIMessage = Record<string, unknown>;

export interface OpenAIToolDefinition {
  type: 'function';
  function: {
    name: string;
    description?: string;
    parameters?: unknown;
  };
}

export interface OpenAIChatCompletionRequest {
  model: string;
  messages: OpenAIMessage[];
  temperature?: number;
  max_tokens?: number;
  stream?: boolean;
  top_p?: number;
  frequency_penalty?: number;
  presence_penalty?: number;
  tools?: OpenAIToolDefinition[];
  tool_choice?: 'auto' | 'required' | 'none';
  parallel_tool_calls?: boolean;
  [key: string]: unknown;
}

export interface OpenAIChatCompletionChunk {
  id: string;
  object: string;
  created: number;
  model: string;
  choices: Array<{
    index: number;
    delta: {
      role?: string;
      content?: string;
      tool_calls?: Array<{
        id: string;
        type: 'function';
        function: {
          name: string;
          arguments: string;
        };
      }>;
    };
    finish_reason: string | null;
  }>;
  /**
   * Present only on the trailing chunk when the request asked for
   * `stream_options.include_usage = true`. OpenAI omits `choices` (or sends
   * an empty array) on that chunk.
   */
  usage?: OpenAIUsage;
}

/**
 * OpenAI-compatible token usage stats. `prompt_tokens_details.cached_tokens`
 * is supported by OpenAI and a growing set of compatible servers; absent
 * elsewhere, we default to 0 when surfacing to VS Code.
 */
export interface OpenAIUsage {
  prompt_tokens: number;
  completion_tokens: number;
  total_tokens: number;
  prompt_tokens_details?: {
    cached_tokens?: number;
  };
}

export interface OpenAIChatCompletionResponse {
  id: string;
  object: string;
  created: number;
  model: string;
  choices: Array<{
    index: number;
    message: {
      role: string;
      content: string;
    };
    finish_reason: string;
  }>;
  usage: OpenAIUsage;
}

export interface GatewayConfig {
  serverUrl: string;
  apiKey?: string;
  requestTimeout: number;
  defaultMaxTokens: number;
  defaultMaxOutputTokens: number;
  enableImageInput: boolean;
  enableToolCalling: boolean;
  parallelToolCalling: boolean;
  agentTemperature: number;
  verboseLogging: boolean;
  customHeaders: Record<string, string>;
  extraModelOptions: Record<string, unknown>;
  perModelSettings: Record<string, Record<string, unknown>>;
  stripFakeToolCallText: boolean;
  retryFakeToolCalls: boolean;
  /**
   * When true (default), route requests for Claude models to the upstream's
   * `/v1/messages` (Anthropic Messages API) instead of `/v1/chat/completions`.
   * This eliminates the lossy OpenAI→Anthropic translation on the upstream
   * side and resolves tool-call self-poisoning, tool-pairing errors, and
   * `<invoke>` XML issues. Non-Claude models always use the OpenAI path.
   *
   * Can be overridden per model via `perModelSettings.<id>.transport`.
   */
  useAnthropicNative: boolean;
  /**
   * When true, bypass all OpenAI / Anthropic format conversion and send
   * `LanguageModelChatMessage[]` directly to the upstream Gateway via
   * `POST /lm/chat`. The upstream is responsible for all model-specific
   * format translation. Responses come back as a simple SSE stream with
   * typed events (`text`, `tool_call`, `thinking`, `done`, `error`).
   *
   * Can be overridden per model via `perModelSettings.<id>.transport`
   * set to `'lm-passthrough'`.
   */
  useLmPassthrough: boolean;
}
