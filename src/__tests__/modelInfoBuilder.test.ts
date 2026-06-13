import { test, describe } from 'node:test';
import assert from 'node:assert/strict';
import {
  PROVIDER_DETAIL_LABEL,
  PROVIDER_MULTIPLIER_NUMERIC,
  buildModelInfo,
} from '../modelInfoBuilder';
import { TOKEN_CONSTANTS } from '../tokenBudget';
import { OpenAIModel } from '../types';

function baseModel(overrides: Partial<OpenAIModel> = {}): OpenAIModel {
  return {
    id: 'qwen/Qwen3-8B',
    object: 'model',
    created: 0,
    owned_by: 'vllm',
    ...overrides,
  };
}

describe('buildModelInfo first-party look-and-feel fields', () => {
  test('sets detail to the provider label so the picker groups models', () => {
    const { info } = buildModelInfo({
      model: baseModel(),
      defaultMaxTokens: 8192,
      defaultMaxOutputTokens: 2048,
      capabilities: {},
    });
    assert.equal(info.detail, PROVIDER_DETAIL_LABEL);
    assert.equal(info.detail, 'LLM Gateway');
  });

  test('sets multiplierNumeric to 0 so BYOK models do not appear premium', () => {
    const { info } = buildModelInfo({
      model: baseModel(),
      defaultMaxTokens: 8192,
      defaultMaxOutputTokens: 2048,
      capabilities: {},
    });
    assert.equal(info.multiplierNumeric, 0);
    assert.equal(info.multiplierNumeric, PROVIDER_MULTIPLIER_NUMERIC);
  });

  test('marks the model user-selectable for the chat picker (1.120 requirement)', () => {
    const { info } = buildModelInfo({
      model: baseModel(),
      defaultMaxTokens: 8192,
      defaultMaxOutputTokens: 2048,
      capabilities: {},
    });
    assert.equal(info.isUserSelectable, true);
  });
});

describe('buildModelInfo id-derived fields', () => {
  test('uses the friendly (post-slash) name', () => {
    const { info } = buildModelInfo({
      model: baseModel({ id: 'meta-llama/Llama-3.1-8B-Instruct' }),
      defaultMaxTokens: 8192,
      defaultMaxOutputTokens: 2048,
      capabilities: {},
    });
    assert.equal(info.name, 'Llama-3.1-8B-Instruct');
    assert.equal(info.version, 'Llama-3.1-8B-Instruct');
    assert.equal(info.id, 'meta-llama/Llama-3.1-8B-Instruct');
  });

  test('infers a known family when the id matches', () => {
    const { info } = buildModelInfo({
      model: baseModel({ id: 'mistralai/Mistral-7B' }),
      defaultMaxTokens: 8192,
      defaultMaxOutputTokens: 2048,
      capabilities: {},
    });
    assert.equal(info.family, 'mistral');
  });

  test('falls back to the llm-gateway family for unknown ids', () => {
    const { info } = buildModelInfo({
      model: baseModel({ id: 'unknown-org/unknown-model' }),
      defaultMaxTokens: 8192,
      defaultMaxOutputTokens: 2048,
      capabilities: {},
    });
    assert.equal(info.family, 'llm-gateway');
  });
});

describe('buildModelInfo context resolution', () => {
  test('treats max_input_tokens as the usable prompt budget (passed through, not reduced)', () => {
    const { totalContext, info, hasServerReportedContext } = buildModelInfo({
      model: baseModel({
        max_input_tokens: 936000,
      }),
      defaultMaxTokens: 9999,
      defaultMaxOutputTokens: 2048,
      capabilities: {},
    });
    // The gateway's max_input_tokens is already a usable prompt budget
    // (upstream max_prompt_tokens), so it is surfaced as-is.
    assert.equal(info.maxInputTokens, 936000);
    // No family-table entry and no full-window field, so totalContext =
    // prompt budget + output budget.
    assert.equal(totalContext, 936000 + info.maxOutputTokens);
    assert.equal(hasServerReportedContext, true);
  });

  test('uses max_model_len as a full window when no prompt budget is reported', () => {
    const { totalContext, info, hasServerReportedContext } = buildModelInfo({
      model: baseModel({
        max_model_len: 131072,
        context_length: 8192,
        context_window: 4096,
      }),
      defaultMaxTokens: 9999,
      defaultMaxOutputTokens: 2048,
      capabilities: {},
    });
    assert.equal(totalContext, 131072);
    assert.equal(info.maxInputTokens, 131072);
    assert.equal(hasServerReportedContext, true);
  });

  test('falls back to context_length when max_model_len is absent', () => {
    const { totalContext, hasServerReportedContext } = buildModelInfo({
      model: baseModel({ context_length: 8192 }),
      defaultMaxTokens: 9999,
      defaultMaxOutputTokens: 2048,
      capabilities: {},
    });
    assert.equal(totalContext, 8192);
    assert.equal(hasServerReportedContext, true);
  });

  test('falls back to context_window when max_model_len and context_length are absent', () => {
    const { totalContext, hasServerReportedContext } = buildModelInfo({
      model: baseModel({ context_window: 4096 }),
      defaultMaxTokens: 9999,
      defaultMaxOutputTokens: 2048,
      capabilities: {},
    });
    assert.equal(totalContext, 4096);
    assert.equal(hasServerReportedContext, true);
  });

  test('falls back to defaultMaxTokens when the server reports no context size', () => {
    const { totalContext, hasServerReportedContext } = buildModelInfo({
      model: baseModel(),
      defaultMaxTokens: 32768,
      defaultMaxOutputTokens: 2048,
      capabilities: {},
    });
    assert.equal(totalContext, 32768);
    assert.equal(hasServerReportedContext, false);
  });
});

describe('buildModelInfo output token math', () => {
  test('uses the configured default when the family is unknown', () => {
    const { info } = buildModelInfo({
      model: baseModel({ max_model_len: 131072 }),
      defaultMaxTokens: 32768,
      defaultMaxOutputTokens: 2048,
      capabilities: {},
    });
    assert.equal(info.maxOutputTokens, 2048);
  });

  test('uses the version-table maxOutputTokens when the version is known', () => {
    const { info } = buildModelInfo({
      model: baseModel({ id: 'gpt-5.5', version: 'gpt-5.5' }),
      defaultMaxTokens: 32768,
      defaultMaxOutputTokens: 2048,
      capabilities: {},
    });
    // gpt-5.5 advertises 128000 output tokens upstream.
    assert.equal(info.maxOutputTokens, 128000);
  });

  test('never drops below MIN_OUTPUT_TOKENS', () => {
    const { info } = buildModelInfo({
      model: baseModel({ max_model_len: TOKEN_CONSTANTS.MIN_OUTPUT_TOKENS }),
      defaultMaxTokens: 32768,
      defaultMaxOutputTokens: 0,
      capabilities: {},
    });
    assert.equal(info.maxOutputTokens, TOKEN_CONSTANTS.MIN_OUTPUT_TOKENS);
  });
});

describe('buildModelInfo description and tooltip', () => {
  test('includes description when describeModel returns content', () => {
    const { info } = buildModelInfo({
      model: baseModel({ max_model_len: 32768, owned_by: 'vllm' }),
      defaultMaxTokens: 8192,
      defaultMaxOutputTokens: 2048,
      capabilities: {},
    });
    assert.ok(info.description, 'expected description to be set');
    assert.ok(info.description!.includes('ctx'));
    assert.equal(info.tooltip, `qwen/Qwen3-8B — ${info.description}`);
  });

  test('omits description when describeModel returns an empty string', () => {
    const { info } = buildModelInfo({
      // No context fields + filtered-out owned_by leaves describeModel empty.
      model: baseModel({ owned_by: 'organization-owner' }),
      defaultMaxTokens: 8192,
      defaultMaxOutputTokens: 2048,
      capabilities: {},
    });
    assert.equal(info.description, undefined);
    assert.equal(info.tooltip, 'qwen/Qwen3-8B');
  });
});

describe('buildModelInfo capabilities resolution', () => {
  test('falls back to caller-provided defaults for unknown models (no guessing)', () => {
    const { info } = buildModelInfo({
      // An unknown id isn't in the capability table; with no inference, the
      // caller-provided config defaults are used verbatim.
      model: baseModel({ id: 'some-unknown-custom-model' }),
      defaultMaxTokens: 8192,
      defaultMaxOutputTokens: 2048,
      capabilities: { imageInput: true, toolCalling: 16 },
    });
    assert.deepEqual(info.capabilities, { toolCalling: 16, imageInput: true });
  });

  test('leaves capabilities undefined for unknown models when no defaults are given', () => {
    const { info } = buildModelInfo({
      model: baseModel({ id: 'some-unknown-custom-model' }),
      defaultMaxTokens: 8192,
      defaultMaxOutputTokens: 2048,
      capabilities: {},
    });
    assert.deepEqual(info.capabilities, { toolCalling: undefined, imageInput: undefined });
  });

  test('resolves capabilities from the version table, ignoring caller defaults', () => {
    const { info } = buildModelInfo({
      model: baseModel({ id: 'claude-opus-4.8', version: 'claude-opus-4.8' }),
      defaultMaxTokens: 8192,
      defaultMaxOutputTokens: 2048,
      capabilities: { imageInput: false, toolCalling: false },
    });
    // Version table says claude-opus-4.8 supports tools + vision.
    assert.deepEqual(info.capabilities, { toolCalling: true, imageInput: true });
  });
});

describe('buildModelInfo Thinking Effort configurationSchema', () => {
  test('adds a reasoningEffort navigation property for reasoning models', () => {
    const { info } = buildModelInfo({
      model: baseModel({ id: 'gpt-5.5', version: 'gpt-5.5' }),
      defaultMaxTokens: 8192,
      defaultMaxOutputTokens: 2048,
      capabilities: {},
    });
    const prop = info.configurationSchema?.properties?.reasoningEffort;
    assert.ok(prop, 'expected a reasoningEffort schema property');
    assert.equal(prop!.group, 'navigation');
    assert.deepEqual(prop!.enum, ['none', 'low', 'medium', 'high', 'xhigh']);
    assert.equal(prop!.default, 'high');
  });

  test('omits configurationSchema for non-reasoning models', () => {
    const { info } = buildModelInfo({
      model: baseModel({ id: 'gpt-4o', version: 'gpt-4o-2024-11-20' }),
      defaultMaxTokens: 8192,
      defaultMaxOutputTokens: 2048,
      capabilities: {},
    });
    assert.equal(info.configurationSchema, undefined);
  });

  test('prefers the server-provided name/family/version over id-derived values', () => {
    const { info, totalContext } = buildModelInfo({
      model: baseModel({
        id: 'claude-opus-4.8',
        name: 'Claude Opus 4.8',
        family: 'claude-opus-4.8',
        version: 'claude-opus-4.8',
        max_input_tokens: 935793,
      }),
      defaultMaxTokens: 8192,
      defaultMaxOutputTokens: 2048,
      capabilities: {},
    });
    assert.equal(info.name, 'Claude Opus 4.8');
    assert.equal(info.family, 'claude-opus-4.8');
    assert.equal(info.version, 'claude-opus-4.8');
    assert.equal(info.maxInputTokens, 935793);
    assert.equal(info.maxOutputTokens, 64000);
    // Full window comes from the family table (1M), independent of the
    // server-reported prompt budget.
    assert.equal(totalContext, 1_000_000);
  });
});
