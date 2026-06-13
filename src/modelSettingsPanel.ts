/**
 * QuickPick-based per-model settings editor.
 *
 * Flow: Pick a model → Pick a parameter → Enter/select value → loop.
 * Changes are written to `github.copilot.llm-gateway.perModelSettings`
 * in workspace settings immediately.
 */

import * as vscode from 'vscode';
import { ModelSummary } from './statusSnapshot';

const CONFIG_SECTION = 'github.copilot.llm-gateway';
const CONFIG_KEY = 'perModelSettings';

/** Well-known parameters with dropdown or typed input. */
const KNOWN_PARAMS: Array<{
  key: string;
  label: string;
  description: string;
  kind: 'dropdown' | 'number' | 'string';
  options?: string[];
}> = [
  {
    key: 'temperature',
    label: 'temperature',
    description: 'Sampling temperature (0-2)',
    kind: 'number',
  },
  {
    key: 'max_tokens',
    label: 'max_tokens',
    description: 'Override maximum output tokens',
    kind: 'number',
  },
];

let outputChannel: vscode.OutputChannel | undefined;
function log(msg: string): void {
  if (!outputChannel) {
    outputChannel = vscode.window.createOutputChannel('GitHub Copilot LLM Gateway');
  }
  outputChannel.appendLine(`[ModelSettings] ${msg}`);
}

function readSettings(): Record<string, Record<string, unknown>> {
  const config = vscode.workspace.getConfiguration(CONFIG_SECTION);
  const value = config.get<Record<string, Record<string, unknown>>>(CONFIG_KEY) ?? {};
  log(`readSettings: ${JSON.stringify(value)}`);
  // VS Code returns a deeply-frozen object (wrapped in a Proxy). Mutating it
  // directly throws "'isExtensible' on proxy: trap result does not reflect
  // extensibility of proxy target". Clone into a plain, mutable object so
  // callers can freely add/delete keys before writing back.
  return cloneMutable(value);
}

/**
 * Deep clone an arbitrary JSON-compatible value into a plain, mutable object
 * tree. Used to escape VS Code's frozen configuration proxy.
 */
function cloneMutable<T>(value: T): T {
  if (value === null || typeof value !== 'object') {
    return value;
  }
  if (Array.isArray(value)) {
    return value.map((v) => cloneMutable(v)) as unknown as T;
  }
  const out: Record<string, unknown> = {};
  for (const k of Object.keys(value as Record<string, unknown>)) {
    out[k] = cloneMutable((value as Record<string, unknown>)[k]);
  }
  return out as unknown as T;
}

async function writeSettings(settings: Record<string, Record<string, unknown>>): Promise<void> {
  log(`writeSettings: ${JSON.stringify(settings)}`);
  const config = vscode.workspace.getConfiguration(CONFIG_SECTION);
  try {
    await config.update(CONFIG_KEY, settings, vscode.ConfigurationTarget.Global);
    log(`writeSettings: SUCCESS`);
  } catch (err) {
    log(`writeSettings: FAILED — ${err instanceof Error ? err.message : String(err)}`);
    throw err;
  }
}

function formatCurrentValue(value: unknown): string {
  if (value === undefined || value === null || value === '') {
    return '(default)';
  }
  return String(value);
}

interface ModelPickItem extends vscode.QuickPickItem {
  modelId: string;
}

interface ParamPickItem extends vscode.QuickPickItem {
  action: 'set' | 'clear' | 'clearAll' | 'custom' | 'done';
  paramKey?: string;
}

/**
 * Main entry point — opens the multi-step QuickPick flow.
 */
export async function openModelSettingsQuickPick(
  models: readonly ModelSummary[]
): Promise<void> {
  if (models.length === 0) {
    vscode.window.showWarningMessage(
      'LLM Gateway: No models available. Connect to a server first.'
    );
    return;
  }

  // Step 1: Pick a model
  const allSettings = readSettings();

  const modelItems: ModelPickItem[] = models.map((m) => {
    const overrides = allSettings[m.id];
    const overrideCount = overrides ? Object.keys(overrides).length : 0;
    const suffix = overrideCount > 0 ? ` (${overrideCount} override${overrideCount > 1 ? 's' : ''})` : '';
    return {
      label: m.name,
      description: `${m.id}${suffix}`,
      detail: m.contextLabel || undefined,
      modelId: m.id,
    };
  });

  const modelPick = await vscode.window.showQuickPick(modelItems, {
    title: 'LLM Gateway — Model Settings',
    placeHolder: 'Select a model to configure',
    matchOnDescription: true,
  });

  if (!modelPick) { return; }

  // Step 2: Parameter loop for the selected model
  await editModelParams(modelPick.modelId, modelPick.label);
}

async function editModelParams(modelId: string, modelName: string): Promise<void> {
  while (true) {
    const allSettings = readSettings();
    const modelSettings = allSettings[modelId] ?? {};

    const items: ParamPickItem[] = [];

    // Done
    items.push({
      label: '$(check) Done',
      description: 'Return to model list',
      action: 'done',
    });

    // Known params with current values
    for (const p of KNOWN_PARAMS) {
      const current = formatCurrentValue(modelSettings[p.key]);
      items.push({
        label: `$(symbol-parameter) ${p.label}`,
        description: current,
        detail: p.description,
        action: 'set',
        paramKey: p.key,
      });
    }

    // Custom params already set
    const customKeys = Object.keys(modelSettings).filter(
      (k) => !KNOWN_PARAMS.some((p) => p.key === k)
    );
    for (const k of customKeys) {
      items.push({
        label: `$(symbol-key) ${k}`,
        description: formatCurrentValue(modelSettings[k]),
        detail: 'Custom parameter',
        action: 'set',
        paramKey: k,
      });
    }

    // Add custom
    items.push({
      label: '$(add) Add custom parameter...',
      description: '',
      action: 'custom',
    });

    // Clear all
    if (Object.keys(modelSettings).length > 0) {
      items.push({
        label: '$(trash) Clear all overrides for this model',
        description: '',
        action: 'clearAll',
      });
    }

    const pick = await vscode.window.showQuickPick(items, {
      title: `Model Settings — ${modelName}`,
      placeHolder: 'Select a parameter to edit',
    });

    if (!pick || pick.action === 'done') { return; }

    if (pick.action === 'clearAll') {
      const confirm = await vscode.window.showWarningMessage(
        `Clear all overrides for ${modelName}?`,
        { modal: true },
        'Clear'
      );
      if (confirm === 'Clear') {
        const s = readSettings();
        delete s[modelId];
        await writeSettings(s);
        vscode.window.showInformationMessage(`Cleared all overrides for ${modelName}.`);
      }
      continue;
    }

    if (pick.action === 'custom') {
      const key = await vscode.window.showInputBox({
        title: `${modelName} — Add Custom Parameter`,
        prompt: 'Parameter name (e.g. top_k, repetition_penalty, seed)',
        placeHolder: 'parameter_name',
      });
      if (!key) { continue; }

      const value = await vscode.window.showInputBox({
        title: `${modelName} — ${key}`,
        prompt: `Enter value for ${key}`,
        placeHolder: 'value',
      });
      if (value === undefined) { continue; }

      const s = readSettings();
      if (!s[modelId]) { s[modelId] = {}; }
      // Try to parse as number/boolean
      s[modelId][key] = parseValue(value);
      await writeSettings(s);
      continue;
    }

    if (pick.action === 'set' && pick.paramKey) {
      await editSingleParam(modelId, modelName, pick.paramKey);
    }
  }
}

async function editSingleParam(modelId: string, modelName: string, paramKey: string): Promise<void> {
  log(`editSingleParam: modelId=${modelId} paramKey=${paramKey}`);
  const known = KNOWN_PARAMS.find((p) => p.key === paramKey);
  log(`editSingleParam: knownParam=${known ? `kind=${known.kind}` : '(unknown — free-form)'}`);
  const allSettings = readSettings();
  const currentValue = allSettings[modelId]?.[paramKey];
  log(`editSingleParam: currentValue=${JSON.stringify(currentValue)}`);

  if (known?.kind === 'dropdown' && known.options) {
    // Dropdown with clear option
    const options = [
      { label: '$(close) Clear (use default)', value: undefined as string | undefined },
      ...known.options.map((o) => ({
        label: currentValue === o ? `$(check) ${o}` : `    ${o}`,
        value: o as string | undefined,
      })),
    ];

    const pick = await vscode.window.showQuickPick(options, {
      title: `${modelName} — ${paramKey}`,
      placeHolder: `Current: ${formatCurrentValue(currentValue)}`,
    });

    if (pick === undefined) { return; }

    const s = readSettings();
    if (pick.value === undefined) {
      if (s[modelId]) {
        delete s[modelId][paramKey];
        if (Object.keys(s[modelId]).length === 0) { delete s[modelId]; }
      }
    } else {
      if (!s[modelId]) { s[modelId] = {}; }
      s[modelId][paramKey] = pick.value;
    }
    await writeSettings(s);
    return;
  }

  // Number or string: jump straight to input box. Empty input = clear.
  const promptHint = currentValue !== undefined
    ? `Current: ${formatCurrentValue(currentValue)} — leave empty to clear`
    : known?.description ?? 'Enter value';
  const value = await vscode.window.showInputBox({
    title: `${modelName} — ${paramKey}`,
    prompt: promptHint,
    value: currentValue !== undefined ? String(currentValue) : '',
    placeHolder: known?.kind === 'number' ? '0' : 'value (empty = clear)',
  });

  if (value === undefined) { log(`editSingleParam: cancelled at input box`); return; }
  log(`editSingleParam: raw input value=${JSON.stringify(value)}`);

  const s = readSettings();
  if (value === '') {
    // Empty input clears the override
    if (s[modelId]) {
      delete s[modelId][paramKey];
      if (Object.keys(s[modelId]).length === 0) { delete s[modelId]; }
    }
    log(`editSingleParam: cleared (empty input)`);
    await writeSettings(s);
    return;
  }

  const parsed = parseValue(value);
  log(`editSingleParam: parsed=${JSON.stringify(parsed)} (typeof=${typeof parsed})`);
  if (!s[modelId]) { s[modelId] = {}; }
  s[modelId][paramKey] = parsed;
  await writeSettings(s);
}

function parseValue(raw: string): unknown {
  if (raw === '') { return ''; }
  if (raw === 'true') { return true; }
  if (raw === 'false') { return false; }
  const n = Number(raw);
  if (!isNaN(n) && raw.trim() !== '') { return n; }
  return raw;
}
