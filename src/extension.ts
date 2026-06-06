import * as vscode from 'vscode';
import * as fs from 'node:fs';
import * as path from 'node:path';
import { GatewayProvider, RequestStateEvent } from './provider';
import { openModelSettingsQuickPick } from './modelSettingsPanel';
import {
  StatusBarState,
  TokenUsage,
  extractHost,
  renderStatusBar,
} from './statusBarController';
import { StatusSnapshot } from './statusSnapshot';
import { renderStatusTooltipHtml } from './statusTooltip';
import { CopilotProxyServer, CopilotProxyConfig, ProxyUpstreamConfig } from './copilotProxyServer';

const STATUS_BAR_PROBE_DELAY_MS = 1500;
/** How long the "responded" pulse stays in the bar before reverting to idle. */
const RESPONDED_DISPLAY_MS = 10_000;

// ---------- User settings.json helpers ----------
// VS Code's configuration API refuses to write settings that aren't registered
// by any installed extension. `github.copilot.advanced.debug.overrideProxyUrl`
// is registered by the Copilot extension at runtime — if it isn't installed
// (or hasn't activated yet) the write fails. These helpers patch the file
// directly so the proxy can always configure itself.

/**
 * Resolve the user-global `settings.json` path. Matches the platform
 * conventions VS Code itself uses.
 */
function getUserSettingsPath(): string {
  const appData = process.platform === 'win32'
    ? process.env.APPDATA ?? path.join(process.env.USERPROFILE ?? '', 'AppData', 'Roaming')
    : process.platform === 'darwin'
      ? path.join(process.env.HOME ?? '', 'Library', 'Application Support')
      : process.env.XDG_CONFIG_HOME ?? path.join(process.env.HOME ?? '', '.config');
  return path.join(appData, 'Code', 'User', 'settings.json');
}

/**
 * Write a dotted key (e.g. `"github.copilot.advanced.debug.overrideProxyUrl"`)
 * into the user `settings.json`, preserving all other entries.
 */
async function writeUserSetting(key: string, value: unknown): Promise<void> {
  const file = getUserSettingsPath();
  let settings: Record<string, unknown> = {};
  try {
    const raw = await fs.promises.readFile(file, 'utf8');
    settings = JSON.parse(raw);
  } catch { /* file doesn't exist or isn't valid JSON — start fresh */ }
  settings[key] = value;
  await fs.promises.writeFile(file, JSON.stringify(settings, null, 4) + '\n', 'utf8');
}

/**
 * Remove a dotted key from the user `settings.json`.
 */
async function removeUserSetting(key: string): Promise<void> {
  const file = getUserSettingsPath();
  let settings: Record<string, unknown> = {};
  try {
    const raw = await fs.promises.readFile(file, 'utf8');
    settings = JSON.parse(raw);
  } catch { return; /* nothing to remove */ }
  delete settings[key];
  await fs.promises.writeFile(file, JSON.stringify(settings, null, 4) + '\n', 'utf8');
}

/**
 * Drives the LLM Gateway status bar. Pure rendering lives in
 * `statusBarController.ts`; this class only handles the VS Code-side state
 * machine: timers, in-flight counting, mapping events onto state transitions.
 */
class StatusBarManager implements vscode.Disposable {
  private state: StatusBarState;
  private respondedRevertTimer?: NodeJS.Timeout;
  private activeRequestCount = 0;
  private cachedIdle: { host: string; modelIds: readonly string[] } = {
    host: '',
    modelIds: [],
  };

  constructor(
    private readonly item: vscode.StatusBarItem,
    private readonly getServerUrl: () => string,
    private readonly getSnapshot: () => StatusSnapshot
  ) {
    this.state = { kind: 'probing', host: extractHost(this.getServerUrl()) };
    this.render();
  }

  /**
   * Called from outside whenever the provider's snapshot changes (session
   * totals, last request, models, connection state). The tooltip is rebuilt
   * from the snapshot, so any new data shows up the next time the user hovers
   * the status bar — even if the bar's icon state hasn't changed.
   */
  refreshTooltip(): void {
    this.render();
  }

  dispose(): void {
    this.cancelRespondedRevert();
  }

  setIdle(modelIds: readonly string[]): void {
    this.cachedIdle = { host: this.host(), modelIds };
    this.cancelRespondedRevert();
    this.applyIdle();
  }

  setNoModels(): void {
    this.cancelRespondedRevert();
    this.state = { kind: 'noModels', host: this.host() };
    this.render();
  }

  setError(errorMessage: string): void {
    this.cancelRespondedRevert();
    this.state = { kind: 'error', host: this.host(), errorMessage };
    this.render();
  }

  onRequest(event: RequestStateEvent): void {
    switch (event.kind) {
      case 'start':
        this.onRequestStart(event);
        return;
      case 'complete':
        this.onRequestComplete(event);
        return;
      case 'error':
        this.onRequestError(event);
        return;
      default: {
        const _never: never = event;
        throw new Error(`Unexpected request state kind: ${String(_never)}`);
      }
    }
  }

  private onRequestStart(event: Extract<RequestStateEvent, { kind: 'start' }>): void {
    this.cancelRespondedRevert();
    this.activeRequestCount++;
    this.state = {
      kind: 'streaming',
      host: this.host(),
      modelId: event.modelId,
      modelName: event.modelName,
      activeCount: this.activeRequestCount,
    };
    this.render();
  }

  private onRequestComplete(
    event: Extract<RequestStateEvent, { kind: 'complete' }>
  ): void {
    this.activeRequestCount = Math.max(0, this.activeRequestCount - 1);
    if (this.activeRequestCount > 0) {
      // Other requests still streaming — keep the bar in streaming state with
      // an updated count rather than briefly flashing "responded".
      this.state = {
        kind: 'streaming',
        host: this.host(),
        modelId: event.modelId,
        modelName: event.modelName,
        activeCount: this.activeRequestCount,
      };
      this.render();
      return;
    }
    this.state = {
      kind: 'responded',
      host: this.host(),
      modelId: event.modelId,
      modelName: event.modelName,
      ...(event.usage ? { usage: this.toUsage(event.usage) } : {}),
    };
    this.render();
    this.scheduleRespondedRevert();
  }

  private onRequestError(event: Extract<RequestStateEvent, { kind: 'error' }>): void {
    this.activeRequestCount = Math.max(0, this.activeRequestCount - 1);
    this.setError(event.errorMessage);
  }

  private toUsage(usage: TokenUsage): TokenUsage {
    return { prompt: usage.prompt, completion: usage.completion, total: usage.total };
  }

  private applyIdle(): void {
    this.state = {
      kind: 'idle',
      host: this.cachedIdle.host,
      modelCount: this.cachedIdle.modelIds.length,
      modelIds: this.cachedIdle.modelIds,
    };
    this.render();
  }

  private scheduleRespondedRevert(): void {
    this.cancelRespondedRevert();
    this.respondedRevertTimer = setTimeout(() => {
      this.respondedRevertTimer = undefined;
      this.applyIdle();
    }, RESPONDED_DISPLAY_MS);
  }

  private cancelRespondedRevert(): void {
    if (this.respondedRevertTimer) {
      clearTimeout(this.respondedRevertTimer);
      this.respondedRevertTimer = undefined;
    }
  }

  private host(): string {
    return extractHost(this.getServerUrl());
  }

  private render(): void {
    // Bar text stays minimal (vm-active/vm-disconnect + host) — that's the
    // "is the gateway up" signal. All the rich data goes into the hover
    // tooltip, which is the closest stable-API approximation to GHCP's
    // floating popup (`chatStatusItem` is proposed-API-only).
    const { text } = renderStatusBar(this.state);
    this.item.text = text;
    // Tooltip renders as the GHCP-style popup: HTML card with theme icons,
    // section headers, and command-link buttons. MarkdownString runs the value
    // through VS Code's hover renderer, which is the closest stable-API path
    // to a click-triggered floating popup (`chatStatusItem` is proposed-only).
    const tooltipHtml = renderStatusTooltipHtml(this.getSnapshot());
    const md = new vscode.MarkdownString(tooltipHtml);
    md.isTrusted = true;
    md.supportThemeIcons = true;
    md.supportHtml = true;
    this.item.tooltip = md;
  }
}

/**
 * Extension activation. Async so we can pull the API key + custom headers
 * out of SecretStorage (and migrate legacy plain-text settings, issue #28)
 * before registering the provider — otherwise the first model fetch races
 * the secret load and is sent unauthenticated.
 */
export async function activate(context: vscode.ExtensionContext): Promise<void> {
  const provider = new GatewayProvider(context);
  await provider.loadSecrets();

  const disposable = vscode.lm.registerLanguageModelChatProvider(
    'copilot-llm-gateway',
    provider
  );

  context.subscriptions.push(disposable);

  // Status bar entry so users can see connection state at a glance and
  // quickly refresh the model list. Without this, failed model fetches were
  // invisible unless users happened to open the model picker. The visible
  // label is context-aware (host when idle, model name during streaming,
  // model + token count after) — see statusBarController.ts.
  const statusBar = vscode.window.createStatusBarItem(
    vscode.StatusBarAlignment.Right,
    100
  );
  statusBar.name = 'LLM Gateway';
  // Click refreshes the gateway. The rich GHCP-style popup is the hover
  // tooltip — it's the closest stable-API approximation to a floating
  // status-bar popup. Clicking is wired to a useful action so the bar
  // isn't dead.
  statusBar.command = 'github.copilot.llm-gateway.refreshModels';
  statusBar.show();
  context.subscriptions.push(statusBar);

  const statusManager = new StatusBarManager(
    statusBar,
    () =>
      vscode.workspace
        .getConfiguration('github.copilot.llm-gateway')
        .get<string>('serverUrl', 'http://localhost:8000'),
    () => provider.getStatusSnapshot()
  );
  context.subscriptions.push(statusManager);

  // Live request state: streaming → responded → idle, with errors flashing in
  // place. The provider fires `start` / `complete` / `error` events around
  // each provideLanguageModelChatResponse call.
  context.subscriptions.push(
    provider.onDidChangeRequestState((event) => statusManager.onRequest(event))
  );

  // Rich hover tooltip is rebuilt from the provider's snapshot — refresh it
  // whenever the snapshot changes (model refresh, request completion, session
  // totals tick) so a hovering user always sees current numbers.
  context.subscriptions.push(
    provider.onDidChangeStatusSnapshot(() => statusManager.refreshTooltip())
  );

  // Status dialog (opened by clicking the status bar) — the GHCP-style
  // QuickPick with connection state, session totals, models, feature toggles,
  // and quick actions. The controller subscribes to the provider's snapshot
  // event while open so the values stay fresh without polling.
  // Tooltip's "Show output log" link needs a registered command (command-link
  // anchors can't call class methods directly). Tiny wrapper around
  // provider.showOutput.
  context.subscriptions.push(
    vscode.commands.registerCommand('github.copilot.llm-gateway.showOutput', () =>
      provider.showOutput()
    )
  );

  /**
   * Probe the gateway silently (no error toast) and render the result in the
   * status bar. Uses the provider's cached fetch so it doesn't double-hit the
   * server when VS Code is already asking for models.
   */
  const refreshStatusBar = async (): Promise<void> => {
    const cts = new vscode.CancellationTokenSource();
    try {
      const models = await provider.provideLanguageModelChatInformation(
        { silent: true },
        cts.token
      );
      if (models.length > 0) {
        statusManager.setIdle(models.map((m) => m.id));
      } else {
        statusManager.setNoModels();
      }
    } catch (error) {
      statusManager.setError(error instanceof Error ? error.message : String(error));
    } finally {
      cts.dispose();
    }
  };

  // Initial silent probe shortly after activation, once VS Code has settled.
  // The timer is registered as a disposable so it can't fire into a
  // disposed provider if the extension is deactivated in the interim.
  const initialProbeTimer = setTimeout(() => {
    void refreshStatusBar();
  }, STATUS_BAR_PROBE_DELAY_MS);
  context.subscriptions.push({ dispose: () => clearTimeout(initialProbeTimer) });

  const testCommand = vscode.commands.registerCommand(
    'github.copilot.llm-gateway.testConnection',
    async () => {
      const cts = new vscode.CancellationTokenSource();
      try {
        const models = await provider.provideLanguageModelChatInformation(
          { silent: false },
          cts.token
        );

        if (models.length > 0) {
          statusManager.setIdle(models.map((m) => m.id));
          vscode.window.showInformationMessage(
            `GitHub Copilot LLM Gateway: Successfully connected! Found ${models.length} model(s): ${models.map((m) => m.name).join(', ')}`
          );
        } else {
          statusManager.setNoModels();
          vscode.window.showWarningMessage(
            'GitHub Copilot LLM Gateway: Connected but no models found.'
          );
        }
      } catch (error) {
        statusManager.setError(error instanceof Error ? error.message : String(error));
        vscode.window.showErrorMessage(
          `GitHub Copilot LLM Gateway: Connection test failed. ${error instanceof Error ? error.message : String(error)}`
        );
      } finally {
        cts.dispose();
      }
    }
  );

  context.subscriptions.push(testCommand);

  // "Manage" command — triggered by the Copilot Language Models panel's
  // "configure setting" button via the managementCommand contribution.
  // Shows a QuickPick menu offering Configure Server, Model Settings, etc.
  const manageCommand = vscode.commands.registerCommand(
    'github.copilot.llm-gateway.manage',
    async () => {
      const pick = await vscode.window.showQuickPick(
        [
          { label: '$(server) Configure Server', description: 'Set server URL and API key', id: 'server' },
          { label: '$(list-unordered) Model Settings', description: 'Per-model reasoning_effort, temperature, etc.', id: 'modelSettings' },
          { label: '$(globe) Copilot Proxy', description: 'Route Copilot background services through Gateway', id: 'copilotProxy' },
          { label: '$(symbol-class) Sub-agent Settings', description: 'Configure Copilot sub-agents to use Gateway models', id: 'subagentSettings' },
          { label: '$(database) Copilot Memory Settings', description: 'Toggle Copilot Memory write/read (cross-session context)', id: 'copilotMemory' },
          { label: '$(edit) Edit Custom Headers', description: 'Manage HTTP headers (stored in secret storage)', id: 'headers' },
          { label: '$(settings-gear) Open Settings', description: 'All LLM Gateway settings', id: 'settings' },
          { label: '$(refresh) Refresh Models', description: 'Re-fetch model list from server', id: 'refresh' },
        ],
        { title: 'LLM Gateway', placeHolder: 'Choose an action' }
      );
      if (!pick) { return; }

      switch (pick.id) {
        case 'server':
          await configureServerFlow(provider, refreshStatusBar);
          break;
        case 'modelSettings':
          await vscode.commands.executeCommand('github.copilot.llm-gateway.modelSettings');
          break;
        case 'copilotProxy':
          await vscode.commands.executeCommand('github.copilot.llm-gateway.copilotProxySettings');
          break;
        case 'subagentSettings':
          await vscode.commands.executeCommand('github.copilot.llm-gateway.subagentSettings');
          break;
        case 'copilotMemory':
          await vscode.commands.executeCommand('github.copilot.llm-gateway.copilotMemorySettings');
          break;
        case 'headers':
          await vscode.commands.executeCommand('github.copilot.llm-gateway.editCustomHeaders');
          break;
        case 'settings':
          await vscode.commands.executeCommand('workbench.action.openSettings', 'github.copilot.llm-gateway');
          break;
        case 'refresh':
          await vscode.commands.executeCommand('github.copilot.llm-gateway.refreshModels');
          break;
      }
    }
  );

  context.subscriptions.push(manageCommand);

  // "Edit Custom Headers" command — lets users manage additional HTTP
  // headers (e.g. `Authorization: Token …`, `Anthropic-Version`) without
  // touching settings.json. Values are persisted via SecretStorage because
  // these headers commonly carry credentials (issue #28).
  const editHeadersCommand = vscode.commands.registerCommand(
    'github.copilot.llm-gateway.editCustomHeaders',
    async () => {
      await editCustomHeadersFlow(provider);
      provider.invalidateModelCache();
      provider.refreshModels();
      await refreshStatusBar();
    }
  );

  context.subscriptions.push(editHeadersCommand);

  // Explicit "Refresh Models" command — previously users could only trigger
  // a re-fetch by editing settings, which was confusing when models
  // temporarily went missing.
  const refreshCommand = vscode.commands.registerCommand(
    'github.copilot.llm-gateway.refreshModels',
    async () => {
      // Invalidate the provider's cache so the next fetch is fresh, then
      // fire the change event (VS Code will re-call
      // provideLanguageModelChatInformation on its own schedule) and
      // update the status bar immediately.
      provider.invalidateModelCache();
      provider.refreshModels();
      await refreshStatusBar();
    }
  );

  context.subscriptions.push(refreshCommand);

  const modelSettingsCommand = vscode.commands.registerCommand(
    'github.copilot.llm-gateway.modelSettings',
    async () => {
      const snapshot = provider.getStatusSnapshot();
      await openModelSettingsQuickPick(snapshot.models);
    }
  );

  context.subscriptions.push(modelSettingsCommand);

  // ---------- Copilot proxy server ----------

  const outputChannel = provider.getOutputChannel();
  let copilotProxy: CopilotProxyServer | undefined;

  /**
   * Read copilotProxy settings from VS Code configuration. Decoupled from
   * the provider — the proxy only needs the upstream URL, key, headers, and
   * its own model-mapping table.
   */
  const readProxyConfigs = (): { upstream: ProxyUpstreamConfig; proxy: CopilotProxyConfig } => {
    const cfg = vscode.workspace.getConfiguration('github.copilot.llm-gateway');
    return {
      upstream: {
        serverUrl: cfg.get<string>('serverUrl', 'http://localhost:8000'),
        apiKey: provider.getResolvedApiKey(),
        customHeaders: provider.getCustomHeadersSnapshot(),
        requestTimeout: cfg.get<number>('requestTimeout', 60000),
        // When false, [CopilotProxy] only logs errors + lifecycle events.
        // Per-request flow (← method url, Proxy: x → y, pass-through) stays
        // quiet to keep the output channel readable in normal use.
        verboseLogging: cfg.get<boolean>('verboseLogging', false),
      },
      proxy: {
        enabled: cfg.get<boolean>('copilotProxy.enabled', false),
        modelMapping: cfg.get<Record<string, string>>('copilotProxy.modelMapping', {}),
      },
    };
  };

  /**
   * Start the proxy server and auto-configure `overrideProxyUrl`.
   */
  const startProxy = async (): Promise<void> => {
    if (copilotProxy?.isRunning) { return; }
    const { upstream, proxy } = readProxyConfigs();
    copilotProxy = new CopilotProxyServer(
      upstream,
      proxy,
      (msg) => outputChannel.appendLine(`[CopilotProxy] ${msg}`)
    );
    try {
      const port = await copilotProxy.start();
      context.subscriptions.push({ dispose: () => copilotProxy?.dispose() });

      // Auto-configure Copilot URL overrides so background services route
      // through us. Copilot has two URL types:
      //   - overrideCapiUrl  → ChatCompletions (copilot-fast, copilot-base)
      //   - overrideProxyUrl → ProxyChatCompletions (agentic search)
      // We set both so all request types are intercepted.
      // These are owned by the Copilot extension — VS Code's config API
      // refuses to write unregistered keys, so we patch settings.json directly.
      const proxyUrl = `http://127.0.0.1:${port}`;
      await writeUserSetting('github.copilot.advanced.debug.overrideCapiUrl', proxyUrl);
      await writeUserSetting('github.copilot.advanced.debug.overrideProxyUrl', proxyUrl);

      outputChannel.appendLine(
        `[CopilotProxy] Started on port ${port}, overrideCapiUrl + overrideProxyUrl → ${proxyUrl}`
      );
      vscode.window.showInformationMessage(
        `LLM Gateway: Copilot proxy started on port ${port}`
      );
    } catch (err) {
      const message = err instanceof Error ? err.message : String(err);
      outputChannel.appendLine(`[CopilotProxy] Failed to start: ${message}`);
      vscode.window.showErrorMessage(`LLM Gateway: Copilot proxy failed to start — ${message}`);
    }
  };

  /**
   * Stop the proxy server and clear `overrideProxyUrl`.
   */
  const stopProxy = async (): Promise<void> => {
    if (!copilotProxy) { return; }
    copilotProxy.dispose();
    copilotProxy = undefined;

    // Clear both Copilot URL overrides so it reverts to its default endpoints
    await removeUserSetting('github.copilot.advanced.debug.overrideCapiUrl');
    await removeUserSetting('github.copilot.advanced.debug.overrideProxyUrl');

    outputChannel.appendLine('[CopilotProxy] Stopped, overrideProxyUrl cleared');
    vscode.window.showInformationMessage('LLM Gateway: Copilot proxy stopped');
  };

  // Copilot Proxy Settings command — QuickPick-based interactive flow
  const proxySettingsCommand = vscode.commands.registerCommand(
    'github.copilot.llm-gateway.copilotProxySettings',
    async () => {
      const snapshot = provider.getStatusSnapshot();
      const availableModels = snapshot.models.map((m) => m.id);
      await copilotProxySettingsFlow(
        () => readProxyConfigs().proxy,
        copilotProxy?.isRunning === true,
        startProxy,
        stopProxy,
        availableModels
      );
    }
  );
  context.subscriptions.push(proxySettingsCommand);

  // Sub-agent Settings command — configure Copilot sub-agents to use
  // Gateway models. Independent of the HTTP proxy.
  const subagentSettingsCommand = vscode.commands.registerCommand(
    'github.copilot.llm-gateway.subagentSettings',
    async () => {
      const snapshot = provider.getStatusSnapshot();
      const availableModels = snapshot.models.map((m) => m.id);
      await subagentSettingsFlow(availableModels);
    }
  );
  context.subscriptions.push(subagentSettingsCommand);

  // Copilot Memory Settings command — toggle Copilot's Memory feature
  // (write/read) to control cross-session context transfer.
  const copilotMemorySettingsCommand = vscode.commands.registerCommand(
    'github.copilot.llm-gateway.copilotMemorySettings',
    async () => {
      await copilotMemorySettingsFlow();
    }
  );
  context.subscriptions.push(copilotMemorySettingsCommand);

  // Auto-start if enabled in settings
  const { proxy: initialProxyConfig } = readProxyConfigs();
  if (initialProxyConfig.enabled) {
    void startProxy();
  }

  // React to config changes — restart proxy when upstream or proxy settings change
  context.subscriptions.push(
    vscode.workspace.onDidChangeConfiguration((e) => {
      if (!e.affectsConfiguration('github.copilot.llm-gateway')) { return; }
      if (copilotProxy?.isRunning) {
        const { upstream, proxy } = readProxyConfigs();
        copilotProxy.updateConfig(upstream, proxy);
        outputChannel.appendLine('[CopilotProxy] Configuration updated');
      }
    })
  );
}

/**
 * Extension deactivation
 */
export function deactivate(): void {
  // Proxy cleanup is handled by context.subscriptions disposables
}

// ---------- Copilot Proxy settings flow ----------

/** Well-known Copilot model names that users will likely need to map. */
/** Well-known Copilot model aliases (label shown to user → actual model name). */
const COPILOT_MODELS = [
  { label: 'copilot-fast', copilotName: 'gpt-4o-mini', description: 'title, summary, classify, commit message, etc.' },
  { label: 'copilot-base', copilotName: 'gpt-4.1-2025-04-14', description: 'some chat requests' },
];

interface ProxyPickItem extends vscode.QuickPickItem {
  action: 'toggle' | 'mapping' | 'addMapping' | 'done';
  copilotModel?: string;
}

/**
 * Copilot Proxy settings — for the local HTTP proxy that routes Copilot's
 * background services (commit message, title generation, etc.) to the
 * upstream LLM. Sub-agent configuration is in a separate menu.
 *
 * Menu (flat, no submenus):
 *   1. Enable/Disable Proxy
 *   2. copilot-fast / copilot-base mappings + custom mappings
 *   3. Add custom mapping
 */
async function copilotProxySettingsFlow(
  getConfig: () => CopilotProxyConfig,
  isRunning: boolean,
  startProxy: () => Promise<void>,
  stopProxy: () => Promise<void>,
  availableModels: string[]
): Promise<void> {
  while (true) {
    const config = getConfig();
    const mapping = config.modelMapping;

    const items: ProxyPickItem[] = [];

    // 1. Toggle Proxy
    if (config.enabled) {
      items.push({
        label: '$(debug-stop) Disable Proxy',
        description: isRunning ? '(running)' : '(enabled but not running)',
        action: 'toggle',
      });
    } else {
      items.push({
        label: '$(play) Enable Proxy',
        description: 'Start routing Copilot background services through Gateway',
        action: 'toggle',
      });
    }

    items.push({
      label: '',
      kind: vscode.QuickPickItemKind.Separator,
      action: 'done',
    });

    // 2. Model mappings — well-known Copilot aliases first
    for (const m of COPILOT_MODELS) {
      const mapped = mapping[m.copilotName];
      items.push({
        label: `$(symbol-parameter) ${m.label}`,
        description: mapped ? `→ ${mapped}` : '(not mapped — passes through unchanged)',
        detail: `${m.copilotName} — ${m.description}`,
        action: 'mapping',
        copilotModel: m.copilotName,
      });
    }

    // Custom mappings not in the well-known list
    const customKeys = Object.keys(mapping).filter(
      (k) => !COPILOT_MODELS.some((m) => m.copilotName === k)
    );
    for (const k of customKeys) {
      items.push({
        label: `$(symbol-key) ${k}`,
        description: `→ ${mapping[k]}`,
        detail: 'Custom model mapping',
        action: 'mapping',
        copilotModel: k,
      });
    }

    items.push({
      label: '$(add) Add custom model mapping...',
      description: '',
      action: 'addMapping',
    });

    items.push({
      label: '',
      kind: vscode.QuickPickItemKind.Separator,
      action: 'done',
    });

    items.push({
      label: '$(check) Done',
      description: 'Close',
      action: 'done',
    });

    const pick = await vscode.window.showQuickPick(items, {
      title: 'LLM Gateway — Copilot Proxy',
      placeHolder: config.enabled
        ? 'Proxy is enabled. Edit model mappings or disable.'
        : 'Proxy is disabled. Enable to route Copilot services through Gateway.',
    });

    if (!pick || pick.action === 'done') { return; }

    if (pick.action === 'toggle') {
      const cfg = vscode.workspace.getConfiguration('github.copilot.llm-gateway');
      if (config.enabled) {
        await cfg.update('copilotProxy.enabled', false, vscode.ConfigurationTarget.Global);
        await stopProxy();
        isRunning = false;
      } else {
        await cfg.update('copilotProxy.enabled', true, vscode.ConfigurationTarget.Global);
        await startProxy();
        isRunning = true;
      }
      continue;
    }

    if (pick.action === 'addMapping') {
      const copilotModel = await vscode.window.showInputBox({
        title: 'Copilot Proxy — Copilot Model Name',
        prompt: 'Enter the Copilot model name (e.g. gpt-4o-mini)',
        placeHolder: 'gpt-4o-mini',
      });
      if (!copilotModel) { continue; }
      await editProxyModelMapping(copilotModel, mapping, availableModels);
      continue;
    }

    if (pick.action === 'mapping' && pick.copilotModel) {
      await editProxyModelMapping(pick.copilotModel, mapping, availableModels);
    }
  }
}

// ---------- Sub-agent model configuration ----------

/** VS Code settings that control sub-agent model selection. */
// ---------- Sub-agent Settings (top-level menu) ----------

interface SubagentPickItem extends vscode.QuickPickItem {
  action: 'chooseDefault' | 'toggleSearch' | 'toggleExecution' | 'done';
}

/**
 * Sub-agent Settings — independent top-level menu (not under Copilot Proxy).
 *
 * Sub-agents route through `vscode.lm` API → Gateway's
 * LanguageModelChatProvider, completely independent of the HTTP proxy.
 *
 * Menu:
 *   1. Default Model — chat.exploreAgent.defaultModel (with vendor suffix)
 *   2. Search Sub-agent enable/disable
 *   3. Execution Sub-agent enable/disable
 */
async function subagentSettingsFlow(availableModels: string[]): Promise<void> {
  while (true) {
    const items: SubagentPickItem[] = [];

    // 1. Default model (chat.exploreAgent.defaultModel)
    const exploreDefault = readVSCodeSetting('chat.exploreAgent.defaultModel');
    items.push({
      label: '$(rocket) Default Model',
      description: exploreDefault ? `→ ${exploreDefault}` : '(not set — Copilot default)',
      detail: 'chat.exploreAgent.defaultModel — routes explore sub-agents through Gateway',
      action: 'chooseDefault',
    });

    items.push({
      label: '',
      kind: vscode.QuickPickItemKind.Separator,
      action: 'done',
    });

    // 2. Search Sub-agent toggle.
    // Enable → enabled=true + useAgenticProxy=false + model="" (inherit parent)
    // Disable → enabled=false
    const searchEnabled = readVSCodeBooleanSetting(
      'github.copilot.chat.searchSubagent.enabled',
      false
    );
    items.push({
      label: searchEnabled
        ? '$(check) Search Sub-agent: Enabled'
        : '$(circle-slash) Search Sub-agent: Disabled',
      description: 'searchSubagent.enabled',
      detail: 'Enable → search uses Gateway via parent model; Disable → hide tool from LLM',
      action: 'toggleSearch',
    });

    // 3. Execution Sub-agent toggle.
    // Enable → enabled=true + model="" (inherit parent)
    // Disable → enabled=false
    const executionEnabled = readVSCodeBooleanSetting(
      'github.copilot.chat.executionSubagent.enabled',
      false
    );
    items.push({
      label: executionEnabled
        ? '$(check) Execution Sub-agent: Enabled'
        : '$(circle-slash) Execution Sub-agent: Disabled',
      description: 'executionSubagent.enabled',
      detail: 'Enable → execution uses Gateway via parent model; Disable → hide tool from LLM',
      action: 'toggleExecution',
    });

    items.push({
      label: '',
      kind: vscode.QuickPickItemKind.Separator,
      action: 'done',
    });

    items.push({
      label: '',
      kind: vscode.QuickPickItemKind.Separator,
      action: 'done',
    });

    items.push({
      label: '$(check) Done',
      description: 'Close',
      action: 'done',
    });

    const pick = await vscode.window.showQuickPick(items, {
      title: 'LLM Gateway — Sub-agent Settings',
      placeHolder: 'Configure Copilot sub-agents to use Gateway models',
    });

    if (!pick || pick.action === 'done') { return; }

    if (pick.action === 'chooseDefault') {
      await chooseSubagentDefaultModel(availableModels);
      continue;
    }

    if (pick.action === 'toggleSearch') {
      if (searchEnabled) {
        await writeVSCodeSetting('github.copilot.chat.searchSubagent.enabled', false);
      } else {
        // Enable: turn on + useAgenticProxy=false + model="" (inherit parent)
        await writeVSCodeSetting('github.copilot.chat.searchSubagent.enabled', true);
        await writeVSCodeSetting('github.copilot.chat.searchSubagent.useAgenticProxy', false);
        await writeVSCodeSetting('github.copilot.chat.searchSubagent.model', '');
      }
      continue;
    }

    if (pick.action === 'toggleExecution') {
      if (executionEnabled) {
        await writeVSCodeSetting('github.copilot.chat.executionSubagent.enabled', false);
      } else {
        // Enable: turn on + model="" (inherit parent)
        await writeVSCodeSetting('github.copilot.chat.executionSubagent.enabled', true);
        await writeVSCodeSetting('github.copilot.chat.executionSubagent.model', '');
      }
      continue;
    }
  }
}

// ---------- Copilot Memory Settings (top-level menu) ----------

interface MemoryPickItem extends vscode.QuickPickItem {
  action: 'toggleWrite' | 'toggleRead' | 'done';
}

/**
 * Copilot Memory Settings — standalone menu for managing Copilot's Memory
 * feature. Memory is the main cross-session context-transfer mechanism in
 * Copilot Chat (the `store_memory` tool persists info between conversations
 * across the user/repo/session layers).
 *
 * Toggles:
 *   - Write Copilot Memory → github.copilot.chat.tools.memory.enabled
 *     Controls whether the LLM is given the `store_memory` tool.
 *   - Read Copilot Memory  → github.copilot.chat.copilotMemory.enabled
 *     Controls whether stored memories are injected into the chat context.
 *
 * Disable both to fully isolate sessions (no new memories saved, no old
 * memories loaded).
 */
async function copilotMemorySettingsFlow(): Promise<void> {
  while (true) {
    const items: MemoryPickItem[] = [];

    const writeEnabled = readVSCodeBooleanSetting(
      'github.copilot.chat.tools.memory.enabled',
      true
    );
    items.push({
      label: writeEnabled
        ? '$(check) Write Copilot Memory: Enabled'
        : '$(circle-slash) Write Copilot Memory: Disabled',
      description: 'github.copilot.chat.tools.memory.enabled',
      detail: 'Enable → LLM can save new memories via the store_memory tool; Disable → no new memories saved',
      action: 'toggleWrite',
    });

    const readEnabled = readVSCodeBooleanSetting(
      'github.copilot.chat.copilotMemory.enabled',
      true
    );
    items.push({
      label: readEnabled
        ? '$(check) Read Copilot Memory: Enabled'
        : '$(circle-slash) Read Copilot Memory: Disabled',
      description: 'github.copilot.chat.copilotMemory.enabled',
      detail: 'Enable → stored memories are injected into chat context; Disable → ignore stored memories',
      action: 'toggleRead',
    });

    items.push({
      label: '',
      kind: vscode.QuickPickItemKind.Separator,
      action: 'done',
    });

    items.push({
      label: '$(check) Done',
      description: 'Close',
      action: 'done',
    });

    const pick = await vscode.window.showQuickPick(items, {
      title: 'LLM Gateway — Copilot Memory Settings',
      placeHolder: 'Toggle Copilot Memory write/read to control cross-session context transfer',
    });

    if (!pick || pick.action === 'done') { return; }

    if (pick.action === 'toggleWrite') {
      await writeVSCodeSetting(
        'github.copilot.chat.tools.memory.enabled',
        !writeEnabled
      );
      continue;
    }

    if (pick.action === 'toggleRead') {
      await writeVSCodeSetting(
        'github.copilot.chat.copilotMemory.enabled',
        !readEnabled
      );
      continue;
    }
  }
}

/**
 * Pick a Gateway model for `chat.exploreAgent.defaultModel`. Writes in the
 * `"id (copilot-llm-gateway)"` format VS Code uses to resolve models by
 * display name (vendor-qualified to avoid colliding with Copilot's official
 * same-named models).
 */
async function chooseSubagentDefaultModel(availableModels: string[]): Promise<void> {
  if (availableModels.length === 0) {
    vscode.window.showWarningMessage('LLM Gateway: No models available. Connect to a server first.');
    return;
  }

  const current = readVSCodeSetting('chat.exploreAgent.defaultModel');
  // Strip optional "(vendor)" suffix so we can match exact model ids.
  const currentModelId = current?.replace(/\s*\([^)]*\)\s*$/, '').trim();

  interface ModelPickItem extends vscode.QuickPickItem {
    modelId?: string;
    action: 'select' | 'clear';
  }

  const items: ModelPickItem[] = [];
  if (current) {
    items.push({
      label: '$(trash) Clear (use Copilot default)',
      description: `Currently: ${current}`,
      action: 'clear',
    });
    items.push({
      label: '',
      kind: vscode.QuickPickItemKind.Separator,
      action: 'clear',
    });
  }
  for (const modelId of availableModels) {
    items.push({
      label: modelId === currentModelId ? `$(check) ${modelId}` : `    ${modelId}`,
      description: '(copilot-llm-gateway)',
      action: 'select',
      modelId,
    });
  }

  const pick = await vscode.window.showQuickPick(items, {
    title: 'Sub-agent Default Model',
    placeHolder: 'Pick a Gateway model for chat.exploreAgent.defaultModel',
  });

  if (!pick) { return; }

  if (pick.action === 'clear') {
    await writeVSCodeSetting('chat.exploreAgent.defaultModel', undefined);
    return;
  }

  if (pick.action === 'select' && pick.modelId) {
    // Vendor-qualified format so Copilot's resolver picks our model, not
    // its own same-named model (vendor=copilot).
    await writeVSCodeSetting(
      'chat.exploreAgent.defaultModel',
      `${pick.modelId} (copilot-llm-gateway)`
    );
  }
}

/** Read a flat-dotted boolean VS Code setting, with fallback to settings.json. */
function readVSCodeBooleanSetting(key: string, defaultValue: boolean): boolean {
  const lastDot = key.lastIndexOf('.');
  if (lastDot < 0) { return defaultValue; }
  const section = key.slice(0, lastDot);
  const property = key.slice(lastDot + 1);
  const fromApi = vscode.workspace.getConfiguration(section).get<boolean>(property);
  if (typeof fromApi === 'boolean') { return fromApi; }
  try {
    const raw = fs.readFileSync(getUserSettingsPath(), 'utf8');
    const settings = JSON.parse(raw) as Record<string, unknown>;
    const direct = settings[key];
    if (typeof direct === 'boolean') { return direct; }
  } catch { /* ignore */ }
  return defaultValue;
}


/**
 * Read a flat-dotted VS Code setting. Tries the configuration API first;
 * if the key isn't registered (e.g. Copilot's internal sub-agent settings),
 * falls back to reading directly from user settings.json.
 */
function readVSCodeSetting(key: string): string | undefined {
  const lastDot = key.lastIndexOf('.');
  if (lastDot < 0) { return undefined; }
  const section = key.slice(0, lastDot);
  const property = key.slice(lastDot + 1);
  const value = vscode.workspace.getConfiguration(section).get<string>(property);
  if (value && value.length > 0) { return value; }

  // Fallback: read directly from user settings.json (for unregistered keys)
  try {
    const raw = fs.readFileSync(getUserSettingsPath(), 'utf8');
    const settings = JSON.parse(raw) as Record<string, unknown>;
    const direct = settings[key];
    return typeof direct === 'string' && direct.length > 0 ? direct : undefined;
  } catch {
    return undefined;
  }
}

/**
 * Write a flat-dotted VS Code setting. Tries the configuration API first;
 * if the key isn't registered by any extension's schema (e.g. Copilot's
 * internal `github.copilot.chat.planAgent.model`), falls back to directly
 * patching the user settings.json.
 */
async function writeVSCodeSetting(key: string, value: unknown): Promise<void> {
  const lastDot = key.lastIndexOf('.');
  if (lastDot < 0) { return; }
  const section = key.slice(0, lastDot);
  const property = key.slice(lastDot + 1);
  try {
    await vscode.workspace
      .getConfiguration(section)
      .update(property, value, vscode.ConfigurationTarget.Global);
  } catch {
    // Setting not registered — patch settings.json directly
    if (value === undefined) {
      await removeUserSetting(key);
    } else {
      await writeUserSetting(key, value);
    }
  }
}

/**
 * Edit or clear a single model mapping entry. Shows the upstream models
 * fetched from the server's `/models` endpoint as a picker list.
 */
async function editProxyModelMapping(
  copilotModel: string,
  currentMapping: Record<string, string>,
  availableModels: string[]
): Promise<void> {
  const currentValue = currentMapping[copilotModel] ?? '';

  // Build picker: available upstream models + clear option
  interface ModelMapPickItem extends vscode.QuickPickItem {
    modelId?: string;
    action: 'select' | 'clear';
  }

  const items: ModelMapPickItem[] = [];

  if (currentValue) {
    items.push({
      label: '$(trash) Clear mapping',
      description: `Remove current mapping (${currentValue})`,
      action: 'clear',
    });
    items.push({
      label: '',
      kind: vscode.QuickPickItemKind.Separator,
      action: 'clear',
    });
  }

  if (availableModels.length === 0) {
    items.push({
      label: '$(warning) No models available',
      description: 'Connect to a server first',
      action: 'clear',
    });
  } else {
    for (const modelId of availableModels) {
      items.push({
        label: modelId === currentValue ? `$(check) ${modelId}` : `    ${modelId}`,
        description: modelId === currentValue ? '(current)' : undefined,
        action: 'select',
        modelId,
      });
    }
  }

  const pick = await vscode.window.showQuickPick(items, {
    title: `Copilot Proxy — Map "${copilotModel}" to`,
    placeHolder: currentValue
      ? `Current: ${copilotModel} → ${currentValue}`
      : `Select an upstream model for ${copilotModel}`,
  });

  if (!pick) { return; }

  const cfg = vscode.workspace.getConfiguration('github.copilot.llm-gateway');
  const mapping = { ...cfg.get<Record<string, string>>('copilotProxy.modelMapping', {}) };

  if (pick.action === 'clear') {
    delete mapping[copilotModel];
  } else if (pick.modelId) {
    mapping[copilotModel] = pick.modelId;
  }

  await cfg.update('copilotProxy.modelMapping', mapping, vscode.ConfigurationTarget.Global);
}

// ---------- Configure Server flow ----------

/**
 * The original "Configure Server" flow extracted from the manage command.
 * Prompts for server URL + API key, saves, refreshes models.
 */
async function configureServerFlow(
  provider: GatewayProvider,
  refreshStatusBar: () => Promise<void>
): Promise<void> {
  const config = vscode.workspace.getConfiguration('github.copilot.llm-gateway');
  const currentUrl = config.get<string>('serverUrl', 'http://localhost:8000');

  const url = await vscode.window.showInputBox({
    title: 'LLM Gateway — Server URL',
    prompt: 'Enter the inference server URL (OpenAI-compatible endpoint)',
    value: currentUrl,
    placeHolder: 'http://localhost:8000',
    ignoreFocusOut: true,
    validateInput: (value) => {
      try {
        new URL(value);
        return undefined;
      } catch {
        return 'Please enter a valid URL';
      }
    },
  });
  if (url === undefined) { return; }

  const apiKey = await vscode.window.showInputBox({
    title: 'LLM Gateway — API Key',
    prompt: 'Enter the API key — saved to VS Code\'s secret storage. Leave empty to clear.',
    password: true,
    placeHolder: 'Optional',
    ignoreFocusOut: true,
  });
  if (apiKey === undefined) { return; }

  const target = await pickConfigurationTarget(config);
  if (target === undefined) { return; }

  await config.update('serverUrl', url, target);
  await provider.setApiKey(apiKey);

  provider.invalidateModelCache();
  provider.refreshModels();
  await refreshStatusBar();

  await offerAdvancedSettings(provider);
}

/**
 * After the basic Configure Server flow, offer the user a chance to edit
 * custom headers (kept in SecretStorage, issue #28) or jump to the Settings
 * UI for the remaining non-secret options.
 */
async function offerAdvancedSettings(provider: GatewayProvider): Promise<void> {
  const completePick: vscode.QuickPickItem = {
    label: 'Complete',
    description: 'Finish configuration',
  };
  const headersPick: vscode.QuickPickItem = {
    label: 'Edit custom headers...',
    description: 'Add or remove HTTP headers (stored in secret storage)',
  };
  const advancedPick: vscode.QuickPickItem = {
    label: 'Edit advanced settings...',
    description: 'Extra model options, timeouts, logging',
  };

  const pick = await vscode.window.showQuickPick(
    [completePick, headersPick, advancedPick],
    {
      title: 'LLM Gateway — Configuration saved',
      placeHolder: 'Done, or continue to advanced options?',
      ignoreFocusOut: true,
    }
  );
  if (pick === headersPick) {
    await editCustomHeadersFlow(provider);
  } else if (pick === advancedPick) {
    await vscode.commands.executeCommand(
      'workbench.action.openSettings',
      'github.copilot.llm-gateway'
    );
  }
}

interface HeaderQuickPickItem extends vscode.QuickPickItem {
  action: 'add' | 'edit' | 'clear' | 'done';
  headerName?: string;
}

/**
 * Quick-pick driven editor for custom headers persisted in SecretStorage.
 * Shows only header names (not values) so peeking at someone else's screen
 * doesn't leak credentials, and supports add / edit / delete / clear-all.
 */
async function editCustomHeadersFlow(provider: GatewayProvider): Promise<void> {
  while (true) {
    const headers = provider.getCustomHeadersSnapshot();
    const headerNames = Object.keys(headers).sort((a, b) => a.localeCompare(b));
    const items = buildHeaderQuickPickItems(headerNames);

    const pick = await vscode.window.showQuickPick(items, {
      title: `LLM Gateway — Custom Headers (${headerNames.length})`,
      placeHolder:
        headerNames.length === 0
          ? 'No custom headers yet. Add one or close.'
          : 'Select a header to edit, or add a new one',
      ignoreFocusOut: true,
    });
    if (!pick || pick.action === 'done') { return; }

    if (pick.action === 'add') {
      await addHeader(provider, headers);
    } else if (pick.action === 'clear') {
      await confirmAndClearHeaders(provider, headerNames.length);
    } else if (pick.action === 'edit' && pick.headerName) {
      await editOrDeleteHeader(provider, headers, pick.headerName);
    }
  }
}

/**
 * Build the quick-pick items for the custom-headers editor. Pulled out so
 * `editCustomHeadersFlow` stays under SonarCloud's cognitive-complexity
 * budget — and so the item shape lives next to its uses.
 */
function buildHeaderQuickPickItems(headerNames: readonly string[]): HeaderQuickPickItem[] {
  const items: HeaderQuickPickItem[] = [
    { label: 'Done', description: 'Save and close', action: 'done' },
    { label: '$(add) Add header...', description: 'Add a new header', action: 'add' },
  ];
  if (headerNames.length === 0) {
    return items;
  }
  items.push(
    {
      label: '$(trash) Clear all headers',
      description: 'Remove every custom header',
      action: 'clear',
    },
    {
      label: '',
      kind: vscode.QuickPickItemKind.Separator,
      action: 'done',
    },
    ...headerNames.map<HeaderQuickPickItem>((name) => ({
      label: name,
      description: 'Edit or remove (value hidden)',
      action: 'edit',
      headerName: name,
    }))
  );
  return items;
}

async function confirmAndClearHeaders(provider: GatewayProvider, count: number): Promise<void> {
  const confirm = await vscode.window.showWarningMessage(
    `Remove all ${count} custom header(s)?`,
    { modal: true },
    'Remove'
  );
  if (confirm === 'Remove') {
    await provider.setCustomHeaders({});
  }
}

async function addHeader(
  provider: GatewayProvider,
  current: Record<string, string>
): Promise<void> {
  const name = await vscode.window.showInputBox({
    title: 'LLM Gateway — New header name',
    prompt: 'e.g. Authorization, Anthropic-Version, HTTP-Referer',
    ignoreFocusOut: true,
    validateInput: (value) => {
      if (value.trim().length === 0) { return 'Header name cannot be empty'; }
      if (/[^\w-]/.test(value)) { return 'Header names typically only contain letters, digits, and dashes'; }
      return undefined;
    },
  });
  if (!name) { return; }
  const value = await vscode.window.showInputBox({
    title: `LLM Gateway — Value for ${name}`,
    prompt: 'Saved to VS Code\'s secret storage',
    password: true,
    ignoreFocusOut: true,
  });
  if (value === undefined) { return; }
  await provider.setCustomHeaders({ ...current, [name.trim()]: value });
}

async function editOrDeleteHeader(
  provider: GatewayProvider,
  current: Record<string, string>,
  name: string
): Promise<void> {
  const action = await vscode.window.showQuickPick(
    [
      { label: 'Edit value', description: 'Replace the current value' },
      { label: 'Remove header', description: 'Delete this header entirely' },
    ],
    {
      title: `LLM Gateway — ${name}`,
      placeHolder: 'Choose an action',
      ignoreFocusOut: true,
    }
  );
  if (!action) { return; }

  if (action.label === 'Remove header') {
    const next = { ...current };
    delete next[name];
    await provider.setCustomHeaders(next);
    return;
  }

  const value = await vscode.window.showInputBox({
    title: `LLM Gateway — New value for ${name}`,
    prompt: 'Saved to VS Code\'s secret storage',
    password: true,
    ignoreFocusOut: true,
  });
  if (value === undefined) { return; }
  await provider.setCustomHeaders({ ...current, [name]: value });
}

/**
 * Asks the user whether to save settings to Workspace or User (Global) scope.
 * Returns undefined if cancelled, or skips the prompt and returns Global when
 * no workspace folder is open (the only meaningful scope in that case).
 *
 * Defaults the highlighted option to whichever scope already has a value, and
 * otherwise prefers Workspace when a folder is open — most users hitting this
 * picker want per-window configuration (issue #23).
 */
async function pickConfigurationTarget(
  config: vscode.WorkspaceConfiguration
): Promise<vscode.ConfigurationTarget | undefined> {
  const hasWorkspaceFolder = (vscode.workspace.workspaceFolders?.length ?? 0) > 0;
  if (!hasWorkspaceFolder) {
    return vscode.ConfigurationTarget.Global;
  }

  const inspection = config.inspect('serverUrl');
  const workspacePick: vscode.QuickPickItem = {
    label: 'Workspace Settings',
    description: inspection?.workspaceValue === undefined ? undefined : '(currently set)',
    detail: 'Apply to this workspace only — different VS Code windows can use different servers.',
  };
  const globalPick: vscode.QuickPickItem = {
    label: 'User Settings (Global)',
    description: inspection?.globalValue === undefined ? undefined : '(currently set)',
    detail: 'Apply to all VS Code windows.',
  };

  const items = inspection?.globalValue !== undefined && inspection?.workspaceValue === undefined
    ? [globalPick, workspacePick]
    : [workspacePick, globalPick];

  const pick = await vscode.window.showQuickPick(items, {
    title: 'LLM Gateway — Save settings to',
    placeHolder: 'Choose where these settings should apply',
    ignoreFocusOut: true,
  });
  if (!pick) { return undefined; }

  return pick === workspacePick
    ? vscode.ConfigurationTarget.Workspace
    : vscode.ConfigurationTarget.Global;
}
