import { spawn } from "node:child_process";
import { appendFileSync } from "node:fs";
import { createInterface } from "node:readline";

const publish = (type, data) => appendFileSync(process.env.OTA_METADATA_FILE, JSON.stringify({ type, ...data }) + "\n", { mode: 0o600 });

export default async function tradingTools(pi) {
  pi.on("cache_warming_decision", () => ({ action: "stop" }));
  pi.on("before_provider_request", (event) => {
    publish("ota_provider_request", { payload: event.payload });
    return event.payload;
  });
  pi.on("provider_stream_event", (event) => {
    const data = event.data;
    const usage = data?.usage ?? data?.response?.usage ?? data?.message?.usage;
    if (usage) publish("ota_provider_usage", { usage });
  });
  const command = JSON.parse(process.env.OTA_TOOL_COMMAND || "[]");
  const environment = JSON.parse(process.env.OTA_TOOL_ENV || "{}");
  const allowed = new Set(JSON.parse(process.env.OTA_TOOLS || "[]"));
  const nativeTools = JSON.parse(process.env.OTA_NATIVE_TOOLS || "[]");
  const activeTools = [...nativeTools, ...allowed];
  const maximumRounds = Number(process.env.OTA_MAX_TOOL_ROUNDS || 20);
  let rounds = 0;
  pi.on("input", () => { rounds = 0; pi.setActiveTools(activeTools); return { action: "continue" }; });
  pi.on("turn_end", (event) => {
    if (!event.toolResults?.length) return;
    rounds += 1;
    if (rounds >= maximumRounds) {
      pi.setActiveTools([]);
      pi.sendMessage({ customType: "tool_budget", content: "The tool-round budget is exhausted. Write your final report using the evidence already collected, identifying missing evidence.", display: true });
      publish("ota_budget", { rounds, maximumRounds });
    }
  });
  if (!command.length) { publish("ota_ready", { tools: nativeTools }); return; }
  const child = spawn(command[0], command.slice(1), { stdio: ["pipe", "pipe", "pipe"], env: { ...process.env, ...environment } });
  let sequence = 0;
  const waiting = new Map();
  let diagnostic = "";
  child.stderr.on("data", (data) => { diagnostic = (diagnostic + data.toString()).slice(-4000); });
  const fail = (error) => {
    for (const { reject, timer } of waiting.values()) { clearTimeout(timer); reject(error); }
    waiting.clear();
  };
  child.on("error", fail);
  child.stdin.on("error", fail);
  child.on("exit", (code) => fail(new Error(`Financial tool server exited (${code}): ${diagnostic}`)));
  const lines = createInterface({ input: child.stdout });
  lines.on("line", (line) => {
    let record;
    try { record = JSON.parse(line); } catch { fail(new Error("Financial tool server emitted invalid JSON")); return; }
    const pending = waiting.get(record.id);
    if (pending) {
      waiting.delete(record.id);
      clearTimeout(pending.timer);
      if (record.error) pending.reject(new Error(JSON.stringify(record.error)));
      else pending.resolve(record.result);
    } else if (record.id !== undefined) {
      child.stdin.write(JSON.stringify({ jsonrpc: "2.0", id: record.id, error: { code: -32601, message: "Unsupported server request" } }) + "\n");
    }
  });
  const request = (method, params, signal) => new Promise((resolve, reject) => {
    const id = ++sequence;
    let timer;
    const finish = (callback, value) => {
      clearTimeout(timer); waiting.delete(id); signal?.removeEventListener("abort", cancel); callback(value);
    };
    const cancel = () => {
      child.stdin.write(JSON.stringify({ jsonrpc: "2.0", method: "notifications/cancelled", params: { requestId: id } }) + "\n");
      finish(reject, new Error("Tool execution cancelled"));
    };
    timer = setTimeout(() => finish(reject, new Error(`Financial tool ${method} timed out`)), 120000);
    waiting.set(id, { resolve: (value) => finish(resolve, value), reject: (error) => finish(reject, error), timer });
    if (signal?.aborted) { cancel(); return; }
    signal?.addEventListener("abort", cancel, { once: true });
    child.stdin.write(JSON.stringify({ jsonrpc: "2.0", id, method, params }) + "\n");
  });
  pi.on("session_shutdown", () => { child.kill(); lines.close(); });
  process.once("exit", () => child.kill());
  await request("initialize", { protocolVersion: "2025-03-26", capabilities: {}, clientInfo: { name: "open-trading-agents", version: "0.1.0" } });
  child.stdin.write(JSON.stringify({ jsonrpc: "2.0", method: "notifications/initialized" }) + "\n");
  const registered = new Set();
  let cursor;
  do {
    const result = await request("tools/list", cursor ? { cursor } : {});
    for (const tool of result.tools) {
      if (!allowed.has(tool.name)) continue;
      registered.add(tool.name);
      pi.registerTool({
        name: tool.name, label: tool.name, description: tool.description || tool.name, parameters: tool.inputSchema,
        async execute(_id, params, signal) {
          const result = await request("tools/call", { name: tool.name, arguments: params }, signal);
          if (result.isError) throw new Error(result.content.filter((part) => part.type === "text").map((part) => part.text).join("\n"));
          return { content: result.content, details: {} };
        },
      });
    }
    cursor = result.nextCursor;
  } while (cursor);
  const missing = [...allowed].filter((name) => !registered.has(name));
  if (missing.length) { child.kill(); throw new Error(`Financial tools missing from MCP server: ${missing.join(", ")}`); }
  publish("ota_ready", { tools: [...nativeTools, ...registered] });
}
