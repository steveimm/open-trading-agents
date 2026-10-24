import assert from "node:assert/strict";
import { pathToFileURL } from "node:url";

const { default: extension } = await import(pathToFileURL(process.env.EXTENSION));
const handlers = new Map();
const tools = new Map();
const messages = [];
let active = [];
const pi = {
  on: (name, callback) => handlers.set(name, callback),
  registerTool: (tool) => tools.set(tool.name, tool),
  setActiveTools: (names) => { active = names; },
  sendMessage: (message) => messages.push(message),
};

await extension(pi);
assert.deepEqual([...tools.keys()], ["price"]);
handlers.get("input")();
assert.deepEqual(active, ["price"]);

const result = await tools.get("price").execute("call1", { symbol: "PROBE" });
assert.equal(result.content[0].text, "123.45");
handlers.get("turn_end")({ toolResults: [result] });
assert.deepEqual(active, []);
assert.match(messages[0].content, /budget is exhausted/);

let aborted = false;
const boundary = handlers.get("turn_end")({ toolResults: [{ isError: true }] }, { abort: () => { aborted = true; } });
assert.equal(aborted, true);
assert.equal(boundary.continue, false);
assert.equal(messages.length, 1);

handlers.get("input")();
assert.deepEqual(active, ["price"]);
handlers.get("turn_end")({ toolResults: [] }, { abort: () => { throw new Error("Completed report was aborted"); } });

const payload = { messages: [{ role: "assistant", reasoning: "retained", tool_calls: [{ id: "original" }] }] };
assert.equal(handlers.get("before_provider_request")({ payload }), payload);
assert.equal(handlers.get("cache_warming_decision")().action, "stop");

handlers.get("provider_stream_event")({
  data: { usageMetadata: { promptTokenCount: 100, cachedContentTokenCount: 0, thoughtsTokenCount: 0 } },
});
handlers.get("provider_stream_event")({
  data: { response: { usageMetadata: { promptTokenCount: 100, cachedContentTokenCount: 30 } } },
});
handlers.get("provider_stream_event")({
  data: { metadata: { usage: { inputTokens: 70, cacheReadInputTokens: 20, cacheWriteInputTokens: 10 } } },
});
handlers.get("provider_stream_event")({ data: { choices: [] } });

handlers.get("session_shutdown")();
