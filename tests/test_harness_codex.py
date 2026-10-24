import sys
from pathlib import Path

import pytest

from open_trading_agents.domain import AgentEvent, AgentSpec, HarnessConfig
from open_trading_agents.harnesses import open_session


@pytest.mark.asyncio
@pytest.mark.parametrize("effort", [None, "high"])
async def test_codex_persistent_turns_and_cumulative_usage(tmp_path: Path, effort: str | None) -> None:
    """Verify JSON-RPC initialization, native tool events, and usage deduplication."""
    script = tmp_path / "codex_fixture.py"
    script.write_text('''import json,os,sys
n=0
for line in sys.stdin:
 r=json.loads(line);method=r.get('method');result={}
 if method=='initialized':continue
 if method=='config/read':result={'config':{'mcp_servers':{'ambient':{}}}}
 if method=='thread/start':
  assert 'environments' not in r['params']
  assert r['params']['sandbox']=='workspace-write'
  assert r['params']['config']['mcp_servers']['ambient']['enabled'] is False
  result={'thread':{'id':'thread1'},'model':'native-default','modelProvider':'native-provider'}
 if method=='turn/start':
  effort=os.environ['EXPECTED_EFFORT']
  assert ('effort' in r['params'])==bool(effort)
  if effort:assert r['params']['effort']==effort
  n+=1;tid=f'turn{n}';result={'turn':{'id':tid}}
 print(json.dumps({'id':r['id'],'result':result}),flush=True)
 if method=='turn/start':
  total={'inputTokens':100*n,'outputTokens':10*n,'cachedInputTokens':30*n,'reasoningOutputTokens':2*n,'totalTokens':110*n}
  events=[('turn/started',{'turn':{'id':tid}}),
   ('model/rerouted',{'threadId':'thread1','turnId':tid,'fromModel':'native-default','toModel':'native-rerouted','reason':'highRiskCyberActivity'}),
   ('item/agentMessage/delta',{'itemId':tid,'delta':'hello'}),
   ('item/reasoning/summaryTextDelta',{'itemId':'r','delta':'consider'}),
   ('item/started',{'item':{'type':'mcpToolCall','id':'tool','tool':'get_news','arguments':{'symbol':'NVDA'}}}),
   ('item/completed',{'item':{'type':'mcpToolCall','id':'tool','tool':'get_news','result':{'content':[]}}}),
   ('thread/tokenUsage/updated',{'turnId':tid,'tokenUsage':{'total':total}}),
   ('thread/tokenUsage/updated',{'turnId':tid,'tokenUsage':{'total':total}}),
   ('item/completed',{'item':{'type':'agentMessage','id':tid,'text':str(n)}}),
   ('turn/completed',{'turn':{'id':tid,'status':'completed'}})]
  for method,params in events:print(json.dumps({'method':method,'params':params}),flush=True)
''')
    events: list[AgentEvent] = []

    async def emit(event: AgentEvent) -> None:
        """Collect normalized adapter events."""
        events.append(event)

    spec = AgentSpec("market", "analysis", "system", tmp_path, tmp_path / "sessions")
    session = await open_session(HarnessConfig(kind="codex", command=(sys.executable, str(script)), effort=effort,
                                              environment={"EXPECTED_EFFORT": effort or ""}), spec, emit)
    try:
        assert (await session.prompt("first")).text == "1"
        assert (await session.prompt("second")).text == "2"
    finally:
        await session.close()
    assert sum(bool(event.data.get("prompt_accepted")) for event in events) == 2
    assert [event.data for event in events if event.kind == "model"] == [
        {"harness": "codex", "provider": "native-provider", "model": "native-default"},
        {"harness": "codex", "provider": "native-provider", "model": "native-rerouted"},
    ]
    usage = [event.data for event in events if event.kind == "usage"]
    assert len(usage) == 2
    assert [row["input_tokens"] for row in usage] == [100, 100]
    assert [row["cache_read_tokens"] for row in usage] == [30, 30]
    assert all(row["cache_write_tokens"] is None for row in usage)
    assert any(event.kind == "reasoning" for event in events)
    assert any(event.kind == "tool_result" for event in events)


@pytest.mark.asyncio
async def test_codex_patch_failure_keeps_streamed_native_output(tmp_path: Path) -> None:
    """Retain sandbox failure output when final patch records contain no output field."""
    from open_trading_agents.harnesses.codex import CodexSession

    events: list[AgentEvent] = []

    async def emit(event: AgentEvent) -> None:
        """Collect normalized tool output."""
        events.append(event)

    session = CodexSession(HarnessConfig(kind="codex"), AgentSpec("a", "analysis", "", tmp_path, tmp_path), emit)
    session.tool_output = {}
    await session.handle({"method": "item/fileChange/outputDelta", "params": {"itemId": "patch", "delta": "bwrap failed"}})
    await session.handle({"method": "item/completed", "params": {"item": {"type": "fileChange", "id": "patch", "status": "failed"}}})
    assert [event.text for event in events] == ["bwrap failed", "bwrap failed"]
    assert events[0].data["delta"] is True
    assert events[1].data["final"] is True
    assert events[1].data["is_error"] is True
