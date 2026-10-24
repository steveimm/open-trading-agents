import asyncio
import json
import sys
from pathlib import Path

import pytest

from open_trading_agents.domain import AgentEvent, AgentSpec, HarnessConfig
from open_trading_agents.harnesses import open_session
from open_trading_agents.harnesses.process import HarnessError


@pytest.mark.asyncio
@pytest.mark.parametrize("allow_execution", [False, True])
@pytest.mark.parametrize("effort", [None, "high"])
async def test_pi_native_protocol_preserves_session_streams_and_unknown_cache(tmp_path: Path, allow_execution: bool, effort: str | None) -> None:
    """Exercise native subprocess framing, session continuation, and authoritative usage."""
    script = tmp_path / "fake_pi.py"
    script.write_text('''import json,os,sys,time
native_allowed='--tools' in sys.argv and 'bash' in sys.argv[sys.argv.index('--tools')+1].split(',')
assert native_allowed==(os.environ['ALLOW_EXECUTION']=='True')
effort=os.environ['EXPECTED_EFFORT']
assert ('--thinking' in sys.argv)==bool(effort)
if effort:assert sys.argv[sys.argv.index('--thinking')+1]==effort
metadata=os.environ['OTA_METADATA_FILE']
with open(metadata,'a') as handle:handle.write(json.dumps({'type':'ota_ready','tools':[]})+'\\n')
count=0
for line in sys.stdin:
 command=json.loads(line)
 kind=command['type']
 if kind=='get_state':
  model={'id':'native-default','provider':'native-provider','baseUrl':'http://private-endpoint','apiKey':'not-for-report'}
  print(json.dumps({'id':command['id'],'type':'response','success':True,'data':{'sessionFile':'persisted-session.jsonl','model':model}}),flush=True)
 elif kind=='prompt':
  count+=1
  print(json.dumps({'id':command['id'],'type':'response','success':True}),flush=True)
  while not os.path.exists(os.environ['ACCEPTANCE_GATE']):time.sleep(0.01)
  events=[{'type':'ota_provider_request','payload':{'messages':[{'content':command['message']}] }},
   {'type':'ota_provider_usage','usage':{'prompt_tokens':100,'completion_tokens':5}},
   {'type':'message_start','message':{'role':'assistant'}},
   {'type':'message_update','assistantMessageEvent':{'type':'thinking_delta','delta':'consider'}},
   {'type':'message_update','assistantMessageEvent':{'type':'text_delta','delta':'answer'}},
   {'type':'tool_execution_start','toolName':'get_news','toolCallId':'t1','args':{'symbol':'NVDA'}},
   {'type':'tool_execution_end','toolName':'get_news','toolCallId':'t1','result':{'content':[]}},
   {'type':'message_end','message':{'role':'assistant','stopReason':'stop','content':[{'type':'text','text':str(count)}],
      'usage':{'input':100,'output':5,'cacheRead':0,'cacheWrite':0,'reasoning':2}}},
   {'type':'agent_end'}, {'type':'auto_retry_start'}, {'type':'auto_retry_end'}, {'type':'agent_settled'}]
  for event in events:
   if event['type'].startswith('ota_'):
    with open(metadata,'a') as handle:handle.write(json.dumps(event)+'\\n')
   else:print(json.dumps(event),flush=True)
''')
    events: list[AgentEvent] = []
    accepted = asyncio.Event()
    gate = tmp_path / "finish_allowed"

    async def emit(event: AgentEvent) -> None:
        """Collect normalized events from the adapter."""
        events.append(event)
        if event.data.get("prompt_accepted"):
            accepted.set()

    spec = AgentSpec("market", "analysis", "system", tmp_path, tmp_path / "sessions", allow_execution=allow_execution)
    session = await open_session(HarnessConfig(command=(sys.executable, str(script)), effort=effort,
                                              environment={"ACCEPTANCE_GATE": str(gate), "ALLOW_EXECUTION": str(allow_execution),
                                                           "EXPECTED_EFFORT": effort or ""}), spec, emit)
    try:
        task = asyncio.create_task(session.prompt("first"))
        await asyncio.wait_for(accepted.wait(), 2)
        assert not task.done()
        gate.touch()
        assert (await task).text == "1"
        assert (await session.prompt("second")).text == "2"
        assert session.session_id == "persisted-session.jsonl"
    finally:
        await session.close()
    assert sum(bool(event.data.get("prompt_accepted")) for event in events) == 2
    assert [event.data for event in events if event.kind == "model"] == [
        {"harness": "pi", "provider": "native-provider", "model": "native-default"},
    ]
    usage = [event for event in events if event.kind == "usage"]
    assert len(usage) == 2
    assert usage[0].data["input_tokens"] == 100
    assert usage[0].data["cache_read_tokens"] is None
    assert usage[0].data["reasoning_tokens"] == 2
    assert usage[0].data["request_id"] != usage[1].data["request_id"]
    assert any(event.kind == "reasoning" and event.text == "consider" for event in events)
    assert any(event.kind == "tool_start" and event.data["tool"] == "get_news" for event in events)
    assert [event.text for event in events if event.kind == "prompt"] == ["first", "second"]
    assert any(json.loads(event.text)["messages"][0]["content"] == "first"
               for event in events if event.kind == "system" and event.data.get("source") == "native_request")


@pytest.mark.asyncio
async def test_process_exit_does_not_leave_prompt_waiting(tmp_path: Path) -> None:
    """Surface a crashed native executable instead of hanging during startup."""
    async def emit(event: AgentEvent) -> None:
        """Accept events from the failing subprocess."""
        return None

    spec = AgentSpec("market", "analysis", "system", tmp_path, tmp_path / "sessions")
    with pytest.raises(HarnessError, match="exited"):
        await open_session(HarnessConfig(command=(sys.executable, "-c", "raise SystemExit(1)")), spec, emit)


@pytest.mark.asyncio
async def test_pi_auxiliary_compaction_usage_counts_cached_input(tmp_path: Path) -> None:
    """Include native compaction costs without counting cached input twice."""
    from open_trading_agents.harnesses.pi import PiSession

    events: list[AgentEvent] = []

    async def emit(event: AgentEvent) -> None:
        """Collect normalized usage."""
        events.append(event)

    session = PiSession(HarnessConfig(), AgentSpec("a", "analysis", "", tmp_path, tmp_path), emit)
    await session.auxiliary_usage({"input": 20, "cacheRead": 30, "cacheWrite": 0, "output": 5}, "native-entry", "compaction")
    assert events[0].data == {
        "request_id": "compaction:native-entry", "source": "compaction", "input_tokens": 50, "output_tokens": 5,
        "cache_read_tokens": 30, "cache_write_tokens": None, "reasoning_tokens": None,
    }
