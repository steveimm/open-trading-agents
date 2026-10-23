import json
import sys
from pathlib import Path

import pytest

from open_trading_agents.domain import AgentEvent, AgentSpec, HarnessConfig
from open_trading_agents.harnesses import open_session
from open_trading_agents.harnesses.process import HarnessError


@pytest.mark.asyncio
async def test_pi_native_protocol_preserves_session_streams_and_unknown_cache(tmp_path: Path) -> None:
    """Exercise native subprocess framing, session continuation, and authoritative usage."""
    script = tmp_path / "fake_pi.py"
    script.write_text('''import json,sys
count=0
for line in sys.stdin:
 command=json.loads(line)
 kind=command['type']
 if kind=='get_state':
  print(json.dumps({'id':command['id'],'type':'response','success':True,'data':{'sessionFile':'persisted-session.jsonl'}}),flush=True)
 elif kind=='prompt':
  count+=1
  print(json.dumps({'id':command['id'],'type':'response','success':True}),flush=True)
  events=[{'type':'ota_provider_request','payload':{'messages':[{'content':command['message']}] }},
   {'type':'ota_provider_usage','usage':{'prompt_tokens':100,'completion_tokens':5}},
   {'type':'message_start','message':{'role':'assistant'}},
   {'type':'message_update','assistantMessageEvent':{'type':'thinking_delta','delta':'consider'}},
   {'type':'message_update','assistantMessageEvent':{'type':'text_delta','delta':'answer'}},
   {'type':'tool_execution_start','toolName':'get_news','toolCallId':'t1','args':{'symbol':'NVDA'}},
   {'type':'tool_execution_end','toolName':'get_news','toolCallId':'t1','result':{'content':[]}},
   {'type':'message_end','message':{'role':'assistant','stopReason':'stop','content':[{'type':'text','text':str(count)}],
      'usage':{'input':80,'output':5,'cacheRead':20,'cacheWrite':0,'reasoning':2}}},
   {'type':'agent_end'}, {'type':'auto_retry_start'}, {'type':'auto_retry_end'}, {'type':'agent_settled'}]
  for event in events: print(json.dumps(event),flush=True)
''')
    events: list[AgentEvent] = []

    async def emit(event: AgentEvent) -> None:
        """Collect normalized events from the adapter."""
        events.append(event)

    spec = AgentSpec("market", "analysis", "system", tmp_path, tmp_path / "sessions")
    session = await open_session(HarnessConfig(command=(sys.executable, str(script))), spec, emit)
    try:
        assert (await session.prompt("first")).text == "1"
        assert (await session.prompt("second")).text == "2"
        assert session.session_id == "persisted-session.jsonl"
    finally:
        await session.close()
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
