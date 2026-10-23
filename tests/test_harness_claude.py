import sys
from pathlib import Path

import pytest

from open_trading_agents.domain import AgentEvent, AgentSpec, HarnessConfig
from open_trading_agents.harnesses import open_session


@pytest.mark.asyncio
async def test_claude_raw_stream_usage_avoids_final_and_cumulative_duplicates(tmp_path: Path) -> None:
    """Count complete provider messages once despite repeated assistant and result counters."""
    script = tmp_path / "claude_fixture.py"
    script.write_text('''import json,sys
n=0
for line in sys.stdin:
 r=json.loads(line)
 if r['type']!='user':continue
 n+=1;mid=f'message{n}'
 events=[{'type':'system','subtype':'init','session_id':r['session_id']},
  {'type':'stream_event','event':{'type':'message_start','message':{'id':mid,'usage':{'input_tokens':70,'output_tokens':0,'cache_read_input_tokens':30}}}},
  {'type':'stream_event','event':{'type':'content_block_delta','index':0,'delta':{'type':'text_delta','text':'hello'}}},
  {'type':'stream_event','event':{'type':'message_delta','usage':{'output_tokens':10}}},
  {'type':'stream_event','event':{'type':'message_stop'}},
  {'type':'assistant','message':{'id':mid,'content':[{'type':'text','text':str(n)}],'usage':{'input_tokens':0,'output_tokens':0}}},
  {'type':'result','subtype':'success','is_error':False,'result':str(n),'usage':{'input_tokens':70*n,'output_tokens':10*n,'cache_read_input_tokens':30*n}}]
 for event in events:print(json.dumps(event),flush=True)
''')
    events: list[AgentEvent] = []

    async def emit(event: AgentEvent) -> None:
        """Collect normalized adapter events."""
        events.append(event)

    spec = AgentSpec("market", "analysis", "system", tmp_path, tmp_path / "sessions")
    session = await open_session(HarnessConfig(kind="claude", command=(sys.executable, str(script))), spec, emit)
    try:
        assert (await session.prompt("first")).text == "1"
        assert (await session.prompt("second")).text == "2"
    finally:
        await session.close()
    usage = [event.data for event in events if event.kind == "usage"]
    assert len(usage) == 2
    assert [row["input_tokens"] for row in usage] == [100, 100]
    assert [row["output_tokens"] for row in usage] == [10, 10]
    assert all(row["cache_read_tokens"] == 30 for row in usage)
    assert all(row["cache_write_tokens"] is None for row in usage)
    assert all(event.data.get("message_id") == "message1:0" for event in events if event.kind == "text" and event.text == "1")
