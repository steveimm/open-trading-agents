from open_trading_agents.domain import AgentSpec, EventSink, HarnessConfig, HarnessSession
from open_trading_agents.harnesses.process import start_checked


async def open_session(config: HarnessConfig, spec: AgentSpec, emit: EventSink) -> HarnessSession:
    """Create an isolated conversation through the selected native CLI.

    Args:
        config: Native executable, model selection, and environment overrides.
        spec: Agent identity, instructions, session storage, and financial tools.
        emit: Async sink receiving normalized native events.

    Returns:
        A ready persistent harness conversation.
    """
    if config.kind == "pi":
        from open_trading_agents.harnesses.pi import PiSession

        session = PiSession(config, spec, emit)
    elif config.kind == "codex":
        from open_trading_agents.harnesses.codex import CodexSession

        session = CodexSession(config, spec, emit)
    elif config.kind == "claude":
        from open_trading_agents.harnesses.claude import ClaudeSession

        session = ClaudeSession(config, spec, emit)
    else:
        raise ValueError(f"Unknown harness: {config.kind}")
    await start_checked(session, session.start)
    return session
