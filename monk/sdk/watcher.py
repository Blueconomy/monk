"""Context manager for scoped monk monitoring."""
from __future__ import annotations
from monk.sdk.instrument import instrument
from monk.sdk.buffer import SpanBuffer


class watch:
    """
    Context manager that instruments LLM calls within a block.

    Usage:
        with monk.watch() as session:
            result = agent.run(task)
        session.flush()
    """
    def __init__(self, flush_every: int = 5, on_finding=None, session_id: str | None = None):
        self.flush_every = flush_every
        self.on_finding = on_finding
        self.session_id = session_id
        self.buffer: SpanBuffer | None = None

    def __enter__(self) -> SpanBuffer:
        self.buffer = instrument(
            flush_every=self.flush_every,
            on_finding=self.on_finding,
            session_id=self.session_id,
        )
        return self.buffer

    def __exit__(self, *args):
        if self.buffer:
            self.buffer.flush()
