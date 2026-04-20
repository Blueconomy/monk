"""monk — Agentic workflow blind spot detector."""
__version__ = "0.4.8"

from monk.sdk import instrument, SpanBuffer
from monk.sdk.watcher import watch

__all__ = ["instrument", "SpanBuffer", "watch"]
