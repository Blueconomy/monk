from .retry_loop import RetryLoopDetector
from .empty_return import EmptyReturnDetector
from .model_overkill import ModelOverkillDetector
from .context_bloat import ContextBloatDetector
from .agent_loop import AgentLoopDetector

ALL_DETECTORS = [
    RetryLoopDetector(),
    EmptyReturnDetector(),
    ModelOverkillDetector(),
    ContextBloatDetector(),
    AgentLoopDetector(),
]

__all__ = [
    "RetryLoopDetector",
    "EmptyReturnDetector",
    "ModelOverkillDetector",
    "ContextBloatDetector",
    "AgentLoopDetector",
    "ALL_DETECTORS",
]
