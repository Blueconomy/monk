from .retry_loop import RetryLoopDetector
from .empty_return import EmptyReturnDetector
from .model_overkill import ModelOverkillDetector
from .context_bloat import ContextBloatDetector
from .agent_loop import AgentLoopDetector
from .latency_spike import LatencySpikeDetector
from .error_cascade import ErrorCascadeDetector
from .tool_dependency import ToolDependencyDetector
from .cross_turn_memory import CrossTurnMemoryDetector

# Detectors that run on normalised TraceCall lists (work on all formats)
TRACE_DETECTORS = [
    RetryLoopDetector(),
    EmptyReturnDetector(),
    ModelOverkillDetector(),
    ContextBloatDetector(),
    AgentLoopDetector(),
]

# Detectors that require OTEL span trees (deeper analysis)
SPAN_DETECTORS = [
    LatencySpikeDetector(),
    ErrorCascadeDetector(),
    ToolDependencyDetector(),
    CrossTurnMemoryDetector(),
]

# Legacy alias — all detectors, used by CLI
ALL_DETECTORS = TRACE_DETECTORS + SPAN_DETECTORS

__all__ = [
    "RetryLoopDetector",
    "EmptyReturnDetector",
    "ModelOverkillDetector",
    "ContextBloatDetector",
    "AgentLoopDetector",
    "LatencySpikeDetector",
    "ErrorCascadeDetector",
    "ToolDependencyDetector",
    "CrossTurnMemoryDetector",
    "TRACE_DETECTORS",
    "SPAN_DETECTORS",
    "ALL_DETECTORS",
]
