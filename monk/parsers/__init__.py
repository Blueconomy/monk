from .auto import parse_traces, TraceCall
from .otel import parse_spans, spans_to_trace_calls, is_otel_format, Span

__all__ = [
    "parse_traces",
    "TraceCall",
    "parse_spans",
    "spans_to_trace_calls",
    "is_otel_format",
    "Span",
]
