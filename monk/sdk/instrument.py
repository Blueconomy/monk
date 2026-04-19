"""
Auto-instrumentation for OpenAI and Anthropic SDKs.
Wraps API calls to capture TraceCall records into the SpanBuffer.
"""
from __future__ import annotations
import functools
import time
import uuid
from monk.sdk.buffer import SpanBuffer
from monk.parsers.auto import TraceCall, ToolCall

_buffer: SpanBuffer | None = None
_session_id: str = ""
_call_index: int = 0


def instrument(
    flush_every: int = 5,
    on_finding=None,
    session_id: str | None = None,
) -> SpanBuffer:
    """
    Patch OpenAI and Anthropic clients to auto-capture trace data.

    Usage:
        import monk
        monk.instrument()   # that's it

    Returns the SpanBuffer so you can call .flush() or .get_calls() manually.
    """
    global _buffer, _session_id, _call_index
    _session_id = session_id or str(uuid.uuid4())[:8]
    _call_index = 0
    _buffer = SpanBuffer(flush_every=flush_every, on_finding=on_finding)

    _patch_openai()
    _patch_anthropic()

    return _buffer


def _patch_openai() -> None:
    try:
        import openai
        # Patch the create method on chat completions
        original = openai.chat.completions.create

        @functools.wraps(original)
        def patched(*args, **kwargs):
            start = time.time()
            response = original(*args, **kwargs)
            _record_openai(kwargs, response, time.time() - start)
            return response

        openai.chat.completions.create = patched
    except (ImportError, AttributeError):
        pass


def _patch_anthropic() -> None:
    try:
        import anthropic
        client_cls = anthropic.Anthropic
        original_create = client_cls.messages.create.__func__ if hasattr(client_cls.messages.create, '__func__') else None

        # Patch at the module level for the messages.create method
        orig = anthropic.resources.messages.Messages.create

        @functools.wraps(orig)
        def patched(self, *args, **kwargs):
            start = time.time()
            response = orig(self, *args, **kwargs)
            _record_anthropic(kwargs, response, time.time() - start)
            return response

        anthropic.resources.messages.Messages.create = patched
    except (ImportError, AttributeError):
        pass


def _record_openai(kwargs: dict, response, duration_s: float) -> None:
    global _call_index
    if _buffer is None:
        return

    model = kwargs.get("model", "")
    usage = getattr(response, "usage", None)
    input_tok = getattr(usage, "prompt_tokens", 0) or 0
    output_tok = getattr(usage, "completion_tokens", 0) or 0

    # Extract tool calls from response
    tool_calls = []
    choices = getattr(response, "choices", [])
    for choice in choices:
        msg = getattr(choice, "message", None)
        if msg and hasattr(msg, "tool_calls") and msg.tool_calls:
            for tc in msg.tool_calls:
                fn = getattr(tc, "function", None)
                if fn:
                    tool_calls.append(ToolCall(
                        name=getattr(fn, "name", ""),
                        arguments=getattr(fn, "arguments", ""),
                    ))

    # Extract input/output text for sample collection
    messages = kwargs.get("messages", [])
    system_tok = 0
    input_text = ""
    output_text = ""

    for m in messages:
        if isinstance(m, dict):
            role = m.get("role", "")
            content = m.get("content", "")
            if role == "system":
                system_tok = len(str(content)) // 4  # rough estimate
            input_text += f"[{role}]: {content}\n"

    for choice in choices:
        msg = getattr(choice, "message", None)
        if msg:
            output_text = str(getattr(msg, "content", "") or "")

    call = TraceCall(
        session_id=_session_id,
        call_index=_call_index,
        model=model,
        input_tokens=input_tok,
        output_tokens=output_tok,
        system_prompt_tokens=system_tok,
        tool_calls=tool_calls,
        raw={
            "input_text": input_text[:2000],     # capped for memory
            "output_text": output_text[:2000],
            "input_length_chars": len(input_text),
            "output_length_chars": len(output_text),
            "duration_s": duration_s,
            "latency_ms": duration_s * 1000,
        }
    )
    _call_index += 1
    _buffer.add(call)


def _record_anthropic(kwargs: dict, response, duration_s: float) -> None:
    global _call_index
    if _buffer is None:
        return

    model = kwargs.get("model", "")
    usage = getattr(response, "usage", None)
    input_tok = getattr(usage, "input_tokens", 0) or 0
    output_tok = getattr(usage, "output_tokens", 0) or 0

    messages = kwargs.get("messages", [])
    system = kwargs.get("system", "")
    system_tok = len(str(system)) // 4
    input_text = f"[system]: {system}\n" if system else ""
    for m in messages:
        if isinstance(m, dict):
            input_text += f"[{m.get('role','')}]: {m.get('content','')}\n"

    output_text = ""
    content_blocks = getattr(response, "content", [])
    for block in content_blocks:
        if hasattr(block, "text"):
            output_text += block.text

    tool_calls = []
    for block in content_blocks:
        if getattr(block, "type", "") == "tool_use":
            tool_calls.append(ToolCall(
                name=getattr(block, "name", ""),
                arguments=str(getattr(block, "input", "")),
            ))

    call = TraceCall(
        session_id=_session_id,
        call_index=_call_index,
        model=model,
        input_tokens=input_tok,
        output_tokens=output_tok,
        system_prompt_tokens=system_tok,
        tool_calls=tool_calls,
        raw={
            "input_text": input_text[:2000],
            "output_text": output_text[:2000],
            "input_length_chars": len(input_text),
            "output_length_chars": len(output_text),
            "duration_s": duration_s,
            "latency_ms": duration_s * 1000,
        }
    )
    _call_index += 1
    _buffer.add(call)
