"""Outbound flood guard for the messaging gateway.

The gateway turns one logical agent reply into as many platform messages as
the payload needs (``BasePlatformAdapter.truncate_message``).  That split was
unbounded and content-blind: a single degenerate model turn — the classic
"repeat one token until the cap" failure mode, seen right after a provider
fallback switch or under rate limiting — was faithfully fanned out into dozens
of consecutive chat messages of pure noise (#118: 41 messages of a repeated
``熊``, 20 of them inside the same minute).

Two independent guards live here, both pure functions so the adapter layer can
call them from any send path:

- :func:`describe_degenerate_repetition` recognises a payload that is a short
  unit repeated to fill the buffer.  The gateway replaces it with ONE visible
  error instead of N messages of garbage — a degenerate loop should surface as
  a failure the operator can act on, not as chat flood.
- :func:`cap_chunk_fanout` bounds how many platform messages one reply may
  become, whatever the content.  The tail is replaced by a visible notice; the
  full text remains in the session transcript.

Both are advisory-by-default-on and tunable through the environment so an
operator who genuinely wants unbounded fan-out can opt out.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List, Optional, Sequence

# Default ceiling on how many platform messages a single reply may occupy.
# 12 Telegram messages is ~49k characters — already far past any reply a human
# reads in a chat window, and an order of magnitude below the #118 incident.
DEFAULT_MAX_OUTBOUND_CHUNKS = 12

# A degenerate payload that fits in this many platform messages is noise, not
# a flood, and is delivered as-is: the guard exists to stop a burst, not to
# police repetitive content.
DEGENERATE_MIN_CHUNKS = 3

# A payload shorter than this is never treated as a degenerate loop: short
# repeated strings are legitimate (ASCII art, "----" separators, a row of
# emoji reactions).
DEGENERATE_MIN_LENGTH = 2000

# A degenerate loop repeats a *short* unit.  Anything longer than this is a
# repeating template, not a broken decoder, and is left alone.
DEGENERATE_MAX_PERIOD = 64

# The repeated unit must fill the payload this many times over.
DEGENERATE_MIN_REPEATS = 100

# Cheap pre-filter: real prose, code, and base64 all blow past this alphabet
# size immediately, so the periodicity scan below only ever runs on payloads
# that are already suspicious.
DEGENERATE_MAX_DISTINCT_CHARS = 8


@dataclass(frozen=True)
class DegenerateRepetition:
    """A payload identified as one short unit repeated to fill the buffer."""

    unit: str
    repeats: int
    length: int


def resolve_max_outbound_chunks() -> int:
    """Ceiling on platform messages per reply.

    ``HERMES_MAX_OUTBOUND_CHUNKS`` overrides the default; ``0`` (or any
    non-positive value) disables the cap entirely.  A malformed value falls
    back to the default rather than raising on a send path.
    """
    raw = os.environ.get("HERMES_MAX_OUTBOUND_CHUNKS")
    if raw is None or not str(raw).strip():
        return DEFAULT_MAX_OUTBOUND_CHUNKS
    try:
        return int(str(raw).strip())
    except (TypeError, ValueError):
        return DEFAULT_MAX_OUTBOUND_CHUNKS


def degenerate_repetition_guard_enabled() -> bool:
    """True unless ``HERMES_DEGENERATE_REPETITION_GUARD`` opts out."""
    raw = os.environ.get("HERMES_DEGENERATE_REPETITION_GUARD")
    if raw is None:
        return True
    return str(raw).strip().lower() not in {"0", "false", "no", "off"}


def describe_degenerate_repetition(
    text: str,
    min_length: int = DEGENERATE_MIN_LENGTH,
) -> Optional[DegenerateRepetition]:
    """Return a description when ``text`` is one short unit repeated, else None.

    The check is deliberately conservative — it fires only on payloads that are
    long, drawn from a tiny alphabet, AND exactly periodic — so an ordinary
    reply can never be mistaken for a decoder loop.  A trailing partial unit is
    tolerated: a loop truncated at the provider's token cap does not end on a
    unit boundary.
    """
    if not isinstance(text, str):
        return None
    length = len(text)
    if length < min_length:
        return None
    if len(set(text)) > DEGENERATE_MAX_DISTINCT_CHARS:
        return None
    for period in range(1, DEGENERATE_MAX_PERIOD + 1):
        if length < period * DEGENERATE_MIN_REPEATS:
            break
        # ``text`` is p-periodic iff every character equals the one p positions
        # back, which is exactly this slice equality.
        if text[period:] == text[:-period]:
            return DegenerateRepetition(
                unit=text[:period],
                repeats=length // period,
                length=length,
            )
    return None


def degenerate_repetition_notice(found: DegenerateRepetition) -> str:
    """One-message operator-visible replacement for a degenerate payload."""
    unit = found.unit if len(found.unit) <= 8 else found.unit[:8] + "…"
    return (
        "⚠️ Reply suppressed: the model emitted a degenerate loop — "
        f"{unit!r} repeated ~{found.repeats:,} times ({found.length:,} chars) "
        "with no other content. Sending it would have flooded this chat with "
        "dozens of junk messages. The raw output is still in the session "
        "transcript. This usually means the provider degraded mid-turn "
        "(common right after a fallback switch or under rate limiting) — "
        "retry the request or switch models."
    )


def chunk_cap_notice(dropped: int, max_chunks: int) -> str:
    """Visible replacement for the tail of an over-long multi-message reply."""
    return (
        f"⚠️ Reply truncated: {dropped:,} more message(s) were suppressed by "
        f"the outbound flood cap ({max_chunks} messages per reply). The full "
        "text is in the session transcript. Raise or disable the cap with "
        "HERMES_MAX_OUTBOUND_CHUNKS."
    )


def cap_chunk_fanout(
    chunks: Sequence[str],
    max_chunks: Optional[int] = None,
) -> List[str]:
    """Bound a split reply to ``max_chunks`` platform messages.

    The last kept slot carries a visible notice instead of content, so the
    truncation is never silent.  ``max_chunks <= 0`` disables the cap.
    """
    result = list(chunks)
    limit = resolve_max_outbound_chunks() if max_chunks is None else max_chunks
    if limit <= 0 or len(result) <= limit:
        return result
    if limit == 1:
        return [chunk_cap_notice(len(result), limit)]
    kept = result[: limit - 1]
    kept.append(chunk_cap_notice(len(result) - (limit - 1), limit))
    return kept
