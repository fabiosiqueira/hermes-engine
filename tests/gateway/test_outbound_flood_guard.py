"""Outbound flood guard — regression cover for the 熊-spam incident.

fabiosiqueira/hermes-binance#118: 41 chat messages made of a single repeated
CJK character, 20 of them inside one minute, with no other content.  The
characters came from a degenerate model turn; the 41 *messages* came from
``BasePlatformAdapter.truncate_message``, which split any oversized payload
into as many platform messages as it took, unbounded and content-blind.

These tests pin both halves of the fix: a degenerate payload collapses to one
visible error, and any other over-long payload is capped with a visible
notice — while ordinary long replies keep splitting exactly as before.
"""

import os
from unittest.mock import patch

from gateway.outbound_flood_guard import (
    DEFAULT_MAX_OUTBOUND_CHUNKS,
    cap_chunk_fanout,
    describe_degenerate_repetition,
    resolve_max_outbound_chunks,
)
from gateway.platforms.base import BasePlatformAdapter

TELEGRAM_LIMIT = 4096


def _prose(chars: int) -> str:
    """Ordinary long reply: many distinct characters, no periodicity."""
    line = "The strategist reviewed the position and adjusted the floor. "
    words = (line * (chars // len(line) + 2))[:chars]
    return words


class TestDegenerateRepetitionDetector:
    def test_detects_the_incident_payload(self):
        """A wall of one repeated CJK character is a degenerate loop."""
        found = describe_degenerate_repetition("熊" * 40000)
        assert found is not None
        assert found.unit == "熊"
        assert found.repeats == 40000

    def test_tolerates_a_truncated_final_unit(self):
        """A loop cut off at the provider's token cap is still degenerate."""
        found = describe_degenerate_repetition("ab" * 5000 + "a")
        assert found is not None
        assert found.unit == "ab"
        assert found.length == 10001

    def test_ignores_ordinary_long_prose(self):
        assert describe_degenerate_repetition(_prose(40000)) is None

    def test_ignores_short_repetition(self):
        """Separators and emoji rows are legitimate; only long floods count."""
        assert describe_degenerate_repetition("-" * 500) is None

    def test_ignores_a_long_repeating_template(self):
        """A repeated multi-line block is a report, not a broken decoder."""
        block = "| BTCUSDT | 1.234 | open   |\n"
        assert describe_degenerate_repetition(block * 3000) is None


class TestChunkFanoutCap:
    def test_caps_and_leaves_a_visible_notice(self):
        capped = cap_chunk_fanout([f"chunk{i}" for i in range(30)], max_chunks=5)
        assert len(capped) == 5
        assert capped[:4] == ["chunk0", "chunk1", "chunk2", "chunk3"]
        assert "26" in capped[-1]  # 30 - 4 kept = 26 suppressed
        assert "truncated" in capped[-1].lower()

    def test_under_the_cap_is_untouched(self):
        chunks = [f"chunk{i}" for i in range(4)]
        assert cap_chunk_fanout(chunks, max_chunks=5) == chunks

    def test_zero_disables_the_cap(self):
        chunks = [f"chunk{i}" for i in range(50)]
        assert cap_chunk_fanout(chunks, max_chunks=0) == chunks

    def test_env_override(self):
        with patch.dict(os.environ, {"HERMES_MAX_OUTBOUND_CHUNKS": "3"}):
            assert resolve_max_outbound_chunks() == 3
        with patch.dict(os.environ, {"HERMES_MAX_OUTBOUND_CHUNKS": "not-a-number"}):
            assert resolve_max_outbound_chunks() == DEFAULT_MAX_OUTBOUND_CHUNKS


class TestTruncateMessageGuard:
    def test_degenerate_payload_becomes_one_visible_error(self):
        """#118: 40k of 熊 must not fan out into ten chat messages."""
        chunks = BasePlatformAdapter.truncate_message(
            "熊" * 40000, TELEGRAM_LIMIT,
        )
        assert len(chunks) == 1
        assert "熊" * 100 not in chunks[0]
        assert "degenerate loop" in chunks[0]

    def test_degenerate_guard_can_be_disabled(self):
        with patch.dict(os.environ, {"HERMES_DEGENERATE_REPETITION_GUARD": "0"}):
            chunks = BasePlatformAdapter.truncate_message(
                "熊" * 40000, TELEGRAM_LIMIT,
            )
        assert len(chunks) > 1

    def test_oversized_prose_is_capped_not_unbounded(self):
        chunks = BasePlatformAdapter.truncate_message(
            _prose(TELEGRAM_LIMIT * 40), TELEGRAM_LIMIT,
        )
        assert len(chunks) == DEFAULT_MAX_OUTBOUND_CHUNKS
        assert "truncated" in chunks[-1].lower()

    def test_capped_chunks_still_carry_indicators(self):
        """The cap runs before numbering, so (i/N) matches what is sent."""
        chunks = BasePlatformAdapter.truncate_message(
            _prose(TELEGRAM_LIMIT * 40), TELEGRAM_LIMIT,
        )
        total = len(chunks)
        assert chunks[0].endswith(f" (1/{total})")
        assert chunks[-1].endswith(f" ({total}/{total})")

    def test_normal_multi_chunk_reply_is_unchanged(self):
        """A three-message reply must split exactly as it always did."""
        chunks = BasePlatformAdapter.truncate_message(
            _prose(TELEGRAM_LIMIT * 3 - 100), TELEGRAM_LIMIT,
        )
        assert 2 <= len(chunks) <= 4
        assert all("truncated" not in c.lower() for c in chunks)

    def test_short_message_passes_through(self):
        assert BasePlatformAdapter.truncate_message("hi", TELEGRAM_LIMIT) == ["hi"]
