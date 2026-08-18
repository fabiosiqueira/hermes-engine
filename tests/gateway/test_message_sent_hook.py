"""``message:sent`` — outbound delivery accounting on the hook bus.

fabiosiqueira/hermes-binance#124: a host counting channel volume could see the
lanes it drives itself (cron, webhook) but never the replies the agent sends
back to the operator, because the gateway published no event when something
actually reached the chat.  ``agent:end`` says a turn finished — not that a
message was delivered, and never how many.

These tests pin the contract the host counter depends on: one event per
delivered reply, carrying the lane and the platform-message count.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.platforms.base import (
    BasePlatformAdapter,
    MessageEvent,
    SendResult,
    delivered_message_count,
    outbound_slash_command,
)
from gateway.session import SessionSource
from gateway.config import Platform


class _StubAdapter(BasePlatformAdapter):
    """Minimal concrete adapter — the hook helper needs no transport."""

    name = "telegram"  # base exposes ``name`` as a read-only property

    async def connect(self):  # pragma: no cover - unused
        return True

    async def disconnect(self):  # pragma: no cover - unused
        return True

    async def get_chat_info(self, chat_id):  # pragma: no cover - unused
        return {}

    async def send(self, chat_id, content, reply_to=None, metadata=None):  # pragma: no cover
        return SendResult(success=True, message_id="1")


def _adapter_with_hooks():
    adapter = object.__new__(_StubAdapter)
    hooks = MagicMock()
    hooks.emit = AsyncMock()
    runner = MagicMock()
    runner.hooks = hooks
    handler = MagicMock()
    handler.__self__ = runner
    adapter._message_handler = handler
    return adapter, hooks


def _event(text="/status", internal=False):
    source = SessionSource(
        platform=Platform.TELEGRAM,
        user_id="42",
        chat_id="-100777",
    )
    event = MessageEvent(text=text, source=source)
    if internal:
        event.internal = True
    return event


class TestDeliveredMessageCount:
    def test_counts_an_adapter_reported_id_list(self):
        """Telegram reports the whole fan-out under raw_response."""
        result = SendResult(
            success=True,
            message_id="10",
            raw_response={"message_ids": ["10", "11", "12"]},
        )
        assert delivered_message_count(result) == 3

    def test_counts_continuation_ids_from_the_edit_overflow_path(self):
        result = SendResult(
            success=True,
            message_id="10",
            continuation_message_ids=("11", "12"),
        )
        assert delivered_message_count(result) == 3

    def test_single_message_default(self):
        assert delivered_message_count(SendResult(success=True, message_id="10")) == 1

    def test_failed_send_delivered_nothing(self):
        assert delivered_message_count(SendResult(success=False, error="boom")) == 0
        assert delivered_message_count(None) == 0


class TestOutboundSlashCommand:
    def test_parses_the_command_name(self):
        assert outbound_slash_command(_event("/Status now")) == "status"

    def test_free_form_message_has_no_command(self):
        assert outbound_slash_command(_event("como está o BTC?")) == ""


class TestMessageSentHook:
    def test_operator_reply_is_attributed_to_the_command_lane(self):
        adapter, hooks = _adapter_with_hooks()
        result = SendResult(
            success=True,
            message_id="10",
            raw_response={"message_ids": ["10", "11"]},
        )
        asyncio.run(adapter._emit_message_sent_hook(
            _event("/status"),
            result,
            origin=adapter._outbound_reply_origin(_event("/status")),
            session_key="telegram:-100777",
            content="x" * 5000,
        ))
        hooks.emit.assert_awaited_once()
        name, ctx = hooks.emit.await_args.args
        assert name == "message:sent"
        assert ctx["origin"] == "command"
        assert ctx["slash_command"] == "status"
        assert ctx["messages"] == 2
        assert ctx["chars"] == 5000
        assert ctx["platform"] == "telegram"
        assert ctx["chat_id"] == "-100777"
        assert ctx["session_key"] == "telegram:-100777"
        assert ctx["success"] is True

    def test_free_form_operator_message_is_still_the_command_lane(self):
        """The agent answers a command and a plain question identically, and
        #124 asks for one label covering the operator-driven volume."""
        event = _event("qual o risco agora?")
        adapter, hooks = _adapter_with_hooks()
        asyncio.run(adapter._emit_message_sent_hook(
            event,
            SendResult(success=True, message_id="10"),
            origin=adapter._outbound_reply_origin(event),
        ))
        _, ctx = hooks.emit.await_args.args
        assert ctx["origin"] == "command"
        assert ctx["slash_command"] == ""
        assert ctx["messages"] == 1

    def test_self_injected_notification_is_not_command_volume(self):
        event = _event("background job finished", internal=True)
        adapter, hooks = _adapter_with_hooks()
        asyncio.run(adapter._emit_message_sent_hook(
            event,
            SendResult(success=True, message_id="10"),
            origin=adapter._outbound_reply_origin(event),
        ))
        _, ctx = hooks.emit.await_args.args
        assert ctx["origin"] == "internal"

    def test_failed_delivery_reports_zero_messages(self):
        adapter, hooks = _adapter_with_hooks()
        asyncio.run(adapter._emit_message_sent_hook(
            _event(),
            SendResult(success=False, error="chat not found"),
            origin="command",
        ))
        _, ctx = hooks.emit.await_args.args
        assert ctx["success"] is False
        assert ctx["messages"] == 0

    def test_a_raising_handler_never_breaks_delivery(self):
        adapter, hooks = _adapter_with_hooks()
        hooks.emit = AsyncMock(side_effect=RuntimeError("hook exploded"))
        asyncio.run(adapter._emit_message_sent_hook(
            _event(),
            SendResult(success=True, message_id="10"),
            origin="command",
        ))  # must not raise

    def test_no_runner_is_a_no_op(self):
        adapter = object.__new__(_StubAdapter)
        adapter._message_handler = None
        asyncio.run(adapter._emit_message_sent_hook(
            _event(),
            SendResult(success=True, message_id="10"),
            origin="command",
        ))  # must not raise


class TestStreamConsumerAccounting:
    def test_counts_every_message_the_stream_creates(self):
        """The streamed path delivers without an adapter send at the gateway
        boundary, so the consumer must carry its own tally."""
        from gateway.stream_consumer import GatewayStreamConsumer

        consumer = object.__new__(GatewayStreamConsumer)
        consumer._messages_created = 0
        consumer.adapter = MagicMock()
        consumer.adapter.send = AsyncMock(return_value=SendResult(
            success=True,
            message_id="10",
            raw_response={"message_ids": ["10", "11", "12"]},
        ))
        asyncio.run(consumer._counted_send(chat_id="c", content="x"))
        asyncio.run(consumer._counted_send(chat_id="c", content="y"))
        assert consumer.messages_created == 6

    def test_failed_send_adds_nothing(self):
        from gateway.stream_consumer import GatewayStreamConsumer

        consumer = object.__new__(GatewayStreamConsumer)
        consumer._messages_created = 0
        consumer.adapter = MagicMock()
        consumer.adapter.send = AsyncMock(return_value=SendResult(success=False))
        asyncio.run(consumer._counted_send(chat_id="c", content="x"))
        assert consumer.messages_created == 0
