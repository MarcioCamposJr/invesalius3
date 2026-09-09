import os
import sys
from unittest.mock import call

import pytest

from invesalius.data.viewer_events import ViewEventRouter
from invesalius.pubsub.pub import (
    add_sendMessage_hook,
    sendMessage,
    sendMessage_no_hook,
    subscribe,
    unsubscribe,
)


@pytest.fixture
def mock_publisher(mocker):
    return mocker.patch("invesalius.pubsub.pub.Publisher")


def test_subscribe(mock_publisher, mocker):
    mock_publisher.subscribe.return_value = ("mocked_listener", True)
    mock_listener = mocker.Mock()
    listener, success = subscribe(mock_listener, "dummy_topic")
    mock_publisher.subscribe.assert_called_once_with(mock_listener, "dummy_topic")
    assert listener == "mocked_listener"
    assert success is True


def test_unsubscribe(mocker):
    mock_publisher = mocker.patch("invesalius.pubsub.pub.Publisher.unsubscribe")
    mock_listener = mocker.Mock()
    unsubscribe(mock_listener, "test_topic", arg1="value")
    mock_publisher.assert_called_once_with(mock_listener, "test_topic", arg1="value")
    assert unsubscribe(mock_listener, "test_topic") is None


def test_send_message(mock_publisher):
    """Test that sendMessage() calls Publisher.sendMessage() correctly."""
    sendMessage("test_topic", key="value")
    mock_publisher.sendMessage.assert_called_once_with("test_topic", key="value")


def test_send_message_no_hook(mock_publisher):
    """Test that sendMessage_no_hook() calls Publisher.sendMessage() and does not trigger hooks."""
    sendMessage_no_hook("test_topic", key="value")
    mock_publisher.sendMessage.assert_called_once_with("test_topic", key="value")


def test_send_message_with_hook(mock_publisher, mocker):
    """Test that sendMessage() triggers the added hook function."""
    mock_hook = mocker.Mock()
    add_sendMessage_hook(mock_hook)
    sendMessage("test_topic", key="value")
    mock_publisher.sendMessage.assert_called_once_with("test_topic", key="value")
    mock_hook.assert_called_once_with("test_topic", {"key": "value"})


def test_send_message_no_hook_does_not_trigger_hook(mock_publisher, mocker):
    """Test that sendMessage_no_hook() does not call the hook function."""
    mock_hook = mocker.Mock()
    add_sendMessage_hook(mock_hook)
    sendMessage_no_hook("test_topic", key="value")
    mock_publisher.sendMessage.assert_called_once_with("test_topic", key="value")
    mock_hook.assert_not_called()


def test_send_message_hook_is_called(mocker):
    mock_publisher = mocker.patch("invesalius.pubsub.pub.Publisher.sendMessage")
    mock_hook1 = mocker.Mock()
    mock_hook2 = mocker.Mock()
    add_sendMessage_hook(mock_hook1)
    add_sendMessage_hook(mock_hook2)  # This overwrites mock_hook1
    sendMessage("test_topic", key="value")
    mock_publisher.assert_called_once_with("test_topic", key="value")
    # Since hook1 was overwritten, it should NOT be called
    mock_hook1.assert_not_called()
    mock_hook2.assert_called_once_with("test_topic", {"key": "value"})


def test_view_event_router_suspends_inactive_scene_events():
    router = ViewEventRouter({"test.scene.model"})
    received = []

    class View:
        def on_model(self, value):
            received.append(("model", value))

        def on_interaction(self, value):
            received.append(("interaction", value))

    view = View()
    subscribe(view.on_model, "test.scene.model")
    subscribe(view.on_interaction, "test.scene.interaction")
    try:
        router.manage(view)
        sendMessage("test.scene.model", value=1)
        sendMessage("test.scene.interaction", value=2)
        assert received == [("model", 1)]

        router.active = True
        sendMessage("test.scene.interaction", value=3)
        assert received == [("model", 1), ("interaction", 3)]
    finally:
        router.dispose()


def test_view_event_router_keeps_unmanaged_listeners_active():
    router = ViewEventRouter()
    received = []

    class Listener:
        def update(self, value):
            received.append((self, value))

    scene, shared_model = Listener(), Listener()
    subscribe(scene.update, "test.scene.shared")
    subscribe(shared_model.update, "test.scene.shared")
    try:
        router.manage(scene)
        sendMessage("test.scene.shared", value=1)
        assert received == [(shared_model, 1)]
    finally:
        router.dispose()
        unsubscribe(shared_model.update, "test.scene.shared")
