"""Subscriptions owned by one 3D scene, with explicit background model updates."""

from functools import wraps

from invesalius.pubsub import pub as Publisher


class ViewEventRouter:
    def __init__(self, background_topics=()):
        self.active = False
        self.disposed = False
        self.background_topics = frozenset(background_topics)
        self._listeners = {}

    def subscribe(self, listener, topicName, **curriedArgs):
        key = (listener, topicName)
        if key not in self._listeners:

            @wraps(listener)
            def dispatch(*args, **kwargs):
                if not self.disposed and (self.active or topicName in self.background_topics):
                    return listener(*args, **kwargs)

            self._listeners[key] = dispatch
        return Publisher.subscribe(self._listeners[key], topicName, **curriedArgs)

    def unsubscribe_owner(self, owner):
        for (listener, topic), dispatch in list(self._listeners.items()):
            if getattr(listener, "__self__", None) is owner:
                Publisher.unsubscribe(dispatch, topic)
                del self._listeners[listener, topic]

    def dispose(self):
        self.disposed = True
        self.active = False
        for (_, topic), dispatch in self._listeners.items():
            Publisher.unsubscribe(dispatch, topic)
        self._listeners.clear()
