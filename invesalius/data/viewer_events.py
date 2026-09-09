"""Activate ordinary PubSub subscriptions with their owning 3D scene."""

from pubsub import pub as pubsub

from invesalius.pubsub import pub as Publisher


class ViewEventRouter:
    def __init__(self, background_topics=()):
        self._active = False
        self.disposed = False
        self.background_topics = frozenset(background_topics)
        self._listeners = {}

    @property
    def active(self):
        return self._active

    @active.setter
    def active(self, active):
        if self.disposed or self._active == active:
            return
        self._active = active
        for listener, topic in self._listeners:
            self._update_subscription(listener, topic)

    def _update_subscription(self, listener, topic):
        if self.active or topic in self.background_topics:
            Publisher.subscribe(listener, topic)
        else:
            Publisher.unsubscribe(listener, topic)

    def manage(self, owner):
        """Take ownership of this object's existing, ordinary subscriptions.

        Inspect only at object creation, never on the pose/render path. Shared
        model singletons are deliberately not included among scene owners.
        """
        if self.disposed:
            return
        pending = [pubsub.getDefaultTopicMgr().getRootAllTopics()]
        while pending:
            topic = pending.pop()
            pending.extend(topic.getSubtopics())
            for registered in topic.getListeners():
                callback = registered.getCallable()
                if getattr(callback, "__self__", None) is owner:
                    name = topic.getName()
                    self._listeners[callback, name] = callback
                    self._update_subscription(callback, name)

    def manage_view(self, view):
        view._event_router = self
        self.manage(view)
        for name in (
            "style",
            "slice_plane",
            "coil_visualizer",
            "marker_visualizer",
            "probe_visualizer",
            "robot_force_visualizer",
            "vector_field_visualizer",
        ):
            owner = getattr(view, name, None)
            if owner is not None:
                self.manage(owner)

    def unsubscribe_owner(self, owner):
        for listener, topic in list(self._listeners):
            if getattr(listener, "__self__", None) is owner:
                Publisher.unsubscribe(listener, topic)
                del self._listeners[listener, topic]

    def dispose(self):
        self.disposed = True
        self._active = False
        for listener, topic in self._listeners:
            Publisher.unsubscribe(listener, topic)
        self._listeners.clear()
