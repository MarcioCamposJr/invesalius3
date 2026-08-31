from dataclasses import dataclass


@dataclass
class SliceNavigationUpdateState:
    updates_enabled: bool = True
    pending_position: tuple | None = None

    def defer_position(self, position, navigation_active):
        if navigation_active and not self.updates_enabled:
            self.pending_position = tuple(position[:3])
            return True

        self.pending_position = None
        return False

    def set_updates_enabled(self, enabled):
        self.updates_enabled = enabled
        if not enabled:
            return None

        position = self.pending_position
        self.pending_position = None
        return position

    def should_render(self, navigation_active):
        return not navigation_active or self.updates_enabled
