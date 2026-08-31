from unittest.mock import Mock

import invesalius.data.slice_  # noqa: F401 - establishes the application's import order
from invesalius.data.viewer_slice import Viewer


class NavigationViewer:
    UpdateSlicesPosition = Viewer.UpdateSlicesPosition
    SetNavigationUpdatesEnabled = Viewer.SetNavigationUpdatesEnabled
    UpdateRender = Viewer.UpdateRender

    def __init__(self, navigation_active, updates_enabled):
        self.nav_status = navigation_active
        self._slice_navigation_updates_enabled = updates_enabled
        self._pending_navigation_position = None
        self._apply_navigation_position = Mock()
        self.interactor = Mock()


def test_hidden_navigation_viewer_applies_only_latest_position_when_reenabled():
    viewer = NavigationViewer(navigation_active=True, updates_enabled=False)

    viewer.UpdateSlicesPosition((1, 2, 3, 4, 5, 6))
    viewer.UpdateSlicesPosition((7, 8, 9, 10, 11, 12))
    viewer.UpdateRender()

    viewer._apply_navigation_position.assert_not_called()
    viewer.interactor.Render.assert_not_called()

    viewer.SetNavigationUpdatesEnabled(True)

    viewer._apply_navigation_position.assert_called_once_with((7, 8, 9))
    assert viewer._pending_navigation_position is None


def test_disabled_updates_do_not_affect_viewer_outside_navigation():
    viewer = NavigationViewer(navigation_active=False, updates_enabled=False)

    viewer.UpdateSlicesPosition((1, 2, 3, 4, 5, 6))
    viewer.UpdateRender()

    viewer._apply_navigation_position.assert_called_once_with((1, 2, 3))
    viewer.interactor.Render.assert_called_once_with()
