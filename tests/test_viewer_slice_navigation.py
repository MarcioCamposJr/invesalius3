from invesalius.data.viewer_slice_state import SliceNavigationUpdateState


def test_hidden_navigation_viewer_keeps_only_latest_position():
    state = SliceNavigationUpdateState(updates_enabled=False)

    first_deferred = state.defer_position((1, 2, 3, 4, 5, 6), navigation_active=True)
    second_deferred = state.defer_position((7, 8, 9, 10, 11, 12), navigation_active=True)

    assert first_deferred is True
    assert second_deferred is True
    assert state.pending_position == (7, 8, 9)


def test_enabling_navigation_updates_applies_latest_position():
    state = SliceNavigationUpdateState(
        updates_enabled=False,
        pending_position=(7, 8, 9),
    )

    position = state.set_updates_enabled(True)

    assert position == (7, 8, 9)
    assert state.updates_enabled is True
    assert state.pending_position is None


def test_hidden_navigation_viewer_skips_render():
    state = SliceNavigationUpdateState(updates_enabled=False)

    assert state.should_render(navigation_active=True) is False


def test_visible_or_idle_viewer_renders():
    enabled = SliceNavigationUpdateState(updates_enabled=True)
    hidden_but_idle = SliceNavigationUpdateState(updates_enabled=False)

    assert enabled.should_render(navigation_active=True) is True
    assert hidden_but_idle.should_render(navigation_active=False) is True
