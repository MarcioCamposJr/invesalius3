from invesalius.navigation.navigation import NavigationRenderScheduler


def test_navigation_render_requests_are_coalesced_at_independent_rates():
    scheduler = NavigationRenderScheduler()

    scheduler.request_render(volume=True, slices=True)
    scheduler.request_render(volume=True, slices=True)
    assert scheduler.consume_ready(now=0.0) == (True, True)

    scheduler.request_render(volume=True, slices=True)
    scheduler.request_render(volume=True, slices=True)
    assert scheduler.consume_ready(now=0.009) == (False, False)
    assert scheduler.consume_ready(now=0.01) == (True, False)
    assert scheduler.consume_ready(now=0.10) == (False, True)
