from unittest.mock import MagicMock

import invesalius.constants as const
import invesalius.data.coordinates as dco


def test_debug_tracker_uses_only_configured_coordinate_delay(mocker):
    mock_sleep = mocker.patch.object(dco, "sleep")
    tracker_connection = MagicMock(n_coils=1)

    dco.DebugCoordRandom(tracker_connection, const.DEBUGTRACKRANDOM, const.DEFAULT_REF_MODE)

    mock_sleep.assert_not_called()
