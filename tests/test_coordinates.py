from unittest.mock import MagicMock

import invesalius.constants as const
import invesalius.data.coordinates as dco


def test_debug_tracker_uses_only_configured_coordinate_delay(mocker):
    mock_sleep = mocker.patch.object(dco, "sleep")
    tracker_connection = MagicMock(n_coils=1)

    dco.DebugCoordRandom(tracker_connection, const.DEBUGTRACKRANDOM, const.DEFAULT_REF_MODE)

    mock_sleep.assert_not_called()


def test_coordinate_receiver_yields_after_every_tracker_read(mocker):
    receiver = dco.ReceiveCoordinates.__new__(dco.ReceiveCoordinates)
    receiver.event = MagicMock()
    receiver.event.is_set.side_effect = [False, True]
    receiver.sleep_coord = 0.1
    receiver.tracker_connection = MagicMock()
    receiver.tracker_id = const.DEBUGTRACKRANDOM
    receiver.TrackerCoordinates = MagicMock()
    mocker.patch.object(dco, "GetCoordinatesForThread", return_value=([], []))

    receiver.run()

    receiver.event.wait.assert_called_once_with(0.1)
