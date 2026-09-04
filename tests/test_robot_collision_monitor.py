import importlib
import sys
import types

import invesalius.gui  # noqa: F401 - initialize package before stubbing its dialogs


def make_registration(object_id):
    return {
        "obj_id": object_id,
        "fiducials": [
            [-1, 0, 0],
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 0],
        ],
        "orientations": [[0, 0, 0]] * 4,
    }


def test_collision_monitor_accepts_one_robotized_coil(monkeypatch):
    # robot.py imports the GUI dialog module, which has an unrelated circular
    # dependency when loaded in isolation during tests.
    monkeypatch.setitem(
        sys.modules, "invesalius.gui.dialogs", types.ModuleType("invesalius.gui.dialogs")
    )
    sys.modules.pop("invesalius.navigation.robot", None)
    robot_module = importlib.import_module("invesalius.navigation.robot")

    created_monitors = []

    class FakeMonitor:
        is_running = False

        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)
            created_monitors.append(self)

        def start(self):
            return True

    monkeypatch.setattr(robot_module, "CoilCollisionMonitor", FakeMonitor)

    registrations = {
        "robotized-coil": make_registration(2),
        "manual-coil": make_registration(3),
    }
    navigation = types.SimpleNamespace(coil_registrations=registrations)
    tracker_coordinates = types.SimpleNamespace(GetCoordinates=lambda: None)
    robot = types.SimpleNamespace(
        navigation=navigation,
        tracker=types.SimpleNamespace(TrackerCoordinates=tracker_coordinates),
    )

    manager = object.__new__(robot_module.Robots)
    manager.robots_by_id = {0: robot}
    manager.robots_by_coil = {"robotized-coil": robot}
    manager._collision_monitor = None
    manager._collision_monitoring_active = False

    assert manager.StartCoilCollisionMonitoring()
    assert created_monitors[0].calculator.object_ids == (2, 3)
