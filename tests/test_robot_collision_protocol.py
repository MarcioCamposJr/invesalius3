import importlib
import sys
import types

import invesalius.gui  # noqa: F401 - initialize package before stubbing its dialogs


def make_registration(object_id):
    return {
        "obj_id": object_id,
        "fiducials": [[-1, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 0]],
        "orientations": [[0, 0, 0]] * 4,
        "path": "not-needed-by-robot.stl",
        "tracker_id": 1,
    }


def load_robot_module(monkeypatch):
    monkeypatch.setitem(
        sys.modules, "invesalius.gui.dialogs", types.ModuleType("invesalius.gui.dialogs")
    )
    sys.modules.pop("invesalius.navigation.robot", None)
    return importlib.import_module("invesalius.navigation.robot")


def test_sends_both_collision_registrations_to_associated_robot(monkeypatch):
    robot_module = load_robot_module(monkeypatch)
    messages = []
    monkeypatch.setattr(
        robot_module.Publisher,
        "sendMessage",
        lambda topic, **data: messages.append((topic, data)),
    )

    registrations = {
        "robotized": make_registration(2),
        "manual": make_registration(3),
    }
    robot = object.__new__(robot_module.Robot)
    robot.robot_id = 0
    robot.coil_name = "robotized"
    robot.navigation = types.SimpleNamespace(coil_registrations=registrations)

    assert robot.SendCollisionRegistrations()
    assert messages == [
        (
            "Neuronavigation to Robot: Set coil collision registrations",
            {
                "registrations": {
                    coil_name: {
                        "obj_id": registration["obj_id"],
                        "fiducials": registration["fiducials"],
                        "orientations": registration["orientations"],
                    }
                    for coil_name, registration in registrations.items()
                },
                "coil_idx": 2,
                "robot_id": 0,
            },
        )
    ]


def test_does_not_send_until_two_coils_are_selected(monkeypatch):
    robot_module = load_robot_module(monkeypatch)
    messages = []
    monkeypatch.setattr(
        robot_module.Publisher,
        "sendMessage",
        lambda topic, **data: messages.append((topic, data)),
    )

    robot = object.__new__(robot_module.Robot)
    robot.robot_id = 0
    robot.coil_name = "robotized"
    robot.navigation = types.SimpleNamespace(
        coil_registrations={"robotized": make_registration(2)}
    )

    assert not robot.SendCollisionRegistrations()
    assert messages == []
