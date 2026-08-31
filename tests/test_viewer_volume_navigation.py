from unittest.mock import Mock

import pytest

import invesalius.data.slice_  # noqa: F401 - establishes the application's import order
from invesalius.data.viewer_volume import Viewer


class TargetNavigationViewer:
    _UpdateTargetCameraZoom = Viewer._UpdateTargetCameraZoom

    def __init__(self, camera):
        self.ren = Mock()
        self.ren.GetActiveCamera.return_value = camera
        self._target_camera_base_parallel_scale = 200.0
        self._target_camera_base_view_angle = 30.0


class TargetGuideViewer:
    _CreateTargetGuideArrow = Viewer._CreateTargetGuideArrow
    _UpdateTargetGuideArrow = staticmethod(Viewer._UpdateTargetGuideArrow)

    def __init__(self):
        self._target_guide_arrow_source = None


def test_target_camera_zoom_is_absolute_between_updates():
    camera = Mock()
    camera.GetParallelProjection.return_value = True
    viewer = TargetNavigationViewer(camera)

    viewer._UpdateTargetCameraZoom(1.0)
    viewer._UpdateTargetCameraZoom(100.0)

    camera.SetParallelScale.assert_any_call(40.0)
    camera.SetParallelScale.assert_called_with(200.0 / 1.0004)
    camera.Zoom.assert_not_called()


def test_target_guide_arrow_transform_preserves_actor_position():
    viewer = TargetGuideViewer()
    actor = viewer._CreateTargetGuideArrow([-55, -35, 0], [-55, -65, 0], (0, 1, 0))
    actor.SetPosition(0, -150, 0)
    actor.RotateZ(180)

    bounds = actor.GetBounds()
    center = tuple((bounds[index] + bounds[index + 1]) / 2.0 for index in range(0, 6, 2))

    assert center == pytest.approx((55.0, -100.0, 0.0))
