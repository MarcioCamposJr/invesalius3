from unittest.mock import Mock

import invesalius.data.slice_  # noqa: F401 - establishes the application's import order
from invesalius.data.viewer_volume import Viewer


class TargetNavigationViewer:
    _UpdateTargetCameraZoom = Viewer._UpdateTargetCameraZoom

    def __init__(self, camera):
        self.ren = Mock()
        self.ren.GetActiveCamera.return_value = camera
        self._target_camera_base_parallel_scale = 200.0
        self._target_camera_base_view_angle = 30.0


def test_target_camera_zoom_is_absolute_between_updates():
    camera = Mock()
    camera.GetParallelProjection.return_value = True
    viewer = TargetNavigationViewer(camera)

    viewer._UpdateTargetCameraZoom(1.0)
    viewer._UpdateTargetCameraZoom(100.0)

    camera.SetParallelScale.assert_any_call(40.0)
    camera.SetParallelScale.assert_called_with(200.0 / 1.0004)
    camera.Zoom.assert_not_called()
