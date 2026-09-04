import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from invesalius.navigation.coil_collision import (
    OrientedBoundingBox,
    coil_box_from_registration,
    measure_obb_distance,
)


def make_box(center, half_sizes=(1, 1, 1), rotation=None):
    if rotation is None:
        rotation = np.eye(3)
    half_axes = (rotation @ np.diag(half_sizes)).T
    return OrientedBoundingBox(np.asarray(center, dtype=float), half_axes)


def test_measures_axis_aligned_box_distance_and_directions():
    box_a = make_box([0, 0, 0])
    box_b = make_box([3, 0, 0])

    result = measure_obb_distance(box_a, box_b)

    assert result.distance == pytest.approx(1)
    np.testing.assert_allclose(result.closest_point_a, [1, 0, 0], atol=1e-7)
    np.testing.assert_allclose(result.closest_point_b, [2, 0, 0], atol=1e-7)
    np.testing.assert_allclose(result.brake_direction_a, [-1, 0, 0], atol=1e-7)
    np.testing.assert_allclose(result.brake_direction_b, [1, 0, 0], atol=1e-7)


def test_measures_diagonal_distance():
    box_a = make_box([0, 0, 0])
    box_b = make_box([3, 4, 0])

    result = measure_obb_distance(box_a, box_b)

    assert result.distance == pytest.approx(np.sqrt(5))


def test_returns_zero_for_intersecting_boxes():
    box_a = make_box([0, 0, 0])
    box_b = make_box([1, 0, 0])

    result = measure_obb_distance(box_a, box_b)

    assert result.distance == 0
    np.testing.assert_array_equal(result.brake_direction_a, np.zeros(3))
    np.testing.assert_array_equal(result.brake_direction_b, np.zeros(3))


def test_supports_rotated_boxes():
    rotation = Rotation.from_euler("z", 45, degrees=True).as_matrix()
    box_a = make_box([0, 0, 0], half_sizes=(2, 1, 1), rotation=rotation)
    box_b = make_box([5, 0, 0])

    result = measure_obb_distance(box_a, box_b)

    rotated_extent_x = 2 * np.cos(np.radians(45)) + np.sin(np.radians(45))
    expected = 5 - rotated_extent_x - 1
    assert result.distance == pytest.approx(expected)


def test_transforms_local_box_to_world_coordinates():
    local = make_box([1, 0, 0], half_sizes=(2, 1, 0.5))
    rotation = Rotation.from_euler("z", 90, degrees=True).as_matrix()

    world = local.transformed([10, 20, 30], rotation)

    np.testing.assert_allclose(world.center, [10, 21, 30], atol=1e-12)
    np.testing.assert_allclose(world.half_axes[0], [0, 2, 0], atol=1e-12)


def test_rejects_degenerate_box():
    with pytest.raises(ValueError):
        OrientedBoundingBox(np.zeros(3), np.zeros((3, 3)))


def test_builds_marker_local_box_from_coil_registration():
    registration = {
        "fiducials": [
            [-20, 0, 0],
            [20, 0, 0],
            [0, 30, 0],
            [10, 20, 30],
        ],
        "orientations": [[0, 0, 0]] * 4,
    }

    box = coil_box_from_registration(registration, half_thickness=5)

    np.testing.assert_allclose(box.center, [-10, -20, -30])
    np.testing.assert_allclose(box.half_axes, [[20, 0, 0], [0, 30, 0], [0, 0, 5]])


def test_reconstructs_registered_box_at_initial_marker_pose():
    registration = {
        "fiducials": [
            [10, 18, 30],
            [10, 22, 30],
            [7, 20, 30],
            [10, 20, 30],
        ],
        "orientations": [[0, 0, 0]] * 3 + [[90, 0, 0]],
    }
    local_box = coil_box_from_registration(registration, half_thickness=1)
    marker_rotation = Rotation.from_euler("ZYX", [90, 0, 0], degrees=True).as_matrix()

    reconstructed = local_box.transformed([10, 20, 30], marker_rotation)

    np.testing.assert_allclose(reconstructed.center, [10, 20, 30], atol=1e-12)
    np.testing.assert_allclose(reconstructed.half_axes[0], [0, 2, 0], atol=1e-12)
    np.testing.assert_allclose(reconstructed.half_axes[1], [-3, 0, 0], atol=1e-12)


@pytest.mark.parametrize(
    "registration",
    [
        {},
        {"fiducials": [], "orientations": []},
        {
            "fiducials": [[0, 0, 0]] * 4,
            "orientations": [[0, 0, 0]] * 4,
        },
    ],
)
def test_rejects_invalid_coil_registration(registration):
    with pytest.raises(ValueError):
        coil_box_from_registration(registration)
