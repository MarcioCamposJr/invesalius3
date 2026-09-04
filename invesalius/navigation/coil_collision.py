from dataclasses import dataclass

import numpy as np
from scipy.optimize import lsq_linear
from scipy.spatial.transform import Rotation

DEFAULT_COIL_HALF_THICKNESS_MM = 7.0


@dataclass(frozen=True)
class OrientedBoundingBox:
    """An oriented box represented by its center and three half-axis vectors."""

    center: np.ndarray
    half_axes: np.ndarray

    def __post_init__(self):
        center = np.asarray(self.center, dtype=float)
        half_axes = np.asarray(self.half_axes, dtype=float)

        if center.shape != (3,):
            raise ValueError("OBB center must contain three values")
        if half_axes.shape != (3, 3):
            raise ValueError("OBB half_axes must have shape (3, 3)")
        if not np.all(np.isfinite(center)) or not np.all(np.isfinite(half_axes)):
            raise ValueError("OBB values must be finite")
        if abs(np.linalg.det(half_axes)) <= 1e-9:
            raise ValueError("OBB half-axes must define a non-degenerate volume")

        object.__setattr__(self, "center", center.copy())
        object.__setattr__(self, "half_axes", half_axes.copy())

    def transformed(self, position, rotation) -> "OrientedBoundingBox":
        position = np.asarray(position, dtype=float)
        rotation = np.asarray(rotation, dtype=float)
        if position.shape != (3,) or rotation.shape != (3, 3):
            raise ValueError("Expected a 3D position and a 3x3 rotation")
        if not np.all(np.isfinite(position)) or not np.all(np.isfinite(rotation)):
            raise ValueError("OBB transformation values must be finite")

        return OrientedBoundingBox(
            center=rotation @ self.center + position,
            half_axes=(rotation @ self.half_axes.T).T,
        )


@dataclass(frozen=True)
class CoilCollisionMeasurement:
    distance: float
    closest_point_a: np.ndarray
    closest_point_b: np.ndarray
    brake_direction_a: np.ndarray
    brake_direction_b: np.ndarray


@dataclass(frozen=True)
class NamedCoilCollisionMeasurement:
    coil_a: str
    coil_b: str
    measurement: CoilCollisionMeasurement


class CoilCollisionCalculator:
    """Calculate collision data for two registered tracker coils."""

    def __init__(self, registrations, half_thickness=DEFAULT_COIL_HALF_THICKNESS_MM):
        if not isinstance(registrations, dict) or len(registrations) != 2:
            raise ValueError("Collision calculation requires exactly two coils")

        self._coils = []
        object_ids = set()
        for coil_name, registration in registrations.items():
            object_id = registration.get("obj_id")
            if not isinstance(object_id, int) or object_id < 0:
                raise ValueError(f"Coil {coil_name} has an invalid tracker object ID")
            if object_id in object_ids:
                raise ValueError("Collision coils must use distinct tracker object IDs")
            object_ids.add(object_id)
            self._coils.append(
                (
                    coil_name,
                    object_id,
                    coil_box_from_registration(registration, half_thickness),
                )
            )

    @property
    def object_ids(self):
        return tuple(coil[1] for coil in self._coils)

    def measure(self, tracker_coordinates) -> NamedCoilCollisionMeasurement:
        world_boxes = []
        for coil_name, object_id, local_box in self._coils:
            try:
                pose = np.asarray(tracker_coordinates[object_id], dtype=float)
            except (IndexError, TypeError, ValueError) as error:
                raise ValueError(f"Tracker pose for coil {coil_name} is unavailable") from error

            if pose.ndim != 1 or pose.size < 6 or not np.all(np.isfinite(pose[:6])):
                raise ValueError(f"Tracker pose for coil {coil_name} is invalid")

            rotation = Rotation.from_euler("ZYX", pose[3:6], degrees=True).as_matrix()
            world_boxes.append(local_box.transformed(pose[:3], rotation))

        measurement = measure_obb_distance(world_boxes[0], world_boxes[1])
        return NamedCoilCollisionMeasurement(
            coil_a=self._coils[0][0],
            coil_b=self._coils[1][0],
            measurement=measurement,
        )


def coil_box_from_registration(
    registration, half_thickness=DEFAULT_COIL_HALF_THICKNESS_MM
) -> OrientedBoundingBox:
    """Build a marker-local coil OBB from an InVesalius coil registration."""

    if not np.isfinite(half_thickness) or half_thickness <= 0:
        raise ValueError("Coil half-thickness must be a positive finite value")
    if not isinstance(registration, dict):
        raise ValueError("Coil registration must be a dictionary")

    try:
        fiducials = np.asarray(registration["fiducials"], dtype=float)
        orientations = np.asarray(registration["orientations"], dtype=float)
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("Coil registration has invalid fiducials or orientations") from error

    if fiducials.ndim != 2 or fiducials.shape[0] < 4 or fiducials.shape[1] < 3:
        raise ValueError("Coil registration must contain four 3D fiducials")
    if orientations.ndim != 2 or orientations.shape[0] < 4 or orientations.shape[1] < 3:
        raise ValueError("Coil registration must contain four orientations")
    if not np.all(np.isfinite(fiducials)) or not np.all(np.isfinite(orientations)):
        raise ValueError("Coil registration values must be finite")

    left, right, anterior, marker_position = fiducials[:4, :3]
    marker_orientation = orientations[3, :3]
    center = (left + right) / 2
    half_width = (right - left) / 2
    half_depth = anterior - center

    normal = np.cross(half_width, half_depth)
    normal_norm = np.linalg.norm(normal)
    if normal_norm <= 1e-9:
        raise ValueError("Coil registration fiducials must not be collinear")
    half_normal = normal / normal_norm * half_thickness

    marker_to_tracker = Rotation.from_euler("ZYX", marker_orientation, degrees=True).as_matrix()
    tracker_to_marker = marker_to_tracker.T

    return OrientedBoundingBox(
        center=tracker_to_marker @ (center - marker_position),
        half_axes=np.array(
            [
                tracker_to_marker @ half_width,
                tracker_to_marker @ half_depth,
                tracker_to_marker @ half_normal,
            ]
        ),
    )


def measure_obb_distance(
    box_a: OrientedBoundingBox, box_b: OrientedBoundingBox
) -> CoilCollisionMeasurement:
    """Return the minimum surface distance and separation directions for two OBBs."""

    # A point inside each box is center + half_axes.T @ coefficients, where
    # every coefficient is constrained to [-1, 1]. Finding the two closest
    # points is therefore a bounded linear least-squares problem with 6 vars.
    matrix = np.column_stack((box_a.half_axes.T, -box_b.half_axes.T))
    target = box_b.center - box_a.center
    solution = lsq_linear(matrix, target, bounds=(-1.0, 1.0), tol=1e-12)
    if not solution.success:
        raise RuntimeError(f"Unable to calculate OBB distance: {solution.message}")

    coefficients_a = solution.x[:3]
    coefficients_b = solution.x[3:]
    point_a = box_a.center + box_a.half_axes.T @ coefficients_a
    point_b = box_b.center + box_b.half_axes.T @ coefficients_b

    separation = point_a - point_b
    distance = float(np.linalg.norm(separation))
    if distance > 1e-9:
        direction_a = separation / distance
        direction_b = -direction_a
    else:
        # A direction is unnecessary while the boxes intersect: the robot-side
        # controller treats zero distance as an immediate stop.
        direction_a = np.zeros(3, dtype=float)
        direction_b = np.zeros(3, dtype=float)
        distance = 0.0

    return CoilCollisionMeasurement(
        distance=distance,
        closest_point_a=point_a,
        closest_point_b=point_b,
        brake_direction_a=direction_a,
        brake_direction_b=direction_b,
    )
