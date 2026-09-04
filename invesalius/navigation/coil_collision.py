from dataclasses import dataclass

import numpy as np
from scipy.optimize import lsq_linear


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
