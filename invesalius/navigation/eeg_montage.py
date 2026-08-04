# --------------------------------------------------------------------------
# Software:     InVesalius - Software de Reconstrucao 3D de Imagens Medicas
# Copyright:    (C) 2001  Centro de Pesquisas Renato Archer
# Homepage:     http://www.softwarepublico.gov.br
# Contact:      invesalius@cti.gov.br
# License:      GNU - GPL 2 (LICENSE.txt/LICENCA.txt)
# --------------------------------------------------------------------------
#    Este programa e software livre; voce pode redistribui-lo e/ou
#    modifica-lo sob os termos da Licenca Publica Geral GNU, conforme
#    publicada pela Free Software Foundation; de acordo com a versao 2
#    da Licenca.
#
#    Este programa eh distribuido na expectativa de ser util, mas SEM
#    QUALQUER GARANTIA; sem mesmo a garantia implicita de
#    COMERCIALIZACAO ou de ADEQUACAO A QUALQUER PROPOSITO EM
#    PARTICULAR. Consulte a Licenca Publica Geral GNU para obter mais
#    detalhes.
# --------------------------------------------------------------------------

"""
EEG Montage Manager — Point Cloud Matching for HD electrode digitization.

Responsibility:
- Load EEG montage templates via MNE-Python.
- Store coordinates of anatomical fiducials (Nasion, LPA, RPA).
- Accumulate the anonymous point cloud captured by the tracker.
- Execute the ICP alignment (fiducial pre-alignment + refinement).
- Automatically label electrodes by minimum Euclidean distance.
- Export in EEG-BIDS format.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from invesalius.utils import Singleton


class DigitizationState(Enum):
    """State machine for the digitization workflow."""

    IDLE = "idle"
    TEMPLATE_SELECTED = "template_selected"
    FIDUCIALS_REGISTERED = "fiducials_registered"
    CAPTURING_POINTS = "capturing_points"
    MATCHING_DONE = "matching_done"
    EXPORTED = "exported"


class ConfidenceLevel(Enum):
    """Confidence levels based on the distance between captured point and template point."""

    HIGH = "high"  # green — dist < 5mm
    MEDIUM = "medium"  # yellow — 5mm <= dist < 10mm
    LOW = "low"  # red — dist >= 10mm


@dataclass
class LabeledElectrode:
    """Result of the matching: an electrode with an assigned label and metrics."""

    label: str  # e.g., "Cz", "Fp1"
    position_inv: np.ndarray  # coordinates in InVesalius space (3,)
    position_world: np.ndarray  # coordinates in Scanner RAS space (3,)
    template_position: np.ndarray  # expected position from template (3,)
    distance_mm: float  # Euclidean distance between point and template
    confidence: ConfidenceLevel
    manually_corrected: bool = False


class EEGMontage(metaclass=Singleton):
    def __init__(self) -> None:
        self.state: DigitizationState = DigitizationState.IDLE

        # Template data
        self.template_name: Optional[str] = None
        self.template_positions: Optional[np.ndarray] = None  # (N, 3)
        self.template_labels: Optional[List[str]] = None  # N labels

        # Fiducials (3 points: Nasion, LPA, RPA) in tracker->InVesalius space
        self.fiducials_inv: Dict[str, Optional[np.ndarray]] = {
            "nasion": None,
            "lpa": None,
            "rpa": None,
        }

        # Captured point cloud (M anonymous points)
        self.point_cloud: List[np.ndarray] = []

        # Matching results
        self.labeled_electrodes: List[LabeledElectrode] = []
        self.icp_transform: Optional[np.ndarray] = None  # (4,4) affine matrix
        self.mean_error_mm: Optional[float] = None

    def reset(self) -> None:
        """Reset the state for a new digitization session."""
        self.__init__()

    # --- Template Loading (via MNE) ---

    @staticmethod
    def get_available_templates() -> List[str]:
        """Returns a list of available EEG montage templates from MNE."""
        try:
            import mne

            builtin = mne.channels.get_builtin_montages()
            return sorted(builtin)
        except ImportError:
            # Fallback if MNE is not installed
            return ["standard_1020", "standard_1005"]

    def load_template(self, template_name: str) -> None:
        """Load an EEG montage template via MNE and extract its 3D positions."""
        import mne

        # Load standard montage
        montage = mne.channels.make_standard_montage(template_name)

        # Positions in meters (MNE default), convert to mm
        positions_dict = montage.get_positions()
        ch_pos = positions_dict["ch_pos"]  # dict: label -> (3,) array in meters

        self.template_labels = list(ch_pos.keys())
        self.template_positions = np.array(list(ch_pos.values())) * 1000  # meters to mm
        self.template_name = template_name

        # Extract fiducials from the template (in MNE frame)
        self._template_fiducials = {
            "nasion": positions_dict.get("nasion"),
            "lpa": positions_dict.get("lpa"),
            "rpa": positions_dict.get("rpa"),
        }

        # Convert fiducials to mm
        for key in self._template_fiducials:
            if self._template_fiducials[key] is not None:
                self._template_fiducials[key] = self._template_fiducials[key] * 1000

        self.state = DigitizationState.TEMPLATE_SELECTED

    # --- Point Cloud Capture ---

    def add_point(self, position: np.ndarray) -> int:
        """Add a point to the point cloud. Returns the index of the point."""
        self.point_cloud.append(np.array(position[:3]))
        if self.state == DigitizationState.FIDUCIALS_REGISTERED:
            self.state = DigitizationState.CAPTURING_POINTS
        return len(self.point_cloud) - 1

    def remove_last_point(self) -> bool:
        """Remove the last captured point."""
        if self.point_cloud:
            self.point_cloud.pop()
            return True
        return False

    def remove_point(self, index: int) -> bool:
        """Remove a point at a specific index."""
        if 0 <= index < len(self.point_cloud):
            self.point_cloud.pop(index)
            return True
        return False

    def get_point_cloud_array(self) -> np.ndarray:
        """Return the point cloud as an (M, 3) numpy array."""
        if not self.point_cloud:
            return np.empty((0, 3))
        return np.vstack(self.point_cloud)

    # --- Outlier and Duplicate Filtering ---

    def filter_outliers(self, std_factor: float = 2.5) -> List[int]:
        """
        Remove outlier points (accidental clicks outside the head).
        Uses distance to centroid with a threshold based on standard deviation.
        Returns the indices of the removed points.
        """
        if len(self.point_cloud) < 4:
            return []

        points = self.get_point_cloud_array()
        centroid = points.mean(axis=0)
        distances = np.linalg.norm(points - centroid, axis=1)

        mean_dist = distances.mean()
        std_dist = distances.std()
        threshold = mean_dist + std_factor * std_dist

        outlier_mask = distances > threshold
        outlier_indices = list(np.where(outlier_mask)[0])

        # Remove in reverse order to preserve indices
        for idx in sorted(outlier_indices, reverse=True):
            self.point_cloud.pop(idx)

        return outlier_indices

    def filter_duplicates(self, min_distance_mm: float = 3.0) -> int:
        """
        Remove duplicate points (points that are too close to each other).
        Returns the number of removed points.
        """
        if len(self.point_cloud) < 2:
            return 0

        from scipy.spatial.distance import pdist, squareform

        points = self.get_point_cloud_array()
        dist_matrix = squareform(pdist(points))

        to_remove = set()
        for i in range(len(points)):
            if i in to_remove:
                continue
            for j in range(i + 1, len(points)):
                if j in to_remove:
                    continue
                if dist_matrix[i, j] < min_distance_mm:
                    to_remove.add(j)

        for idx in sorted(to_remove, reverse=True):
            self.point_cloud.pop(idx)

        return len(to_remove)

    # --- Fiducial Registration and ICP ---

    def set_fiducial(self, name: str, position: np.ndarray) -> None:
        """Register an anatomical fiducial. name: 'nasion', 'lpa', 'rpa'."""
        assert name in self.fiducials_inv
        self.fiducials_inv[name] = np.array(position[:3])

        if self.are_fiducials_set() and self.state == DigitizationState.TEMPLATE_SELECTED:
            self.state = DigitizationState.FIDUCIALS_REGISTERED

    def are_fiducials_set(self) -> bool:
        """Return True if all three fiducials are set."""
        return all(v is not None for v in self.fiducials_inv.values())

    def compute_fiducial_alignment(self) -> np.ndarray:
        """
        Compute initial rigid alignment using the 3 fiducials.
        Calculates transformation from MNE template space to InVesalius space.
        Returns a 4x4 transformation matrix.
        """
        from invesalius.data import transformations as tr

        # Source points: template fiducials
        src = np.array(
            [
                self._template_fiducials["lpa"],
                self._template_fiducials["rpa"],
                self._template_fiducials["nasion"],
            ]
        )  # (3, 3)

        # Target points: captured fiducials
        dst = np.array(
            [
                self.fiducials_inv["lpa"],
                self.fiducials_inv["rpa"],
                self.fiducials_inv["nasion"],
            ]
        )  # (3, 3)

        # affine_matrix_from_points(v0, v1) returns matrix mapping v0 to v1
        m_fiducial = tr.affine_matrix_from_points(src.T, dst.T, shear=False, scale=False)

        return m_fiducial

    def apply_transform_to_template(self, transform: np.ndarray) -> np.ndarray:
        """Apply a 4x4 transformation matrix to template positions."""
        positions_h = np.hstack(
            [self.template_positions, np.ones((len(self.template_positions), 1))]
        )  # (N, 4)

        transformed = (transform @ positions_h.T).T[:, :3]  # (N, 3)
        return transformed

    def _run_vtk_icp(self, source_points: np.ndarray, target_points: np.ndarray) -> np.ndarray:
        """
        Run ICP using VTK.
        Source = template (to be moved), Target = captured cloud (fixed).
        """
        from vtkmodules.vtkCommonCore import vtkPoints
        from vtkmodules.vtkCommonDataModel import vtkIterativeClosestPointTransform, vtkPolyData

        src_vtk = vtkPoints()
        for pt in source_points:
            src_vtk.InsertNextPoint(pt)
        src_poly = vtkPolyData()
        src_poly.SetPoints(src_vtk)

        tgt_vtk = vtkPoints()
        for pt in target_points:
            tgt_vtk.InsertNextPoint(pt)
        tgt_poly = vtkPolyData()
        tgt_poly.SetPoints(tgt_vtk)

        icp = vtkIterativeClosestPointTransform()
        icp.SetSource(src_poly)
        icp.SetTarget(tgt_poly)
        icp.GetLandmarkTransform().SetModeToRigidBody()
        icp.SetMaximumNumberOfIterations(500)
        icp.SetMaximumNumberOfLandmarks(len(source_points))
        icp.Modified()
        icp.Update()

        m = np.eye(4)
        vtk_matrix = icp.GetMatrix()
        for i in range(4):
            for j in range(4):
                m[i, j] = vtk_matrix.GetElement(i, j)

        return m

    # --- Matching Pipeline and Labeling ---

    def run_icp_matching(
        self, progress_callback: Optional[Callable[[int, str], None]] = None
    ) -> Tuple[float, List[LabeledElectrode]]:
        """
        Complete Point Cloud Matching pipeline:
        1. Filter duplicates and outliers
        2. Initial alignment via fiducials (Nasion/LPA/RPA)
        3. ICP refinement (template -> point cloud)
        4. Labeling via minimum Euclidean distance (Hungarian algorithm)

        Returns: (mean_error_mm, list of LabeledElectrode)
        """
        from scipy.optimize import linear_sum_assignment
        from scipy.spatial.distance import cdist

        from invesalius.data import imagedata_utils

        if not self.are_fiducials_set():
            raise ValueError("Fiducials are not registered")

        if len(self.point_cloud) < 3:
            raise ValueError("Insufficient point cloud (minimum 3 points)")

        # Step 1: Filter
        if progress_callback:
            progress_callback(0, _("Filtering points..."))
        self.filter_duplicates()
        self.filter_outliers()

        # Step 2: Fiducial alignment
        if progress_callback:
            progress_callback(1, _("Initial fiducial alignment..."))
        m_fiducial = self.compute_fiducial_alignment()
        template_aligned = self.apply_transform_to_template(m_fiducial)

        # Step 3: ICP refinement
        if progress_callback:
            progress_callback(2, _("ICP refinement..."))
        m_icp = self._run_vtk_icp(
            source_points=template_aligned,
            target_points=self.get_point_cloud_array(),
        )

        # Total transform
        self.icp_transform = m_icp @ m_fiducial
        template_final = self.apply_transform_to_template(self.icp_transform)

        # Step 4: Labeling (Hungarian assignment)
        if progress_callback:
            progress_callback(3, _("Labeling assignments..."))
        point_cloud = self.get_point_cloud_array()
        cost_matrix = cdist(point_cloud, template_final)  # (M, N)

        if len(point_cloud) <= len(template_final):
            row_ind, col_ind = linear_sum_assignment(cost_matrix)
        else:
            # More points than channels, transpose
            row_ind_t, col_ind_t = linear_sum_assignment(cost_matrix.T)
            row_ind, col_ind = col_ind_t, row_ind_t

        self.labeled_electrodes = []
        for pt_idx, tmpl_idx in zip(row_ind, col_ind):
            dist = cost_matrix[pt_idx, tmpl_idx]

            # Confidence thresholds
            if dist < 5.0:
                confidence = ConfidenceLevel.HIGH
            elif dist < 10.0:
                confidence = ConfidenceLevel.MEDIUM
            else:
                confidence = ConfidenceLevel.LOW

            pos_inv = point_cloud[pt_idx]
            # Convert to world (Scanner RAS)
            pos_world, _ori = imagedata_utils.convert_invesalius_to_world(
                position=list(pos_inv), orientation=[0, 0, 0]
            )
            pos_world = np.array(pos_world) if pos_world[0] is not None else pos_inv

            electrode = LabeledElectrode(
                label=self.template_labels[tmpl_idx],
                position_inv=pos_inv,
                position_world=pos_world,
                template_position=template_final[tmpl_idx],
                distance_mm=float(dist),
                confidence=confidence,
            )
            self.labeled_electrodes.append(electrode)

        self.mean_error_mm = float(np.mean([e.distance_mm for e in self.labeled_electrodes]))
        self.state = DigitizationState.MATCHING_DONE

        return self.mean_error_mm, self.labeled_electrodes

    # --- Manual Correction ---

    def swap_labels(self, label_a: str, label_b: str) -> bool:
        """Swap the labels between two electrodes."""
        elec_a = next((e for e in self.labeled_electrodes if e.label == label_a), None)
        elec_b = next((e for e in self.labeled_electrodes if e.label == label_b), None)
        if elec_a is None or elec_b is None:
            return False
        elec_a.label, elec_b.label = elec_b.label, elec_a.label
        elec_a.manually_corrected = True
        elec_b.manually_corrected = True
        return True

    def reassign_label(self, old_label: str, new_label: str) -> bool:
        """Reassign the label of an electrode."""
        elec = next((e for e in self.labeled_electrodes if e.label == old_label), None)
        if elec is None:
            return False
        elec.label = new_label
        elec.manually_corrected = True
        return True

    # --- Marker Integration ---

    def create_markers(self) -> List[int]:
        """
        Instantiate Marker objects for each labeled electrode and add them to MarkersControl.
        Returns the list of marker IDs created.
        """
        from invesalius.data.markers.marker import Marker, MarkerType
        from invesalius.navigation.markers import MarkersControl

        marker_ids = []
        markers_control = MarkersControl()

        for elec in self.labeled_electrodes:
            marker = Marker()
            marker.name = elec.label
            marker.marker_type = MarkerType.EEG_ELECTRODE
            # Store in InVesalius space
            marker.coord = np.array(elec.position_inv)
            # Default orientation
            marker.orient = np.array([0, 0, 0])

            # Map confidence to color (R, G, B in 0-1 range)
            if elec.confidence == ConfidenceLevel.HIGH:
                marker.colour = (0.0, 1.0, 0.0)
            elif elec.confidence == ConfidenceLevel.MEDIUM:
                marker.colour = (1.0, 1.0, 0.0)
            else:
                marker.colour = (1.0, 0.0, 0.0)

            # Add to MarkersControl
            marker_id = markers_control.AddMarker(marker)
            marker_ids.append(marker_id)

        return marker_ids

    # --- BIDS Export ---

    def export_bids(self, output_dir: str, subject_id: str = "01") -> Tuple[str, str]:
        """
        Export in EEG-BIDS format:
        - electrodes.tsv: name, x, y, z (Scanner RAS in mm)
        - coordsystem.json: coordinate system metadata

        Returns: tuple (path_electrodes, path_coordsystem)
        """
        import json
        import os

        import pandas as pd

        from invesalius.data import imagedata_utils

        os.makedirs(output_dir, exist_ok=True)

        # electrodes.tsv
        rows = []
        for elec in sorted(self.labeled_electrodes, key=lambda e: e.label):
            rows.append(
                {
                    "name": elec.label,
                    "x": round(float(elec.position_world[0]), 2),
                    "y": round(float(elec.position_world[1]), 2),
                    "z": round(float(elec.position_world[2]), 2),
                }
            )

        df = pd.DataFrame(rows)
        electrodes_path = os.path.join(output_dir, f"sub-{subject_id}_electrodes.tsv")
        df.to_csv(electrodes_path, sep="\t", index=False)

        # coordsystem.json
        coordsystem = {
            "EEGCoordinateSystem": "Other",
            "EEGCoordinateUnits": "mm",
            "EEGCoordinateSystemDescription": (
                "Scanner RAS coordinate system derived from the subject MRI affine transformation."
            ),
            "IntendedFor": "",
            "AnatomicalLandmarkCoordinateSystem": "Other",
            "AnatomicalLandmarkCoordinateUnits": "mm",
            "AnatomicalLandmarkCoordinates": {},
            "DigitizationMethod": "InVesalius Navigator - Point Cloud ICP Matching",
            "DigitizationTemplate": self.template_name or "unknown",
            "ICPMeanErrorMM": self.mean_error_mm,
        }

        # Add fiducials to coordsystem
        for fid_name, fid_pos in self.fiducials_inv.items():
            if fid_pos is not None:
                pos_world, _ori = imagedata_utils.convert_invesalius_to_world(
                    position=list(fid_pos), orientation=[0, 0, 0]
                )
                if pos_world[0] is not None:
                    key = fid_name.upper()
                    coordsystem["AnatomicalLandmarkCoordinates"][key] = {
                        "x": round(float(pos_world[0]), 2),
                        "y": round(float(pos_world[1]), 2),
                        "z": round(float(pos_world[2]), 2),
                    }

        coordsystem_path = os.path.join(output_dir, f"sub-{subject_id}_coordsystem.json")
        with open(coordsystem_path, "w") as f:
            json.dump(coordsystem, f, indent=2)

        self.state = DigitizationState.EXPORTED
        return electrodes_path, coordsystem_path
