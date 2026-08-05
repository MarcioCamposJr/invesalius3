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

from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum

import numpy as np
import wx

from invesalius.utils import Singleton

_ = wx.GetTranslation


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
        self.template_name: str | None = None
        self.template_positions: np.ndarray | None = None  # (N, 3)
        self.template_labels: list[str] | None = None  # N labels

        # Fiducials (3 points: Nasion, LPA, RPA) in tracker->InVesalius space
        self.fiducials_inv: dict[str, np.ndarray | None] = {
            "nasion": None,
            "lpa": None,
            "rpa": None,
        }

        # Captured point cloud (M anonymous points)
        self.point_cloud: list[np.ndarray] = []

        # Matching results
        self.labeled_electrodes: list[LabeledElectrode] = []
        self.icp_transform: np.ndarray | None = None  # (4,4) affine matrix
        self.head_scale_factor: float = 1.0  # captured/template head size ratio
        self.mean_error_mm: float | None = None
        self.show_electrodes: bool = True

        from pubsub import pub as Publisher

        Publisher.subscribe(self._on_enable_state_project, "Enable state project")

    def _on_enable_state_project(self, state: bool) -> None:
        if state:
            self.LoadState()
            self._broadcast_electrodes()

    def _broadcast_electrodes(self) -> None:
        from pubsub import pub as Publisher

        has_matches = bool(self.labeled_electrodes)
        eeg_data = []

        for i, coord in enumerate(self.point_cloud):
            name = f"E{i + 1}"
            color = (0.5, 0.5, 0.5)

            if has_matches and i < len(self.labeled_electrodes):
                res = self.labeled_electrodes[i]
                name = res.label
                if res.confidence.value == "high":
                    color = (0.0, 1.0, 0.0)
                elif res.confidence.value == "medium":
                    color = (1.0, 1.0, 0.0)
                elif res.confidence.value == "low":
                    color = (1.0, 0.0, 0.0)

            # Convert to VTK space (negate Y)
            vtk_coord = list(coord)
            vtk_coord[1] = -vtk_coord[1]

            eeg_data.append(
                {
                    "name": name,
                    "position": vtk_coord,
                    "normal": [0.0, 0.0, 1.0],  # Default normal, re-projected when dialog opens
                    "color": color,
                }
            )

        Publisher.sendMessage(
            "Update EEG electrodes", electrodes_data=eeg_data, show=self.show_electrodes
        )

    def reset(self) -> None:
        """Reset the state for a new digitization session."""
        self.__init__()

    def SaveState(self) -> None:
        import invesalius.session as ses

        state = {
            "template_name": self.template_name,
            "show_electrodes": self.show_electrodes,
            "point_cloud": [p.tolist() for p in self.point_cloud],
            "labeled_electrodes": [
                {
                    "label": elec.label,
                    "position_inv": elec.position_inv.tolist(),
                    "position_world": elec.position_world.tolist(),
                    "template_position": elec.template_position.tolist(),
                    "distance_mm": elec.distance_mm,
                    "confidence": elec.confidence.value,
                }
                for elec in self.labeled_electrodes
            ],
        }
        ses.Session().SetState("eeg_montage", state)

    def LoadState(self) -> None:
        import invesalius.session as ses

        state = ses.Session().GetState("eeg_montage")
        if not state:
            return

        self.template_name = state.get("template_name", "standard_1020")
        self.show_electrodes = state.get("show_electrodes", True)
        self.point_cloud = [np.array(p) for p in state.get("point_cloud", [])]

        if self.template_name:
            self.load_template(self.template_name)

        elec_data = state.get("labeled_electrodes", state.get("matched_labels", []))
        if elec_data:
            self.labeled_electrodes = []
            _zero = np.zeros(3)
            for m in elec_data:
                conf = ConfidenceLevel(m["confidence"])

                pos_inv = np.array(m.get("position_inv", _zero))
                pos_world = np.array(m.get("position_world", _zero))
                tmpl_pos = np.array(m.get("template_position", _zero))

                elec = LabeledElectrode(
                    label=m["label"],
                    position_inv=pos_inv,
                    position_world=pos_world,
                    template_position=tmpl_pos,
                    distance_mm=m["distance_mm"],
                    confidence=conf,
                )
                self.labeled_electrodes.append(elec)

    # --- Template Loading (via MNE) ---

    @staticmethod
    def get_available_templates() -> list[str]:
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

        # MNE is RAS (Y=Anterior). InVesalius is RPS (Y=Posterior). Invert Y to match InVesalius space.
        self.template_positions[:, 1] = -self.template_positions[:, 1]

        self.template_name = template_name

        # Extract fiducials from the template (in MNE frame)
        self._template_fiducials = {
            "nasion": positions_dict.get("nasion"),
            "lpa": positions_dict.get("lpa"),
            "rpa": positions_dict.get("rpa"),
        }

        # Convert fiducials to mm and invert Y
        for key in self._template_fiducials:
            if self._template_fiducials[key] is not None:
                self._template_fiducials[key] = self._template_fiducials[key] * 1000
                self._template_fiducials[key][1] = -self._template_fiducials[key][1]

        self.state = DigitizationState.TEMPLATE_SELECTED

    # --- Point Cloud Capture ---

    def add_point(self, position: np.ndarray) -> int:
        """Add a point to the point cloud. Returns the index of the point."""
        self.point_cloud.append(np.array(position[:3]))
        if self.state == DigitizationState.FIDUCIALS_REGISTERED:
            self.state = DigitizationState.CAPTURING_POINTS
        self.SaveState()
        return len(self.point_cloud) - 1

    def remove_last_point(self) -> bool:
        """Remove the last captured point."""
        if self.point_cloud:
            self.point_cloud.pop()
            self.SaveState()
            return True
        return False

    def remove_point(self, index: int) -> bool:
        """Remove a point at a specific index."""
        if 0 <= index < len(self.point_cloud):
            self.point_cloud.pop(index)
            self.SaveState()
            return True
        return False

    def get_point_cloud_array(self) -> np.ndarray:
        """Return the point cloud as an (M, 3) numpy array."""
        if not self.point_cloud:
            return np.empty((0, 3))
        return np.vstack(self.point_cloud)

    # --- Outlier and Duplicate Filtering ---

    def filter_outliers(self, std_factor: float = 2.5) -> list[int]:
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
        Includes uniform scaling to compensate for head size differences.
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

        # Compute uniform scale factor from inter-fiducial distances
        self.head_scale_factor = self._compute_scale_factor(src, dst)

        # affine_matrix_from_points with scale=True computes rigid + uniform scale
        # This compensates for head size differences between template and patient
        m_fiducial = tr.affine_matrix_from_points(src.T, dst.T, shear=False, scale=True)

        return m_fiducial

    @staticmethod
    def _compute_scale_factor(template_fids: np.ndarray, captured_fids: np.ndarray) -> float:
        """
        Compute uniform scale factor from inter-fiducial distances.
        Compares LPA-RPA, LPA-Nasion, RPA-Nasion distances between template
        and captured fiducials to determine the head size ratio.

        Returns: scale factor (captured_size / template_size).
        Values > 1.0 mean the patient's head is larger than the template.
        """
        # Inter-fiducial distance pairs: (LPA-RPA), (LPA-Nasion), (RPA-Nasion)
        pairs = [(0, 1), (0, 2), (1, 2)]

        template_dists = [np.linalg.norm(template_fids[i] - template_fids[j]) for i, j in pairs]
        captured_dists = [np.linalg.norm(captured_fids[i] - captured_fids[j]) for i, j in pairs]

        # Ratio of each pair, then take the mean for a robust estimate
        ratios = [c / t for c, t in zip(captured_dists, template_dists) if t > 0]
        return float(np.mean(ratios)) if ratios else 1.0

    def apply_transform_to_template(self, transform: np.ndarray) -> np.ndarray:
        """Apply a 4x4 transformation matrix to template positions."""
        positions_h = np.hstack(
            [self.template_positions, np.ones((len(self.template_positions), 1))]
        )  # (N, 4)

        transformed = (transform @ positions_h.T).T[:, :3]  # (N, 3)
        return transformed

    # --- Matching Pipeline and Labeling ---

    def run_icp_matching(
        self, progress_callback: Callable[[int, str], None] | None = None
    ) -> tuple[float, list[LabeledElectrode]]:
        """
        Complete Point Cloud Matching pipeline:
        1. Filter duplicates and outliers
        2. Initial alignment via fiducials (Nasion/LPA/RPA) with uniform scaling
        3. Template subsetting (keep only the M closest channels to point cloud)
        4. Iterative ICP refinement + Hungarian assignment (up to max_iterations)
        5. Final labeling with confidence metrics

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

        # Step 2: Fiducial alignment (with scale)
        if progress_callback:
            progress_callback(1, _("Initial fiducial alignment..."))
        m_fiducial = self.compute_fiducial_alignment()
        template_aligned = self.apply_transform_to_template(m_fiducial)

        point_cloud = self.get_point_cloud_array()
        M = len(point_cloud)
        N = len(template_aligned)

        # Step 3: Template subsetting when M < N
        # Keep only the closest channels to avoid ICP distortion from many
        # unmatched template points pulling the registration off.
        if M < N:
            subset_indices = self._select_template_subset(template_aligned, point_cloud)
        else:
            subset_indices = np.arange(N)

        template_subset = template_aligned[subset_indices]

        # Step 4: Iterative Assignment + Kabsch refinement
        if progress_callback:
            progress_callback(2, _("Iterative refinement..."))

        max_iterations = 5
        convergence_threshold_mm = 0.1
        outlier_threshold_mm = 50.0  # Generous threshold to pull in template initially
        prev_mean_error = float("inf")

        current_template = template_subset.copy()
        current_transform = m_fiducial.copy()
        best_transform = current_transform.copy()

        from invesalius.data import transformations as tr

        for iteration in range(max_iterations):
            # 1. Hungarian assignment on current positions
            cost_matrix = cdist(point_cloud, current_template)
            if M <= len(current_template):
                row_ind, col_ind = linear_sum_assignment(cost_matrix)
            else:
                row_ind_t, col_ind_t = linear_sum_assignment(cost_matrix.T)
                row_ind, col_ind = col_ind_t, row_ind_t

            # Compute per-pair distances
            pair_dists = np.array([cost_matrix[r, c] for r, c in zip(row_ind, col_ind)])
            mean_error = float(pair_dists.mean())

            # Check convergence
            improvement = prev_mean_error - mean_error
            if abs(improvement) < convergence_threshold_mm:
                best_transform = current_transform.copy()
                break

            prev_mean_error = mean_error
            best_transform = current_transform.copy()

            # 2. Filter outlier pairs for Kabsch (rigid + scale transform)
            good_mask = pair_dists < outlier_threshold_mm
            if good_mask.sum() < 3:
                break  # Not enough good pairs to compute transform

            # 3. Compute absolute transform from ORIGINAL template space to point cloud
            good_template_indices = col_ind[good_mask]
            src_points = self.template_positions[subset_indices[good_template_indices]]
            dst_points = point_cloud[row_ind[good_mask]]

            m_kabsch = tr.affine_matrix_from_points(
                src_points.T, dst_points.T, shear=False, scale=True
            )

            # Update transform and template for next iteration
            current_transform = m_kabsch
            current_template = self._apply_transform_to_points(
                self.template_positions[subset_indices], current_transform
            )

        # Step 5: Final assignment with the best transform (using ALL template channels)
        if progress_callback:
            progress_callback(3, _("Final labeling..."))

        self.icp_transform = best_transform
        template_final = self.apply_transform_to_template(best_transform)
        point_cloud = self.get_point_cloud_array()

        cost_matrix = cdist(point_cloud, template_final)
        if len(point_cloud) <= len(template_final):
            row_ind, col_ind = linear_sum_assignment(cost_matrix)
        else:
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

        # Step 6: Topological consistency check — fix neighbor swaps
        self._fix_topological_swaps(template_final)

        self.mean_error_mm = float(np.mean([e.distance_mm for e in self.labeled_electrodes]))
        self.state = DigitizationState.MATCHING_DONE

        return self.mean_error_mm, self.labeled_electrodes

    def _select_template_subset(
        self, template_positions: np.ndarray, point_cloud: np.ndarray
    ) -> np.ndarray:
        """
        Select a subset of template channels closest to the captured point cloud.
        For each captured point, find the nearest template channel and build a
        unique set. Then add a margin of nearby channels to improve ICP robustness.

        Returns: array of indices into template_positions.
        """
        from scipy.spatial.distance import cdist

        M = len(point_cloud)
        N = len(template_positions)

        # For each captured point, find the nearest template channel
        dist_matrix = cdist(point_cloud, template_positions)  # (M, N)
        nearest_per_point = np.argmin(dist_matrix, axis=1)  # (M,)
        selected = set(nearest_per_point)

        # Add margin: for each selected channel, also include its k nearest
        # template neighbors to give ICP more geometric context
        k_neighbors = min(3, N // max(M, 1))
        if k_neighbors > 0:
            tmpl_dist = cdist(template_positions, template_positions)  # (N, N)
            for idx in list(selected):
                neighbor_indices = np.argsort(tmpl_dist[idx])[1 : k_neighbors + 1]
                selected.update(neighbor_indices)

        # Cap at min(2*M, N) to avoid including the whole template
        selected_arr = np.array(sorted(selected))
        max_subset = min(2 * M, N)
        if len(selected_arr) > max_subset:
            # Keep the ones with smallest distance to any captured point
            min_dists = dist_matrix[:, selected_arr].min(axis=0)
            keep = np.argsort(min_dists)[:max_subset]
            selected_arr = selected_arr[keep]

        return selected_arr

    @staticmethod
    def _apply_transform_to_points(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
        """Apply a 4x4 transform to an (N, 3) array of points."""
        points_h = np.hstack([points, np.ones((len(points), 1))])
        return (transform @ points_h.T).T[:, :3]

    def _fix_topological_swaps(self, template_final: np.ndarray) -> int:
        """
        Post-assignment topological consistency check.

        For each pair of labeled electrodes that are template neighbors
        (within a neighborhood radius), check if swapping their labels
        would reduce the total pairwise distance. If so, perform the swap.

        This catches cases where the Hungarian algorithm assigns incorrect
        labels to adjacent electrodes due to small inter-electrode spacing
        (common in 32ch and 64ch systems).

        Returns: number of swaps performed.
        """
        if len(self.labeled_electrodes) < 2:
            return 0

        # Build lookup: label -> index in template_labels
        label_to_tmpl_idx = {lbl: i for i, lbl in enumerate(self.template_labels)}

        # Compute neighborhood radius from template (median inter-electrode distance)
        from scipy.spatial.distance import pdist

        if len(template_final) > 1:
            all_dists = pdist(template_final)
            neighbor_radius = float(np.median(all_dists)) * 0.6
        else:
            return 0

        n_swaps = 0
        max_swap_rounds = 3  # Limit iterations to avoid infinite loops

        for _round in range(max_swap_rounds):
            swapped_this_round = False

            for i in range(len(self.labeled_electrodes)):
                elec_a = self.labeled_electrodes[i]
                tmpl_idx_a = label_to_tmpl_idx.get(elec_a.label)
                if tmpl_idx_a is None:
                    continue

                for j in range(i + 1, len(self.labeled_electrodes)):
                    elec_b = self.labeled_electrodes[j]
                    tmpl_idx_b = label_to_tmpl_idx.get(elec_b.label)
                    if tmpl_idx_b is None:
                        continue

                    # Only check template neighbors (close in template space)
                    tmpl_dist = np.linalg.norm(
                        template_final[tmpl_idx_a] - template_final[tmpl_idx_b]
                    )
                    if tmpl_dist > neighbor_radius:
                        continue

                    # Current cost: a↔tmpl_a + b↔tmpl_b
                    cost_current = elec_a.distance_mm + elec_b.distance_mm

                    # Swapped cost: a↔tmpl_b + b↔tmpl_a
                    dist_a_to_b = float(
                        np.linalg.norm(elec_a.position_inv - template_final[tmpl_idx_b])
                    )
                    dist_b_to_a = float(
                        np.linalg.norm(elec_b.position_inv - template_final[tmpl_idx_a])
                    )
                    cost_swapped = dist_a_to_b + dist_b_to_a

                    # Swap if it reduces total cost by at least 1mm
                    if cost_swapped < cost_current - 1.0:
                        elec_a.label, elec_b.label = elec_b.label, elec_a.label
                        elec_a.template_position = template_final[tmpl_idx_b]
                        elec_b.template_position = template_final[tmpl_idx_a]
                        elec_a.distance_mm = dist_a_to_b
                        elec_b.distance_mm = dist_b_to_a

                        # Update confidence
                        for elec in (elec_a, elec_b):
                            if elec.distance_mm < 5.0:
                                elec.confidence = ConfidenceLevel.HIGH
                            elif elec.distance_mm < 10.0:
                                elec.confidence = ConfidenceLevel.MEDIUM
                            else:
                                elec.confidence = ConfidenceLevel.LOW

                        n_swaps += 1
                        swapped_this_round = True

            if not swapped_this_round:
                break

        return n_swaps

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

    def create_markers(self) -> list[int]:
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
            marker.label = elec.label
            marker.marker_type = MarkerType.EEG_ELECTRODE
            # Store in InVesalius space
            marker.position = list(elec.position_inv)
            # Default orientation
            marker.orientation = [0, 0, 0]

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

    def export_bids(self, output_dir: str, subject_id: str = "01") -> tuple[str, str]:
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

    def export_hpts(self, filepath: str) -> str:
        """
        Export in MNE Head Points (.hpts) format.
        Includes fiducials (cardinal) and electrodes (eeg).
        Uses world coordinates (Scanner RAS).
        """
        from invesalius.data import imagedata_utils

        lines = [
            "# Digitized points exported by InVesalius 3",
            "# Coordinate system: Scanner RAS",
        ]

        # Fiducial mapping: 1=nasion, 2=lpa, 3=rpa
        fid_map = {"nasion": 1, "lpa": 2, "rpa": 3}
        for fid_name, fid_id in fid_map.items():
            fid_pos = self.fiducials_inv.get(fid_name)
            if fid_pos is not None:
                pos_world, _ori = imagedata_utils.convert_invesalius_to_world(
                    position=list(fid_pos), orientation=[0, 0, 0]
                )
                if pos_world[0] is not None:
                    x, y, z = pos_world[:3]
                    lines.append(f"cardinal {fid_id} {x:.2f} {y:.2f} {z:.2f}")

        # Electrodes
        for elec in sorted(self.labeled_electrodes, key=lambda e: e.label):
            x, y, z = elec.position_world
            lines.append(f"eeg {elec.label} {x:.2f} {y:.2f} {z:.2f}")

        with open(filepath, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")

        self.state = DigitizationState.EXPORTED
        return filepath
