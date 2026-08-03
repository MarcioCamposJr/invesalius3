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
from typing import Dict, List, Optional, Tuple

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
