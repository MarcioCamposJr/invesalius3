# --------------------------------------------------------------------------
# Software:     InVesalius - Software de Reconstrucao 3D de Imagens Medicas
# Copyright:    (C) 2001  Centro de Pesquisas Renato Archer
# Homepage:     http://www.softwarepublico.gov.br
# Contact:      invesalius@cti.gov.br
# License:      GNU - GPL 2 (LICENSE.txt/LICENCA.txt)
# --------------------------------------------------------------------------

"""
VTK actors for EEG electrode visualization.
Used in the EEG Digitization wizard.
"""

from typing import List, Tuple

import numpy as np
from vtkmodules.vtkCommonCore import vtkPoints
from vtkmodules.vtkCommonDataModel import vtkPolyData
from vtkmodules.vtkFiltersSources import vtkSphereSource
from vtkmodules.vtkRenderingCore import (
    vtkActor,
    vtkFollower,
    vtkPolyDataMapper,
    vtkProperty,
    vtkRenderer,
)
from vtkmodules.vtkRenderingFreeType import vtkVectorText

from invesalius.navigation.eeg_montage import ConfidenceLevel, LabeledElectrode


def confidence_to_color(confidence: ConfidenceLevel) -> Tuple[float, float, float]:
    """Map ConfidenceLevel to RGB color."""
    if confidence == ConfidenceLevel.HIGH:
        return (0.0, 0.8, 0.0)  # Green
    elif confidence == ConfidenceLevel.MEDIUM:
        return (1.0, 0.8, 0.0)  # Yellow
    else:
        return (1.0, 0.0, 0.0)  # Red


def create_electrode_sphere(
    position: np.ndarray, radius: float, color: Tuple[float, float, float]
) -> vtkActor:
    """Create a spherical vtkActor for an electrode."""
    sphere = vtkSphereSource()
    sphere.SetRadius(radius)
    sphere.SetCenter(position[0], position[1], position[2])

    mapper = vtkPolyDataMapper()
    mapper.SetInputConnection(sphere.GetOutputPort())

    prop = vtkProperty()
    prop.SetColor(*color)

    actor = vtkActor()
    actor.SetMapper(mapper)
    actor.SetProperty(prop)

    return actor


def create_label_follower(
    text: str, position: np.ndarray, camera, color: Tuple[float, float, float] = (1.0, 1.0, 1.0)
) -> vtkFollower:
    """Create a vtkFollower with text that follows the camera."""
    vector_text = vtkVectorText()
    vector_text.SetText(text)
    vector_text.Update()

    mapper = vtkPolyDataMapper()
    mapper.SetInputConnection(vector_text.GetOutputPort())

    follower = vtkFollower()
    follower.SetMapper(mapper)
    follower.SetScale(2.5, 2.5, 2.5)
    # Offset slightly above the sphere
    follower.SetPosition(position[0], position[1], position[2] + 4.0)
    follower.GetProperty().SetColor(*color)
    follower.SetCamera(camera)

    return follower


def update_electrode_actors(
    renderer: vtkRenderer, labeled_electrodes: List[LabeledElectrode], invert_y: bool = True
) -> List[vtkActor]:
    """
    Generate the visual actors for a list of labeled electrodes.
    Returns the list of generated actors to be added to the renderer.
    """
    actors = []

    # Needs camera for followers
    camera = renderer.GetActiveCamera()

    for elec in labeled_electrodes:
        # Display coordinate logic from ICP dialog: x, -y, z
        pos = list(elec.position_inv)
        if invert_y:
            pos[1] = -pos[1]

        color = confidence_to_color(elec.confidence)

        # Create sphere
        sphere_actor = create_electrode_sphere(pos, radius=3.0, color=color)
        actors.append(sphere_actor)

        # Create label
        label_actor = create_label_follower(elec.label, pos, camera, color=(1.0, 1.0, 1.0))
        actors.append(label_actor)

    return actors
