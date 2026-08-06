# --------------------------------------------------------------------------
# Software:     InVesalius - Software de Reconstrucao 3D de Imagens Medicas
# Copyright:    (C) 2001  Centro de Pesquisas Renato Archer
# Homepage:     http://www.softwarepublico.gov.br
# Contact:      invesalius@cti.gov.br
# License:      GNU - GPL 2 (LICENSE.txt/LICENCA.txt)
# --------------------------------------------------------------------------

"""
VTK actors and visualizer for EEG electrode visualization.
Delegates the 3D rendering of electrodes to the main viewer.
"""

import math

import numpy as np
import vtk
from vtkmodules.vtkRenderingCore import vtkBillboardTextActor3D, vtkRenderer


class EEGVisualizer:
    def __init__(self, renderer: vtkRenderer):
        self.ren = renderer
        self.actors = {}

    def UpdateElectrodes(self, electrodes_data, show=True, highlight_name=None):
        # Clear existing
        for actor in self.actors.values():
            self.ren.RemoveActor(actor)
        self.actors.clear()

        if not electrodes_data:
            return

        for i, elec in enumerate(electrodes_data):
            name = elec.get("name", f"E{i}")
            position = elec.get("position", [0, 0, 0])
            normal = elec.get("normal", [0, 0, 1])
            color = elec.get("color", (0.5, 0.5, 0.5))
            text_color = elec.get("text_color", (1.0, 1.0, 1.0))

            # Create torus
            source = vtk.vtkParametricTorus()
            source.SetRingRadius(3.0)
            source.SetCrossSectionRadius(0.8)

            source_fn = vtk.vtkParametricFunctionSource()
            source_fn.SetParametricFunction(source)
            source_fn.SetUResolution(40)
            source_fn.SetVResolution(40)
            source_fn.Update()

            mapper = vtk.vtkPolyDataMapper()
            mapper.SetInputConnection(source_fn.GetOutputPort())

            actor = vtk.vtkActor()
            actor.SetMapper(mapper)
            actor.GetProperty().SetOpacity(0.8)

            source_z = np.array([0, 0, 1])
            target_z = np.array(normal)
            if np.linalg.norm(target_z) > 1e-6:
                target_z = target_z / np.linalg.norm(target_z)
            else:
                target_z = np.array([0, 0, 1])

            # Ensure normal points outward (approximate center at origin)
            if np.dot(target_z, np.array(position)) < 0:
                target_z = -target_z

            source_z = np.array([0, 0, 1])
            axis = np.cross(source_z, target_z)
            axis_norm = np.linalg.norm(axis)

            transform = vtk.vtkTransform()
            transform.Translate(position)

            if axis_norm > 1e-6:
                axis = axis / axis_norm
                angle = math.degrees(math.acos(np.clip(np.dot(source_z, target_z), -1.0, 1.0)))
                transform.RotateWXYZ(angle, axis[0], axis[1], axis[2])
            elif np.dot(source_z, target_z) < 0:
                transform.RotateWXYZ(180, 1, 0, 0)

            actor.SetUserTransform(transform)
            actor.SetVisibility(show)

            if highlight_name and name == highlight_name:
                actor.GetProperty().SetColor(0.0, 0.5, 1.0)
            else:
                actor.GetProperty().SetColor(*color)

            self.ren.AddActor(actor)
            self.actors[name] = actor

            # Create text label (Billboard)
            text_actor = vtkBillboardTextActor3D()
            text_actor.SetInput(name)

            text_prop = text_actor.GetTextProperty()
            text_prop.SetFontSize(28)
            text_prop.SetColor(*text_color)
            text_prop.SetBold(True)
            text_prop.SetShadow(True)
            text_prop.SetShadowOffset(2, -2)

            text_prop.SetFrame(True)
            text_prop.SetFrameColor(0.2, 0.2, 0.2)
            text_prop.SetFrameWidth(2)
            text_prop.SetBackgroundColor(0.3, 0.3, 0.3)
            text_prop.SetBackgroundOpacity(0.85)

            offset_pos = np.array(position) + np.array(target_z) * 6.0
            text_actor.SetPosition(offset_pos)
            text_actor.SetVisibility(show)
            self.ren.AddActor(text_actor)
            self.actors[f"{name}_text"] = text_actor

    def SetVisibility(self, show):
        for actor in self.actors.values():
            actor.SetVisibility(show)
