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

import numpy as np
from vtkmodules.util.numpy_support import numpy_to_vtk
from vtkmodules.vtkCommonDataModel import vtkPolyData
from vtkmodules.vtkRenderingAnnotation import vtkScalarBarActor
from vtkmodules.vtkRenderingCore import vtkActor, vtkPolyDataMapper


class SurfaceScalarLayer:
    """Display scalar point data with independent attributes and shared geometry."""

    def __init__(self, poly_data, actor=None):
        # Separate the attributes so coloring a layer cannot change the source surface.
        self.poly_data = vtkPolyData()
        self.poly_data.ShallowCopy(poly_data)

        self.mapper = vtkPolyDataMapper()
        self.mapper.SetInputData(self.poly_data)
        self.mapper.SetScalarModeToUsePointData()
        self.mapper.SetColorModeToMapScalars()

        self.actor = actor if actor is not None else vtkActor()
        self.actor.SetMapper(self.mapper)
        self.scalar_bar = vtkScalarBarActor()

    def SetPointScalars(self, name, values, point_ids=None):
        """Replace point values, using NaN for points outside a partial update."""
        values = np.asarray(values, dtype=float)
        number_of_points = self.poly_data.GetNumberOfPoints()
        if values.ndim != 1:
            raise ValueError("Scalar values must be a one-dimensional array")

        if point_ids is None:
            if len(values) != number_of_points:
                raise ValueError("A scalar value is required for each surface point")
            scalar_values = values
        else:
            point_ids = np.asarray(point_ids)
            if point_ids.ndim != 1 or len(point_ids) != len(values):
                raise ValueError("A scalar value is required for each point ID")
            if point_ids.size and not np.issubdtype(point_ids.dtype, np.integer):
                raise ValueError("Point IDs must be integers")
            if np.any(point_ids < 0) or np.any(point_ids >= number_of_points):
                raise ValueError("Point ID is outside the surface point range")

            scalar_values = np.full(number_of_points, np.nan)
            scalar_values[point_ids.astype(np.intp, copy=False)] = values

        scalars = numpy_to_vtk(scalar_values, deep=True)
        scalars.SetName(name)
        self.poly_data.GetPointData().SetScalars(scalars)
        self.mapper.ScalarVisibilityOn()
        self.poly_data.Modified()

    def SetLookupTable(self, lookup_table):
        self.mapper.SetLookupTable(lookup_table)
        self.scalar_bar.SetLookupTable(lookup_table)
