from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
from vtkmodules.vtkCommonCore import vtkLookupTable
from vtkmodules.vtkFiltersSources import vtkSphereSource
from vtkmodules.vtkRenderingCore import vtkActor, vtkPolyDataMapper, vtkRenderer

from invesalius.data import simnibs_efield as simnibs
from invesalius.data.actor_factory import ActorFactory
from invesalius.data.surface import Surface, SurfaceManager
from invesalius.data.visualization import surface_painting
from invesalius.data.visualization.surface_map_visualizer import SurfaceMap
from invesalius.project import Project
from invesalius.pubsub import pub


@pytest.fixture
def field_view(monkeypatch):
    project = Project()
    monkeypatch.setattr(project, "surface_dict", {})
    monkeypatch.setattr(project, "peel", None)
    monkeypatch.setattr(surface_painting.SurfacePainting, "instance", None)
    monkeypatch.setattr(surface_painting, "Session", lambda: SimpleNamespace(GetConfig=lambda _: 0))
    monkeypatch.setattr(pub, "sendMessage_hook", None)
    painting = surface_painting.SurfacePainting()
    manager = SurfaceManager.__new__(SurfaceManager)
    manager.actors_dict = {}
    manager.map_visualizers = {}
    manager.painting = painting
    manager.last_surface_index = 0
    renderer = vtkRenderer()
    pub.subscribe(manager.GetActor, "Get Actor")
    pub.subscribe(manager.UpdateSurfacePresentation, "Update surface presentation")
    pub.subscribe(manager.OnShowSurface, "Show surface")

    def append_actor(actor):
        renderer.AddActor(actor)

    def remove_actor(actor):
        renderer.RemoveActor(actor)

    pub.subscribe(append_actor, "AppendActor")
    pub.subscribe(remove_actor, "Remove surface actor from viewer")
    viewer = simnibs.SimnibsEfieldVisualizer(renderer, ActorFactory(), lambda: None)

    def add_surface(name="GM", radius=1.0):
        sphere = vtkSphereSource()
        sphere.SetRadius(radius)
        sphere.Update()
        surface = Surface(name=name)
        surface.polydata = sphere.GetOutput()
        surface.index = len(project.surface_dict)
        project.surface_dict[surface.index] = surface
        mapper = vtkPolyDataMapper()
        mapper.SetInputData(surface.polydata)
        actor = vtkActor()
        actor.SetMapper(mapper)
        renderer.AddActor(actor)
        manager.actors_dict[surface.index] = actor
        return surface

    def load(surfaces, vectors=True):
        mesh = surfaces[0].polydata
        values = np.linspace(10.0, 100.0, mesh.GetNumberOfPoints())
        field_vectors = np.tile([1.0, 0.0, 0.0], (len(values), 1)) if vectors else None
        monkeypatch.setattr(simnibs, "ReadEfieldSurface", lambda _: (mesh, values, field_vectors))
        viewer.Load("result.vtp", target_surfaces=[surface.index for surface in surfaces])
        return values

    yield SimpleNamespace(
        viewer=viewer,
        manager=manager,
        painting=painting,
        renderer=renderer,
        add_surface=add_surface,
        load=load,
    )
    viewer.dispose()
    painting.close()
    for owner in (manager, painting, painting.peeling, renderer):
        pub.unsubscribe_owner(owner)
    pub.unsubscribe(append_actor, "AppendActor")
    pub.unsubscribe(remove_actor, "Remove surface actor from viewer")


def test_replaces_existing_map_and_restores_frozen_map(field_view):
    surface = field_view.add_surface()
    lut = vtkLookupTable()
    lut.Build()
    previous = SurfaceMap(surface.polydata, lut, (0.0, 1.0), "Snapshot", "V/m")
    surface.frozen_map = surface.display_map = previous
    field_view.manager.UpdateSurfacePresentation(surface)
    original = field_view.manager.actors_dict[surface.index]
    mapper = original.GetMapper()

    field_view.load([surface])
    display = field_view.manager.map_visualizers[surface]
    assert surface.display_map is not previous
    assert display.actor.GetVisibility()
    assert not original.GetVisibility()
    assert original.GetMapper() is mapper
    assert surface.displayed_polydata.GetPointData().GetArray("Colors") is not None

    field_view.viewer.Remove()
    assert surface.display_map is previous
    assert original.GetMapper() is mapper
    assert not field_view.painting.static_maps


def test_static_map_survives_navigation_mode_changes_and_hides_peeling(field_view):
    surface = field_view.add_surface()
    surface.anatomical_role = "brain"
    peeling = field_view.painting.peeling
    peel_actor = vtkActor()
    peeling.actor = peel_actor
    field_view.painting.navigation_mode = True
    field_view.renderer.AddActor(peel_actor)
    field_view.load([surface])
    current_map = surface.display_map

    field_view.painting.set_navigation_mode(False)
    field_view.painting.set_navigation_mode(True)

    assert surface.display_map is current_map
    assert field_view.manager.map_visualizers[surface].actor.GetVisibility()
    assert not peel_actor.GetVisibility()


def test_multiple_surfaces_share_one_scale_and_restore_opacity(field_view):
    inner = field_view.add_surface()
    outer = field_view.add_surface("Skin", radius=2.0)
    field_view.load([inner, outer])
    displays = field_view.manager.map_visualizers
    assert displays[inner].scalar_bar.GetVisibility()
    assert not displays[outer].scalar_bar.GetVisibility()
    assert displays[inner].mapper.GetScalarRange() == (0.0, 100.0)
    assert displays[outer].actor.GetProperty().GetOpacity() == pytest.approx(0.25)

    field_view.viewer.SetOpacity(0.5)
    assert displays[inner].actor.GetProperty().GetOpacity() == pytest.approx(0.5)
    field_view.viewer.Remove()
    assert all(
        actor.GetProperty().GetOpacity() == 1.0 for actor in field_view.manager.actors_dict.values()
    )


def test_selecting_original_clears_all_simnibs_maps_and_cuts(field_view):
    surfaces = [field_view.add_surface(), field_view.add_surface("Skin", radius=2.0)]
    field_view.load(surfaces)
    cuts = Mock()

    def on_cuts(polydatas):
        cuts(polydatas=polydatas)

    pub.subscribe(on_cuts, "Set SimNIBS efield cross sections")
    try:
        field_view.painting.select(surfaces[0])
        assert field_view.viewer.source is None
        assert not field_view.viewer.arrows
        assert all(surface.display_map is None for surface in surfaces)
        cuts.assert_called_with(polydatas=[])
    finally:
        pub.unsubscribe(on_cuts, "Set SimNIBS efield cross sections")


def test_duplicate_keeps_simnibs_colors_as_a_snapshot(field_view, monkeypatch):
    surface = field_view.add_surface()
    field_view.load([surface])
    create = Mock()
    monkeypatch.setattr(field_view.manager, "CreateSurfaceFromPolydata", create)
    field_view.manager.OnDuplicate([surface.index])
    snapshot = create.call_args.kwargs["frozen_map"]
    assert snapshot is not None
    field_view.viewer.Remove()
    assert snapshot.polydata.GetPointData().GetArray("Colors") is not None
    assert snapshot.polydata is not surface.polydata
