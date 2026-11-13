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
from invesalius.navigation.efield_navigation import EFieldNavigation
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


@pytest.fixture
def field_navigation(field_view):
    scene = SimpleNamespace(ren=field_view.renderer, tracts_status=False, actor_tracts=None)
    navigation = EFieldNavigation(scene, Mock(), ActorFactory(), Mock())
    yield navigation
    navigation.dispose()


def test_static_result_feeds_contours_and_cortex_queries(field_view, field_navigation):
    surface = field_view.add_surface()
    values = field_view.load([surface])
    assert field_navigation.static_efield
    assert field_navigation.efield_painting_token is None
    np.testing.assert_allclose(field_navigation.e_field_norms, values)

    pub.sendMessage("Show Efield contours", enable=True)
    assert field_navigation.visualizer.edge_actor is not None
    cortex = []

    def on_cortex(CoGposition, CoGorientation):
        cortex.append((CoGposition, CoGorientation))

    pub.subscribe(on_cortex, "Update Cortex Marker")
    try:
        pub.sendMessage("Get Cortex position")
        assert len(cortex) == 1
        np.testing.assert_allclose(cortex[0][1], [1.0, 0.0, 0.0])
        field_view.viewer.Remove()
        assert not field_navigation.static_efield
        assert field_navigation.efield_mesh is None
        assert not field_navigation.Id_list
        assert field_navigation.e_field_norms_to_save is None
        assert field_navigation.visualizer.edge_actor is None
        pub.sendMessage("Get Cortex position")
        assert len(cortex) == 1
    finally:
        pub.unsubscribe(on_cortex, "Update Cortex Marker")


def test_navigation_created_after_load_receives_existing_result(field_view):
    surface = field_view.add_surface()
    values = field_view.load([surface])
    navigation = EFieldNavigation(
        SimpleNamespace(ren=field_view.renderer, tracts_status=False, actor_tracts=None),
        Mock(),
        ActorFactory(),
        Mock(),
    )
    try:
        assert navigation.static_efield
        np.testing.assert_allclose(navigation.e_field_norms, values)
    finally:
        navigation.dispose()


def test_threshold_updates_static_tools_without_replacing_map(field_view, field_navigation):
    surface = field_view.add_surface()
    field_view.load([surface])
    pub.sendMessage("Show Efield contours", enable=True)
    field_view.viewer.SetThreshold(0.5)
    assert field_navigation.efield_threshold == 0.5
    assert all(
        value > 50.0
        for value in field_navigation.e_field_norms[
            np.array(field_navigation.cell_id_indexes_above_threshold)
        ]
    )
    assert field_view.painting.static_maps[surface] is surface.display_map
    assert field_navigation.efield_painting_token is None


@pytest.mark.parametrize("zero_field", [False, True])
def test_scalar_only_or_zero_field_does_not_emit_invalid_cortex_marker(
    field_view, field_navigation, monkeypatch, zero_field
):
    surface = field_view.add_surface()
    if zero_field:
        mesh = surface.polydata
        monkeypatch.setattr(
            simnibs,
            "ReadEfieldSurface",
            lambda _: (
                mesh,
                np.zeros(mesh.GetNumberOfPoints()),
                np.zeros((mesh.GetNumberOfPoints(), 3)),
            ),
        )
        field_view.viewer.Load("zero.vtp", target_surfaces=[surface.index])
    else:
        field_view.load([surface], vectors=False)
    callback = Mock()

    def on_cortex(CoGposition, CoGorientation):
        callback(CoGposition, CoGorientation)

    pub.subscribe(on_cortex, "Update Cortex Marker")
    try:
        pub.sendMessage("Get Cortex position")
        callback.assert_not_called()
    finally:
        pub.unsubscribe(on_cortex, "Update Cortex Marker")


def test_live_efield_replaces_static_result(field_view, field_navigation):
    from invesalius.data.e_field import EFieldBrain

    surface = field_view.add_surface()
    field_view.load([surface])
    token = object()
    pub.sendMessage(
        "Initialize E-field brain",
        e_field_brain=EFieldBrain(surface.polydata),
        painting_token=token,
    )
    assert field_view.viewer.source is None
    assert not field_view.painting.static_maps
    assert not field_navigation.static_efield
    assert field_navigation.efield_painting_token is token
    assert field_navigation.e_field_norms is None


def test_center_of_gravity_uses_values_of_selected_points(field_view, field_navigation):
    surface = field_view.add_surface()
    field_view.load([surface])
    selected = [7, 20]
    values = dict(zip(field_navigation.Id_list, field_navigation.e_field_norms))
    points = np.array([surface.polydata.GetPoint(i) for i in selected])
    expected = np.average(points, axis=0, weights=[values[i] for i in selected])
    locator = Mock()

    def closest(point, result, *args):
        result[:] = point

    locator.FindClosestPoint.side_effect = closest
    field_navigation.locator_efield_cell = locator
    actual = field_navigation.FindCenterofGravity(selected, points.tolist())
    np.testing.assert_allclose(actual, expected)


def test_static_target_value_and_targeting_tools_update_immediately(field_view, field_navigation):
    surface = field_view.add_surface()
    values = field_view.load([surface])
    point = list(surface.polydata.GetPoint(8))
    point[1] = -point[1]
    pub.sendMessage("Set as Efield target at cortex", position=point, orientation=[0, 0, 0])
    legend = field_navigation.visualizer.EfieldAtTargetLegend
    assert legend.actor.GetMapper().GetInput() == "Efield at Target: " + f"{values[8]:04.2f}"

    pub.sendMessage("Enable Efield tools", enable=True)
    assert field_navigation.distance_efield is not None
    assert field_navigation.focal_factor_members


def test_clearing_result_resets_panel_state():
    from invesalius.gui.task_simnibs import InnerTaskPanel

    panel = SimpleNamespace(
        _efield_loaded=True,
        _efield_targets=["GM"],
        _retarget_on_new_surface=True,
        combo_surface=Mock(),
        lbl_efield=Mock(),
    )
    InnerTaskPanel._on_efield_painted(panel, painted=[], refused=[])
    assert not panel._efield_loaded
    assert not panel._efield_targets
    assert not panel._retarget_on_new_surface
    panel.combo_surface.SetChecked.assert_called_once_with([])
    panel.lbl_efield.SetLabel.assert_called_once()
