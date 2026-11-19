"""Experimental A–B tract display driven by the original cortical source signals."""

from itertools import combinations

import numpy as np
from vtkmodules.util.numpy_support import numpy_to_vtk
from vtkmodules.vtkCommonCore import vtkPoints
from vtkmodules.vtkCommonDataModel import vtkPolyData
from vtkmodules.vtkFiltersCore import vtkAppendPolyData
from vtkmodules.vtkFiltersGeneral import vtkVertexGlyphFilter
from vtkmodules.vtkRenderingCore import vtkActor, vtkPolyDataMapper

import invesalius.constants as const
from invesalius.data.tractography import compute_directions, compute_tubes
from invesalius.pubsub import pub as Publisher


class SourceTractConnection:
    def __init__(self, mode):
        self.mode = mode
        self.series = None
        self.seeds = [None, None]
        self.pairs = ((0, 1),)
        self.pair_index = 0
        self.traces = ()
        self.actors = {}
        self.counts = {}
        self.markers = []
        self.show_points = True
        self.count = 0
        self.revision = 0
        Publisher.subscribe(self.on_source_changed, "Source reconstruction changed")

    def clear(self):
        actors = [*self.markers, *self.actors.values()]
        Publisher.sendMessage("Remove actors " + str(const.SURFACE), actors=actors)
        self.seeds = [None, None]
        self.pairs = ((0, 1),)
        self.pair_index = 0
        self.traces = ()
        self.actors = {}
        self.counts = {}
        self.markers = []
        self.count = 0
        self.revision += 1
        if any(actor is not None for actor in actors):
            Publisher.sendMessage("Render volume viewer")

    def set_seed(self, index):
        mode = self.mode
        if index not in (0, 1) or mode.series is None or mode.selected_source is None:
            raise ValueError("Select a cortical source with the cross cursor first.")
        source_index = mode.selected_source
        vertex = int(mode.series.source_vertices[source_index])
        if mode.series.cortex_mask is not None and not mode.series.cortex_mask[vertex]:
            raise ValueError("Select a source inside the FreeSurfer cortical mask.")
        point = mode.series.anatomical_points()[vertex]
        if len(self.seeds) > 2:
            self.clear()
        self.seeds[index] = (source_index, point[None].copy(), np.array([vertex]))
        self._update_seeds()

    def detect_seeds(self, radius):
        values = self.mode.read_surface_frame(
            self.mode.playback.index, self.mode.series.base_representation
        )
        regions = self.mode.series.detect_regions(values, radius)
        self.mode.playback.pause()
        self.seeds = regions
        self._update_seeds()

    def select_pair(self, index):
        if not 0 <= index < len(self.pairs):
            raise ValueError("Select a valid source pair.")
        self.pair_index = index
        self._update_traces()
        Publisher.sendMessage("Source reconstruction changed", mode=self.mode)

    def _update_traces(self):
        self.traces = tuple(
            (f"{name}: {self.series.source_label(seed[0])}", self.series.read_source(seed[0]))
            for name, seed in zip(("A", "B"), (self.seeds[i] for i in self.pairs[self.pair_index]))
            if seed is not None
        )

    def _update_seeds(self):
        Publisher.sendMessage(
            "Remove actors " + str(const.SURFACE), actors=list(self.actors.values())
        )
        self.actors.clear()
        self.counts.clear()
        self.count = 0
        self.pairs = tuple(combinations(range(len(self.seeds)), 2))
        self.pair_index = 0
        self._update_traces()
        self.revision += 1
        Publisher.sendMessage("Remove actors " + str(const.SURFACE), actors=self.markers)
        self.markers = []
        for seed in self.seeds:
            if seed is None:
                continue
            points = vtkPoints()
            points.SetData(numpy_to_vtk(seed[1], deep=True))
            polydata = vtkPolyData()
            polydata.SetPoints(points)
            glyphs = vtkVertexGlyphFilter()
            glyphs.SetInputData(polydata)
            glyphs.Update()
            marker = self._actor(glyphs.GetOutput())
            marker.GetProperty().SetPointSize(5)
            marker.GetProperty().RenderPointsAsSpheresOn()
            self.markers.append(marker)
        Publisher.sendMessage("Source reconstruction changed", mode=self.mode)
        Publisher.sendMessage("Render volume viewer")

    def _actor(self, polydata):
        polydata = self.mode.surface_manager.ConvertPolydataToInv(polydata)
        mapper = vtkPolyDataMapper()
        mapper.SetInputData(polydata)
        actor = vtkActor()
        actor.SetMapper(mapper)
        actor.PickableOff()
        Publisher.sendMessage("AppendActor", actor=actor)
        return actor

    def set_tracks(self, tracks, revision):
        if revision != self.revision or self.series is not self.mode.series:
            return
        if not any(tracks.values()):
            raise ValueError("No A–B tract was found. Check alignment, seed locations and radius.")
        actors = {}
        try:
            for pair, paths in tracks.items():
                if not paths:
                    continue
                combined = vtkAppendPolyData()
                for track in paths:
                    tube = compute_tubes(track, compute_directions(track))
                    combined.AddInputData(tube.GetOutput())
                combined.Update()
                actors[pair] = self._actor(combined.GetOutput())
        except Exception:
            Publisher.sendMessage(
                "Remove actors " + str(const.SURFACE), actors=list(actors.values())
            )
            raise
        Publisher.sendMessage(
            "Remove actors " + str(const.SURFACE), actors=list(self.actors.values())
        )
        self.actors = actors
        self.counts = {pair: len(paths) for pair, paths in tracks.items()}
        self.count = sum(self.counts.values())
        Publisher.sendMessage("Source reconstruction changed", mode=self.mode)
        Publisher.sendMessage("Render volume viewer")

    def on_source_changed(self, mode):
        if mode is not self.mode:
            return
        if mode.series is not self.series:
            self.clear()
            self.series = mode.series
        visible = mode.display_active and mode.representation != "inflated"
        selected = self.pairs[self.pair_index]
        colors = {selected[0]: (0.4, 1.0, 0.4), selected[1]: (0.9, 0.4, 1.0)}
        indices = [i for i, seed in enumerate(self.seeds) if seed is not None]
        for index, marker in zip(indices, self.markers):
            marker.SetVisibility(visible and self.show_points)
            marker.GetProperty().SetColor(colors.get(index, (0.6, 0.6, 0.6)))
        strength = None
        if self.actors:
            frame = mode.read_surface_frame(mode.playback.index, mode.series.base_representation)
            strength = mode.series.activation_strength(frame)
        for pair, actor in self.actors.items():
            actor.SetVisibility(visible)
            activation = max(strength[self.seeds[i][2]].max() for i in pair)
            actor.GetProperty().SetOpacity(0.02 + 0.98 * activation)
