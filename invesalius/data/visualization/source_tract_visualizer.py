"""Experimental A–B tract display driven by the original cortical source signals."""

from itertools import combinations

import numpy as np
import wx
from vtkmodules.util.numpy_support import numpy_to_vtk
from vtkmodules.vtkCommonCore import vtkPoints
from vtkmodules.vtkCommonDataModel import vtkPolyData
from vtkmodules.vtkFiltersCore import vtkAppendPolyData
from vtkmodules.vtkFiltersGeneral import vtkVertexGlyphFilter
from vtkmodules.vtkRenderingCore import vtkActor, vtkPolyDataMapper

import invesalius.constants as const
from invesalius.data.tractography import compute_directions, compute_tubes, source_pair_targets
from invesalius.pubsub import pub as Publisher


class SourceTractConnection:
    def __init__(self, mode):
        self.mode = mode
        self.series = None
        self.temporal = None
        self.delay_limits = (2.0, 20.0)
        self.pair_events = {}
        self.search_pairs = ()
        # shortcut: session-only cache for this seed geometry; persist it if cross-session reuse is needed.
        self.track_cache = {}
        self.render_counts = {}
        self.attempted_pairs = set()
        self.cache_context = None
        self.cache_radius = None
        self._seed_geometry = None
        self.seeds = [None, None]
        self.pairs = ((0, 1),)
        self.pair_index = 0
        self.traces = ()
        self.actors = {}
        self.counts = {}
        self.markers = []
        self.show_points = True
        self.fiber_multiplier = 2.0
        self.max_fibers = 2000
        self.search_seconds = 45.0
        self.count = 0
        self.revision = 0
        Publisher.subscribe(self.on_source_changed, "Source reconstruction changed")
        for topic in ("Update Trekker object", "Enable ACT", "Update ACT data"):
            Publisher.subscribe(self.on_tracking_changed, topic)

    @property
    def targets(self):
        if any(seed is None for seed in self.seeds):
            return {}
        targets = source_pair_targets(
            [seed[1] for seed in self.seeds], self.fiber_multiplier, self.max_fibers
        )
        return (
            {pair: targets[pair] for pair in self.search_pairs}
            if self.temporal is not None
            else targets
        )

    def search_summary(self):
        targets = self.targets
        available = sum(
            min(len(self.track_cache.get(pair, ())), count) for pair, count in targets.items()
        )
        return {
            "pairs": len(targets),
            "target": sum(targets.values()),
            "available": available,
            "missing": sum(targets.values()) - available,
            "pending_pairs": sum(
                len(self.track_cache.get(pair, ())) < count for pair, count in targets.items()
            ),
            "attempted": len(self.attempted_pairs.intersection(targets)),
        }

    def invalidate_tracks(self):
        Publisher.sendMessage(
            "Remove actors " + str(const.SURFACE), actors=list(self.actors.values())
        )
        self.actors.clear()
        self.counts.clear()
        self.count = 0
        self.track_cache.clear()
        self.render_counts.clear()
        self.attempted_pairs.clear()
        self.cache_context = self.cache_radius = None

    def on_tracking_changed(self, data):
        if not wx.IsMainThread():
            wx.CallAfter(self.on_tracking_changed, data)
            return
        self.invalidate_tracks()
        self.revision += 1
        Publisher.sendMessage("Source reconstruction changed", mode=self.mode)
        Publisher.sendMessage("Render volume viewer")

    def prepare_cache(self, radius, context):
        if self.cache_radius != radius or self.cache_context != context:
            self.invalidate_tracks()
        self.cache_radius, self.cache_context = radius, context

    def clear(self):
        actors = [*self.markers, *self.actors.values()]
        Publisher.sendMessage("Remove actors " + str(const.SURFACE), actors=actors)
        self.seeds = [None, None]
        self.temporal = None
        self.pair_events = {}
        self.search_pairs = ()
        self.track_cache.clear()
        self.render_counts.clear()
        self.attempted_pairs.clear()
        self.cache_context = self.cache_radius = self._seed_geometry = None
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
        if self.temporal is not None or len(self.seeds) > 2:
            self.clear()
        self.seeds[index] = (source_index, point[None].copy(), np.array([vertex]))
        self._update_seeds()

    def detect_seeds(self, radius):
        values = self.mode.read_surface_frame(
            self.mode.playback.index, self.mode.series.base_representation
        )
        regions = self.mode.series.detect_regions(values, radius)
        if len(regions) < 2:
            raise ValueError(
                "Fewer than two active regions were found. "
                "Review the current time, Min/Med color limits and search radius."
            )
        self.mode.playback.pause()
        self.temporal = None
        self.pair_events = {}
        self.seeds = regions
        self._update_seeds()

    def set_temporal_regions(self, result, revision):
        if revision != self.revision or self.series is not self.mode.series:
            return
        self.temporal = result
        self.seeds = result.regions
        self.search_pairs = result.anatomical_pairs(self.series.axis.dt, self.delay_limits)
        self._update_seeds()

    def update_time(self):
        if self.temporal is None:
            return
        sample = self.temporal.index_at(self.mode.playback.index)
        self.pair_events = self.temporal.preceding_pairs(
            sample, self.series.axis.dt, self.delay_limits
        )
        pairs = tuple(sorted(self.pair_events))
        selected = self.pairs[self.pair_index] if self.pairs else None
        if pairs != self.pairs:
            self.pairs = pairs
            self.pair_index = pairs.index(selected) if selected in pairs else 0
            self._update_traces()

    def set_delay_limits(self, minimum, maximum):
        if not np.isfinite((minimum, maximum)).all() or not 0 < minimum <= maximum:
            raise ValueError("Onset delays must satisfy 0 < minimum <= maximum in milliseconds.")
        self.delay_limits = (minimum, maximum)
        if self.temporal is not None:
            self.search_pairs = self.temporal.anatomical_pairs(
                self.series.axis.dt, self.delay_limits
            )
        self.update_time()
        Publisher.sendMessage("Source reconstruction changed", mode=self.mode)
        Publisher.sendMessage("Render volume viewer")

    def select_pair(self, index):
        if not 0 <= index < len(self.pairs):
            raise ValueError("Select a valid source pair.")
        self.pair_index = index
        self._update_traces()
        Publisher.sendMessage("Source reconstruction changed", mode=self.mode)
        Publisher.sendMessage("Render volume viewer")

    def _update_traces(self):
        if not self.pairs:
            self.traces = ()
            return
        self.traces = tuple(
            (f"{name}: {self.series.source_label(seed[0])}", self.series.read_source(seed[0]))
            for name, seed in zip(("A", "B"), (self.seeds[i] for i in self.pairs[self.pair_index]))
            if seed is not None
        )

    def _update_seeds(self):
        geometry = tuple(seed[1].tobytes() if seed is not None else None for seed in self.seeds)
        if self.temporal is None or geometry != self._seed_geometry or self.cache_context is None:
            self.invalidate_tracks()
        self._seed_geometry = geometry
        self.pairs = tuple(combinations(range(len(self.seeds)), 2))
        self.pair_index = 0
        self.update_time()
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

    def set_cached_tracks(self, tracks, revision, completed):
        if revision != self.revision or self.series is not self.mode.series:
            return
        for pair, paths in tracks.items():
            self.track_cache[pair] = paths
            self.render_counts.pop(pair, None)
        self.attempted_pairs.update(completed)
        self.counts = {pair: len(paths) for pair, paths in self.track_cache.items()}
        self.count = sum(self.counts.values())
        Publisher.sendMessage("Source reconstruction changed", mode=self.mode)
        Publisher.sendMessage("Render volume viewer")

    def _show_cached_tracks(self, visible):
        targets = source_pair_targets(
            [seed[1] for seed in self.seeds], self.fiber_multiplier, self.max_fibers
        )
        sample = self.temporal.index_at(self.mode.playback.index)
        strengths = {}
        for pair, (earlier, _later) in self.pair_events.items():
            left, right = (
                self.series.activation_strength(
                    np.where(
                        self.temporal.point_activity[region][at],
                        self.temporal.point_values[region][at],
                        0,
                    )
                ).max()
                for region, at in ((pair[0], earlier), (pair[1], sample))
            )
            key = tuple(sorted(pair))
            strength = max(left, right) if min(left, right) > 0 else 0
            strengths[key] = max(strengths.get(key, 0), strength)
        for pair, paths in self.track_cache.items():
            count = min(len(paths), targets[pair])
            if not count:
                continue
            if self.render_counts.get(pair) != count:
                combined = vtkAppendPolyData()
                for path in paths[:count]:
                    tube = compute_tubes(path, compute_directions(path))
                    combined.AddInputData(tube.GetOutput())
                combined.Update()
                actor = self._actor(combined.GetOutput())
                if pair in self.actors:
                    Publisher.sendMessage(
                        "Remove actors " + str(const.SURFACE), actors=[self.actors[pair]]
                    )
                self.actors[pair] = actor
                self.render_counts[pair] = count
            self.actors[pair].SetVisibility(visible)
            self.actors[pair].GetProperty().SetOpacity(0.02 + 0.98 * strengths.get(pair, 0))

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
        selected = self.pairs[self.pair_index] if self.pairs else ()
        colors = dict(zip(selected, ((0.4, 1.0, 0.4), (0.9, 0.4, 1.0))))
        indices = [i for i, seed in enumerate(self.seeds) if seed is not None]
        for index, marker in zip(indices, self.markers):
            marker.SetVisibility(visible and self.show_points)
            marker.GetProperty().SetColor(colors.get(index, (0.6, 0.6, 0.6)))
            if self.temporal is not None:
                sample = self.temporal.index_at(mode.playback.index)
                if selected and index == selected[0]:
                    sample = self.pair_events[selected][0]
                active = self.temporal.point_activity[index][sample]
                rgba = np.empty((len(active), 4), dtype=np.uint8)
                rgba[:, :3] = np.asarray(colors.get(index, (0.6, 0.6, 0.6))) * 255
                rgba[:, 3] = active * 255
                marker.GetMapper().GetInput().GetPointData().SetScalars(
                    numpy_to_vtk(rgba, deep=True)
                )
                marker.GetMapper().SetColorModeToDirectScalars()
                marker.SetVisibility(visible and self.show_points and active.any())
        if self.temporal is not None:
            self._show_cached_tracks(visible)
            return
        strength = None
        if self.actors:
            frame = mode.read_surface_frame(mode.playback.index, mode.series.base_representation)
            strength = mode.series.activation_strength(frame)
        for pair, actor in self.actors.items():
            actor.SetVisibility(visible)
            activation = max(strength[self.seeds[i][2]].max() for i in pair)
            actor.GetProperty().SetOpacity(0.02 + 0.98 * activation)
