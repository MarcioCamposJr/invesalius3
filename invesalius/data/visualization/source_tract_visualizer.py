"""Experimental A–B tract display driven by the original cortical source signals."""

from itertools import combinations

import numpy as np
import wx
from scipy.sparse import csr_matrix
from scipy.spatial import cKDTree
from vtkmodules.util.numpy_support import numpy_to_vtk, numpy_to_vtkIdTypeArray, vtk_to_numpy
from vtkmodules.vtkCommonCore import vtkPoints
from vtkmodules.vtkCommonDataModel import vtkCellArray, vtkPolyData
from vtkmodules.vtkFiltersCore import vtkAppendPolyData, vtkTubeFilter
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
        self.track_contacts = {}
        self.render_counts = {}
        self.render_values = {}
        self._color_nodes = None
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
        self.track_contacts.clear()
        self.render_counts.clear()
        self.render_values.clear()
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
        self.track_contacts.clear()
        self.render_counts.clear()
        self.render_values.clear()
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
            contacts = []
            for region, endpoint in zip(pair, (0, -1)):
                centers = self.seeds[region][1]
                if paths:
                    hits = cKDTree(centers).query_ball_point(
                        np.asarray([path[endpoint] for path in paths]), self.cache_radius + 1e-6
                    )
                    offsets = np.r_[0, np.cumsum([len(indices) for indices in hits])]
                    indices = (
                        np.concatenate(hits).astype(int) if offsets[-1] else np.empty(0, dtype=int)
                    )
                    contacts.append(
                        csr_matrix(
                            (np.ones(len(indices), dtype=bool), indices, offsets),
                            shape=(len(paths), len(centers)),
                        )
                    )
                else:
                    contacts.append(csr_matrix((0, len(centers)), dtype=bool))
            self.track_contacts[pair] = tuple(contacts)
            self.render_counts.pop(pair, None)
            self.render_values.pop(pair, None)
        self.attempted_pairs.update(completed)
        self.counts = {pair: len(paths) for pair, paths in self.track_cache.items()}
        self.count = sum(self.counts.values())
        Publisher.sendMessage("Source reconstruction changed", mode=self.mode)
        Publisher.sendMessage("Render volume viewer")

    def _paths_actor(self, paths):
        lengths = np.asarray([len(path) for path in paths])
        points = vtkPoints()
        points.SetData(numpy_to_vtk(np.vstack(paths), deep=True))
        lines = vtkCellArray()
        lines.SetData(
            numpy_to_vtkIdTypeArray(np.r_[0, lengths.cumsum()].astype(np.int64), deep=True),
            numpy_to_vtkIdTypeArray(np.arange(lengths.sum(), dtype=np.int64), deep=True),
        )
        data = vtkPolyData()
        data.SetPoints(points)
        data.SetLines(lines)
        data.GetPointData().SetScalars(
            numpy_to_vtk(np.zeros((lengths.sum(), 4), dtype=np.uint8), deep=True)
        )
        positions = []
        for path in paths:
            distance = np.r_[0.0, np.linalg.norm(np.diff(path, axis=0), axis=1).cumsum()]
            positions.append(distance / distance[-1] if distance[-1] else distance)
        for name, values in (
            ("SourceFiberId", np.repeat(np.arange(len(paths), dtype=np.int32), lengths)),
            ("SourceFiberPosition", np.concatenate(positions).astype(np.float32)),
        ):
            array = numpy_to_vtk(values, deep=True)
            array.SetName(name)
            data.GetFieldData().AddArray(array)
        point_ids = numpy_to_vtk(np.arange(lengths.sum(), dtype=np.int64), deep=True)
        point_ids.SetName("SourcePointId")
        data.GetPointData().AddArray(point_ids)
        tubes = vtkTubeFilter()
        tubes.SetInputData(data)
        tubes.SetRadius(0.5)
        tubes.SetNumberOfSides(4)
        tubes.Update()
        return self._actor(tubes.GetOutput())

    @staticmethod
    def _contact_values(contacts, values, count):
        offsets = contacts.indptr[: count + 1]
        rows = np.repeat(np.arange(count), np.diff(offsets))
        samples = values[contacts.indices[: offsets[-1]]]
        magnitude = np.zeros(count)
        np.maximum.at(magnitude, rows, np.abs(samples))
        result = np.zeros(count)
        selected = np.abs(samples) == magnitude[rows]
        result[rows[selected]] = samples[selected]
        return result

    def _show_cached_tracks(self, visible):
        if not self.track_cache:
            return
        targets = source_pair_targets(
            [seed[1] for seed in self.seeds], self.fiber_multiplier, self.max_fibers
        )
        signals = {}
        sample = self.temporal.index_at(self.mode.playback.index)
        for pair, (earlier, _later) in self.pair_events.items():
            key = tuple(sorted(pair))
            if key not in self.track_cache:
                continue
            count = min(len(self.track_cache[key]), targets[key])
            left, right = (
                self._contact_values(
                    self.track_contacts[key][key.index(region)],
                    np.where(
                        self.temporal.point_activity[region][at],
                        self.temporal.point_values[region][at],
                        0.0,
                    ),
                    count,
                )
                for region, at in ((pair[0], earlier), (pair[1], sample))
            )
            active = (self.series.activation_strength(left) > 0) & (
                self.series.activation_strength(right) > 0
            )
            # Stored paths always run from the lower region index to the higher one.
            values = np.column_stack((left, right) if pair == key else (right, left))
            values[~active] = 0.0
            previous = signals.get(key, np.zeros((count, 2)))
            stronger = np.abs(values).max(axis=1) > np.abs(previous).max(axis=1)
            signals[key] = np.where(stronger[:, None], values, previous)
        lookup = self.mode.surface_map.lookup_table
        factor = self.series.display_scale[0]
        appearance = lookup.GetMTime(), factor, self.series.color_limits, self.series.color_range
        for pair, paths in self.track_cache.items():
            count = min(len(paths), targets[pair])
            if not count:
                continue
            if self.render_counts.get(pair) != count:
                actor = self._paths_actor(paths[:count])
                if pair in self.actors:
                    Publisher.sendMessage(
                        "Remove actors " + str(const.SURFACE), actors=[self.actors[pair]]
                    )
                self.actors[pair] = actor
                self.render_counts[pair] = count
                self.render_values.pop(pair, None)
            actor = self.actors[pair]
            actor.SetVisibility(visible)
            actor.GetProperty().SetOpacity(1)
            if not visible:
                continue
            values = signals.get(pair, np.zeros((count, 2)))
            previous = self.render_values.get(pair)
            if (
                previous is not None
                and previous[0] == appearance
                and np.array_equal(previous[1], values)
            ):
                continue
            polydata = actor.GetMapper().GetInput()
            fiber_ids = vtk_to_numpy(polydata.GetFieldData().GetArray("SourceFiberId"))
            position = vtk_to_numpy(polydata.GetFieldData().GetArray("SourceFiberPosition"))
            # Color each centerline point once, then reuse it around the tube's cross-section.
            gradient = values[fiber_ids, 0] + position * (values[:, 1] - values[:, 0])[fiber_ids]
            if self._color_nodes is None or self._color_nodes[0] != lookup.GetMTime():
                nodes = np.empty((lookup.GetSize(), 6))
                for index, node in enumerate(nodes):
                    lookup.GetNodeValue(index, node)
                self._color_nodes = lookup.GetMTime(), nodes
            nodes = self._color_nodes[1]
            display_values = gradient * factor
            rgba = np.empty((len(gradient), 4), dtype=np.uint8)
            for channel in range(3):
                rgba[:, channel] = np.rint(
                    255 * np.interp(display_values, nodes[:, 0], nodes[:, channel + 1])
                )
            rgba[:, 3] = np.rint(255 * (0.02 + 0.98 * self.series.activation_strength(gradient)))
            data = polydata.GetPointData()
            ids = vtk_to_numpy(data.GetArray("SourcePointId"))
            colors = data.GetScalars()
            np.take(rgba, ids, axis=0, out=vtk_to_numpy(colors), mode="clip")
            colors.Modified()
            self.render_values[pair] = appearance, values

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
