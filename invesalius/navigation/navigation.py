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

import logging
import queue
import threading
import time

import numpy as np
import wx

import invesalius.constants as const
import invesalius.data.bases as db
import invesalius.data.coregistration as dcr
import invesalius.data.e_field as e_field
import invesalius.data.polydata_utils as pu
import invesalius.data.serial_port_connection as spc
import invesalius.data.slice_ as sl
import invesalius.data.tractography as dti
import invesalius.data.transformations as tr
import invesalius.data.vtk_utils as vtk_utils
import invesalius.project as prj
import invesalius.session as ses
from invesalius.data.markers.marker import MarkerType
from invesalius.data.visualization.mep_visualizer import MEPVisualizer
from invesalius.i18n import tr as _
from invesalius.navigation.image import Image
from invesalius.navigation.iterativeclosestpoint import IterativeClosestPoint
from invesalius.navigation.markers import MarkersControl
from invesalius.navigation.robot import Robots
from invesalius.navigation.tracker import Tracker
from invesalius.net.neuronavigation_api import NeuronavigationApi
from invesalius.net.pedal_connection import PedalConnector
from invesalius.pubsub import pub as Publisher
from invesalius.utils import Singleton

NAVIGATION_STOP_TIMEOUT_SECONDS = 2.0


class NavigationHub(metaclass=Singleton):
    """
    Class to initialize and store references to navigation components.
    """

    def __init__(self, window=None):
        self.tracker = Tracker()
        self.image = Image()
        self.icp = IterativeClosestPoint()
        self.neuronavigation_api = NeuronavigationApi()
        self.pedal_connector = PedalConnector(self.neuronavigation_api, window)
        self.robots = Robots()
        self.navigation = Navigation(
            pedal_connector=self.pedal_connector,
            neuronavigation_api=self.neuronavigation_api,
        )

        session = ses.Session()
        n_coils = session.GetConfig("navigation", {}).get("n_coils", 1)

        # Initialize the robots based on number of coils config
        while len(self.robots.robots_by_id) < n_coils:
            robot = self.robots.AddRobot(
                tracker=self.tracker,
                navigation=self.navigation,
                icp=self.icp,
            )
            # Break when we have 2 robots
            if len(self.robots.robots_by_id) == 2:
                break
        self.markers = MarkersControl(navigation=self.navigation)
        self.mep_visualizer = MEPVisualizer()
        Publisher.sendMessage("Add navigation context to interactive shell")


class QueueCustom(queue.Queue):
    """
    A custom queue subclass that provides a :meth:`clear` method.
    https://stackoverflow.com/questions/6517953/clear-all-items-from-the-queue
    Pose producers can replace waiting entries with ``put_latest`` without
    changing the task accounting of an item already being processed.
    """

    def clear(self):
        """
        Clears all items from the queue.
        """

        with self.mutex:
            unfinished = self.unfinished_tasks - len(self.queue)
            if unfinished <= 0:
                if unfinished < 0:
                    raise ValueError("task_done() called too many times")
                self.all_tasks_done.notify_all()
            self.unfinished_tasks = unfinished
            self.queue.clear()
            self.not_full.notify_all()

    def put_latest(self, item):
        with self.not_full:
            while self._qsize():
                self._get()
                self.unfinished_tasks -= 1
            self._put(item)
            self.unfinished_tasks += 1
            self.not_empty.notify()


class UpdateNavigationScene(threading.Thread):
    def __init__(self, vis_queues, vis_components, event, sle, neuronavigation_api):
        """Class (threading) to update the navigation scene with all graphical elements.

        Sleep function in run method is used to avoid blocking GUI and more fluent, real-time navigation

        :param affine_vtk: Affine matrix in vtkMatrix4x4 instance to update objects position in 3D scene
        :type affine_vtk: vtkMatrix4x4
        :param visualization_queue: Queue instance that manage coordinates to be visualized
        :type visualization_queue: queue.Queue
        :param event: Threading event to coordinate when tasks as done and allow UI release
        :type event: threading.Event
        :param sle: Sleep pause in seconds
        :type sle: float
        :param neuronavigation_api: An API object for communicating the coil position.
        :type neuronavigation_api: invesalius.net.neuronavigation_api.NeuronavigationAPI
        """

        threading.Thread.__init__(self, name="UpdateScene")
        (
            self.serial_port_enabled,
            self.view_tracts,
            self.peel_loaded,
            self.e_field_loaded,
            self.plot_efield_vectors,
        ) = vis_components
        (
            self.coord_queue,
            self.serial_port_queue,
            self.tracts_queue,
            self.e_field_norms_queue,
            self.e_field_IDs_queue,
        ) = vis_queues
        self.sle = sle
        self.event = event
        self.neuronavigation_api = neuronavigation_api
        self.navigation = Navigation()
        self._slice_render_interval = max(self.sle, 1.0 / 10.0)
        self._dispatch_lock = threading.Lock()
        self._pending_update = None
        self._dispatch_timer = None
        self._last_dispatch = 0.0
        self._dispatch_pending = False
        self._last_slice_render = 0.0

    def _dispatch_updates(
        self,
        *,
        update_pose,
        slice_render,
        coord,
        probe_visible,
        probe_coord,
        probe_m_img,
        coil_visible,
        main_coil,
        coords,
        m_imgs,
        tracts_payload,
        trigger_on,
        enorm_data,
    ):
        if self.event.is_set():
            return
        if tracts_payload is not None:
            bundle, affine_vtk, coord_offset, coord_offset_w = tracts_payload
            Publisher.sendMessage("Remove tracts", coil_name=main_coil)
            Publisher.sendMessage(
                "Update tracts",
                coil_name=main_coil,
                root=bundle,
                affine_vtk=affine_vtk,
                coord_offset=coord_offset,
                coord_offset_w=coord_offset_w,
            )

        if update_pose:
            Publisher.sendMessage("Update slices position", position=coord[:3])
            Publisher.sendMessage("Set cross focal point", position=coord)
            Publisher.sendMessage(
                "Update volume viewer pointer",
                position=[coord[0], -coord[1], coord[2]],
            )

            if coil_visible:
                Publisher.sendMessage("Update coil poses", m_imgs=m_imgs, coords=coords)
                Publisher.sendMessage(
                    "Update coil pose",
                    m_img=m_imgs[main_coil],
                    coord=coords[main_coil],
                )
                Publisher.sendMessage(
                    "From Neuronavigation: Send coil pose",
                    coord=list(coords[main_coil]),
                )
                Publisher.sendMessage(
                    "Update object arrow matrix",
                    coil_name=main_coil,
                    m_img=m_imgs[main_coil],
                    coord=coords[main_coil],
                    flag=self.peel_loaded,
                )

                if self.e_field_loaded:
                    Publisher.sendMessage(
                        "Update point location for e-field calculation",
                        m_img=m_imgs[main_coil],
                        coord=coords[main_coil],
                        queue_IDs=self.e_field_IDs_queue,
                    )
            if probe_visible:
                Publisher.sendMessage(
                    "Update probe pose",
                    m_img=probe_m_img,
                    coord=probe_coord,
                )

        for _trigger in range(trigger_on):
            Publisher.sendMessage("Create marker", marker_type=MarkerType.COIL_POSE)

        if coil_visible and self.e_field_loaded:
            if enorm_data is not None:
                Publisher.sendMessage(
                    "Get enorm",
                    enorm_data=enorm_data,
                    plot_vector=self.plot_efield_vectors,
                    current_revision=self.navigation.e_field_revision,
                )
            if self.view_tracts:
                Publisher.sendMessage(
                    "Update tract seed based efield",
                    coord_tracts_queue=self.navigation.coord_tracts_queue,
                    fallback_m_img=m_imgs[main_coil],
                    current_revision=self.navigation.e_field_revision,
                )

        Publisher.sendMessage("Render volume viewer")
        if slice_render:
            Publisher.sendMessage("Update slice viewer")

    def _queue_update(self, update):
        with self._dispatch_lock:
            if self.event.is_set():
                return
            previous = self._pending_update
            if previous is not None:
                update["update_pose"] |= previous["update_pose"]
                update["trigger_on"] += previous["trigger_on"]
                if previous["main_coil"] == update["main_coil"]:
                    for key in ("tracts_payload", "enorm_data"):
                        if update[key] is None:
                            update[key] = previous[key]
            self._pending_update = update
            schedule = not self._dispatch_pending
            self._dispatch_pending = True
        if schedule:
            wx.CallAfter(self._flush_updates)

    def _flush_updates(self):
        self._dispatch_timer = None
        if self.event.is_set():
            self.CancelDispatch()
            return
        interval = max(self.sle, 1.0 / 100.0)
        delay = interval - (time.perf_counter() - self._last_dispatch)
        if delay > 0:
            self._dispatch_timer = wx.CallLater(max(1, int(delay * 1000) + 1), self._flush_updates)
            return
        with self._dispatch_lock:
            update = self._pending_update
            self._pending_update = None
        try:
            if update is not None:
                now = time.perf_counter()
                self._last_dispatch = now
                update["slice_render"] = (
                    now - self._last_slice_render >= self._slice_render_interval
                )
                if update["slice_render"]:
                    self._last_slice_render = now
                self._dispatch_updates(**update)
        finally:
            with self._dispatch_lock:
                schedule = self._pending_update is not None and not self.event.is_set()
                self._dispatch_pending = schedule
            if schedule:
                wx.CallAfter(self._flush_updates)

    def CancelDispatch(self):
        if self._dispatch_timer is not None:
            self._dispatch_timer.Stop()
            self._dispatch_timer = None
        with self._dispatch_lock:
            self._pending_update = None
            self._dispatch_pending = False

    def run(self):
        latest = None
        while not self.event.is_set():
            new_pose = False
            try:
                latest = self.coord_queue.get(timeout=0.05)
            except queue.Empty:
                pass
            else:
                self.coord_queue.task_done()
                new_pose = True
            if self.event.is_set() or latest is None:
                continue

            coords, marker_visibilities, m_imgs = latest
            main_coil = self.navigation.main_coil
            coil_visible = (
                any(marker_visibilities[2:]) and main_coil in coords and main_coil in m_imgs
            )
            track_this = main_coil if self.navigation.track_coil else "probe"
            coord = coords.get(track_this)
            if coord is None:
                continue
            probe_coord, probe_m_img = coords["probe"], m_imgs["probe"]
            coords = {name: pose for name, pose in coords.items() if name != "probe"}
            m_imgs = {name: matrix for name, matrix in m_imgs.items() if name != "probe"}

            tracts_payload = None
            if self.view_tracts:
                try:
                    tracts_payload = self.tracts_queue.get_nowait()
                except queue.Empty:
                    pass
                else:
                    self.tracts_queue.task_done()

            trigger_on = 0
            if self.serial_port_enabled:
                try:
                    trigger_on = int(bool(self.serial_port_queue.get_nowait()))
                except queue.Empty:
                    pass
                else:
                    self.serial_port_queue.task_done()

            enorm_data = None
            if coil_visible and self.e_field_loaded:
                try:
                    enorm_data = self.e_field_norms_queue.get_nowait()
                except queue.Empty:
                    pass
                else:
                    self.e_field_norms_queue.task_done()

            if new_pose or tracts_payload is not None or trigger_on or enorm_data is not None:
                self._queue_update(
                    dict(
                        update_pose=new_pose,
                        slice_render=False,
                        coord=coord,
                        probe_visible=marker_visibilities[0],
                        probe_coord=probe_coord,
                        probe_m_img=probe_m_img,
                        coil_visible=coil_visible,
                        main_coil=main_coil,
                        coords=coords,
                        m_imgs=m_imgs,
                        tracts_payload=tracts_payload,
                        trigger_on=trigger_on,
                        enorm_data=enorm_data,
                    )
                )


class Navigation(metaclass=Singleton):
    def __init__(self, pedal_connector, neuronavigation_api):
        if getattr(self, "_workers", None):
            self.StopNavigation()
        self.pedal_connector = pedal_connector
        self.neuronavigation_api = neuronavigation_api

        self.target = None
        self.n_coils = 1
        self.coil_registrations = {}
        self.track_coil = False
        self.main_coil = None  # Which coil to track with pointer
        self.m_change = None
        self.r_stylus = None
        self.obj_datas = None  # This is accessed by the robot, gets value at StartNavigation

        self.all_fiducials = np.zeros((6, 6))
        self.event = threading.Event()
        self.event.set()
        self._workers = []
        self._CreateQueues()

        # Tracker parameters
        self.ref_mode_id = const.DEFAULT_REF_MODE

        self.e_field_loaded = False
        self.plot_efield_vectors = False
        self.debug_efield_enorm = None
        self.e_field_revision = 0

        # Tractography parameters
        self.trk_inp = None
        self.trekker = None
        self.n_threads = None
        self.view_tracts = False
        self.peel_loaded = False
        self.enable_act = False
        self.act_data = None
        self.n_tracts = const.N_TRACTS

        # Sleep parameters
        session = ses.Session()
        sleep_nav = session.GetConfig("sleep_nav", const.SLEEP_NAVIGATION)
        self.sleep_nav = sleep_nav

        self.seed_offset = const.SEED_OFFSET
        self.seed_radius = const.SEED_RADIUS

        # Serial port
        self.serial_port_in_use = False
        self.com_port = None
        self.baud_rate = None
        self.serial_port_connection = None

        # During navigation
        self.lock_to_target = False
        self.coil_at_target = False

        self.LoadConfig()

        self.__bind_events()

    def _CreateQueues(self):
        self.coord_queue = QueueCustom(maxsize=1)
        self.object_at_target_queue = QueueCustom(maxsize=1)
        self.efield_queue = QueueCustom(maxsize=1)
        self.e_field_norms_queue = QueueCustom(maxsize=1)
        self.e_field_IDs_queue = QueueCustom(maxsize=1)
        self.serial_port_queue = QueueCustom(maxsize=1)
        self.coord_tracts_queue = QueueCustom(maxsize=1)
        self.tracts_queue = QueueCustom(maxsize=1)

    def __bind_events(self):
        Publisher.subscribe(self.CoilAtTarget, "Coil at target")
        Publisher.subscribe(self.SetNoOfCoils, "Reset coil selection")
        Publisher.subscribe(self.SelectCoil, "Select coil")
        Publisher.subscribe(self.UpdateSerialPort, "Update serial port")
        Publisher.subscribe(self.TrackObject, "Track object")

    def SaveConfig(self, key=None, value=None):
        """
        Save either the whole state, or a specific key-value pair into navigation configuration
        """
        session = ses.Session()
        if key is None:  # Save the whole state
            state = {
                "selected_coils": list(self.coil_registrations),
                "n_coils": self.n_coils,
                "track_coil": self.track_coil,
            }
            if self.main_coil is not None:
                state["main_coil"] = self.main_coil
            if self.r_stylus is not None:
                state["r_stylus"] = self.r_stylus.tolist()
            session.SetConfig("navigation", state)

        elif value is not None:  # Save a specific state attribute
            state = session.GetConfig("navigation", {})
            state[key] = value
            session.SetConfig("navigation", state)

    def LoadConfig(self):
        session = ses.Session()
        state = session.GetConfig("navigation")

        # Get the dict of all coil_registrations saved to config file
        saved_coil_registrations = session.GetConfig("coil_registrations")

        if state is not None:
            self.main_coil = state.get("main_coil", None)
            self.n_coils = state.get("n_coils", 1)
            if self.n_coils == 1:
                self.main_coil = "default_coil"

            self.track_coil = state.get("track_coil", False)

            # Try to load selected_coils (the list of names of coils to use for navigation)
            if ("selected_coils" in state) and (saved_coil_registrations is not None):
                selected_coils = state["selected_coils"]
                self.coil_registrations = {
                    coil_name: saved_coil_registrations[coil_name]
                    for coil_name in selected_coils
                    if coil_name in saved_coil_registrations
                }
                if self.coil_registrations:
                    self.main_coil = self.main_coil or next(iter(self.coil_registrations))

            # Try to load stylus orientation data
            if "r_stylus" in state:
                self.r_stylus = np.array(state["r_stylus"])

    def CoilSelectionDone(self):
        return len(self.coil_registrations) == self.n_coils

    def SelectCoil(self, coil_name, coil_registration):
        if coil_registration is not None:  # Add the coil to selection
            self.coil_registrations[coil_name] = coil_registration
            if self.main_coil is None:
                self.main_coil = coil_name
        else:  # Remove the coil from selection
            self.coil_registrations.pop(coil_name, None)
            if self.main_coil == coil_name:
                self.main_coil = None
                self.SaveConfig()

        self.SaveConfig()

    def CoilAtTarget(self, state, coil_name=None):
        self.coil_at_target = state

    def SetTarget(self, marker):
        coord = marker.position + marker.orientation
        coord[1] = -coord[1]
        self.target = coord

    def UnsetTarget(self):
        self.target = None

    def UpdateNavSleep(self, sleep):
        self.sleep_nav = sleep
        # self.serial_port_connection.sleep_nav = sleep

    def UpdateSerialPort(self, serial_port_in_use, com_port=None, baud_rate=None):
        self.serial_port_in_use = serial_port_in_use
        self.com_port = com_port
        self.baud_rate = baud_rate

    def TrackObject(self, enabled=False):
        self.track_coil = enabled
        self.SaveConfig()

    def SetLockToTarget(self, value):
        self.lock_to_target = value

    def SetNoOfCoils(self, n_coils, clear_all=False):
        self.n_coils = n_coils
        self.SaveConfig("n_coils", n_coils)

        # Reset coil selection
        self.coil_registrations = {}
        self.main_coil = None
        self.SaveConfig()

    def SetMainCoil(self, main_coil):
        self.main_coil = main_coil
        self.SaveConfig("main_coil", main_coil)

        # Send the polydata of the main coil to the connection
        polydata = pu.LoadPolydata(self.coil_registrations[main_coil]["path"])
        self.neuronavigation_api.update_coil_mesh(polydata)

    def SetReferenceMode(self, value):
        self.ref_mode_id = value

    def GetReferenceMode(self):
        return self.ref_mode_id

    def UpdateFiducialRegistrationError(self, tracker, image):
        tracker_fiducials, tracker_fiducials_raw = tracker.GetTrackerFiducials()
        image_fiducials = image.GetImageFiducials()

        self.all_fiducials = np.vstack([image_fiducials, tracker_fiducials])

        self.fre = db.calculate_fre(
            tracker_fiducials_raw, self.all_fiducials, self.ref_mode_id, self.m_change
        )

    def GetFiducialRegistrationError(self, icp):
        fre = icp.icp_fre if icp.use_icp else self.fre
        return fre, fre <= const.FIDUCIAL_REGISTRATION_ERROR_THRESHOLD

    def PedalStateChanged(self, state):
        if not self.serial_port_in_use:
            return

        permission_to_stimulate = (
            self.lock_to_target and self.coil_at_target
        ) or not self.lock_to_target

        if state and permission_to_stimulate:
            self.serial_port_connection.SendPulse()

    def EstimateTrackerToInVTransformationMatrix(self, tracker, image):
        tracker_fiducials, tracker_fiducials_raw = tracker.GetTrackerFiducials()
        image_fiducials = image.GetImageFiducials()

        self.all_fiducials = np.vstack([image_fiducials, tracker_fiducials])

        self.m_change = tr.affine_matrix_from_points(
            self.all_fiducials[3:, :].T, self.all_fiducials[:3, :].T, shear=False, scale=False
        )

    def OnRecordStylusOrientation(self, coord_raw):
        if self.m_change is not None:
            m_probe = dcr.compute_marker_transformation(coord_raw, 0)

            # transform probe to reference system if dynamic ref_mode
            if self.ref_mode_id:
                up_trk = dcr.object_to_reference(coord_raw, m_probe)[:3, :3]
            else:
                up_trk = m_probe[:3, :3]

            # up_trk: orientation 'stylus pointing up along head' in tracker space
            # up_vtk: orientation 'stylus pointing up along head' in vtk space
            up_vtk = tr.euler_matrix(*np.radians([90.0, 0.0, 0.0]), axes="rxyz")[:3, :3]

            # Rotate 90 degrees around z-axis from trk system where stylus points in x-axis
            # to vtk-system where stylus points in y-axis
            R = tr.euler_matrix(*np.radians([0, 0, -90]), axes="rxyz")[:3, :3]

            # Reflect/flip the orientation in the vtk x-axis
            x_flip = np.eye(3)
            x_flip[0] = -x_flip[0]

            # Rotation from tracker to VTK coordinate system (apply x_flip to orient stylus with NIFTI file)
            self.r_stylus = x_flip @ up_vtk @ np.linalg.inv(R @ up_trk @ np.linalg.inv(R))
            # Save r_stylus to config file
            self.SaveConfig("r_stylus", self.r_stylus.tolist())
            return True
        else:
            return False

    def StartNavigation(self, tracker, icp):
        if self._workers:
            if not self.event.is_set():
                return
            self.StopNavigation()

        self.event = threading.Event()
        self._CreateQueues()
        self.serial_port_connection = None
        try:
            self._StartNavigation(tracker, icp)
        except Exception:
            self.StopNavigation()
            raise
        finally:
            if not self._workers:
                self.event.set()

    def _StartNavigation(self, tracker, icp):
        jobs_list = self._workers

        vis_components = [
            self.serial_port_in_use,
            self.view_tracts,
            self.peel_loaded,
            self.e_field_loaded,
            self.plot_efield_vectors,
        ]
        vis_queues = [
            self.coord_queue,
            self.serial_port_queue,
            self.tracts_queue,
            self.e_field_norms_queue,
            self.e_field_IDs_queue,
        ]

        if not self.CoilSelectionDone():
            wx.MessageBox(
                _(
                    f"No. of coils to track is {self.n_coils}, but no. of coils registered is {len(self.coil_registrations)}"
                ),
                _("InVesalius 3"),
            )
        else:
            # Pre-compute obj_datas: data/matrices for each coil to be used in coregistration
            # data is accessed from dict by coil name
            obj_datas = {}
            for coil_name in self.coil_registrations:
                if self.ref_mode_id:
                    coord_raw, marker_visibilities = tracker.TrackerCoordinates.GetCoordinates()
                else:
                    coord_raw = np.array([None])

                coil_registration = self.coil_registrations[coil_name]
                obj_id = coil_registration["obj_id"]

                # Check that the object index is in range of Tracker coordinates
                if obj_id >= coord_raw.shape[0]:
                    wx.MessageBox(
                        _(
                            f"Coil index {obj_id} for coil {coil_name} is out of range of current Tracker! Change the coil index in Preferences - TMS Coil tab"
                        ),
                        _("InVesalius 3"),
                    )
                    return

                # Pre-compute obj_datas: data/matrices for each coil to be used in coregistration
                obj_data = (obj_id,) + db.object_registration(
                    np.array(coil_registration["fiducials"]),
                    np.array(coil_registration["orientations"]),
                    coord_raw,
                    self.m_change,
                )
                obj_datas[coil_name] = obj_data

            self.obj_datas = obj_datas

            coreg_data = [self.m_change, self.r_stylus]

            queues = [
                self.coord_queue,
                self.coord_tracts_queue,
                self.object_at_target_queue,
                self.efield_queue,
            ]
            jobs_list.append(
                dcr.CoordinateCorregistrate(
                    self.ref_mode_id,
                    tracker,
                    coreg_data,
                    obj_datas,
                    self.view_tracts,
                    queues,
                    self.event,
                    self.sleep_nav,
                    tracker.tracker_id,
                    self.target,
                    icp,
                    self.e_field_loaded,
                    self,
                )
            )

            # TODO: Test the serial port thread
            if self.serial_port_in_use:
                self.serial_port_connection = spc.SerialPortConnection(
                    com_port=self.com_port,
                    baud_rate=self.baud_rate,
                    serial_port_queue=self.serial_port_queue,
                    event=self.event,
                    sleep_nav=self.sleep_nav,
                )
                if self.serial_port_connection.Connect():
                    # Only add to jobs_list if connection succeeded
                    jobs_list.append(self.serial_port_connection)
                else:
                    # Connection failed - show error and disable serial port
                    wx.MessageBox(
                        "Failed to connect to serial port. Navigation will continue without serial port support.",
                        "InVesalius 3",
                    )
                    self.serial_port_in_use = False

            if self.view_tracts:
                # initialize Trekker parameters
                slic = sl.Slice()
                affine, affine_vtk, img_shift = slic.get_world_to_invesalius_vtk_affine()

                Publisher.sendMessage("Update marker offset state", create=True)

                self.trk_inp = (
                    self.trekker,
                    affine,
                    self.seed_offset,
                    self.n_tracts,
                    self.seed_radius,
                    self.n_threads,
                    self.act_data,
                    affine_vtk,
                    img_shift,
                )
                # print("Appending the tract computation thread!")
                queues = [self.coord_tracts_queue, self.tracts_queue]
                if self.enable_act:
                    jobs_list.append(
                        dti.ComputeTractsACTThread(self.trk_inp, queues, self.event, self.sleep_nav)
                    )
                else:
                    jobs_list.append(
                        dti.ComputeTractsThread(self.trk_inp, queues, self.event, self.sleep_nav)
                    )

            if self.e_field_loaded:
                queues = [self.efield_queue, self.e_field_norms_queue, self.e_field_IDs_queue]
                jobs_list.append(
                    e_field.Visualize_E_field_Thread(
                        queues,
                        self.event,
                        2 * self.sleep_nav,
                        self.neuronavigation_api,
                        self.debug_efield_enorm,
                        self.plot_efield_vectors,
                    )
                )

            vis_components[0] = self.serial_port_in_use
            jobs_list.append(
                UpdateNavigationScene(
                    vis_queues=vis_queues,
                    vis_components=vis_components,
                    event=self.event,
                    sle=self.sleep_nav,
                    neuronavigation_api=self.neuronavigation_api,
                )
            )

            Publisher.sendMessage("Navigation status", nav_status=True, vis_status=vis_components)
            for worker in jobs_list:
                run = worker.run
                event = self.event
                worker.run = lambda run=run, event=event: self._RunWorker(run, event)
                worker.start()

            if not self.event.is_set():
                self.pedal_connector.add_callback("navigation", self.PedalStateChanged)

    def _RunWorker(self, run, event):
        try:
            run()
        except Exception as error:
            logging.exception("Navigation worker failed")
            event.set()
            wx.CallAfter(self._OnWorkerError, event, str(error))

    def _OnWorkerError(self, event, message):
        if event is self.event and self._workers:
            Publisher.sendMessage("Stop navigation")
            wx.MessageBox(
                _("Navigation stopped because a worker failed: {error}").format(error=message),
                _("Navigation"),
                wx.OK | wx.ICON_ERROR,
            )

    def StopNavigation(self):
        self.event.set()
        for worker in self._workers:
            if isinstance(worker, UpdateNavigationScene):
                worker.CancelDispatch()

        self.pedal_connector.remove_callback("navigation")

        deadline = time.monotonic() + NAVIGATION_STOP_TIMEOUT_SECONDS
        for worker in self._workers:
            if worker.ident is not None:
                worker.join(timeout=max(0.0, deadline - time.monotonic()))
        self._workers = [worker for worker in self._workers if worker.is_alive()]
        if self._workers:
            raise TimeoutError(_("Navigation workers are still stopping. Try again shortly."))

        if self.serial_port_connection is not None and self.serial_port_connection.ident is None:
            self.serial_port_connection.Disconnect()
        for pending in (
            self.coord_queue,
            self.object_at_target_queue,
            self.serial_port_queue,
            self.coord_tracts_queue,
            self.tracts_queue,
            self.efield_queue,
            self.e_field_norms_queue,
            self.e_field_IDs_queue,
        ):
            pending.clear()

        vis_components = [
            self.serial_port_in_use,
            self.view_tracts,
            self.peel_loaded,
            self.e_field_loaded,
            self.plot_efield_vectors,
        ]
        Publisher.sendMessage("Navigation status", nav_status=False, vis_status=vis_components)

    def MarkEfieldParametersChanged(self):
        self.e_field_revision += 1
