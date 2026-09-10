"""Switch general and navigation scenes inside one wx/VTK viewport."""

import wx
from vtkmodules.wx.wxVTKRenderWindowInteractor import wxVTKRenderWindowInteractor

import invesalius.constants as const
import invesalius.session as ses
from invesalius.data.viewer_events import ViewEventRouter
from invesalius.data.viewer_navigation import NavigationView
from invesalius.data.viewer_volume import VolumeView
from invesalius.i18n import tr as _
from invesalius.pubsub import pub as Publisher

# Keep project data synchronized without sending user interaction or pose updates
# to a detached scene. The actors supplied by the project remain shared.
PROJECT_TOPICS = {
    "Load surface actor into viewer",
    "Remove surface actor from viewer",
    "Load volume into viewer",
    "Unload volume",
    "Remove Volume",
    "Add actors " + str(const.SURFACE),
    "Remove actors " + str(const.SURFACE),
    "Remove all volume actors",
    "Close project data",
    "Load slice plane",
}

NAVIGATION_DATA_TOPICS = {
    "Navigation status",
    "Add marker",
    "Update marker",
    "Delete markers",
    "Delete marker",
    "Set target",
    "Unset target",
    "Set target transparency",
    "Set target mode",
    "Reset coil selection",
    "Select coil",
    "Set vector field",
    "Update vector field",
    "Remove sensors ID",
}


class VolumeViewHost(wx.Panel):
    """Own one interactor and keep each scene's camera and renderers alive."""

    def __init__(self, parent):
        super().__init__(parent)
        self._disposed = False
        self.interactor = wxVTKRenderWindowInteractor(self, -1, size=self.GetSize())
        sizer = wx.BoxSizer(wx.VERTICAL)
        sizer.Add(self.interactor, 1, wx.EXPAND)
        self.SetSizer(sizer)

        self.volume_events = ViewEventRouter(PROJECT_TOPICS)
        self.volume_view = VolumeView(self, interactor=self.interactor)
        self.volume_events.manage_view(self.volume_view)

        # NavigationView inherits the general volume tools and adds navigation state.
        self.navigation_view = NavigationView(self, interactor=self.interactor)
        self.navigation_events = ViewEventRouter(PROJECT_TOPICS | NAVIGATION_DATA_TOPICS)
        self.navigation_events.manage_view(self.navigation_view)
        self.navigation_views = [self.navigation_view]
        self.navigation_event_routers = [self.navigation_events]

        self.volume_view.Hide()
        self.navigation_view.Hide()
        self.active_views = []
        self.active_view = None
        self._navigation_visited = False
        self.Bind(wx.EVT_SIZE, self.OnSize)
        self.Bind(wx.EVT_WINDOW_DESTROY, self.OnDestroy)
        Publisher.subscribe(self.SetNavigationMode, "Set navigation mode")
        Publisher.subscribe(self.OnCloseProject, "Close project data")
        Publisher.subscribe(self.dispose, "Exit")
        self.SetNavigationMode(ses.Session().GetConfig("mode") == const.MODE_NAVIGATOR)

    def SetNavigationMode(self, status):
        if self._disposed:
            return
        next_views = self.navigation_views[:1] if status else [self.volume_view]
        if next_views == self.active_views:
            return

        for view in self.active_views:
            view._event_router.active = False
            view.set_active(False)

        if status and not self._navigation_visited:
            camera = self.volume_view.GetCameraSettings()
            for view in next_views:
                if not view.target_mode:
                    view.ApplyCameraSettings(camera)
            self._navigation_visited = True

        self._configure_viewports(next_views)
        self.active_views = next_views
        self.active_view = next_views[0]
        for view in next_views:
            view.SetSize(self.GetClientSize())
            view._event_router.active = True
            view.set_active(True)

        Publisher.sendMessage("Send orientation cube visibility status")
        Publisher.sendMessage("Send ruler visibility status")
        Publisher.sendMessage(
            "Update viewer caption",
            viewer_name="Volume",
            caption=_("Navigation") if status else _("Volume"),
        )

    def _configure_viewports(self, views):
        width = 1.0 / len(views)
        for index, view in enumerate(views):
            view.SetSceneViewport((index * width, 0.0, (index + 1) * width, 1.0))

    def OnSize(self, evt):
        for view in (self.volume_view, *self.navigation_views):
            view.SetSize(self.GetClientSize())
        evt.Skip()

    def OnCloseProject(self):
        self._navigation_visited = False
        for view in self.navigation_views:
            if view.target_coord is not None:
                view.OnUnsetTarget(None)

    def OnDestroy(self, evt):
        if evt.GetEventObject() is self:
            self.dispose()
        evt.Skip()

    def dispose(self):
        if self._disposed:
            return
        self._disposed = True
        Publisher.unsubscribe(self.SetNavigationMode, "Set navigation mode")
        Publisher.unsubscribe(self.OnCloseProject, "Close project data")
        Publisher.unsubscribe(self.dispose, "Exit")
        self.volume_events.active = False
        for router in self.navigation_event_routers:
            router.active = False
        for view in self.active_views:
            view.set_active(False)
        self.volume_view.dispose()
        for view in self.navigation_views:
            view.dispose()
