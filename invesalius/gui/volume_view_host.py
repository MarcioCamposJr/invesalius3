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
        self.navigation_events = ViewEventRouter(PROJECT_TOPICS | NAVIGATION_DATA_TOPICS)
        self.volume_view = VolumeView(self, interactor=self.interactor)
        self.volume_events.manage_view(self.volume_view)
        # NavigationView inherits the general volume tools and adds navigation state.
        self.navigation_view = NavigationView(self, interactor=self.interactor)
        self.navigation_events.manage_view(self.navigation_view)
        self.volume_view.Hide()
        self.navigation_view.Hide()
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
        next_view = self.navigation_view if status else self.volume_view
        if next_view is self.active_view:
            return
        if self.active_view is not None:
            self.active_view._event_router.active = False
            self.active_view.set_active(False)
        if status and not self._navigation_visited:
            if not next_view.target_mode:
                next_view.ApplyCameraSettings(self.volume_view.GetCameraSettings())
            self._navigation_visited = True
        self.active_view = next_view
        next_view.SetSize(self.GetClientSize())
        next_view._event_router.active = True
        next_view.set_active(True)
        Publisher.sendMessage("Send orientation cube visibility status")
        Publisher.sendMessage("Send ruler visibility status")
        Publisher.sendMessage(
            "Update viewer caption",
            viewer_name="Volume",
            caption=_("Navigation") if status else _("Volume"),
        )

    def OnSize(self, evt):
        for view in (self.volume_view, self.navigation_view):
            view.SetSize(self.GetClientSize())
        evt.Skip()

    def OnCloseProject(self):
        self._navigation_visited = False
        if self.navigation_view.target_coord is not None:
            self.navigation_view.OnUnsetTarget(None)

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
        self.navigation_events.active = False
        # Detach the active scene before disposing the inactive one.
        if self.active_view is not None:
            self.active_view.set_active(False)
        self.volume_view.dispose()
        self.navigation_view.dispose()
