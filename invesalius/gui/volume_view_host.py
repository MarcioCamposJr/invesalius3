"""Switch one 3D volume view between general and navigation modes."""

import wx
from vtkmodules.wx.wxVTKRenderWindowInteractor import wxVTKRenderWindowInteractor

import invesalius.constants as const
import invesalius.session as ses
from invesalius.data.viewer_navigation import NavigationController
from invesalius.data.viewer_volume import VolumeView
from invesalius.i18n import tr as _
from invesalius.pubsub import pub as Publisher


class VolumeViewHost(wx.Panel):
    """Own one interactor, one volume view and navigation while it is active."""

    def __init__(self, parent):
        super().__init__(parent)
        self._disposed = False
        self.interactor = wxVTKRenderWindowInteractor(self, -1, size=self.GetSize())
        sizer = wx.BoxSizer(wx.VERTICAL)
        sizer.Add(self.interactor, 1, wx.EXPAND)
        self.SetSizer(sizer)

        self.volume_view = VolumeView(self, interactor=self.interactor)
        self.volume_view.set_active(True)

        self.navigation = None
        self.navigation_view = None

        self.volume_view.Hide()
        self.active_views = [self.volume_view]
        self.active_view = self.volume_view
        self._navigation_mode = None
        self.Bind(wx.EVT_SIZE, self.OnSize)
        self.Bind(wx.EVT_WINDOW_DESTROY, self.OnDestroy)
        Publisher.subscribe(self.SetNavigationMode, "Set navigation mode")
        Publisher.subscribe(self.OnCloseProject, "Close project data")
        Publisher.subscribe(self.dispose, "Exit")
        self.SetNavigationMode(ses.Session().GetConfig("mode") == const.MODE_NAVIGATOR)

    def SetNavigationMode(self, status):
        if self._disposed:
            return
        status = bool(status)
        if status == self._navigation_mode:
            return

        if status:
            self.navigation = NavigationController(self.volume_view)
            self.navigation_view = self.navigation
            self.navigation.activate()
            self.active_view = self.navigation
        else:
            if self.navigation is not None:
                self.navigation.dispose()
                self.navigation = None
                self.navigation_view = None
            self.active_view = self.volume_view
        self._navigation_mode = status
        self.volume_view.SetSize(self.GetClientSize())

        Publisher.sendMessage("Send orientation cube visibility status")
        Publisher.sendMessage("Send ruler visibility status")
        Publisher.sendMessage(
            "Update viewer caption",
            viewer_name="Volume",
            caption=_("Navigation") if status else _("Volume"),
        )

    def OnSize(self, evt):
        self.volume_view.SetSize(self.GetClientSize())
        evt.Skip()

    def OnCloseProject(self):
        if self.navigation is not None and self.navigation.target_coord is not None:
            self.navigation.OnUnsetTarget(None)

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
        if self.navigation is not None:
            self.navigation.dispose()
            self.navigation = None
            self.navigation_view = None
        self.volume_view.dispose()
        try:
            self.interactor.Disable()
        except Exception:
            pass
