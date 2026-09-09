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
# from math import cos, sin
import os

import wx

# TODO: Check that these imports are not used -- vtkLookupTable, vtkMinimalStandardRandomSequence, vtkPoints, vtkUnsignedCharArray
from vtkmodules.vtkFiltersHybrid import vtkRenderLargeImage
from vtkmodules.vtkIOExport import (
    vtkIVExporter,
    vtkOBJExporter,
    vtkPOVExporter,
    vtkRIBExporter,
    vtkVRMLExporter,
    vtkX3DExporter,
)
from vtkmodules.vtkIOImage import (
    vtkBMPWriter,
    vtkJPEGWriter,
    vtkPNGWriter,
    vtkPostScriptWriter,
    vtkTIFFWriter,
)
from vtkmodules.vtkRenderingCore import (
    vtkWindowToImageFilter,
)

import invesalius.constants as const
import invesalius.utils as utils
from invesalius.data.viewer_base import (
    Base3DView,
    SlicePlane,  # noqa: F401 - historical public import
    _has_win32api,
    win32api,
)
from invesalius.data.viewer_navigation import NavigationView
from invesalius.i18n import tr as _
from invesalius.pubsub import pub as Publisher


class VolumeView(Base3DView):
    """General volume tools, mask previews, seeding and export."""

    def _bind_events(self):
        super()._bind_events()
        Publisher.subscribe(self.OnExportSurface, "Export surface to file")
        Publisher.subscribe(self.OnExportPicture, "Export picture to file")
        Publisher.subscribe(self.OnStartSeed, "Create surface by seeding - start")
        Publisher.subscribe(self.OnEndSeed, "Create surface by seeding - end")
        Publisher.subscribe(self.load_mask_preview, "Load mask preview")
        Publisher.subscribe(self.remove_mask_preview, "Remove mask preview")

    def OnStartSeed(self):
        self.seed_points = []

    def OnEndSeed(self):
        Publisher.sendMessage("Create surface from seeds", seeds=self.seed_points)

    def OnExportPicture(self, orientation, filename, filetype):
        if orientation == const.VOLUME:
            Publisher.sendMessage("Begin busy cursor")
            if _has_win32api:
                utils.touch(filename)
                win_filename = win32api.GetShortPathName(filename)  # type: ignore
                self._export_picture(orientation, win_filename, filetype)
            else:
                self._export_picture(orientation, filename, filetype)
            Publisher.sendMessage("End busy cursor")

    def _export_picture(self, id, filename, filetype):
        if filetype == const.FILETYPE_POV:
            renwin = self.interactor.GetRenderWindow()
            image = vtkWindowToImageFilter()
            image.SetInput(renwin)
            writer = vtkPOVExporter()
            writer.SetFileName(filename.encode(const.FS_ENCODE))
            writer.SetRenderWindow(renwin)
            writer.Write()
        else:
            # Use tiling to generate a large rendering.
            image = vtkRenderLargeImage()
            image.SetInput(self.ren)
            image.SetMagnification(1)
            image.Update()

            image = image.GetOutput()

            # write image file
            if filetype == const.FILETYPE_BMP:
                writer = vtkBMPWriter()
            elif filetype == const.FILETYPE_JPG:
                writer = vtkJPEGWriter()
            elif filetype == const.FILETYPE_PNG:
                writer = vtkPNGWriter()
            elif filetype == const.FILETYPE_PS:
                writer = vtkPostScriptWriter()
            elif filetype == const.FILETYPE_TIF:
                writer = vtkTIFFWriter()
                filename = "{}.tif".format(filename.strip(".tif"))

            writer.SetInputData(image)
            writer.SetFileName(filename.encode(const.FS_ENCODE))
            writer.Write()

        if not os.path.exists(filename):
            wx.MessageBox(
                _("InVesalius was not able to export this picture"), _("Export picture error")
            )

    def OnExportSurface(self, filename, filetype, convert_to_world=False):
        if filetype not in (
            const.FILETYPE_STL,
            const.FILETYPE_VTP,
            const.FILETYPE_PLY,
            const.FILETYPE_STL_ASCII,
            const.FILETYPE_3MF,
        ):
            if _has_win32api:
                utils.touch(filename)
                win_filename = win32api.GetShortPathName(filename)  # type: ignore
                self._export_surface(win_filename, filetype)
            else:
                self._export_surface(filename, filetype)

    def ChangeRenderOrderToExportFile(self):
        """
        Due to the need for self.target_guide_renderer, which is the neuronavigation renderer,
        to be added first, it is necessary to change the order of the renderers in order to
        correctly export the files in obj, vrml, etc. formats. So that only ren and
        canvas_renderer are kept.

        TODO: It is recommended to improve the order in which renderers work in the future.
        """
        if self.target_guide_renderer is None:
            return

        renwin = self.interactor.GetRenderWindow()
        for r in self.renderers:
            renwin.RemoveRenderer(r)

        renwin.SetNumberOfLayers(2)
        renwin.AddRenderer(self.renderers[1])
        renwin.AddRenderer(self.renderers[2])
        renwin.AddRenderer(self.renderers[3])

        self.interactor.Render()

    def RestoreRenderOrderAfterExportFile(self):
        """
        Restores renderer order after export, keeping self.target_guide_renderer,
        ren and canvas_renderer
        """
        if self.target_guide_renderer is None:
            return

        renwin = self.interactor.GetRenderWindow()
        renwin.RemoveRenderer(self.renderers[1])
        renwin.RemoveRenderer(self.renderers[2])

        renwin.AddRenderer(self.renderers[0])
        renwin.SetNumberOfLayers(2)
        renwin.AddRenderer(self.renderers[1])
        renwin.AddRenderer(self.renderers[2])
        renwin.AddRenderer(self.renderers[3])

        self.interactor.Render()

    def _export_surface(self, filename, filetype):
        fileprefix = filename.split(".")[-2]
        renwin = self.interactor.GetRenderWindow()

        self.ChangeRenderOrderToExportFile()

        progress = wx.ProgressDialog(
            "Exporting",
            "Preparing export...",
            maximum=100,
            style=wx.PD_APP_MODAL | wx.PD_AUTO_HIDE | wx.PD_CAN_ABORT | wx.PD_ELAPSED_TIME,
        )
        progress_destroyed = False

        try:
            num_updates = 20
            for i in range(num_updates):
                percent = int(i * 89 / num_updates)
                keep_going, _ = progress.Update(percent, f"Exporting file: {percent}%")
                if not keep_going:
                    progress.Destroy()
                    progress_destroyed = True
                    wx.MessageBox(
                        "Export cancelled by user.", "Export Cancelled", wx.OK | wx.ICON_INFORMATION
                    )
                    return  # If User cancels
                wx.MilliSleep(30)
                wx.Yield()

            progress.Update(90, "Finalizing export...")
            wx.MilliSleep(100)
            wx.Yield()

            if filetype == const.FILETYPE_RIB:
                writer = vtkRIBExporter()
                writer.SetFilePrefix(fileprefix)
                writer.SetTexturePrefix(fileprefix)
                writer.SetInput(renwin)
                writer.Write()
            elif filetype == const.FILETYPE_VRML:
                writer = vtkVRMLExporter()
                writer.SetFileName(filename)
                writer.SetInput(renwin)
                writer.Write()
            elif filetype == const.FILETYPE_X3D:
                writer = vtkX3DExporter()
                writer.SetInput(renwin)
                writer.SetFileName(filename)
                writer.Update()
                writer.Write()
            elif filetype == const.FILETYPE_OBJ:
                writer = vtkOBJExporter()
                writer.SetFilePrefix(fileprefix)
                writer.SetInput(renwin)
                writer.Write()
            elif filetype == const.FILETYPE_IV:
                writer = vtkIVExporter()
                writer.SetFileName(filename)
                writer.SetInput(renwin)
                writer.Write()
            else:
                raise ValueError("Unsupported filetype")

            progress.Update(100, "Export Complete")
            wx.MilliSleep(100)
            # Used by surface.py if needed
            wx.Yield()
            self.export_successful = True

            wx.MessageBox(
                "Export completed successfully.", "Export success", wx.OK | wx.ICON_INFORMATION
            )

        except Exception as e:
            wx.MessageBox(f"Export failed: {e}", "Export Error", wx.OK | wx.ICON_ERROR)
        finally:
            self.RestoreRenderOrderAfterExportFile()
            if progress and not progress_destroyed:
                try:
                    progress.Destroy()
                except Exception:
                    pass

    def load_mask_preview(self, mask_3d_actor, flag=True):
        if flag:
            self.ren.AddVolume(mask_3d_actor)
        else:
            self.ren.RemoveVolume(mask_3d_actor)

        if flag:
            if not self.view_angle:
                self.SetViewAngle(const.VOL_FRONT)
                self.view_angle = 1

            # Match the parallel projection used by AddSurface/LoadVolume so that
            # GetCompositeProjectionTransformMatrix produces a correct world-to-screen
            # matrix for the 3D mask editor (fixes #1086 – "Edit in 3D" without a
            # surface generated first).
            self.ren.GetActiveCamera().ParallelProjectionOn()

        self.UpdateRender()

    def remove_mask_preview(self, mask_3d_actor):
        self.ren.RemoveVolume(mask_3d_actor)


class Viewer(NavigationView, VolumeView):
    """Compatibility view combining existing tools in one panel.

    Keep this entry point until the UI explicitly selects a specialized view.
    Its original subscriptions are intentionally registered in their original order.
    """

    def _bind_events(self):
        Publisher.subscribe(self.AddSurface, "Load surface actor into viewer")
        Publisher.subscribe(self.RemoveSurface, "Remove surface actor from viewer")
        # Publisher.subscribe(self.OnShowSurface, 'Show surface')
        Publisher.subscribe(self.UpdateRender, "Render volume viewer")
        Publisher.subscribe(self.ChangeBackgroundColour, "Change volume viewer background colour")

        # Related to raycasting
        Publisher.subscribe(self.LoadVolume, "Load volume into viewer")
        Publisher.subscribe(self.UnloadVolume, "Unload volume")
        Publisher.subscribe(self.OnSetWindowLevelText, "Set volume window and level text")
        Publisher.subscribe(self.OnHideRaycasting, "Hide raycasting volume")
        Publisher.subscribe(self.OnShowRaycasting, "Update raycasting preset")
        ###
        Publisher.subscribe(self.AppendActor, "AppendActor")
        Publisher.subscribe(self.SetWidgetInteractor, "Set Widget Interactor")
        Publisher.subscribe(self.OnSetViewAngle, "Set volume view angle")

        Publisher.subscribe(
            self.OnDisableBrightContrast, "Set interaction mode " + str(const.MODE_SLICE_EDITOR)
        )

        Publisher.subscribe(self.OnExportSurface, "Export surface to file")

        Publisher.subscribe(self.LoadSlicePlane, "Load slice plane")

        Publisher.subscribe(self.ResetCamClippingRange, "Reset cam clipping range")
        Publisher.subscribe(self.SendActiveCamera, "Send volume viewer active camera")
        Publisher.subscribe(self.SendViewerSize, "Send volume viewer size")

        Publisher.subscribe(self.enable_style, "Enable style")
        Publisher.subscribe(self.OnDisableStyle, "Disable style")

        Publisher.subscribe(self.OnHideText, "Hide text actors on viewers")

        Publisher.subscribe(self.AddActors, "Add actors " + str(const.SURFACE))
        Publisher.subscribe(self.RemoveActors, "Remove actors " + str(const.SURFACE))

        Publisher.subscribe(self.OnShowText, "Show text actors on viewers")
        Publisher.subscribe(self.OnShowRuler, "Show rulers on viewers")
        Publisher.subscribe(self.OnHideRuler, "Hide rulers on viewers")
        Publisher.subscribe(self.OnRulerVisibilityStatus, "Receive ruler visibility status")
        Publisher.subscribe(self.OnShowOrientationCube, "Show orientation cube")
        Publisher.subscribe(self.OnCloseProject, "Close project data")
        Publisher.subscribe(self.FocusCamera, "Focus volume camera")

        Publisher.subscribe(self.RemoveAllActors, "Remove all volume actors")

        Publisher.subscribe(self.OnExportPicture, "Export picture to file")

        Publisher.subscribe(self.OnStartSeed, "Create surface by seeding - start")
        Publisher.subscribe(self.OnEndSeed, "Create surface by seeding - end")

        Publisher.subscribe(self.SetStereoMode, "Set stereo mode")

        Publisher.subscribe(self.Reposition3DPlane, "Reposition 3D Plane")

        Publisher.subscribe(self.UpdatePointer, "Update volume viewer pointer")

        Publisher.subscribe(self.RemoveVolume, "Remove Volume")

        Publisher.subscribe(self.OnSensors, "Sensors ID")
        Publisher.subscribe(self.OnRemoveSensorsID, "Remove sensors ID")

        # TODO: This shouldn't be here, rather in marker_visualizer.py. The problem is that
        #   e-field-related markers are stored in this class, even though all other marker
        #   types have been moved to MarkerViewer.
        Publisher.subscribe(self.DeleteEFieldMarkers, "Delete markers")

        # Related to object tracking during neuronavigation
        Publisher.subscribe(self.OnNavigationStatus, "Navigation status")
        Publisher.subscribe(self.UpdateArrowPose, "Update object arrow matrix")
        Publisher.subscribe(
            self.UpdateEfieldPointLocation, "Update point location for e-field calculation"
        )
        Publisher.subscribe(self.GetEnorm, "Get enorm")
        Publisher.subscribe(self.TrackObject, "Track object")
        Publisher.subscribe(self.SetTargetMode, "Set target mode")
        Publisher.subscribe(self.OnUpdateCoilPose, "Update coil pose")
        Publisher.subscribe(self.OnSetTarget, "Set target")
        Publisher.subscribe(self.OnUnsetTarget, "Unset target")
        Publisher.subscribe(self.OnUpdateAngleThreshold, "Update angle threshold")
        Publisher.subscribe(self.OnUpdateDistanceThreshold, "Update distance threshold")
        Publisher.subscribe(self.OnUpdateTracts, "Update tracts")
        Publisher.subscribe(self.OnUpdateEfieldvis, "Update efield vis")
        Publisher.subscribe(self.InitializeColorArray, "Initialize color array")
        Publisher.subscribe(self.OnRemoveTracts, "Remove tracts")
        Publisher.subscribe(self.UpdateSeedOffset, "Update seed offset")
        Publisher.subscribe(self.UpdateMarkerOffsetState, "Update marker offset state")
        Publisher.subscribe(self.AddPeeledSurface, "Update peel")
        Publisher.subscribe(self.InitEfield, "Initialize E-field brain")
        Publisher.subscribe(self.GetPeelCenters, "Get peel centers and normals")
        Publisher.subscribe(self.InitLocatorViewer, "Get init locator")
        Publisher.subscribe(self.GetPeelCenters, "Get peel centers and normals")
        Publisher.subscribe(self.InitLocatorViewer, "Get init locator")
        Publisher.subscribe(self.load_mask_preview, "Load mask preview")
        Publisher.subscribe(self.remove_mask_preview, "Remove mask preview")
        Publisher.subscribe(self.GetEfieldActor, "Send Actor")
        Publisher.subscribe(self.ReturnToDefaultColorActor, "Recolor again")
        Publisher.subscribe(self.SaveEfieldData, "Save Efield data")
        Publisher.subscribe(self.SavedAllEfieldData, "Save all Efield data")
        Publisher.subscribe(self.SaveEfieldTargetData, "Save target data")
        Publisher.subscribe(self.ClearSaveEfieldData, "Clear saved efield data")
        Publisher.subscribe(self.GetTargetSavedEfieldData, "Get target index efield")
        Publisher.subscribe(self.CheckStatusSavedEfieldData, "Check efield data")
        Publisher.subscribe(self.GetNeuronavigationApi, "Get Neuronavigation Api")
        Publisher.subscribe(self.UpdateEfieldPointLocationOffline, "Update interseccion offline")
        Publisher.subscribe(self.MaxEfieldActor, "Show max Efield actor")
        Publisher.subscribe(self.CoGEfieldActor, "Show CoG Efield actor")
        Publisher.subscribe(
            self.CalculateDistanceMaxEfieldCoGE, "Show distance between Max and CoG Efield"
        )
        Publisher.subscribe(self.EfieldVectors, "Show Efield vectors")
        Publisher.subscribe(self.RecolorEfieldActor, "Recolor efield actor")
        Publisher.subscribe(self.GetScalpEfield, "Send scalp index")
        # Related to robot tracking during neuronavigation
        Publisher.subscribe(
            self.OnUpdateRobotWarning, "Robot to Neuronavigation: Update robot warning"
        )
        Publisher.subscribe(self.GetCoilPosition, "Calculate position and rotation")
        Publisher.subscribe(
            self.CreateCortexProjectionOnScalp, "Send efield target position on brain"
        )
        Publisher.subscribe(self.UpdateEfieldThreshold, "Update Efield Threshold")
        Publisher.subscribe(self.UpdateEfieldROISize, "Update Efield ROI size")
        Publisher.subscribe(self.SetEfieldTargetAtCortex, "Set as Efield target at cortex")
        Publisher.subscribe(self.EnableShowEfieldAboveThreshold, "Show area above threshold")
        Publisher.subscribe(self.ShowEfieldContours, "Show Efield contours")
        Publisher.subscribe(self.EnableEfieldTools, "Enable Efield tools")
        Publisher.subscribe(self.ClearTargetAtCortex, "Clear efield target at cortex")
        Publisher.subscribe(self.CoGEforCortexMarker, "Get Cortex position")
        Publisher.subscribe(self.AddCortexMarkerActor, "Add cortex marker actor")
        Publisher.subscribe(self.CortexMarkersVisualization, "Display efield markers at cortex")
        Publisher.subscribe(self.GetTargetPositions, "Get targets Ids for mtms")
        Publisher.subscribe(self.GetTargetPathmTMS, "Send targeting file path")
        Publisher.subscribe(self.GetdIsfromCoord, "Send mtms coords")
        Publisher.subscribe(
            self.EnableSaveAutomaticallyEfieldData, "Save automatically efield data"
        )
        Publisher.subscribe(self.Getdiperdtforreport, "Get diperdt used in efield calculation")
        Publisher.subscribe(self.UpdateTractSeedBasedEfield, "Update tract seed based efield")
        Publisher.subscribe(self.Get_meshes_paths_to_report, "Get path meshes")

        # SSAO related
        Publisher.subscribe(self._EnableSSAO, "Enable SSAO")
        Publisher.subscribe(self._DisableSSAO, "Disable SSAO")
        Publisher.subscribe(self._ApplySSAOAfterProjectLoad, "Project loaded successfully")
