# A tkinter panel for per-camera settings (file writing, and - later -
# compression), decoupled from acquisition/monitoring so the user can configure
# cameras without initializing acquisition. The per-camera data itself lives in
# PyVAQ (keyed by camera serial); this panel is just the editor/view, populated
# from the list of currently-selected cameras.

import tkinter as tk
import tkinter.ttk as ttk
from collections import OrderedDict as odict
from fileWritingEntry import FileWritingEntry


class CameraSettingsEntry(ttk.LabelFrame):
    """Settings for a single camera: file writing (directory / base filename /
    enable-write). The frame label is the camera serial.
    """
    def __init__(self, master, *args, camSerial='', initialDirectory='',
                 initialBaseFileName='', initialWriteEnable=True, **kwargs):
        ttk.LabelFrame.__init__(self, master, *args, text=camSerial, **kwargs)
        self.camSerial = camSerial

        self.enableWriteChangeHandler = lambda *a: None

        self.fileWidget = FileWritingEntry(
            self,
            defaultDirectory=initialDirectory,
            defaultBaseFileName=initialBaseFileName,
            purposeText='video writing for camera {s}'.format(s=camSerial),
            text='Video file writing'
            )

        # Enable-write checkbox. Set the value before adding the trace so the
        #   change handler doesn't fire during construction.
        self.enableWriteVar = tk.BooleanVar(); self.enableWriteVar.set(initialWriteEnable)
        self.enableWriteVar.trace('w', self._onEnableWriteChange)
        self.enableWriteCheckButton = tk.Checkbutton(self, text="Enable write", variable=self.enableWriteVar, offvalue=False, onvalue=True)
        self._updateEnableWriteColor()

        self.fileWidget.grid(row=0, column=0, sticky=tk.NSEW)
        self.enableWriteCheckButton.grid(row=1, column=0, sticky=tk.W)

    def _onEnableWriteChange(self, *args):
        self._updateEnableWriteColor()
        self.enableWriteChangeHandler()

    def _updateEnableWriteColor(self):
        self.enableWriteCheckButton['fg'] = 'green' if self.getEnableWrite() else 'red'

    # --- Getters ---
    def getDirectory(self):
        return self.fileWidget.getDirectory()

    def getBaseFileName(self):
        return self.fileWidget.getBaseFileName()

    def getEnableWrite(self):
        return self.enableWriteVar.get()

    # --- Setters (used when a value is changed programmatically, e.g. on
    #     loading settings) ---
    def setDirectory(self, directory):
        self.fileWidget.setDirectory(directory)

    def setBaseFileName(self, baseFileName):
        self.fileWidget.setBaseFileName(baseFileName)

    def setWriteEnable(self, enableWrite):
        self.enableWriteVar.set(enableWrite)

    # --- Change-handler registration ---
    def setDirectoryChangeHandler(self, function):
        self.fileWidget.setDirectoryChangeHandler(function)

    def setBaseFileNameChangeHandler(self, function):
        self.fileWidget.setBaseFileNameChangeHandler(function)

    def setEnableWriteChangeHandler(self, function):
        self.enableWriteChangeHandler = function


class CameraSettingsPanel(ttk.Frame):
    """Container holding one CameraSettingsEntry per selected camera, plus an
    "Advanced (FLIR)..." button for the legacy FLIR attribute configuration.

    Populated via updateCameras() from the list of selected camera serials, so
    it works whether or not acquisition has been initialized.
    """
    def __init__(self, master, *args, advancedCommand=None, **kwargs):
        ttk.Frame.__init__(self, master, *args, **kwargs)
        self.cameraEntries = odict()  # camSerial -> CameraSettingsEntry
        self.advancedCommand = advancedCommand

        # Handlers PyVAQ registers; applied to every per-camera entry.
        self.directoryChangeHandler = lambda *a: None
        self.baseFileNameChangeHandler = lambda *a: None
        self.enableWriteChangeHandler = lambda *a: None

        self.entryFrame = ttk.Frame(self)

        self.noCamerasLabel = ttk.Label(
            self,
            text='No cameras selected.\nUse "Select audio/digital/video inputs" to choose cameras.'
            )

        self.advancedButton = ttk.Button(
            self, text="Advanced (FLIR)…",
            command=(advancedCommand if advancedCommand is not None else (lambda *a: None))
            )

        self._layout()

    def _layout(self):
        self.entryFrame.grid(row=0, column=0, sticky=tk.NSEW)
        if len(self.cameraEntries) == 0:
            self.noCamerasLabel.grid(row=1, column=0, sticky=tk.NSEW)
        else:
            self.noCamerasLabel.grid_remove()
        if self.advancedCommand is not None:
            self.advancedButton.grid(row=2, column=0, sticky=tk.W, pady=(4, 0))

    # --- Change-handler registration (applied to all current & future entries) ---
    def setDirectoryChangeHandler(self, function):
        self.directoryChangeHandler = function
        for entry in self.cameraEntries.values():
            entry.setDirectoryChangeHandler(function)

    def setBaseFileNameChangeHandler(self, function):
        self.baseFileNameChangeHandler = function
        for entry in self.cameraEntries.values():
            entry.setBaseFileNameChangeHandler(function)

    def setEnableWriteChangeHandler(self, function):
        self.enableWriteChangeHandler = function
        for entry in self.cameraEntries.values():
            entry.setEnableWriteChangeHandler(function)

    def updateCameras(self, camSerials, directories=None, baseFileNames=None, writeEnables=None):
        """Rebuild the per-camera entries for the given list of camera serials,
        seeding each from the supplied per-camera dicts (keyed by serial).
        """
        directories = directories if directories is not None else {}
        baseFileNames = baseFileNames if baseFileNames is not None else {}
        writeEnables = writeEnables if writeEnables is not None else {}

        # Destroy old entries
        for camSerial in list(self.cameraEntries.keys()):
            self.cameraEntries[camSerial].grid_forget()
            self.cameraEntries[camSerial].destroy()
            del self.cameraEntries[camSerial]

        # Create new entries, seeded from the current per-camera settings
        for k, camSerial in enumerate(camSerials):
            entry = CameraSettingsEntry(
                self.entryFrame,
                camSerial=camSerial,
                initialDirectory=directories.get(camSerial, ''),
                initialBaseFileName=baseFileNames.get(camSerial, ''),
                initialWriteEnable=writeEnables.get(camSerial, True)
                )
            # Apply the currently-registered change handlers
            entry.setDirectoryChangeHandler(self.directoryChangeHandler)
            entry.setBaseFileNameChangeHandler(self.baseFileNameChangeHandler)
            entry.setEnableWriteChangeHandler(self.enableWriteChangeHandler)
            entry.grid(row=k, column=0, sticky=tk.NSEW, pady=2)
            self.cameraEntries[camSerial] = entry

        self._layout()

    # --- Dict getters over the current entries (keyed by camera serial) ---
    def getDirectories(self):
        return dict((s, e.getDirectory()) for s, e in self.cameraEntries.items())

    def getBaseFileNames(self):
        return dict((s, e.getBaseFileName()) for s, e in self.cameraEntries.items())

    def getWriteEnables(self):
        return dict((s, e.getEnableWrite()) for s, e in self.cameraEntries.items())
