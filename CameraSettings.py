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
    # Valid range for both nvenc -cq and libx264 -crf quality values.
    MIN_QUALITY = 0
    MAX_QUALITY = 51

    def __init__(self, master, *args, camSerial='', initialDirectory='',
                 initialBaseFileName='', initialWriteEnable=True,
                 initialCQ=23, initialCRF=23, **kwargs):
        ttk.LabelFrame.__init__(self, master, *args, text=camSerial, **kwargs)
        self.camSerial = camSerial

        self.enableWriteChangeHandler = lambda *a: None
        self.compressionChangeHandler = lambda *a: None
        self._defaultCQ = initialCQ
        self._defaultCRF = initialCRF

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

        # Compression quality. The camera's active encoder (nvenc vs libx264)
        #   depends on GPU availability at acquisition time; both values are
        #   kept and only the relevant one is used.
        self.compressionFrame = ttk.LabelFrame(self, text="Compression quality (lower = better/larger)")
        self.cqVar = tk.StringVar(); self.cqVar.set(str(initialCQ))
        self.crfVar = tk.StringVar(); self.crfVar.set(str(initialCRF))
        self.cqLabel = ttk.Label(self.compressionFrame, text="GPU nvenc -cq:")
        self.cqEntry = ttk.Entry(self.compressionFrame, width=5, textvariable=self.cqVar)
        self.crfLabel = ttk.Label(self.compressionFrame, text="CPU libx264 -crf:")
        self.crfEntry = ttk.Entry(self.compressionFrame, width=5, textvariable=self.crfVar)
        self.cqEntry.bind('<FocusOut>', self._onCompressionChange)
        self.crfEntry.bind('<FocusOut>', self._onCompressionChange)
        self.cqLabel.grid(row=0, column=0, sticky=tk.E)
        self.cqEntry.grid(row=0, column=1, sticky=tk.W)
        self.crfLabel.grid(row=1, column=0, sticky=tk.E)
        self.crfEntry.grid(row=1, column=1, sticky=tk.W)

        self.fileWidget.grid(row=0, column=0, sticky=tk.NSEW)
        self.enableWriteCheckButton.grid(row=1, column=0, sticky=tk.W)
        self.compressionFrame.grid(row=2, column=0, sticky=tk.NSEW)

    def _onEnableWriteChange(self, *args):
        self._updateEnableWriteColor()
        self.enableWriteChangeHandler()

    def _updateEnableWriteColor(self):
        self.enableWriteCheckButton['fg'] = 'green' if self.getEnableWrite() else 'red'

    def _onCompressionChange(self, *args):
        # Sanitize both fields, then notify.
        self.cqVar.set(str(self._sanitizeQuality(self.cqVar.get(), self._defaultCQ)))
        self.crfVar.set(str(self._sanitizeQuality(self.crfVar.get(), self._defaultCRF)))
        self.compressionChangeHandler()

    def _sanitizeQuality(self, value, fallback):
        # Coerce to an int clamped to the valid quality range.
        try:
            value = int(float(value))
        except (ValueError, TypeError):
            return fallback
        return max(self.MIN_QUALITY, min(self.MAX_QUALITY, value))

    # --- Getters ---
    def getDirectory(self):
        return self.fileWidget.getDirectory()

    def getBaseFileName(self):
        return self.fileWidget.getBaseFileName()

    def getEnableWrite(self):
        return self.enableWriteVar.get()

    def getCQ(self):
        return self._sanitizeQuality(self.cqVar.get(), self._defaultCQ)

    def getCRF(self):
        return self._sanitizeQuality(self.crfVar.get(), self._defaultCRF)

    # --- Setters (used when a value is changed programmatically, e.g. on
    #     loading settings) ---
    def setDirectory(self, directory):
        self.fileWidget.setDirectory(directory)

    def setBaseFileName(self, baseFileName):
        self.fileWidget.setBaseFileName(baseFileName)

    def setWriteEnable(self, enableWrite):
        self.enableWriteVar.set(enableWrite)

    def setCQ(self, cq):
        self.cqVar.set(str(cq))

    def setCRF(self, crf):
        self.crfVar.set(str(crf))

    # --- Change-handler registration ---
    def setDirectoryChangeHandler(self, function):
        self.fileWidget.setDirectoryChangeHandler(function)

    def setBaseFileNameChangeHandler(self, function):
        self.fileWidget.setBaseFileNameChangeHandler(function)

    def setEnableWriteChangeHandler(self, function):
        self.enableWriteChangeHandler = function

    def setCompressionChangeHandler(self, function):
        self.compressionChangeHandler = function


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
        self.compressionChangeHandler = lambda *a: None

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

    def setCompressionChangeHandler(self, function):
        self.compressionChangeHandler = function
        for entry in self.cameraEntries.values():
            entry.setCompressionChangeHandler(function)

    def updateCameras(self, camSerials, directories=None, baseFileNames=None,
                      writeEnables=None, cqs=None, crfs=None,
                      defaultCQ=23, defaultCRF=23):
        """Rebuild the per-camera entries for the given list of camera serials,
        seeding each from the supplied per-camera dicts (keyed by serial).
        Cameras without a stored cq/crf fall back to defaultCQ/defaultCRF.
        """
        directories = directories if directories is not None else {}
        baseFileNames = baseFileNames if baseFileNames is not None else {}
        writeEnables = writeEnables if writeEnables is not None else {}
        cqs = cqs if cqs is not None else {}
        crfs = crfs if crfs is not None else {}

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
                initialWriteEnable=writeEnables.get(camSerial, True),
                initialCQ=cqs.get(camSerial, defaultCQ),
                initialCRF=crfs.get(camSerial, defaultCRF)
                )
            # Apply the currently-registered change handlers
            entry.setDirectoryChangeHandler(self.directoryChangeHandler)
            entry.setBaseFileNameChangeHandler(self.baseFileNameChangeHandler)
            entry.setEnableWriteChangeHandler(self.enableWriteChangeHandler)
            entry.setCompressionChangeHandler(self.compressionChangeHandler)
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

    def getCQs(self):
        return dict((s, e.getCQ()) for s, e in self.cameraEntries.items())

    def getCRFs(self):
        return dict((s, e.getCRF()) for s, e in self.cameraEntries.items())
