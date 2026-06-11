# A tkinter panel for per-camera settings (file writing and video compression),
# decoupled from acquisition/monitoring so the user can configure cameras
# without initializing acquisition. The per-camera data itself lives in PyVAQ
# (keyed by camera serial); this panel is just the editor/view, populated from
# the list of currently-selected cameras.

import tkinter as tk
import tkinter.ttk as ttk
from collections import OrderedDict as odict
from fileWritingEntry import FileWritingEntry


class CameraSettingsEntry(ttk.LabelFrame):
    """Settings for a single camera: file writing (directory / base filename /
    enable-write) and video compression (GPU-vs-CPU encoder + a compression
    level). The frame label is the camera serial.

    The compression level is a single value interpreted as nvenc's -cq when GPU
    encoding is selected, or libx264's -crf otherwise. For both, higher means
    more compression (smaller files, worse quality).
    """
    # Valid range for the compression level (both nvenc -cq and libx264 -crf).
    MIN_LEVEL = 0
    MAX_LEVEL = 51

    def __init__(self, master, *args, camSerial='', initialDirectory='',
                 initialBaseFileName='', initialWriteEnable=True,
                 initialCompressionLevel=23, initialGPUVEnc=False, **kwargs):
        ttk.LabelFrame.__init__(self, master, *args, text=camSerial, **kwargs)
        self.camSerial = camSerial

        self.enableWriteChangeHandler = lambda *a: None
        self.compressionChangeHandler = lambda *a: None
        self.gpuVEncChangeHandler = lambda *a: None
        self._defaultLevel = initialCompressionLevel

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

        # GPU (nvenc) vs CPU (libx264) encoding for this camera. Determines how
        #   the compression level is interpreted (and labelled).
        self.gpuVEncVar = tk.BooleanVar(); self.gpuVEncVar.set(initialGPUVEnc)
        self.gpuVEncVar.trace('w', self._onGPUVEncChange)
        self.gpuVEncCheckButton = tk.Checkbutton(self, text="GPU encoding (nvenc)", variable=self.gpuVEncVar, offvalue=False, onvalue=True)

        # Compression level (single value; meaning depends on the encoder).
        self.compressionFrame = ttk.LabelFrame(self, text="Compression level (higher = more compression / smaller)")
        self.compressionLevelVar = tk.StringVar(); self.compressionLevelVar.set(str(initialCompressionLevel))
        self.compressionLevelLabel = ttk.Label(self.compressionFrame)
        self.compressionLevelEntry = ttk.Entry(self.compressionFrame, width=5, textvariable=self.compressionLevelVar)
        self.compressionLevelEntry.bind('<FocusOut>', self._onCompressionChange)
        self._updateCompressionLabel()
        self.compressionLevelLabel.grid(row=0, column=0, sticky=tk.E)
        self.compressionLevelEntry.grid(row=0, column=1, sticky=tk.W)

        self.fileWidget.grid(row=0, column=0, sticky=tk.NSEW)
        self.enableWriteCheckButton.grid(row=1, column=0, sticky=tk.W)
        self.gpuVEncCheckButton.grid(row=2, column=0, sticky=tk.W)
        self.compressionFrame.grid(row=3, column=0, sticky=tk.NSEW)

    def _onEnableWriteChange(self, *args):
        self._updateEnableWriteColor()
        self.enableWriteChangeHandler()

    def _updateEnableWriteColor(self):
        self.enableWriteCheckButton['fg'] = 'green' if self.getEnableWrite() else 'red'

    def _onGPUVEncChange(self, *args):
        self._updateCompressionLabel()
        self.gpuVEncChangeHandler()

    def _updateCompressionLabel(self):
        # The compression level is nvenc -cq for GPU encoding, libx264 -crf for
        #   CPU encoding. Update the field label to match.
        if self.gpuVEncVar.get():
            self.compressionLevelLabel.config(text="nvenc -cq:")
        else:
            self.compressionLevelLabel.config(text="libx264 -crf:")

    def _onCompressionChange(self, *args):
        self.compressionLevelVar.set(str(self._sanitizeLevel(self.compressionLevelVar.get())))
        self.compressionChangeHandler()

    def _sanitizeLevel(self, value):
        # Coerce to an int clamped to the valid compression-level range.
        try:
            value = int(float(value))
        except (ValueError, TypeError):
            return self._defaultLevel
        return max(self.MIN_LEVEL, min(self.MAX_LEVEL, value))

    # --- Getters ---
    def getDirectory(self):
        return self.fileWidget.getDirectory()

    def getBaseFileName(self):
        return self.fileWidget.getBaseFileName()

    def getEnableWrite(self):
        return self.enableWriteVar.get()

    def getCompressionLevel(self):
        return self._sanitizeLevel(self.compressionLevelVar.get())

    def getGPUVEnc(self):
        return self.gpuVEncVar.get()

    # --- Setters (used when a value is changed programmatically, e.g. on
    #     loading settings) ---
    def setDirectory(self, directory):
        self.fileWidget.setDirectory(directory)

    def setBaseFileName(self, baseFileName):
        self.fileWidget.setBaseFileName(baseFileName)

    def setWriteEnable(self, enableWrite):
        self.enableWriteVar.set(enableWrite)

    def setCompressionLevel(self, level):
        self.compressionLevelVar.set(str(level))

    def setGPUVEnc(self, gpuVEnc):
        self.gpuVEncVar.set(gpuVEnc)

    # --- Change-handler registration ---
    def setDirectoryChangeHandler(self, function):
        self.fileWidget.setDirectoryChangeHandler(function)

    def setBaseFileNameChangeHandler(self, function):
        self.fileWidget.setBaseFileNameChangeHandler(function)

    def setEnableWriteChangeHandler(self, function):
        self.enableWriteChangeHandler = function

    def setCompressionChangeHandler(self, function):
        self.compressionChangeHandler = function

    def setGPUVEncChangeHandler(self, function):
        self.gpuVEncChangeHandler = function


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
        self.gpuVEncChangeHandler = lambda *a: None

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

    def setGPUVEncChangeHandler(self, function):
        self.gpuVEncChangeHandler = function
        for entry in self.cameraEntries.values():
            entry.setGPUVEncChangeHandler(function)

    def updateCameras(self, camSerials, directories=None, baseFileNames=None,
                      writeEnables=None, compressionLevels=None, gpuVEncs=None,
                      defaultCompressionLevel=23):
        """Rebuild the per-camera entries for the given list of camera serials,
        seeding each from the supplied per-camera dicts (keyed by serial).
        Cameras without a stored compression level fall back to
        defaultCompressionLevel; without a stored GPU flag, to CPU encoding.
        """
        directories = directories if directories is not None else {}
        baseFileNames = baseFileNames if baseFileNames is not None else {}
        writeEnables = writeEnables if writeEnables is not None else {}
        compressionLevels = compressionLevels if compressionLevels is not None else {}
        gpuVEncs = gpuVEncs if gpuVEncs is not None else {}

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
                initialCompressionLevel=compressionLevels.get(camSerial, defaultCompressionLevel),
                initialGPUVEnc=gpuVEncs.get(camSerial, False)
                )
            # Apply the currently-registered change handlers
            entry.setDirectoryChangeHandler(self.directoryChangeHandler)
            entry.setBaseFileNameChangeHandler(self.baseFileNameChangeHandler)
            entry.setEnableWriteChangeHandler(self.enableWriteChangeHandler)
            entry.setCompressionChangeHandler(self.compressionChangeHandler)
            entry.setGPUVEncChangeHandler(self.gpuVEncChangeHandler)
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

    def getCompressionLevels(self):
        return dict((s, e.getCompressionLevel()) for s, e in self.cameraEntries.items())

    def getGPUVEncs(self):
        return dict((s, e.getGPUVEnc()) for s, e in self.cameraEntries.items())
