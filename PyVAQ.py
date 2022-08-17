import sys
import os
import time
import json
import math
import numpy as np
import multiprocessing as mp
import nidaqmx.system as nisys
import tkinter as tk
import tkinter.ttk as ttk
from tkinter.filedialog import askdirectory, asksaveasfilename, askopenfilename
from tkinter.messagebox import showinfo, showwarning
import queue
from PIL import Image
import pprint
import traceback
from collections import deque
import re
import datetime
import unicodedata
from TimeInput import TimeVar, TimeEntry
from ParamDialog import ParamDialog, Param
from fileWritingEntry import FileWritingEntry
import warnings
import cProfile, pstats, io
# For audio monitor graph embedding:
from matplotlib.backends.backend_tkagg import (
    FigureCanvasTkAgg, NavigationToolbar2Tk)
# Implement the default Matplotlib key bindings.
# from matplotlib.backend_bases import key_press_handler
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
# from matplotlib.lines import Line2D
# from pympler import tracker
try:
    import PySpin
except ModuleNotFoundError:
    # pip seems to install PySpin as pyspin sometimes...
    import pyspin as PySpin

from MonitorWidgets import AudioMonitor, CameraMonitor
from DockableFrame import Docker
from StateMachineProcesses import States, Messages, Trigger, StdoutManager, AVMerger, Synchronizer, AudioTriggerer, AudioAcquirer, AudioWriter, VideoAcquirer, VideoWriter, ContinuousTriggerer, syncPrint, SimpleVideoWriter, SimpleAudioWriter
import inspect
import CollapsableFrame as cf
import PySpinUtilities as psu
import ctypes

VERSION='0.3.0'

try:
    # Inform windows that this App should display its own icon, not python's
    # Thanks to StackOverflow user @DamonJW - https://stackoverflow.com/a/1552105/1460057
    myappid = '{company}.{product}.{version}'.format(company='glab', product='PyVAQ', version=VERSION) # arbitrary string
    ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(myappid)
except:
    # Whatever, it's not that important.
    print('PyVAQ icon display failed')

# Todo:

# Verbosity levels (cumulative):
# 0 - Errors
# 1 - Occasional major status updates
# 2 - Routine status messages
# 3 - Everything

ICON_PATH = r'Resources\PyVAQ.ico'

#plt.style.use("dark_background")

np.set_printoptions(linewidth=200)

def getOptimalMonitorGrid(numCameras):
    if numCameras == 0:
        return (0,0)
    divisors = []
    for k in range(1, numCameras+1):
        if numCameras % k == 0:
            divisors.append(k)
    bestDivisor = min(divisors, key=lambda d:abs(d - (numCameras/d)))
    return sorted([bestDivisor, int(numCameras/bestDivisor)], key=lambda x:-x)

def flattenList(l):
    return [item for sublist in l for item in sublist]

def discoverDAQAudioChannels():
    s = nisys.System.local()
    channels = {}
    for d in s.devices:
        channels[d.name] = [c.name for c in d.ai_physical_chans]
    return channels

def discoverDAQTerminals():
    s = nisys.System.local()
    channels = {}
    for d in s.devices:
        channels[d.name] = d.terminals
    return channels

def discoverDAQClockChannels():
    s = nisys.System.local()
    channels = {}
    for d in s.devices:
        channels[d.name] = [c.name for c in d.co_physical_chans]
    return channels

def serializableToTime(serializable):
    return datetime.time(
        hour=serializable['hour'],
        minute=serializable['minute'],
        second=serializable['second'],
        microsecond=serializable['microsecond']
    )

def timeToSerializable(time):
    return dict(
        hour=time.hour,
        minute=time.minute,
        second=time.second,
        microsecond=time.microsecond
    )

def defaultSerializer(obj):
    if isinstance(obj, datetime.datetime):
        return timeToSerializable(obj)
    else:
        raise TypeError("Object is not serializable")

def convertExposureTimeToDutyCycle(exposureTime, frameRate):
    # Exposure time in s, frameRate in Hz
    return exposureTime * frameRate

def convertDutyCycleToExposureTime(dutyCycle, frameRate):
    # framerate in Hz, exposure time in s
    return dutyCycle / frameRate

def format_diff(diff):
    # Diff is a list of the form output by pympler.summary.diff()
    output = '\n'.join(str(d) for d in sorted(diff, key=lambda dd:-dd[2]))

def slugify(value, allow_unicode=False):
    """
    Convert to ASCII if 'allow_unicode' is False. Convert spaces to hyphens.
    Remove characters that aren't alphanumerics, underscores, or hyphens.
    Convert to lowercase. Also strip leading and trailing whitespace.

    Adapted from Django utils
    """
    value = str(value)
    if allow_unicode:
        value = unicodedata.normalize('NFKC', value)
    else:
        value = unicodedata.normalize('NFKD', value).encode('ascii', 'ignore').decode('ascii')
    value = re.sub(r'[^\w\s-]', '', value).strip().lower()
    return re.sub(r'[-\s]+', '-', value)

def getSharedString(sharedString, block=False):
    L = sharedString.get_lock()
    locked = L.acquire(block=block)
    if locked:
        # Strip off spaces and null bytes
        value = str.rstrip(sharedString[:], u' \x00')
        L.release()
    else:
        value = None
    return value

LINE_STYLES = [c+'-' for c in 'bykcmgr']
WIDGET_COLORS = [
    '#050505', # near black
    '#e6f5ff', # very light blue
    '#c1ffc1', # light green
    '#FFC1C1'  # light red
]

class GeneralVar:
    def __init__(self):
        self.value = None
        self.getCallback = None
        self.setCallback = None
    def get(self):
        if self.getCallback is not None:
            self.getCallback()
        return self.value
    def set(self, value):
        self.value = value
        if self.setCallback is not None:
            self.setCallback()
    def trace(self, mode, callback):
        if mode == 'w':
            self.setCallback = callback
        elif mode == 'r':
            self.readCallback = callback

class PyVAQ:
    def __init__(self, master, settingsFilePath=None):
        self.master = master
        try:
            self.master.wm_iconbitmap(ICON_PATH)
        except:
            print('Icon load failed')
        self.master.resizable(height=False, width=False)  # Disallow resizing window
        self.master.minsize(300, 0)
        self.customTitleBar = False
        if self.customTitleBar:
            self.master.overrideredirect(True) # Disable title bar
        self.master.title("PyVAQ v{version}".format(version=VERSION))
        self.master.protocol("WM_DELETE_WINDOW", self.cleanupAndExit)
        self.style = ttk.Style()
        self.style.theme_use('default')
        if False:
            # Man, ttk styling is hard to get into...
            self.style.configure('.', background=WIDGET_COLORS[0])
            self.style.configure('.', foreground=WIDGET_COLORS[1])
            self.style.configure('TButton', background=WIDGET_COLORS[1], foreground=WIDGET_COLORS[0], borderwidth=3)
            self.style.configure('TEntry', borderwidth=0, fieldbackground=WIDGET_COLORS[1], foreground=WIDGET_COLORS[0])
            self.style.configure('TLabel', borderwidth=0, background=WIDGET_COLORS[1], fieldbackground=WIDGET_COLORS[1], foreground=WIDGET_COLORS[0])
            self.style.configure('TRadiobutton', indicatorbackground=WIDGET_COLORS[1], indicatorcolor=WIDGET_COLORS[0])
            self.style.configure('TCheckbutton', indicatorbackground=WIDGET_COLORS[1], indicatorcolor=WIDGET_COLORS[0])
            self.style.configure('TCombobox', fieldbackground=WIDGET_COLORS[1], foreground=WIDGET_COLORS[0])
        self.style.configure('SingleContainer.TLabelframe', padding=5)

#        self.style.configure('TFrame', background='#050505', foreground='#aaaadd')
        self.style.configure('ValidDirectory.TEntry', fieldbackground=WIDGET_COLORS[2])
        self.style.configure('InvalidDirectory.TEntry', fieldbackground=WIDGET_COLORS[3])
#        self.style.map('Directory.TEntry.label', background=[(('!invalid',), 'green'),(('invalid',), 'red')])

        self.ID = 'GUI'
        self.stdoutBuffer = []

        ########### PARAMETERS not directly associated with permanent widgets ##########
        self.videoBaseFileNames = GeneralVar(); self.videoBaseFileNames.set({})
        self.videoDirectories = GeneralVar(); self.videoDirectories.set({})
        self.audioBaseFileName = GeneralVar(); self.audioBaseFileName.set('')
        self.audioDirectory = GeneralVar(); self.audioDirectory.set('')
        self.mergeBaseFileName = GeneralVar(); self.mergeBaseFileName.set('')
        self.mergeDirectory = GeneralVar(); self.mergeDirectory.set('')
        self.audioDAQChannels = GeneralVar(); self.audioDAQChannels.set([])
        self.camSerials = GeneralVar(); self.camSerials.set([])  # Cam serials selected for acquisition
        self.audioSyncTerminal = GeneralVar(); self.audioSyncTerminal.set(None)
        self.videoSyncTerminal = GeneralVar(); self.videoSyncTerminal.set(None)
        self.audioSyncSource = GeneralVar(); self.audioSyncSource.set("PFI5")
        self.videoSyncSource = GeneralVar(); self.videoSyncSource.set("PFI4")
        self.videoWriteEnable = GeneralVar(); self.videoWriteEnable.set({})
        self.audioWriteEnable = tk.BooleanVar(); self.audioWriteEnable.set(True)
        self.acquisitionSignalChannel = GeneralVar(); self.acquisitionSignalChannel.set(None)
        self.audioChannelConfiguration = GeneralVar(); self.audioChannelConfiguration.set(None)
        self.videoMonitorDisplaySize = GeneralVar(); self.videoMonitorDisplaySize.set((400, 300))
        self.videoMonitorDisplaySize.trace('w', self.updateVideoMonitorDisplaySize)

        ########### GUI WIDGETS #####################

        BG_COLOR = '#eeeeff'

        collapsableFrameStyle = {
            'relief':tk.GROOVE,
            'borderwidth':3,
            'bg':BG_COLOR
            }
        collapsableFrameButtonStyle = {
            'bg':BG_COLOR
        }

        self.mainFrame = ttk.Frame(self.master)

        self.menuBar = tk.Menu(self.master, tearoff=False)
        self.settingsMenu = tk.Menu(self.menuBar, tearoff=False)
        self.settingsMenu.add_command(label='Save settings...', command=self.saveSettings)
        self.settingsMenu.add_command(label='Load settings...', command=self.loadSettings)
        self.settingsMenu.add_command(label='Save default settings...', command=lambda *args: self.saveSettings(path='default.pvs'))
        self.settingsMenu.add_command(label='Load default settings...', command=lambda *args: self.loadSettings(path='default.pvs'))

        self.helpMenu = tk.Menu(self.menuBar, tearoff=False)
        self.helpMenu.add_command(label="Help", command=self.showHelpDialog)
        self.helpMenu.add_command(label="About", command=self.showAboutDialog)

        self.debugMenu = tk.Menu(self.menuBar, tearoff=False)
        self.debugMenu.add_command(label="Set verbosity", command=self.setVerbosity)
        self.debugMenu.add_command(label="Check states", command=self.checkStates)
        self.debugMenu.add_command(label="Get PIDs", command=self.getPIDs)
        self.debugMenu.add_command(label="Get Queue Sizes", command=self.getQueueSizes)

        self.monitoringMenu = tk.Menu(self.menuBar, tearoff=False)
        self.monitoringMenu.add_command(label="Configure audio monitoring", command=self.configureAudioMonitoring)
        self.monitoringMenu.add_command(label="Configure video monitoring", command=self.configureVideoMonitoring)

        self.menuBar.add_cascade(label="Settings", menu=self.settingsMenu)
        self.menuBar.add_cascade(label="Monitoring", menu=self.monitoringMenu)
        self.menuBar.add_cascade(label="Debug", menu=self.debugMenu)
        self.menuBar.add_cascade(label="Help", menu=self.helpMenu)

        self.master.config(menu=self.menuBar)

        self.titleBarFrame = ttk.Frame(self.master)
        self.closeButton = ttk.Button(self.titleBarFrame, text="X", command=self.cleanupAndExit)

        self.monitorMasterFrame = ttk.Frame(self.mainFrame)

        # Set up undockable frame for video monitoring
        def unDockFunction(d):
            d.unDockButton.grid_forget()
            d.reDockButton.grid(row=0, column=0, sticky=tk.NW)
            self.update()

        def reDockFunction(d):
            d.reDockButton.grid_forget()
            d.unDockButton.grid(row=0, column=0, sticky=tk.NW)
            d.docker.grid(row=0, column=0)
            self.update()

        self.videoMonitorDocker = Docker(
            self.monitorMasterFrame, root=self.master,
            unDockFunction=unDockFunction, reDockFunction=reDockFunction,
            unDockText='undock', reDockText='dock', background='#d9d9d9')
        self.videoMonitorDocker.unDockButton.grid(row=0, column=0, sticky=tk.NW)
        self.videoMonitorDocker.reDockButton.grid(row=0, column=0, sticky=tk.NW)
        self.videoMonitorDocker.reDockButton.grid_forget()

        self.videoMonitorMasterFrame = self.videoMonitorDocker.docker # ttk.Frame(self.monitorMasterFrame)

        self.audioMonitor = None  #ttk.Frame(self.monitorMasterFrame)

        self.cameraAttributes = {}
        self.cameraMonitors = {}

        self.controlFrame = ttk.Frame(self.mainFrame)

        self.statusFrame = cf.CollapsableFrame(self.controlFrame, collapseText="Status", **collapsableFrameStyle); self.statusFrame.stateChangeButton.config(**collapsableFrameButtonStyle)
        self.childStatusText = tk.Text(self.statusFrame, tabs=('7c',))

        self.acquisitionControlFrame = cf.CollapsableFrame(self.controlFrame, collapseText="Acquisition Control", **collapsableFrameStyle); self.acquisitionControlFrame.stateChangeButton.config(**collapsableFrameButtonStyle)
        self.acquisitionControlFrame.expand()
        # self.acquisitionFrame = ttk.LabelFrame(self.controlFrame, text="Acquisition")

        self.initializeAcquisitionButton =      ttk.Button(self.acquisitionControlFrame, text="Initialize acquisition", command=self.initializeAcquisition)
        self.haltAcquisitionButton =            ttk.Button(self.acquisitionControlFrame, text='Halt acquisition', command=self.haltAcquisition)
        self.restartAcquisitionButton =         ttk.Button(self.acquisitionControlFrame, text='Restart acquisition', command=self.restartAcquisition)
        self.shutDownAcquisitionButton =        ttk.Button(self.acquisitionControlFrame, text='Shut down acquisition', command=self.shutDownAcquisition)

        self.acquisitionParametersFrame = cf.CollapsableFrame(self.controlFrame, collapseText="Acquisition Parameters", **collapsableFrameStyle); self.acquisitionParametersFrame.stateChangeButton.config(**collapsableFrameButtonStyle)

        self.audioFrequencyFrame =  ttk.LabelFrame(self.acquisitionParametersFrame, text="Audio freq. (Hz)", style='SingleContainer.TLabelframe')
        self.audioFrequencyVar =    tk.StringVar(); self.audioFrequencyVar.set("44100")
        self.audioFrequencyEntry =  ttk.Entry(self.audioFrequencyFrame, width=16, textvariable=self.audioFrequencyVar);
        self.audioFrequencyVar.trace('w', self.updateAudioFrequency)

        self.videoFrequencyFrame =  ttk.LabelFrame(self.acquisitionParametersFrame, text="Video freq (fps)", style='SingleContainer.TLabelframe')
        self.videoFrequencyVar =    tk.StringVar(); self.videoFrequencyVar.set("30")
        self.videoFrequencyEntry =  ttk.Entry(self.videoFrequencyFrame, width=16, textvariable=self.videoFrequencyVar)
        self.videoFrequencyVar.trace('w', self.updateVideoFrequency)

        self.videoExposureTimeFrame =    ttk.LabelFrame(self.acquisitionParametersFrame, text="Video exposure time (ms):", style='SingleContainer.TLabelframe')
        self.videoExposureTimeVar =      tk.StringVar(); self.videoExposureTimeVar.set("3")
        self.videoExposureTimeEntry =    ttk.Entry(self.videoExposureTimeFrame, width=16, textvariable=self.videoExposureTimeVar)
        self.videoExposureTimeEntry.bind('<FocusOut>', self.validateVideoExposureTime)

        self.gainFrame =    ttk.LabelFrame(self.acquisitionParametersFrame, text="Gain", style='SingleContainer.TLabelframe')
        self.gainVar =      tk.StringVar(); self.gainVar.set("10")
        self.gainEntry =    ttk.Entry(self.gainFrame, width=16, textvariable=self.gainVar)
        self.gainEntry.bind('<FocusOut>', self.validateGain)

        self.preTriggerTimeFrame =  ttk.LabelFrame(self.acquisitionParametersFrame, text="Pre-trigger record time (s)", style='SingleContainer.TLabelframe')
        self.preTriggerTimeVar =    tk.StringVar(); self.preTriggerTimeVar.set("4.5")
        self.preTriggerTimeEntry =  ttk.Entry(self.preTriggerTimeFrame, width=16, textvariable=self.preTriggerTimeVar)

        self.acquisitionBufferSizeFrame =  ttk.LabelFrame(self.acquisitionParametersFrame, text="Acquisition buffer size (s)", style='SingleContainer.TLabelframe')
        self.acquisitionBufferSizeVar =    tk.StringVar(); self.acquisitionBufferSizeVar.set("4.5")
        self.acquisitionBufferSizeEntry =  ttk.Entry(self.acquisitionBufferSizeFrame, width=16, textvariable=self.acquisitionBufferSizeVar)

        self.recordTimeFrame =      ttk.LabelFrame(self.acquisitionParametersFrame, text="Record time (s)", style='SingleContainer.TLabelframe')
        self.recordTimeVar =        tk.StringVar(); self.recordTimeVar.set("10.0")
        self.recordTimeEntry =      ttk.Entry(self.recordTimeFrame, width=16, textvariable=self.recordTimeVar)

        self.acquisitionSignalParametersFrame = ttk.LabelFrame(self.acquisitionParametersFrame, text="Acquisition HW signaling", style='SingleContainer.TLabelframe')
        self.startOnHWSignalVar = tk.BooleanVar(); self.startOnHWSignalVar.set(False)
        self.startOnHWSignalCheckbutton = ttk.Checkbutton(self.acquisitionSignalParametersFrame, text="Start acquisition on HW signal", variable=self.startOnHWSignalVar, offvalue=False, onvalue=True)
        self.writeEnableOnHWSignalVar = tk.BooleanVar(); self.writeEnableOnHWSignalVar.set(False)
        self.writeEnableOnHWSignalCheckbutton = ttk.Checkbutton(self.acquisitionSignalParametersFrame, text="Write enable based on HW signal", variable=self.writeEnableOnHWSignalVar, offvalue=False, onvalue=True)

        DEFAULT_NUM_GPU_VENC_SESSIONS = 3
        self.maxGPUVencFrame = ttk.LabelFrame(self.acquisitionParametersFrame, text="Max GPU VEnc sessions", style='SingleContainer.TLabelframe')
        self.maxGPUVEncVar = tk.StringVar(); self.maxGPUVEncVar.set(str(DEFAULT_NUM_GPU_VENC_SESSIONS))
        self.maxGPUVEncEntry = ttk.Entry(self.maxGPUVencFrame, width=16, textvariable=self.maxGPUVEncVar)

        self.selectAcquisitionHardwareButton =  ttk.Button(self.acquisitionParametersFrame, text="Select audio/video inputs", command=self.selectAcquisitionHardware)
        self.acquisitionHardwareText = tk.Text(self.acquisitionParametersFrame)

        self.chunkSizeVar =         tk.StringVar(); self.chunkSizeVar.set(1000)

        self.mergeFrame = cf.CollapsableFrame(self.controlFrame, collapseText="AV File Merging", **collapsableFrameStyle); self.mergeFrame.stateChangeButton.config(**collapsableFrameButtonStyle)
        # self.mergeFrame = ttk.LabelFrame(self.acquisitionFrame, text="AV File merging")

        self.mergeFileWidget = FileWritingEntry(
            self.mergeFrame,
            defaultDirectory='',
            defaultBaseFileName='mergeWrite',
            purposeText='merging audio/video',
            text="Merged A/V Writing"
        )
        self.mergeFileWidget.setDirectoryChangeHandler(self.mergeDirectoryChangeHandler)
        self.mergeFileWidget.setBaseFileNameChangeHandler(self.mergeBaseFileNameChangeHandler)

        self.mergeFilesVar =        tk.BooleanVar(); self.mergeFilesVar.set(False)
        self.mergeFilesCheckbutton = ttk.Checkbutton(self.mergeFrame, text="Merge audio/video", variable=self.mergeFilesVar, offvalue=False, onvalue=True)
        self.mergeFilesVar.trace('w', self.updateAVMergerState)

        self.deleteMergedFilesFrame = ttk.LabelFrame(self.mergeFrame, text="Delete merged...")

        self.deleteMergedAudioFilesVar = tk.BooleanVar(); self.deleteMergedAudioFilesVar.set(False)
        self.deleteMergedAudioFilesCheckbutton = ttk.Checkbutton(self.deleteMergedFilesFrame, text="Audio files", variable=self.deleteMergedAudioFilesVar, offvalue=False, onvalue=True)
        self.deleteMergedAudioFilesVar.trace('w', lambda *args: self.changeAVMergerParams(deleteMergedAudioFiles=self.deleteMergedAudioFilesVar.get()))

        self.deleteMergedVideoFilesVar = tk.BooleanVar(); self.deleteMergedVideoFilesVar.set(True)
        self.deleteMergedVideoFilesCheckbutton = ttk.Checkbutton(self.deleteMergedFilesFrame, text="Video files", variable=self.deleteMergedVideoFilesVar, offvalue=False, onvalue=True)
        self.deleteMergedVideoFilesVar.trace('w', lambda *args: self.changeAVMergerParams(deleteMergedVideoFiles=self.deleteMergedVideoFilesVar.get()))

        self.montageMergeVar = tk.BooleanVar(); self.montageMergeVar.set(False)
        self.montageMergeCheckbutton = ttk.Checkbutton(self.mergeFrame, text="Montage-merge videos", variable=self.montageMergeVar, offvalue=False, onvalue=True)
        self.montageMergeVar.trace('w', lambda *args: self.changeAVMergerParams(montage=self.montageMergeVar.get()))

        self.mergeCompressionFrame = ttk.LabelFrame(self.mergeFrame, text="Compression:")
        self.mergeCompressionVar = tk.StringVar(); self.mergeCompressionVar.set('23')
        self.mergeCompression = ttk.Combobox(self.mergeCompressionFrame, textvariable=self.mergeCompressionVar, values=AVMerger.COMPRESSION_OPTIONS, width=12)
        self.mergeCompressionVar.trace('w', lambda *args: self.changeAVMergerParams(compression=self.mergeCompressionVar.get()))

        self.fileSettingsFrame = cf.CollapsableFrame(self.controlFrame, collapseText="File writing settings", **collapsableFrameStyle); self.fileSettingsFrame.stateChangeButton.config(**collapsableFrameButtonStyle)

        self.daySubfoldersVar = tk.BooleanVar(); self.daySubfoldersVar.set(True)
        self.daySubfoldersCheckbutton = ttk.Checkbutton(self.fileSettingsFrame, text="File in day subfolders", variable=self.daySubfoldersVar)
        self.daySubfoldersVar.trace('w', lambda *args: self.updateDaySubfolderSetting())

        self.scheduleFrame = cf.CollapsableFrame(self.controlFrame, collapseText="Recording Schedule", **collapsableFrameStyle); self.scheduleFrame.stateChangeButton.config(**collapsableFrameButtonStyle)

        self.scheduleEnabledVar = tk.BooleanVar(); self.scheduleEnabledVar.set(False)
        self.scheduleEnabledVar.trace('w', self.updateChildSchedulingState)
        self.scheduleEnabledCheckbutton = ttk.Checkbutton(self.scheduleFrame, text="Restrict trigger to schedule", variable=self.scheduleEnabledVar)
        self.scheduleStartVar = TimeVar(); self.scheduleStartVar.trace('w', self.updateChildSchedulingState)
        self.scheduleStartTimeEntry = TimeEntry(self.scheduleFrame, text="Start time", style=self.style)
        self.scheduleStopVar = TimeVar(); self.scheduleStopVar.trace('w', self.updateChildSchedulingState)
        self.scheduleStopTimeEntry = TimeEntry(self.scheduleFrame, text="Stop time")

        self.triggerFrame = cf.CollapsableFrame(self.controlFrame, collapseText="Triggering", **collapsableFrameStyle); self.triggerFrame.stateChangeButton.config(**collapsableFrameButtonStyle)
        # self.triggerFrame = ttk.LabelFrame(self.controlFrame, text='Triggering')

        self.triggerModes = ['Manual', 'Audio', 'Continuous', 'SimpleContinuous', 'None']
        self.triggerModeChooserFrame = ttk.Frame(self.triggerFrame)
        self.triggerModeVar = tk.StringVar(); self.triggerModeVar.set(self.triggerModes[0])
        self.triggerModeLabel = ttk.Label(self.triggerModeChooserFrame, text='Trigger mode:')
        self.triggerModeRadioButtons = {}
        self.triggerModeControlGroupFrames = {}

        self.triggerControlTabs = ttk.Notebook(self.triggerFrame)
        for mode in self.triggerModes:
            self.triggerModeRadioButtons[mode] = ttk.Radiobutton(self.triggerModeChooserFrame, text=mode, variable=self.triggerModeVar, value=mode)
            self.triggerModeControlGroupFrames[mode] = ttk.Frame(self.triggerControlTabs)
            self.triggerControlTabs.add(self.triggerModeControlGroupFrames[mode], text=mode)

        defaultMode = 'SimpleContinuous'
        self.triggerControlTabs.select(self.triggerModeControlGroupFrames[defaultMode])
        self.triggerModeVar.set(defaultMode)
        self.triggerModeVar.trace('w', self.updateTriggerMode)

        # Manual controls
        self.manualWriteTriggerButton = ttk.Button(self.triggerModeControlGroupFrames['Manual'], text="Manual write trigger", command=self.writeButtonClickHandler)

        # Audio trigger controls
        self.triggerHiLoFrame = ttk.Frame(self.triggerModeControlGroupFrames['Audio'])
        self.triggerHighLevelFrame = ttk.LabelFrame(self.triggerHiLoFrame, text="High volume threshold", style='SingleContainer.TLabelframe')
        self.triggerHighLevelVar = tk.StringVar(); self.triggerHighLevelVar.set("0.1")
        self.triggerHighLevelEntry = ttk.Entry(self.triggerHighLevelFrame, textvariable=self.triggerHighLevelVar); self.triggerHighLevelEntry.bind('<FocusOut>', self.updateAudioTriggerSettings)

        self.triggerLowLevelFrame = ttk.LabelFrame(self.triggerHiLoFrame, text="Low volume threshold", style='SingleContainer.TLabelframe')
        self.triggerLowLevelVar = tk.StringVar(); self.triggerLowLevelVar.set("0.05")
        self.triggerLowLevelEntry = ttk.Entry(self.triggerLowLevelFrame, textvariable=self.triggerLowLevelVar); self.triggerLowLevelEntry.bind('<FocusOut>', self.updateAudioTriggerSettings)

        self.triggerHighTimeFrame = ttk.LabelFrame(self.triggerHiLoFrame, text="High threshold time", style='SingleContainer.TLabelframe')
        self.triggerHighTimeVar = tk.StringVar(); self.triggerHighTimeVar.set("0.5")
        self.triggerHighTimeEntry = ttk.Entry(self.triggerHighTimeFrame, textvariable=self.triggerHighTimeVar); self.triggerHighTimeEntry.bind('<FocusOut>', self.updateAudioTriggerSettings)

        self.triggerLowTimeFrame = ttk.LabelFrame(self.triggerHiLoFrame, text="Low threshold time", style='SingleContainer.TLabelframe')
        self.triggerLowTimeVar = tk.StringVar(); self.triggerLowTimeVar.set("2.0")
        self.triggerLowTimeEntry = ttk.Entry(self.triggerLowTimeFrame, textvariable=self.triggerLowTimeVar); self.triggerLowTimeEntry.bind('<FocusOut>', self.updateAudioTriggerSettings)

        self.triggerHighFractionFrame = ttk.LabelFrame(self.triggerHiLoFrame, text="Frac. of time above high threshold", style='SingleContainer.TLabelframe')
        self.triggerHighFractionVar = tk.StringVar(); self.triggerHighFractionVar.set("0.1")
        self.triggerHighFractionEntry = ttk.Entry(self.triggerHighFractionFrame, textvariable=self.triggerHighFractionVar); self.triggerHighFractionEntry.bind('<FocusOut>', self.updateAudioTriggerSettings)

        self.triggerLowFractionFrame = ttk.LabelFrame(self.triggerHiLoFrame, text="Frac. of time below low threshold", style='SingleContainer.TLabelframe')
        self.triggerLowFractionVar = tk.StringVar(); self.triggerLowFractionVar.set("0.99")
        self.triggerLowFractionEntry = ttk.Entry(self.triggerLowFractionFrame, textvariable=self.triggerLowFractionVar); self.triggerLowFractionEntry.bind('<FocusOut>', self.updateAudioTriggerSettings)

        self.triggerHighBandpassFrame = ttk.LabelFrame(self.triggerHiLoFrame, text="High bandpass cutoff freq. (Hz)", style='SingleContainer.TLabelframe')
        self.triggerHighBandpassVar = tk.StringVar(); self.triggerHighBandpassVar.set("7000")
        self.triggerHighBandpassEntry = ttk.Entry(self.triggerHighBandpassFrame, textvariable=self.triggerHighBandpassVar); self.triggerHighBandpassEntry.bind('<FocusOut>', self.updateAudioTriggerSettings)

        self.triggerLowBandpassFrame = ttk.LabelFrame(self.triggerHiLoFrame, text="Low bandpass cutoff freq. (Hz)", style='SingleContainer.TLabelframe')
        self.triggerLowBandpassVar = tk.StringVar(); self.triggerLowBandpassVar.set("100")
        self.triggerLowBandpassEntry = ttk.Entry(self.triggerLowBandpassFrame, textvariable=self.triggerLowBandpassVar); self.triggerLowBandpassEntry.bind('<FocusOut>', self.updateAudioTriggerSettings)

        self.maxAudioTriggerTimeFrame = ttk.LabelFrame(self.triggerModeControlGroupFrames['Audio'], text="Max. audio trigger record time", style='SingleContainer.TLabelframe')
        self.maxAudioTriggerTimeVar = tk.StringVar(); self.maxAudioTriggerTimeVar.set("20.0")
        self.maxAudioTriggerTimeEntry = ttk.Entry(self.maxAudioTriggerTimeFrame, textvariable=self.maxAudioTriggerTimeVar); self.maxAudioTriggerTimeEntry.bind('<FocusOut>', self.updateAudioTriggerSettings)

        self.multiChannelStartBehaviorFrame = ttk.LabelFrame(self.triggerModeControlGroupFrames['Audio'], text="Start recording when...", style='SingleContainer.TLabelframe')
        self.multiChannelStartBehaviorVar = tk.StringVar(); self.multiChannelStartBehaviorVar.set("OR"); self.multiChannelStartBehaviorVar.trace('w', self.updateAudioTriggerSettings)
        self.multiChannelStartBehaviorOR = ttk.Radiobutton(self.multiChannelStartBehaviorFrame, text="...any channels stay above threshold", variable=self.multiChannelStartBehaviorVar, value="OR")
        self.multiChannelStartBehaviorAND = ttk.Radiobutton(self.multiChannelStartBehaviorFrame, text="...all channels stay above threshold", variable=self.multiChannelStartBehaviorVar, value="AND")

        self.multiChannelStopBehaviorFrame = ttk.LabelFrame(self.triggerModeControlGroupFrames['Audio'], text="Stop recording when...", style='SingleContainer.TLabelframe')
        self.multiChannelStopBehaviorVar = tk.StringVar(); self.multiChannelStopBehaviorVar.set("AND"); self.multiChannelStopBehaviorVar.trace('w', self.updateAudioTriggerSettings)
        self.multiChannelStopBehaviorOR = ttk.Radiobutton(self.multiChannelStopBehaviorFrame, text="...any channels stay below threshold", variable=self.multiChannelStopBehaviorVar, value="OR")
        self.multiChannelStopBehaviorAND = ttk.Radiobutton(self.multiChannelStopBehaviorFrame, text="...all channels stay below threshold", variable=self.multiChannelStopBehaviorVar, value="AND")

        self.audioTriggerStateFrame = ttk.LabelFrame(self.triggerModeControlGroupFrames['Audio'], text="Trigger state", style='SingleContainer.TLabelframe')
        self.audioTriggerStateLabel = ttk.Label(self.audioTriggerStateFrame)

        # Continuous trigger controls
        self.continuousTriggerModeStart = ttk.Button(self.triggerModeControlGroupFrames['Continuous'], text="Begin continuous write", command=self.continuousTriggerStartButtonClick)
        self.continuousTriggerModeStop = ttk.Button(self.triggerModeControlGroupFrames['Continuous'], text="End continuous write", command=self.continuousTriggerStopButtonClick)

        self.continuousTriggerPeriodFrame = ttk.LabelFrame(self.triggerModeControlGroupFrames['Continuous'], text="Video chunk length (sec)", style='SingleContainer.TLabelframe')
        self.continuousTriggerPeriodVar = tk.StringVar(); self.continuousTriggerPeriodVar.set("20")
        self.continuousTriggerPeriodEntry = ttk.Entry(self.continuousTriggerPeriodFrame, textvariable=self.continuousTriggerPeriodVar); self.continuousTriggerPeriodEntry.bind('<FocusOut>', self.updateContinuousTriggerSettings)

        self.audioTagContinuousTrigsVar = tk.BooleanVar(); self.audioTagContinuousTrigsVar.set(True)
        self.audioTagContinuousTrigsCheckbutton = ttk.Checkbutton(self.triggerModeControlGroupFrames['Continuous'], text="Tag files with high audio", variable=self.audioTagContinuousTrigsVar, offvalue=False, onvalue=True)
        self.audioTagContinuousTrigsVar.trace('w', self.updateContinuousTriggerSettings)

        # Simple continuous trigger controls
        self.manualSyncStartButton = ttk.Button(self.triggerModeControlGroupFrames['SimpleContinuous'], text="Sync start", command=self.startSyncProcess)

        # None trigger mode widgets
        self.noneTriggerModeLabel = ttk.Label(self.triggerModeControlGroupFrames['None'], text='No write triggers, only monitoring')

        # Audio analysis monitoring widgets
        self.audioAnalysisMonitorFrame = ttk.LabelFrame(self.triggerModeControlGroupFrames['Audio'], text="Audio analysis")
        self.audioAnalysisWidgets = {}
        self.analysisSummaryHistoryChunkLength = 100
        self.audioAnalysisSummaryHistory = deque(maxlen=self.analysisSummaryHistoryChunkLength)
        ########### Child process objects #####################

        # Monitoring queues for collecting audio and video data for user monitoring purposes
        self.monitorMasterFrameRate = 15

        # Pointers to processes
        self.videoWriteProcesses = {}
        self.videoAcquireProcesses = {}
        self.audioWriteProcess = None
        self.audioAcquireProcess = None
        self.audioTriggerProcess = None
        self.continuousTriggerProcess = None
        self.syncProcess = None
        self.mergeProcess = None
        self.StdoutManager = StdoutManager()
        self.StdoutManager.start()

        # Actual a/v frequency shared vars
        self.actualVideoFrequency = None
        self.actualAudioFrequency = None

        # Verbosity of child processes
        #   0=Errors, 1=Occasional important status updates
        #   2=Minor status updates, 3=Continuous status messages
        self.audioAcquireVerbose = 1
        self.audioWriteVerbose = 1
        self.videoAcquireVerbose = 1
        self.videoWriteVerbose = 1
        self.syncVerbose = 1
        self.mergeVerbose = 1
        self.audioTriggerVerbose = 1
        self.continuousTriggerVerbose = 1

        # self.profiler =  cProfile.Profile()

        # the params dict defines how to access and set all the parameters in the GUI
        self.paramInfo = {
            'audioFrequency':                   dict(get=lambda:int(self.audioFrequencyVar.get()),              set=self.audioFrequencyVar.set),
            'videoFrequency':                   dict(get=lambda:int(self.videoFrequencyVar.get()),              set=self.videoFrequencyVar.set),
            'chunkSize':                        dict(get=lambda:int(self.chunkSizeVar.get()),                   set=self.chunkSizeVar.set),
            "maxGPUVEnc":                       dict(get=lambda:int(self.maxGPUVEncVar.get()),                  set=self.maxGPUVEncVar.set),
            # 'exposureTime':                     dict(get=lambda:int(self.exposureTimeVar.get()),                set=self.exposureTimeVar.set),
            'gain':                             dict(get=lambda:float(self.gainVar.get()),                      set=self.gainVar.set),
            'preTriggerTime':                   dict(get=lambda:float(self.preTriggerTimeVar.get()),            set=self.preTriggerTimeVar.set),
            'acquisitionBufferSize':            dict(get=lambda:float(self.acquisitionBufferSizeVar.get()),     set=self.acquisitionBufferSizeVar.set),
            'recordTime':                       dict(get=lambda:float(self.recordTimeVar.get()),                set=self.recordTimeVar.set),
            'triggerHighLevel':                 dict(get=lambda:float(self.triggerHighLevelVar.get()),          set=self.triggerHighLevelVar.set),
            'triggerLowLevel':                  dict(get=lambda:float(self.triggerLowLevelVar.get()),           set=self.triggerLowLevelVar.set),
            'triggerHighTime':                  dict(get=lambda:float(self.triggerHighTimeVar.get()),           set=self.triggerHighTimeVar.set),
            'triggerLowTime':                   dict(get=lambda:float(self.triggerLowTimeVar.get()),            set=self.triggerLowTimeVar.set),
            'triggerHighFraction':              dict(get=lambda:float(self.triggerHighFractionVar.get()),       set=self.triggerHighFractionVar.set),
            'triggerLowFraction':               dict(get=lambda:float(self.triggerLowFractionVar.get()),        set=self.triggerLowFractionVar.set),
            'triggerHighBandpass':              dict(get=lambda:float(self.triggerHighBandpassVar.get()),       set=self.triggerHighBandpassVar.set),
            'triggerLowBandpass':               dict(get=lambda:float(self.triggerLowBandpassVar.get()),        set=self.triggerLowBandpassVar.set),
            'maxAudioTriggerTime':              dict(get=lambda:float(self.maxAudioTriggerTimeVar.get()),       set=self.maxAudioTriggerTimeVar.set),
            "videoExposureTime":                dict(get=lambda:float(self.videoExposureTimeVar.get()),            set=self.videoExposureTimeVar.set),
            'videoBaseFileNames':               dict(get=self.videoBaseFileNames.get,                            set=self.setVideoBaseFileNames),
            'videoDirectories':                 dict(get=self.videoDirectories.get,                              set=self.setVideoDirectories),
            'audioBaseFileName':                dict(get=self.audioBaseFileName.get,                             set=self.setAudioBaseFileName),
            'audioDirectory':                   dict(get=self.audioDirectory.get,                                set=self.setAudioDirectory),
            'mergeBaseFileName':                dict(get=self.mergeBaseFileName.get,                             set=self.setMergeBaseFileName),
            'mergeDirectory':                   dict(get=self.mergeDirectory.get,                                set=self.setMergeDirectory),
            'mergeFiles':                       dict(get=self.mergeFilesVar.get,                                set=self.mergeFilesVar.set),
            'deleteMergedAudioFiles':           dict(get=self.deleteMergedAudioFilesVar.get,                    set=self.deleteMergedAudioFilesVar.set),
            'deleteMergedVideoFiles':           dict(get=self.deleteMergedVideoFilesVar.get,                    set=self.deleteMergedVideoFilesVar.set),
            'montageMerge':                     dict(get=self.montageMergeVar.get,                              set=self.montageMergeVar.set),
            'mergeCompression':                 dict(get=self.mergeCompressionVar.get,                          set=self.mergeCompressionVar.set),
            'scheduleEnabled':                  dict(get=self.scheduleEnabledVar.get,                           set=self.scheduleEnabledVar.set),
            'scheduleStart':                    dict(get=self.scheduleStartVar.get,                             set=self.scheduleStartVar.set),
            'scheduleStop':                     dict(get=self.scheduleStopVar.get,                              set=self.scheduleStopVar.set),
            'triggerMode':                      dict(get=self.triggerModeVar.get,                               set=self.triggerModeVar.set),
            'multiChannelStopBehavior':         dict(get=self.multiChannelStopBehaviorVar.get,                  set=self.multiChannelStopBehaviorVar.set),
            'multiChannelStartBehavior':        dict(get=self.multiChannelStartBehaviorVar.get,                 set=self.multiChannelStartBehaviorVar.set),
            "bufferSizeSeconds":                dict(get=self.getBufferSizeSeconds,                             set=self.setBufferSizeSeconds),
            "bufferSizeAudioChunks":            dict(get=self.getBufferSizeAudioChunks,                         set=self.setBufferSizeAudioChunks),
            "numStreams":                       dict(get=self.getNumStreams,                                    set=self.setNumStreams),
            "numProcesses":                     dict(get=self.getNumProcesses,                                  set=self.setNumProcesses),
            "numSyncedProcesses":               dict(get=self.getNumSyncedProcesses,                            set=self.setNumSyncedProcesses),
            "acquireSettings":                  dict(get=self.getCameraSettings,                               set=self.setAcquireSettings),
            "continuousTriggerPeriod":          dict(get=lambda:float(self.continuousTriggerPeriodVar.get()),   set=self.continuousTriggerPeriodVar.set),
            "audioTagContinuousTrigs":          dict(get=self.audioTagContinuousTrigsVar.get,                   set=self.audioTagContinuousTrigsVar.set),
            "daySubfolders":                    dict(get=self.daySubfoldersVar.get,                             set=self.daySubfoldersVar.set),
            "audioDAQChannels":                 dict(get=self.audioDAQChannels.get,                             set=self.audioDAQChannels.set),
            "camSerials":                       dict(get=self.camSerials.get,                                   set=self.camSerials.set),
            "audioSyncTerminal":                dict(get=self.audioSyncTerminal.get,                            set=self.audioSyncTerminal.set),
            "videoSyncTerminal":                dict(get=self.videoSyncTerminal.get,                            set=self.videoSyncTerminal.set),
            "audioSyncSource":                  dict(get=self.audioSyncSource.get,                              set=self.audioSyncSource.set),
            "videoSyncSource":                  dict(get=self.videoSyncSource.get,                              set=self.videoSyncSource.set),
            "acquisitionSignalChannel":         dict(get=self.acquisitionSignalChannel.get,                     set=self.acquisitionSignalChannel.set),
            "audioChannelConfiguration":        dict(get=self.audioChannelConfiguration.get,                    set=self.audioChannelConfiguration.set),
            "videoWriteEnable":                 dict(get=self.videoWriteEnable.get,                             set=self.setVideoWriteEnable),
            "audioWriteEnable":                 dict(get=self.audioWriteEnable.get,                             set=self.setAudioWriteEnable),
            "startOnHWSignal":                  dict(get=self.startOnHWSignalVar.get,                           set=self.startOnHWSignalVar.set),
            "writeEnableOnHWSignal":            dict(get=self.writeEnableOnHWSignalVar.get,                     set=self.writeEnableOnHWSignalVar.set),
            "videoMonitorDisplaySize":          dict(get=self.videoMonitorDisplaySize.get,                      set=self.videoMonitorDisplaySize.set),
        }

        self.createAudioAnalysisMonitor()

        self.setupInputMonitoringWidgets()

        self.updateAcquisitionHardwareDisplay()

        self.update()

        # Start automatic updating of video and audio monitors
        self.audioMonitorUpdateJob = None
        self.videoMonitorUpdateJob = None
        self.audioAnalysisMonitorUpdateJob = None
        self.triggerIndicatorUpdateJob = None
        self.autoUpdateVideoMonitors()
        self.autoUpdateAudioMonitors()
        if self.triggerModeVar.get() == "Audio":
            self.autoUpdateAudioAnalysisMonitors()

        self.updateStatusDisplayJob = None
        self.metaState = None
        self.updateStatusDisplay()

        self.master.update_idletasks()

        # If provided, load settings file
        if settingsFilePath is not None:
            self.loadSettings(path=settingsFilePath)

    def log(self, msg, *args, **kwargs):
        """Add another tagged message to the currently accumulating log entry.

        Note that you must call endLog to cause accumulated messages to be
        logged together.

        Args:
            msg (str): The message to log
            *args: Any arguments to pass to the print statement
            **kwargs: Unused, I think

        Returns:
            None

        """
        # Add another message to the currently accumulating log entry
        syncPrint('|| {ID} - {msg}'.format(ID=self.ID, msg=msg), *args, buffer=self.stdoutBuffer, **kwargs)

    def endLog(self, state):
        """End accumulated log entry and output it.

        Args:
            state (str): Indication of what state the message sender is in.
                To use the function name as the state, use the inspect module:
                inspect.currentframe().f_code.co_name

        Returns:
            None

        """
        #
        if len(self.stdoutBuffer) > 0:
            self.log(r'*********************************** /\ {ID} {state} /\ ********************************************'.format(ID=self.ID, state=state))
            self.flushStdout()

    def flushStdout(self):
        """Flush the accumulated log and clear the buffer for the next entry.

        Returns:
            None

        """
        # Output currently accumulated log entry and clear buffer
        if len(self.stdoutBuffer) > 0:
            if self.StdoutManager is not None:
                self.StdoutManager.queue.put(self.stdoutBuffer)
            else:
                print('Warning, logging failed - stdout queue not created.')
                for msgBundle in self.stdoutBuffer:
                    args, kwargs = msgBundle
                    print(*args, **kwargs)
        self.stdoutBuffer = []

    def cleanupAndExit(self):
        """Attempt to gracefully shut everything down and exit.

        Cancel any automatic update jobs.

        Returns:
            None

        """
        self.stopMonitors()
        self.log("Stopping acquisition")
        self.haltChildProcesses()
        self.log("Destroying master")
        self.master.destroy()
        self.master.quit()
        self.log("Everything should be closed now!")
        self.endLog(inspect.currentframe().f_code.co_name)

    def setVerbosity(self):
        """Produce popup dialog box for setting child process logging verbosity.

        Get new verbosity settings, then update all child processes accordingly.

        Returns:
            None

        """
        verbosityOptions = ['0', '1', '2', '3']
        names = [
            'AudioAcquirer verbosity',
            'AudioWriter verbosity',
            'Synchronizer verbosity',
            'AVMerger verbosity',
            'AudioTriggerer verbosity',
            'ContinuousTriggerer verbosity',
            'VideoAcquirer verbosity',
            'VideoWriter verbosity'
        ]
        defaults = [
            str(int(self.audioAcquireVerbose)),
            str(int(self.audioWriteVerbose)),
            str(int(self.syncVerbose)),
            str(int(self.mergeVerbose)),
            str(int(self.audioTriggerVerbose)),
            str(int(self.continuousTriggerVerbose)),
            str(int(self.videoAcquireVerbose)),
            str(int(self.videoWriteVerbose))
        ]
        params = []
        for name, default in zip(names, defaults):
            params.append(
                Param(name=name, widgetType=Param.MONOCHOICE, default=default, options=verbosityOptions)
            )
        pd = ParamDialog(self.master, params=params, title="Set child process verbosity levels", arrangement=ParamDialog.BOX)
        choices = pd.results
        if choices is not None:
            self.audioAcquireVerbose = int(choices['AudioAcquirer verbosity'])
            self.audioWriteVerbose = int(choices['AudioWriter verbosity'])
            self.syncVerbose = int(choices['Synchronizer verbosity'])
            self.mergeVerbose = int(choices['AVMerger verbosity'])
            self.audioTriggerVerbose = int(choices['AudioTriggerer verbosity'])
            self.continuousTriggerVerbose = int(choices['ContinuousTriggerer verbosity'])
            self.videoAcquireVerbose = int(choices['VideoAcquirer verbosity'])
            self.videoWriteVerbose = int(choices['VideoWriter verbosity'])
        self.updateChildProcessVerbosity()

    def updateChildProcessVerbosity(self):
        """Update child process logging verbosity based on current settings

        Returns:
            None

        """
        self.sendMessage(self.audioAcquireProcess, (Messages.SETPARAMS, {'verbose':self.audioAcquireVerbose}))
        self.sendMessage(self.audioWriteProcess, (Messages.SETPARAMS, {'verbose':self.audioWriteVerbose}))
        self.sendMessage(self.syncProcess, (Messages.SETPARAMS, {'verbose':self.syncVerbose}))
        self.sendMessage(self.mergeProcess, (Messages.SETPARAMS, {'verbose':self.mergeVerbose}))
        self.sendMessage(self.audioTriggerProcess, (Messages.SETPARAMS, {'verbose':self.audioTriggerVerbose}))
        self.sendMessage(self.continuousTriggerProcess, (Messages.SETPARAMS, {'verbose':self.continuousTriggerVerbose}))
        for camSerial in self.videoAcquireProcesses:
            self.sendMessage(self.videoAcquireProcesses[camSerial], (Messages.SETPARAMS, {'verbose':self.videoAcquireVerbose}))
            self.sendMessage(self.videoWriteProcesses[camSerial], (Messages.SETPARAMS, {'verbose':self.videoWriteVerbose}))

    def updateDaySubfolderSetting(self, *args):
        """Change day subfolder setting in all child processes.

        Args:
            *args (any): Dummy variable to hold unused event data

        Returns:
            None

        """
        daySubfolders = self.getParams('daySubfolders')
        self.sendMessage(self.audioWriteProcess, (Messages.SETPARAMS, {'daySubfolders':daySubfolders}))
        self.sendMessage(self.mergeProcess, (Messages.SETPARAMS, {'daySubfolders':daySubfolders}))
        for camSerial in self.videoWriteProcesses:
            self.sendMessage(self.videoWriteProcesses[camSerial], (Messages.SETPARAMS, {'daySubfolders':daySubfolders}))

    def validateVideoExposureTime(self, *args):
        """Sanitize current video exposure time settings.

        Args:
            *args (any): Dummy variable to hold unused event data

        Returns:
            None

        """
        videoExposureTime = self.getParams('videoExposureTime')/1000
        videoFrequency = self.getParams('videoFrequency')

        minHighLowTime = 0.001
        maxDutyCycle = 1 - (minHighLowTime * videoFrequency)
        minDutyCycle = minHighLowTime * videoFrequency

        videoDutyCycle = convertExposureTimeToDutyCycle(videoExposureTime, videoFrequency)

        if videoDutyCycle > maxDutyCycle:
            videoExposureTime = convertDutyCycleToExposureTime(maxDutyCycle, videoFrequency)
        if videoDutyCycle < minDutyCycle:
            videoDutyCycle = convertDutyCycleToExposureTime(minDutyCycle, videoFrequency)

        self.setParams(videoExposureTime=videoExposureTime*1000)

    def validateGain(self, *args):
        """Sanitize current gain time settings.

        Args:
            *args (any): Dummy variable to hold unused event data

        Returns:
            None

        """
        gain = self.getParams('gain')
        if gain < 0:
            gain = 0
        # videoFrequency = self.getParams('videoFrequency')
        # maxExposureTime = 1000000 * 0.95 / videoFrequency
        # if exposureTime > maxExposureTime:
        #     exposureTime = maxExposureTime
        # exposureTime = int(exposureTime)
        self.setParams(gain=gain)

    def showHelpDialog(self, *args):
        """Show help dialog box (not implemented).

        Args:
            *args (any): Dummy variable to hold unused event data

        Returns:
            None

        """
        msg = 'Sorry, nothing here yet.'
        showinfo('PyVAQ Help', msg)

    def showAboutDialog(self, *args):
        """Show about dialog box.

        Args:
            *args (any): Dummy variable to hold unused event data

        Returns:
            None

        """
        msg = """Welcome to PyVAQ version {version}!

If it's working perfectly, then contact Brian Kardon (bmk27@cornell.edu) to let \
him know. Otherwise, I had nothing to do with it.

""".format(version=VERSION)

        showinfo('About PyVAQ', msg)

    def configureAudioMonitoring(self):
        """Show popup for user to select audio monitoring options.

        Returns:
            None

        """
        if self.audioMonitor is None:
            showinfo('Please initialize acquisition before configuring audio monitor')
            return

        p = self.getParams()
        audioMonitorSampleLength = round(self.audioMonitor.historyLength / p['audioFrequency'], 2)
        params = [
            Param(name='Audio autoscale', widgetType=Param.MONOCHOICE, options=['Auto', 'Manual'], default=('Auto' if self.audioMonitor.autoscale else 'Manual')),
            Param(name='Audio range', widgetType=Param.TEXT, options=None, default=str(self.audioMonitor.displayAmplitude)),
            Param(name='Audio history length (s)', widgetType=Param.TEXT, options=None, default=str(audioMonitorSampleLength))
        ]
        pd = ParamDialog(self.master, params=params, title="Configure audio monitoring")
        choices = pd.results
        if choices is not None:
            if 'Audio autoscale' in choices and len(choices['Audio autoscale']) > 0:
                try:
                    if choices['Audio autoscale'] == "Manual":
                        self.audioMonitor.autoscale = False
                    elif choices['Audio autoscale'] == "Auto":
                        self.audioMonitor.autoscale = True
                except ValueError:
                    pass
            if 'Audio range' in choices and len(choices['Audio range']) > 0:
                try:
                    self.audioMonitor.displayAmplitude = abs(float(choices['Audio range']))
                except ValueError:
                    pass
            if 'Audio history length' in choices and len(choices['Audio history length']) > 0:
                try:
                    self.audioMonitor.historyLength = float(choices['Audio history length']) * p['audioFrequency']
                except ValueError:
                    pass

    def configureVideoMonitoring(self):
        """Show popup for user to select video monitoring options.

        Returns:
            None

        """
        if len(self.cameraMonitors) == 0:
            showinfo('Please initialize acquisition before configuring audio monitor')
            return

        # Get current video monitor display size, to use as default
        w, h = self.getParams('videoMonitorDisplaySize')

        params = [
            Param(name='Video display width',  widgetType=Param.TEXT, options=None, default=str(w)),
            Param(name='Video display height', widgetType=Param.TEXT, options=None, default=str(h))
        ]
        pd = ParamDialog(self.master, params=params, title="Configure video monitoring")
        choices = pd.results
        if choices is not None:
            if 'Video display width' in choices:
                w = int(choices['Video display width'])
            if 'Video display height' in choices:
                h = int(choices['Video display height'])
        # Set new video monitor display size
        self.videoMonitorDisplaySize.set((w, h))

    def updateVideoMonitorDisplaySize(self):
        """Update all video monitors with the current size setting.

        Returns:
            None

        """
        #
        newSize = self.videoMonitorDisplaySize.get()
        for camSerial in self.cameraMonitors:
            self.cameraMonitors[camSerial].setDisplaySize(newSize)

    def selectAcquisitionHardware(self, *args):
        """Create a popup for the user to select acquisition hardware.

        Args:
            *args (any): Dummy variable to hold unused event data

        Returns:
            None

        """
        # Create a popup for the user to select acquisition options

        # Get current settings to use as defaults
        p = self.getParams(
            "audioDAQChannels",
            "camSerials",
            "audioSyncTerminal",
            "videoSyncTerminal",
            "audioSyncSource",
            "videoSyncSource",
            "acquisitionSignalChannel",
            "audioChannelConfiguration"
            )

        defaultAudioDAQChannels = p["audioDAQChannels"]
        defaultCamSerials = p["camSerials"]
        defaultAudioSyncTerminal = p["audioSyncTerminal"]
        defaultVideoSyncTerminal = p["videoSyncTerminal"]
        defaultAudioSyncSource = p["audioSyncSource"]
        defaultVideoSyncSource = p["videoSyncSource"]
        defaultAcquisitionSignalChannel = p["acquisitionSignalChannel"]
        defaultAudioChannelConfiguration = p["audioChannelConfiguration"]

        # Query the system to determine what DAQ channels and cameras are
        #   currently available
        availableAudioChannels = flattenList(discoverDAQAudioChannels().values())
        availableClockChannels = flattenList(discoverDAQClockChannels().values()) + ['None']
        availableDigitalChannels = ['None'] + flattenList(discoverDAQTerminals().values())
        availableCamSerials = psu.discoverCameras()
        audioChannelConfigurations = [
            "DEFAULT",
            "DIFFERENTIAL",
            "NRSE",
            "PSEUDODIFFERENTIAL",
            "RSE"
        ]

        # Define and create GUI elements
        params = []
        if len(availableAudioChannels) > 0:
            params.append(Param(name='Audio Channels', widgetType=Param.MULTICHOICE, options=availableAudioChannels, default=defaultAudioDAQChannels))
        if len(availableCamSerials) > 0:
            params.append(Param(name='Cameras', widgetType=Param.MULTICHOICE, options=availableCamSerials, default=defaultCamSerials))
        if len(availableClockChannels) > 0:
            params.append(Param(name='Audio Sync Channel', widgetType=Param.MONOCHOICE, options=availableClockChannels, default=defaultAudioSyncTerminal))
            params.append(Param(name='Video Sync Channel', widgetType=Param.MONOCHOICE, options=availableClockChannels, default=defaultVideoSyncTerminal))
            params.append(Param(name='Audio Sync PFI Interface', widgetType=Param.TEXT, options=None, default=defaultAudioSyncSource, description="This must match your selection for Audio Sync Channel. Check DAQ pinout for matching PFI channel."))
            params.append(Param(name='Video Sync PFI Interface', widgetType=Param.TEXT, options=None, default=defaultVideoSyncSource, description="This must match your selection for Video Sync Channel. Check DAQ pinout for matching PFI channel."))
        params.append(Param(name='Audio channel configuration', widgetType=Param.MONOCHOICE, options=audioChannelConfigurations, default=defaultAudioChannelConfiguration, description="Choose an analog channel configuration for audio acquisition. Recommend differential if you have a 3-wire XLR-type output, RSE if you only use two wires."))
        params.append(Param(name='Acquisition start trigger channel', widgetType=Param.MONOCHOICE, options=availableDigitalChannels, default=defaultAcquisitionSignalChannel, description="Choose a channel that will trigger the acquisition start with a rising edge. Leave as None if you wish the acquisition to start without waiting for a digital trigger."))
        params.append(Param(name='Start acquisition immediately', widgetType=Param.MONOCHOICE, options=['Yes', 'No'], default='No'))

        choices = None
        if len(params) > 0:
            pd = ParamDialog(self.master, params=params, title="Choose audio/video inputs to use", maxHeight=35, arrangement=ParamDialog.HYBRID)
            choices = pd.results
            if choices is not None:
                # We're changing acquisition settings, so stop everything
                self.stopMonitors()
                # self.updateAcquisitionButton()
                self.destroyChildProcesses()

                # Extract chosen parameters from GUI
                if 'Audio Channels' in choices:
                    audioDAQChannels = choices['Audio Channels']
                else:
                    audioDAQChannels = []
                if 'Cameras' in choices:
                    camSerials = choices['Cameras']
                else:
                    camSerials = []
                if 'Audio Sync Channel' in choices and choices['Audio Sync Channel'] != "None":
                    audioSyncTerminal = choices['Audio Sync Channel']
                else:
                    audioSyncTerminal = None
                if 'Video Sync Channel' in choices and choices['Video Sync Channel'] != "None":
                    videoSyncTerminal = choices['Video Sync Channel']
                else:
                    videoSyncTerminal = None
                if 'Audio Sync PFI Interface' in choices and len(choices['Audio Sync PFI Interface']) > 0:
                    audioSyncSource = choices['Audio Sync PFI Interface']
                else:
                    audioSyncSource = None
                if 'Video Sync PFI Interface' in choices and len(choices['Video Sync PFI Interface']) > 0:
                    videoSyncSource = choices['Video Sync PFI Interface']
                else:
                    videoSyncSource = None
                if 'Acquisition signal channel' in choices and len(choices['Acquisition start trigger channel']) > 0:
                    if choices['Acquisition start trigger channel'] == 'None':
                        acquisitionSignalChannel = None
                    else:
                        acquisitionSignalChannel = choices['Acquisition start trigger channel']
                else:
                    acquisitionSignalChannel = None
                if 'Audio channel configuration' in choices and len(choices['Audio channel configuration']) > 0:
                    audioChannelConfiguration = choices['Audio channel configuration']

                # Set chosen parameters
                self.setParams(
                    audioDAQChannels=audioDAQChannels,
                    camSerials=camSerials,
                    audioSyncTerminal=audioSyncTerminal,
                    videoSyncTerminal=videoSyncTerminal,
                    audioSyncSource=audioSyncSource,
                    videoSyncSource=videoSyncSource,
                    acquisitionSignalChannel=acquisitionSignalChannel,
                    audioChannelConfiguration=audioChannelConfiguration
                    )

                self.log('Got audioDAQChannels:', audioDAQChannels)
                self.log('Got camSerials:', camSerials)

                # Create GUI elements for monitoring the chosen inputs
                self.setupInputMonitoringWidgets()

                # Update display text
                self.updateAcquisitionHardwareDisplay()

                # Restart child processes with new acquisition values
                self.createChildProcesses()
                if 'Start acquisition immediately' in choices and choices['Start acquisition immediately'] == 'Yes':
                    self.initializeChildProcesses()
                # self.updateAcquisitionButton()
                self.startMonitors()
            else:
                self.log('User input cancelled.')
        else:
            showinfo('No inputs', 'No compatible audio/video inputs found. Please connect at least one USB3 vision camera for video input and/or a NI USB DAQ for audio input and synchronization.')

        self.endLog(inspect.currentframe().f_code.co_name)

    def updateAcquisitionHardwareDisplay(self):
        """Update display showing selected acquisition hardware.

        Returns:
            None

        """
        lines = []

        p = self.getParams(
            "audioDAQChannels",
            "camSerials",
            "audioSyncTerminal",
            "videoSyncTerminal",
            "audioSyncSource",
            "videoSyncSource",
            "acquisitionSignalChannel",
            "audioChannelConfiguration"
            )

        lines.extend([
            'Acquisition hardware selections:',
            '  Audio DAQ channels:   {audioDAQChannels}'.format(audioDAQChannels=', '.join(p['audioDAQChannels'])),
            '  Cameras:              {camSerials}'.format(camSerials=', '.join(p['camSerials'])),
            '  Audio sync terminal:  {audioSyncTerminal}'.format(audioSyncTerminal=p['audioSyncTerminal']),
            '  Video sync terminal:  {videoSyncTerminal}'.format(videoSyncTerminal=p['videoSyncTerminal']),
            '  Audio sync source:    {audioSyncSource}'.format(audioSyncSource=p['audioSyncSource']),
            '  Video sync source:    {videoSyncSource}'.format(videoSyncSource=p['videoSyncSource']),
            '  Acq signal channel:   {acquisitionSignalChannel}'.format(acquisitionSignalChannel=p['acquisitionSignalChannel']),
            '  Audio channel config: {audioChannelConfiguration}'.format(audioChannelConfiguration=p['audioChannelConfiguration'])
        ])

        self.acquisitionHardwareText.delete('0.0', tk.END)
        self.acquisitionHardwareText['height'] = len(lines)
        self.acquisitionHardwareText.insert('0.0', '\n'.join(lines))

    def destroyInputMonitoringWidgets(self):
        """Destroy camera and audo monitor widgets.

        Returns:
            None

        """
        oldCamSerials = self.cameraMonitors.keys()
        for camSerial in oldCamSerials:
            self.cameraMonitors[camSerial].grid_forget()
            self.cameraMonitors[camSerial].destroy()
            del self.cameraMonitors[camSerial]
        self.audioMonitor

    def setupInputMonitoringWidgets(self):
        """Set up widgets for selected audio and video inputs.

        Returns:
            None

        """
        # Set up widgets and other entities for specific selected audio and video inputs

        p = self.getParams(
            'camSerials',
            'audioDAQChannels',
            'audioBaseFileName',
            'audioDirectory',
            'videoBaseFileNames',
            'videoDirectories'
            )
        camSerials = p["camSerials"]
        audioDAQChannels = p["audioDAQChannels"]
        audioBaseFileName = p["audioBaseFileName"]
        audioDirectory = p["audioDirectory"]
        videoBaseFileNames = p["videoBaseFileNames"]
        videoDirectories = p["videoDirectories"]

        # Destroy old video stream monitoring widgets
        oldCamSerials = list(self.cameraMonitors.keys())
        for camSerial in oldCamSerials:
            self.cameraMonitors[camSerial].grid_forget()
            self.cameraMonitors[camSerial].destroy()
            del self.cameraMonitors[camSerial]

        self.cameraSpeeds = dict([(camSerial, psu.checkCameraSpeed(camSerial=camSerial)) for camSerial in camSerials])

        # Create new video stream monitoring widgets and other entities
        for camSerial in camSerials:
            if camSerial in videoDirectories:
                videoDirectory = videoDirectories[camSerial]
            else:
                videoDirectory = ''
            if camSerial in videoBaseFileNames:
                videoBaseFileName = videoBaseFileNames[camSerial]
            else:
                videoBaseFileName = ''
            self.cameraMonitors[camSerial] = CameraMonitor(
                self.videoMonitorMasterFrame,
                displaySize=self.videoMonitorDisplaySize.get(),
                camSerial=camSerial,
                speedText=self.cameraSpeeds[camSerial],
                initialDirectory=videoDirectory,
                initialBaseFileName=videoBaseFileName
            )
            self.cameraMonitors[camSerial].setDirectoryChangeHandler(self.videoDirectoryChangeHandler)
            self.cameraMonitors[camSerial].setBaseFileNameChangeHandler(self.videoBaseFileNameChangeHandler)
            self.cameraMonitors[camSerial].setEnableWriteChangeHandler(self.videoWriteEnableChangeHandler)

        if len(camSerials) == 0:
            # Don't display docker buttons
            self.videoMonitorDocker.unDockButton.grid_forget()
            self.videoMonitorDocker.reDockButton.grid_forget()
        else:
            # Re-dock video monitor, which includes making sure docker buttons
            #   are displayed properly.
            self.videoMonitorDocker.reDock()

        # Create new audio stream monitoring widgets
        if self.audioMonitor is None:

            def unDockFunction(d):
                d.unDockButton.grid_forget()
                d.reDockButton.grid(row=0, column=0, sticky=tk.NW)
                self.update()
            def reDockFunction(d):
                d.reDockButton.grid_forget()
                d.unDockButton.grid(row=0, column=0, sticky=tk.NW)
                d.docker.grid(row=1, column=0)
                self.update()

            self.audioMonitorDocker = Docker(
                self.monitorMasterFrame, root=self.master,
                unDockFunction=unDockFunction, reDockFunction=reDockFunction,
                unDockText='undock', reDockText='dock', background='#d9d9d9')
            self.audioMonitorDocker.unDockButton.grid(row=0, column=0, sticky=tk.NW)
            self.audioMonitorDocker.reDockButton.grid(row=0, column=0, sticky=tk.NW)
            self.audioMonitorDocker.reDockButton.grid_forget()

            self.audioMonitor = AudioMonitor(
                self.audioMonitorDocker.docker,
                initialDirectory=audioDirectory,
                initialBaseFileName=audioBaseFileName
                )
            self.audioMonitor.grid(row=1, column=0)

            self.audioMonitor.setEnableWriteChangeHandler(self.audioWriteEnableChangeHandler)
        if audioDAQChannels is None or len(audioDAQChannels) == 0:
            # Don't display docker buttons
            self.audioMonitorDocker.unDockButton.grid_forget()
            self.audioMonitorDocker.reDockButton.grid_forget()
        else:
            # Re-dock audio monitor, which includes making sure docker buttons
            #   are displayed properly.
            self.audioMonitorDocker.reDock()
        self.audioMonitor.updateChannels(audioDAQChannels)
        self.audioMonitor.setDirectoryChangeHandler(self.audioDirectoryChangeHandler)
        self.audioMonitor.setBaseFileNameChangeHandler(self.audioBaseFileNameChangeHandler)
        self.update()

    def updateAudioTriggerSettings(self, *args):
        """Update settings that determine parameters for audio-based triggering

        Args:
            *args (any): Dummy variable to hold unused event data

        Returns:
            None

        """
        # Update settings that determine for what audio values the GUI will
        #   send a record trigger.
        if self.audioTriggerProcess is not None:
            paramList = [
                'triggerHighLevel',
                'triggerLowLevel',
                'triggerHighTime',
                'triggerLowTime',
                'triggerHighFraction',
                'triggerLowFraction',
                'maxAudioTriggerTime',
                'multiChannelStartBehavior',
                'multiChannelStopBehavior'
            ]
            params = self.getParams(*paramList)
            self.sendMessage(self.audioTriggerProcess, (Messages.SETPARAMS, params))

    def updateContinuousTriggerSettings(self, *args):
        """Update settings for continuous triggering

        Continuous triggererer sends consecutive triggers, one after another

        Args:
            *args (any): Dummy variable to hold unused event data

        Returns:
            None

        """
        if self.continuousTriggerProcess is not None:
            paramList = [
                'continuousTriggerPeriod',
            ]
            params = self.getParams(*paramList, mapping=True)
            self.sendMessage(self.continuousTriggerProcess, (Messages.SETPARAMS, params))

            if self.audioTriggerProcess is not None:
                if self.getParams('audioTagContinuousTrigs'):
                    # Tell audio triggerer to start analyzing and send any tag triggers.
                    self.sendMessage(self.audioTriggerProcess, (Messages.STARTANALYZE, None))
                    self.sendMessage(self.audioTriggerProcess, (Messages.SETPARAMS, dict(tagTriggerEnabled=True)))
                else:
                    # Tell audio triggerer to stop analyzing and don't send any tag triggers.
                    self.sendMessage(self.audioTriggerProcess, (Messages.STOPANALYZE, None))
                    self.sendMessage(self.audioTriggerProcess, (Messages.SETPARAMS, dict(tagTriggerEnabled=False)))
            else:
                self.log('Warning, audio trigger process not available for continuous trigger tagging')
                self.endLog(inspect.currentframe().f_code.co_name)

    def updateAVMergerState(self, *args):
        """Update GUI & AVMerger process to reflect current AV merging settings

        Args:
            *args (any): Dummy variable to hold unused event data

        Returns:
            None

        """
        merging = self.mergeFilesVar.get()
        if merging:
            # User requests AV merging; enable settings and inform child process
            self.deleteMergedVideoFilesCheckbutton.config(state=tk.NORMAL)
            self.deleteMergedAudioFilesCheckbutton.config(state=tk.NORMAL)
            self.montageMergeCheckbutton.config(state=tk.NORMAL)
            self.sendMessage(self.mergeProcess, (Messages.START, None))
        else:
            # User has not requested AV merging; disable settings and inform
            #   child process
            self.deleteMergedVideoFilesCheckbutton.config(state=tk.DISABLED)
            self.deleteMergedAudioFilesCheckbutton.config(state=tk.DISABLED)
            self.montageMergeCheckbutton.config(state=tk.DISABLED)
            self.sendMessage(self.mergeProcess, (Messages.CHILL, None))

    def updateChildSchedulingState(self, *args):
        """Inform relevant child processes of current record schedule settings.

        Args:
            *args (any): Dummy variable to hold unused event data

        Returns:
            None

        """
        scheduleParams = self.getParams('scheduleEnabled', 'scheduleStart', 'scheduleStop')
        for camSerial in self.videoWriteProcesses:
            self.sendMessage(self.videoWriteProcesses[camSerial], (Messages.SETPARAMS, scheduleParams))
        if self.audioWriteProcess is not None:
            self.sendMessage(self.audioWriteProcess, (Messages.SETPARAMS, scheduleParams))

    def updateAudioFrequency(self, *args):
        """Send message to Synchronizer to update audio frequency

        Note that this will not take effect until the Synchronizer passes
        through the INITIALIZING state.

        Args:
            *args (any): Dummy variable to hold unused event data

        Returns:
            None

        """
        # Get current audio frequency parameter
        newFrequency = self.getParams('audioFrequency')
        # Send it to the Synchronizer
        self.sendMessage(self.syncProcess, (Messages.SETPARAMS, {'audioFrequency':newFrequency}))

    def updateVideoFrequency(self, *args):
        """Send message to Synchronizer to update video frequency

        Note that this will not take effect until the Synchronizer passes
        through the INITIALIZING state.

        Args:
            *args (any): Dummy variable to hold unused event data

        Returns:
            None

        """
        # Get current video frequency parameter
        newFrequency = self.getParams('videoFrequency')
        # Send it to the Synchronizer
        self.sendMessage(self.syncProcess, (Messages.SETPARAMS, {'videoFrequency':newFrequency}))

    def changeAVMergerParams(self, **params):
        """Inform AVMerger child process of current file merge settings.

        Args:
            *args (any): Dummy variable to hold unused event data

        Returns:
            None

        """
        self.sendMessage(self.mergeProcess, (Messages.SETPARAMS, params))

    def audioWriteEnableChangeHandler(self, *args):
        """Handle changes in audioWriteEnable

        Args:
            *args (any): Dummy variable to hold unused event data

        Returns:
            None

        """
        audioWriteEnable = self.audioMonitor.getEnableWrite()
        self.setAudioWriteEnable(audioWriteEnable, updateGUI=False)
    def videoWriteEnableChangeHandler(self, *args):
        """Handle changes in videoWriteEnable

        Args:
            *args (any): Dummy variable to hold unused event data

        Returns:
            None

        """
        videoWriteEnables = {}
        for camSerial in self.cameraMonitors:
            videoWriteEnables[camSerial] = self.cameraMonitors[camSerial].getEnableWrite()
        self.setVideoWriteEnable(videoWriteEnables, updateGUI=False)
    def videoBaseFileNameChangeHandler(self, *args):
        """Handle changes in videoBaseFileName

        Args:
            *args (any): Dummy variable to hold unused event data

        Returns:
            None

        """
        videoBaseFileNames = {}
        for camSerial in self.cameraMonitors:
            videoBaseFileNames[camSerial] = self.cameraMonitors[camSerial].getBaseFileName()
        self.setVideoBaseFileNames(videoBaseFileNames, updateGUI=False)
    def videoDirectoryChangeHandler(self, *args):
        """Handle changes in videoDirectory

        Args:
            *args (any): Dummy variable to hold unused event data

        Returns:
            None

        """
        videoDirectories = {}
        for camSerial in self.cameraMonitors:
            videoDirectories[camSerial] = self.cameraMonitors[camSerial].getDirectory()
        self.setVideoDirectories(videoDirectories, updateGUI=False)
    def audioBaseFileNameChangeHandler(self, *args):
        """Handle changes in audioBaseFileName

        Args:
            *args (any): Dummy variable to hold unused event data

        Returns:
            None

        """
        newAudioBaseFileName = self.audioMonitor.getBaseFileName()
        self.setAudioBaseFileName(newAudioBaseFileName, updateGUI=False)
    def audioDirectoryChangeHandler(self, *args):
        """Handle changes in audioDirectory

        Args:
            *args (any): Dummy variable to hold unused event data

        Returns:
            None

        """
        newAudioDirectory = self.audioMonitor.getDirectory()
        self.setAudioDirectory(newAudioDirectory, updateGUI=False)
    def mergeBaseFileNameChangeHandler(self, *args):
        """Handle changes in mergeBaseFileName

        Args:
            *args (any): Dummy variable to hold unused event data

        Returns:
            None

        """
        newMergeBaseFileName = self.mergeFileWidget.getBaseFileName()
        self.setMergeBaseFileName(newMergeBaseFileName, updateGUI=False)
    def mergeDirectoryChangeHandler(self, *args):
        """Handle changes in mergeDirectory

        Args:
            *args (any): Dummy variable to hold unused event data

        Returns:
            None

        """
        newMergeDirectory = self.mergeFileWidget.getDirectory()
        self.setMergeDirectory(newMergeDirectory, updateGUI=False)

    def updateTriggerMode(self, *args):
        """Handle a user selection of a new trigger mode.

        Args:
            *args (type): Description of parameter `*args`.

        Returns:
            type: Description of returned object.

        """
        newMode = self.triggerModeVar.get()

        if newMode in ["Continuous", "SimpleContinuous"]:
            # PreTrigger value is not relevant
            self.preTriggerTimeEntry['state'] = tk.DISABLED
        else:
            self.preTriggerTimeEntry.config(state=tk.NORMAL)

        if newMode != "Continuous":
            self.sendMessage(self.continuousTriggerProcess, (Messages.STOP, None))
        if newMode != "Audio":
            # May as well stop analyzing audio if we're not in audio mode.
            self.sendMessage(self.audioTriggerProcess, (Messages.STOPANALYZE, None))

        if self.audioAnalysisMonitorUpdateJob is not None:
            # If there was already an audio analysis monitoring job running, cancel it
            self.master.after_cancel(self.audioAnalysisMonitorUpdateJob)

        if newMode == "Audio":
            # User selected "Audio" trigger mode
            self.sendMessage(self.audioTriggerProcess, (Messages.STARTANALYZE, None))
            self.sendMessage(self.audioTriggerProcess, (Messages.SETPARAMS, dict(writeTriggerEnabled=True, tagTriggerEnabled=False)))
            self.autoUpdateAudioAnalysisMonitors()
        elif newMode == "Continuous":
            # User selected "Continuous" trigger mode
            self.sendMessage(self.continuousTriggerProcess, (Messages.START, None))
            if self.getParams('audioTagContinuousTrigs'):
                self.sendMessage(self.audioTriggerProcess, (Messages.STARTANALYZE, None))
                self.sendMessage(self.audioTriggerProcess, (Messages.SETPARAMS, dict(writeTriggerEnabled=False, tagTriggerEnabled=True)))
            else:
                self.sendMessage(self.audioTriggerProcess, (Messages.STOPANALYZE, None))
                self.sendMessage(self.audioTriggerProcess, (Messages.SETPARAMS, dict(writeTriggerEnabled=False, tagTriggerEnabled=False)))
        # elif newMode == "SimpleContinuous":
        #     # User selected "SimpleContinuous" trigger mode
        #     self.restartAcquisition()

        self.update()

    def createAudioAnalysisMonitor(self):
        """Set up axes/plots to display audio analysis data from AudioTriggerer.

        Uses matplotlib axes and plots to display audio analysis data from
        AudioTriggerer object

        Returns:
            None

        """
        audioDAQChannels = self.getParams("audioDAQChannels")

        # Create figure
        self.audioAnalysisWidgets['figure'] = fig = Figure(figsize=(7, 0.75), dpi=100, facecolor=WIDGET_COLORS[1])

        gs = gridspec.GridSpec(1, 2, width_ratios=[6, 1])

        # Create plot for volume vs time trace
        chunkSize = self.getParams('chunkSize')
        t = np.arange(int(self.analysisSummaryHistoryChunkLength))
        self.audioAnalysisWidgets['volumeTraceAxes'] = vtaxes = fig.add_subplot(gs[0])  # Axes to display
        vtaxes.autoscale(enable=True)
        vtaxes.plot(t, 0 * t, LINE_STYLES[0], linewidth=1)
        vtaxes.plot([0, 0], [0, 1], color='r')
        vtaxes.plot([0, 0], [0, 1], color='g')
        vtaxes.relim()
        vtaxes.autoscale_view(True, True, True)
        vtaxes.margins(x=0, y=0)

        # Create bar chart for current high/low volume fraction
        self.audioAnalysisWidgets['volumeFracAxes'] = vfaxes = fig.add_subplot(gs[1])   # Axes to display fraction of time above/below high/low threshold
        vfaxes.autoscale(enable=True)
        xValues = [k for k in range(len(audioDAQChannels))]
        zeroChannelValues = [0 for k in range(len(audioDAQChannels))]
        oneChannelValues = [1 for k in range(len(audioDAQChannels))]
        self.audioAnalysisWidgets['lowFracBars'] = []
        self.audioAnalysisWidgets['highFracBars'] = []
        vfaxes.relim()
        vfaxes.autoscale_view(True, True, True)
        vfaxes.margins(x=0, y=0)

        #fig.tight_layout()
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0, hspace=0)
        self.audioAnalysisWidgets['canvas'] = FigureCanvasTkAgg(fig, master=self.audioAnalysisMonitorFrame)  # A tk.DrawingArea.
        self.audioAnalysisWidgets['canvas'].draw()

    def stopMonitors(self):
        """Stop automatic update jobs.

        Returns:
            None

        """
        if self.audioMonitorUpdateJob is not None:
            self.master.after_cancel(self.audioMonitorUpdateJob)
            self.audioMonitorUpdateJob = None
        if self.videoMonitorUpdateJob is not None:
            self.master.after_cancel(self.videoMonitorUpdateJob)
            self.videoMonitorUpdateJob = None
        if self.audioAnalysisMonitorUpdateJob is not None:
            self.master.after_cancel(self.audioAnalysisMonitorUpdateJob)
            self.audioAnalysisMonitorUpdateJob = None
        if self.triggerIndicatorUpdateJob is not None:
            self.master.after_cancel(self.triggerIndicatorUpdateJob)
            self.triggerIndicatorUpdateJob = None
        # if self.updateStatusDisplayJob is not None:
        #     print('stopping update state display job...')
        #     self.master.after_cancel(self.updateStatusDisplayJob)
        #     self.updateStatusDisplayJob = None
        #     print('...done stopping update state display job')

    def startMonitors(self):
        """Start automatic update jobs.

        Returns:
            None

        """
        self.autoUpdateAudioMonitors()
        self.autoUpdateVideoMonitors()
        self.autoUpdateTriggerIndicator()
        self.autoUpdateAudioAnalysisMonitors()
        # self.updateStatusDisplay()

    def startSyncProcess(self):
        """Instruct Synchronizer process to begin syncing

        Returns:
            None

        """
        self.sendMessage(self.syncProcess, (Messages.SYNC, None))

    def autoUpdateAudioAnalysisMonitors(self, beginAuto=True):
        """Begin updating audio analysis monitors

        Args:
            beginAuto (bool): Automatically continue updating on a time
                interval? Defaults to True.

        Returns:
            None

        """
        if self.audioTriggerProcess is not None:
            analysisSummary = None
            try:
                while True:
                    analysisSummary = self.audioTriggerProcess.analysisMonitorQueue.get(block=True, timeout=0.01)

                    self.audioAnalysisSummaryHistory.append(analysisSummary)

            except queue.Empty:
                pass

            if analysisSummary is not None:
                # self.log(analysisSummary)
                lag = time.time_ns()/1000000000 - analysisSummary['chunkStartTime']
                if lag > 1.5:
                    self.log("WARNING, high analysis monitoring lag:", lag, 's', 'qsize:', self.audioTriggerProcess.analysisMonitorQueue.qsize())

                # Update bar charts using last received analysis summary

                self.audioAnalysisWidgets['volumeFracAxes'].clear()
                self.audioAnalysisWidgets['lowFracBars'] = []
                self.audioAnalysisWidgets['highFracBars'] = []

                audioDAQChannels = self.getParams('audioDAQChannels')
                for c in range(len(audioDAQChannels)):
                    self.audioAnalysisWidgets['lowFracBars'].append(
                        self.audioAnalysisWidgets['volumeFracAxes'].bar(x=c,      width=0.5, bottom=0, height=analysisSummary['lowFrac'][c],  color='r', align='edge')
                        ) # low frac bar
                    self.audioAnalysisWidgets['highFracBars'].append(
                        self.audioAnalysisWidgets['volumeFracAxes'].bar(x=c+0.5, width=0.5, bottom=0,  height=analysisSummary['highFrac'][c], color='g', align='edge')
                        ) # High frac bar

                self.audioAnalysisWidgets['volumeFracAxes'].axis(xmin=0, xmax=len(audioDAQChannels), ymin=0, ymax=1)

                if len(self.audioAnalysisSummaryHistory) > 0:
                    # Update volume plot

                    # Remove old lines
                    self.audioAnalysisWidgets['volumeTraceAxes'].clear()

                    # Construct a c x n array of volume measurements
                    volumeTrace           = np.array([sum['volume']           for sum in self.audioAnalysisSummaryHistory]).transpose()
                    triggerLowLevelTrace  = np.array([sum['triggerLowLevel']  for sum in self.audioAnalysisSummaryHistory]).transpose()
                    triggerHighLevelTrace = np.array([sum['triggerHighLevel'] for sum in self.audioAnalysisSummaryHistory]).transpose()
                    t                     = np.array([sum['chunkStartTime']   for sum in self.audioAnalysisSummaryHistory]).transpose()
                    numChannels = volumeTrace.shape[0]
    #                tMultiChannel = np.stack([t for k in range(numChannels)], axis=0)
                    yMax = 1.1 * max([volumeTrace.max(), triggerLowLevelTrace.max(), triggerHighLevelTrace.max()])
                    # Plot volume traces for all channels
                    for c in range(numChannels):
                        self.audioAnalysisWidgets['volumeTraceAxes'].plot(t, volumeTrace[c, :], LINE_STYLES[c % len(LINE_STYLES)], linewidth=1)
                    # Plot low level trigger level demarcation
                    self.audioAnalysisWidgets['volumeTraceAxes'].plot(t, triggerLowLevelTrace, 'r-', linewidth=1)
                    # Plot high level trigger level demarcation
                    self.audioAnalysisWidgets['volumeTraceAxes'].plot(t, triggerHighLevelTrace, 'g-', linewidth=1)
                    try:
                        tLow  = t[-1] - (analysisSummary['triggerLowChunks'] -1)*analysisSummary['chunkSize']/analysisSummary['audioFrequency']
                        tHigh = t[-1] - (analysisSummary['triggerHighChunks']-1)*analysisSummary['chunkSize']/analysisSummary['audioFrequency']
                    except TypeError:
                        self.log('weird analysis monitoring error:')
                        traceback.print_exc()
                        self.log('t:', t)
                    # Plot low level time period demarcation
                    self.audioAnalysisWidgets['volumeTraceAxes'].plot([tLow,  tLow],  [0, yMax], 'r-', linewidth=1)
                    # Plot high level time period demarcation
                    self.audioAnalysisWidgets['volumeTraceAxes'].plot([tHigh, tHigh], [0, yMax], 'g-', linewidth=1)
                    self.audioAnalysisWidgets['volumeTraceAxes'].relim()
                    self.audioAnalysisWidgets['volumeTraceAxes'].autoscale_view(True, True, True)
                    self.audioAnalysisWidgets['volumeTraceAxes'].axis(ymin=0, ymax=yMax)
                    self.audioAnalysisWidgets['volumeTraceAxes'].margins(x=0, y=0)
                    self.audioAnalysisWidgets['canvas'].draw()
                    self.audioAnalysisWidgets['canvas'].flush_events()

                # Update trigger state display
                msg = 'No active trigger'
                trigger = analysisSummary['activeTrigger']
                if trigger is not None:
                    now = time.time_ns()/1000000000
                    triggerState = trigger.state(now)
                    if triggerState == 0:
                        pct = 100 * (now - trigger.startTime) / (trigger.endTime - trigger.startTime)
                        msg = "Active trigger {pct:.0f}% complete".format(pct=pct)

                self.audioTriggerStateLabel.config(text=msg)

        if beginAuto:
            self.audioAnalysisMonitorUpdateJob = self.master.after(100, self.autoUpdateAudioAnalysisMonitors)

        self.endLog(inspect.currentframe().f_code.co_name)

    def autoUpdateAudioMonitors(self, beginAuto=True):
        """Begin updating audio monitors

        Args:
            beginAuto (bool): Automatically continue updating on a time
                interval? Defaults to True.

        Returns:
            None

        """
        if self.audioAcquireProcess is not None:
            newAudioData = None
            try:
                for chunkCount in range(100):
                    # Get audio data from monitor queue
                    channels, chunkStartTime, audioData = self.audioAcquireProcess.monitorQueue.get(block=True, timeout=0.001)
                    # Accumulate all new audio chunks together
                    if newAudioData is not None:
                        newAudioData = np.concatenate((newAudioData, audioData), axis=1)
                    else:
                        newAudioData = audioData
                self.log("WARNING! Audio monitor is not getting data fast enough to keep up with stream.")
            except queue.Empty:
                pass

            if newAudioData is not None:
                self.audioMonitor.addAudioData(newAudioData)

        if beginAuto:
            # Schedule another automatic call to autoUpdateAudioMonitors
            self.audioMonitorUpdateJob = self.master.after(100, self.autoUpdateAudioMonitors)

        self.endLog(inspect.currentframe().f_code.co_name)

    def autoUpdateVideoMonitors(self, beginAuto=True):
        """Begin updating video monitors

        Args:
            beginAuto (bool): Automatically continue updating on a time
                interval? Defaults to True.

        Returns:
            None

        """
        if self.videoAcquireProcesses is not None:
            availableImages = {}
            pixelFormats = {}
            for camSerial in self.videoAcquireProcesses:
                try:
                    # Get all available images with the associated pixel format information fromt he monitor queue
                    availableImages[camSerial], metadata = self.videoAcquireProcesses[camSerial].monitorImageReceiver.get(includeMetadata=True)
                    if 'done' in metadata:
                        # Acquire process has indicated it's done sending images for now
                        self.cameraMonitors[camSerial].idle()
                        pixelFormats[camSerial] = None
                        availableImages[camSerial] = None
                    else:
                        self.cameraMonitors[camSerial].active()
                        pixelFormats[camSerial] = metadata['pixelFormat']
                except queue.Empty:
                    pass

            for camSerial in availableImages:
                # Display the most recent available image for each camera
                if availableImages[camSerial] is not None:
                    self.cameraMonitors[camSerial].updateImage(availableImages[camSerial], pixelFormat=pixelFormats[camSerial])

        if beginAuto:
            # Schedule another automatic call to autoUpdateAudioMonitors
            period = int(round(1000.0/(2*self.monitorMasterFrameRate)))
            self.videoMonitorUpdateJob = self.master.after(period, self.autoUpdateVideoMonitors)

    def autoUpdateTriggerIndicator(self, beginAuto=True):
        if beginAuto:
            self.triggerIndicatorUpdateJob = self.master.after(100, self.autoUpdateTriggerIndicator)

    def createCameraAttributeBrowser(self, camSerial):
        """Create a window allowing user to browse camera settings.

        Args:
            camSerial (str): String representing the serial number of a camera
                currently attached to the computer

        Returns:
            None

        """
        main = tk.Toplevel()
        nb = ttk.Notebook(main)
        nb.grid(row=0)
        tooltipLabel = ttk.Label(main, text="temp")
        tooltipLabel.grid(row=1)

        #self.cameraAttributesWidget[camSerial]
        widgets = self.createAttributeBrowserNode(self.cameraAttributes[camSerial], nb, tooltipLabel, 1)

    def createAttributeBrowserNode(self, attributeNode, parent, tooltipLabel, gridRow):
        """Create widgets for one camera attribute node in the browser.

        See createCameraAttributeBrowser

        Args:
            attributeNode (type): Description of parameter `attributeNode`.
            parent (type): Description of parameter `parent`.
            tooltipLabel (type): Description of parameter `tooltipLabel`.
            gridRow (type): Description of parameter `gridRow`.

        Returns:
            dict: Dictionary containing widgets created, organized by type

        """
        frame = ttk.Frame(parent)
        frame.bind("<Enter>", lambda event: tooltipLabel.config(text=attributeNode["tooltip"]))  # Set tooltip rollover callback
        frame.grid(row=gridRow)

        # syncPrint()
        # pp = pprint.PrettyPrinter(indent=1, depth=1)
        # pp.pprint(attributeNode)
        # syncPrint.log()

        widgets = [frame]
        childWidgets = []
        childCategoryHolder = None
        childCategoryWidgets = []

        if attributeNode['type'] == "category":
            children = []
            parent.add(frame, text=attributeNode['displayName'])
            if len(attributeNode['subcategories']) > 0:
                # If this category has subcategories, create a notebook to hold them
                childCategoryHolder = ttk.Notebook(frame)
                childCategoryHolder.grid(row=0)
                widgets.append(childCategoryHolder)
                for subcategoryAttributeNode in attributeNode['subcategories']:
                    childCategoryWidgets.append(self.createAttributeBrowserNode(subcategoryAttributeNode, childCategoryHolder, tooltipLabel, 0))
            for k, childAttributeNode in enumerate(attributeNode['children']):
                childWidgets.append(self.createAttributeBrowserNode(childAttributeNode, frame, tooltipLabel, k+1))
        else:
            if attributeNode['accessMode'] == "RW":
                # Read/write attribute
                accessState = 'normal'
            else:
                # Read only attribute
                accessState = 'readonly'
            if attributeNode['type'] == "command":
                commandButton = ttk.Button(frame, text=attributeNode['displayName'])
                commandButton.grid()
                widgets.append(commandButton)
            elif attributeNode['type'] == "enum":
                enumLabel = ttk.Label(frame, text=attributeNode['displayName'])
                enumLabel.grid(column=0, row=0)
                options = list(attributeNode['options'].values())
                enumSelector = ttk.Combobox(frame, state=accessState, values=options)
                enumSelector.set(attributeNode['value'][1])
                enumSelector.grid(column=1, row=0)
                widgets.append(enumLabel)
                widgets.append(enumSelector)
            else:
                entryLabel = ttk.Label(frame, text=attributeNode['displayName'])
                entryLabel.grid(column=0, row=0)
                entry = ttk.Entry(frame, state=accessState)
                entry.insert(0, attributeNode['value'])
                entry.grid(column=1, row=0)
                widgets.append(entryLabel)
                widgets.append(entry)

        return {'widgets':widgets, 'childWidgets':childWidgets, 'childCategoryWidgets':childCategoryWidgets, 'childCategoryHolder':childCategoryHolder}

    def updateAllCamerasAttributes(self):
        """Update the current camera attributes from some camera??.

        Returns:
            None

        """
        self.cameraAttributes = psu.getAllCamerasAttributes()

    def getQueueSizes(self, verbose=True):
        """Determine the sizes of the various queues used by child processes

        Args:
            verbose (bool): Should the queue sizes be printed to stdout?
                Defaults to True.

        Returns:
            dict: Queue sizes organized in a dictionary like so:
                {
                    videoMonitorQueueSizes={
                        [[cam serial 1]]:[[queue size 1]],
                        [[cam serial 2]]:[[queue size 2]],
                        ...
                        [[cam serial N]]:[[queue size N]],
                    },
                    imageQueueSizes={
                        [[cam serial 1]]:[[queue size 1]],
                        [[cam serial 2]]:[[queue size 2]],
                        ...
                        [[cam serial N]]:[[queue size N]],
                    },
                    audioAnalysisQueueSize=         [[queue size]],
                    audioMonitorQueueSize=          [[queue size]],
                    audioQueueSize=                 [[queue size]],
                    audioAnalysisMonitorQueueSize=  [[queue size]],
                    mergeQueueSize=                 [[queue size]],
                    stdoutQueueSize=                [[queue size]],
                }

        """
        # Initialize an empty dictionary to hold queue sizes
        queueSizes = dict(
            videoMonitorQueueSizes={},
            imageQueueSizes={},
            audioAnalysisQueueSize=None,
            audioMonitorQueueSize=None,
            audioQueueSize=None,
            audioAnalysisMonitorQueueSize=None,
            mergeQueueSize=None,
            stdoutQueueSize=None,
        )

        for camSerial in self.videoAcquireProcesses:
            queueSizes['videoMonitorQueueSizes'][camSerial] = self.videoAcquireProcesses[camSerial].monitorImageReceiver.qsize()
            queueSizes['imageQueueSizes'][camSerial] = self.videoAcquireProcesses[camSerial].imageQueue.qsize()
        if self.audioAcquireProcess is not None:
            queueSizes['audioAnalysisQueueSize'] = self.audioAcquireProcess.audioQueue.qsize()
            queueSizes['audioMonitorQueueSize'] = self.audioAcquireProcess.analysisQueue.qsize()
            queueSizes['audioQueueSize'] = self.audioAcquireProcess.monitorQueue.qsize()
        if self.audioTriggerProcess is not None:
            queueSizes['audioAnalysisMonitorQueueSize'] = self.audioTriggerProcess.analysisMonitorQueue.qsize()
        if self.mergeProcess is not None:
            queueSizes['mergeQueueSize'] = self.mergeProcess.msgQueue.qsize()
        if self.StdoutManager is not None:
            queueSizes['stdoutQueueSize'] = self.StdoutManager.queue.qsize()

        if verbose:
            self.log("Get qsizes...")
            for camSerial in self.videoAcquireProcesses:
                self.log("  videoMonitorQueues[", camSerial, "] size:", queueSizes['videoMonitorQueueSizes'][camSerial])
                self.log("  imageQueues[", camSerial, "] size:", queueSizes['imageQueueSizes'][camSerial])
            if self.audioAcquireProcess is not None:
                self.log("  audioAcquireProcess.audioQueue size:", queueSizes['audioAnalysisQueueSize'])
                self.log("  audioAnalysisQueue size:", queueSizes['audioMonitorQueueSize'])
                self.log("  audioMonitorQueue size:", queueSizes['audioQueueSize'])
            if self.audioTriggerProcess is not None:
                self.log("  audioAnalysisMonitorQueue size:", queueSizes['audioAnalysisMonitorQueueSize'])
            if self.mergeProcess is not None:
                self.log("  mergeMessageQueue size:", queueSizes['mergeQueueSize'])
            if self.StdoutManager is not None:
                self.log("  stdoutQueue size:", queueSizes['stdoutQueueSize'])
            self.log("...get qsizes")
            self.endLog(inspect.currentframe().f_code.co_name)

        return queueSizes

    def getPIDs(self, verbose=True):
        """Determine the PIDs (process IDs) of the various child processes

        Knowing the child PIDs is useful when (for example) you want to find
            them in the Windows Task Manager or Unix top output to monitor their
            resource usage.

        Args:
            verbose (bool): Should the PIDs be printed to stdout?
                Defaults to True.

        Returns:
            dict: Queue sizes organized in a dictionary like so:
                {
                    videoWritePIDs={
                        [[cam serial 1]]:[[PID 1]],
                        [[cam serial 2]]:[[PID 2]],
                        ...
                        [[cam serial N]]:[[PID N]],
                    },
                    videoAcquirePIDs={
                        [[cam serial 1]]:[[PID 1]],
                        [[cam serial 2]]:[[PID 2]],
                        ...
                        [[cam serial N]]:[[PID N]],
                    },
                    audioWritePID=         [[PID]],
                    audioAcquirePID=       [[PID]],
                    audioTriggerPID=       [[PID]],
                    continuousTriggerPID=  [[PID]],
                    syncPID=               [[PID]],
                    mergePID=              [[PID]],
                }

        """
        # Initialize an empty dictionary to hold PIDs
        PIDs = dict(
            videoWritePIDs = {},
            videoAcquirePIDs = {},
            audioWritePID = 'None',
            audioAcquirePID = 'None',
            audioTriggerPID = 'None',
            continuousTriggerPID = 'None',
            syncPID = 'None',
            mergePID = 'None'
        )

        for camSerial in self.videoWriteProcesses:
            if self.videoWriteProcesses[camSerial] is not None:
                PIDs['videoWritePIDs'][camSerial] = self.videoWriteProcesses[camSerial].PID.value
        for camSerial in self.videoAcquireProcesses:
            if self.videoAcquireProcesses[camSerial] is not None:
                PIDs['videoAcquirePIDs'][camSerial] = self.videoAcquireProcesses[camSerial].PID.value
        if self.audioWriteProcess is not None:
            PIDs['audioWritePID'] = self.audioWriteProcess.PID.value
        if self.audioAcquireProcess is not None:
            PIDs['audioAcquirePID'] = self.audioAcquireProcess.PID.value
        if self.audioTriggerProcess is not None:
            PIDs['audioTriggerPID'] = self.audioTriggerProcess.PID.value
        if self.continuousTriggerProcess is not None:
            PIDs['continuousTriggerPID'] = self.continuousTriggerProcess.PID.value
        if self.syncProcess is not None:
            PIDs['syncPID'] = self.syncProcess.PID.value
        if self.mergeProcess is not None:
            PIDs['mergePID'] = self.mergeProcess.PID.value

        if verbose:
            self.log("PIDs...")
            self.log("main thread:", os.getpid())
            for camSerial in self.videoWriteProcesses:
                self.log("  videoWritePID["+camSerial+"]:", PIDs['videoWritePIDs'])[camSerial]
            for camSerial in self.videoAcquireProcesses:
                self.log("  videoAcquirePID["+camSerial+"]:", PIDs['videoAcquirePIDs'])[camSerial]
            self.log("  audioWritePID:", PIDs['audioWritePID'])
            self.log("  audioAcquirePID:", PIDs['audioAcquirePID'])
            self.log("  audioTriggerPID:", PIDs['audioTriggerPID'])
            self.log("  continuousTriggerPID:", PIDs['continuousTriggerPID'])
            self.log("  syncPID:", PIDs['syncPID'])
            self.log("  mergePID:", PIDs['mergePID'])
            self.log("...PIDs:")

        self.endLog(inspect.currentframe().f_code.co_name)
        return PIDs

    def checkStates(self, verbose=True):
        """Determine the current published states of the various child processes

        See StateMachineProcesses.States for a complete list of states.

        Args:
            verbose (bool): Should the states be printed to stdout?
                Defaults to True.

        Returns:
            dict: Child process states organized in a dictionary like so:
                {
                    videoWriteStates={
                        [[cam serial 1]]:[[state 1]],
                        [[cam serial 2]]:[[state 2]],
                        ...
                        [[cam serial N]]:[[state N]],
                    },
                    videoAcquireStates={
                        [[cam serial 1]]:[[state 1]],
                        [[cam serial 2]]:[[state 2]],
                        ...
                        [[cam serial N]]:[[state N]],
                    },
                    audioWriteState=         [[state]],
                    audioAcquireState=       [[state]],
                    audioTriggerState=       [[state]],
                    continuousTriggerState=  [[state]],
                    syncState=               [[state]],
                    mergeState=              [[state]],
                }

        """
        states = dict(
            videoWriteStates = {},
            videoAcquireStates = {},
            audioWriteState = None,
            audioAcquireState = None,
            syncState = None,
            mergeState = None,
            audioTriggerState = None,
            continuousTriggerState = None
        )
        stateNames = dict(
            videoWriteStates = {},
            videoAcquireStates = {},
            audioWriteState = 'None',
            audioAcquireState = 'None',
            syncState = 'None',
            mergeState = 'None',
            audioTriggerState = 'None',
            continuousTriggerState = 'None'
        )

        for camSerial in self.videoWriteProcesses:
            if self.videoWriteProcesses[camSerial] is not None:
                states['videoWriteStates'][camSerial] = self.videoWriteProcesses[camSerial].publishedStateVar.value
                stateNames['videoWriteStates'][camSerial] = VideoWriter.stateList[states['videoWriteStates'][camSerial]]
        for camSerial in self.videoAcquireProcesses:
            if self.videoAcquireProcesses[camSerial] is not None:
                states['videoAcquireStates'][camSerial] = self.videoAcquireProcesses[camSerial].publishedStateVar.value
                stateNames['videoAcquireStates'][camSerial] = VideoAcquirer.stateList[states['videoAcquireStates'][camSerial]]
        if self.audioWriteProcess is not None:
            states['audioWriteState'] = self.audioWriteProcess.publishedStateVar.value
            stateNames['audioWriteState'] = AudioWriter.stateList[states['audioWriteState']]
        if self.audioAcquireProcess is not None:
            states['audioAcquireState'] = self.audioAcquireProcess.publishedStateVar.value
            stateNames['audioAcquireState'] = AudioAcquirer.stateList[states['audioAcquireState']]
        if self.syncProcess is not None:
            states['syncState'] = self.syncProcess.publishedStateVar.value
            stateNames['syncState'] = Synchronizer.stateList[states['syncState']]
        if self.mergeProcess is not None:
            states['mergeState'] = self.mergeProcess.publishedStateVar.value
            stateNames['mergeState'] = AVMerger.stateList[states['mergeState']]
        if self.audioTriggerProcess is not None:
            states['audioTriggerState'] = self.audioTriggerProcess.publishedStateVar.value
            stateNames['audioTriggerState'] = AudioTriggerer.stateList[states['audioTriggerState']]
        if self.continuousTriggerProcess is not None:
            states['continuousTriggerState'] = self.continuousTriggerProcess.publishedStateVar.value
            stateNames['continuousTriggerState'] = ContinuousTriggerer.stateList[states['continuousTriggerState']]

        if verbose:
            self.log("Check states...")
            for camSerial in stateNames['videoWriteStates']:
                self.log("videoWriteStates[", camSerial, "]:", stateNames['videoWriteStates'][camSerial])
            for camSerial in stateNames['videoAcquireStates']:
                self.log("videoAcquireStates[", camSerial, "]:", stateNames['videoAcquireStates'][camSerial])
            self.log("audioWriteState:", stateNames['audioWriteState'])
            self.log("audioAcquireState:", stateNames['audioAcquireState'])
            self.log("syncState:", stateNames['syncState'])
            self.log("mergeState:", stateNames['mergeState'])
            self.log("audioTriggerState:", stateNames['audioTriggerState'])
            self.log("continuousTriggerState:", stateNames['continuousTriggerState'])
            self.log("...check states")
            self.endLog(inspect.currentframe().f_code.co_name)

        return states, stateNames

    def checkInfo(self, verbose=True):
        # Check supplementary status information variable shared by processes
        # Note: All processes have this variable, but only the writers currently publish any info.
        """Gather supplementary published info from the various child processes

        Not all processes currently publish useful info, so only some are
        currently gathered

        Args:
            verbose (bool): Should the info be printed to stdout?
                Defaults to True.

        Returns:
            dict: Child published info organized in a dictionary like so:
                {
                    videoWriteInfo={
                        [[cam serial 1]]:[[info 1]],
                        [[cam serial 2]]:[[info 2]],
                        ...
                        [[cam serial N]]:[[info N]],
                    },
                    audioWriteInfo=[[info]],
                }

        """

        info = dict(
            videoWriteInfo = {},
            audioWriteInfo = 'None'
        )
        for camSerial in self.videoWriteProcesses:
            if self.videoWriteProcesses[camSerial] is not None:
                info['videoWriteInfo'][camSerial] = getSharedString(self.videoWriteProcesses[camSerial].publishedInfoVar)
        if self.audioWriteProcess is not None:
            info['audioWriteInfo'] = getSharedString(self.audioWriteProcess.publishedInfoVar)

        if verbose:
            self.log("Check process info...")
            for camSerial in info['videoWriteInfo']:
                self.log("videoWriteInfo[", camSerial, "]:", info['videoWriteInfo'][camSerial])
            self.log("audioWriteInfo:", info['audioWriteInfo'])
            self.log("...check process info")
            self.endLog(inspect.currentframe().f_code.co_name)

        return info

    def updateStatusDisplay(self, verbose=False, interval=1000, repeat=True):
        """Update the Status display widget with the current status of the app.

        Args:
            verbose (bool): Should the gathered information be printed to
                stdout? Defaults to False.
            interval (int): Interval in ms that this function will be repeated
                if the repeat argument is True Defaults to 1000.
            repeat (bool): Should this function be automatically repeated?
                Defaults to True.

        Returns:
            None

        """

        # Gather all status information
        states, stateNames = self.checkStates(verbose=verbose)
        PIDs = self.getPIDs(verbose=verbose)
        queueSizes = self.getQueueSizes(verbose=verbose)
        info = self.checkInfo(verbose=verbose)

        # Check and update the current app "meta state" (self.metaState)
        self.determineApplicationMetaState(states=states)

        # Make widget controls react to current meta state (mostly by enabling
        #   or disabling)
        self.reactToAcquisitionState()

        # Update meta state in Status frame label as a quick hint to the user
        if self.metaState is not None:
            self.statusFrame.setText('Status: {metaState}'.format(metaState=self.metaState))
        else:
            self.statusFrame.setText('Status')

        # \/\/\/ formatting scheme is not implemented yet. But it's a good idea.
        # Format: Each line is a list of text to include in that line, separated
        #   into chunks based on what tag to apply The last element in the list
        #   is a list of tag names, one for each chunk in the line.

        lines = []
        lines.append(
                    'Overall state: {metaState}'.format(metaState=self.metaState)
        )
        lines.append(
                    'VideoAcquirers' #[['VideoAcquirers:'], ['normal']]
        )
        for camSerial in self.videoWriteProcesses:
            if self.videoWriteProcesses[camSerial] is not None:
                lines.extend([
                    '   {camSerial} ({PID}):\t{state}'.format(camSerial=camSerial, PID=PIDs['videoAcquirePIDs'][camSerial], state=stateNames['videoAcquireStates'][camSerial])
                ])
        lines.append(
                    'VideoWriters:'
        )
        for camSerial in self.videoWriteProcesses:
            if self.videoAcquireProcesses[camSerial] is not None:
                lines.extend([
                    '   {camSerial} ({PID}):\t{state}'.format(camSerial=camSerial, PID=PIDs['videoWritePIDs'][camSerial], state=stateNames['videoWriteStates'][camSerial]),
                    '       Image Queue: {qsize}'.format(qsize=queueSizes['videoMonitorQueueSizes'][camSerial]),
                    '       Monitor Queue: {qsize}'.format(qsize=queueSizes['imageQueueSizes'][camSerial]),
                    '       Info: {info}'.format(info=info['videoWriteInfo'][camSerial])
                ])
        lines.extend([
                    'AudioAcquirer ({PID}):\t{state}'.format(PID=PIDs['audioAcquirePID'], state=stateNames['audioAcquireState']),
                    '   Audio Queue: {qsize}'.format(qsize=queueSizes['audioQueueSize']),
                    '   Analysis Queue: {qsize}'.format(qsize=queueSizes['audioAnalysisQueueSize']),
                    '   Monitor Queue: {qsize}'.format(qsize=queueSizes['audioMonitorQueueSize']),
                    '   Analysis Monitor Queue: {qsize}'.format(qsize=queueSizes['audioAnalysisMonitorQueueSize']),
                    'AudioWriter ({PID}):\t{state}'.format(PID=PIDs['audioWritePID'], state=stateNames['audioWriteState']),
                    '   Info: {info}'.format(info=info['audioWriteInfo']),
                    'Synchronizer ({PID}):\t{state}'.format(PID=PIDs['syncPID'], state=stateNames['syncState']),
                    'ContinuousTrigger ({PID}):\t{state}'.format(PID=PIDs['continuousTriggerPID'], state=stateNames['continuousTriggerState']),
                    'AudioTriggerer ({PID}):\t{state}'.format(PID=PIDs['audioTriggerPID'], state=stateNames['audioTriggerState']),
                    'AVMerger ({PID}):\t{state}'.format(PID=PIDs['mergePID'], state=stateNames['mergeState']),
                    '   Merge Queue: {qsize}'.format(qsize=queueSizes['mergeQueueSize'])
        ])

        self.childStatusText.delete('1.0', tk.END)
        self.childStatusText['height'] = len(lines)
        self.childStatusText.insert(tk.END, '\n'.join(lines))

        if repeat:
            self.updateStatusDisplayJob = self.master.after(interval, self.updateStatusDisplay)

    def getProcesses(self, audio=True, video=True, acquirers=True, writers=True, auxiliary=True):
        """Gather a list of processes of the selected types.

        Args:
            audio (bool): Include "audio" type processes. Defaults to True.
            video (bool): Include "video" type processes. Defaults to True.
            acquirers (bool): Include "acquirers" type processes. Defaults to True.
            writers (bool): Include "writers" type processes. Defaults to True.
            auxiliary (bool): Include "auxiliary" type processes. Defaults to True.

        Returns:
            list: List of references to selected child process

        """
        processes = []
        if video and writers:
            for camSerial in self.videoWriteProcesses:
                processes.append(self.videoWriteProcesses[camSerial])
        if video and acquirers:
            for camSerial in self.videoAcquireProcesses:
                processes.append(self.videoAcquireProcesses[camSerial])
        if audio and writers:
            processes.append(self.audioWriteProcess)
        if audio and acquirers:
            processes.append(self.audioAcquireProcess)
        if auxiliary:
            processes.extend([
                self.audioTriggerProcess,
                self.continuousTriggerProcess,
                self.syncProcess,
                self.mergeProcess
                ])
        return processes

    def determineApplicationMetaState(self, states=None):
        """Determine, record, and return an overall application "meta state".

        The meta state is a representation of the overal gestalt of what the
            child processes are doing.

        List of metaStates
            'initialized'   All processes running (inc at least one acquire),
                                and acquisition/sync processes but in
                                initializing or similar state
            'acquiring'     All processes running (inc at least one acquire),
                                and all in an acquire-ey state
            'halted'        All processes running but in stopped state
            'dead'          No processes running
            'error'         At least one process in error state
            'indeterminate' Some other combination of states

        Args:
            states (dict): Formatted dictionary containing information about
                child process states. See checkStates function for details about
                the format of the dictionary. If states is None, the checkStates
                function will be called to gather the states. Defaults to None.

        Returns:
            str: Meta state of the application

        """
        # If not provided, check the states of the child processes
        if states is None:
            states, _ = self.checkStates()

        # Count up how many processes of various types are in various types of
        #   states
        numProcesses = 0
        numInitializing = 0
        numAcquirersRunning = 0
        numAcquiresAcquiring = 0
        numAuxiliariesRunning = 0
        numWritersRunning = 0
        numStopped = 0
        numError = 0
        stateList = []
        isAcquirer = []
        isWriter = []
        isAux = []
        for camSerial in states['videoWriteStates']:
            state = states['videoWriteStates'][camSerial]
            stateList.append(state); isAcquirer.append(False); isWriter.append(True);  isAux.append(False)
        for camSerial in states['videoAcquireStates']:
            state = states['videoAcquireStates'][camSerial]
            stateList.append(state); isAcquirer.append(True);  isWriter.append(False); isAux.append(False)

        stateList.append(states['audioAcquireState']);      isAcquirer.append(True);  isWriter.append(False); isAux.append(False)
        stateList.append(states['audioWriteState']);        isAcquirer.append(False); isWriter.append(True);  isAux.append(False)
        stateList.append(states['syncState']);              isAcquirer.append(False); isWriter.append(False); isAux.append(True)
        stateList.append(states['mergeState']);             isAcquirer.append(False); isWriter.append(False); isAux.append(True)
        stateList.append(states['audioTriggerState']);      isAcquirer.append(False); isWriter.append(False); isAux.append(True)
        stateList.append(states['continuousTriggerState']); isAcquirer.append(False); isWriter.append(False); isAux.append(True)

        # Define human readable state names
        allStates = {States.UNKNOWN:'UNKNOWN', States.STOPPED:'STOPPED',
            States.INITIALIZING:'INITIALIZING', States.READY:'READY',
            States.STOPPING:'STOPPING', States.ERROR:'ERROR',
            States.EXITING:'EXITING', States.DEAD:'DEAD',
            States.IGNORING:'IGNORING', States.MERGING:'MERGING',
            States.SYNCHRONIZING:'SYNCHRONIZING', States.WAITING:'WAITING',
            States.ANALYZING:'ANALYZING', States.AUDIOINIT:'AUDIOINIT',
            States.WRITING:'WRITING', States.BUFFERING:'BUFFERING',
            States.ACQUIRING:'ACQUIRING', States.VIDEOINIT:'VIDEOINIT',
            States.TRIGGERING:'TRIGGERING', None:'None'}

        # Tally up the various types of states processes are in
        for state, acquirer, aux, writer in zip(stateList, isAcquirer, isAux, isWriter):
            if state is not None and state != States.EXITING:
                numProcesses += 1
                if acquirer:
                    numAcquirersRunning += 1
                if aux:
                    numAuxiliariesRunning += 1
                if writer:
                    numWritersRunning += 1
            if state in [States.INITIALIZING, States.WAITING, States.READY]:
                numInitializing += 1
            if state in [States.ACQUIRING]:
                numAcquiresAcquiring += 1
            if state in [States.STOPPED, States.STOPPING]:
                numStopped += 1
            if state == States.ERROR:
                numError += 1

        # Determine an overall application meta state by examining the tallies
        #   of the child states
        if numProcesses == 0:
            self.metaState = 'dead'
        elif numError > 0:
            self.metaState = 'error'
        elif numInitializing == (numAcquirersRunning + numAuxiliariesRunning):
            self.metaState = 'initialized'
        elif numAcquirersRunning > 0 and numAcquiresAcquiring == numAcquirersRunning:
            self.metaState = 'acquiring'
        elif numStopped == numProcesses:
            self.metaState = 'halted'
        else:
            self.metaState = 'indeterminate'

        return self.metaState

    def reactToAcquisitionState(self):
        """Adjust GUI to reflect the meta state of the application.

        Mostly, enable/disable buttons to prevent users from taking actions that
            are incompatible with the application meta state.

        Returns:
            None

        """
        if self.metaState is None:
            self.selectAcquisitionHardwareButton.config(state=tk.DISABLED)
            self.initializeAcquisitionButton.config(state=tk.DISABLED)
            self.haltAcquisitionButton.config(state=tk.DISABLED)
            self.restartAcquisitionButton.config(state=tk.DISABLED)
            self.shutDownAcquisitionButton.config(state=tk.DISABLED)
            for mode in self.triggerModeRadioButtons:
                self.triggerModeRadioButtons[mode].config(state=tk.DISABLED)
            self.acquisitionParametersFrame.disable()
        elif self.metaState == 'initialized':
            self.selectAcquisitionHardwareButton.config(state=tk.DISABLED)
            self.initializeAcquisitionButton.config(state=tk.DISABLED)
            self.haltAcquisitionButton.config(state=tk.NORMAL)
            self.restartAcquisitionButton.config(state=tk.NORMAL)
            self.shutDownAcquisitionButton.config(state=tk.NORMAL)
            for mode in self.triggerModeRadioButtons:
                self.triggerModeRadioButtons[mode].config(state=tk.DISABLED)
            self.acquisitionParametersFrame.disable()
        elif self.metaState == 'acquiring':
            # Child processes are either in initialized state, or are actively acquiring
            self.selectAcquisitionHardwareButton.config(state=tk.DISABLED)
            self.initializeAcquisitionButton.config(state=tk.DISABLED)
            self.haltAcquisitionButton.config(state=tk.NORMAL)
            self.restartAcquisitionButton.config(state=tk.NORMAL)
            self.shutDownAcquisitionButton.config(state=tk.NORMAL)
            for mode in self.triggerModeRadioButtons:
                self.triggerModeRadioButtons[mode].config(state=tk.NORMAL)
            self.acquisitionParametersFrame.disable()
        elif self.metaState == 'halted':
            # Child processes are running, but in stopped state
            self.selectAcquisitionHardwareButton.config(state=tk.DISABLED)
            self.initializeAcquisitionButton.config(state=tk.NORMAL)
            self.haltAcquisitionButton.config(state=tk.NORMAL)
            self.restartAcquisitionButton.config(state=tk.DISABLED)
            self.shutDownAcquisitionButton.config(state=tk.NORMAL)
            for mode in self.triggerModeRadioButtons:
                self.triggerModeRadioButtons[mode].config(state=tk.NORMAL)
            self.acquisitionParametersFrame.enable()
        elif self.metaState == 'dead':
            # Child processes are not running, or do not exist
            self.selectAcquisitionHardwareButton.config(state=tk.NORMAL)
            self.initializeAcquisitionButton.config(state=tk.NORMAL)
            self.haltAcquisitionButton.config(state=tk.DISABLED)
            self.restartAcquisitionButton.config(state=tk.DISABLED)
            self.shutDownAcquisitionButton.config(state=tk.DISABLED)
            for mode in self.triggerModeRadioButtons:
                self.triggerModeRadioButtons[mode].config(state=tk.NORMAL)
            self.acquisitionParametersFrame.enable()
        elif self.metaState == 'error':
            self.selectAcquisitionHardwareButton.config(state=tk.DISABLED)
            self.initializeAcquisitionButton.config(state=tk.DISABLED)
            self.haltAcquisitionButton.config(state=tk.NORMAL)
            self.restartAcquisitionButton.config(state=tk.NORMAL)
            self.shutDownAcquisitionButton.config(state=tk.NORMAL)
            for mode in self.triggerModeRadioButtons:
                self.triggerModeRadioButtons[mode].config(state=tk.DISABLED)
            self.acquisitionParametersFrame.disable()
        elif self.metaState == 'indeterminate':
            self.selectAcquisitionHardwareButton.config(state=tk.DISABLED)
            self.initializeAcquisitionButton.config(state=tk.DISABLED)
            self.haltAcquisitionButton.config(state=tk.DISABLED)
            self.restartAcquisitionButton.config(state=tk.DISABLED)
            self.shutDownAcquisitionButton.config(state=tk.NORMAL)
            for mode in self.triggerModeRadioButtons:
                self.triggerModeRadioButtons[mode].config(state=tk.DISABLED)
            self.acquisitionParametersFrame.disable()
        else:
            self.selectAcquisitionHardwareButton.config(state=tk.NORMAL)
            self.initializeAcquisitionButton.config(state=tk.NORMAL)
            self.haltAcquisitionButton.config(state=tk.NORMAL)
            self.restartAcquisitionButton.config(state=tk.NORMAL)
            self.shutDownAcquisitionButton.config(state=tk.NORMAL)
            for mode in self.triggerModeRadioButtons:
                self.triggerModeRadioButtons[mode].config(state=tk.NORMAL)
            self.acquisitionParametersFrame.enable()

    def initializeAcquisition(self):
        """Initialize child processes to prepare for acquisition.

        If the child processes are already in the "halted" meta state, we can
            simply send them a start message. Otherwise, we have to recreate
            them.
        Normally, the user should be prevented from calling this method unless
            the application meta state is "halted" or "stopped"

        Returns:
            None

        """
        if self.metaState != 'halted':
            # If state is halted, we can just reinit the processes
            self.setupInputMonitoringWidgets()
            self.createChildProcesses()
        self.initializeChildProcesses()
        # Schedule button update after 100 ms to give child processes a chance to react
        # self.master.after(100, self.updateAcquisitionButton)
    def haltAcquisition(self):
        """Halt child processes.

        Child processes will keep running, but in the "STOPPED" state.

        Returns:
            None

        """
        self.haltChildProcesses()
    def restartAcquisition(self):
        """Restart child processes.

        Returns:
            None

        """
        self.haltChildProcesses()
        self.setupInputMonitoringWidgets()
        self.initializeChildProcesses()
    def shutDownAcquisition(self):
        """Shut down child processes.

        This actually terminates the processes. To acquire, the user will have
        to recreate the processes.

        Returns:
            None

        """
        self.haltChildProcesses()
        self.destroyChildProcesses()

    def writeButtonClickHandler(self):
        """Send a manual trigger to write processes to trigger recording.

        Normally will only be called in "Manual" trigger mode.

        Returns:
            None

        """
        self.sendWriteTrigger()

    def continuousTriggerStartButtonClick(self):
        """Send a message to ContinuousTriggerer process to begin triggering.

        Normally will only be called in "Continuous" trigger mode.

        Returns:
            None

        """
        success = self.sendMessage(self.continuousTriggerProcess, (Messages.START, None))
        if success:
            self.log("Sent start signal to continuous trigger process")
            self.endLog(inspect.currentframe().f_code.co_name)
        else:
            showwarning(title="No continuous trigger process available", message="Continuous triggering process does not appear to be available. Try starting up acquisition first")

    def continuousTriggerStopButtonClick(self):
        """Send a message to ContinuousTriggerer process to stop triggering.

        Normally will only be called in "Continuous" trigger mode.

        Returns:
            None

        """
        success = self.sendMessage(self.continuousTriggerProcess, (Messages.STOP, None))
        if success:
            self.log("Sent stop signal to continuous trigger process")
            self.endLog(inspect.currentframe().f_code.co_name)
        else:
            showwarning(title="No continuous trigger process available", message="Continuous triggering process does not appear to be available. Try starting up acquisition first")

    def saveSettings(self, *args, path=None):
        """Save all application settings/parameters to a settings file.

        Args:
            *args (any): Dummy variable to hold unused event data
            path (str): A string representing the path to save the settings file
                to. If None, a GUI dialog will prompt the user to provide one.
                Defaults to None.

        Returns:
            None

        """
        params = self.getParams()
        # datetime.time objects are not serializable, so we have to extract the time
        params['scheduleStart'] = timeToSerializable(params['scheduleStart'])
        params['scheduleStop'] = timeToSerializable(params['scheduleStop'])
        if path is None:
            path = asksaveasfilename(
                title = "Choose a filename to save current settings to.",
                confirmoverwrite = True,
                defaultextension = 'pvs',
                initialdir = '.'
            )
        if path is not None and len(path) > 0:
            with open(path, 'w') as f:
                f.write(json.dumps(params))

    def loadSettings(self, *args, path=None):
        """Load application settings/parameters from a saved settings file.

        Args:
            *args (any): Dummy variable to hold unused event data
            path (str): A string representing the path to load the settings file
                from. If None, a GUI dialog will prompt the user to provide one.
                Defaults to None.

        Returns:
            None

        """
        if path is None:
            path = askopenfilename(
                title = "Choose a settings file to load.",
                defaultextension = 'pvs',
                initialdir = '.'
            )
        if path is not None and len(path) > 0:
            with open(path, 'r') as f:
                params = json.loads(f.read())
            params['scheduleStart'] = serializableToTime(params['scheduleStart'])
            params['scheduleStop'] = serializableToTime(params['scheduleStop'])
            self.log("Loaded settings:")
            self.log(params)
            self.setParams(**params)
            self.updateAcquisitionHardwareDisplay()
        self.endLog(inspect.currentframe().f_code.co_name)

    def setAudioWriteEnable(self, newAudioWriteEnable, *args, updateGUI=True):
        """Send a message to audio writer process to enable/disable file writing

        Args:
            newAudioWriteEnable (bool): Enable or disable writing? True=enable,
                False=disable.
            *args (any): Dummy variable to hold unused event data
            updateGUI (bool): Should this method update the audio write enable
                GUI checkbox? Set to False when called by the checkbox itself to
                prevent an infinite event loop where the checkbox triggers this
                method, and the method triggers the checkbox.

        Returns:
            None

        """
        self.audioWriteEnable.set(newAudioWriteEnable)
        if updateGUI:
            # Update text field
            self.audioMonitors.setWriteEnable(newAudioWriteEnable)
        # Notify AudioWriter child process of new write enable state
        self.sendMessage(self.audioWriteProcess, (Messages.SETPARAMS, dict(enableWrite=newAudioWriteEnable)))

    def setVideoWriteEnable(self, newVideoWriteEnables, *args, updateGUI=True):
        """Send messages to video writer processes to enable/disable file writing

        Args:
            newVideoWriteEnables (dict): A dictionary with camera serials as
                keys, and booleans indicating whether to enable or disable
                writing as values. True=enable, False=disable.
            *args (any): Dummy variable to hold unused event data
            updateGUI (bool): Should this method update the video write enable
                GUI checkbox? Set to False when called by the checkbox itself to
                prevent an infinite event loop where the checkbox triggers this
                method, and the method triggers the checkbox.

        Returns:
            None

        """
        # Update stored parameter
        self.videoWriteEnable.set(newVideoWriteEnables)
        # Loop over available cameras
        for camSerial in self.cameraMonitors:
            # Check if we're changing enable state for this particular camera
            if camSerial in newVideoWriteEnables:
                newVideoWriteEnable = newVideoWriteEnables[camSerial]
                if updateGUI:
                    # Update text field
                    self.cameraMonitors[camSerial].setWriteEnable(newVideoWriteEnable)
                if camSerial in self.videoWriteProcesses:
                    # Notify VideoWriter child process of new write enable state
                    self.sendMessage(self.videoWriteProcesses[camSerial], (Messages.SETPARAMS, dict(enableWrite=newVideoWriteEnable)))

    def setVideoBaseFileNames(self, newVideoBaseFileNames, *args, updateGUI=True):
        """Send messages to video writer processes to change base filenames

        Args:
            newVideoBaseFileNames (dict): A dictionary with camera serials as
                keys, and strings indicating new base filenames to use as values
            *args (any): Dummy variable to hold unused event data
            updateGUI (bool): Should this method update the video base filename
                GUI textboxes? Set to False when called by the textbox itself to
                prevent an infinite event loop where the textbox triggers this
                method, and the method triggers the textbox.

        Returns:
            None

        """
        self.videoBaseFileNames.set(newVideoBaseFileNames)
        for camSerial in self.cameraMonitors:
            if camSerial in newVideoBaseFileNames:
                newVideoBaseFileName = newVideoBaseFileNames[camSerial]
                if updateGUI:
                    # Update text field
                    self.cameraMonitors[camSerial].fileWidget.setBaseFileName(newVideoBaseFileName)
                if camSerial in self.videoWriteProcesses:
                    # Notify VideoWriter child process of new write base filename
                    self.sendMessage(self.videoWriteProcesses[camSerial], (Messages.SETPARAMS, dict(videoBaseFileName=newVideoBaseFileName)))
    def setVideoDirectories(self, newVideoDirectories, *args, updateGUI=True):
        """Send messages to video writer processes to change video directories

        Args:
            newVideoDirectories (dict): A dictionary with camera serials as
                keys, and strings indicating new video directories to save in
            *args (any): Dummy variable to hold unused event data
            updateGUI (bool): Should this method update the video directory
                GUI textboxes? Set to False when called by the textbox itself to
                prevent an infinite event loop where the textbox triggers this
                method, and the method triggers the textbox.

        Returns:
            None

        """
        self.videoDirectories.set(newVideoDirectories)
        for camSerial in self.cameraMonitors:
            if camSerial in newVideoDirectories:
                newVideoDirectory = newVideoDirectories[camSerial]
                if updateGUI:
                    # Update text field
                    self.cameraMonitors[camSerial].fileWidget.setDirectory(newVideoDirectory)
                if camSerial in self.videoWriteProcesses and (len(newVideoDirectory) == 0 or os.path.isdir(newVideoDirectory)):
                    # Notify VideoWriter child process of new write directory
                    self.sendMessage(self.videoWriteProcesses[camSerial], (Messages.SETPARAMS, dict(videoDirectory=newVideoDirectory)))
    def setAudioBaseFileName(self, newAudioBaseFileName, *args, updateGUI=True):
        """Send message to audio writer process to change base filenames

        Args:
            newAudioBaseFileName (str): A string indicating a new base filename
                to use to save audio files
            *args (any): Dummy variable to hold unused event data
            updateGUI (bool): Should this method update the base filename GUI
                textbox? Set to False when called by the textbox itself to
                prevent an infinite event loop where the textbox triggers this
                method, and the method triggers the textbox.

        Returns:
            None

        """
        self.audioBaseFileName.set(newAudioBaseFileName)
        if updateGUI and self.audioMonitor is not None:
            # Update text field
            self.audioMonitor.fileWidget.setBaseFileName(newAudioBaseFileName)
        # Notify AudioWriter child process of new write base filename
        self.sendMessage(self.audioWriteProcess, (Messages.SETPARAMS, dict(audioBaseFileName=newAudioBaseFileName)))
    def setAudioDirectory(self, newAudioDirectory, *args, updateGUI=True):
        """Send message to audio writer process to change audio directory

        Args:
            newAudioDirectory (str): A string indicating a new directory to use
                to save audio files
            *args (any): Dummy variable to hold unused event data
            updateGUI (bool): Should this method update the audio directory GUI
                textbox? Set to False when called by the textbox itself to
                prevent an infinite event loop where the textbox triggers this
                method, and the method triggers the textbox.

        Returns:
            None

        """
        self.audioDirectory.set(newAudioDirectory)
        if updateGUI and self.audioMonitor is not None:
            # Update text field
            self.audioMonitor.fileWidget.setDirectory(newAudioDirectory)
        if len(newAudioDirectory) == 0 or os.path.isdir(newAudioDirectory):
            # Notify AudioWriter child process of new write directory
            self.sendMessage(self.audioWriteProcess, (Messages.SETPARAMS, dict(audioDirectory=newAudioDirectory)))
    def setMergeBaseFileName(self, newMergeBaseFileName, *args, updateGUI=True):
        """Send message to AVMerger process to change merge base filename

        Args:
            newMergeBaseFileName (str): A string indicating a new base filename
                to use to create merged audio/video files
            *args (any): Dummy variable to hold unused event data
            updateGUI (bool): Should this method update the merge base filename
                GUI textbox? Set to False when called by the textbox itself to
                prevent an infinite event loop where the textbox triggers this
                method, and the method triggers the textbox.

        Returns:
            None

        """
        self.mergeBaseFileName.set(newMergeBaseFileName)
        if updateGUI and self.mergeFileWidget is not None:
            # Update text field
            self.mergeFileWidget.setBaseFileName(newMergeBaseFileName)
        # Notify AVMerger child process of new write base filename
        self.sendMessage(self.mergeProcess, (Messages.SETPARAMS, dict(mergeBaseFileName=newMergeBaseFileName)))
    def setMergeDirectory(self, newMergeDirectory, *args, updateGUI=True):
        """Send message to AVMerger process to change merge directory

        Args:
            newMergeDirectory (str): A string indicating a new directory to use
                to merge audio/video files into
            *args (any): Dummy variable to hold unused event data
            updateGUI (bool): Should this method update the merge directory
                textbox? Set to False when called by the textbox itself to
                prevent an infinite event loop where the textbox triggers this
                method, and the method triggers the textbox.

        Returns:
            None

        """
        self.mergeDirectory.set(newMergeDirectory)
        if updateGUI and self.mergeFileWidget is not None:
            # Update text field
            self.mergeFileWidget.setDirectory(newMergeDirectory)
        if len(newMergeDirectory) == 0 or os.path.isdir(newMergeDirectory):
            # Notify AVMerger child process of new write directory
            self.sendMessage(self.mergeProcess, (Messages.SETPARAMS, dict(directory=newMergeDirectory)))

    def setParams(self, ignoreErrors=True, **params):
        """Set one or more GUI parameters.

        The PyVAQ.paramInfo instance attribute is a dictionary containing the
            getters and setters for all GUI parameters; essentially all the
            user-alterable settings. The setParams function is a convenient way
            to use one or more setters to set one or more GUI parameters.

        See also: getParams

        Args:
            ignoreErrors (bool): Continue through dictionary of parameters to
                set even if an unexpected error is encountered when setting one
                of them? Defaults to True.
            **params (dict): Keyword pairs of parameter names and values in the
                form of a dictionary to set.

        Returns:
            None

        """
        for paramName in params:
            try:
                self.paramInfo[paramName]["set"](params[paramName])
            except (NotImplementedError, AttributeError) as e:
                print('Unable to set param: {paramName}'.format(paramName=paramName))
                print('\tReason: {reason}'.format(reason=str(e)))
                # traceback.print_exc()
            except KeyError as e:
                print('Unable to set param: {paramName}'.format(paramName=paramName))
                print('\tReason: {reason}'.format(reason='This parameter is not valid - it will be ignored.'))
            except Exception as e:
                if ignoreErrors:
                    # Print error message, then move on...
                    traceback.print_exc()
                else:
                    # Stop the train, we've got an error!
                    raise e

    def getParams(self, *paramNames, mapping=False):
        """Get the current value of one or more GUI parameters.

        The PyVAQ.paramInfo instance attribute is a dictionary containing the
            getters and setters for all GUI parameters; essentially all the
            user-alterable settings. The getParams function is a convenient way
            to use one or more getters to get one or more GUI parameter values.

        See also: setParams

        Args:
            mappping (bool): Return a dictionary of parameters, even if only one
                parameter is requested. If False, requests for multiple
                parameters will still return a dictionary, but requests for
                single parameters will return just the plain value, not wrapped
                in a dictionary, for convenience. Defaults to False.
            *paramNames (list): A list of one or more parameter names,
                corresponding to one or more keys of the PyVAQ.paramInfo
                dictionary.

        Returns:
            dict: A dictionary composed of pairs of parameter names and values,
                with parameter names corresponding to the requested parameters.

        """
        if len(paramNames) == 1 and (not mapping):
            # If just one param was requested, just return the value, not a dictionary
            return self.paramInfo[paramNames[0]]["get"]()
        else:
            # If multiple params are requested, return them in a paramName:value dictionary
            if len(paramNames) == 0:
                # Get all params
                paramNames = self.paramInfo.keys()
            params = {}
            for paramName in paramNames:
                params[paramName] = self.paramInfo[paramName]["get"]()
            return params

    def sendMessage(self, process, msg):
        """Send a message to a child process.

        This is a convenient way to safely send a message to a process.

        Args:
            process (StateMachineProces): A descendant of StateMachineProcess
            msg (2-tuple): A tuple of the following form:
                (Messages.[[message type]], [[argument]])
                Where the first element is a message type, as defined by the
                StateMachineProcesses.Messages class, and the second argument is
                an argument that depends on the message type. For example,
                    (Messages.SETPARAMS, {'verbose':2})
                or
                    (Messages.START, None)

        Returns:
            bool: A boolean indicating whether sending the message succeeded

        """
        if process is None:
            return False
        else:
            if process.msgQueue is None:
                return False
            else:
                process.msgQueue.put(msg)
                return True

    def setBufferSizeSeconds(self, *args):
        """Placeholder param setter indicating this property is not settable"""
        raise AttributeError('This attribute is a derived property, and is not directly settable')
    def setBufferSizeAudioChunks(self, *args):
        """Placeholder param setter indicating this property is not settable"""
        raise AttributeError('This attribute is a derived property, and is not directly settable')
    def setNumStreams(self, *args):
        """Placeholder param setter indicating this property is not settable"""
        raise AttributeError('This attribute is a derived property, and is not directly settable')
    def setNumProcesses(self, *args):
        """Placeholder param setter indicating this property is not settable"""
        raise AttributeError('This attribute is a derived property, and is not directly settable')
    def setNumSyncedProcesses(self, *args):
        """Placeholder param setter indicating this property is not settable"""
        raise AttributeError('This attribute is a derived property, and is not directly settable')

    def getBufferSizeSeconds(self):
        """Set bufferSizeSeconds parameter.

        This is the old way of deterining an appropriate buffer
            size based on the pre-trigger time requested. It is not used by
            VideoAcquirer any more - see param "acquisitionBufferSize" instead.

        See also: PyVAQ.paramInfo

        Returns:
            float: Calculated size of acquisition buffer in seconds

        """
        preTriggerTime = self.getParams('preTriggerTime')
        return preTriggerTime * 2 + 1    # Twice the pretrigger time to make sure we don't miss stuff, plus one second for good measure
    def getBufferSizeAudioChunks(self):
        """Calculate the number of chunks in the audio buffer - not in use"""
        p = self.getParams('bufferSizeSeconds', 'audioFrequency', 'chunkSize')
        return p['bufferSizeSeconds'] * p['audioFrequency'] / p['chunkSize']   # Will be rounded up to nearest integer
    def getNumStreams(self):
        """Get # of audio/video streams in the current configuration."""
        audioDAQChannels = self.getParams('audioDAQChannels')
        camSerials = self.getParams('camSerials')
        return (len(audioDAQChannels)>0) + len(camSerials)
    def getNumProcesses(self):
        """Get # of child processes in the current configuration."""
        audioDAQChannels = self.getParams('audioDAQChannels')
        camSerials = self.getParams('camSerials')
        return (len(audioDAQChannels)>0) + len(camSerials)*2 + 2
    def getNumSyncedProcesses(self):
        """Get # of processes subject to synchronization in current config"""
        audioDAQChannels = self.getParams('audioDAQChannels')
        camSerials = self.getParams('camSerials')
        return (len(audioDAQChannels)>0) + len(camSerials) + 1  # 0 or 1 audio acquire processes, N video acquire processes, and 1 sync process
    def getCameraSettings(self):
        """Get the current set of camera settings.

        These are settings for FLIR USB3 Vision cameras, such as the Flea3
        and Blackfly S series of cameras.

        See also: StateMachineProcesses.VideoAcquirer.setCameraAttributes

        Returns:
            list of tuples: A list of camera settings, formatted as a list of
                3-tuples, where each tuple is of the form:
                    ([[setting name]], [[setting value]], [[setting type]])

        """
        gain = self.getParams('gain')
        return [
            ('AcquisitionMode', 'Continuous', 'enum'),
            ('TriggerMode', 'Off', 'enum'),
            ('TriggerSelector', 'FrameStart', 'enum'),
            ('TriggerSource', 'Line0', 'enum'),
            ('TriggerActivation', 'RisingEdge', 'enum'),
            ('PixelFormat', 'BayerRG8', 'enum'),
            # ('ExposureMode', 'TriggerWidth'),
            # ('Width', 800, 'integer'),
            # ('Height', 800, 'integer'),
            ('TriggerMode', 'On', 'enum'),
            ('GainAuto', 'Off', 'enum'),
            ('Gain', gain, 'float'),
            ('ExposureAuto', 'Off', 'enum'),
            ('ExposureMode', 'TriggerWidth', 'enum')]
#            ('ExposureTime', exposureTime, 'float')]

    def setAcquireSettings(self, *args):
        """Placeholder param setter indicating this property is not settable"""
        raise NotImplementedError()

    def waitForChildProcessesToStop(self, attempts=10, timeout=5):
        """Wait for all state machine child processes to stop.

        Wait for until all child processes stop, or all attempts have been
        exhausted.

        Args:
            attempts (int): Number of attempts before giving up. Defaults to 10.
            timeout (int): Amount of time in seconds to spread the attempts
                over. Defaults to 5.

        Returns:
            bool: Returns true if all processes were found to have stopped,
                false if not.

        """
        for attempts in range(attempts):
            allStopped = False
            states, stateNames = self.checkStates(verbose=False)
            if 'videoWriteStates' in states:
                for camSerial in states['videoWriteStates']:
                    if not (states['videoWriteStates'][camSerial] == States.STOPPED):
                        break;
            if 'videoAcquireStates' in states:
                for camSerial in states['videoAcquireStates']:
                    if not (states['videoAcquireStates'][camSerial] == States.STOPPED):
                        break;
            if 'audioWriteState' in states:
                if not (states['audioWriteState'] == States.STOPPED):
                    break;
            if 'audioAcquireState' in states:
                if not (states['audioAcquireState'] == States.STOPPED):
                    break;
            if 'syncState' in states:
                if not (states['syncState'] == States.STOPPED):
                    break;
            if 'mergeState' in states:
                if not (states['mergeState'] == States.STOPPED):
                    break;
            allStopped = True

            if allStopped:
                return allStopped

            time.sleep(timeout/attempts)

        return False

    def createChildProcesses(self):
        """Instantiate all child processes.

        This method instantiates all child processes, but does not start them.
            Each child process is a subclass of multiprocessing.Process.

        In some cases, the type and number of child processes to start is
            dictated by various parameters.

        Returns:
            None

        """
        self.log("Creating child processes")
        p = self.getParams()

        self.log('Number of synced processes = {k}'.format(k=p["numSyncedProcesses"]))
        ready = mp.Barrier(p["numSyncedProcesses"], timeout=0.5)

        if self.StdoutManager is None:
            self.StdoutManager = StdoutManager()
            self.StdoutManager.start()

        # Shared values so all processes can access actual DAQ frequencies
        #   determined by Synchronizer process. This value should only change
        #   once when the Synchronizer is initialized, and not again until
        #   all child processes are stopped and restarted.
        self.actualVideoFrequency = mp.Value('d', -1)
        self.actualAudioFrequency = mp.Value('d', -1)

        startTime = mp.Value('d', -1)

        if p["mergeFiles"] and p["numStreams"] >= 2:
            # Create merge process
            self.mergeProcess = AVMerger(
                directory=p["mergeDirectory"],
                numFilesPerTrigger=p["numStreams"],
                verbose=self.mergeVerbose,
                stdoutQueue=self.StdoutManager.queue,
                baseFileName=p["mergeBaseFileName"],
                montage=p["montageMerge"],
                deleteMergedAudioFiles=p["deleteMergedAudioFiles"],
                deleteMergedVideoFiles=p["deleteMergedVideoFiles"],
                compression=p["mergeCompression"]
                )
            mergeMsgQueue = self.mergeProcess.msgQueue
        else:
            mergeMsgQueue = None


        if p["audioSyncTerminal"] is not None or p["videoSyncTerminal"] is not None:
            # Create sync process
            self.syncProcess = Synchronizer(
                actualVideoFrequency=self.actualVideoFrequency,
                actualAudioFrequency=self.actualAudioFrequency,
                startTime=startTime,
                signalChannel=p['acquisitionSignalChannel'],
                startOnHWSignal=p['startOnHWSignal'],
                writeEnableOnHWSignal=p['writeEnableOnHWSignal'],
                audioSyncChannel=p["audioSyncTerminal"],
                videoSyncChannel=p["videoSyncTerminal"],
                videoDutyCycle=convertExposureTimeToDutyCycle(p["videoExposureTime"]/1000, p["videoFrequency"]),
                requestedAudioFrequency=p["audioFrequency"],
                requestedVideoFrequency=p["videoFrequency"],
                verbose=self.syncVerbose,
                ready=ready,
                stdoutQueue=self.StdoutManager.queue)

        copyToMonitoringQueue = True
        copyToAnalysisQueue = p["triggerMode"] != "SimpleContinuous"

        if len(p["audioDAQChannels"]) > 0:
            audioQueue = mp.Queue()
            self.audioAcquireProcess = AudioAcquirer(
                startTime=startTime,
                audioQueue=audioQueue,
                chunkSize=p["chunkSize"],
                audioFrequency=self.actualAudioFrequency,
                bufferSize=None,
                channelNames=p["audioDAQChannels"],
                channelConfig=p["audioChannelConfiguration"],
                syncChannel=p["audioSyncSource"],
                verbose=self.audioAcquireVerbose,
                ready=ready,
                copyToMonitoringQueue=copyToMonitoringQueue,
                copyToAnalysisQueue=copyToAnalysisQueue,
                stdoutQueue=self.StdoutManager.queue)
            if p["triggerMode"] == "SimpleContinuous":
                if mergeMsgQueue is not None:
                    self.log('Warning: SimpleAudioWriter does not support A/V merging yet.')
                self.audioWriteProcess = SimpleAudioWriter(
                    audioDirectory=p["audioDirectory"],
                    audioBaseFileName=p["audioBaseFileName"],
                    channelNames=p["audioDAQChannels"],
                    audioQueue=audioQueue,
                    audioFrequency=self.actualAudioFrequency,
                    frameRate=self.actualVideoFrequency,
                    numChannels=len(p["audioDAQChannels"]),
                    videoLength=p["recordTime"],
                    mergeMessageQueue=mergeMsgQueue,
                    daySubfolders=p['daySubfolders'],
                    verbose=self.audioWriteVerbose,
                    stdoutQueue=self.StdoutManager.queue)
            elif p["triggerMode"] != 'None':
                self.audioWriteProcess = AudioWriter(
                    audioDirectory=p["audioDirectory"],
                    audioBaseFileName=p["audioBaseFileName"],
                    channelNames=p["audioDAQChannels"],
                    audioQueue=audioQueue,
                    mergeMessageQueue=mergeMsgQueue,
                    chunkSize=p["chunkSize"],
                    bufferSizeSeconds=p["bufferSizeSeconds"],
                    audioFrequency=self.actualAudioFrequency,
                    numChannels=len(p["audioDAQChannels"]),
                    daySubfolders=p['daySubfolders'],
                    verbose=self.audioWriteVerbose,
                    stdoutQueue=self.StdoutManager.queue)

        gpuCount = 0
        for camSerial in p["camSerials"]:
            if camSerial in p["videoDirectories"]:
                videoDirectory = p["videoDirectories"][camSerial]
            else:
                videoDirectory = ''
            if camSerial in p["videoBaseFileNames"]:
                videoBaseFileName = p["videoBaseFileNames"][camSerial]
            else:
                videoBaseFileName = ''

            processes = {}

            videoAcquireProcess = VideoAcquirer(
                startTime=startTime,
                camSerial=camSerial,
                acquireSettings=p["acquireSettings"],
                frameRate = self.actualVideoFrequency,
                requestedFrameRate=p["videoFrequency"],
                monitorFrameRate=self.monitorMasterFrameRate,
                verbose=self.videoAcquireVerbose,
                bufferSizeSeconds=p["acquisitionBufferSize"],
                ready=ready,
                stdoutQueue=self.StdoutManager.queue)

            if camSerial in p['videoWriteEnable']:
                videoWriteEnable = p['videoWriteEnable'][camSerial]
            else:
                videoWriteEnable = True

            if p["triggerMode"] == "SimpleContinuous":
                gpuOk = (gpuCount < p['maxGPUVEnc'])
                if p['maxGPUVEnc'] > 0 and not gpuOk:
                    # Some GPU video encoder sessions requested, but not enough for all cameras.
                    self.log('Warning: Cannot use GPU acceleration for all cameras - not enough GPU VEnc sessions allowed.')
                videoWriteProcess = SimpleVideoWriter(
                    camSerial=camSerial,
                    videoDirectory=videoDirectory,
                    videoBaseFileName=videoBaseFileName,
                    imageQueue=videoAcquireProcess.imageQueueReceiver,
                    frameRate=self.actualVideoFrequency,
                    requestedFrameRate=p["videoFrequency"],
                    mergeMessageQueue=mergeMsgQueue,
                    videoLength=p["recordTime"],
                    daySubfolders=p['daySubfolders'],
                    verbose=self.videoWriteVerbose,
                    stdoutQueue=self.StdoutManager.queue,
                    gpuVEnc=gpuOk,
                    scheduleEnabled=p['scheduleEnabled'],
                    scheduleStartTime=p['scheduleStart'],
                    scheduleStopTime=p['scheduleStop'],
                    enableWrite=videoWriteEnable
                    )
                gpuCount += 1
            elif p["triggerMode"] == 'None':
                videoWriteProcess = None
            else:
                videoWriteProcess = VideoWriter(
                    camSerial=camSerial,
                    videoDirectory=videoDirectory,
                    videoBaseFileName=videoBaseFileName,
                    imageQueue=videoAcquireProcess.imageQueueReceiver,
                    frameRate=self.actualVideoFrequency,
                    requestedFrameRate=p["videoFrequency"],
                    mergeMessageQueue=mergeMsgQueue,
                    bufferSizeSeconds=p["bufferSizeSeconds"],
                    daySubfolders=p['daySubfolders'],
                    verbose=self.videoWriteVerbose,
                    stdoutQueue=self.StdoutManager.queue
                    )
            self.videoAcquireProcesses[camSerial] = videoAcquireProcess
            self.videoWriteProcesses[camSerial] = videoWriteProcess

        if p["triggerMode"] != "SimpleContinuous":
            # Create (but don't start) continuous trigger process for sending
            #   automatic, continuous, and consecutive triggers to audio and video
            #   writers
            self.continuousTriggerProcess = ContinuousTriggerer(
                startTime=startTime,
                recordPeriod=p['continuousTriggerPeriod'],
                verbose=self.continuousTriggerVerbose,
                audioMessageQueue=self.audioWriteProcess.msgQueue if self.audioWriteProcess else None,
                videoMessageQueues=dict([(camSerial, self.videoWriteProcesses[camSerial].msgQueue) for camSerial in self.videoWriteProcesses if self.videoWriteProcesses[camSerial] is not None]),
                stdoutQueue=self.StdoutManager.queue
            )

        if self.continuousTriggerProcess is None:
            taggerQueues = []
        else:
            [self.continuousTriggerProcess.msgQueue]

        # If we have an audioAcquireProcess, create (but don't start) an
        #   audioTriggerProcess to generate audio-based triggers
        if self.audioAcquireProcess is not None and \
                self.getParams('triggerMode') == "Audio" and \
                len(p["audioDAQChannels"]) > 0:
            self.audioTriggerProcess = AudioTriggerer(
                audioQueue=self.audioAcquireProcess.analysisQueue,
                audioFrequency=self.actualAudioFrequency,
                chunkSize=p["chunkSize"],
                triggerHighLevel=p["triggerHighLevel"],
                triggerLowLevel=p["triggerLowLevel"],
                triggerHighTime=p["triggerHighTime"],
                triggerLowTime=p["triggerLowTime"],
                triggerHighFraction=p["triggerHighFraction"],
                triggerLowFraction=p["triggerLowFraction"],
                maxAudioTriggerTime=p["maxAudioTriggerTime"],
                preTriggerTime=p["preTriggerTime"],
                multiChannelStartBehavior=p["multiChannelStartBehavior"],
                multiChannelStopBehavior=p["multiChannelStopBehavior"],
                bandpassFrequencies=(p['triggerLowBandpass'], p['triggerHighBandpass']),
                taggerQueues=taggerQueues,
                verbose=self.audioTriggerVerbose,
                audioMessageQueue=self.audioWriteProcess.msgQueue,
                videoMessageQueues=dict([(camSerial, self.videoWriteProcesses[camSerial].msgQueue) for camSerial in self.videoWriteProcesses if self.videoWriteProcesses[camSerial] is not None]),
                stdoutQueue=self.StdoutManager.queue
                )

        # Start all audio-related processes
        if self.audioTriggerProcess is not None:
            self.audioTriggerProcess.start()
            if self.getParams('triggerMode') == "Audio":
                self.sendMessage(self.audioTriggerProcess, (Messages.STARTANALYZE, None))

        if len(p["audioDAQChannels"]) > 0:
            self.audioWriteProcess.start()
            self.audioAcquireProcess.start()

        # Start all video-related processes
        for camSerial in p["camSerials"]:
            if self.videoWriteProcesses[camSerial] is not None:
                self.videoWriteProcesses[camSerial].start()
            self.videoAcquireProcesses[camSerial].start()

        # Start other processes
        if self.syncProcess is not None: self.syncProcess.start()
        if self.mergeProcess is not None: self.mergeProcess.start()
        if self.continuousTriggerProcess is not None: self.continuousTriggerProcess.start()

        self.endLog(inspect.currentframe().f_code.co_name)

    def initializeChildProcesses(self):
        """Send messages to all child processes to start.

        Returns:
            None

        """
        p = self.getParams('audioDAQChannels', 'camSerials', 'triggerMode')

        if len(p["audioDAQChannels"]) > 0:
            # Start audio trigger process
            self.sendMessage(self.audioTriggerProcess, (Messages.START, None))
            self.updateTriggerMode()

            # Start AudioWriter
            self.sendMessage(self.audioWriteProcess, (Messages.START, None))

            # Start AudioAcquirer
            self.sendMessage(self.audioAcquireProcess, (Messages.START, None))

        # Start continuous trigger process
        if self.getParams('triggerMode') == 'Continuous':
            self.sendMessage(self.continuousTriggerProcess, (Messages.START, None))

        # For each camera
        for camSerial in p["camSerials"]:
            # Start VideoWriter
            self.sendMessage(self.videoWriteProcesses[camSerial], (Messages.START, None))
            # Start VideoAcquirer
            self.sendMessage(self.videoAcquireProcesses[camSerial], (Messages.START, None))

        if len(p["audioDAQChannels"]) + len(p["camSerials"]) >= 1:
            # Start sync process
            self.sendMessage(self.syncProcess, (Messages.START, None))

        if len(p["audioDAQChannels"]) + len(p["camSerials"]) >= 2:
            # Start merge process
            self.updateAVMergerState()

    # def restartAcquisition(self):
    #     self.haltChildProcesses()
    #     stopped = self.waitForChildProcessesToStop()
    #     if stopped:
    #         self.initializeChildProcesses()
    #     else:
    #         self.log("Attempted to restart child processes, but could not get them to stop.")
    #         self.endLog(inspect.currentframe().f_code.co_name)

    def haltChildProcesses(self):
        """Tell all child processes to stop (but not close).

        Send a message to all child processes to enter the "STOPPED" state, but
            not to close/exit/stop running.

        Returns:
            None

        """
        # Tell all child processes to go to stopped state
        self.sendMessage(self.audioTriggerProcess, (Messages.STOP, None))
        self.sendMessage(self.continuousTriggerProcess, (Messages.STOP, None))
        for camSerial in self.videoAcquireProcesses:
            self.sendMessage(self.videoAcquireProcesses[camSerial], (Messages.STOP, None))
        for camSerial in self.videoAcquireProcesses:
            self.sendMessage(self.videoWriteProcesses[camSerial], (Messages.STOP, None))
        self.sendMessage(self.audioAcquireProcess, (Messages.STOP, None))
        self.sendMessage(self.audioWriteProcess, (Messages.STOP, None))
        self.sendMessage(self.mergeProcess, (Messages.STOP, None))
        self.sendMessage(self.syncProcess, (Messages.STOP, None))

    def exitChildProcesses(self):
        """Tell all child processes to exit.

        Returns:
            None

        """
        self.sendMessage(self.audioTriggerProcess, (Messages.EXIT, None))
        self.sendMessage(self.continuousTriggerProcess, (Messages.EXIT, None))
        for camSerial in self.videoAcquireProcesses:
            self.sendMessage(self.videoAcquireProcesses[camSerial], (Messages.EXIT, None))
        for camSerial in self.videoWriteProcesses:
            self.sendMessage(self.videoWriteProcesses[camSerial], (Messages.EXIT, None))
        self.sendMessage(self.audioAcquireProcess, (Messages.EXIT, None))
        self.sendMessage(self.audioWriteProcess, (Messages.EXIT, None))
        self.sendMessage(self.mergeProcess, (Messages.EXIT, None))
        self.sendMessage(self.syncProcess, (Messages.EXIT, None))
        #self.StdoutManager.queue.put(Messages.EXIT)

    def destroyChildProcesses(self):
        """Exit then dereference all child processes.

        Probably not really any different in effect from exitChildProcesses...

        Returns:
            None

        """
        self.exitChildProcesses()

        self.actualVideoFrequency = None
        self.actualAudioFrequency = None

        # Give children a chance to register exit message
        time.sleep(0.5)

        # try:
        #     s = io.StringIO()
        #     ps = pstats.Stats(self.profiler, stream=s)
        #     ps.print_stats()
        #     self.log('', s.getvalue())
        # except:
        #     self.log('Error printing profiler stats')

        self.audioTriggerProcess = None
        self.continuousTriggerProcess = None
        self.audioAcquireProcess = None
        self.audioWriteProcess = None
        self.videoAcquireProcesses = {}
        self.videoWriteProcesses = {}
        self.mergeProcess = None
        self.syncProcess = None
        self.StdoutManager = None

        self.endLog(inspect.currentframe().f_code.co_name)

    def sendWriteTrigger(self, t=None):
        """Send a manual trigger to write processes to trigger recording.

        Normally will only be called in "Manual" trigger mode.

        See writeButtonClickHandler

        Returns:
            None

        """
        p = self.getParams('preTriggerTime', 'recordTime')
        if t is None:
            t = time.time_ns()/1000000000
        trig = Trigger(t-p['preTriggerTime'], t, t + p['recordTime'] - p['preTriggerTime'], idspace='GUI')
        self.log("Sending manual trigger!")
        for camSerial in self.getParams('camSerials'):
            success = self.sendMessage(self.videoWriteProcesses[camSerial], (Messages.TRIGGER, trig))
            if success:
                self.log("...sent to", camSerial, "video writer")
        success = self.sendMessage(self.audioWriteProcess, (Messages.TRIGGER, trig))
        if success:
            self.log("...sent to audio writer")
        self.endLog(inspect.currentframe().f_code.co_name)

    def update(self):
        """Update GUI widget layout.

        Layout sketch:

        root window
            titleBarFrame
            mainFrame
            monitorFrame
                videoMonitorMasterFrame
                audioMonitorMasterFrame
            controlFrame
                acquisitionControlFrame
                statusFrame
                acquisitionParametersFrame
                mergeFrame
                fileSettingsFrame
                scheduleFrame
                triggerFrame

        Returns:
            None

        """
        if self.customTitleBar:
            self.titleBarFrame.grid(row=0, column=0, sticky=tk.NSEW)
            self.closeButton.grid(sticky=tk.E)
        else:
            self.titleBarFrame.grid_forget()
            self.closeButton.grid_forget()
        self.mainFrame.grid(row=1, column=1, sticky=tk.NSEW)
        # self.mainFrame.columnconfigure(0, weight=1)
        # self.mainFrame.columnconfigure(1, weight=1)
        # self.mainFrame.rowconfigure(0, weight=1)
        # self.mainFrame.rowconfigure(1, weight=1)

        p = self.getParams(
            'camSerials',
            'audioDAQChannels',
            )
        camSerials = p["camSerials"]
        audioDAQChannels = p["audioDAQChannels"]

        if (self.audioMonitorDocker.isDocked() and len(audioDAQChannels) > 0) or (self.videoMonitorDocker.isDocked() and len(camSerials) > 0):
            self.monitorMasterFrame.grid(row=0, column=1, sticky=tk.NSEW)
        else:
            self.monitorMasterFrame.grid_forget()

        if self.videoMonitorDocker.isDocked():
            self.videoMonitorMasterFrame.grid(row=0, column=0, sticky=tk.NSEW)
        camSerials = self.getParams('camSerials')
        wV, hV = getOptimalMonitorGrid(len(camSerials))
        for k, camSerial in enumerate(camSerials):
            self.cameraMonitors[camSerial].grid(row=1+2*(k // wV), column = k % wV)
            # self.cameraAttributeBrowserButtons[camSerial].grid(row=1, column=0)

        if self.audioMonitorDocker.isDocked():
            self.audioMonitorDocker.docker.grid(row=1, column=0, sticky=tk.NSEW)
#        self.audioMonitor.grid(row=1, column=0, sticky=tk.NSEW)

        self.controlFrame.grid(row=0, column=0, sticky=tk.NSEW)
        # self.controlFrame.columnconfigure(0, weight=1)
        # self.controlFrame.columnconfigure(1, weight=1)
        # self.controlFrame.columnconfigure(1, weight=1)
        # self.controlFrame.rowconfigure(0, weight=1)
        # self.controlFrame.rowconfigure(1, weight=1)

        self.acquisitionControlFrame.grid(   row=0, column=0, sticky=tk.NSEW)
        self.statusFrame.grid(               row=1, column=0, sticky=tk.NSEW)
        self.acquisitionParametersFrame.grid(row=2, column=0, sticky=tk.NSEW)
        self.mergeFrame.grid(                row=3, column=0, sticky=tk.NSEW)
        self.fileSettingsFrame.grid(         row=4, column=0, sticky=tk.NSEW)
        self.scheduleFrame.grid(             row=5, column=0, sticky=tk.NSEW)
        self.triggerFrame.grid(              row=6, column=0, sticky=tk.NSEW)

        #### Children of self.statusFrame
        self.childStatusText.grid()

        # for c in range(3):
        #     self.acquisitionFrame.columnconfigure(c, weight=1)
        # for r in range(4):
        #     self.acquisitionFrame.rowconfigure(r, weight=1)

        #### Children of self.acquisitionControlFrame
        self.initializeAcquisitionButton.grid(row=0, column=0, sticky=tk.NSEW)
        self.haltAcquisitionButton.grid(      row=0, column=1, sticky=tk.NSEW)
        self.restartAcquisitionButton.grid(   row=0, column=2, sticky=tk.NSEW)
        self.shutDownAcquisitionButton.grid(  row=0, column=3, sticky=tk.NSEW)

        #### Children of self.acquisitionParametersFrame
        self.audioFrequencyFrame.grid(              row=1, column=0, sticky=tk.EW)
        self.audioFrequencyEntry.grid()
        self.videoFrequencyFrame.grid(              row=1, column=1, sticky=tk.EW)
        self.videoFrequencyEntry.grid()
        self.videoExposureTimeFrame.grid(           row=1, column=2, sticky=tk.EW)
        self.videoExposureTimeEntry.grid()
        self.gainFrame.grid(                        row=1, column=3, sticky=tk.EW)
        self.gainEntry.grid()
        self.acquisitionBufferSizeFrame.grid(           row=2, column=0, sticky=tk.EW)
        self.acquisitionBufferSizeEntry.grid()
        self.preTriggerTimeFrame.grid(              row=2, column=1, sticky=tk.EW)
        self.preTriggerTimeEntry.grid()
        self.recordTimeFrame.grid(                  row=2, column=2, sticky=tk.EW)
        self.recordTimeEntry.grid()
        self.maxGPUVencFrame.grid(                  row=2, column=3, sticky=tk.NSEW)
        self.maxGPUVEncEntry.grid()
        self.acquisitionSignalParametersFrame.grid( row=3, column=0, columnspan=4, sticky=tk.NSEW)
        self.startOnHWSignalCheckbutton.grid(row=0, column=0)
        self.writeEnableOnHWSignalCheckbutton.grid(row=0, column=1)
        self.selectAcquisitionHardwareButton.grid(  row=4, column=0, columnspan=4, sticky=tk.NSEW)
        self.acquisitionHardwareText.grid(          row=5, column=0, columnspan=4)

        #### Children of self.mergeFrame
        self.mergeFilesCheckbutton.grid(            row=1, column=0, sticky=tk.NW)
        self.deleteMergedFilesFrame.grid(           row=2, column=0, sticky=tk.NW)
        self.mergeCompressionFrame.grid(            row=2, column=1, sticky=tk.NW)
        self.mergeCompression.grid()
        self.deleteMergedAudioFilesCheckbutton.grid(row=0, column=0, sticky=tk.NW)
        self.deleteMergedVideoFilesCheckbutton.grid(row=0, column=1, sticky=tk.NW)
        self.montageMergeCheckbutton.grid(          row=3, column=0, sticky=tk.NW)

        self.mergeFileWidget.grid(                  row=4, column=0, columnspan=2)

        #### Children of self.fileSettingsFrame
        self.daySubfoldersCheckbutton.grid( row=0, column=0)

        #### Children of self.scheduleFrame
        self.scheduleEnabledCheckbutton.grid(   row=0, column=0, sticky=tk.NW)
        self.scheduleStartTimeEntry.grid(       row=1, column=0, sticky=tk.NW)
        self.scheduleStopTimeEntry.grid(        row=2, column=0, sticky=tk.NW)

        #### Children of self.triggerFrame
        self.triggerModeChooserFrame.grid(          row=0, column=0, sticky=tk.NW)
        self.triggerModeLabel.grid(                 row=0, column=0)
        for k, mode in enumerate(self.triggerModes):
            self.triggerModeRadioButtons[mode].grid(row=0, column=k+1)
        self.triggerControlTabs.grid(               row=1, column=0)
        # if mode == self.triggerModeVar.get():
        #     self.triggerModeControlGroupFrames[mode].grid(row=1, column=0)
        # else:
        #     self.triggerModeControlGroupFrames[mode].grid_forget()
        self.manualWriteTriggerButton.grid(row=1, column=0)

        self.manualSyncStartButton.grid(row=1, column=1)

        self.triggerHiLoFrame.grid(row=0, column=0, columnspan=4, sticky=tk.NW)
        self.triggerHighLevelFrame.grid(row=0, column=0)
        self.triggerHighLevelEntry.grid()
        self.triggerLowLevelFrame.grid(row=1, column=0)
        self.triggerLowLevelEntry.grid()
        self.triggerHighTimeFrame.grid(row=0, column=1)
        self.triggerHighTimeEntry.grid()
        self.triggerLowTimeFrame.grid(row=1, column=1)
        self.triggerLowTimeEntry.grid()
        self.triggerHighFractionFrame.grid(row=0, column=2)
        self.triggerHighFractionEntry.grid()
        self.triggerLowFractionFrame.grid(row=1, column=2)
        self.triggerLowFractionEntry.grid()
        self.triggerHighBandpassFrame.grid(row=0, column=3)
        self.triggerHighBandpassEntry.grid()
        self.triggerLowBandpassFrame.grid(row=1, column=3)
        self.triggerLowBandpassEntry.grid()

        self.multiChannelStartBehaviorFrame.grid(row=1, rowspan=2, column=0, sticky=tk.NW)
        self.multiChannelStartBehaviorOR.grid(row=0)
        self.multiChannelStartBehaviorAND.grid(row=1)

        self.multiChannelStopBehaviorFrame.grid(row=1, rowspan=2, column=1, sticky=tk.NW)
        self.multiChannelStopBehaviorOR.grid(row=0)
        self.multiChannelStopBehaviorAND.grid(row=1)

        self.maxAudioTriggerTimeFrame.grid(row=1, column=2, sticky=tk.NW)
        self.maxAudioTriggerTimeEntry.grid()

        self.audioTriggerStateFrame.grid(row=2, column=2, sticky=tk.NW)
        self.audioTriggerStateLabel.grid()

        self.continuousTriggerModeStart.grid(row=0, column=0)
        self.continuousTriggerModeStop.grid(row=0, column=1)
        self.continuousTriggerPeriodFrame.grid(row=1, column=0)
        self.continuousTriggerPeriodEntry.grid(row=0, column=0)
        self.audioTagContinuousTrigsCheckbutton.grid(row=1, column=1)

        self.noneTriggerModeLabel.grid()

        self.audioAnalysisMonitorFrame.grid(row=4, column=0, columnspan=3)
        self.audioAnalysisWidgets['canvas'].get_tk_widget().pack(side=tk.LEFT, fill=tk.BOTH, expand=1)

if __name__ == "__main__":
    simulatedHardware = False
    settingsFilePath = None
    for arg in sys.argv[1:]:
        if arg == '-s' or arg == '--sim':
            # Use simulated harddware instead of physical cameras and DAQs
            simulatedHardware = True
        else:
            # Any other parameter is the settings file path
            settingsFilePath = arg

    if simulatedHardware:
        import PySpinSim.PySpinSim as PySpin
        import nidaqmxSim.system as nisys

    root = tk.Tk()
    p = PyVAQ(root, settingsFilePath=settingsFilePath)
    root.mainloop()


r"""
cd "C:\Users\Brian Kardon\Dropbox\Documents\Work\Cornell Lab Tech\Projects\Video VI\PyVAQ\Source"
python PyVAQ.py
"""
