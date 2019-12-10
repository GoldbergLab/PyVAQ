import sys
import os
import time
import json
import numpy as np
import multiprocessing as mp
import nidaqmx.system as nisys
import tkinter as tk
import tkinter.ttk as ttk
from tkinter.filedialog import askdirectory, asksaveasfilename, askopenfilename
from tkinter.messagebox import showinfo
import queue
from PIL import Image
# import pprint
import traceback
from collections import deque
import re
import unicodedata
from TimeInput import TimeVar, TimeEntry
from ParamDialog import ParamDialog, Param
from fileWritingEntry import FileWritingEntry
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
from pympler import tracker
try:
    import PySpin
except ModuleNotFoundError:
    # pip seems to install PySpin as pyspin sometimes...
    import pyspin as PySpin
from MonitorWidgets import AudioMonitor, CameraMonitor
from StateMachineProcesses import Trigger, StdoutManager, AVMerger, Synchronizer, AudioTriggerer, AudioAcquirer, AudioWriter, VideoAcquirer, VideoWriter, nodeAccessorFunctions, nodeAccessorTypes

VERSION='0.2.0'

# Todo:
#  - Add filename/directory entry for each stream
#  - Find and plug memory leak
#  - Add video frameRate indicator
#  - Make attributes settable
#  - Make saved avis not gigantic (maybe switch to opencv for video writing?)
#  - Add external record triggering
# Done
#  - Add help dialog that includes version
#  - Fix acquire/write indicator positioing
#  - Add volume-based triggering
#  - Camera commands are not being collected properly
#  - Separate acquire and write modes so it's possible to monitor w/o writing
#  - Rework with each process as an individual state machine
#  - Turn all acquire/write/sync processes into state machines
#  - Figure out why video task isn't closing down properly
#  - Add buffering capability
#  - Fix camera monitor
#  - Add video & audio frequency controls
#  - Fix audio


# Verbosity levels (cumulative):
# 0 - Errors
# 1 - Occasional major status updates
# 2 - Routine status messages
# 3 - Everything

r'''
cd "C:\Users\Brian Kardon\Dropbox\Documents\Work\Cornell Lab Tech\Projects\Video VI\PyVAQ\Source"
python PyVAQ.py
git add * & git commit -m "" & git push origin master
'''

#plt.style.use("dark_background")

pixelFormats = [
"PixelFormat_Mono8",
"PixelFormat_Mono16",
"PixelFormat_RGB8Packed",
"PixelFormat_BayerGR8",
"PixelFormat_BayerRG8",
"PixelFormat_BayerGB8",
"PixelFormat_BayerBG8",
"PixelFormat_BayerGR16",
"PixelFormat_BayerRG16",
"PixelFormat_BayerGB16",
"PixelFormat_BayerBG16",
"PixelFormat_Mono12Packed",
"PixelFormat_BayerGR12Packed",
"PixelFormat_BayerRG12Packed",
"PixelFormat_BayerGB12Packed",
"PixelFormat_BayerBG12Packed",
"PixelFormat_YUV411Packed",
"PixelFormat_YUV422Packed",
"PixelFormat_YUV444Packed",
"PixelFormat_Mono12p",
"PixelFormat_BayerGR12p",
"PixelFormat_BayerRG12p",
"PixelFormat_BayerGB12p",
"PixelFormat_BayerBG12p",
"PixelFormat_YCbCr8",
"PixelFormat_YCbCr422_8",
"PixelFormat_YCbCr411_8",
"PixelFormat_BGR8",
"PixelFormat_BGRa8",
"PixelFormat_Mono10Packed",
"PixelFormat_BayerGR10Packed",
"PixelFormat_BayerRG10Packed",
"PixelFormat_BayerGB10Packed",
"PixelFormat_BayerBG10Packed",
"PixelFormat_Mono10p",
"PixelFormat_BayerGR10p",
"PixelFormat_BayerRG10p",
"PixelFormat_BayerGB10p",
"PixelFormat_BayerBG10p",
"PixelFormat_Mono1p",
"PixelFormat_Mono2p",
"PixelFormat_Mono4p",
"PixelFormat_Mono8s",
"PixelFormat_Mono10",
"PixelFormat_Mono12",
"PixelFormat_Mono14",
"PixelFormat_Mono16s",
"PixelFormat_Mono32f",
"PixelFormat_BayerBG10",
"PixelFormat_BayerBG12",
"PixelFormat_BayerGB10",
"PixelFormat_BayerGB12",
"PixelFormat_BayerGR10",
"PixelFormat_BayerGR12",
"PixelFormat_BayerRG10",
"PixelFormat_BayerRG12",
"PixelFormat_RGBa8",
"PixelFormat_RGBa10",
"PixelFormat_RGBa10p",
"PixelFormat_RGBa12",
"PixelFormat_RGBa12p",
"PixelFormat_RGBa14",
"PixelFormat_RGBa16",
"PixelFormat_RGB8",
"PixelFormat_RGB8_Planar",
"PixelFormat_RGB10",
"PixelFormat_RGB10_Planar",
"PixelFormat_RGB10p",
"PixelFormat_RGB10p32",
"PixelFormat_RGB12",
"PixelFormat_RGB12_Planar",
"PixelFormat_RGB12p",
"PixelFormat_RGB14",
"PixelFormat_RGB16",
"PixelFormat_RGB16s",
"PixelFormat_RGB32f",
"PixelFormat_RGB16_Planar",
"PixelFormat_RGB565p",
"PixelFormat_BGRa10",
"PixelFormat_BGRa10p",
"PixelFormat_BGRa12",
"PixelFormat_BGRa12p",
"PixelFormat_BGRa14",
"PixelFormat_BGRa16",
"PixelFormat_RGBa32f",
"PixelFormat_BGR10",
"PixelFormat_BGR10p",
"PixelFormat_BGR12",
"PixelFormat_BGR12p",
"PixelFormat_BGR14",
"PixelFormat_BGR16",
"PixelFormat_BGR565p",
"PixelFormat_R8",
"PixelFormat_R10",
"PixelFormat_R12",
"PixelFormat_R16",
"PixelFormat_G8",
"PixelFormat_G10",
"PixelFormat_G12",
"PixelFormat_G16",
"PixelFormat_B8",
"PixelFormat_B10",
"PixelFormat_B12",
"PixelFormat_B16",
"PixelFormat_Coord3D_ABC8",
"PixelFormat_Coord3D_ABC8_Planar",
"PixelFormat_Coord3D_ABC10p",
"PixelFormat_Coord3D_ABC10p_Planar",
"PixelFormat_Coord3D_ABC12p",
"PixelFormat_Coord3D_ABC12p_Planar",
"PixelFormat_Coord3D_ABC16",
"PixelFormat_Coord3D_ABC16_Planar",
"PixelFormat_Coord3D_ABC32f",
"PixelFormat_Coord3D_ABC32f_Planar",
"PixelFormat_Coord3D_AC8",
"PixelFormat_Coord3D_AC8_Planar",
"PixelFormat_Coord3D_AC10p",
"PixelFormat_Coord3D_AC10p_Planar",
"PixelFormat_Coord3D_AC12p",
"PixelFormat_Coord3D_AC12p_Planar",
"PixelFormat_Coord3D_AC16",
"PixelFormat_Coord3D_AC16_Planar",
"PixelFormat_Coord3D_AC32f",
"PixelFormat_Coord3D_AC32f_Planar",
"PixelFormat_Coord3D_A8",
"PixelFormat_Coord3D_A10p",
"PixelFormat_Coord3D_A12p",
"PixelFormat_Coord3D_A16",
"PixelFormat_Coord3D_A32f",
"PixelFormat_Coord3D_B8",
"PixelFormat_Coord3D_B10p",
"PixelFormat_Coord3D_B12p",
"PixelFormat_Coord3D_B16",
"PixelFormat_Coord3D_B32f",
"PixelFormat_Coord3D_C8",
"PixelFormat_Coord3D_C10p",
"PixelFormat_Coord3D_C12p",
"PixelFormat_Coord3D_C16",
"PixelFormat_Coord3D_C32f",
"PixelFormat_Confidence1",
"PixelFormat_Confidence1p",
"PixelFormat_Confidence8",
"PixelFormat_Confidence16",
"PixelFormat_Confidence32f",
"PixelFormat_BiColorBGRG8",
"PixelFormat_BiColorBGRG10",
"PixelFormat_BiColorBGRG10p",
"PixelFormat_BiColorBGRG12",
"PixelFormat_BiColorBGRG12p",
"PixelFormat_BiColorRGBG8",
"PixelFormat_BiColorRGBG10",
"PixelFormat_BiColorRGBG10p",
"PixelFormat_BiColorRGBG12",
"PixelFormat_BiColorRGBG12p",
"PixelFormat_SCF1WBWG8",
"PixelFormat_SCF1WBWG10",
"PixelFormat_SCF1WBWG10p",
"PixelFormat_SCF1WBWG12",
"PixelFormat_SCF1WBWG12p",
"PixelFormat_SCF1WBWG14",
"PixelFormat_SCF1WBWG16",
"PixelFormat_SCF1WGWB8",
"PixelFormat_SCF1WGWB10",
"PixelFormat_SCF1WGWB10p",
"PixelFormat_SCF1WGWB12",
"PixelFormat_SCF1WGWB12p",
"PixelFormat_SCF1WGWB14",
"PixelFormat_SCF1WGWB16",
"PixelFormat_SCF1WGWR8",
"PixelFormat_SCF1WGWR10",
"PixelFormat_SCF1WGWR10p",
"PixelFormat_SCF1WGWR12",
"PixelFormat_SCF1WGWR12p",
"PixelFormat_SCF1WGWR14",
"PixelFormat_SCF1WGWR16",
"PixelFormat_SCF1WRWG8",
"PixelFormat_SCF1WRWG10",
"PixelFormat_SCF1WRWG10p",
"PixelFormat_SCF1WRWG12",
"PixelFormat_SCF1WRWG12p",
"PixelFormat_SCF1WRWG14",
"PixelFormat_SCF1WRWG16",
"PixelFormat_YCbCr8_CbYCr",
"PixelFormat_YCbCr10_CbYCr",
"PixelFormat_YCbCr10p_CbYCr",
"PixelFormat_YCbCr12_CbYCr",
"PixelFormat_YCbCr12p_CbYCr",
"PixelFormat_YCbCr411_8_CbYYCrYY",
"PixelFormat_YCbCr422_8_CbYCrY",
"PixelFormat_YCbCr422_10",
"PixelFormat_YCbCr422_10_CbYCrY",
"PixelFormat_YCbCr422_10p",
"PixelFormat_YCbCr422_10p_CbYCrY",
"PixelFormat_YCbCr422_12",
"PixelFormat_YCbCr422_12_CbYCrY",
"PixelFormat_YCbCr422_12p",
"PixelFormat_YCbCr422_12p_CbYCrY",
"PixelFormat_YCbCr601_8_CbYCr",
"PixelFormat_YCbCr601_10_CbYCr",
"PixelFormat_YCbCr601_10p_CbYCr",
"PixelFormat_YCbCr601_12_CbYCr",
"PixelFormat_YCbCr601_12p_CbYCr",
"PixelFormat_YCbCr601_411_8_CbYYCrYY",
"PixelFormat_YCbCr601_422_8",
"PixelFormat_YCbCr601_422_8_CbYCrY",
"PixelFormat_YCbCr601_422_10",
"PixelFormat_YCbCr601_422_10_CbYCrY",
"PixelFormat_YCbCr601_422_10p",
"PixelFormat_YCbCr601_422_10p_CbYCrY",
"PixelFormat_YCbCr601_422_12",
"PixelFormat_YCbCr601_422_12_CbYCrY",
"PixelFormat_YCbCr601_422_12p",
"PixelFormat_YCbCr601_422_12p_CbYCrY",
"PixelFormat_YCbCr709_8_CbYCr",
"PixelFormat_YCbCr709_10_CbYCr",
"PixelFormat_YCbCr709_10p_CbYCr",
"PixelFormat_YCbCr709_12_CbYCr",
"PixelFormat_YCbCr709_12p_CbYCr",
"PixelFormat_YCbCr709_411_8_CbYYCrYY",
"PixelFormat_YCbCr709_422_8",
"PixelFormat_YCbCr709_422_8_CbYCrY",
"PixelFormat_YCbCr709_422_10",
"PixelFormat_YCbCr709_422_10_CbYCrY",
"PixelFormat_YCbCr709_422_10p",
"PixelFormat_YCbCr709_422_10p_CbYCrY",
"PixelFormat_YCbCr709_422_12",
"PixelFormat_YCbCr709_422_12_CbYCrY",
"PixelFormat_YCbCr709_422_12p",
"PixelFormat_YCbCr709_422_12p_CbYCrY",
"PixelFormat_YUV8_UYV",
"PixelFormat_YUV411_8_UYYVYY",
"PixelFormat_YUV422_8",
"PixelFormat_YUV422_8_UYVY",
"PixelFormat_Polarized8",
"PixelFormat_Polarized10p",
"PixelFormat_Polarized12p",
"PixelFormat_Polarized16",
"PixelFormat_BayerRGPolarized8",
"PixelFormat_BayerRGPolarized10p",
"PixelFormat_BayerRGPolarized12p",
"PixelFormat_BayerRGPolarized16",
"PixelFormat_LLCMono8",
"PixelFormat_LLCBayerRG8",
"PixelFormat_JPEGMono8",
"PixelFormat_JPEGColor8",
"PixelFormat_Raw16",
"PixelFormat_Raw8",
"PixelFormat_R12_Jpeg",
"PixelFormat_GR12_Jpeg",
"PixelFormat_GB12_Jpeg",
"PixelFormat_B12_Jpeg"
]

np.set_printoptions(linewidth=200)

def queryAttributeNode(nodePtr, nodeType):
    """
    Retrieves and prints the display name and value of any node.
    """
    try:
        # Create string node
        (nodeTypeName, nodeAccessorFunction) = nodeAccessorFunctions[nodeType]
        node = nodeAccessorFunction(nodePtr)

        # Retrieve string node value
        try:
            display_name = node.GetDisplayName()
        except:
            display_name = None

        # Ensure that the value length is not excessive for printing
        try:
            value = node.GetValue()
        except AttributeError:
            try:
                valueEntry = node.GetCurrentEntry()
                value = (valueEntry.GetName(), valueEntry.GetDisplayName())
            except:
                value = None
        except:
            value = None

        try:
            symbolic  = node.GetSymbolic()
        except AttributeError:
            symbolic = None
        except:
            symbolic = None

        try:
            tooltip = node.GetToolTip()
        except AttributeError:
            tooltip = None
        except:
            tooltip = None

        try:
            accessMode = PySpin.EAccessModeClass_ToString(node.GetAccessMode())
        except AttributeError:
            accessMode = None
        except:
            accessMode = None

        try:
            options = {}
            optionsPtrs = node.GetEntries()
            for optionsPtr in optionsPtrs:
                options[optionsPtr.GetName()] = optionsPtr.GetDisplayName()
        except:
            if nodeTypeName == "enum":
                print("Failed to get options from enum!")
                traceback.print_exc()
            options = {}

        try:
            subcategories = []
            children = []
            for childNode in node.GetFeatures():
                # Ensure node is available and readable
                if not PySpin.IsAvailable(childNode) or not PySpin.IsReadable(childNode):
                    continue
                nodeType = childNode.GetPrincipalInterfaceType()
                if nodeType not in nodeAccessorFunctions:
                    print("Unknown node type:", nodeType)
                    continue
                (childNodeTypeName, nodeAccessorFunction) = nodeAccessorFunctions[nodeType]
                if childNodeTypeName == "category":
                    subcategories.append(queryAttributeNode(childNode, nodeType))
                else:
                    children.append(queryAttributeNode(childNode, nodeType))
        except AttributeError:
            # Not a category node
            pass
        except:
            pass

        try:
            name = node.GetName()
        except:
            name = None

        return {'type':nodeTypeName, 'name':name, 'symbolic':symbolic, 'displayName':display_name, 'value':value, 'tooltip':tooltip, 'accessMode':accessMode, 'options':options, 'subcategories':subcategories, 'children':children}

    except PySpin.SpinnakerException as ex:
        print('Error: %s' % ex)
        traceback.print_exc()
        return None

def getAllCameraAttributes(cam):
    # cam must be initialized before being passed to this function
    try:
        nodeData = {'type':'category', 'name':'Master', 'symbolic':'Master', 'displayName':'Master', 'value':None, 'tooltip':'Camera attributes', 'accessMode':'RO', 'options':{}, 'subcategories':[], 'children':[]}

        nodemap_gentl = cam.GetTLDeviceNodeMap()

        nodeDataTL = queryAttributeNode(nodemap_gentl.GetNode('Root'), PySpin.intfICategory)
        nodeDataTL['displayName'] = "Transport layer settings"
        nodeData['subcategories'].append(nodeDataTL)

        nodemap_tlstream = cam.GetTLStreamNodeMap()

        nodeDataTLStream = queryAttributeNode(nodemap_tlstream.GetNode('Root'), PySpin.intfICategory)
        nodeDataTLStream['displayName'] = "Transport layer stream settings"
        nodeData['subcategories'].append(nodeDataTLStream)

        nodemap_applayer = cam.GetNodeMap()

        nodeDataAppLayer = queryAttributeNode(nodemap_applayer.GetNode('Root'), PySpin.intfICategory)
        nodeDataAppLayer['displayName'] = "Camera settings"
        nodeData['subcategories'].append(nodeDataAppLayer)

        return nodeData

    except PySpin.SpinnakerException as ex:
        print('Error: %s' % ex)
        traceback.print_exc()
        return None

def getOptimalMonitorGrid(numCameras):
    if numCameras == 0:
        return (0,0)
    divisors = []
    for k in range(1, numCameras+1):
        if numCameras % k == 0:
            divisors.append(k)
    bestDivisor = min(divisors, key=lambda d:abs(d - (numCameras/d)))
    return sorted([bestDivisor, int(numCameras/bestDivisor)], key=lambda x:-x)

def checkCameraSpeed(camSerial):
    try:
        system = PySpin.System.GetInstance()
        camList = system.GetCameras()
        cam = camList.GetBySerial(camSerial)
        cam.Init()
        cameraSpeedValue, cameraSpeed = getCameraAttribute(cam.GetTLDeviceNodeMap(), 'DeviceCurrentSpeed', PySpin.CEnumerationPtr)
        cameraBaudValue, cameraBaud =   getCameraAttribute(cam.GetNodeMap(), 'SerialPortBaudRate', PySpin.CEnumerationPtr)
        cameraSpeed = cameraSpeed + ' ' + cameraBaud
        cam.DeInit()
        del cam
        camList.Clear()
        system.ReleaseInstance()
        return cameraSpeed
    except:
        return "Unknown speed"

def flattenList(l):
    return [item for sublist in l for item in sublist]

def discoverDAQAudioChannels():
    s = nisys.System.local()
    channels = {}
    for d in s.devices:
        channels[d.name] = [c.name for c in d.ai_physical_chans]
    return channels

def discoverDAQClockChannels():
    s = nisys.System.local()
    channels = {}
    for d in s.devices:
        channels[d.name] = [c.name for c in d.co_physical_chans]
    return channels

def discoverCameras(numFakeCameras=0):
    system = PySpin.System.GetInstance()
    camList = system.GetCameras()
    camSerials = []
    for cam in camList:
        cam.Init()
        camSerials.append(getCameraAttribute(cam.GetTLDeviceNodeMap(), 'DeviceSerialNumber', PySpin.CStringPtr))
        cam.DeInit()
        del cam
    for k in range(numFakeCameras):
        camSerials.append('fake_camera_'+str(k))
    camList.Clear()
    system.ReleaseInstance()
    return camSerials

def getCameraAttribute(nodemap, attributeName, attributeTypePtrFunction):
    nodeAttribute = attributeTypePtrFunction(nodemap.GetNode(attributeName))

    if not PySpin.IsAvailable(nodeAttribute) or not PySpin.IsReadable(nodeAttribute):
        raise AttributeError('Unable to retrieve '+attributeName+'. Aborting...')
        return None

    try:
        value = nodeAttribute.GetValue()
    except AttributeError:
        # Maybe it's an enum?
        valueEntry = nodeAttribute.GetCurrentEntry()
        value = (valueEntry.GetName(), valueEntry.GetDisplayName())
    return value

def timeToSerializable(time):
    return dict(
        hour=time.hour,
        minute=time.minute,
        second=time.second,
        microsecond=time.microsecond
    )

def format_diff(diff):
    # Diff is a list of the form output by pympler.summary.diff()
    output = '\n'.join(str(d) for d in sorted(diff, key=lambda dd:-dd[2]))

def slugify(value, allow_unicode=False):
    """
    Convert to ASCII if 'allow_unicode' is False. Convert spaces to hyphens.
    Remove characters that aren't alphanumerics, underscores, or hyphens.
    Convert to lowercase. Also strip leading and trailing whitespace.

    Adapter from Django utils
    """
    value = str(value)
    if allow_unicode:
        value = unicodedata.normalize('NFKC', value)
    else:
        value = unicodedata.normalize('NFKD', value).encode('ascii', 'ignore').decode('ascii')
    value = re.sub(r'[^\w\s-]', '', value).strip().lower()
    return re.sub(r'[-\s]+', '-', value)

LINE_STYLES = [c+'-' for c in 'bykcmgr']
WIDGET_COLORS = [
    '#050505', # near black
    '#e6f5ff', # very light blue
    '#c1ffc1', # light green
    '#FFC1C1'  # light red
]

class PyVAQ:
    def __init__(self, master):
        self.master = master
        self.master.resizable(height=False, width=False)  # Disallow resizing window
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

        self.chunkSize = 1000

        self.audioDAQChannels = []

        self.audioSyncSource = None
        self.audioSyncTerminal = None
        self.videoSyncSource = None
        self.videoSyncTerminal = None

        ########### GUI WIDGETS #####################

        self.mainFrame = ttk.Frame(self.master)

        self.menuBar = tk.Menu(self.master, tearoff=False)
        self.settingsMenu = tk.Menu(self.menuBar, tearoff=False)
        self.settingsMenu.add_command(label='Save settings...', command=self.saveSettings)
        self.settingsMenu.add_command(label='Load settings...', command=self.loadSettings)
        self.settingsMenu.add_command(label='Save default settings...', command=lambda *args: self.saveSettings(path='default.pvs'))
        self.settingsMenu.add_command(label='Save default settings...', command=lambda *args: self.loadSettings(path='default.pvs'))

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

        self.menuBar.add_cascade(label="Settings", menu=self.settingsMenu)
        self.menuBar.add_cascade(label="Monitoring", menu=self.monitoringMenu)
        self.menuBar.add_cascade(label="Debug", menu=self.debugMenu)
        self.menuBar.add_cascade(label="Help", menu=self.helpMenu)

        self.master.config(menu=self.menuBar)

        self.titleBarFrame = ttk.Frame(self.master)
        self.closeButton = ttk.Button(self.titleBarFrame, text="X", command=self.cleanupAndExit)

        self.monitorMasterFrame = ttk.Frame(self.mainFrame)
        self.videoMonitorMasterFrame = ttk.Frame(self.monitorMasterFrame)
        self.audioMonitor = None  #ttk.Frame(self.monitorMasterFrame)

        self.cameraAttributes = {}
        self.cameraMonitors = {}

        self.camSerials = []

        self.controlFrame = ttk.Frame(self.mainFrame)

        self.acquisitionFrame = ttk.LabelFrame(self.controlFrame, text="Acquisition")
        self.startChildProcessesButton = ttk.Button(self.acquisitionFrame, text="Start acquisition", command=self.acquireButtonClick)

        self.audioFrequencyFrame =  ttk.LabelFrame(self.acquisitionFrame, text="Audio freq. (Hz)", style='SingleContainer.TLabelframe')
        self.audioFrequencyVar =    tk.StringVar(); self.audioFrequencyVar.set("22050")
        self.audioFrequencyEntry =  ttk.Entry(self.audioFrequencyFrame, width=15, textvariable=self.audioFrequencyVar);

        self.videoFrequencyFrame =  ttk.LabelFrame(self.acquisitionFrame, text="Video freq (fps)", style='SingleContainer.TLabelframe')
        self.videoFrequencyVar =    tk.StringVar(); self.videoFrequencyVar.set("30")
        self.videoFrequencyEntry =  ttk.Entry(self.videoFrequencyFrame, width=15, textvariable=self.videoFrequencyVar)

        self.exposureTimeFrame =    ttk.LabelFrame(self.acquisitionFrame, text="Exposure time (us):", style='SingleContainer.TLabelframe')
        self.exposureTimeVar =      tk.StringVar(); self.exposureTimeVar.set("8000")
        self.exposureTimeEntry =    ttk.Entry(self.exposureTimeFrame, width=18, textvariable=self.exposureTimeVar)

        self.preTriggerTimeFrame =  ttk.LabelFrame(self.acquisitionFrame, text="Pre-trigger record time (s)", style='SingleContainer.TLabelframe')
        self.preTriggerTimeVar =    tk.StringVar(); self.preTriggerTimeVar.set("2.0")
        self.preTriggerTimeEntry =  ttk.Entry(self.preTriggerTimeFrame, width=26, textvariable=self.preTriggerTimeVar)

        self.recordTimeFrame =      ttk.LabelFrame(self.acquisitionFrame, text="Record time (s)", style='SingleContainer.TLabelframe')
        self.recordTimeVar =        tk.StringVar(); self.recordTimeVar.set("4.0")
        self.recordTimeEntry =      ttk.Entry(self.recordTimeFrame, width=14, textvariable=self.recordTimeVar)


        self.updateInputsButton =   ttk.Button(self.acquisitionFrame, text="Select audio/video inputs", command=self.selectInputs)

        self.mergeFrame = ttk.LabelFrame(self.acquisitionFrame, text="AV File merging")

        self.mergeFileWidget = FileWritingEntry(
            self.mergeFrame,
            defaultDirectory=r'C:\Users\Brian Kardon\Documents\Cornell Lab Tech non-syncing\PyVAQ test videos\merge',
            defaultBaseFileName='mergeWrite',
            purposeText='merging audio/video',
            text="Merged A/V Writing"
        )
        self.mergeFileWidget.setDirectoryChangeHandler(self.mergeDirectoryChangeHandler)

        self.mergeFilesVar =        tk.BooleanVar(); self.mergeFilesVar.set(True)
        self.mergeFilesCheckbutton = ttk.Checkbutton(self.mergeFrame, text="Merge audio/video", variable=self.mergeFilesVar, offvalue=False, onvalue=True)
        self.mergeFilesVar.trace('w', self.updateAVMergerState)

        self.deleteMergedFilesFrame = ttk.LabelFrame(self.mergeFrame, text="Delete merged...")

        self.deleteMergedAudioFilesVar = tk.BooleanVar(); self.deleteMergedAudioFilesVar.set(False)
        self.deleteMergedAudioFilesCheckbutton = ttk.Checkbutton(self.deleteMergedFilesFrame, text="Audio files", variable=self.deleteMergedAudioFilesVar, offvalue=False, onvalue=True)
        self.deleteMergedAudioFilesVar.trace('w', lambda *args: self.changeAVMergerParams(deleteMergedAudioFiles=self.deleteMergedAudioFilesVar.get()))

        self.deleteMergedVideoFilesVar = tk.BooleanVar(); self.deleteMergedVideoFilesVar.set(False)
        self.deleteMergedVideoFilesCheckbutton = ttk.Checkbutton(self.deleteMergedFilesFrame, text="Video files", variable=self.deleteMergedVideoFilesVar, offvalue=False, onvalue=True)
        self.deleteMergedVideoFilesVar.trace('w', lambda *args: self.changeAVMergerParams(deleteMergedVideoFiles=self.deleteMergedVideoFilesVar.get()))

        self.montageMergeVar = tk.BooleanVar(); self.montageMergeVar.set(False)
        self.montageMergeCheckbutton = ttk.Checkbutton(self.mergeFrame, text="Montage-merge videos", variable=self.montageMergeVar, offvalue=False, onvalue=True)
        self.montageMergeVar.trace('w', lambda *args: self.changeAVMergerParams(montage=self.montageMergeVar.get()))

        self.mergeCompressionFrame = ttk.LabelFrame(self.mergeFrame, text="Compression:")
        self.mergeCompressionVar = tk.StringVar(); self.mergeCompressionVar.set('0')
        self.mergeCompression = ttk.Combobox(self.mergeCompressionFrame, textvariable=self.mergeCompressionVar, values=[str(k) for k in range(52)], width=12)
        self.mergeCompressionVar.trace('w', lambda *args: self.changeAVMergerParams(mergeCompression=self.mergeCompressionVar.get()))

        self.scheduleFrame = ttk.LabelFrame(self.acquisitionFrame, text="Trigger enable schedule")
        self.scheduleEnabledVar = tk.BooleanVar(); self.scheduleEnabledVar.set(False)
        self.scheduleEnabledCheckbutton = ttk.Checkbutton(self.scheduleFrame, text="Restrict trigger to schedule", variable=self.scheduleEnabledVar)
        self.scheduleStartVar = TimeVar()
        self.scheduleStartTimeEntry = TimeEntry(self.scheduleFrame, text="Start time", style=self.style)
        self.scheduleStopVar = TimeVar()
        self.scheduleStopTimeEntry = TimeEntry(self.scheduleFrame, text="Stop time")

        self.triggerFrame = ttk.LabelFrame(self.controlFrame, text='Triggering')
        self.triggerModes = ['Manual', 'Audio']
        self.triggerModeChooserFrame = ttk.Frame(self.triggerFrame)
        self.triggerModeVar = tk.StringVar(); self.triggerModeVar.set(self.triggerModes[0])
        self.triggerModeVar.trace('w', self.updateTriggerMode)
        self.triggerModeLabel = ttk.Label(self.triggerModeChooserFrame, text='Trigger mode:')
        self.triggerModeRadioButtons = {}
        self.triggerModeControlGroupFrames = {}

        for mode in self.triggerModes:
            self.triggerModeRadioButtons[mode] = ttk.Radiobutton(self.triggerModeChooserFrame, text=mode, variable=self.triggerModeVar, value=mode)
            self.triggerModeControlGroupFrames[mode] = ttk.Frame(self.triggerFrame)

        # Manual controls
        self.manualWriteTriggerButton = ttk.Button(self.triggerModeControlGroupFrames['Manual'], text="Manual write trigger", command=self.writeButtonClick)

        # Audio trigger controls
        self.triggerHighLevelFrame = ttk.LabelFrame(self.triggerModeControlGroupFrames['Audio'], text="High volume threshold", style='SingleContainer.TLabelframe')
        self.triggerHighLevelVar = tk.StringVar(); self.triggerHighLevelVar.set("0.1")
        self.triggerHighLevelEntry = ttk.Entry(self.triggerHighLevelFrame, textvariable=self.triggerHighLevelVar); self.triggerHighLevelEntry.bind('<FocusOut>', self.updateAudioTriggerSettings)

        self.triggerLowLevelFrame = ttk.LabelFrame(self.triggerModeControlGroupFrames['Audio'], text="Low volume threshold", style='SingleContainer.TLabelframe')
        self.triggerLowLevelVar = tk.StringVar(); self.triggerLowLevelVar.set("0.05")
        self.triggerLowLevelEntry = ttk.Entry(self.triggerLowLevelFrame, textvariable=self.triggerLowLevelVar); self.triggerLowLevelEntry.bind('<FocusOut>', self.updateAudioTriggerSettings)

        self.triggerHighTimeFrame = ttk.LabelFrame(self.triggerModeControlGroupFrames['Audio'], text="High threshold time", style='SingleContainer.TLabelframe')
        self.triggerHighTimeVar = tk.StringVar(); self.triggerHighTimeVar.set("0.5")
        self.triggerHighTimeEntry = ttk.Entry(self.triggerHighTimeFrame, textvariable=self.triggerHighTimeVar); self.triggerHighTimeEntry.bind('<FocusOut>', self.updateAudioTriggerSettings)

        self.triggerLowTimeFrame = ttk.LabelFrame(self.triggerModeControlGroupFrames['Audio'], text="Low threshold time", style='SingleContainer.TLabelframe')
        self.triggerLowTimeVar = tk.StringVar(); self.triggerLowTimeVar.set("2.0")
        self.triggerLowTimeEntry = ttk.Entry(self.triggerLowTimeFrame, textvariable=self.triggerLowTimeVar); self.triggerLowTimeEntry.bind('<FocusOut>', self.updateAudioTriggerSettings)

        self.triggerHighFractionFrame = ttk.LabelFrame(self.triggerModeControlGroupFrames['Audio'], text="Frac. of time above high threshold", style='SingleContainer.TLabelframe')
        self.triggerHighFractionVar = tk.StringVar(); self.triggerHighFractionVar.set("0.1")
        self.triggerHighFractionEntry = ttk.Entry(self.triggerHighFractionFrame, textvariable=self.triggerHighFractionVar); self.triggerHighFractionEntry.bind('<FocusOut>', self.updateAudioTriggerSettings)

        self.triggerLowFractionFrame = ttk.LabelFrame(self.triggerModeControlGroupFrames['Audio'], text="Frac. of time below low threshold", style='SingleContainer.TLabelframe')
        self.triggerLowFractionVar = tk.StringVar(); self.triggerLowFractionVar.set("0.99")
        self.triggerLowFractionEntry = ttk.Entry(self.triggerLowFractionFrame, textvariable=self.triggerLowFractionVar); self.triggerLowFractionEntry.bind('<FocusOut>', self.updateAudioTriggerSettings)

        self.triggerHighBandpassFrame = ttk.LabelFrame(self.triggerModeControlGroupFrames['Audio'], text="High bandpass cutoff freq. (Hz)", style='SingleContainer.TLabelframe')
        self.triggerHighBandpassVar = tk.StringVar(); self.triggerHighBandpassVar.set("7000")
        self.triggerHighBandpassEntry = ttk.Entry(self.triggerHighBandpassFrame, textvariable=self.triggerHighBandpassVar); self.triggerHighBandpassEntry.bind('<FocusOut>', self.updateAudioTriggerSettings)

        self.triggerLowBandpassFrame = ttk.LabelFrame(self.triggerModeControlGroupFrames['Audio'], text="Low bandpass cutoff freq. (Hz)", style='SingleContainer.TLabelframe')
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

        # Audio analysis monitoring widgets
        self.audioAnalysisMonitorFrame = ttk.LabelFrame(self.triggerModeControlGroupFrames['Audio'], text="Audio analysis")
        self.audioAnalysisWidgets = {}
        self.analysisSummaryHistoryChunkLength = 100
        self.audioAnalysisSummaryHistory = deque(maxlen=self.analysisSummaryHistoryChunkLength)
        self.createAudioAnalysisMonitor()

        self.setupInputMonitoringWidgets(camSerials=self.camSerials, audioDAQChannels=self.audioDAQChannels)

        ########### Child process objects #####################

        # Monitoring queues for collecting audio and video data for user monitoring purposes
        self.monitorMasterFrameRate = 15

        # Pointers to processes
        self.videoWriteProcesses = {}
        self.videoAcquireProcesses = {}
        self.audioWriteProcess = None
        self.audioAcquireProcess = None
        self.audioTriggerProcess = None
        self.syncProcess = None
        self.mergeProcess = None
        self.StdoutManager = None

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

        self.profiler =  cProfile.Profile()

        self.update()

#        self.createChildProcesses()
#        self.startChildProcesses()

        # Start automatic updating of video and audio monitors
        self.audioMonitorUpdateJob = None
        self.videoMonitorUpdateJob = None
        self.audioAnalysisMonitorUpdateJob = None
        self.triggerIndicatorUpdateJob = None
        self.autoUpdateVideoMonitors()
        self.autoUpdateAudioMonitors()
        if self.triggerModeVar.get() == "Audio":
            self.autoUpdateAudioAnalysisMonitors()

        self.autoDebugAllJob = None
        self.autoDebugAll()

        self.master.update_idletasks()

    # def createSetting(self, settingName, parent, varType, initialValue, labelText, width=None):
    #     # Creates a set of widgets (label, input widget, variable). Only good for Entry-type inputs
    #     newVar = varType()
    #     newVar.set(initialValue)
    #     newEntry = ttk.Entry(parent, width=width, )
    #     setattr(self, settingName+"Var")

    def cleanupAndExit(self):
        # Cancel automatic update jobs
        self.stopMonitors()
        print("main>> Stopping acquisition")
        self.stopChildProcesses()
        print("main>> Destroying master")
        self.master.destroy()
        self.master.quit()
        print("main>> Everything should be closed now!")

    def setVerbosity(self):
        verbosityOptions = ['0', '1', '2', '3']
        names = [
            'AudioAcquirer verbosity',
            'AudioWriter verbosity',
            'Synchronizer verbosity',
            'AVMerger verbosity',
            'AudioTriggerer verbosity',
            'VideoAcquirer verbosity',
            'VideoWriter verbosity'
        ]
        defaults = [
            str(int(self.audioAcquireVerbose)),
            str(int(self.audioWriteVerbose)),
            str(int(self.syncVerbose)),
            str(int(self.mergeVerbose)),
            str(int(self.audioTriggerVerbose)),
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
            self.videoAcquireVerbose = int(choices['VideoAcquirer verbosity'])
            self.videoWriteVerbose = int(choices['VideoWriter verbosity'])
        self.updateChildProcessVerbosity()

    def updateChildProcessVerbosity(self):
        if self.audioAcquireProcess is not None:
            self.audioAcquireProcess.msgQueue.put((AudioAcquirer.SETPARAMS, {'verbose':self.audioAcquireVerbose}))
        if self.audioWriteProcess is not None:
            self.audioWriteProcess.msgQueue.put((AudioWriter.SETPARAMS, {'verbose':self.audioWriteVerbose}))
        if self.syncProcess is not None:
            self.syncProcess.msgQueue.put((Synchronizer.SETPARAMS, {'verbose':self.syncVerbose}))
        if self.mergeProcess is not None:
            self.mergeProcess.msgQueue.put((AVMerger.SETPARAMS, {'verbose':self.mergeVerbose}))
        if self.audioTriggerProcess is not None:
            self.audioTriggerProcess.msgQueue.put((AudioTriggerer.SETPARAMS, {'verbose':self.audioTriggerVerbose}))
        for camSerial in self.videoAcquireProcesses:
            self.videoAcquireProcesses[camSerial].msgQueue.put((VideoAcquirer.SETPARAMS, {'verbose':self.videoAcquireVerbose}))
            self.videoWriteProcesses[camSerial].msgQueue.put((VideoWriter.SETPARAMS, {'verbose':self.videoWriteVerbose}))

    def showHelpDialog(self, *args):
        msg = 'Sorry, nothing here yet.'
        showinfo('PyVAQ Help', msg)

    def showAboutDialog(self, *args):
        msg = '''Welcome to PyVAQ version {version}!

If it's working perfectly, then contact Brian Kardon (bmk27@cornell.edu) to let \
him know. Otherwise, I had nothing to do with it.

'''.format(version=VERSION)

        showinfo('About PyVAQ', msg)

    def configureAudioMonitoring(self):
        if self.audioMonitor is None:
            showinfo('Please start acquisition before configuring audio monitor')
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
                    self.self.audioMonitor.historyLength = float(choices['Audio history length']) * p['audioFrequency']
                except ValueError:
                    pass

    def selectInputs(self, *args):
        debug = False
        if debug:
            print("main>> GUI DEBUG MODE - using fake cameras and DAQ channels")
            audioDAQChannels = ['fakeDebugAudioChannel1', 'fakeDebugAudioChannel2']
            # availableClockChannels = ['fakeDebugClockChannel1', 'fakeDebugClockChannel2', 'fakeDebugClockChannel3', 'fakeDebugClockChannel4', 'fakeDebugClockChannel5', 'fakeDebugClockChannel6']
            camSerials = ['fakeDebugCam1', 'fakeDebugCam2']
            self.setupInputMonitoringWidgets(camSerials=camSerials, audioDAQChannels=audioDAQChannels)
            return

        availableAudioChannels = flattenList(discoverDAQAudioChannels().values())
        availableClockChannels = flattenList(discoverDAQClockChannels().values()) + ['None']
        availableCamSerials = discoverCameras()

        params = []
        if len(availableAudioChannels) > 0:
            params.append(Param(name='Audio Channels', widgetType=Param.MULTICHOICE, options=availableAudioChannels, default=None))
        if len(availableCamSerials) > 0:
            params.append(Param(name='Cameras', widgetType=Param.MULTICHOICE, options=availableCamSerials, default=None))
        if len(availableClockChannels) > 0:
            params.append(Param(name='Audio Sync Channel', widgetType=Param.MONOCHOICE, options=availableClockChannels, default="None"))
            params.append(Param(name='Video Sync Channel', widgetType=Param.MONOCHOICE, options=availableClockChannels, default="None"))
            params.append(Param(name='Audio Sync PFI Interface', widgetType=Param.TEXT, options=None, default="PFI4", description="This must match your selection for Audio Sync Channel. Check DAQ pinout for matching PFI channel."))
            params.append(Param(name='Video Sync PFI Interface', widgetType=Param.TEXT, options=None, default="PFI5", description="This must match your selection for Video Sync Channel. Check DAQ pinout for matching PFI channel."))
        params.append(Param(name='Start acquisition immediately', widgetType=Param.MONOCHOICE, options=['Yes', 'No'], default='Yes'))

        choices = None
        if len(params) > 0:
            pd = ParamDialog(self.master, params=params, title="Choose audio/video inputs to use")
            choices = pd.results
            if choices is not None:
                self.stopMonitors()
                self.updateAcquisitionButton()
                self.destroyChildProcesses()

                if 'Audio Channels' in choices:
                    audioDAQChannels = choices['Audio Channels']
                else:
                    audioDAQChannels = []
                if 'Cameras' in choices:
                    camSerials = choices['Cameras']
                else:
                    camSerials = []
                if 'Audio Sync Channel' in choices and choices['Audio Sync Channel'] != "None":
                    self.audioSyncTerminal = choices['Audio Sync Channel']
                else:
                    self.audioSyncTerminal = None
                if 'Video Sync Channel' in choices and choices['Video Sync Channel'] != "None":
                    self.videoSyncTerminal = choices['Video Sync Channel']
                else:
                    self.videoSyncTerminal = None
                if 'Audio Sync PFI Interface' in choices and len(choices['Audio Sync PFI Interface']) > 0:
                    self.audioSyncSource = choices['Audio Sync PFI Interface']
                else:
                    self.audioSyncSource = None
                if 'Video Sync PFI Interface' in choices and len(choices['Video Sync PFI Interface']) > 0:
                    self.videoSyncSource = choices['Video Sync PFI Interface']
                else:
                    self.videoSyncSource = None

                print('main>> Got audioDAQChannels:', audioDAQChannels)
                print('main>> Got camSerials:', camSerials)

                self.setupInputMonitoringWidgets(camSerials=camSerials, audioDAQChannels=audioDAQChannels)

                self.createChildProcesses()
                if 'Start acquisition immediately' in choices and choices['Start acquisition immediately'] == 'Yes':
                    self.startChildProcesses()
                self.updateAcquisitionButton()
                self.startMonitors()
            else:
                print('main>> User input cancelled.')
        else:
            showinfo('No inputs', 'No compatible audio/video inputs found. Please connect at least one USB3 vision camera for video input and/or a NI USB DAQ for audio input and synchronization.')

    def setupInputMonitoringWidgets(self, camSerials=None, audioDAQChannels=None):
        # Set up widgets and other entities for specific selected audio and video inputs

        # Destroy old video stream monitoring widgets
        for camSerial in self.camSerials:
            self.cameraMonitors[camSerial].grid_forget()
            self.cameraMonitors[camSerial].destroy()
            del self.cameraMonitors[camSerial]

        self.camSerials = camSerials

        self.cameraSpeeds = dict([(camSerial, checkCameraSpeed(camSerial)) for camSerial in self.camSerials])
        # self.updateAllCamerasAttributes()
        # with open('attributes.txt', 'w') as f:
        #     pp = pprint.PrettyPrinter(stream=f, indent=2)
        #     pp.pprint(self.cameraAttributes)

        # Create new video stream monitoring widgets and other entities
        for camSerial in self.camSerials:
            self.cameraMonitors[camSerial] = CameraMonitor(
                self.videoMonitorMasterFrame,
                displaySize=(400, 300),
                camSerial=camSerial,
                speedText=self.cameraSpeeds[camSerial]
            )
            self.cameraMonitors[camSerial].setDirectoryChangeHandler(self.videoDirectoryChangeHandler)

        self.audioDAQChannels = audioDAQChannels

        # Create new audio stream monitoring widgets
        if self.audioMonitor is None:
            self.audioMonitor = AudioMonitor(self.monitorMasterFrame)
        self.audioMonitor.updateChannels(self.audioDAQChannels)
        self.audioMonitor.setDirectoryChangeHandler(self.audioDirectoryChangeHandler)
        self.update()

    def updateAudioTriggerSettings(self, *args):
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
            # params = dict(
            #     triggerHighLevel=float(self.triggerHighLevelVar.get()),
            #     triggerLowLevel=float(self.triggerLowLevelVar.get()),
            #     triggerHighTime=float(self.triggerHighTimeVar.get()),
            #     triggerLowTime=float(self.triggerLowTimeVar.get()),
            #     triggerHighFraction=float(self.triggerHighFractionVar.get()),
            #     triggerLowFraction=float(self.triggerLowFractionVar.get()),
            #     maxAudioTriggerTime=float(self.maxAudioTriggerTimeVar.get()),
            #     multiChannelStartBehavior=self.multiChannelStartBehaviorVar.get(),
            #     multiChannelStopBehavior=self.multiChannelStopBehaviorVar.get()
            # )
            self.audioTriggerProcess.msgQueue.put((AudioTriggerer.SETPARAMS, params))

    def updateAVMergerState(self, *args):
        merging = self.mergeFilesVar.get()
        if merging:
            self.deleteMergedVideoFilesCheckbutton.config(state=tk.NORMAL)
            self.deleteMergedAudioFilesCheckbutton.config(state=tk.NORMAL)
            self.montageMergeCheckbutton.config(state=tk.NORMAL)
        else:
            self.deleteMergedVideoFilesCheckbutton.config(state=tk.DISABLED)
            self.deleteMergedAudioFilesCheckbutton.config(state=tk.DISABLED)
            self.montageMergeCheckbutton.config(state=tk.DISABLED)

        if self.mergeProcess is not None:
            if merging:
                self.mergeProcess.msgQueue.put((AVMerger.START, None))
            else:
                self.mergeProcess.msgQueue.put((AVMerger.CHILL, None))

    def changeAVMergerParams(self, **params):
        if self.mergeProcess is not None:
            self.mergeProcess.msgQueue.put((AVMerger.SETPARAMS, params))

    def videoDirectoryChangeHandler(self, *args):
        print("main>> videoDirectoryChangeHandler")
        p = self.getParams(
            'videoDirectories',
            'videoBaseFileNames'
            )
        for camSerial in self.videoWriteProcesses:
            if len(p['videoDirectories'][camSerial]) == 0 or os.path.isdir(p['videoDirectories'][camSerial]):
                # Notify VideoWriter child process of new write directory
                print("main>> sending new video write directory: "+p['videoDirectories'][camSerial])
                self.videoWriteProcesses[camSerial].msgQueue.put((VideoWriter.SETPARAMS, dict(videoDirectory=p['videoDirectories'][camSerial])))
    def audioDirectoryChangeHandler(self, *args):
        print("main>> audioDirectoryChangeHandler")
        p = self.getParams(
            'audioDirectory',
            'audioBaseFileName'
            )
        if self.audioWriteProcess is not None:
            if len(p['audioDirectory']) == 0 or os.path.isdir(p['audioDirectory']):
                # Notify AudioWriter child process of new write directory
                print("main>> sending new audio write directory: "+p['audioDirectory'])
                self.audioWriteProcess.msgQueue.put((AudioWriter.SETPARAMS, dict(audioDirectory=p['audioDirectory'])))
    def mergeDirectoryChangeHandler(self, *args):
        print("main>> mergeDirectoryChangeHandler")
        p = self.getParams(
            'mergeDirectory',
            'mergeBaseFileName'
        )
        if self.mergeProcess is not None:
            if len(p['mergeDirectory']) == 0 or os.path.isdir(p['mergeDirectory']):
                # Notify AVMerger child process of new write directory
                print("main>> sending new video write directory: "+p['mergeDirectory'])
                self.mergeProcess.msgQueue.put((AVMerger.SETPARAMS, dict(directory=p['mergeDirectory'])))

    def selectMergedWriteDirectory(self, *args):
        directory = askdirectory(
#            initialdir = ,
#            message = "Choose a directory to write video and audio files to.",
            mustexist = False,
            title = "Choose a directory to write merged audio/video files to."
        )
        if len(directory) > 0:
            self.mergedDirectoryVar.set(directory)
            self.mergedDirectoryEntry.xview_moveto(0.5)
            self.mergedDirectoryEntry.update_idletasks()
            self.mergedDirectoryChangeHandler()

    def updateTriggerMode(self, *args):
        newMode = self.triggerModeVar.get()

        if self.audioAnalysisMonitorUpdateJob is not None:
            self.master.after_cancel(self.audioAnalysisMonitorUpdateJob)

        if newMode == "Audio":
            if self.audioTriggerProcess is not None:
                self.audioTriggerProcess.msgQueue.put((AudioTriggerer.STARTANALYZE, None))
            self.autoUpdateAudioAnalysisMonitors()
        else:
            if self.audioTriggerProcess is not None:
                self.audioTriggerProcess.msgQueue.put((AudioTriggerer.STOPANALYZE, None))

        self.update()

    def createAudioAnalysisMonitor(self):
        # Set up matplotlib axes and plots to display audio analysis data from AudioTriggerer object

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
        xValues = [k for k in range(len(self.audioDAQChannels))]
        zeroChannelValues = [0 for k in range(len(self.audioDAQChannels))]
        oneChannelValues = [1 for k in range(len(self.audioDAQChannels))]
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
        if self.audioMonitorUpdateJob is not None:
            self.master.after_cancel(self.audioMonitorUpdateJob)
        if self.videoMonitorUpdateJob is not None:
            self.master.after_cancel(self.videoMonitorUpdateJob)
        if self.audioAnalysisMonitorUpdateJob is not None:
            self.master.after_cancel(self.audioAnalysisMonitorUpdateJob)
        if self.triggerIndicatorUpdateJob is not None:
            self.master.after_cancel(self.triggerIndicatorUpdateJob)
        if self.autoDebugAllJob is not None:
            self.master.after_cancel(self.autoDebugAllJob)

    def startMonitors(self):
        self.autoUpdateAudioMonitors()
        self.autoUpdateVideoMonitors()
        self.autoUpdateTriggerIndicator()
        self.autoUpdateAudioAnalysisMonitors()

    def autoUpdateAudioAnalysisMonitors(self, beginAuto=True):
        if self.audioTriggerProcess is not None:
            analysisSummary = None
            try:
                while True:
                    analysisSummary = self.audioTriggerProcess.analysisMonitorQueue.get(block=True, timeout=0.01)

                    self.audioAnalysisSummaryHistory.append(analysisSummary)

            except queue.Empty:
                pass

            if analysisSummary is not None:
                # print(analysisSummary)
                lag = time.time_ns()/1000000000 - analysisSummary['chunkStartTime']
                if lag > 1.5:
                    print("main>> WARNING, high analysis monitoring lag:", lag, 's', 'qsize:', self.audioTriggerProcess.analysisMonitorQueue.qsize())

                # Update bar charts using last received analysis summary

                self.audioAnalysisWidgets['volumeFracAxes'].clear()
                self.audioAnalysisWidgets['lowFracBars'] = []
                self.audioAnalysisWidgets['highFracBars'] = []

                for c in range(len(self.audioDAQChannels)):
                    self.audioAnalysisWidgets['lowFracBars'].append(
                        self.audioAnalysisWidgets['volumeFracAxes'].bar(x=c,      width=0.5, bottom=0, height=analysisSummary['lowFrac'][c],  color='r', align='edge')
                        ) # low frac bar
                    self.audioAnalysisWidgets['highFracBars'].append(
                        self.audioAnalysisWidgets['volumeFracAxes'].bar(x=c+0.5, width=0.5, bottom=0,  height=analysisSummary['highFrac'][c], color='g', align='edge')
                        ) # High frac bar

                self.audioAnalysisWidgets['volumeFracAxes'].axis(xmin=0, xmax=len(self.audioDAQChannels), ymin=0, ymax=1)

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
                        print('main>> weird analysis monitoring error:')
                        traceback.print_exc()
                        print('main>> t:', t)
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

    def autoUpdateAudioMonitors(self, beginAuto=True):
        if self.audioAcquireProcess is not None:
            newAudioData = None
            try:
                for chunkCount in range(100):
                    # Get audio data from monitor queue
                    channels, chunkStartTime, audioData = self.audioAcquireProcess.monitorQueue.get(block=True, timeout=0.001)
                    # audioData = np.ones(audioData.shape) * cc
                    # Accumulate all new audio chunks together
                    if newAudioData is not None:
                        newAudioData = np.concatenate((newAudioData, audioData), axis=1)
                    else:
                        newAudioData = audioData
                print("main>> WARNING! Audio monitor is not getting data fast enough to keep up with stream.")
            except queue.Empty:
#                print('main>> exhausted audio monitoring queue, got', chunkCount)
                pass

            if newAudioData is not None:
                self.audioMonitor.addAudioData(newAudioData)

        if beginAuto:
            self.audioMonitorUpdateJob = self.master.after(100, self.autoUpdateAudioMonitors)

    def autoUpdateVideoMonitors(self, beginAuto=True):
        if self.videoAcquireProcesses is not None:
            availableImages = {}
            for camSerial in self.videoAcquireProcesses:
                try:
                    availableImages[camSerial] = self.videoAcquireProcesses[camSerial].monitorImageReceiver.get()
                except queue.Empty:
                    pass

            for camSerial in availableImages:   # Display the most recent available image for each camera
                # pImage = availableImages[camSerial]
                # imData = np.reshape(pImage.data, (pImage.height, pImage.width, 3))
                # im = Image.fromarray(imData)
                self.cameraMonitors[camSerial].updateImage(availableImages[camSerial])

        if beginAuto:
            period = int(round(1000.0/(2*self.monitorMasterFrameRate)))
            self.videoMonitorUpdateJob = self.master.after(period, self.autoUpdateVideoMonitors)

    def autoUpdateTriggerIndicator(self, beginAuto=True):
        if beginAuto:
            self.triggerIndicatorUpdateJob = self.master.after(100, self.autoUpdateTriggerIndicator)

    def createCameraAttributeBrowser(self, camSerial):
        main = tk.Toplevel()
        nb = ttk.Notebook(main)
        nb.grid(row=0)
        tooltipLabel = ttk.Label(main, text="temp")
        tooltipLabel.grid(row=1)

        #self.cameraAttributesWidget[camSerial]
        widgets = self.createAttributeBrowserNode(self.cameraAttributes[camSerial], nb, tooltipLabel, 1)

    def createAttributeBrowserNode(self, attributeNode, parent, tooltipLabel, gridRow):
        frame = ttk.Frame(parent)
        frame.bind("<Enter>", lambda event: tooltipLabel.config(text=attributeNode["tooltip"]))  # Set tooltip rollover callback
        frame.grid(row=gridRow)

        # syncPrint()
        # pp = pprint.PrettyPrinter(indent=1, depth=1)
        # pp.pprint(attributeNode)
        # syncPrint()

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
        system = PySpin.System.GetInstance()
        camList = system.GetCameras()
        for cam in camList:
            cam.Init()

            camSerial = getCameraAttribute(cam.GetTLDeviceNodeMap(), 'DeviceSerialNumber', PySpin.CStringPtr)
            self.updateCameraAttributes(camSerial, cam=cam)

            cam.DeInit()
            del cam
        camList.Clear()
        system.ReleaseInstance()

    def updateCameraAttributes(self, camSerial, cam=None):
        if cam is None:
            getCam = True
        else:
            getCam = False

        if getCam:
            system = PySpin.System.GetInstance()
            camList = system.GetCameras()
            cam = camList.GetBySerial(camSerial)
            cam.Init()

        self.cameraAttributes[camSerial] = getAllCameraAttributes(cam)

        if getCam:
            cam.DeInit()
            del cam
            camList.Clear()
            system.ReleaseInstance()

    def getQueueSizes(self):
        print("main>> Get qsizes...")
        for camSerial in self.videoAcquireProcesses:
            print("main>>   videoMonitorQueues[", camSerial, "] size:", self.videoAcquireProcesses[camSerial].monitorImageReceiver.qsize())
            print("main>>   imageQueues[", camSerial, "] size:", self.videoAcquireProcesses[camSerial].imageQueue.qsize())
        if self.audioAcquireProcess is not None:
            print("main>>   audioAcquireProcess.audioQueue size:", self.audioAcquireProcess.audioQueue.qsize())
            print("main>>   audioAnalysisQueue size:", self.audioAcquireProcess.analysisQueue.qsize())
            print("main>>   audioMonitorQueue size:", self.audioAcquireProcess.monitorQueue.qsize())
        if self.audioTriggerProcess is not None:
            print("main>>   audioAnalysisMonitorQueue size:", self.audioTriggerProcess.analysisMonitorQueue.qsize())
        if self.mergeProcess is not None:
            print("main>>   mergeMessageQueue size:", self.mergeProcess.msgQueue.qsize())
        if self.StdoutManager is not None:
            print("main>>   stdoutQueue size:", self.StdoutManager.queue.qsize())
        print("main>> ...get qsizes")

    def getPIDs(self):
        videoWritePIDs = {}
        videoAcquirePIDs = {}
        audioWritePID = None
        audioAcquirePID = None
        syncPID = None
        mergePID = None

        print("main>> PIDs...")
        print("main>> main thread:", os.getpid())
        for camSerial in self.videoWriteProcesses:
            videoWritePIDs[camSerial] = self.videoWriteProcesses[camSerial].PID.value
            print("main>>   videoWritePIDs["+camSerial+"]:", videoWritePIDs[camSerial])
        for camSerial in self.videoAcquireProcesses:
            videoAcquirePIDs[camSerial] = self.videoAcquireProcesses[camSerial].PID.value
            print("main>>   videoAcquirePIDs["+camSerial+"]:", videoAcquirePIDs[camSerial])
        if self.audioWriteProcess is not None:
            audioWritePID = self.audioWriteProcess.PID.value
            print("main>>   audioWritePID:", audioWritePID)
        if self.audioAcquireProcess is not None:
            audioAcquirePID = self.audioAcquireProcess.PID.value
            print("main>>   audioAcquirePID:", audioAcquirePID)
        if self.syncProcess is not None:
            syncPID = self.syncProcess.PID.value
            print("main>>   syncPID:", syncPID)
        if self.mergeProcess is not None:
            mergePID = self.mergeProcess.PID.value
            print("main>>   mergePID:", mergePID)
        print("main>> ...PIDs:")

    def checkStates(self):
        videoWriteStates = {}
        videoAcquireStates = {}
        audioWriteState = None
        audioAcquireState = None
        syncState = None
        mergeState = None

        print("main>> Check states...")
        for camSerial in self.videoWriteProcesses:
            # print("main>> Getting VideoWriter {camSerial} state...".format(camSerial=camSerial))
            videoWriteStates[camSerial] = VideoWriter.stateList[self.videoWriteProcesses[camSerial].publishedStateVar.value]
            # print("main>> ...one getting VideoWriter {camSerial} state".format(camSerial=camSerial))
        for camSerial in self.videoAcquireProcesses:
            # print("main>> Getting VideoAcquirer {camSerial} state...".format(camSerial=camSerial))
            videoAcquireStates[camSerial] = VideoAcquirer.stateList[self.videoAcquireProcesses[camSerial].publishedStateVar.value]
            # print("main>> ...one getting VideoAcquirer {camSerial} state".format(camSerial=camSerial))
        if self.audioWriteProcess is not None:
            # print("main>> Getting AudioWriter state...")
            audioWriteState = AudioWriter.stateList[self.audioWriteProcess.publishedStateVar.value]
            # print("main>> ...done getting AudioWriter state")
        if self.audioAcquireProcess is not None:
            # print("main>> Getting AudioAcquirer state...")
            audioAcquireState = AudioAcquirer.stateList[self.audioAcquireProcess.publishedStateVar.value]
            # print("main>> ...done getting AudioAcquirer state")
        if self.syncProcess is not None:
            # print("main>> Getting Synchronizer state...")
            syncState = Synchronizer.stateList[self.syncProcess.publishedStateVar.value]
            # print("main>> ...done getting Synchronizer state...")
        if self.mergeProcess is not None:
            # print("main>> Getting AVMerger state...")
            mergeState = AVMerger.stateList[self.mergeProcess.publishedStateVar.value]
            # print("main>> ...done getting AVMerger state")

        for camSerial in videoWriteStates:
            print("main>> videoWriteStates[", camSerial, "]:", videoWriteStates[camSerial])
        for camSerial in videoAcquireStates:
            print("main>> videoAcquireStates[", camSerial, "]:", videoAcquireStates[camSerial])
        print("main>> audioWriteState:", audioWriteState)
        print("main>> audioAcquireState:", audioAcquireState)
        print("main>> syncState:", syncState)
        print("main>> mergeState:", mergeState)
        print("main>> ...check states")

    def debugAll(self):
        print(r"main>> ****************************** \/ \/ DEBUG ALL \/ \/ *******************************")
        self.getQueueSizes()
        self.getPIDs()
        self.checkStates()
        print(r"main>> ****************************** /\ /\ DEBUG ALL /\ /\ *******************************")

    def autoDebugAll(self, *args, interval=3000, startAuto=True):
        self.debugAll()

        if startAuto:
            self.autoDebugAllJob = self.master.after(interval, self.autoDebugAll)

    def acquisitionActive(self):
        # Check if at least one audio or video process is acquiring

        activeAudioStates = [
            AudioAcquirer.INITIALIZING,
            AudioAcquirer.ACQUIRING,
            AudioAcquirer.ACQUIRE_READY
        ]

        activeVideoStates = [
            VideoAcquirer.INITIALIZING,
            VideoAcquirer.ACQUIRING,
            VideoAcquirer.ACQUIRE_READY
        ]

        for camSerial in self.camSerials:
            try:
                state = self.videoAcquireProcesses[camSerial].publishedStateVar.get(block=False)
            except (queue.Full, queue.Empty):
                state = None
            except (AttributeError, KeyError):
                # No state vars set up yet, so no, not acquiring
                state = None
            if state in activeVideoStates:
                return True
        try:
            state = self.audioAcquireProcess.publishedStateVar.get(block=False)
        except (queue.Full, queue.Empty):
            state = None
        except (AttributeError, KeyError):
            # No state vars set up yet, so no, not acquiring
            state = None
        if state in activeAudioStates:
            return True

        return False

    def acquireButtonClick(self):
        if self.acquisitionActive():
            self.stopChildProcesses()
        else:
            self.startChildProcesses()
        # Schedule button update after 100 ms to give child processes a chance to react
        self.master.after(100, self.updateAcquisitionButton)

    def updateAcquisitionButton(self):
        if self.acquisitionActive():
            self.startChildProcessesButton.config(text="Stop acquisition")
        else:
            self.startChildProcessesButton.config(text= "Start acquisition")

    def writeButtonClick(self):
        self.sendWriteTrigger()

    def saveSettings(self, *args, path=None):
        params = self.getParams()
        # datetime.time objects are not serializable, so we have to extract the time
        params["scheduleStart"] = timeToSerializable(params["scheduleStart"])
        params["scheduleStop"] = timeToSerializable(params["scheduleStop"])
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
        if path is None:
            path = askopenfilename(
                title = "Choose a settings file to load.",
                defaultextension = 'pvs',
                initialdir = '.'
            )
        if path is not None and len(path) > 0:
            with open(path, 'r') as f:
                params = json.loads(f.read())
            params["scheduleStart"] = serializableToTime(params["scheduleStart"])
            params["scheduleStop"] = serializableToTime(params["scheduleStop"])
            print("main>> Loaded settings:")
            print(params)
            self.setParams(params)

    def setParams(self, **params):
        self.audioFrequencyVar.set(params['audioFrequency'])
        self.videoFrequencyVar.set(params['videoFrequency'])
        self.exposureTimeVar.set(params['exposureTime'])
        self.preTriggerTimeVar.set(params['preTriggerTime'])
        self.recordTimeVar.set(params['recordTime'])
        self.baseFileNameVar.set(params['baseFileName'])
        self.directoryVar.set(params['directory'])
        self.mergeFilesVar.set(params['mergeFiles'])
        self.deleteMergedAudioFilesVar.set(params['deleteMergedAudioFiles'])
        self.deleteMergedVideoFilesVar.set(params['deleteMergedVideoFiles'])
        self.montageMergeVar.set(params['montageMerge'])
        self.scheduleEnabledVar.set(params['scheduleEnabled'])
        self.scheduleStartVar.set(params['scheduleStart'])
        self.scheduleStopVar.set(params['scheduleStop'])
        self.triggerModeVar.set(params['triggerMode'])
        self.triggerHighLevelVar.set(params['triggerHighLevel'])
        self.triggerLowLevelVar.set(params['triggerLowLevel'])
        self.triggerHighTimeVar.set(params['triggerHighTime'])
        self.triggerLowTimeVar.set(params['triggerLowTime'])
        self.triggerHighFractionVar.set(params['triggerHighFraction'])
        self.triggerLowFractionVar.set(params['triggerLowFraction'])
        self.triggerHighBandpassVar.set(params['triggerHighBandpass'])
        self.triggerLowBandpassVar.set(params['triggerLowBandpass'])
        self.maxAudioTriggerTimeVar.set(params['maxAudioTriggerTime'])
        self.multiChannelStartBehaviorVar.set(params['multiChannelStartBehavior'])
        self.multiChannelStopBehaviorVar.set(params['multiChannelStopBehavior'])
#        params['chunkSize'] = 1000

    def getParams(self, *paramList):
        # Extract parameters from GUI, and calculate a few derived parameters
        params = {}
        getAllParams = (len(paramList) == 0)
        if getAllParams or 'audioFrequency' in paramList: params['audioFrequency'] = int(self.audioFrequencyVar.get())
        if getAllParams or 'videoFrequency' in paramList: params['videoFrequency'] = int(self.videoFrequencyVar.get())
        if getAllParams or 'exposureTime' in paramList: params['exposureTime'] = float(self.exposureTimeVar.get())
        if getAllParams or 'preTriggerTime' in paramList: params['preTriggerTime'] = float(self.preTriggerTimeVar.get())
        if getAllParams or 'recordTime' in paramList: params['recordTime'] = float(self.recordTimeVar.get())
        if getAllParams or 'videoBaseFileNames' in paramList:
            videoBaseFileNames = {}
            for camSerial in self.camSerials:
                videoBaseFileNames[camSerial] = slugify(self.cameraMonitors[camSerial].getBaseFileName() + '_' + camSerial)
            params['videoBaseFileNames'] = videoBaseFileNames
        if getAllParams or 'videoDirectories' in paramList:
            videoDirectories = {}
            for camSerial in self.camSerials:
                videoDirectories[camSerial] = self.cameraMonitors[camSerial].getDirectory()
            params['videoDirectories'] = videoDirectories
        if getAllParams or 'audioBaseFileName' in paramList:
            params["audioBaseFileName"] = slugify(self.audioMonitor.getBaseFileName()+'_'+','.join(self.audioDAQChannels))
        if getAllParams or 'audioDirectory' in paramList: params['audioDirectory'] = self.audioMonitor.getDirectory()
        if getAllParams or 'mergeBaseFileName' in paramList: params['mergeBaseFileName'] = self.mergeFileWidget.getBaseFileName()
        if getAllParams or 'mergeDirectory' in paramList: params['mergeDirectory'] = self.mergeFileWidget.getDirectory()
        if getAllParams or 'mergeFiles' in paramList: params['mergeFiles'] = self.mergeFilesVar.get()
        if getAllParams or 'deleteMergedAudioFiles' in paramList: params['deleteMergedAudioFiles'] = self.deleteMergedAudioFilesVar.get()
        if getAllParams or 'deleteMergedVideoFiles' in paramList: params['deleteMergedVideoFiles'] = self.deleteMergedVideoFilesVar.get()
        if getAllParams or 'montageMerge' in paramList: params['montageMerge'] = self.montageMergeVar.get()
        if getAllParams or 'mergeCompression' in paramList: params['mergeCompression'] = self.mergeCompressionVar.get()
        if getAllParams or 'scheduleEnabled' in paramList: params['scheduleEnabled'] = self.scheduleEnabledVar.get()
        if getAllParams or 'scheduleStart' in paramList: params['scheduleStart'] = self.scheduleStartVar.get()
        if getAllParams or 'scheduleStop' in paramList: params['scheduleStop'] = self.scheduleStopVar.get()
        if getAllParams or 'triggerMode' in paramList: params['triggerMode'] = self.triggerModeVar.get()
        if getAllParams or 'triggerHighLevel' in paramList: params['triggerHighLevel'] = float(self.triggerHighLevelVar.get())
        if getAllParams or 'triggerLowLevel' in paramList: params['triggerLowLevel'] = float(self.triggerLowLevelVar.get())
        if getAllParams or 'triggerHighTime' in paramList: params['triggerHighTime'] = float(self.triggerHighTimeVar.get())
        if getAllParams or 'triggerLowTime' in paramList: params['triggerLowTime'] = float(self.triggerLowTimeVar.get())
        if getAllParams or 'triggerHighFraction' in paramList: params['triggerHighFraction'] = float(self.triggerHighFractionVar.get())
        if getAllParams or 'triggerLowFraction' in paramList: params['triggerLowFraction'] = float(self.triggerLowFractionVar.get())
        if getAllParams or 'triggerHighBandpass' in paramList: params['triggerHighBandpass'] = float(self.triggerHighBandpassVar.get())
        if getAllParams or 'triggerLowBandpass' in paramList: params['triggerLowBandpass'] = float(self.triggerLowBandpassVar.get())
        if getAllParams or 'maxAudioTriggerTime' in paramList: params['maxAudioTriggerTime'] = float(self.maxAudioTriggerTimeVar.get())
        if getAllParams or 'multiChannelStartBehavior' in paramList: params['multiChannelStartBehavior'] = self.multiChannelStartBehaviorVar.get()
        if getAllParams or 'multiChannelStopBehavior' in paramList: params['multiChannelStopBehavior'] = self.multiChannelStopBehaviorVar.get()
        if getAllParams or 'chunkSize' in paramList: params['chunkSize'] = self.chunkSize

        if getAllParams or 'exposureTime' in paramList:
            if params["exposureTime"] >= 1000000 * 0.95/params["videoFrequency"]:
                oldExposureTime = params["exposureTime"]
                params["exposureTime"] = 1000000*0.95/params["videoFrequency"]
                print('main>> ')
                print("main>> ******WARNING*******")
                print('main>> ')
                print("main>> Exposure time is too long to achieve requested frame rate!")
                print("main>> Shortening exposure time from {a}us to {b}us".format(a=oldExposureTime, b=params["exposureTime"]))
                print('main>> ')
                print("main>> ********************")
                print('main>> ')

        if getAllParams or "bufferSizeSeconds" in paramList: params["bufferSizeSeconds"] = params["preTriggerTime"] * 2 + 1   # Twice the pretrigger time to make sure we don't miss stuff, plus one second for good measure

        if getAllParams or "bufferSizeAudioChunks" in paramList: params["bufferSizeAudioChunks"] = params["bufferSizeSeconds"] * params['audioFrequency'] / params["chunkSize"]   # Will be rounded up to nearest integer

        if getAllParams or "numStreams" in paramList: params["numStreams"] = (len(self.audioDAQChannels)>0) + len(self.camSerials)
        if getAllParams or "numProcesses" in paramList: params["numProcesses"] = (len(self.audioDAQChannels)>0) + len(self.camSerials)*2 + 2
        if getAllParams or "numSyncedProcesses" in paramList: params["numSyncedProcesses"] = (len(self.audioDAQChannels)>0) + len(self.camSerials) + 1  # 0 or 1 audio acquire processes, N video acquire processes, and 1 sync process

        if getAllParams or "acquireSettings" in paramList: params["acquireSettings"] = [
            ('AcquisitionMode', 'Continuous', 'enum'),
            ('TriggerMode', 'Off', 'enum'),
            ('TriggerSelector', 'FrameStart', 'enum'),
            ('TriggerSource', 'Line0', 'enum'),
            ('TriggerActivation', 'RisingEdge', 'enum'),
            ('PixelFormat', 'BGR8', 'enum'),
            # ('ExposureMode', 'TriggerWidth'),
            # ('Width', 800, 'integer'),
            # ('Height', 800, 'integer'),
            ('TriggerMode', 'On', 'enum'),
            ('ExposureAuto', 'Off', 'enum'),
            ('ExposureMode', 'Timed', 'enum'),
            ('ExposureTime', params["exposureTime"], 'float')]   # List of attribute/value pairs to be applied to the camera in the given order

        return params

    def createChildProcesses(self):
        print("main>> Creating child processes")
        p = self.getParams()

        ready = mp.Barrier(p["numSyncedProcesses"])

        self.StdoutManager = StdoutManager()
        self.StdoutManager.start()

        # Shared values so all processes can access actual DAQ frequencies
        #   determined by Synchronizer process. This value should only change
        #   once when the Synchronizer is initialized, and not again until
        #   all child processes are stopped and restarted.
        self.actualVideoFrequency = mp.Value('d', -1)
        self.actualAudioFrequency = mp.Value('d', -1)

        startTime = mp.Value('d', -1)

        if p["numStreams"] >= 2:
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


        if self.audioSyncTerminal is not None or self.videoSyncTerminal is not None:
            # Create sync process
            self.syncProcess = Synchronizer(
                actualVideoFrequency=self.actualVideoFrequency,
                actualAudioFrequency=self.actualAudioFrequency,
                startTime=startTime,
                audioSyncChannel=self.audioSyncTerminal,
                videoSyncChannel=self.videoSyncTerminal,
                requestedAudioFrequency=p["audioFrequency"],
                requestedVideoFrequency=p["videoFrequency"],
                verbose=self.syncVerbose,
                ready=ready,
                stdoutQueue=self.StdoutManager.queue)

        if len(self.audioDAQChannels) > 0:
            audioQueue = mp.Queue()
            self.audioAcquireProcess = AudioAcquirer(
                startTime=startTime,
                audioQueue=audioQueue,
                chunkSize=p["chunkSize"],
                audioFrequency=self.actualAudioFrequency,
                bufferSize=None,
                channelNames=self.audioDAQChannels,
                syncChannel=self.audioSyncSource,
                verbose=self.audioAcquireVerbose,
                ready=ready,
                stdoutQueue=self.StdoutManager.queue)
            self.audioWriteProcess = AudioWriter(
                audioDirectory=p["audioDirectory"],
                audioBaseFileName=p["audioBaseFileName"],
                audioQueue=audioQueue,
                mergeMessageQueue=mergeMsgQueue,
                chunkSize=p["chunkSize"],
                bufferSizeSeconds=p["bufferSizeSeconds"],
                audioFrequency=self.actualAudioFrequency,
                numChannels=len(self.audioDAQChannels),
                verbose=self.audioWriteVerbose,
                stdoutQueue=self.StdoutManager.queue)

        for camSerial in self.camSerials:
            processes = {}

            videoAcquireProcess = VideoAcquirer(
                startTime=startTime,
                camSerial=camSerial,
                acquireSettings=p["acquireSettings"],
                frameRate=p["videoFrequency"],
                monitorFrameRate=self.monitorMasterFrameRate,
                verbose=self.videoAcquireVerbose,
                bufferSizeSeconds=p["bufferSizeSeconds"],
                ready=ready,
                stdoutQueue=self.StdoutManager.queue)
            videoWriteProcess = VideoWriter(
                camSerial=camSerial,
                videoDirectory=p["videoDirectories"][camSerial],
                videoBaseFileName=p["videoBaseFileNames"][camSerial],
                imageQueue=videoAcquireProcess.imageQueueReceiver,
                frameRate=p["videoFrequency"],
                mergeMessageQueue=mergeMsgQueue,
                bufferSizeSeconds=p["bufferSizeSeconds"],
                verbose=self.videoWriteVerbose,
                stdoutQueue=self.StdoutManager.queue
                )
            self.videoAcquireProcesses[camSerial] = videoAcquireProcess
            self.videoWriteProcesses[camSerial] = videoWriteProcess

        if self.audioAcquireProcess is not None:
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
                verbose=self.audioTriggerVerbose,
                audioMessageQueue=self.audioWriteProcess.msgQueue,
                videoMessageQueues=dict([(camSerial, self.videoWriteProcesses[camSerial].msgQueue) for camSerial in self.videoWriteProcesses]),
                stdoutQueue=self.StdoutManager.queue
                )

        if len(self.audioDAQChannels) > 0:
            self.audioTriggerProcess.start()
            if self.getParams('triggerMode') == "Audio":
                self.audioTriggerProcess.msgQueue.put((AudioTriggerer.STARTANALYZE, None))
            self.audioWriteProcess.start()
            self.audioAcquireProcess.start()

        for camSerial in self.camSerials:
            self.videoWriteProcesses[camSerial].start()
            self.videoAcquireProcesses[camSerial].start()
        if self.syncProcess is not None: self.syncProcess.start()
        if self.mergeProcess is not None: self.mergeProcess.start()

    def startChildProcesses(self):
        # Tell all child processes to start

        if len(self.audioDAQChannels) > 0:
            # Start audio trigger process
            self.audioTriggerProcess.msgQueue.put((AudioTriggerer.START, None))
            self.updateTriggerMode()

            # Start AudioWriter
            self.audioWriteProcess.msgQueue.put((AudioWriter.START, None))

            # Start AudioAcquirer
            self.audioAcquireProcess.msgQueue.put((AudioAcquirer.START, None))

        # For each camera
        for camSerial in self.camSerials:
            # Start VideoWriter
            self.videoWriteProcesses[camSerial].msgQueue.put((VideoWriter.START, None))
            # Start VideoAcquirer
            self.videoAcquireProcesses[camSerial].msgQueue.put((VideoAcquirer.START, None))

        if len(self.audioDAQChannels) + len(self.camSerials) >= 2:
            # Start sync process
            self.syncProcess.msgQueue.put((Synchronizer.START, None))

            # Start merge process
            self.updateAVMergerState()

    def stopChildProcesses(self):
        # Tell all child processes to stop

        if self.audioTriggerProcess is not None:
            self.audioTriggerProcess.msgQueue.put((AudioTriggerer.STOP, None))
        for camSerial in self.camSerials:
            self.videoAcquireProcesses[camSerial].msgQueue.put((VideoAcquirer.STOP, None))
        if self.audioAcquireProcess is not None:
            self.audioAcquireProcess.msgQueue.put((AudioAcquirer.STOP, None))
        if self.audioWriteProcess is not None:
            self.audioWriteProcess.msgQueue.put((AudioWriter.STOP, None))
        if self.mergeProcess is not None:
            self.mergeProcess.msgQueue.put((AVMerger.STOP, None))
        if self.syncProcess is not None:
            self.syncProcess.msgQueue.put((Synchronizer.STOP, None))

    def exitChildProcesses(self):
        if self.audioTriggerProcess is not None:
            self.audioTriggerProcess.msgQueue.put((AudioTriggerer.EXIT, None))
        for camSerial in self.camSerials:
            self.videoAcquireProcesses[camSerial].msgQueue.put((VideoAcquirer.EXIT, None))
        if self.audioAcquireProcess is not None:
            self.audioAcquireProcess.msgQueue.put((AudioAcquirer.EXIT, None))
        if self.audioWriteProcess is not None:
            self.audioWriteProcess.msgQueue.put((AudioWriter.EXIT, None))
        if self.mergeProcess is not None:
            self.mergeProcess.msgQueue.put((AVMerger.EXIT, None))
        if self.syncProcess is not None:
            self.syncProcess.msgQueue.put((Synchronizer.EXIT, None))
        if self.StdoutManager is not None:
            self.StdoutManager.queue.put(StdoutManager.EXIT)

    def destroyChildProcesses(self):
        self.exitChildProcesses()

        self.actualVideoFrequency = None
        self.actualAudioFrequency = None

        # Give children a chance to register exit message
        time.sleep(0.5)

        try:
            s = io.StringIO()
            ps = pstats.Stats(self.profiler, stream=s)
            ps.print_stats()
            print('main>> ', s.getvalue())
        except:
            print('main>> Error printing profiler stats')

        self.audioTriggerProcess = None
        self.audioAcquireProcess = None
        self.audioWriteProcess = None
        self.videoAcquireProcesses = {}
        self.videoWriteProcesses = {}
        self.mergeProcess = None
        self.syncProcess = None
        self.StdoutManager = None

    def sendWriteTrigger(self, t=None):
        if t is None:
            t = time.time_ns()/1000000000
        trig = Trigger(t-2, t, t+2)
        print("main>> Sending manual trigger!")
        for camSerial in self.camSerials:
            self.videoWriteProcesses[camSerial].msgQueue.put((VideoWriter.TRIGGER, trig))
            print("main>> ...sent to", camSerial, "video writer")
        if self.audioWriteProcess is not None:
            self.audioWriteProcess.msgQueue.put((AudioWriter.TRIGGER, trig))
            print("main>> ...sent to audio writer")

    def update(self):
        # root window
        #   titleBarFrame
        #   mainFrame
        #       monitorFrame
        #           videoMonitorMasterFrame
        #           audioMonitorMasterFrame
        #       controlFrame
        #           acquisitionFrame
        #           triggerFrame
        #               triggerModeChooserFrame
        #               triggerModeControlGroupFrame (only the active one is gridded)
        #       settingsFrame

        if self.customTitleBar:
            self.titleBarFrame.grid(row=0, column=0, sticky=tk.NSEW)
            self.closeButton.grid(sticky=tk.E)
        else:
            self.titleBarFrame.grid_forget()
            self.closeButton.grid_forget()
        self.mainFrame.grid(row=1, column=1)
        self.mainFrame.columnconfigure(0, weight=1)
        self.mainFrame.columnconfigure(1, weight=1)
        self.mainFrame.rowconfigure(0, weight=1)
        self.mainFrame.rowconfigure(1, weight=1)

        self.monitorMasterFrame.grid(row=0, column=0)

        self.videoMonitorMasterFrame.grid(row=0, column=0, sticky=tk.NSEW)
        wV, hV = getOptimalMonitorGrid(len(self.camSerials))
        for k, camSerial in enumerate(self.camSerials):
            self.cameraMonitors[camSerial].grid(row=2*(k // wV), column = k % wV)
            # self.cameraAttributeBrowserButtons[camSerial].grid(row=1, column=0)

        self.audioMonitor.grid(row=1, column=0, sticky=tk.NSEW)

        self.controlFrame.grid(row=0, column=1, sticky=tk.NSEW)
        # self.controlFrame.columnconfigure(0, weight=1)
        # self.controlFrame.columnconfigure(1, weight=1)
        # self.controlFrame.rowconfigure(0, weight=1)
        # self.controlFrame.rowconfigure(1, weight=1)

        self.acquisitionFrame.grid(row=0, column=0, sticky=tk.NSEW)
        # for c in range(3):
        #     self.acquisitionFrame.columnconfigure(c, weight=1)
        # for r in range(4):
        #     self.acquisitionFrame.rowconfigure(r, weight=1)
        self.startChildProcessesButton.grid(row=0, column=0, columnspan=5, sticky=tk.NSEW)
        self.audioFrequencyFrame.grid(row=1, column=0, sticky=tk.EW)
        self.audioFrequencyEntry.grid()
        self.videoFrequencyFrame.grid(row=1, column=1, sticky=tk.EW)
        self.videoFrequencyEntry.grid()
        self.exposureTimeFrame.grid(row=1, column=2, sticky=tk.EW)
        self.exposureTimeEntry.grid()
        self.preTriggerTimeFrame.grid(row=2, column=0, sticky=tk.EW)
        self.preTriggerTimeEntry.grid()
        self.recordTimeFrame.grid(row=2, column=1, sticky=tk.EW)
        self.recordTimeEntry.grid()

        self.updateInputsButton.grid(row=3, column=0)

        self.mergeFrame.grid(row=4, column=0, sticky=tk.NSEW)
        self.mergeFilesCheckbutton.grid(row=1, column=0, sticky=tk.NW)
        self.deleteMergedFilesFrame.grid(row=2, column=0, sticky=tk.NW)
        self.mergeCompressionFrame.grid(row=2, column=1, sticky=tk.NW)
        self.mergeCompression.grid()
        self.deleteMergedAudioFilesCheckbutton.grid(row=0, column=0, sticky=tk.NW)
        self.deleteMergedVideoFilesCheckbutton.grid(row=0, column=1, sticky=tk.NW)
        self.montageMergeCheckbutton.grid(row=3, column=0, sticky=tk.NW)

        self.mergeFileWidget.grid(row=4, column=0, columnspan=2)

        self.scheduleFrame.grid(row=4, column=1, columnspan=2, sticky=tk.NSEW)
        self.scheduleEnabledCheckbutton.grid(row=0, column=0, sticky=tk.NW)
        self.scheduleStartTimeEntry.grid(row=1, column=0, sticky=tk.NW)
        self.scheduleStopTimeEntry.grid(row=2, column=0, sticky=tk.NW)

        self.triggerFrame.grid(row=1, column=0, sticky=tk.NSEW)
        self.triggerModeChooserFrame.grid(row=0, column=0, sticky=tk.NW)
        self.triggerModeLabel.grid(row=0, column=0)
        for k, mode in enumerate(self.triggerModes):
            self.triggerModeRadioButtons[mode].grid(row=0, column=k+1)
            if mode == self.triggerModeVar.get():
                self.triggerModeControlGroupFrames[mode].grid(row=1, column=0)
            else:
                self.triggerModeControlGroupFrames[mode].grid_forget()
        self.manualWriteTriggerButton.grid(row=1, column=0)

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

        self.multiChannelStartBehaviorFrame.grid(row=2, column=0)
        self.multiChannelStartBehaviorOR.grid(row=0)
        self.multiChannelStartBehaviorAND.grid(row=1)

        self.multiChannelStopBehaviorFrame.grid(row=2, column=1)
        self.multiChannelStopBehaviorOR.grid(row=0)
        self.multiChannelStopBehaviorAND.grid(row=1)

        self.maxAudioTriggerTimeFrame.grid(row=2, column=2)
        self.maxAudioTriggerTimeEntry.grid()

        self.audioTriggerStateFrame.grid(row=2, column=3)
        self.audioTriggerStateLabel.grid()
        self.audioAnalysisMonitorFrame.grid(row=4, column=0, columnspan=3)
        self.audioAnalysisWidgets['canvas'].get_tk_widget().pack(side=tk.LEFT, fill=tk.BOTH, expand=1)

def clearQueue(q):
    if q is not None:
        while True:
            try:
                stuff = q.get(block=True, timeout=0.1)
            except queue.Empty:
                break

if __name__ == "__main__":
    root = tk.Tk()
    p = PyVAQ(root)
    root.mainloop()


r'''
cd "C:\Users\Brian Kardon\Dropbox\Documents\Work\Cornell Lab Tech\Projects\Video VI\PyVAQ\Source"
python PyVAQ.py
'''
