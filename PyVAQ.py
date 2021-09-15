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
from StateMachineProcesses import Trigger, StdoutManager, AVMerger, Synchronizer, AudioTriggerer, AudioAcquirer, AudioWriter, VideoAcquirer, VideoWriter, nodeAccessorFunctions, nodeAccessorTypes, ContinuousTriggerer, syncPrint, SimpleVideoWriter
import inspect

VERSION='0.2.0'

# Todo:
#  - Add filename/directory entry for each stream
#  - Find and plug memory leak #  - Add video frameRate indicator
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

class GeneralVar:
    def __init__(self):
        self.value = None
    def get(self):
        return self.value
    def set(self, value):
        self.value = value

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

        self.ID = 'GUI'
        self.stdoutBuffer = []

        # self.audioDAQChannels = []
        # self.audioSyncSource = None
        # self.audioSyncTerminal = None
        # self.videoSyncSource = None
        # self.videoSyncTerminal = None
        # self.acquisitionStartTriggerSource = None
        # self.audioChannelConfiguration = None
        # self.camSerials = []
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
        self.audioSyncSource = GeneralVar(); self.audioSyncSource.set(None)
        self.videoSyncSource = GeneralVar(); self.videoSyncSource.set(None)
        self.acquisitionStartTriggerSource = GeneralVar(); self.acquisitionStartTriggerSource.set(None)
        self.audioChannelConfiguration = GeneralVar(); self.audioChannelConfiguration.set(None)

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
        self.exposureTimeEntry.bind('<FocusOut>', self.validateExposure)

        self.gainFrame =    ttk.LabelFrame(self.acquisitionFrame, text="Gain", style='SingleContainer.TLabelframe')
        self.gainVar =      tk.StringVar(); self.gainVar.set("10")
        self.gainEntry =    ttk.Entry(self.gainFrame, width=18, textvariable=self.gainVar)
        self.gainEntry.bind('<FocusOut>', self.validateGain)

        self.preTriggerTimeFrame =  ttk.LabelFrame(self.acquisitionFrame, text="Pre-trigger record time (s)", style='SingleContainer.TLabelframe')
        self.preTriggerTimeVar =    tk.StringVar(); self.preTriggerTimeVar.set("2.0")
        self.preTriggerTimeEntry =  ttk.Entry(self.preTriggerTimeFrame, width=26, textvariable=self.preTriggerTimeVar)

        self.recordTimeFrame =      ttk.LabelFrame(self.acquisitionFrame, text="Record time (s)", style='SingleContainer.TLabelframe')
        self.recordTimeVar =        tk.StringVar(); self.recordTimeVar.set("4.0")
        self.recordTimeEntry =      ttk.Entry(self.recordTimeFrame, width=14, textvariable=self.recordTimeVar)

        self.chunkSizeVar =         tk.StringVar(); self.chunkSizeVar.set(1000)

        self.updateInputsButton =   ttk.Button(self.acquisitionFrame, text="Select audio/video inputs", command=self.selectInputs)

        self.mergeFrame = ttk.LabelFrame(self.acquisitionFrame, text="AV File merging")

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
        self.mergeCompression = ttk.Combobox(self.mergeCompressionFrame, textvariable=self.mergeCompressionVar, values=[str(k) for k in range(52)], width=12)
        self.mergeCompressionVar.trace('w', lambda *args: self.changeAVMergerParams(compression=self.mergeCompressionVar.get()))

        self.fileSettingsFrame = ttk.LabelFrame(self.acquisitionFrame, text="File settings")
        self.daySubfoldersVar = tk.BooleanVar(); self.daySubfoldersVar.set(True)
        self.daySubfoldersCheckbutton = ttk.Checkbutton(self.fileSettingsFrame, text="File in day subfolders", variable=self.daySubfoldersVar)
        self.daySubfoldersVar.trace('w', lambda *args: self.updateDaySubfolderSetting())

        self.scheduleFrame = ttk.LabelFrame(self.acquisitionFrame, text="Trigger enable schedule")
        self.scheduleEnabledVar = tk.BooleanVar(); self.scheduleEnabledVar.set(False)
        self.scheduleEnabledCheckbutton = ttk.Checkbutton(self.scheduleFrame, text="Restrict trigger to schedule", variable=self.scheduleEnabledVar)
        self.scheduleStartVar = TimeVar()
        self.scheduleStartTimeEntry = TimeEntry(self.scheduleFrame, text="Start time", style=self.style)
        self.scheduleStopVar = TimeVar()
        self.scheduleStopTimeEntry = TimeEntry(self.scheduleFrame, text="Stop time")

        self.triggerFrame = ttk.LabelFrame(self.controlFrame, text='Triggering')
        self.triggerModes = ['Manual', 'Audio', 'Continuous', 'SimpleContinuous']
        self.triggerModeChooserFrame = ttk.Frame(self.triggerFrame)
        self.triggerModeVar = tk.StringVar(); self.triggerModeVar.set(self.triggerModes[0])
        self.triggerModeVar.set('SimpleContinuous')
        self.triggerModeVar.trace('w', self.updateTriggerMode)
        self.triggerModeLabel = ttk.Label(self.triggerModeChooserFrame, text='Trigger mode:')
        self.triggerModeRadioButtons = {}
        self.triggerModeControlGroupFrames = {}

        self.triggerControlTabs = ttk.Notebook(self.triggerFrame)
        for mode in self.triggerModes:
            self.triggerModeRadioButtons[mode] = ttk.Radiobutton(self.triggerModeChooserFrame, text=mode, variable=self.triggerModeVar, value=mode)
            self.triggerModeControlGroupFrames[mode] = ttk.Frame(self.triggerControlTabs)
            self.triggerControlTabs.add(self.triggerModeControlGroupFrames[mode], text=mode)

        # Manual controls
        self.manualWriteTriggerButton = ttk.Button(self.triggerModeControlGroupFrames['Manual'], text="Manual write trigger", command=self.writeButtonClick)

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

        self.profiler =  cProfile.Profile()

        # the params dict defines how to access and set all the parameters in the GUI
        self.paramInfo = {
            'audioFrequency':                   dict(get=lambda:int(self.audioFrequencyVar.get()),              set=self.audioFrequencyVar.set),
            'videoFrequency':                   dict(get=lambda:int(self.videoFrequencyVar.get()),              set=self.videoFrequencyVar.set),
            'chunkSize':                        dict(get=lambda:int(self.chunkSizeVar.get()),                   set=self.chunkSizeVar.set),
            'exposureTime':                     dict(get=lambda:int(self.exposureTimeVar.get()),                set=self.exposureTimeVar.set),
            'gain':                             dict(get=lambda:float(self.gainVar.get()),                      set=self.gainVar.set),
            'preTriggerTime':                   dict(get=lambda:float(self.preTriggerTimeVar.get()),            set=self.preTriggerTimeVar.set),
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
            "acquireSettings":                  dict(get=self.getAcquireSettings,                               set=self.setAcquireSettings),
            "continuousTriggerPeriod":          dict(get=lambda:float(self.continuousTriggerPeriodVar.get()),   set=self.continuousTriggerPeriodVar.set),
            "audioTagContinuousTrigs":          dict(get=self.audioTagContinuousTrigsVar.get,                   set=self.audioTagContinuousTrigsVar.set),
            "daySubfolders":                    dict(get=self.daySubfoldersVar.get,                             set=self.daySubfoldersVar.set),
            "audioDAQChannels":                 dict(get=self.audioDAQChannels.get,                             set=self.audioDAQChannels.set),
            "camSerials":                       dict(get=self.camSerials.get,                                   set=self.camSerials.set),
            "audioSyncTerminal":                dict(get=self.audioSyncTerminal.get,                            set=self.audioSyncTerminal.set),
            "videoSyncTerminal":                dict(get=self.videoSyncTerminal.get,                            set=self.videoSyncTerminal.set),
            "audioSyncSource":                  dict(get=self.audioSyncSource.get,                              set=self.audioSyncSource.set),
            "videoSyncSource":                  dict(get=self.videoSyncSource.get,                              set=self.videoSyncSource.set),
            "acquisitionStartTriggerSource":    dict(get=self.acquisitionStartTriggerSource.get,                set=self.acquisitionStartTriggerSource.set),
            "audioChannelConfiguration":        dict(get=self.audioChannelConfiguration.get,                    set=self.audioChannelConfiguration.set),
        }

        self.createAudioAnalysisMonitor()

        self.setupInputMonitoringWidgets()

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

        self.autoDebugAllJob = None
        self.autoDebugAll()

        self.master.update_idletasks()

    # def createSetting(self, settingName, parent, varType, initialValue, labelText, width=None):
    #     # Creates a set of widgets (label, input widget, variable). Only good for Entry-type inputs
    #     newVar = varType()
    #     newVar.set(initialValue)
    #     newEntry = ttk.Entry(parent, width=width, )
    #     setattr(self, settingName+"Var")

    def log(self, msg, *args, **kwargs):
        # Add another message to the currently accumulating log entry
        syncPrint('|| {ID} - {msg}'.format(ID=self.ID, msg=msg), *args, buffer=self.stdoutBuffer, **kwargs)

    def endLog(self, state):
        # Output accumulated log entry and output it
        if len(self.stdoutBuffer) > 0:
            self.log(r'*********************************** /\ {ID} {state} /\ ********************************************'.format(ID=self.ID, state=state))
            self.flushStdout()

    def flushStdout(self):
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
        # Attempt to gracefully shut everything down and exit
        # Cancel automatic update jobs
        self.stopMonitors()
        self.log("Stopping acquisition")
        self.stopChildProcesses()
        self.log("Destroying master")
        self.master.destroy()
        self.master.quit()
        self.log("Everything should be closed now!")
        self.endLog(inspect.currentframe().f_code.co_name)

    def setVerbosity(self):
        # Produce popup for setting logging verbosity, then update all process
        #   verbosity based on user selections.
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
        # Update child process logging verbosity based on currently stored settings
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
        if self.continuousTriggerProcess is not None:
            self.continuousTriggerProcess.msgQueue.put((ContinuousTriggerer.SETPARAMS, {'verbose':self.continuousTriggerVerbose}))
        for camSerial in self.videoAcquireProcesses:
            self.videoAcquireProcesses[camSerial].msgQueue.put((VideoAcquirer.SETPARAMS, {'verbose':self.videoAcquireVerbose}))
            self.videoWriteProcesses[camSerial].msgQueue.put((VideoWriter.SETPARAMS, {'verbose':self.videoWriteVerbose}))

    def updateDaySubfolderSetting(self, *args):
        # Change day subfolder setting in all child processes
        daySubfolders = self.getParams('daySubfolders')
        if self.audioWriteProcess is not None:
            self.audioWriteProcess.msgQueue.put((AudioWriter.SETPARAMS, {'daySubfolders':daySubfolders}))
        if self.mergeProcess is not None:
            self.mergeProcess.msgQueue.put((AVMerger.SETPARAMS, {'daySubfolders':daySubfolders}))
        for camSerial in self.videoWriteProcesses:
            self.videoWriteProcesses[camSerial].msgQueue.put((VideoWriter.SETPARAMS, {'daySubfolders':daySubfolders}))

    def validateExposure(self, *args):
        # Sanitize current exposure settings (make sure it's an integer in a
        #   reasonable range)
        exposureTime = self.getParams('exposureTime')
        videoFrequency = self.getParams('videoFrequency')
        maxExposureTime = 1000000 * 0.95 / videoFrequency
        if exposureTime > maxExposureTime:
            exposureTime = maxExposureTime
        exposureTime = int(exposureTime)
        self.setParams(exposureTime=exposureTime)

    def validateGain(self, *args):
        # Sanitize current gain settings (make sure it's >= 0)
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
        msg = 'Sorry, nothing here yet.'
        showinfo('PyVAQ Help', msg)

    def showAboutDialog(self, *args):
        msg = '''Welcome to PyVAQ version {version}!

If it's working perfectly, then contact Brian Kardon (bmk27@cornell.edu) to let \
him know. Otherwise, I had nothing to do with it.

'''.format(version=VERSION)

        showinfo('About PyVAQ', msg)

    def configureAudioMonitoring(self):
        # Show popup for user to select audio monitoring options
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
        # Create a popup for the user to select acquisition options

        # debug = False
        # if debug:
        #     self.log("GUI DEBUG MODE - using fake cameras and DAQ channels")
        #     audioDAQChannels = ['fakeDebugAudioChannel1', 'fakeDebugAudioChannel2']
        #     # availableClockChannels = ['fakeDebugClockChannel1', 'fakeDebugClockChannel2', 'fakeDebugClockChannel3', 'fakeDebugClockChannel4', 'fakeDebugClockChannel5', 'fakeDebugClockChannel6']
        #     camSerials = ['fakeDebugCam1', 'fakeDebugCam2']
        #     self.setupInputMonitoringWidgets(camSerials=camSerials, audioDAQChannels=audioDAQChannels)
        #     return

        # Get current settings to use as defaults
        p = self.getParams(
            "audioDAQChannels",
            "camSerials",
            "audioSyncTerminal",
            "videoSyncTerminal",
            "audioSyncSource",
            "videoSyncSource",
            "acquisitionStartTriggerSource",
            "audioChannelConfiguration"
            )

        defaultAudioDAQChannels = p["audioDAQChannels"]
        defaultCamSerials = p["camSerials"]
        defaultAudioSyncTerminal = p["audioSyncTerminal"]
        defaultVideoSyncTerminal = p["videoSyncTerminal"]
        defaultAudioSyncSource = p["audioSyncSource"]
        defaultVideoSyncSource = p["videoSyncSource"]
        defaultAcquisitionStartTriggerSource = p["acquisitionStartTriggerSource"]
        defaultAudioChannelConfiguration = p["audioChannelConfiguration"]

        # Query the system to determine what DAQ channels and cameras are
        #   currently available
        availableAudioChannels = flattenList(discoverDAQAudioChannels().values())
        availableClockChannels = flattenList(discoverDAQClockChannels().values()) + ['None']
        availableDigitalChannels = ['None'] + flattenList(discoverDAQTerminals().values())
        availableCamSerials = discoverCameras()
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
        params.append(Param(name='Acquisition start trigger channel', widgetType=Param.MONOCHOICE, options=availableDigitalChannels, default=defaultAcquisitionStartTriggerSource, description="Choose a channel that will trigger the acquisition start with a rising edge. Leave as None if you wish the acquisition to start without waiting for a digital trigger."))
        params.append(Param(name='Start acquisition immediately', widgetType=Param.MONOCHOICE, options=['Yes', 'No'], default='Yes'))

        choices = None
        if len(params) > 0:
            pd = ParamDialog(self.master, params=params, title="Choose audio/video inputs to use", maxHeight=24)
            choices = pd.results
            if choices is not None:
                # We're changing acquisition settings, so stop everything
                self.stopMonitors()
                self.updateAcquisitionButton()
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
                if 'Acquisition start trigger channel' in choices and len(choices['Acquisition start trigger channel']) > 0:
                    if choices['Acquisition start trigger channel'] == 'None':
                        acquisitionStartTriggerSource = None
                    else:
                        acquisitionStartTriggerSource = choices['Acquisition start trigger channel']
                else:
                    acquisitionStartTriggerSource = None
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
                    acquisitionStartTriggerSource=acquisitionStartTriggerSource,
                    audioChannelConfiguration=audioChannelConfiguration
                    )

                self.log('Got audioDAQChannels:', audioDAQChannels)
                self.log('Got camSerials:', camSerials)

                # Create GUI elements for monitoring the chosen inputs
                self.setupInputMonitoringWidgets()

                # Restart child processes with new acquisition values
                self.createChildProcesses()
                if 'Start acquisition immediately' in choices and choices['Start acquisition immediately'] == 'Yes':
                    self.startChildProcesses()
                self.updateAcquisitionButton()
                self.startMonitors()
            else:
                self.log('User input cancelled.')
        else:
            showinfo('No inputs', 'No compatible audio/video inputs found. Please connect at least one USB3 vision camera for video input and/or a NI USB DAQ for audio input and synchronization.')

        self.endLog(inspect.currentframe().f_code.co_name)

    def setupInputMonitoringWidgets(self):
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
        oldCamSerials = self.cameraMonitors.keys()
        for camSerial in oldCamSerials:
            self.cameraMonitors[camSerial].grid_forget()
            self.cameraMonitors[camSerial].destroy()
            del self.cameraMonitors[camSerial]

        self.cameraSpeeds = dict([(camSerial, checkCameraSpeed(camSerial)) for camSerial in camSerials])
        # self.updateAllCamerasAttributes()
        # with open('attributes.txt', 'w') as f:
        #     pp = pprint.PrettyPrinter(stream=f, indent=2)
        #     pp.pprint(self.cameraAttributes)

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
                displaySize=(400, 300),
                camSerial=camSerial,
                speedText=self.cameraSpeeds[camSerial],
                initialDirectory=videoDirectory,
                initialBaseFileName=videoBaseFileName
            )
            self.cameraMonitors[camSerial].setDirectoryChangeHandler(self.videoDirectoryChangeHandler)
            self.cameraMonitors[camSerial].setBaseFileNameChangeHandler(self.videoBaseFileNameChangeHandler)

        # Create new audio stream monitoring widgets
        if self.audioMonitor is None:
            self.audioMonitor = AudioMonitor(
                self.monitorMasterFrame,
                initialDirectory=audioDirectory,
                initialBaseFileName=audioBaseFileName
                )
        self.audioMonitor.updateChannels(audioDAQChannels)
        self.audioMonitor.setDirectoryChangeHandler(self.audioDirectoryChangeHandler)
        self.audioMonitor.setBaseFileNameChangeHandler(self.audioBaseFileNameChangeHandler)
        self.update()

    def updateAudioTriggerSettings(self, *args):
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
            self.audioTriggerProcess.msgQueue.put((AudioTriggerer.SETPARAMS, params))

    def updateContinuousTriggerSettings(self, *args):
        # Update settings for continuous triggering (sending consecutive
        #   triggers, one after another)
        if self.continuousTriggerProcess is not None:
            paramList = [
                'continuousTriggerPeriod',
            ]
            params = self.getParams(*paramList, mapping=True)
            self.continuousTriggerProcess.msgQueue.put((ContinuousTriggerer.SETPARAMS, params))

            if self.audioTriggerProcess is not None:
                if self.getParams('audioTagContinuousTrigs'):
                    # Tell audio triggerer to start analyzing and send any tag triggers.
                    self.audioTriggerProcess.msgQueue.put((AudioTriggerer.STARTANALYZE, None))
                    self.audioTriggerProcess.msgQueue.put((AudioTriggerer.SETPARAMS, dict(tagTriggerEnabled=True)))
                else:
                    # Tell audio triggerer to stop analyzing and don't send any tag triggers.
                    self.audioTriggerProcess.msgQueue.put((AudioTriggerer.STOPANALYZE, None))
                    self.audioTriggerProcess.msgQueue.put((AudioTriggerer.SETPARAMS, dict(tagTriggerEnabled=False)))
            else:
                self.log('Warning, audio trigger process not available for continuous trigger tagging')
                self.endLog(inspect.currentframe().f_code.co_name)

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

    def videoBaseFileNameChangeHandler(self, *args):
        videoBaseFileNames = {}
        for camSerial in self.cameraMonitors:
            videoBaseFileNames[camSerial] = self.cameraMonitors[camSerial].getBaseFileName()
        self.setVideoBaseFileNames(videoBaseFileNames, updateTextField=False)
    def videoDirectoryChangeHandler(self, *args):
        videoDirectories = {}
        for camSerial in self.cameraMonitors:
            videoDirectories[camSerial] = self.cameraMonitors[camSerial].getDirectory()
        self.setVideoDirectories(videoDirectories, updateTextField=False)
    def audioBaseFileNameChangeHandler(self, *args):
        newAudioBaseFileName = self.audioMonitor.getBaseFileName()
        self.setAudioBaseFileName(newAudioBaseFileName, updateTextField=False)
    def audioDirectoryChangeHandler(self, *args):
        newAudioDirectory = self.audioMonitor.getDirectory()
        self.setAudioDirectory(newAudioDirectory, updateTextField=False)
    def mergeBaseFileNameChangeHandler(self, *args):
        newMergeBaseFileName = self.mergeFileWidget.getBaseFileName()
        self.setMergeBaseFileName(newMergeBaseFileName, updateTextField=False)
    def mergeDirectoryChangeHandler(self, *args):
        newMergeDirectory = self.mergeFileWidget.getDirectory()
        self.setMergeDirectory(newMergeDirectory, updateTextField=False)

    def updateTriggerMode(self, *args):
        # Handle a user selection of a new trigger mode
        newMode = self.triggerModeVar.get()

        if newMode != "Continuous":
            if self.continuousTriggerProcess is not None and self.continuousTriggerProcess.msgQueue is not None:
                self.continuousTriggerProcess.msgQueue.put((ContinuousTriggerer.STOP, None))
        if newMode != "Audio":
            # May as well stop analyzing audio if we're not in audio mode.
            if self.audioTriggerProcess is not None and self.audioTriggerProcess.msgQueue is not None:
                self.audioTriggerProcess.msgQueue.put((AudioTriggerer.STOPANALYZE, None))

        if self.audioAnalysisMonitorUpdateJob is not None:
            # If there was already an audio analysis monitoring job running, cancel it
            self.master.after_cancel(self.audioAnalysisMonitorUpdateJob)

        if newMode == "Audio":
            # User selected "Audio" trigger mode
            if self.audioTriggerProcess is not None:
                self.audioTriggerProcess.msgQueue.put((AudioTriggerer.STARTANALYZE, None))
                self.audioTriggerProcess.msgQueue.put((AudioTriggerer.SETPARAMS, dict(writeTriggerEnabled=True, tagTriggerEnabled=False)))
            self.autoUpdateAudioAnalysisMonitors()
        elif newMode == "Continuous":
            # User selected "Continuous" trigger mode
            self.continuousTriggerProcess.msgQueue.put((AudioTriggerer.START, None))
            if self.audioTriggerProcess is not None:
                if self.getParams('audioTagContinuousTrigs'):
                    self.audioTriggerProcess.msgQueue.put((AudioTriggerer.STARTANALYZE, None))
                    self.audioTriggerProcess.msgQueue.put((AudioTriggerer.SETPARAMS, dict(writeTriggerEnabled=False, tagTriggerEnabled=True)))
                else:
                    self.audioTriggerProcess.msgQueue.put((AudioTriggerer.STOPANALYZE, None))
                    self.audioTriggerProcess.msgQueue.put((AudioTriggerer.SETPARAMS, dict(writeTriggerEnabled=False, tagTriggerEnabled=False)))
        elif newMode == "SimpleContinuous":
            # User selected "SimpleContinuous" trigger mode
            self.restartAcquisition()

        self.update()

    def createAudioAnalysisMonitor(self):
        # Set up matplotlib axes and plots to display audio analysis data from AudioTriggerer object

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
                self.log("WARNING! Audio monitor is not getting data fast enough to keep up with stream.")
            except queue.Empty:
#                self.log('exhausted audio monitoring queue, got', chunkCount)
                pass

            if newAudioData is not None:
                self.audioMonitor.addAudioData(newAudioData)

        if beginAuto:
            self.audioMonitorUpdateJob = self.master.after(100, self.autoUpdateAudioMonitors)

        self.endLog(inspect.currentframe().f_code.co_name)

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
        self.log("Get qsizes...")
        for camSerial in self.videoAcquireProcesses:
            self.log("  videoMonitorQueues[", camSerial, "] size:", self.videoAcquireProcesses[camSerial].monitorImageReceiver.qsize())
            self.log("  imageQueues[", camSerial, "] size:", self.videoAcquireProcesses[camSerial].imageQueue.qsize())
        if self.audioAcquireProcess is not None:
            self.log("  audioAcquireProcess.audioQueue size:", self.audioAcquireProcess.audioQueue.qsize())
            self.log("  audioAnalysisQueue size:", self.audioAcquireProcess.analysisQueue.qsize())
            self.log("  audioMonitorQueue size:", self.audioAcquireProcess.monitorQueue.qsize())
        if self.audioTriggerProcess is not None:
            self.log("  audioAnalysisMonitorQueue size:", self.audioTriggerProcess.analysisMonitorQueue.qsize())
        if self.mergeProcess is not None:
            self.log("  mergeMessageQueue size:", self.mergeProcess.msgQueue.qsize())
        if self.StdoutManager is not None:
            self.log("  stdoutQueue size:", self.StdoutManager.queue.qsize())
        self.log("...get qsizes")
        self.endLog(inspect.currentframe().f_code.co_name)

    def getPIDs(self):
        videoWritePIDs = {}
        videoAcquirePIDs = {}
        audioWritePID = None
        audioAcquirePID = None
        audioTriggerPID = None
        continuousTriggerPID = None
        syncPID = None
        mergePID = None

        self.log("PIDs...")
        self.log("main thread:", os.getpid())
        for camSerial in self.videoWriteProcesses:
            videoWritePIDs[camSerial] = self.videoWriteProcesses[camSerial].PID.value
            self.log("  videoWritePIDs["+camSerial+"]:", videoWritePIDs[camSerial])
        for camSerial in self.videoAcquireProcesses:
            videoAcquirePIDs[camSerial] = self.videoAcquireProcesses[camSerial].PID.value
            self.log("  videoAcquirePIDs["+camSerial+"]:", videoAcquirePIDs[camSerial])
        if self.audioWriteProcess is not None:
            audioWritePID = self.audioWriteProcess.PID.value
            self.log("  audioWritePID:", audioWritePID)
        if self.audioAcquireProcess is not None:
            audioAcquirePID = self.audioAcquireProcess.PID.value
            self.log("  audioAcquirePID:", audioAcquirePID)
        if self.audioTriggerProcess is not None:
            audioTriggerPID = self.audioTriggerProcess.PID.value
            self.log("  audioTriggerPID:", audioTriggerPID)
        if self.continuousTriggerProcess is not None:
            continuousTriggerPID = self.continuousTriggerProcess.PID.value
            self.log("  continuousTriggerPID:", continuousTriggerPID)
        if self.syncProcess is not None:
            syncPID = self.syncProcess.PID.value
            self.log("  syncPID:", syncPID)
        if self.mergeProcess is not None:
            mergePID = self.mergeProcess.PID.value
            self.log("  mergePID:", mergePID)
        self.log("...PIDs:")
        self.endLog(inspect.currentframe().f_code.co_name)

    def checkStates(self, verbose=True):
        states = dict(
            videoWriteStates = {},
            videoAcquireStates = {},
            audioWriteState = None,
            audioAcquireState = None,
            syncState = None,
            mergeState = None
        )

        self.log("Check states...")
        for camSerial in self.videoWriteProcesses:
            # self.log("Getting VideoWriter {camSerial} state...".format(camSerial=camSerial))
            states['videoWriteStates'][camSerial] = VideoWriter.stateList[self.videoWriteProcesses[camSerial].publishedStateVar.value]
            # self.log("...done getting VideoWriter {camSerial} state".format(camSerial=camSerial))
        for camSerial in self.videoAcquireProcesses:
            # self.log("Getting VideoAcquirer {camSerial} state...".format(camSerial=camSerial))
            states['videoAcquireStates'][camSerial] = VideoAcquirer.stateList[self.videoAcquireProcesses[camSerial].publishedStateVar.value]
            # self.log("...done getting VideoAcquirer {camSerial} state".format(camSerial=camSerial))
        if self.audioWriteProcess is not None:
            # self.log("Getting AudioWriter state...")
            states['audioWriteState'] = AudioWriter.stateList[self.audioWriteProcess.publishedStateVar.value]
            # self.log("...done getting AudioWriter state")
        else:
            states['audioWriteState'] = 'None'
        if self.audioAcquireProcess is not None:
            # self.log("Getting AudioAcquirer state...")
            states['audioAcquireState'] = AudioAcquirer.stateList[self.audioAcquireProcess.publishedStateVar.value]
            # self.log("...done getting AudioAcquirer state")
        else:
            states['audioAcquireState'] = 'None'
        if self.syncProcess is not None:
            # self.log("Getting Synchronizer state...")
            states['syncState'] = Synchronizer.stateList[self.syncProcess.publishedStateVar.value]
            # self.log("...done getting Synchronizer state...")
        else:
            states['syncState'] = 'None'
        if self.mergeProcess is not None:
            # self.log("Getting AVMerger state...")
            states['mergeState'] = AVMerger.stateList[self.mergeProcess.publishedStateVar.value]
            # self.log("...done getting AVMerger state")
        else:
            states['mergeState'] = 'None'
        if self.audioTriggerProcess is not None:
            # self.log("Getting AudioTriggerer state...")
            states['audioTriggerState'] = AudioTriggerer.stateList[self.audioTriggerProcess.publishedStateVar.value]
            # self.log("...done getting AVMerger state")
        else:
            states['audioTriggerState'] = 'None'
        if self.continuousTriggerProcess is not None:
            # self.log("Getting AudioTriggerer state...")
            states['continuousTriggerState'] = ContinuousTriggerer.stateList[self.continuousTriggerProcess.publishedStateVar.value]
            # self.log("...done getting AVMerger state")
        else:
            states['continuousTriggerState'] = 'None'

        if verbose:
            for camSerial in states['videoWriteStates']:
                self.log("videoWriteStates[", camSerial, "]:", states['videoWriteStates'][camSerial])
            for camSerial in states['videoAcquireStates']:
                self.log("videoAcquireStates[", camSerial, "]:", states['videoAcquireStates'][camSerial])
            self.log("audioWriteState:", states['audioWriteState'])
            self.log("audioAcquireState:", states['audioAcquireState'])
            self.log("syncState:", states['syncState'])
            self.log("mergeState:", states['mergeState'])
            self.log("audioTriggerState:", states['audioTriggerState'])
            self.log("continuousTriggerState:", states['continuousTriggerState'])
            self.log("...check states")

        self.endLog(inspect.currentframe().f_code.co_name)

        return states

    def debugAll(self):
#        self.log(r"main>> ****************************** \/ \/ DEBUG ALL \/ \/ *******************************")
        self.getQueueSizes()
        self.getPIDs()
        self.checkStates()
#        self.log(r"main>> ****************************** /\ /\ DEBUG ALL /\ /\ *******************************")

    def autoDebugAll(self, *args, interval=10000, startAuto=True):
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
        for camSerial in self.videoAcquireProcesses:
            state = self.videoAcquireProcesses[camSerial].publishedStateVar.value
            if state in activeVideoStates:
                return True
        if self.audioAcquireProcess is not None:
            state = self.audioAcquireProcess.publishedStateVar.value
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

    def continuousTriggerStartButtonClick(self):
        if self.continuousTriggerProcess is not None:
            self.log("Sending start signal to continuous trigger process")
            self.continuousTriggerProcess.msgQueue.put((ContinuousTriggerer.START, None))
            self.endLog(inspect.currentframe().f_code.co_name)
        else:
            showwarning(title="No continuous trigger process available", message="Continuous triggering process does not appear to be available. Try starting up acquisition first")

    def continuousTriggerStopButtonClick(self):
        if self.continuousTriggerProcess is not None:
            self.log("Sending stop signal to continuous trigger process")
            self.continuousTriggerProcess.msgQueue.put((ContinuousTriggerer.STOP, None))
            self.endLog(inspect.currentframe().f_code.co_name)
        else:
            showwarning(title="No continuous trigger process available", message="Continuous triggering process does not appear to be available. Try starting up acquisition first")

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
            self.log("Loaded settings:")
            self.log(params)
            self.setParams(**params)
        self.endLog(inspect.currentframe().f_code.co_name)

    def setVideoBaseFileNames(self, newVideoBaseFileNames, *args, updateTextField=True):
        # Expects videoBaseFileNames to be a dictionary of camserial:videoBaseFileNames, which will be used
        #   to assign base filenames to the cameras.
        self.videoBaseFileNames.set(newVideoBaseFileNames)
        for camSerial in self.cameraMonitors:
            if camSerial in newVideoBaseFileNames:
                newVideoBaseFileName = newVideoBaseFileNames[camSerial]
                if updateTextField:
                    # Update text field
                    self.cameraMonitors[camSerial].fileWidget.setBaseFileName(newVideoBaseFileName)
                if camSerial in self.videoWriteProcesses:
                    # Notify VideoWriter child process of new write base filename
                    self.videoWriteProcesses[camSerial].msgQueue.put((VideoWriter.SETPARAMS, dict(videoBaseFileName=newVideoBaseFileName)))
    def setVideoDirectories(self, newVideoDirectories, *args, updateTextField=True):
        # See setVideoBaseFileNames for notes
        self.videoDirectories.set(newVideoDirectories)
        for camSerial in self.cameraMonitors:
            if camSerial in newVideoDirectories:
                newVideoDirectory = newVideoDirectories[camSerial]
                if updateTextField:
                    # Update text field
                    self.cameraMonitors[camSerial].fileWidget.setDirectory(newVideoDirectory)
                if camSerial in self.videoWriteProcesses and (len(newVideoDirectory) == 0 or os.path.isdir(newVideoDirectory)):
                    # Notify VideoWriter child process of new write directory
                    self.videoWriteProcesses[camSerial].msgQueue.put((VideoWriter.SETPARAMS, dict(videoDirectory=newVideoDirectory)))
    def setAudioBaseFileName(self, audioBaseFileName, *args, updateTextField=True):
        self.audioBaseFileName.set(audioBaseFileName)
        if updateTextField and self.audioMonitor is not None:
            # Update text field
            self.audioMonitor.fileWidget.setBaseFileName(audioBaseFileName)
        if self.audioWriteProcess is not None:
            # Notify AudioWriter child process of new write base filename
            self.audioWriteProcess.msgQueue.put((AudioWriter.SETPARAMS, dict(audioBaseFileName=audioBaseFileName)))
    def setAudioDirectory(self, audioDirectory, *args, updateTextField=True):
        self.audioDirectory.set(audioDirectory)
        if updateTextField and self.audioMonitor is not None:
            # Update text field
            self.audioMonitor.fileWidget.setDirectory(audioDirectory)
        if self.audioWriteProcess is not None and (len(audioDirectory) == 0 or os.path.isdir(audioDirectory)):
            # Notify AudioWriter child process of new write directory
            self.audioWriteProcess.msgQueue.put((AudioWriter.SETPARAMS, dict(audioDirectory=audioDirectory)))
    def setMergeBaseFileName(self, mergeBaseFileName, *args, updateTextField=True):
        self.mergeBaseFileName.set(mergeBaseFileName)
        if updateTextField and self.mergeFileWidget is not None:
            # Update text field
            self.mergeFileWidget.setBaseFileName(mergeBaseFileName)
        if self.mergeProcess is not None:
            # Notify AVMerger child process of new write base filename
            self.mergeProcess.msgQueue.put((AVMerger.SETPARAMS, dict(mergeBaseFileName=mergeBaseFileName)))
    def setMergeDirectory(self, mergeDirectory, *args, updateTextField=True):
        self.audioDirectory.set(mergeDirectory)
        if updateTextField and self.mergeFileWidget is not None:
            # Update text field
            self.mergeFileWidget.setDirectory(mergeDirectory)
        if self.mergeProcess is not None and (len(mergeDirectory) == 0 or os.path.isdir(mergeDirectory)):
            # Notify AVMerger child process of new write directory
            self.mergeProcess.msgQueue.put((AVMerger.SETPARAMS, dict(directory=mergeDirectory)))

    def setParams(self, ignoreErrors=True, **params):
        for paramName in params:
            try:
                self.paramInfo[paramName]["set"](params[paramName])
            except NotImplementedError:
                traceback.print_exc()
            except:
                traceback.print_exc()

    def getParams(self, *paramNames, mapping=False):
        # Extract parameters from GUI, and calculate a few derived parameters
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

    def setBufferSizeSeconds(self, *args):
        raise AttributeError('This attribute is a derived property, and is not directly settable')
    def setBufferSizeAudioChunks(self, *args):
        raise AttributeError('This attribute is a derived property, and is not directly settable')
    def setNumStreams(self, *args):
        raise AttributeError('This attribute is a derived property, and is not directly settable')
    def setNumProcesses(self, *args):
        raise AttributeError('This attribute is a derived property, and is not directly settable')
    def setNumSyncedProcesses(self, *args):
        raise AttributeError('This attribute is a derived property, and is not directly settable')
    def getBufferSizeSeconds(self):
        preTriggerTime = self.getParams('preTriggerTime')
        return preTriggerTime * 2 + 1    # Twice the pretrigger time to make sure we don't miss stuff, plus one second for good measure
    def getBufferSizeAudioChunks(self):
        p = self.getParams('bufferSizeSeconds', 'audioFrequency', 'chunkSize')
        return p['bufferSizeSeconds'] * p['audioFrequency'] / p['chunkSize']   # Will be rounded up to nearest integer
    def getNumStreams(self):
        audioDAQChannels = self.getParams('audioDAQChannels')
        camSerials = self.getParams('camSerials')
        return (len(audioDAQChannels)>0) + len(camSerials)
    def getNumProcesses(self):
        audioDAQChannels = self.getParams('audioDAQChannels')
        camSerials = self.getParams('camSerials')
        return (len(audioDAQChannels)>0) + len(camSerials)*2 + 2
    def getNumSyncedProcesses(self):
        audioDAQChannels = self.getParams('audioDAQChannels')
        camSerials = self.getParams('camSerials')
        return (len(audioDAQChannels)>0) + len(camSerials) + 1  # 0 or 1 audio acquire processes, N video acquire processes, and 1 sync process
    def getAcquireSettings(self):
        params = self.getParams('exposureTime', 'gain')
        exposureTime = params['exposureTime']
        gain = params['gain']
        return [
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
            ('GainAuto', 'Off', 'enum'),
            ('Gain', gain, 'float'),
            ('ExposureAuto', 'Off', 'enum'),
            ('ExposureMode', 'Timed', 'enum'),
            ('ExposureTime', exposureTime, 'float')]   # List of attribute/value pairs to be applied to the camera in the given order
    def setAcquireSettings(self, *args):
        raise NotImplementedError()

    def waitForChildProcessesToStop(self, attempts=10, timeout=0.5):
        # Wait for all state machine child processes to stop, or until all attempts have been exhausted.
        #   Returns true if all processes were found to have stopped, false if not.
        for attempts in range(attempts):
            allStopped = False
            states = self.checkStates(verbose=False)
            if 'videoWriteStates' in states:
                for camSerial in states['videoWriteStates']:
                    if not (states['videoWriteStates'][camSerial] == VideoWriter.stateList[VideoWriter.STOPPED]):
                        break;
            if 'videoAcquireStates' in states:
                for camSerial in states['videoAcquireStates']:
                    if not (states['videoAcquireStates'][camSerial] == VideoAcquirer.stateList[VideoAcquirer.STOPPED]):
                        break;
            if 'audioWriteState' in states:
                if not (states['audioWriteState'] == AudioWriter.stateList[VideoWriter.STOPPED]):
                    break;
            if 'audioAcquireState' in states:
                if not (states['audioAcquireState'] == AudioAcquirer.stateList[VideoAcquirer.STOPPED]):
                    break;
            if 'syncState' in states:
                if not (states['syncState'] == Synchronizer.stateList[Synchronizer.STOPPED]):
                    break;
            if 'mergeState' in states:
                if not (states['mergeState'] == AVMerger.stateList[AVMerger.STOPPED]):
                    break;
            allStopped = True

            if allStopped:
                return allStopped

            time.sleep(timeout)

        return False

    def createChildProcesses(self):
        self.log("Creating child processes")
        p = self.getParams()

        ready = mp.Barrier(p["numSyncedProcesses"])

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


        if p["audioSyncTerminal"] is not None or p["videoSyncTerminal"] is not None:
            # Create sync process
            self.syncProcess = Synchronizer(
                actualVideoFrequency=self.actualVideoFrequency,
                actualAudioFrequency=self.actualAudioFrequency,
                startTime=startTime,
                startTriggerChannel=p["acquisitionStartTriggerSource"],
                audioSyncChannel=p["audioSyncTerminal"],
                videoSyncChannel=p["videoSyncTerminal"],
                requestedAudioFrequency=p["audioFrequency"],
                requestedVideoFrequency=p["videoFrequency"],
                verbose=self.syncVerbose,
                ready=ready,
                stdoutQueue=self.StdoutManager.queue)

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
                stdoutQueue=self.StdoutManager.queue)
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
                verbose=self.audioWriteVerbose,
                stdoutQueue=self.StdoutManager.queue)

        for camSerial in p["camSerials"]:
            if camSerial in p["videoDirectories"]:
                videoDirectory = p["videoDirectories"]
            else:
                videoDirectory = ''
            if camSerial in p["videoBaseFileNames"]:
                videoBaseFileName = p["videoBaseFileNames"]
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
                bufferSizeSeconds=p["bufferSizeSeconds"],
                ready=ready,
                videoWidth=3208,  # Should not be hardcoded
                videoHeight=2200, # Figure out how to obtain this automatically from camera
                stdoutQueue=self.StdoutManager.queue)
            if p["triggerMode"] == "SimpleContinuous":
                videoWriteProcess = SimpleVideoWriter(
                    camSerial=camSerial,
                    videoDirectory=videoDirectory,
                    videoBaseFileName=videoBaseFileName,
                    imageQueue=videoAcquireProcess.imageQueueReceiver,
                    frameRate=self.actualVideoFrequency,
                    requestedFrameRate=p["videoFrequency"],
                    mergeMessageQueue=mergeMsgQueue,
                    videoLength=p["recordTime"],
                    verbose=self.videoWriteVerbose,
                    stdoutQueue=self.StdoutManager.queue
                    )
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
                    verbose=self.videoWriteVerbose,
                    stdoutQueue=self.StdoutManager.queue
                    )
            self.videoAcquireProcesses[camSerial] = videoAcquireProcess
            self.videoWriteProcesses[camSerial] = videoWriteProcess

        # Create (but don't start) continuous trigger process for sending
        #   automatic, continuous, and consecutive triggers to audio and video
        #   writers
        self.continuousTriggerProcess = ContinuousTriggerer(
            startTime=startTime,
            recordPeriod=p['continuousTriggerPeriod'],
            verbose=self.continuousTriggerVerbose,
            audioMessageQueue=self.audioWriteProcess.msgQueue if self.audioWriteProcess else None,
            videoMessageQueues=dict([(camSerial, self.videoWriteProcesses[camSerial].msgQueue) for camSerial in self.videoWriteProcesses]),
            stdoutQueue=self.StdoutManager.queue
        )

        # If we have an audioAcquireProcess, create (but don't start) an
        #   audioTriggerProcess to generate audio-based triggers
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
                taggerQueues=[self.continuousTriggerProcess.msgQueue],
                verbose=self.audioTriggerVerbose,
                audioMessageQueue=self.audioWriteProcess.msgQueue,
                videoMessageQueues=dict([(camSerial, self.videoWriteProcesses[camSerial].msgQueue) for camSerial in self.videoWriteProcesses]),
                stdoutQueue=self.StdoutManager.queue
                )

        # Start all audio-related processes
        if len(p["audioDAQChannels"]) > 0:
            self.audioTriggerProcess.start()
            if self.getParams('triggerMode') == "Audio":
                self.audioTriggerProcess.msgQueue.put((AudioTriggerer.STARTANALYZE, None))
            self.audioWriteProcess.start()
            self.audioAcquireProcess.start()

        # Start all video-related processes
        for camSerial in p["camSerials"]:
            self.videoWriteProcesses[camSerial].start()
            self.videoAcquireProcesses[camSerial].start()

        # Start other processes
        if self.syncProcess is not None: self.syncProcess.start()
        if self.mergeProcess is not None: self.mergeProcess.start()
        if self.continuousTriggerProcess is not None: self.continuousTriggerProcess.start()

        self.endLog(inspect.currentframe().f_code.co_name)

    def startChildProcesses(self):
        # Tell all child processes to start

        p = self.getParams('audioDAQChannels', 'camSerials')

        if len(p["audioDAQChannels"]) > 0:
            # Start audio trigger process
            self.audioTriggerProcess.msgQueue.put((AudioTriggerer.START, None))
            self.updateTriggerMode()

            # Start AudioWriter
            self.audioWriteProcess.msgQueue.put((AudioWriter.START, None))

            # Start AudioAcquirer
            self.audioAcquireProcess.msgQueue.put((AudioAcquirer.START, None))

        # Start continuous trigger process
        if self.getParams('triggerMode') == 'Continuous':
            self.continuousTriggerProcess.msgQueue.put((AudioTriggerer.START, None))

        # For each camera
        for camSerial in p["camSerials"]:
            # Start VideoWriter
            self.videoWriteProcesses[camSerial].msgQueue.put((VideoWriter.START, None))
            # Start VideoAcquirer
            self.videoAcquireProcesses[camSerial].msgQueue.put((VideoAcquirer.START, None))

        if len(p["audioDAQChannels"]) + len(p["camSerials"]) >= 2:
            # Start sync process
            self.syncProcess.msgQueue.put((Synchronizer.START, None))

            # Start merge process
            self.updateAVMergerState()

    def restartAcquisition(self):
        self.stopChildProcesses()
        stopped = self.waitForChildProcessesToStop()
        if stopped:
            self.startChildProcesses()
        else:
            self.log("Attempted to restart child processes, but could not get them to stop.")
            self.endLog(inspect.currentframe().f_code.co_name)

    def stopChildProcesses(self):
        # Tell all child processes to stop

        if self.audioTriggerProcess is not None:
            self.audioTriggerProcess.msgQueue.put((AudioTriggerer.STOP, None))
        if self.continuousTriggerProcess is not None:
            self.continuousTriggerProcess.msgQueue.put((ContinuousTriggerer.STOP, None))
        for camSerial in self.getParams('camSerials'):
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
            self.audioTriggerProcess.msgQueue.put((ContinuousTriggerer.EXIT, None))
        if self.continuousTriggerProcess is not None:
            self.continuousTriggerProcess.msgQueue.put((ContinuousTriggerer.EXIT, None))
        for camSerial in self.videoAcquireProcesses:
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
            self.log('', s.getvalue())
        except:
            self.log('Error printing profiler stats')

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
        p = self.getParams('preTriggerTime', 'recordTime')
        if t is None:
            t = time.time_ns()/1000000000
        trig = Trigger(t-p['preTriggerTime'], t, t + p['recordTime'] - p['preTriggerTime'], idspace='GUI')
        self.log("Sending manual trigger!")
        for camSerial in self.getParams('camSerials'):
            self.videoWriteProcesses[camSerial].msgQueue.put((VideoWriter.TRIGGER, trig))
            self.log("...sent to", camSerial, "video writer")
        if self.audioWriteProcess is not None:
            self.audioWriteProcess.msgQueue.put((AudioWriter.TRIGGER, trig))
            self.log("...sent to audio writer")
        self.endLog(inspect.currentframe().f_code.co_name)

    def update(self):
        # root window
        #   titleBarFrame
        #   mainFrame
        #       monitorFrame
        #           videoMonitorMasterFrame
        #           audioMonitorMasterFrame
        #       controlFrame
        #           acquisitionFrame
        #               mergeFrame
        #               scheduleFrame
        #               fileSettingsFrame
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
        camSerials = self.getParams('camSerials')
        wV, hV = getOptimalMonitorGrid(len(camSerials))
        for k, camSerial in enumerate(camSerials):
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
        self.gainFrame.grid(row=2, column=2, sticky=tk.EW)
        self.gainEntry.grid()

        self.updateInputsButton.grid(row=3, column=0)

        self.mergeFrame.grid(row=4, rowspan=2, column=0, sticky=tk.NSEW)
        self.mergeFilesCheckbutton.grid(row=1, column=0, sticky=tk.NW)
        self.deleteMergedFilesFrame.grid(row=2, column=0, sticky=tk.NW)
        self.mergeCompressionFrame.grid(row=2, column=1, sticky=tk.NW)
        self.mergeCompression.grid()
        self.deleteMergedAudioFilesCheckbutton.grid(row=0, column=0, sticky=tk.NW)
        self.deleteMergedVideoFilesCheckbutton.grid(row=0, column=1, sticky=tk.NW)
        self.montageMergeCheckbutton.grid(row=3, column=0, sticky=tk.NW)

        self.mergeFileWidget.grid(row=4, column=0, columnspan=2)

        self.fileSettingsFrame.grid(row=4, column=1, columnspan=2, sticky=tk.NSEW)
        self.daySubfoldersCheckbutton.grid(row=0, column=0)

        self.scheduleFrame.grid(row=5, column=1, columnspan=2, sticky=tk.NSEW)
        self.scheduleEnabledCheckbutton.grid(row=0, column=0, sticky=tk.NW)
        self.scheduleStartTimeEntry.grid(row=1, column=0, sticky=tk.NW)
        self.scheduleStopTimeEntry.grid(row=2, column=0, sticky=tk.NW)

        self.triggerFrame.grid(row=1, column=0, sticky=tk.NSEW)
        self.triggerModeChooserFrame.grid(row=0, column=0, sticky=tk.NW)
        self.triggerModeLabel.grid(row=0, column=0)
        for k, mode in enumerate(self.triggerModes):
            self.triggerModeRadioButtons[mode].grid(row=0, column=k+1)
        self.triggerControlTabs.grid(row=1, column=0)
        # if mode == self.triggerModeVar.get():
        #     self.triggerModeControlGroupFrames[mode].grid(row=1, column=0)
        # else:
        #     self.triggerModeControlGroupFrames[mode].grid_forget()
        self.manualWriteTriggerButton.grid(row=1, column=0)

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

        self.audioAnalysisMonitorFrame.grid(row=4, column=0, columnspan=3)
        self.audioAnalysisWidgets['canvas'].get_tk_widget().pack(side=tk.LEFT, fill=tk.BOTH, expand=1)

if __name__ == "__main__":
    root = tk.Tk()
    p = PyVAQ(root)
    root.mainloop()


r'''
cd "C:\Users\Brian Kardon\Dropbox\Documents\Work\Cornell Lab Tech\Projects\Video VI\PyVAQ\Source"
python PyVAQ.py
'''
