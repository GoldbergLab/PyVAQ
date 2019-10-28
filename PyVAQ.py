import re
import sys
import os
import struct
import math
import time
import datetime as dt
import unicodedata
import wave
import numpy as np
import multiprocessing as mp
import nidaqmx
from nidaqmx.stream_readers import AnalogMultiChannelReader
import PySpin
import tkinter as tk
import tkinter.ttk as ttk
from tkinter.filedialog import askdirectory
import queue
from PIL import Image, ImageTk
import pprint
import traceback
from subprocess import check_output
from collections import deque, defaultdict
from threading import BrokenBarrierError
import itertools
from decimal import Decimal
from TimeInput import TimeVar, TimeEntry
import cProfile, pstats, io
from scipy.signal import butter, lfilter

def generateButterBandpassCoeffs(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

# For audio monitor graph embedding:
from matplotlib.backends.backend_tkagg import (
    FigureCanvasTkAgg, NavigationToolbar2Tk)
# Implement the default Matplotlib key bindings.
from matplotlib.backend_bases import key_press_handler
from matplotlib.figure import Figure
import matplotlib.pyplot as plt

# Todo:
#  - Add video frameRate indicator
#  - Make attributes settable
#  - Add filename entry for each stream
#  - Fix acquire/write indicator positioing
#  - Make saved avis not gigantic (maybe switch to opencv for video writing?)
#  - Add external record triggering
#  - Add volume-based triggering
# Done
#  - Camera commands are not being collected properly
#  - Separate acquire and write modes so it's possible to monitor w/o writing
#  - Rework with each process as an individual state machine
#  - Turn all acquire/write/sync processes into state machines
#  - Figure out why video task isn't closing down properly
#  - Add buffering capability
#  - Fix camera monitor
#  - Add video & audio frequency controls
#  - Fix audio

r'''
cd "C:\Users\Brian Kardon\Dropbox\Documents\Work\Cornell Lab Tech\Projects\Video VI\PyVAQ"
python PyVAQ.py
'''

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

def cPrint(*args, color=None, sep='', end='\n', file=sys.stdout, flush=False, lock=None):
    colors = {
        "black"     :'b[30m',
        "red"       :'b[31m',
        "green"     :'b[32m',
        "yellow"    :'b[33m',
        "blue"      :'b[34m',
        "magenta"   :'b[35m',
        "cyan"      :'b[36m',
        "White"     :'b[37m',
        "reset"     :'b[39m',
        None        :''}
    sys.stdout.buffer.write(bytes(0x1B)+bytes('[{color}m'.format(color=colors[color]).encode('utf8')))
    syncPrint(*args, sep=sep, end=end, file=file, flush=flush, lock=lock)
    sys.stdout.buffer.write(bytes(0x1B)+bytes('[{color}m'.format(color=colors["reset"]).encode('utf8')))

class DummyExecutor:
    def __enter__(self):
        pass
    def __exit__(self, type, value, traceback):
        pass

def syncPrintOld(*args, sep=' ', end='\n', file=sys.stdout, flush=True, lock=None):
    d = DummyExecutor()
    if lock is None:
        lock = d
    with lock:
        print(*args, sep=sep, end=end, file=file, flush=flush)

def syncPrint(*args, sep=' ', end='\n', flush=True, buffer=None):
    kwargs = dict(sep=sep, end=end, flush=flush)
    buffer.append((args, kwargs))

def getStringIntersections(s1, s2, minLength=1):
    if minLength < 1:
        raise ValueError('Minimum string intersection length must be an integer greater than zero.')
    s1, s2 = sorted([s1, s2], key=lambda s:len(s))
    intersections = []
    for L in range(minLength, len(s1)+1):
        for k in range(0, len(s1)-L+1):
            if s1[k:k+L] in s2:
                intersections.append(s1[k:k+L])
    return intersections

### Main recording and writing functions

class Stopwatch:
    def __init__(self):
        self.t0 = None
        self.t1 = None

    def click(self):
        self.t0 = self.t1
        self.t1 = time.time()

    def frequency(self):
        period = self.period()
        if period is None:
            return None
        return 1.0 / period

    def period(self):
        if self.t1 is None or self.t0 is None:
            return None
        return (self.t1 - self.t0)

class Trigger():
    newid = itertools.count().__next__   # Source of this clever little idea: https://stackoverflow.com/a/1045724/1460057
    # A class to represent a trigger object
    def __init__(self, startTime, triggerTime, endTime):
        # times in seconds
        if not (endTime >= triggerTime >= startTime):
            raise ValueError("Trigger times must satisfy startTime <= triggerTime <= endTime")
        self.id = Trigger.newid()
        self.startTime = startTime
        self.triggerTime = triggerTime
        self.endTime = endTime

    def state(self, time):
        # Given a frame/sample count and frame/sample rate:
        # If time is before trigger range
        #       return how far before in seconds
        # If time is after trigger range
        #       return how far after in seconds
        # If time is during trigger range
        #       return 0
        if time < self.startTime:
            return time - self.startTime
        if time > self.endTime:
            return time - self.endTime
        return 0

class PickleableImage():
    def __init__(self, width, height, offsetX, offsetY, pixelFormat, data, frameTime):
        self.width = width
        self.height = height
        self.offsetX = offsetX
        self.offsetY = offsetY
        self.pixelFormat = pixelFormat
        self.data = data
        self.frameTime = frameTime

class AudioChunk():
    # A class to wrap a chunk of audio data.
    # It conveniently bundles the audio data with timing statistics about the data,
    # and functions for manipulating and querying time info about the data
    def __init__(self,
                chunkStartTime=None,    # Time of first sample, in seconds since something
                samplingRate=None,      # Audio sampling rate, in Hz
                data=None               # Audio data as a CxN numpy array, C=# of channels, N=# of samples
                ):
        self.data = data
        self.chunkStartTime = chunkStartTime
        self.samplingRate = samplingRate
        self.channelNumber, self.chunkSize = self.data.shape
        self.chunkEndTime = chunkStartTime + (self.chunkSize / self.samplingRate)

    def getTriggerState(self, trigger):
        chunkStartTriggerState = trigger.state(self.chunkStartTime)
        chunkEndTriggerState =   trigger.state(self.chunkEndTime)
        return chunkStartTriggerState, chunkEndTriggerState

    def trimToTrigger(self, trigger, padStart=False):
        # Trim audio chunk so it lies entirely within the trigger period, and update stats accordingly
        # If padStart == True, pad the audio chunk with enough data so it begins at the beginning of the trigger period
        chunkStartTriggerState, chunkEndTriggerState = self.getTriggerState(trigger)

        # Trim chunk start:
        if chunkStartTriggerState < 0:
            # Start of chunk is before start of trigger - truncate start of chunk.
            startSample = abs(int(chunkStartTriggerState * self.samplingRate))
            self.chunkStartTime = trigger.startTime
        elif chunkStartTriggerState == 0:
            # Start of chunk is in trigger period, do not trim start of chunk, pad if padStart=True
            startSample = 0
        else:
            # Start of chunk is after trigger period...chunk must not be in trigger period at all
            startSample = self.chunkSize
            self.chunkStartTime = trigger.endTime

        # Trim chunk end
        if chunkEndTriggerState < 0:
            # End of chunk is before start of trigger...chunk must be entirely before trigger period
            endSample = 0
            self.chunkEndTime = trigger.startTime
        elif chunkEndTriggerState == 0:
            # End of chunk is in trigger period, do not trim end of chunk
            endSample = self.chunkSize
        else:
            # End of chunk is after trigger period - trim chunk to end of trigger period
            endSample = self.chunkSize - (chunkEndTriggerState * self.samplingRate)
            self.chunkEndTime = trigger.endTime

        startSample = round(startSample)
        endSample = round(endSample)
#        print("Trim samples: {first}|{start} --> {end}|{last}".format(start=startSample, end=endSample, first=0, last=self.chunkSize))
        self.data = self.data[:, startSample:endSample]
        if padStart is True and startSample == 0:
            padLength = round((self.chunkStartTime - trigger.startTime) * self.samplingRate)
            pad = np.zeros((self.channelNumber, padLength), dtype='int16')
            self.data = np.concatenate((pad, self.data), axis=1)
        self.chunkSize = self.data.shape[1]

    def getAsBytes(self):
        bytePackingPattern = 'h'*self.data.shape[0]
        packingFunc = lambda x:struct.pack(bytePackingPattern, *x)
#        print(b''.join(list(map(packingFunc, self.data.transpose().tolist())))[0:20])
        return b''.join(map(packingFunc, self.data.transpose().tolist()))

#  audioChunkBytes = b''.join(map(lambda x:struct.pack(bytePackingPattern, *x), audioChunk.transpose().tolist()))

def generateFileName(directory, baseName, extension, trigger):
    timeString = dt.datetime.fromtimestamp(trigger.triggerTime).strftime('%Y-%m-%d-%H-%M-%S-%f')
    fileName = baseName + '_' + timeString + extension
    return os.path.join(directory, fileName)

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

class stdoutManager(mp.Process):
    # A process for printing output to stdout from other processes.
    # Expects the following messageBundle format from queues:
    #   msgBundle = [msg1, msg2, msg3...],
    # Where each message is of the format
    #   msg = ((arg1, arg2, arg3...), {kw1:kwarg1, kw2:kwarg2...})

    STOP = 'stop'

    def __init__(self, queue):
        mp.Process.__init__(self, daemon=True)
        self.queue = queue
        self.timeout = 0.1

    def run(self):
        while True:
            msgBundles = []
            try:
                msgBundles.append(self.queue.get(block=True, timeout=self.timeout))
            except queue.Empty:
                pass

            for msgBundle in msgBundles:
                if msgBundle == stdoutManager.STOP:
                    print("stdoutManager: Received stop signal!")
                    return 0
                for args, kwargs in msgBundle:
                    print(*args, **kwargs)
                print()

class AVMerger(mp.Process):
# Class for merging audio and video files using ffmpeg

    # States:
    STOPPED = 'state_stopped'
    INITIALIZING = 'state_initializing'
    IGNORING = 'state_ignoring'
    WAITING = 'state_waiting'
    MERGING = 'state_merging'
    STOPPING = 'state_stopping'
    ERROR = 'state_error'
    EXITING = 'state_exiting'

    #messages:
    START = 'msg_start'
    MERGE = 'msg_merge'
    STOP = 'msg_stop'
    EXIT = 'msg_exit'
    CHILL = 'msg_chill'
    SETPARAMS = 'msg_setParams'

    #Stream types
    VIDEO = 'video'
    AUDIO = 'audio'

    settableParams = [
        'verbose',
        'directory',
        'numFilesPerTrigger',
        'baseFileName',
        'montage',
        'deleteMergedFiles'
    ]

    def __init__(self,
        messageQueue=None,
        verbose=False,
        numFilesPerTrigger=2,       # Number of files expected per trigger event (audio + video)
        directory='.',              # Directory for writing merged files
        baseFileName='',            # Base filename (sans extension) for writing merged files
        stdoutQueue=None,           # Queue for pushing output message groups to for printing
        deleteMergedFiles=False,    # After merging, delete unmerged originals
        montage=False):             # Combine videos side by side
        mp.Process.__init__(self, daemon=True)
        # Store inputs in instance variables for later access
        self.messageQueue = messageQueue
        self.verbose = verbose
        self.exitFlag = False
        self.ignoreFlag = True
        self.errorMessages = []
        self.stdoutQueue = stdoutQueue
        self.stdoutBuffer = []
        self.numFilesPerTrigger = numFilesPerTrigger
        self.directory = directory
        self.baseFileName = baseFileName
        self.montage = montage
        self.deleteMergedFiles = deleteMergedFiles
        if numFilesPerTrigger < 2:
            raise ValueError("AVMerger can't merge less than 2 files!")

    def setParams(self, **params):
        for key in params:
            if key in AVMerger.settableParams:
                setattr(self, key, params[key])
                if self.verbose: syncPrint("M - Param set: {key}={val}".format(key=key, val=params[key]), buffer=self.stdoutBuffer)
            else:
                syncPrint("M - Param not settable: {key}={val}".format(key=key, val=params[key]), buffer=self.stdoutBuffer)

    def run(self):
        syncPrint("M - PID={pid}".format(pid=os.getpid()), buffer=self.stdoutBuffer)
        state = AVMerger.STOPPED
        nextState = AVMerger.STOPPED
        lastState = AVMerger.STOPPED
        while True:
            try:
# ********************************* STOPPPED *********************************
                if state == AVMerger.STOPPED:
                    # DO STUFF

                    # CHECK FOR MESSAGES
                    try:
                        msg, arg = self.messageQueue.get(block=True, timeout=0.1)
                        if msg == AVMerger.SETPARAMS: self.setParams(**arg); msg = ''; arg=None
                    except queue.Empty: msg = ''; arg = None

                    # CHOOSE NEXT STATE
                    if self.exitFlag:
                        nextState = AVMerger.EXITING
                    elif msg == '':
                        nextState = state
                    elif msg == AVMerger.CHILL:
                        self.ignoreFlag = True
                        nextState = AVMerger.INITIALIZING
                    elif msg == AVMerger.START:
                        self.ignoreFlag = False
                        nextState = AVMerger.INITIALIZING
                    elif msg == AVMerger.EXIT:
                        self.exitFlag = True
                        nextState = AVMerger.EXITING
                    else:
                        raise SyntaxError("Message \"" + msg + "\" not relevant to " + state + " state")
# ********************************* INITIALIZING *********************************
                elif state == AVMerger.INITIALIZING:
                    # DO STUFF
                    receivedFileEventList = []
                    groupedFileEventList = []

                    # CHECK FOR MESSAGES
                    try:
                        msg, arg = self.messageQueue.get(block=False)
                        if msg == AVMerger.SETPARAMS: self.setParams(**arg); msg = ''; arg=None
                    except queue.Empty: msg = ''; arg = None

                    # CHOOSE NEXT STATE
                    if self.exitFlag:
                        nextState = AVMerger.STOPPING
                    elif msg == AVMerger.CHILL or self.ignoreFlag:
                        self.ignoreFlag = True
                        nextState = AVMerger.IGNORING
                    elif msg == AVMerger.START:
                        self.ignoreFlag = False
                        nextState = AVMerger.WAITING
                    elif msg == '':
                        nextState = AVMerger.WAITING
                    elif msg == AVMerger.STOP:
                        nextState = AVMerger.STOPPING
                    elif msg == AVMerger.EXIT:
                        self.exitFlag = True
                        nextState = AVMerger.STOPPING
                    else:
                        raise SyntaxError("Message \"" + msg + "\" not relevant to " + state + " state")
# ********************************* IGNORING *********************************
                elif state == AVMerger.IGNORING:    # ignoring merge requests
                    # DO STUFF
                    # Clear any file events already received
                    receivedFileEventList = []
                    groupedFileEventList = []

                    # Reset ignore flag
                    self.ignoreFlag = False

                    # CHECK FOR MESSAGES
                    try:
                        msg, arg = self.messageQueue.get(block=True, timeout=0.1)
                        if msg == AVMerger.SETPARAMS: self.setParams(**arg); msg = ''; arg=None
                    except queue.Empty: msg = ''; arg = None

                    # CHOOSE NEXT STATE
                    if msg == AVMerger.MERGE and arg is not None:
                        if self.verbose: syncPrint("M - Ignoring {streamType} file event for merging at {time}: {file}".format(file=arg['filePath'], streamType=arg['streamType'], time=arg['trigger'].triggerTime), buffer=self.stdoutBuffer)
                        nextState = AVMerger.IGNORING
                    elif msg == AVMerger.CHILL:
                        self.ignoreFlag = True
                        nextState = AVMerger.IGNORING
                    elif msg == AVMerger.START:
                        self.ignoreFlag = False
                        nextState = AVMerger.WAITING
                    elif msg == '':
                        nextState = AVMerger.IGNORING
                    elif msg == AVMerger.STOP:
                        nextState = AVMerger.STOPPING
                    elif msg == AVMerger.EXIT:
                        self.exitFlag = True
                        nextState = AVMergers.STOPPING
                    else:
                        raise SyntaxError("Message \"" + msg + "\" not relevant to " + state + " state")
# ********************************* WAITING *********************************
                elif state == AVMerger.WAITING:    # Waiting for files to merge
                    # DO STUFF
                    # Waiting for message of type:
                    #   (AVMerger.MERGE, {'filePath':filePath,
                    #                     'streamType':streamType,
                    #                     'trigger':trigger,
                    #                     'streamID':streamID}
                    # Where
                    #   filepath is a string representing an audio or video file to merge
                    #   streamType is one of the defined streamtypes in AVMerger
                    #   trigger is a trigger object
                    #   streamID is a stream ID to use for file naming (only relevant for video files)
                    # If a new file has been received, add it to the list

                    # Check if any two of the files received share a matching trigger
                    IDCounts = defaultdict(lambda:0)
                    foundID = None
                    for fileEvent in receivedFileEventList:
                        IDCounts[fileEvent['trigger'].id] += 1
                        if IDCounts[fileEvent['trigger'].id] >= self.numFilesPerTrigger:
                            foundID = fileEvent['trigger'].id
                            break

                    # If one of the trigger IDs appears enough times, separate out those two files and prepare to merge them.
                    if foundID is not None:
                        groupedFileEventList.append(tuple(filter(lambda fileEvent:fileEvent['trigger'].id == foundID, receivedFileEventList)))
                        for fileEventGroup in groupedFileEventList:
                            for fileEvent in fileEventGroup:
                                try:
                                    receivedFileEventList.remove(fileEvent)
                                except ValueError:
                                    pass  # Already removed

                    if self.verbose > 1:
                        if len(receivedFileEventList) > 0 or len(groupedFileEventList) > 0:
                            syncPrint("Received: ", [p['filePath'] for p in receivedFileEventList], buffer=self.stdoutBuffer)
                            syncPrint("Ready:    ", [tuple([p['filePath'] for p in fileEvent]) for fileEvent in groupedFileEventList], buffer=self.stdoutBuffer)

                    # CHECK FOR MESSAGES
                    try:
                        msg, arg = self.messageQueue.get(block=True, timeout=0.1)
                        if msg == AVMerger.SETPARAMS: self.setParams(**arg); msg = ''; arg=None
                    except queue.Empty: msg = ''; arg = None

                    # CHOOSE NEXT STATE
                    if msg == AVMerger.MERGE and arg is not None:
                        receivedFileEventList.append(arg)
                        if self.verbose: syncPrint("M - Received {streamType} file event for merging at {time}: {file}".format(file=arg['filePath'], streamType=arg['streamType'], time=arg['trigger'].triggerTime), buffer=self.stdoutBuffer)
                        nextState = AVMerger.WAITING
                    elif msg == AVMerger.CHILL or self.ignoreFlag:
                        self.ignoreFlag = True
                        nextState = AVMerger.IGNORING
                    elif len(groupedFileEventList) > 0:
                        # At least one group of unmerged matching files - go to merge
                        nextState = AVMerger.MERGING
                    elif msg == '' or msg == AVMerger.START:
                        nextState = AVMerger.WAITING
                    elif msg == AVMerger.STOP:
                        nextState = AVMerger.STOPPING
                    elif msg == AVMerger.EXIT:
                        self.exitFlag = True
                        nextState = AVMergers.STOPPING
                    else:
                        raise SyntaxError("Message \"" + msg + "\" not relevant to " + state + " state")
# ********************************* MERGING *********************************
                elif state == AVMerger.MERGING:
                    # DO STUFF
                    # If a new file has been received, add it to the list

                    for fileEventGroup in groupedFileEventList:
                        mergeSuccess = True
                         # Merge all audio streams with each video stream individually
                        audioFileEvents = tuple(filter(lambda fileEvent:fileEvent['streamType'] == AVMerger.AUDIO, fileEventGroup))
                        videoFileEvents = tuple(filter(lambda fileEvent:fileEvent['streamType'] == AVMerger.VIDEO, fileEventGroup))
                        # Construct the audio part of the ffmpeg command template
                        audioFileInputText = ' '.join(['-i "{{audioFile{k}}}"'.format(k=k) for k in range(len(audioFileEvents))])
                        if not self.montage:  # Make a separate file for each video stream
                            # Construct command template
                            mergeCommandTemplate = 'ffmpeg -i "{videoFile}" ' + audioFileInputText + ' -shortest -nostdin -y "{outputFile}"'
                            # Set up dictionary of strings to substitute into command template
                            kwargs = dict([('audioFile{k}'.format(k=k), audioFileEvents[k]['filePath']) for k in range(len(audioFileEvents))])
                            for videoFileEvent in videoFileEvents:
                                # Add/update dictionary to reflect this video file
                                kwargs['videoFile'] = videoFileEvent['filePath']
                                baseVideoName = self.baseFileName + '_' + videoFileEvent['streamID']
                                kwargs['outputFile'] = generateFileName(directory=self.directory, baseName=baseVideoName, extension='.avi', trigger=videoFileEvent['trigger'])
                                # Substitute strings into command template
                                mergeCommand = mergeCommandTemplate.format(**kwargs)
                                if self.verbose:
                                    syncPrint("M - Merging with kwargs: "+str(kwargs), buffer=self.stdoutBuffer)
                                    syncPrint("M - Merging with command:", buffer=self.stdoutBuffer)
                                    syncPrint("M - {command}".format(command=mergeCommand), buffer=self.stdoutBuffer)
                                # Execute constructed merge command
                                status = os.system(mergeCommand)
                                mergeSuccess = mergeSuccess and (status == 0)
                                if self.verbose: syncPrint("M - Merge exit status: {status}".format(status=status), buffer=self.stdoutBuffer)
                        else:   # Montage the video streams into one file
                            # Construct the video part of the ffmpeg command template
                            videoFileInputText = ' '.join(['-i "{{videoFile{k}}}"'.format(k=k) for k in range(len(videoFileEvents))])
                            # Construct command template
                            mergeCommandTemplate = "ffmpeg " + videoFileInputText + " " + audioFileInputText + ' -shortest -nostdin -y -filter_complex hstack "{outputFile}"'
                            # Set up dictionary of strings to substitute into command template
                            kwargs = dict(
                                [('audioFile{k}'.format(k=k), audioFileEvents[k]['filePath']) for k in range(len(audioFileEvents))] + \
                                [('videoFile{k}'.format(k=k), videoFileEvents[k]['filePath']) for k in range(len(videoFileEvents))])
                            baseVideoName = self.baseFileName + '_montage'
                            kwargs['outputFile'] = generateFileName(directory=self.directory, baseName=baseVideoName, extension='.avi', trigger=videoFileEvents[0]['trigger'])
                            mergeCommand = mergeCommandTemplate.format(**kwargs)
                            if self.verbose:
                                syncPrint("M - Merging with kwargs: "+str(kwargs), buffer=self.stdoutBuffer)
                                syncPrint("M - Merging with command:", buffer=self.stdoutBuffer)
                                syncPrint("M - {command}".format(command=mergeCommand), buffer=self.stdoutBuffer)
                            # Execute constructed merge command
                            status = os.system(mergeCommand)
                            mergeSuccess = mergeSuccess and (status == 0)
                            if self.verbose: syncPrint("M - Merge exit status: {status}".format(status=status), buffer=self.stdoutBuffer)
                        if self.deleteMergedFiles:
                            if mergeSuccess:
                                for fileEvent in audioFileEvents + videoFileEvents:
                                    if self.verbose: syncPrint('M - deleting source file: {file}'.format(file=fileEvent['filePath']), buffer=self.stdoutBuffer)
                                    os.remove(fileEvent['filePath'])
                            else:
                                if self.verbose: syncPrint('M - Merge failure - keeping source files in place!', buffer=self.stdoutBuffer)

                    # Clear merged files
                    groupedFileEventList = []

                    # CHECK FOR MESSAGES
                    try:
                        msg, arg = self.messageQueue.get(block=True, timeout=0.1)
                        if msg == AVMerger.SETPARAMS: self.setParams(**arg); msg = ''; arg=None
                    except queue.Empty: msg = ''; arg = None

                    # CHOOSE NEXT STATE
                    if self.exitFlag:
                        nextState = AVMerger.STOPPING
                    elif msg == AVMerger.MERGE:
                        if arg is not None:
                            receivedFileEventList.append(arg)
                        nextState = AVMerger.WAITING
                    elif msg == '' or msg == AVMerger.START:
                        nextState = AVMerger.WAITING
                    elif msg == AVMerger.CHILL:
                        self.ignoreFlag = True
                        nextState = AVMerger.IGNORING
                    elif msg == AVMerger.STOP:
                        nextState = AVMerger.STOPPING
                    elif msg == AVMerger.EXIT:
                        self.exitFlag = True
                        nextState = AVMerger.STOPPING
                    else:
                        raise SyntaxError("Message \"" + msg + "\" not relevant to " + state + " state")
# ********************************* STOPPING *********************************
                elif state == AVMerger.STOPPING:
                    # DO STUFF

                    # CHECK FOR MESSAGES
                    try:
                        msg, arg = self.messageQueue.get(block=False)
                        if msg == AVMerger.SETPARAMS: self.setParams(**arg); msg = ''; arg=None
                    except queue.Empty: msg = ''; arg = None

                    # CHOOSE NEXT STATE
                    if self.exitFlag:
                        nextState = AVMerger.STOPPED
                    elif msg == '':
                        nextState = AVMerger.STOPPED
                    elif msg == AVMerger.STOP:
                        nextState = AVMerger.STOPPED
                    elif msg == AVMerger.EXIT:
                        self.exitFlag = True
                        nextState = AVMerger.STOPPED
                    else:
                        raise SyntaxError("Message \"" + msg + "\" not relevant to " + state + " state")
# ********************************* ERROR *********************************
                elif state == AVMerger.ERROR:
                    # DO STUFF
                    syncPrint("M - ERROR STATE. Error messages:\n\n", buffer=self.stdoutBuffer)
                    syncPrint("\n\n".join(self.errorMessages), buffer=self.stdoutBuffer)
                    self.errorMessages = []

                    # CHECK FOR MESSAGES
                    try:
                        msg, arg = self.messageQueue.get(block=False)
                        if msg == AVMerger.SETPARAMS: self.setParams(**arg); msg = ''; arg=None
                    except queue.Empty: msg = ''; arg = None

                    # CHOOSE NEXT STATE
                    if lastState == AVMerger.ERROR:
                        # Error ==> Error, let's just exit
                        nextState = AVMerger.EXIT
                    elif msg == '':
                        nextState = AVMerger.STOPPING
                    elif msg == AVMerger.STOP:
                        nextState = AVMerger.STOPPED
                    elif msg == AVMerger.EXIT:
                        self.exitFlag = True
                        if lastState == AVMerger.STOPPING:
                            nextState = AVMerger.EXITING
                        else:
                            nextState = AVMerger.STOPPING
                    else:
                        raise SyntaxError("Message \"" + msg + "\" not relevant to " + state + " state")
# ********************************* EXIT *********************************
                elif state == AVMerger.EXITING:
                    break
                else:
                    raise KeyError("Unknown state: "+state)
            except KeyboardInterrupt:
                # Handle user using keyboard interrupt
                if self.verbose: syncPrint("S - Keyboard interrupt received - exiting", buffer=self.stdoutBuffer)
                self.exitFlag = True
                nextState = AVMerger.STOPPING
            except:
                # HANDLE UNKNOWN ERROR
                self.errorMessages.append("Error in "+state+" state\n\n"+traceback.format_exc())
                nextState = AVMerger.ERROR

            if self.verbose > 1:
                if msg != '' or self.exitFlag or self.ignoreFlag:
                    syncPrint("msg={msg}, exitFlag={exitFlag}, ignoreFlag={ignoreFlag}".format(msg=msg, exitFlag=self.exitFlag, ignoreFlag=self.ignoreFlag), buffer=self.stdoutBuffer)
                syncPrint('*********************************** /\ M ' + state + ' /\ ********************************************', buffer=self.stdoutBuffer)

            # Prepare to advance to next state
            lastState = state
            state = nextState
            if len(self.stdoutBuffer) > 0: self.stdoutQueue.put(self.stdoutBuffer)
            self.stdoutBuffer = []

        if self.verbose:
            syncPrint("AVMerger process STOPPED", buffer=self.stdoutBuffer)
        if len(self.stdoutBuffer) > 0: self.stdoutQueue.put(self.stdoutBuffer)
        self.stdoutBuffer = []

class Synchronizer(mp.Process):
    # Class for generating two synchronization signals at the same time
    #   - one for video (send via cable to the camera GPIO)
    #   - one for audio (used internally to trigger analog input of microphone
    #     signals)
    # This class inherits from multiprocessing.Process so it can be run in a
    #   separate process, allowing a single script to generate the sync pulses
    #   and also accomplish other tasks.

    # States:
    STOPPED = 'state_stopped'
    INITIALIZING = 'state_initializing'
    SYNCHRONIZING = 'state_synchronizing'
    STOPPING = 'state_stopping'
    SYNC_READY = 'state_sync_ready'
    ERROR = 'state_error'
    EXITING = 'state_exiting'

    #messages:
    START = 'msg_start'
    STOP = 'msg_stop'
    EXIT = 'msg_exit'

    def __init__(self,
        videoFrequency=120,                     # The frequency in Hz of the video sync signal
        audioFrequency=44100,                   # The frequency in Hz of the audio sync signal
        videoSyncChannel="Dev3/ctr0",           # The counter channel on which to generate the video sync signal
        videoDutyCycle=0.5,
        audioSyncChannel="Dev3/ctr1",           # The counter channel on which to generate the audio sync signal
        audioDutyCycle=0.5,
        messageQueue=None,
        startTime=None,                         # Shared value that is set when sync starts, used as start time by all processes (relevant for manual triggers)
        verbose=False,
        ready=None,
        stdoutQueue=None):                            # Synchronization barrier to ensure everyone's ready before beginning
        mp.Process.__init__(self, daemon=True)
        # Store inputs in instance variables for later access
        self.startTime = startTime
        self.videoFrequency = videoFrequency
        self.audioFrequency = audioFrequency
        self.videoSyncChannel = videoSyncChannel
        self.audioSyncChannel = audioSyncChannel
        self.videoDutyCycle = videoDutyCycle
        self.audioDutyCycle = audioDutyCycle
        self.messageQueue = messageQueue
        self.ready = ready
        self.exitFlag = False
        self.errorMessages = []
        self.verbose = verbose
        self.stdoutQueue = stdoutQueue
        self.stdoutBuffer = []

    def run(self):
        syncPrint("S - PID={pid}".format(pid=os.getpid()), buffer=self.stdoutBuffer)
        state = Synchronizer.STOPPED
        nextState = Synchronizer.STOPPED
        lastState = Synchronizer.STOPPED
        while True:
            try:
# ********************************* STOPPPED *********************************
                if state == Synchronizer.STOPPED:
                    # DO STUFF

                    # CHECK FOR MESSAGES
                    try: msg, arg = self.messageQueue.get(block=True, timeout=0.1)
                    except queue.Empty: msg = ''; arg = None

                    # CHOOSE NEXT STATE
                    if self.exitFlag:
                        nextState = Synchronizer.EXITING
                    elif msg == '':
                        nextState = state
                    elif msg == Synchronizer.START:
                        nextState = Synchronizer.INITIALIZING
                    elif msg == Synchronizer.EXIT:
                        self.exitFlag = True
                        nextState = Synchronizer.EXITING
                    else:
                        raise SyntaxError("Message \"" + msg + "\" not relevant to " + state + " state")
# ********************************* INITIALIZING *********************************
                elif state == Synchronizer.INITIALIZING:
                    # DO STUFF
                # Configure and generate synchronization signal
                    trigTask = nidaqmx.Task()                       # Create task
                    trigTask.co_channels.add_co_pulse_chan_freq(
                        counter=self.videoSyncChannel,
                        name_to_assign_to_channel="videoSync",
                        units=nidaqmx.constants.FrequencyUnits.HZ,
                        initial_delay=0.0,
                        freq=self.videoFrequency,
                        duty_cycle=self.videoDutyCycle)     # Prepare a counter output channel for the video sync signal
                    trigTask.co_channels.add_co_pulse_chan_freq(
                        counter=self.audioSyncChannel,
                        name_to_assign_to_channel="audioSync",
                        units=nidaqmx.constants.FrequencyUnits.HZ,
                        initial_delay=0.0,
                        freq=self.audioFrequency,
                        duty_cycle=self.audioDutyCycle)     # Prepare a counter output channel for the audio sync signal
                    trigTask.timing.cfg_implicit_timing(sample_mode=nidaqmx.constants.AcquisitionType.CONTINUOUS)

                    # CHECK FOR MESSAGES
                    try: msg, arg = self.messageQueue.get(block=False)
                    except queue.Empty: msg = ''; arg = None

                    # CHOOSE NEXT STATE
                    if self.exitFlag:
                        nextState = Synchronizer.STOPPING
                    elif msg == '':
                        nextState = Synchronizer.SYNC_READY
                    elif msg == Synchronizer.STOP:
                        nextState = Synchronizer.STOPPING
                    elif msg == Synchronizer.EXIT:
                        self.exitFlag = True
                        nextState = Synchronizer.STOPPING
                    else:
                        raise SyntaxError("Message \"" + msg + "\" not relevant to " + state + " state")
# ********************************* SYNC_READY *********************************
                elif state == Synchronizer.SYNC_READY:
                    # DO STUFF
                    try:
                        if self.ready is not None:
                            self.ready.wait()
                        preTime = time.time_ns()
                        trigTask.start()
                        postTime = time.time_ns()
                        self.startTime.value = (preTime + postTime) / 2000000000
                    except BrokenBarrierError:
                        syncPrint("S - Simultaneous start failure", buffer=self.stdoutBuffer)
                        nextState = Synchronizer.STOPPING

                    # CHECK FOR MESSAGES
                    try: msg, arg = self.messageQueue.get(block=False)
                    except queue.Empty: msg = ''; arg = None

                    # CHOOSE NEXT STATE
                    if msg == '':
                        nextState = Synchronizer.SYNCHRONIZING
                    elif msg == Synchronizer.STOP:
                        nextState = Synchronizer.STOPPING
                    elif msg == Synchronizer.EXIT:
                        self.exitFlag = True
                        nextState = Synchronizer.STOPPING
                    else:
                        raise SyntaxError("Message \"" + msg + "\" not relevant to " + state + " state")

# ********************************* SYNCHRONIZING *********************************
                elif state == Synchronizer.SYNCHRONIZING:
                    # DO STUFF

                    # CHECK FOR MESSAGES
                    try: msg, arg = self.messageQueue.get(block=True, timeout=0.1)
                    except queue.Empty: msg = ''; arg = None

                    # CHOOSE NEXT STATE
                    if self.exitFlag:
                        nextState = Synchronizer.STOPPING
                    elif msg == '':
                        nextState = state
                    elif msg == Synchronizer.STOP:
                        nextState = Synchronizer.STOPPING
                    elif msg == Synchronizer.EXIT:
                        self.exitFlag = True
                        nextState = Synchronizer.STOPPING
                    else:
                        raise SyntaxError("Message \"" + msg + "\" not relevant to " + state + " state")
# ********************************* STOPPING *********************************
                elif state == Synchronizer.STOPPING:
                    # DO STUFF
                    trigTask.close()

                    # CHECK FOR MESSAGES
                    try: msg, arg = self.messageQueue.get(block=False)
                    except queue.Empty: msg = ''; arg = None

                    # CHOOSE NEXT STATE
                    if self.exitFlag:
                        nextState = Synchronizer.STOPPED
                    elif msg == '':
                        nextState = Synchronizer.STOPPED
                    elif msg == Synchronizer.STOP:
                        nextState = Synchronizer.STOPPED
                    elif msg == Synchronizer.EXIT:
                        self.exitFlag = True
                        nextState = Synchronizer.STOPPED
                    else:
                        raise SyntaxError("Message \"" + msg + "\" not relevant to " + state + " state")
# ********************************* ERROR *********************************
                elif state == Synchronizer.ERROR:
                    # DO STUFF
                    syncPrint("S - ERROR STATE. Error messages:\n\n", buffer=self.stdoutBuffer)
                    syncPrint("\n\n".join(self.errorMessages), buffer=self.stdoutBuffer)
                    self.errorMessages = []

                    # CHECK FOR MESSAGES
                    try: msg, arg = self.messageQueue.get(block=False)
                    except queue.Empty: msg = ''; arg = None

                    # CHOOSE NEXT STATE
                    if lastState == Synchronizer.ERROR:
                        # Error ==> Error, let's just exit
                        nextState = Synchronizer.EXIT
                    elif msg == '':
                        nextState = Synchronizer.STOPPING
                    elif msg == Synchronizer.STOP:
                        nextState = Synchronizer.STOPPED
                    elif msg == Synchronizer.EXIT:
                        self.exitFlag = True
                        if lastState == Synchronizer.STOPPING:
                            nextState = Synchronizer.EXITING
                        else:
                            nextState = Synchronizer.STOPPING
                    else:
                        raise SyntaxError("Message \"" + msg + "\" not relevant to " + state + " state")
# ********************************* EXIT *********************************
                elif state == Synchronizer.EXITING:
                    break
                else:
                    raise KeyError("Unknown state: "+state)
            except KeyboardInterrupt:
                # Handle user using keyboard interrupt
                if self.verbose: syncPrint("S - Keyboard interrupt received - exiting", buffer=self.stdoutBuffer)
                self.exitFlag = True
                nextState = Synchronizer.STOPPING
            except:
                # HANDLE UNKNOWN ERROR
                self.errorMessages.append("Error in "+state+" state\n\n"+traceback.format_exc())
                nextState = Synchronizer.ERROR

            if self.verbose:
                if msg != '' or self.exitFlag:
                    syncPrint("msg={msg}, exitFlag={exitFlag}".format(msg=msg, exitFlag=self.exitFlag), buffer=self.stdoutBuffer)
                syncPrint('*********************************** /\ S ' + state + ' /\ ********************************************', buffer=self.stdoutBuffer)
            if len(self.stdoutBuffer) > 0: self.stdoutQueue.put(self.stdoutBuffer)
            self.stdoutBuffer = []

            # Prepare to advance to next state
            lastState = state
            state = nextState

        if self.verbose:
            syncPrint("Synchronization process STOPPED", buffer=self.stdoutBuffer)
        if len(self.stdoutBuffer) > 0: self.stdoutQueue.put(self.stdoutBuffer)
        self.stdoutBuffer = []

class AudioTriggerer(mp.Process):
    # States:
    STOPPED = 'STOPPED'
    INITIALIZING = 'INITIALIZING'
    WAITING = 'WAITING'
    ANALYZING = 'ANALYZING'
    STOPPING = 'STOPPING'
    ERROR = 'ERROR'
    EXITING = 'EXITING'

    #messages:
    START = 'msg_start'
    STARTANALYZE = "msg_startanalyze"
    STOPANALYZE = "msg_stopanalyze"
    STOP = 'msg_stop'
    EXIT = 'msg_exit'
    SETPARAMS = 'msg_setParams'

    multiChannelBehaviors = ['OR', 'AND']

    settableParams = [
        'audioFrequency',
        'chunkSize',
        'triggerHighLevel',
        'triggerLowLevel',
        'triggerHighTime',
        'triggerLowTime',
        'maxAudioTriggerTime',
        'bandpassFrequencies',
        'butterworthOrder',
        'multiChannelStartBehavior',
        'multiChannelStopBehavior',
        'verbose'
        ]

    def __init__(self,
                audioQueue=None,
                audioFrequency=44100,               # Number of audio samples per second
                chunkSize=1000,                     # Number of audio samples per audio chunk
                triggerHighLevel=0.5,               # Volume level above which the audio must stay for triggerHighTime seconds to generate a start trigger
                triggerLowLevel=0.1,                # Volume level below which the audio must stay for triggerLowTime seconds to generate an updated (stop) trigger
                triggerHighTime=2,                  # Length of time that volume must stay above triggerHigh
                triggerLowTime=1,                   # Length of time that volume must stay below triggerLowLevel
                maxAudioTriggerTime=20,                  # Maximum length of trigger regardless of volume levels
                preTriggerTime=2,                   # Time before trigger to record
                triggerHighFraction=0.3,            # Fraction of high time that audio must be above high threshold for trigger start
                triggerLowFraction=0.3,             # Fraction of low time that audio must be below low threshold for trigger stop
                multiChannelStartBehavior='OR',     # How to handle multiple channels of audio. Either 'OR' (start when any channel goes higher than high threshold) or 'AND' (start when all channels go higher than high threshold)
                multiChannelStopBehavior='AND',     # How to handle multiple channels of audio. Either 'OR' (stop when any channel goes lower than low threshold) or 'AND' (stop when all channels go lower than low threshold)
                bandpassFrequencies = (100, 4000),  # A tuple of (lowfreq, highfreq) cutoff frequencies for bandpass filter
                butterworthOrder=20,
                scheduleStartTime=None,
                scheduleStopTime=None,
                verbose=False,
                audioMessageQueue=None,             # Queue to send triggers to audio writers
                videoMessageQueues={},              # Queues to send triggers to video writers
                messageQueue=None,                  # Queue for getting commands to change state
                stdoutQueue=None):
        mp.Process.__init__(self, daemon=True)
        self.audioQueue = audioQueue
        self.audioQueue.cancel_join_thread()
        self.audioMessageQueue = audioMessageQueue
        self.videoMessageQueues = videoMessageQueues
        self.audioFrequency = audioFrequency
        self.chunkSize = chunkSize
        self.triggerHighLevel = triggerHighLevel
        self.triggerLowLevel = triggerLowLevel
        self.maxAudioTriggerTime = maxAudioTriggerTime
        self.preTriggerTime = preTriggerTime
        self.bandpassFrequencies = bandpassFrequencies
        self.butterworthOrder = butterworthOrder
        self.multiChannelStartBehavior = multiChannelStartBehavior
        self.multiChannelStopBehavior = multiChannelStopBehavior
        self.messageQueue = messageQueue
        self.triggerHighTime = triggerHighTime
        self.triggerLowTime = triggerLowTime

        self.triggerHighChunks = None
        self.triggerLowChunks = None
        self.triggerHighFraction = None
        self.triggerLowFraction = None

        self.scheduleStartTime = scheduleStartTime
        self.scheduleStopTime = scheduleStopTime

        # Generate butterworth filter coefficients...or something...
        self.filter = None

        self.errorMessages = []
        self.exitFlag = False
        self.analyzeFlag = False
        self.verbose = verbose
        self.stdoutQueue = stdoutQueue
        self.stdoutBuffer = []

        self.highLevelBuffer = None
        self.lowLevelBuffer = None
        self.updateFilter()
        self.updateHighBuffer()
        self.updateLowBuffer()

    def updateHighBuffer(self):
        self.triggerHighChunks = int(self.triggerHighTime * self.audioFrequency / self.chunkSize)
        if self.highLevelBuffer is not None:
            previousHighLevelBuffer = list(self.highLevelBuffer)
        else:
            previousHighLevelBuffer = []
        self.highLevelBuffer = deque(maxlen=self.triggerHighChunks)
        self.highLevelBuffer.extend(previousHighLevelBuffer)

    def updateLowBuffer(self):
        self.triggerLowChunks = int(self.triggerLowTime * self.audioFrequency / self.chunkSize)
        if self.lowLevelBuffer is not None:
            previousLowLevelBuffer = list(self.lowLevelBuffer)
        else:
            previousLowLevelBuffer = []
        self.lowLevelBuffer = deque(maxlen=self.triggerLowChunks)
        self.lowLevelBuffer.extend(previousLowLevelBuffer)

    def setParams(self, **params):
        for key in params:
            if key in AudioTriggerer.settableParams:
                setattr(self, key, params[key])
                if key in ["triggerHighTime", 'chunkSize', 'audioFrequency']:
                    self.updateHighBuffer()
                if key in ["triggerLowTime", 'chunkSize', 'audioFrequency']:
                    self.updateLowBuffer()
                if key == 'bandpassFrequencies' or key == 'butterworthOrder':
                    self.updateFilter()
                if self.verbose: syncPrint("AT - Param set: {key}={val}".format(key=key, val=params[key]), buffer=self.stdoutBuffer)
            else:
                syncPrint("AT - Param not settable: {key}={val}".format(key=key, val=params[key]), buffer=self.stdoutBuffer)

    def run(self):
        syncPrint("AT - PID={pid}".format(pid=os.getpid()), buffer=self.stdoutBuffer)
        state = AudioTriggerer.STOPPED
        nextState = AudioTriggerer.STOPPED
        lastState = AudioTriggerer.STOPPED

        while True:
            try:
# ********************************* STOPPPED *********************************
                if state == AudioTriggerer.STOPPED:
                    # DO STUFF

                    # CHECK FOR MESSAGES
                    try:
                        msg, arg = self.messageQueue.get(block=True, timeout=0.1)
                        if msg == AudioTriggerer.SETPARAMS: self.setParams(**arg); msg = ''; arg=None
                    except queue.Empty: msg = ''; arg = None

                    # CHOOSE NEXT STATE
                    if msg == AudioTriggerer.EXIT or self.exitFlag:
                        self.exitFlag = True
                        nextState = AudioTriggerer.EXITING
                    elif msg == '':
                        nextState = AudioTriggerer.STOPPED
                    elif msg == AudioTriggerer.START:
                        nextState = AudioTriggerer.INITIALIZING
                    elif msg == AudioTriggerer.STARTANALYZE:
                        self.analyzeFlag = True
                        nextState = AudioTriggerer.INITIALIZING
                    else:
                        raise SyntaxError("Message \"" + msg + "\" not relevant to " + state + " state")
# ********************************* INITIALIZING *****************************
                elif state == AudioTriggerer.INITIALIZING:
                    # DO STUFF
                    activeTrigger = None

                    # CHECK FOR MESSAGES
                    try:
                        msg, arg = self.messageQueue.get(block=False)
                        if msg == AudioTriggerer.SETPARAMS: self.setParams(**arg); msg = ''; arg=None
                    except queue.Empty: msg = ''; arg = None

                    # CHOOSE NEXT STATE
                    if msg == AudioTriggerer.EXIT or self.exitFlag:
                        self.exitFlag = True
                        nextState = AudioTriggerer.STOPPING
                    elif msg == AudioTriggerer.STOP:
                        nextState = AudioTriggerer.STOPPING
                    elif msg == '' or msg == AudioTriggerer.STOPANALYZE:
                        nextState = AudioTriggerer.WAITING
                    elif msg == AudioTriggerer.STARTANALYZE or self.analyzeFlag:
                        nextState = AudioTriggerer.ANALYZING
                    else:
                        raise SyntaxError("Message \"" + msg + "\" not relevant to " + state + " state")
# ********************************* WAITING ********************************
                elif state == AudioTriggerer.WAITING:
                    # DO STUFF

                    # CHECK FOR MESSAGES
                    try:
                        msg, arg = self.messageQueue.get(block=False)
                        if msg == AudioTriggerer.SETPARAMS: self.setParams(**arg); msg = ''; arg=None
                    except queue.Empty: msg = ''; arg = None

                    # CHOOSE NEXT STATE
                    if msg == AudioTriggerer.STOP:
                        nextState = AudioTriggerer.STOPPING
                    elif msg == AudioTriggerer.EXIT:
                        self.exitFlag = True
                        nextState = AudioTriggerer.STOPPING
                    elif msg == AudioTriggerer.STARTANALYZE or self.analyzeFlag:
                        nextState = AudioTriggerer.ANALYZING
                    elif msg == '' or msg == AudioTriggerer.STOPANALYZE:
                        nextState = AudioTriggerer.WAITING
                    else:
                        raise SyntaxError("Message \"" + msg + "\" not relevant to " + state + " state")
# ********************************* ANALYZING *********************************
                elif state == AudioTriggerer.ANALYZING:
                    # DO STUFF
                    self.analyzeFlag = False

                    # n = number of time steps, c = number of channels
                    # Get audio chunk from audio acquirer (c x n)
                    chunkStartTime, audioChunk = self.audioQueue.get(block=True, timeout=0.1)
                    chunkEndTime = chunkStartTime + self.chunkSize / self.audioFrequency

                    # Check if current active trigger is expired, and delete it if it is
                    if activeTrigger is not None:
                        if activeTrigger.state(chunkStartTime) > 0 and activeTrigger.state(chunkEndTime) < 0:
                            # Entire chunk is after the end of the trigger period
                            activeTrigger = None

                    # Bandpass filter audio (c x n)
                    filteredAudioChunk = self.bandpass(audioChunk)
                    # Center audio - remove mean (c x n)
                    filteredCenteredAudioChunk = (filteredAudioChunk.transpose() - filteredAudioChunk.mean(axis=1)).transpose()
                    # RMS audio (c x 1)
                    rmsAudio = np.sqrt((filteredCenteredAUdioChunk ** 2).mean(axis=1))
                    # Threshold
                    high = rmsAudio > self.triggerHighLevel
                    low = rmsAudio < self.triggerLowLevel
                    # Enqueue new high/low indication
                    self.highLevelBuffer.append(high)
                    self.lowLevelBuffer.pop(low)
                    # Calculate fraction of high/low monitoring time signals have been higher than high level or lower than low level
                    highChunks = self.highLevelBuffer.sum(axis=1) * self.chunkSize / self.audioFrequency
                    lowChunks = self.lowLevelBuffer.sum(axis=1) * self.chunkSize / self.audioFrequency
                    # Calculate fraction of high/low monitoring time audio has been above/below high/low level
                    highFrac = highTime / self.triggerHighChunks
                    lowFrac = lowTime / self.triggerLowChunks
                    # Check if levels have been high/low for long enough
                    highTrigger = highFrac >= self.triggerHighFraction
                    lowTrigger = lowFrac <= self.triggerLowFraction
                    # Combine channel outcomes into a single trigger outcome using specified behavior
                    if self.multiChannelStartBehavior == "OR":
                        highTrigger = highTrigger.any()
                    elif self.multiChannelStartBehavior == "AND":
                        highTrigger = highTrigger.all()
                    if self.multiChannelStopBehavior == "OR":
                        lowTrigger = lowTrigger.any()
                    elif self.multiChannelStopBehavior == "AND":
                        lowTrigger = lowTrigger.all()

                    if activeTrigger is None and highTrigger:
                        # Send new trigger! Set to record preTriggerTime before the chunk start, and end maxAudioTriggerTime later.
                        #   If volumes go low enough for long enough, we will send an updated trigger with a new stop time
                        activeTrigger = Trigger(
                            startTime = chunkStartTime - self.preTriggerTime,
                            triggerTime = chunkStartTime,
                            endTime = chunkStartTime - self.preTriggerTime + self.maxAudioTriggerTime)
                        self.sendTrigger(activeTrigger)
                    elif activeTrigger is not None and lowTrigger:
                        # Send updated trigger
                        activeTrigger.stopTime = chunkStartTime
                        self.sendTrigger(activeTrigger)

                    # CHECK FOR MESSAGES
                    try:
                        msg, arg = self.messageQueue.get(block=False)
                        if msg == AudioTriggerer.SETPARAMS: self.setParams(**arg); msg = ''; arg=None
                    except queue.Empty: msg = ''; arg = None

                    # CHOOSE NEXT STATE
                    if self.verbose: syncPrint("AT - |{startState} ---- {endState}|".format(startState=chunkStartTriggerState, endState=chunkEndTriggerState), buffer=self.stdoutBuffer)
                    if msg == AudioTriggerer.EXIT or self.exitFlag:
                        self.exitFlag = True
                        nextState = AudioTriggerer.STOPPING
                    elif msg == AudioTriggerer.STOP:
                        nextState = AudioTriggerer.STOPPING
                    elif msg == AudioTriggerer.STOPANALYZE:
                        nextState = AudioTriggerer.WAITING
                    elif msg == '' or msg == AudioTriggerer.STARTANALYE:
                        nextState = state
                    else:
                        raise SyntaxError("Message \"" + msg + "\" not relevant to " + state + " state")
# ********************************* STOPPING *********************************
                elif state == AudioTriggerer.STOPPING:
                    # DO STUFF

                    # CHECK FOR MESSAGES
                    try:
                        msg, arg = self.messageQueue.get(block=False)
                        if msg == AudioTriggerer.SETPARAMS: self.setParams(**arg); msg = ''; arg=None
                    except queue.Empty: msg = ''; arg = None

                    # CHOOSE NEXT STATE
                    if self.exitFlag:
                        nextState = AudioTriggerer.STOPPED
                    elif msg == '':
                        nextState = AudioTriggerer.STOPPED
                    elif msg == AudioTriggerer.STOP:
                        nextState = AudioTriggerer.STOPPED
                    elif msg == AudioTriggerer.EXIT:
                        self.exitFlag = True
                        nextState = AudioTriggerer.STOPPED
                    else:
                        raise SyntaxError("Message \"" + msg + "\" not relevant to " + state + " state")
# ********************************* ERROR *********************************
                elif state == AudioTriggerer.ERROR:
                    # DO STUFF
                    syncPrint("AT - ERROR STATE. Error messages:\n\n", buffer=self.stdoutBuffer)
                    syncPrint("\n\n".join(self.errorMessages), buffer=self.stdoutBuffer)
                    self.errorMessages = []

                    # CHECK FOR MESSAGES
                    try:
                        msg, arg = self.messageQueue.get(block=False)
                        if msg == AudioTriggerer.SETPARAMS: self.setParams(**arg); msg = ''; arg=None
                    except queue.Empty: msg = ''; arg = None

                    # CHOOSE NEXT STATE
                    if msg == '':
                        nextState = AudioTriggerer.STOPPING
                    elif msg == AudioTriggerer.STOP:
                        nextState = AudioTriggerer.STOPPED
                    elif msg == AudioTriggerer.EXIT:
                        self.exitFlag = True
                        if lastState == AudioTriggerer.STOPPING:
                            nextState = AudioTriggerer.EXITING
                        else:
                            nextState = AudioTriggerer.STOPPING
                    else:
                        raise SyntaxError("Message \"" + msg + "\" not relevant to " + state + " state")
# ********************************* EXIT *********************************
                elif state == AudioTriggerer.EXITING:
                    break
                else:
                    raise KeyError("Unknown state: "+state)
            except KeyboardInterrupt:
                # Handle user using keyboard interrupt
                if self.verbose: syncPrint("AT - Keyboard interrupt received - exiting", buffer=self.stdoutBuffer)
                self.exitFlag = True
                nextState = AudioTriggerer.STOPPING
            except:
                # HANDLE UNKNOWN ERROR
                self.errorMessages.append("Error in "+state+" state\n\n"+traceback.format_exc())
                nextState = AudioTriggerer.ERROR

            if self.verbose:
                if msg != '' or self.exitFlag:
                    syncPrint("msg={msg}, exitFlag={exitFlag}".format(msg=msg, exitFlag=self.exitFlag), buffer=self.stdoutBuffer)
                syncPrint('*********************************** /\ AW ' + state + ' /\ ********************************************', buffer=self.stdoutBuffer)
            if len(self.stdoutBuffer) > 0: self.stdoutQueue.put(self.stdoutBuffer)
            self.stdoutBuffer = []

            # Prepare to advance to next state
            lastState = state
            state = nextState

        if self.verbose: syncPrint("Audio write process STOPPED", buffer=self.stdoutBuffer)
        if len(self.stdoutBuffer) > 0: self.stdoutQueue.put(self.stdoutBuffer)
        self.stdoutBuffer = []

    def updateFilter(self):
        self.filter = generateButterBandpassCoeffs(self.bandpassFrequencies[0], self.bandpassFrequencies[1], self.audioFrequency, order=self.butterworthOrder)

    def thresholds(self, audioChunk):
        # Return (belowLow, belowHigh) where belowLow is true if the audioChunk
        # is below the low level, and aboveHigh is true if the audioChunk is
        # above the high level
        pass

    def bandpass(self, audioChunk):
        b, a = self.filter
        y = lfilter(b, a, audioChunk)  # axis=
        return y

    def sendTrigger(self, trigger):
        self.audioMessageQueue.put((AudioWriter.TRIGGER, trigger))
        for camSerial in self.videoMessageQueues:
            self.videoMessageQueues.put((VideoWriter.TRIGGER, trigger))

class AudioAcquirer(mp.Process):
    # Class for acquiring an audio signal (or any analog signal) at a rate that
    #   is synchronized to the rising edges on the specified synchronization
    #   channel.

    # States:
    STOPPED = 'state_stopped'
    INITIALIZING = 'state_initializing'
    ACQUIRING = 'state_acquiring'
    STOPPING = 'state_stopping'
    ACQUIRE_READY = 'state_acquire_ready'
    ERROR = 'state_error'
    EXITING = 'state_exiting'

    #messages:
    START = 'msg_start'
    STOP = 'msg_stop'
    EXIT = 'msg_exit'

    def __init__(self,
                startTime=None,
                audioQueue = None,                  # A multiprocessing queue to send data to another proces for writing to disk
                audioMonitorQueue = None,           # A multiprocessing queue to send data to the UI to monitor the audio
                audioAnalysisQueue = None,          # A multiprocessing queue to send data to the audio triggerer process for analysis
                chunkSize = 4410,                   # Size of the read chunk in samples
                samplingRate = 44100,               # Maximum expected rate of the specified synchronization channel
                bufferSize = None,                  # Size of device buffer. Defaults to 1 second's worth of data
                channelNames = [],                  # Channel name for analog input (microphone signal)
                syncChannel = None,                 # Channel name for synchronization source
                messageQueue = None,
                verbose = False,
                ready=None,                         # Synchronization barrier to ensure everyone's ready before beginning
                stdoutQueue=None):
        mp.Process.__init__(self, daemon=True)
        # Store inputs in instance variables for later access
        self.startTimeSharedValue = startTime
        if bufferSize is None:
            self.bufferSize = chunkSize / samplingRate  # Device buffer size defaults to One second's worth of buffer
        else:
            self.bufferSize = bufferSize
        self.acquireTimeout = 2*chunkSize / samplingRate
        self.audioQueue = audioQueue
        if self.audioQueue is not None:
            self.audioQueue.cancel_join_thread()
        self.audioMonitorQueue = audioMonitorQueue
        self.audioAnalysisQueue = audioAnalysisQueue
        # if len(self.audioMonitorQueue) > 0:
        #     self.audioMonitorQueue.cancel_join_thread()
        self.chunkSize = chunkSize
        self.samplingRate = samplingRate
        self.inputChannels = channelNames
        self.syncChannel = syncChannel
        self.ready = ready
        self.errorMessages = []
        self.messageQueue = messageQueue
        self.verbose = verbose
        self.exitFlag = False
        self.stdoutQueue = stdoutQueue
        self.stdoutBuffer = []

    def rescaleAudio(data, maxV=10, minV=-10, maxD=32767, minD=-32767):
        return (data * ((maxD-minD)/(maxV-minV))).astype('int16')

    def run(self):
        syncPrint("AA - PID={pid}".format(pid=os.getpid()), buffer=self.stdoutBuffer)
        state = AudioAcquirer.STOPPED
        nextState = AudioAcquirer.STOPPED
        lastState = AudioAcquirer.STOPPED
        while True:
            try:
# ********************************* STOPPPED *********************************
                if state == AudioAcquirer.STOPPED:
                    # DO STUFF

                    # CHECK FOR MESSAGES
                    try: msg, arg = self.messageQueue.get(block=True, timeout=0.1)
                    except queue.Empty: msg = ''; arg = None

                    # CHOOSE NEXT STATE
                    if self.exitFlag:
                        nextState = AudioAcquirer.EXITING
                    elif msg == '' or msg == AudioAcquirer.STOP:
                        nextState = state
                    elif msg == AudioAcquirer.START:
                        nextState = AudioAcquirer.INITIALIZING
                    elif msg == AudioAcquirer.EXIT:
                        if self.verbose: syncPrint("Stopped state, received exit, going to exit", buffer=self.stdoutBuffer)
                        self.exitFlag = True
                        nextState = AudioAcquirer.EXITING
                    else:
                        raise SyntaxError("Message \"" + msg + "\" not relevant to " + state + " state")
# ********************************* INITIALIZING *********************************
                elif state == AudioAcquirer.INITIALIZING:
                    # DO STUFF
                    data = np.zeros((len(self.inputChannels), self.chunkSize), dtype='float')   # A pre-allocated array to receive audio data
                    processedData = data.copy()
                    readTask = nidaqmx.Task(new_task_name="audioTask!")                            # Create task
                    reader = AnalogMultiChannelReader(readTask.in_stream)  # Set up an analog stream reader
                    for inputChannel in self.inputChannels:
                        readTask.ai_channels.add_ai_voltage_chan(               # Set up analog input channel
                            inputChannel,
                            terminal_config=nidaqmx.constants.TerminalConfiguration.DIFFERENTIAL,
                            max_val=10,
                            min_val=-10)
                    readTask.timing.cfg_samp_clk_timing(                    # Configure clock source for triggering each analog read
                        rate=self.samplingRate,
                        source=self.syncChannel,                            # Specify a timing source!
                        active_edge=nidaqmx.constants.Edge.RISING,
                        sample_mode=nidaqmx.constants.AcquisitionType.CONTINUOUS,
                        samps_per_chan=self.chunkSize)
                    startTime = None
                    sampleCount = 0
                    # CHECK FOR MESSAGES
                    try: msg, arg = self.messageQueue.get(block=False)
                    except queue.Empty: msg = ''; arg = None

                    # CHOOSE NEXT STATE
                    if self.exitFlag:
                        nextState = AudioAcquirer.STOPPING
                    elif msg == '':
                        nextState = AudioAcquirer.ACQUIRE_READY
                    elif msg == AudioAcquirer.STOP:
                        nextState = AudioAcquirer.STOPPING
                    elif msg == AudioAcquirer.EXIT:
                        self.exitFlag = True
                        nextState = AudioAcquirer.STOPPING
                    else:
                        raise SyntaxError("AA - Message \"" + msg + "\" not relevant to " + state + " state")
# ********************************* ACQUIRE_READY *********************************
                elif state == AudioAcquirer.ACQUIRE_READY:
                    # DO STUFF
                    try:
                        if self.ready is not None:
#                            if self.verbose: syncPrint('AA ready: {parties} {n_waiting}'.format(parties=self.ready.parties, n_waiting=self.ready.n_waiting), buffer=self.stdoutBuffer)
                            self.ready.wait()
                    except BrokenBarrierError:
                        if self.verbose: syncPrint("AA - Simultaneous start failure", buffer=self.stdoutBuffer)
                        nextState = AudioAcquirer.STOPPING

#                    if self.verbose: syncPrint('AA passed barrier', buffer=self.stdoutBuffer)

                    # CHECK FOR MESSAGES
                    try: msg, arg = self.messageQueue.get(block=False)
                    except queue.Empty: msg = ''; arg = None

                    # CHOOSE NEXT STATE
                    if msg == '':
                        nextState = AudioAcquirer.ACQUIRING
                    elif msg == AudioAcquirer.STOP:
                        nextState = AudioAcquirer.STOPPING
                    elif msg == AudioAcquirer.EXIT:
                        self.exitFlag = True
                        nextState = AudioAcquirer.STOPPING
                    else:
                        raise SyntaxError("Message \"" + msg + "\" not relevant to " + state + " state")
# ********************************* ACQUIRING *********************************
                elif state == AudioAcquirer.ACQUIRING:
                    # DO STUFF
                    try:
                        reader.read_many_sample(                            # Read a chunk of audio data
                            data,
                            number_of_samples_per_channel=self.chunkSize,
                            timeout=self.acquireTimeout)

                        # Get timestamp of first audio chunk acquisition
                        if startTime is None:
                            if self.verbose: syncPrint("AA - Getting start time from sync process...", buffer=self.stdoutBuffer)
                            while startTime == -1 or startTime is None:
                                startTime = self.startTimeSharedValue.value
                            if self.verbose: syncPrint("AA - Got start time from sync process:"+str(startTime), buffer=self.stdoutBuffer)
#                            startTime = time.time_ns() / 1000000000 - self.chunkSize / self.samplingRate

                        chunkStartTime = startTime + sampleCount / self.samplingRate
                        sampleCount += self.chunkSize
                        processedData = AudioAcquirer.rescaleAudio(data)
                        audioChunk = AudioChunk(chunkStartTime = chunkStartTime, samplingRate = self.samplingRate, data = processedData)
                        if self.audioQueue is not None:
                            self.audioQueue.put(audioChunk)              # If a data queue is provided, queue up the new data
                        else:
                            if self.verbose: syncPrint(processedData, buffer=self.stdoutBuffer)

                        if self.audioMonitorQueue is not None:
                            self.audioMonitorQueue.put((self.inputChannels, data))      # If a monitoring queue is provided, queue up the data
                        if self.audioAnalysisQueue is not None:
                            self.audioAnalysisQueue.put((chunkStartTime, data))
                    except nidaqmx.errors.DaqError:
#                        traceback.print_exc()
                        syncPrint("AA - Audio Chunk acquisition timed out.", buffer=self.stdoutBuffer)

                    # CHECK FOR MESSAGES
                    try: msg, arg = self.messageQueue.get(block=False)
                    except queue.Empty: msg = ''; arg = None

                    # CHOOSE NEXT STATE
                    if self.exitFlag:
                        nextState = AudioAcquirer.STOPPING
                    elif msg == '':
                        nextState = AudioAcquirer.ACQUIRING
                    elif msg == AudioAcquirer.STOP:
                        nextState = AudioAcquirer.STOPPING
                    elif msg == AudioAcquirer.EXIT:
                        self.exitFlag = True
                        nextState = AudioAcquirer.STOPPING
                    else:
                        raise SyntaxError("Message \"" + msg + "\" not relevant to " + state + " state")
# ********************************* STOPPING *********************************
                elif state == AudioAcquirer.STOPPING:
                    # DO STUFF
                    if readTask is not None:
                        readTask.close()

                    # CHECK FOR MESSAGES
                    try: msg, arg = self.messageQueue.get(block=False)
                    except queue.Empty: msg = ''; arg = None

                    # CHOOSE NEXT STATE
                    if self.exitFlag:
                        nextState = AudioAcquirer.STOPPED
                    elif msg == '':
                        nextState = AudioAcquirer.STOPPED
                    elif msg == AudioAcquirer.STOP:
                        nextState = AudioAcquirer.STOPPED
                    elif msg == AudioAcquirer.EXIT:
                        self.exitFlag = True
                        nextState = AudioAcquirer.STOPPED
                    else:
                        raise SyntaxError("Message \"" + msg + "\" not relevant to " + state + " state")
# ********************************* ERROR *********************************
                elif state == AudioAcquirer.ERROR:
                    # DO STUFF
                    syncPrint("AA - ERROR STATE. Error messages:\n\n", buffer=self.stdoutBuffer)
                    syncPrint("\n\n".join(self.errorMessages), buffer=self.stdoutBuffer)
                    self.errorMessages = []

                    # CHECK FOR MESSAGES
                    try: msg, arg = self.messageQueue.get(block=False)
                    except queue.Empty: msg = ''; arg = None

                    # CHOOSE NEXT STATE
                    if lastState == AudioAcquirer.ERROR:
                        # Error ==> Error, let's just exit
                        nextState = AudioAcquirer.EXIT
                    elif msg == '':
                        nextState = AudioAcquirer.STOPPING
                    elif msg == AudioAcquirer.STOP:
                        nextState = AudioAcquirer.STOPPING
                    elif msg == AudioAcquirer.EXIT:
                        self.exitFlag = True
                        if lastState == AudioAcquirer.STOPPING:
                            nextState = AudioAcquirer.EXITING
                        else:
                            nextState = AudioAcquirer.STOPPING
                    else:
                        raise SyntaxError("Message \"" + msg + "\" not relevant to " + state + " state")
# ********************************* EXIT *********************************
                elif state == AudioAcquirer.EXITING:
                    if self.verbose: syncPrint('AA - Exiting!', buffer=self.stdoutBuffer)
                    break
                else:
                    raise KeyError("Unknown state: "+state)
            except KeyboardInterrupt:
                # Handle user using keyboard interrupt
                if self.verbose: syncPrint("AA - Keyboard interrupt received - exiting", buffer=self.stdoutBuffer)
                self.exitFlag = True
                nextState = AudioAcquirer.STOPPING
            except:
                # HANDLE UNKNOWN ERROR
                self.errorMessages.append("Error in "+state+" state\n\n"+traceback.format_exc())
                nextState = AudioAcquirer.ERROR

            if self.verbose:
                if msg != '' or self.exitFlag:
                    syncPrint("msg={msg}, exitFlag={exitFlag}".format(msg=msg, exitFlag=self.exitFlag), buffer=self.stdoutBuffer)
                syncPrint('*********************************** /\ AA ' + state + ' /\ ********************************************', buffer=self.stdoutBuffer)
            if len(self.stdoutBuffer) > 0: self.stdoutQueue.put(self.stdoutBuffer)
            self.stdoutBuffer = []

            # Prepare to advance to next state
            lastState = state
            state = nextState

        clearQueue(self.messageQueue)
        if self.verbose: syncPrint("Audio acquire process STOPPED", buffer=self.stdoutBuffer)
        if len(self.stdoutBuffer) > 0: self.stdoutQueue.put(self.stdoutBuffer)
        self.stdoutBuffer = []

class AudioWriter(mp.Process):
    # States:
    STOPPED = 'STOPPED'
    INITIALIZING = 'INITIALIZING'
    WRITING = 'WRITING'
    BUFFERING = 'BUFFERING'
    STOPPING = 'STOPPING'
    ERROR = 'ERROR'
    EXITING = 'EXITING'

    #messages:
    START = 'msg_start'
    STOP = 'msg_stop'
    EXIT = 'msg_exit'
    TRIGGER = 'msg_trigger'
    SETPARAMS = 'msg_setParams'

    settableParams = [
        'verbose',
        'audioBaseFileName',
        'audioDirectory'
        ]

    def __init__(self,
                audioBaseFileName='audioFile',
                audioDirectory='.',
                audioQueue=None,
                audioFrequency=44100,
                numChannels=1,
                bufferSizeSeconds=4,     # Buffer size in chunks - must be equal to the buffer size of associated videowriters, and equal to an integer # of audio chunks
                chunkSize=None,
                verbose=False,
                audioDepthBytes=2,
                mergeMessageQueue=None,            # Queue to put (filename, trigger) in for merging
                messageQueue=None,          # Queue for getting commands to change state
                stdoutQueue=None):
        mp.Process.__init__(self, daemon=True)
        self.audioDirectory = audioDirectory
        self.audioBaseFileName = audioBaseFileName
        self.audioQueue = audioQueue
        self.audioQueue.cancel_join_thread()
        self.audioFrequency = audioFrequency
        self.numChannels = numChannels
        self.messageQueue = messageQueue
        self.mergeMessageQueue = mergeMessageQueue
        self.bufferSize = round(bufferSizeSeconds * self.audioFrequency / chunkSize)
        self.buffer = deque(maxlen=self.bufferSize)
        self.errorMessages = []
        self.exitFlag = False
        self.verbose = verbose
        self.audioDepthBytes = audioDepthBytes
        self.stdoutQueue = stdoutQueue
        self.stdoutBuffer = []

    def setParams(self, **params):
        for key in params:
            if key in AudioWriter.settableParams:
                setattr(self, key, params[key])
                if self.verbose: syncPrint("AW - Param set: {key}={val}".format(key=key, val=params[key]), buffer=self.stdoutBuffer)
            else:
                syncPrint("AW - Param not settable: {key}={val}".format(key=key, val=params[key]), buffer=self.stdoutBuffer)

    def run(self):
        syncPrint("AW - PID={pid}".format(pid=os.getpid()), buffer=self.stdoutBuffer)
        state = AudioWriter.STOPPED
        nextState = AudioWriter.STOPPED
        lastState = AudioWriter.STOPPED
        while True:
            try:
# ********************************* STOPPPED *********************************
                if state == AudioWriter.STOPPED:
                    # DO STUFF

                    # CHECK FOR MESSAGES
                    try:
                        msg, arg = self.messageQueue.get(block=True, timeout=0.1)
                        if msg == AudioWriter.SETPARAMS: self.setParams(**arg); msg = ''; arg=None
                    except queue.Empty: msg = ''; arg = None

                    # CHOOSE NEXT STATE
                    if self.exitFlag:
                        nextState = AudioWriter.EXITING
                    elif msg == '':
                        nextState = AudioWriter.STOPPED
                    elif msg == AudioWriter.START:
                        nextState = AudioWriter.INITIALIZING
                    elif msg == AudioWriter.EXIT:
                        self.exitFlag = True
                        nextState = AudioWriter.EXITING
                    else:
                        raise SyntaxError("Message \"" + msg + "\" not relevant to " + state + " state")
# ********************************* INITIALIZING *****************************
                elif state == AudioWriter.INITIALIZING:
                    # DO STUFF
                    triggers = []
                    audioChunk = None
                    audioFile = None

                    # CHECK FOR MESSAGES
                    try:
                        msg, arg = self.messageQueue.get(block=False)
                        if msg == AudioWriter.SETPARAMS: self.setParams(**arg); msg = ''; arg=None
                    except queue.Empty: msg = ''; arg = None

                    # CHOOSE NEXT STATE
                    if self.exitFlag:
                        nextState = AudioWriter.STOPPING
                    elif msg == '':
                        nextState = AudioWriter.BUFFERING
                    elif msg == AudioWriter.STOP:
                        nextState = AudioWriter.STOPPING
                    elif msg == AudioWriter.EXIT:
                        self.exitFlag = True
                        nextState = AudioWriter.STOPPING
                    else:
                        raise SyntaxError("Message \"" + msg + "\" not relevant to " + state + " state")
# ********************************* BUFFERING ********************************
                elif state == AudioWriter.BUFFERING:
                    # DO STUFF
                    if len(self.buffer) >= self.buffer.maxlen:
                        # If buffer is full, pull oldest audio chunk from buffer
                        if self.verbose: syncPrint("AW - Pulling audio chunk from buffer (buffer: {len}/{maxlen})".format(len=len(self.buffer), maxlen=self.buffer.maxlen), buffer=self.stdoutBuffer)
                        audioChunk = self.buffer.popleft()

                    try:
                        # Get new audio chunk and push it into the buffer
                        newAudioChunk = self.audioQueue.get(block=True, timeout=0.1)
                        if self.verbose: syncPrint("AW - Got audio chunk from acquirer. Pushing into the buffer.", buffer=self.stdoutBuffer)
                        self.buffer.append(newAudioChunk)
                    except queue.Empty: # None available
                        newAudioChunk = None

                    # CHECK FOR MESSAGES
                    try:
                        msg, arg = self.messageQueue.get(block=False)
                        if msg == AudioWriter.SETPARAMS: self.setParams(**arg); msg = ''; arg=None
                    except queue.Empty: msg = ''; arg = None

                    # CHOOSE NEXT STATE
                    if msg == AudioWriter.STOP:
                        nextState = AudioWriter.STOPPING
                    elif msg == AudioWriter.EXIT:
                        self.exitFlag = True
                        nextState = AudioWriter.STOPPING
                    elif msg == AudioWriter.TRIGGER or len(triggers) > 0:
                        if self.verbose: syncPrint("AW - Trigger is active", buffer=self.stdoutBuffer)
                        # Either we are just receiving a trigger message, or a trigger was previously received and not started yet
                        if arg is not None:
                            # New or updated trigger from message
                            trigger = arg
                            if len(triggers) > 0 and trigger.startTime == triggers[-1].startTime and trigger.triggerTime == triggers[-1].triggerTime:
                                # This is an updated trigger, not a new trigger
                                if self.verbose: syncPrint("AW - Updating trigger", buffer=self.stdoutBuffer)
                                triggers[-1] = trigger
                            else:
                                # This is a new trigger
                                if self.verbose: syncPrint("AW - Adding new trigger", buffer=self.stdoutBuffer)
                                triggers.append(trigger)
                        if audioChunk is not None:
                            # At least one audio chunk has been received - we can check if trigger period has begun
                            chunkStartTriggerState, chunkEndTriggerState = audioChunk.getTriggerState(triggers[0])
                            if self.verbose: syncPrint("AW - |{startState} ---- {endState}|".format(startState=chunkStartTriggerState, endState=chunkEndTriggerState), buffer=self.stdoutBuffer)
                            if chunkStartTriggerState < 0 and chunkEndTriggerState < 0:
                                # Entire chunk is before trigger range
                                nextState = AudioWriter.BUFFERING
                            elif chunkStartTriggerState == 0 or chunkEndTriggerState == 0 or (chunkStartTriggerState < 0 and chunkStartTriggerState > 0):
                                # Time is now in trigger range
                                if self.verbose: syncPrint("AW - Missed trigger period!", buffer=self.stdoutBuffer)
                                timeWrote = 0
                                nextState = AudioWriter.WRITING
                            else:
                                # Time is after trigger range...
                                if self.verbose: syncPrint("Missed video trigger start by", triggerState, "seconds!", buffer=self.stdoutBuffer)
                                timeWrote = 0
                                nextState = AudioWriter.BUFFERING
                                triggers.pop(0)   # Pop off trigger that we missed
                        else:
                            # No audio chunks have been received yet, can't evaluate if trigger time has begun yet
                            if self.verbose: syncPrint("AW - No audio chunks yet, can't begin trigger yet (buffer: {len}/{maxlen})".format(len=len(self.buffer), maxlen=self.buffer.maxlen), buffer=self.stdoutBuffer)
                            nextState = AudioWriter.BUFFERING
                    elif msg == '':
                        nextState = AudioWriter.BUFFERING
                    else:
                        raise SyntaxError("Message \"" + msg + "\" not relevant to " + state + " state")
# ********************************* WRITING *********************************
                elif state == AudioWriter.WRITING:
                    # DO STUFF
                    if audioChunk is not None:
                        if audioFile is None:
                            # Start new audio file
                            audioFileName = generateFileName(directory=self.audioDirectory, baseName=self.audioBaseFileName, extension='.wav', trigger=triggers[0])
                            audioFile = wave.open(audioFileName, 'w')
                            audioFile.audioFileName = audioFileName
                            # setParams: (nchannels, sampwidth, frameRate, nframes, comptype, compname)
                            audioFile.setparams((self.numChannels, self.audioDepthBytes, self.audioFrequency, 0, 'NONE', 'not compressed'))
                        #     padStart = True  # Because this is the first chunk, pad the start of the chunk if it starts after the trigger period start
                        # else:
                        #     padStart = False
                        if self.verbose: syncPrint("AW - Trimming audio", buffer=self.stdoutBuffer)
                        audioChunk.trimToTrigger(triggers[0]) #, padStart=padStart)

                        # Write chunk of audio to file that was previously retrieved from the buffer
                        if self.verbose:
                            syncPrint("AW - Writing audio", buffer=self.stdoutBuffer)
                            timeWrote += (audioChunk.data.shape[1] / audioChunk.samplingRate)
                            syncPrint("AW - Time wrote: {time}".format(time=timeWrote), buffer=self.stdoutBuffer)
                        audioFile.writeframes(audioChunk.getAsBytes())

                    try:
                        # Pop the oldest buffered audio chunk from the back of the buffer.
                        audioChunk = self.buffer.popleft()
                        if self.verbose: syncPrint("AW - Pulled audio from buffer", buffer=self.stdoutBuffer)
                    except IndexError:
                        if self.verbose: syncPrint("AW - No data in buffer", buffer=self.stdoutBuffer)
                        audioChunk = None  # Buffer was empty

                    # Pull new audio chunk from AudioAcquirer and add to the front of the buffer.
                    try:
                        newAudioChunk = self.audioQueue.get(True, 0.05)
                        if self.verbose: syncPrint("AW - Got audio chunk from acquirer. Pushing into buffer.", buffer=self.stdoutBuffer)
                        self.buffer.append(newAudioChunk)
                    except queue.Empty:
                        # No data in audio queue yet - pass.
                        if self.verbose: syncPrint("AW - No audio available from acquirer", buffer=self.stdoutBuffer)

                    # CHECK FOR MESSAGES (and consume certain messages that don't trigger state transitions)
                    try:
                        msg, arg = self.messageQueue.get(block=False)
                        if msg == AudioWriter.SETPARAMS: self.setParams(**arg); msg = ''; arg=None
                        elif msg == AudioWriter.TRIGGER: self.updateTriggers(triggers, arg); msg = ''; arg=None
                    except queue.Empty: msg = ''; arg = None

                    # CHOOSE NEXT STATE
                    if self.verbose: syncPrint("AW - |{startState} ---- {endState}|".format(startState=chunkStartTriggerState, endState=chunkEndTriggerState), buffer=self.stdoutBuffer)
                    if self.exitFlag:
                        nextState = AudioWriter.STOPPING
                    elif msg == AudioWriter.STOP:
                        nextState = AudioWriter.STOPPING
                    elif msg == AudioWriter.EXIT:
                        self.exitFlag = True
                        nextState = AudioWriter.STOPPING
                    elif len(triggers) > 0 and audioChunk is not None:
                        chunkStartTriggerState, chunkEndTriggerState = audioChunk.getTriggerState(triggers[0])
                        if chunkStartTriggerState * chunkEndTriggerState > 0:
                            # Trigger period does not overlap the chunk at all - return to buffering
                            if self.verbose: syncPrint("AW - Audio chunk does not overlap trigger. Switching to buffering.", buffer=self.stdoutBuffer)
                            nextState = AudioWriter.BUFFERING
                            if audioFile is not None:
                                # Done with trigger, close file and clear audioFile
                                audioFile.writeframes(b'')  # Recompute header info?
                                audioFile.close()
                                if self.mergeMessageQueue is not None:
                                    # Send file for AV merging:
                                    fileEvent = dict(
                                        filePath=audioFile.audioFileName,
                                        streamType=AVMerger.AUDIO,
                                        trigger=triggers[0],
                                        streamID='audio'
                                    )
                                    self.mergeMessageQueue.put((AVMerger.MERGE, fileEvent))
                                audioFile = None
                            # Remove current trigger
                            triggers.pop(0)
                        else:
                            # Audio chunk does overlap with trigger period. Continue writing.
                            nextState = AudioWriter.WRITING
                    elif msg == '':
                        nextState = state
                    else:
                        raise SyntaxError("Message \"" + msg + "\" not relevant to " + state + " state")
# ********************************* STOPPING *********************************
                elif state == AudioWriter.STOPPING:
                    # DO STUFF
                    if audioFile is not None:
                        audioFile.writeframes(b'')  # Recompute header info?
                        audioFile.close()
                        if self.mergeMessageQueue is not None:
                            # Send file for AV merging:
                            fileEvent = dict(
                                filePath=audioFile.audioFileName,
                                streamType=AVMerger.AUDIO,
                                trigger=triggers[0],
                                streamID='audio'
                            )
                            self.mergeMessageQueue.put((AVMerger.MERGE, fileEvent))
                        audioFile = None
                    triggers = []

                    # CHECK FOR MESSAGES
                    try:
                        msg, arg = self.messageQueue.get(block=False)
                        if msg == AudioWriter.SETPARAMS: self.setParams(**arg); msg = ''; arg=None
                    except queue.Empty: msg = ''; arg = None

                    # CHOOSE NEXT STATE
                    if self.exitFlag:
                        nextState = AudioWriter.STOPPED
                    elif msg == '':
                        nextState = AudioWriter.STOPPED
                    elif msg == AudioWriter.STOP:
                        nextState = AudioWriter.STOPPED
                    elif msg == AudioWriter.EXIT:
                        self.exitFlag = True
                        nextState = AudioWriter.STOPPED
                    else:
                        raise SyntaxError("Message \"" + msg + "\" not relevant to " + state + " state")
# ********************************* ERROR *********************************
                elif state == AudioWriter.ERROR:
                    # DO STUFF
                    syncPrint("AW - ERROR STATE. Error messages:\n\n", buffer=self.stdoutBuffer)
                    syncPrint("\n\n".join(self.errorMessages), buffer=self.stdoutBuffer)
                    self.errorMessages = []

                    # CHECK FOR MESSAGES
                    try:
                        msg, arg = self.messageQueue.get(block=False)
                        if msg == AudioWriter.SETPARAMS: self.setParams(**arg); msg = ''; arg=None
                    except queue.Empty: msg = ''; arg = None

                    # CHOOSE NEXT STATE
                    if msg == '':
                        nextState = AudioWriter.STOPPING
                    elif msg == AudioWriter.STOP:
                        nextState = AudioWriter.STOPPED
                    elif msg == AudioWriter.EXIT:
                        self.exitFlag = True
                        if lastState == AudioWriter.STOPPING:
                            nextState = AudioWriter.EXITING
                        else:
                            nextState = AudioWriter.STOPPING
                    else:
                        raise SyntaxError("Message \"" + msg + "\" not relevant to " + state + " state")
# ********************************* EXIT *********************************
                elif state == AudioWriter.EXITING:
                    break
                else:
                    raise KeyError("Unknown state: "+state)
            except KeyboardInterrupt:
                # Handle user using keyboard interrupt
                if self.verbose: syncPrint("AW - Keyboard interrupt received - exiting", buffer=self.stdoutBuffer)
                self.exitFlag = True
                nextState = AudioWriter.STOPPING
            except:
                # HANDLE UNKNOWN ERROR
                self.errorMessages.append("Error in "+state+" state\n\n"+traceback.format_exc())
                nextState = AudioWriter.ERROR

            if self.verbose:
                if msg != '' or self.exitFlag:
                    syncPrint("msg={msg}, exitFlag={exitFlag}".format(msg=msg, exitFlag=self.exitFlag), buffer=self.stdoutBuffer)
                syncPrint('*********************************** /\ AW ' + state + ' /\ ********************************************', buffer=self.stdoutBuffer)
            if len(self.stdoutBuffer) > 0: self.stdoutQueue.put(self.stdoutBuffer)
            self.stdoutBuffer = []

            # Prepare to advance to next state
            lastState = state
            state = nextState

        if self.verbose: syncPrint("Audio write process STOPPED", buffer=self.stdoutBuffer)
        if len(self.stdoutBuffer) > 0: self.stdoutQueue.put(self.stdoutBuffer)
        self.stdoutBuffer = []

    def updateTriggers(self, triggers, trigger):
        if len(triggers) > 0 and trigger.id == triggers[-1].id:
            # This is an updated trigger, not a new trigger
            if self.verbose: syncPrint("AW - Updating trigger", buffer=self.stdoutBuffer)
            triggers[-1] = trigger
        else:
            # This is a new trigger
            if self.verbose: syncPrint("AW - Adding new trigger", buffer=self.stdoutBuffer)
            triggers.append(trigger)

class VideoAcquirer(mp.Process):
    # States:
    STOPPED = 'state_stopped'
    INITIALIZING = 'state_initializing'
    ACQUIRING = 'state_acquiring'
    STOPPING = 'state_stopping'
    ACQUIRE_READY = 'state_acquire_ready'
    ERROR = 'state_error'
    EXITING = 'state_exiting'

    #messages:
    START = 'msg_start'
    STOP = 'msg_stop'
    EXIT = 'msg_exit'

    def __init__(self,
                startTime=None,
                camSerial='',
                imageQueue=None,
                monitorImageQueue=None,
                acquireSettings={},
                frameRate=None,
                monitorFrameRate=15,
                messageQueue=None,
                verbose=False,
                ready=None,                        # Synchronization barrier to ensure everyone's ready before beginning
                stdoutQueue=None):
        mp.Process.__init__(self, daemon=True)
        self.startTimeSharedValue = startTime
        self.camSerial = camSerial
        self.ID = 'VA_'+self.camSerial
        self.acquireSettings = acquireSettings
        self.frameRate = frameRate
        self.imageQueue = imageQueue
        if self.imageQueue is not None:
            self.imageQueue.cancel_join_thread()
        self.monitorImageQueue = monitorImageQueue
        if self.monitorImageQueue is not None:
            self.monitorImageQueue.cancel_join_thread()
        self.monitorFrameRate = monitorFrameRate
        self.ready = ready
        self.frameStopwatch = Stopwatch()
        self.monitorStopwatch = Stopwatch()
        self.exitFlag = False
        self.errorMessages = []
        self.messageQueue = messageQueue
        self.verbose = verbose
        self.stdoutQueue = stdoutQueue
        self.stdoutBuffer = []

    def run(self):
        if self.verbose > 1: profiler = cProfile.Profile()
        syncPrint(self.ID + " PID={pid}".format(pid=os.getpid()), buffer=self.stdoutBuffer)

        state = VideoAcquirer.STOPPED
        nextState = VideoAcquirer.STOPPED
        lastState = VideoAcquirer.STOPPED
        while True:
            try:
# ********************************* STOPPPED *********************************
                if state == VideoAcquirer.STOPPED:
                    # DO STUFF

                    # CHECK FOR MESSAGES
                    try: msg, arg = self.messageQueue.get(block=True, timeout=0.1)
                    except queue.Empty: msg = ''; arg = None

                    # CHOOSE NEXT STATE
                    if self.exitFlag:
                        nextState = VideoAcquirer.EXITING
                    elif msg == '' or msg == VideoAcquirer.STOP:
                        nextState = state
                    elif msg == VideoAcquirer.START:
                        nextState = VideoAcquirer.INITIALIZING
                    elif msg == VideoAcquirer.EXIT:
                        if self.verbose: syncPrint("Stopped state, received exit, going to exit", buffer=self.stdoutBuffer)
                        self.exitFlag = True
                        nextState = VideoAcquirer.EXITING
                    else:
                        raise SyntaxError("Message \"" + msg + "\" not relevant to " + state + " state")
# ********************************* INITIALIZING *********************************
                elif state == VideoAcquirer.INITIALIZING:
                    # DO STUFF
                    system = PySpin.System.GetInstance()
                    camList = system.GetCameras()
                    cam = camList.GetBySerial(self.camSerial)
                    cam.Init()
                    nodemap = cam.GetNodeMap()
                    self.setCameraAttributes(nodemap, self.acquireSettings)

                    monitorFramePeriod = 1.0/self.monitorFrameRate
                    syncPrint("Monitoring with period", monitorFramePeriod, buffer=self.stdoutBuffer)
                    # syncPrint("Video monitor frame period:", monitorFramePeriod)
                    thisTime = 0
                    lastTime = time.time()
                    imageCount = 0
                    im = imp = imageResult = None
                    startTime = None
                    frameTime = None

                    # CHECK FOR MESSAGES
                    try: msg, arg = self.messageQueue.get(block=False)
                    except queue.Empty: msg = ''; arg = None

                    # CHOOSE NEXT STATE
                    if self.exitFlag:
                        nextState = VideoAcquirer.STOPPING
                    elif msg == '':
                        nextState = VideoAcquirer.ACQUIRE_READY
                    elif msg == VideoAcquirer.STOP:
                        nextState = VideoAcquirer.STOPPING
                    elif msg == VideoAcquirer.EXIT:
                        self.exitFlag = True
                        nextState = VideoAcquirer.STOPPING
                    else:
                        raise SyntaxError(self.ID + " Message \"" + msg + "\" not relevant to " + state + " state")
# ********************************* ACQUIRE_READY *********************************
                elif state == VideoAcquirer.ACQUIRE_READY:
                    # DO STUFF
                    cam.BeginAcquisition()
                    try:
                        if self.ready is not None:
                            if self.verbose: syncPrint('{ID} ready: {parties} {n_waiting}'.format(ID=self.ID, parties=self.ready.parties, n_waiting=self.ready.n_waiting), buffer=self.stdoutBuffer)
                            self.ready.wait()
                    except BrokenBarrierError:
                        syncPrint(self.ID + " Simultaneous start failure", buffer=self.stdoutBuffer)
                        nextState = VideoAcquirer.STOPPING

                    if self.verbose: syncPrint('{ID} passed barrier'.format(ID=self.ID), buffer=self.stdoutBuffer)

                    # CHECK FOR MESSAGES
                    try: msg, arg = self.messageQueue.get(block=False)
                    except queue.Empty: msg = ''; arg = None

                    # CHOOSE NEXT STATE
                    if msg == '':
                        nextState = VideoAcquirer.ACQUIRING
                    elif msg == VideoAcquirer.STOP:
                        nextState = VideoAcquirer.STOPPING
                    elif msg == VideoAcquirer.EXIT:
                        self.exitFlag = True
                        nextState = VideoAcquirer.STOPPING
                    else:
                        raise SyntaxError("Message \"" + msg + "\" not relevant to " + state + " state")
# ********************************* ACQUIRING *********************************
                elif state == VideoAcquirer.ACQUIRING:
                    if self.verbose > 1: profiler.enable()
                    # DO STUFF
                    try:
                        #  Retrieve next received image
                        imageResult = cam.GetNextImage()
                        # Get timestamp of first image acquisition
                        if startTime is None:
                            if self.verbose: syncPrint(self.ID+" - Getting start time from sync process...", buffer=self.stdoutBuffer)
                            while startTime == -1 or startTime is None:
                                startTime = self.startTimeSharedValue.value
                            if self.verbose: syncPrint(self.ID+" - Got start time from sync process:"+str(startTime), buffer=self.stdoutBuffer)
#                            startTime = time.time_ns() / 1000000000

                        # Time frames, as an extra check
                        self.frameStopwatch.click()
                        if self.verbose: syncPrint(self.ID + " Video freq: ", self.frameStopwatch.frequency(), buffer=self.stdoutBuffer)

                        #  Ensure image completion
                        if imageResult.IsIncomplete():
                            syncPrint('VA - Image incomplete with image status %d...' % imageResult.GetImageStatus(), buffer=self.stdoutBuffer)
                        else:
                            imageCount += 1
                            frameTime = startTime + imageCount / self.frameRate

                            if self.verbose: syncPrint(self.ID + " Got image from camera, t="+str(frameTime), buffer=self.stdoutBuffer)

                            imp = PickleableImage(imageResult.GetWidth(), imageResult.GetHeight(), 0, 0, imageResult.GetPixelFormat(), imageResult.GetData(), frameTime)

                            # Put image into image queue
                            self.imageQueue.put(imp)
                            if self.verbose: syncPrint(self.ID + " Pushed image into buffer", buffer=self.stdoutBuffer)

                            if self.monitorImageQueue is not None:
                                # Put the occasional image in the monitor queue for the UI
                                thisTime = time.time()
                                actualMonitorFramePeriod = thisTime - lastTime
                                if (thisTime - lastTime) >= monitorFramePeriod:
                                    try:
                                        self.monitorImageQueue.put(imp, block=False)
                                        if self.verbose: syncPrint(self.ID + " Sent frame for monitoring", buffer=self.stdoutBuffer)
                                        lastTime = thisTime
                                    except queue.Full:
                                        pass

                        imageResult.Release()
                    except PySpin.SpinnakerException:
                        print(traceback.format_exc())
                        syncPrint(self.ID + " Video frame acquisition timeed out.", buffer=self.stdoutBuffer)

                    # CHECK FOR MESSAGES
                    try: msg, arg = self.messageQueue.get(block=False)
                    except queue.Empty: msg = ''; arg = None

                    # CHOOSE NEXT STATE
                    if self.exitFlag:
                        nextState = VideoAcquirer.STOPPING
                    elif msg == '':
                        nextState = VideoAcquirer.ACQUIRING
                    elif msg == VideoAcquirer.STOP:
                        nextState = VideoAcquirer.STOPPING
                    elif msg == VideoAcquirer.EXIT:
                        self.exitFlag = True
                        nextState = VideoAcquirer.STOPPING
                    else:
                        raise SyntaxError("Message \"" + msg + "\" not relevant to " + state + " state")
                    if self.verbose > 1: profiler.disable()
# ********************************* STOPPING *********************************
                elif state == VideoAcquirer.STOPPING:
                    # DO STUFF
                    if cam is not None:
                        camList.Clear()
                        cam.EndAcquisition()
                        cam.DeInit()
                        cam = None
                    if system is not None:
                        system.ReleaseInstance()

                    # CHECK FOR MESSAGES
                    try: msg, arg = self.messageQueue.get(block=False)
                    except queue.Empty: msg = ''; arg = None

                    # CHOOSE NEXT STATE
                    if self.exitFlag:
                        nextState = VideoAcquirer.STOPPED
                    elif msg == '':
                        nextState = VideoAcquirer.STOPPED
                    elif msg == AudioWriter.STOP:
                        nextState = VideoAcquirer.STOPPED
                    elif msg == VideoAcquirer.EXIT:
                        self.exitFlag = True
                        nextState = VideoAcquirer.STOPPED
                    else:
                        raise SyntaxError("Message \"" + msg + "\" not relevant to " + state + " state")
# ********************************* ERROR *********************************
                elif state == VideoAcquirer.ERROR:
                    # DO STUFF
                    syncPrint(self.ID + " ERROR STATE. Error messages:\n\n", buffer=self.stdoutBuffer)
                    syncPrint("\n\n".join(self.errorMessages), buffer=self.stdoutBuffer)
                    self.errorMessages = []

                    # CHECK FOR MESSAGES
                    try: msg, arg = self.messageQueue.get(block=False)
                    except queue.Empty: msg = ''; arg = None

                    # CHOOSE NEXT STATE
                    if lastState == VideoAcquirer.ERROR:
                        # Error ==> Error, Let's just exit
                        nextState = VideoAcquirer.EXIT
                    if msg == '':
                        nextState = VideoAcquirer.STOPPING
                    elif msg == VideoAcquirer.STOP:
                        nextState = VideoAcquirer.STOPPING
                    elif msg == VideoAcquirer.EXIT:
                        self.exitFlag = True
                        if lastState == VideoAcquirer.STOPPING:
                            nextState = VideoAcquirer.EXITING
                        else:
                            nextState = VideoAcquirer.STOPPING
                    else:
                        raise SyntaxError("Message \"" + msg + "\" not relevant to " + state + " state")
# ********************************* EXIT *********************************
                elif state == VideoAcquirer.EXITING:
                    break
                else:
                    raise KeyError("Unknown state: "+state)
            except KeyboardInterrupt:
                # Handle user using keyboard interrupt
                if self.verbose: syncPrint(self.ID + " Keyboard interrupt received - exiting", buffer=self.stdoutBuffer)
                self.exitFlag = True
                nextState = VideoAcquirer.STOPPING
            except:
                # HANDLE UNKNOWN ERROR
                self.errorMessages.append("Error in "+state+" state\n\n"+traceback.format_exc())
                nextState = VideoAcquirer.ERROR

            if self.verbose:
                if msg != '' or self.exitFlag:
                    syncPrint("msg={msg}, exitFlag={exitFlag}".format(msg=msg, exitFlag=self.exitFlag), buffer=self.stdoutBuffer)
                syncPrint('*********************************** /\ ' + self.ID + ' ' + state + ' /\ ********************************************', buffer=self.stdoutBuffer)
            if len(self.stdoutBuffer) > 0: self.stdoutQueue.put(self.stdoutBuffer)
            self.stdoutBuffer = []

            # Prepare to advance to next state
            lastState = state
            state = nextState

        if self.verbose: syncPrint("Video acquire process STOPPED", buffer=self.stdoutBuffer)
        if self.verbose > 1:
            s = io.StringIO()
            ps = pstats.Stats(profiler, stream=s)
            ps.print_stats()
            syncPrint(s.getvalue(), buffer=self.stdoutBuffer)
        if len(self.stdoutBuffer) > 0: self.stdoutQueue.put(self.stdoutBuffer)
        self.stdoutBuffer = []

    def setCameraAttribute(self, nodemap, attributeName, attributeValue, type='enum'):
        # Set camera attribute. Retrusn True if successful, False otherwise.
        if self.verbose: syncPrint('Setting', attributeName, 'to', attributeValue, 'as', type, buffer=self.stdoutBuffer)
        nodeAttribute = nodeAccessorTypes[type](nodemap.GetNode(attributeName))
        if not PySpin.IsAvailable(nodeAttribute) or not PySpin.IsWritable(nodeAttribute):
            if self.verbose: syncPrint('Unable to set '+str(attributeName)+' to '+str(attributeValue)+' (enum retrieval). Aborting...', buffer=self.stdoutBuffer)
            return False

        if type == 'enum':
            # Retrieve entry node from enumeration node
            nodeAttributeValue = nodeAttribute.GetEntryByName(attributeValue)
            if not PySpin.IsAvailable(nodeAttributeValue) or not PySpin.IsReadable(nodeAttributeValue):
                if self.verbose: syncPrint('Unable to set '+str(attributeName)+' to '+str(attributeValue)+' (entry retrieval). Aborting...', buffer=self.stdoutBuffer)
                return False

            # Set value
            attributeValue = nodeAttributeValue.GetValue()
            nodeAttribute.SetIntValue(attributeValue)
        else:
            nodeAttribute.SetValue(attributeValue)
        return True

    def setCameraAttributes(self, nodemap, attributeValueTriplets):
    #    print('Setting attributes')
        for attribute, value, type in attributeValueTriplets:
            result = self.setCameraAttribute(nodemap, attribute, value, type=type)
            if not result:
                print("Failed to set", str(attribute), " to ", str(value))

class VideoWriter(mp.Process):
    # States:
    STOPPED = 'state_stopped'
    INITIALIZING = 'state_initializing'
    WRITING = 'state_writing'
    BUFFERING = 'state_buffering'
    STOPPING = 'state_stopping'
    ERROR = 'state_error'
    EXITING = 'state_exiting'

    #messages:
    START = 'msg_start'
    STOP = 'msg_stop'
    EXIT = 'msg_exit'
    TRIGGER = 'msg_trigger'
    SETPARAMS = 'msg_setParams'

    settableParams = [
        'verbose',
        'videoBaseFileName',
        'videoDirectory'
        ]

    def __init__(self,
                videoDirectory='.',
                videoBaseFilename='videoFile',
                imageQueue=None,
                frameRate=10,
                messageQueue=None,
                mergeMessageQueue=None,            # Queue to put (filename, trigger) in for merging
                bufferSizeSeconds=5,
                camSerial='',
                verbose=False,
                stdoutQueue=None):
        mp.Process.__init__(self, daemon=True)
        self.camSerial = camSerial
        self.ID = 'VW_' + self.camSerial
        self.videoDirectory=videoDirectory
        self.videoBaseFilename = videoBaseFilename
        self.imageQueue = imageQueue
        self.imageQueue.cancel_join_thread()
        self.frameRate = frameRate
        self.messageQueue = messageQueue
        self.mergeMessageQueue = mergeMessageQueue
        self.bufferSize = int(bufferSizeSeconds * self.frameRate)
        self.buffer = deque(maxlen=self.bufferSize)
        self.errorMessages = []
        self.exitFlag = False
        self.verbose = verbose
        self.stdoutQueue = stdoutQueue
        self.stdoutBuffer = []

    def setParams(self, **params):
        for key in params:
            if key in VideoWriter.settableParams:
                setattr(self, key, params[key])
                if self.verbose: syncPrint(self.ID + " - Param set: {key}={val}".format(key=key, val=params[key]), buffer=self.stdoutBuffer)
            else:
                syncPrint(self.ID + " - Param not settable: {key}={val}".format(key=key, val=params[key]), buffer=self.stdoutBuffer)

    def run(self):
        if self.verbose > 1: profiler = cProfile.Profile()
        syncPrint(self.ID + " - PID={pid}".format(pid=os.getpid()), buffer=self.stdoutBuffer)
        state = VideoWriter.STOPPED
        nextState = VideoWriter.STOPPED
        lastState = VideoWriter.STOPPED
        while True:
            try:
# ********************************* STOPPPED *********************************
                if state == VideoWriter.STOPPED:
                    # DO STUFF

                    # CHECK FOR MESSAGES
                    try:
                        msg, arg = self.messageQueue.get(block=True, timeout=0.1)
                        if msg == VideoWriter.SETPARAMS: self.setParams(**arg); msg = ''; arg=None
                    except queue.Empty: msg = ''; arg = None

                    # CHOOSE NEXT STATE
                    if self.exitFlag:
                        nextState = VideoWriter.EXITING
                    elif msg == '':
                        nextState = VideoWriter.STOPPED
                    elif msg == VideoWriter.START:
                        nextState = VideoWriter.INITIALIZING
                    elif msg == VideoWriter.EXIT:
                        self.exitFlag = True
                        nextState = VideoWriter.EXITING
                    else:
                        raise SyntaxError("Message \"" + msg + "\" not relevant to " + state + " state")
# ********************************* INITIALIZING *********************************
                elif state == VideoWriter.INITIALIZING:
                    # DO STUFF
                    triggers = []
                    imp = None
                    aviRecorder = None

                    # CHECK FOR MESSAGES
                    try:
                        msg, arg = self.messageQueue.get(block=False)
                        if msg == VideoWriter.SETPARAMS: self.setParams(**arg); msg = ''; arg=None
                    except queue.Empty: msg = ''; arg = None

                    # CHOOSE NEXT STATE
                    if self.exitFlag:
                        nextState = VideoWriter.STOPPING
                    elif msg == '':
                        nextState = VideoWriter.BUFFERING
                    elif msg == VideoWriter.STOP:
                        nextState = VideoWriter.STOPPING
                    elif msg == VideoWriter.EXIT:
                        self.exitFlag = True
                        nextState = VideoWriter.STOPPING
                    else:
                        raise SyntaxError("Message \"" + msg + "\" not relevant to " + state + " state")
# ********************************* BUFFERING *********************************
                elif state == VideoWriter.BUFFERING:
                    # DO STUFF
                    if self.verbose:
                        syncPrint("Image queue size: ", self.imageQueue.qsize(), buffer=self.stdoutBuffer)
                        syncPrint("Images in buffer: ", len(self.buffer), buffer=self.stdoutBuffer)
                    if len(self.buffer) >= self.buffer.maxlen:
                        # If buffer is full, pull oldest video frame from buffer
                        imp = self.buffer.popleft()
                        if self.verbose: syncPrint(self.ID + " - Pulling video frame from buffer. t="+str(imp.frameTime) + " (buffer: {len}/{maxlen})".format(len=len(self.buffer), maxlen=self.buffer.maxlen), buffer=self.stdoutBuffer)

                    try:
                        # Get new video frame and push it into the buffer
                        newImp = self.imageQueue.get(block=True, timeout=0.1)
                        if self.verbose: syncPrint(self.ID + " - Got video frame from acquirer. Pushing into the buffer. t="+str(newImp.frameTime), buffer=self.stdoutBuffer)
                        self.buffer.append(newImp)
                    except queue.Empty: # None available
                        newImp = None

                    # CHECK FOR MESSAGES
                    try:
                        msg, arg = self.messageQueue.get(block=False)
                        if msg == VideoWriter.SETPARAMS: self.setParams(**arg); msg = ''; arg=None
                    except queue.Empty: msg = ''; arg = None

                    # CHOOSE NEXT STATE
                    if msg == VideoWriter.STOP:
                        nextState = VideoWriter.STOPPING
                    elif msg == VideoWriter.EXIT:
                        self.exitFlag = True
                        nextState = VideoWriter.STOPPING
                    elif msg == VideoWriter.TRIGGER or len(triggers) > 0:
                        if self.verbose: syncPrint(self.ID + " - Trigger is active", buffer=self.stdoutBuffer)
                        # Either we are just receiving a trigger message, or a trigger was previously received and not started yet
                        if arg is not None:
                            # New or updated trigger from message
                            self.updateTriggers(triggers, arg)
                        if imp is not None:
                            # At least one video frame has been received - we can check if trigger period has begun
                            triggerState = triggers[0].state(imp.frameTime)
                            if self.verbose: syncPrint(self.ID + " - Trigger state: {state}".format(state=triggerState), buffer=self.stdoutBuffer)
                            if triggerState < 0:        # Time is before trigger range
                                nextState = VideoWriter.BUFFERING
                            elif triggerState == 0:     # Time is now in trigger range
                                timeWrote = 0
                                nextState = VideoWriter.WRITING
                            else:                       # Time is after trigger range
                                if self.verbose: syncPrint(self.ID + " - Missed trigger start by", triggerState, "seconds!", buffer=self.stdoutBuffer)
                                timeWrote = 0
                                nextState = VideoWriter.BUFFERING
                                triggers.pop(0)
                        else:
                            # No audio chunks have been received yet, can't evaluate if trigger time has begun yet
                            if self.verbose: syncPrint(self.ID + " - No frames yet, can't begin trigger yet (buffer: {len}/{maxlen})".format(len=len(self.buffer), maxlen=self.buffer.maxlen), buffer=self.stdoutBuffer)
                            nextState = VideoWriter.BUFFERING

                    elif msg == '':
                        nextState = VideoWriter.BUFFERING
                    else:
                        raise SyntaxError("Message \"" + msg + "\" not relevant to " + state + " state")
# ********************************* WRITING *********************************
                elif state == VideoWriter.WRITING:
                    if self.verbose > 1: profiler.enable()
                    # DO STUFF
                    if self.verbose:
                        syncPrint("Image queue size: ", self.imageQueue.qsize(), buffer=self.stdoutBuffer)
                        syncPrint("Images in buffer: ", len(self.buffer), buffer=self.stdoutBuffer)
                    if imp is not None:
                        if aviRecorder is None:
                            # Start new video file
                            videoFileName = generateFileName(directory=self.videoDirectory, baseName=self.videoBaseFilename, extension='', trigger=triggers[0])
                            aviRecorder = PySpin.SpinVideo()
                            option = PySpin.AVIOption()
                            option.frameRate = self.frameRate
                            if self.verbose: syncPrint(self.ID + " - Opening file to save video with frameRate ", option.frameRate, buffer=self.stdoutBuffer)
                            aviRecorder.Open(videoFileName, option)
                            stupidChangedVideoNameThanksABunchFLIR = videoFileName + '-0000.avi'
                            aviRecorder.videoFileName = stupidChangedVideoNameThanksABunchFLIR

                        # Reconstitute PySpin image from PickleableImage
                        im = PySpin.Image.Create(imp.width, imp.height, imp.offsetX, imp.offsetY, imp.pixelFormat, imp.data)
                        # Convert image to desired format
                        im = im.Convert(PySpin.PixelFormat_RGB8, PySpin.HQ_LINEAR)
                        # Write video frame to file that was previously retrieved from the buffer
                        if self.verbose:
                            timeWrote += 1/self.frameRate
                            syncPrint(self.ID + " - Writing video frame, dt="+str(timeWrote), buffer=self.stdoutBuffer)
                        aviRecorder.Append(im)
                        # try:
                        #     im.Release()
                        # except PySpin.SpinnakerException:
                        #     if self.verbose: syncPrint("Error releasing PySpin image after appending to AVI.", buffer=self.stdoutBuffer)
                        del im

                    try:
                        # Pop the oldest image frame from the back of the buffer.
                        imp = self.buffer.popleft()
                        if self.verbose: syncPrint(self.ID + " - Pulled image from buffer", buffer=self.stdoutBuffer)
                    except IndexError:
                        if self.verbose: syncPrint(self.ID + " - No images in buffer", buffer=self.stdoutBuffer)
                        imp = None  # Buffer was empty

                    # Pull new image from VideoAcquirer and add to the front of the buffer.
                    try:
                        newImp = self.imageQueue.get(True, 0.1)
                        if self.verbose: syncPrint(self.ID + " - Got image from acquirer. Pushing into buffer.", buffer=self.stdoutBuffer)
                        self.buffer.append(newImp)
                    except queue.Empty:
                        # No data in image queue yet - pass.
                        if self.verbose: syncPrint(self.ID + " - No images available from acquirer", buffer=self.stdoutBuffer)

                    # CHECK FOR MESSAGES (and consume certain messages that don't trigger state transitions)
                    try:
                        msg, arg = self.messageQueue.get(block=False)
                        if msg == VideoWriter.SETPARAMS: self.setParams(**arg); msg = ''; arg=None
                        elif msg == VideoWriter.TRIGGER: self.updateTriggers(triggers, arg); msg = ''; arg=None
                    except queue.Empty: msg = ''; arg = None

                    # CHOOSE NEXT STATE
                    if self.exitFlag:
                        nextState = VideoWriter.STOPPING
                    elif msg == VideoWriter.STOP:
                        nextState = VideoWriter.STOPPING
                    elif msg == VideoWriter.EXIT:
                        self.exitFlag = True
                        nextState = VideoWriter.STOPPING
                    elif len(triggers) > 0 and imp is not None:
                        triggerState = triggers[0].state(imp.frameTime)
                        if triggerState != 0:
                            # Frame is not in trigger period - return to buffering
                            if self.verbose: syncPrint(self.ID + " - Frame does not overlap trigger. Switching to buffering.", buffer=self.stdoutBuffer)
                            nextState = VideoWriter.BUFFERING
                            # Remove current trigger
                            if aviRecorder is not None:
                                # Done with trigger, close file and clear video file
                                aviRecorder.Close()
                                if self.mergeMessageQueue is not None:
                                    # Send file for AV merging:
                                    fileEvent = dict(
                                        filePath=aviRecorder.videoFileName,
                                        streamType=AVMerger.VIDEO,
                                        trigger=triggers[0],
                                        streamID=self.camSerial
                                    )
                                    if self.verbose: syncPrint(self.ID + " - Sending video filename to merger", buffer=self.stdoutBuffer)
                                    self.mergeMessageQueue.put((AVMerger.MERGE, fileEvent))
                                aviRecorder = None
                            triggers.pop(0)
                        else:
                            # Frame is in trigger period - continue writing
                            nextState = VideoWriter.WRITING
                    elif msg == '':
                        nextState = state
                    else:
                        raise SyntaxError("Message \"" + msg + "\" not relevant to " + state + " state")
                    if self.verbose > 1: profiler.disable()
# ********************************* STOPPING *********************************
                elif state == VideoWriter.STOPPING:
                    # DO STUFF
                    if aviRecorder is not None:
                        aviRecorder.Close()
                        if self.mergeMessageQueue is not None:
                            # Send file for AV merging:
                            fileEvent = dict(
                                filePath=aviRecorder.videoFileName,
                                streamType=AVMerger.VIDEO,
                                trigger=triggers[0],
                                streamID=self.camSerial
                            )
                            self.mergeMessageQueue.put((AVMerger.MERGE, fileEvent))
                        aviRecorder = None
                    triggers = []

                    # CHECK FOR MESSAGES
                    try:
                        msg, arg = self.messageQueue.get(block=False)
                        if msg == VideoWriter.SETPARAMS: self.setParams(**arg); msg = ''; arg=None
                    except queue.Empty: msg = ''; arg = None

                    # CHOOSE NEXT STATE
                    if self.exitFlag:
                        nextState = VideoWriter.STOPPED
                    elif msg == '':
                        nextState = VideoWriter.BUFFERING
                    elif msg == VideoWriter.STOP:
                        nextState = VideoWriter.STOPPED
                    elif msg == VideoWriter.EXIT:
                        self.exitFlag = True
                        nextState = VideoWriter.STOPPED
                    else:
                        raise SyntaxError("Message \"" + msg + "\" not relevant to " + state + " state")
# ********************************* ERROR *********************************
                elif state == VideoWriter.ERROR:
                    # DO STUFF
                    syncPrint(self.ID + " - ERROR STATE. Error messages:\n\n", buffer=self.stdoutBuffer)
                    syncPrint("\n\n".join(self.errorMessages), buffer=self.stdoutBuffer)
                    self.errorMessages = []

                    # CHECK FOR MESSAGES
                    try:
                        msg, arg = self.messageQueue.get(block=False)
                        if msg == VideoWriter.SETPARAMS: self.setParams(**arg); msg = ''; arg=None
                    except queue.Empty: msg = ''; arg = None

                    # CHOOSE NEXT STATE
                    if msg == '':
                        nextState = VideoWriter.STOPPING
                    elif msg == VideoWriter.STOP:
                        nextState = VideoWriter.STOPPED
                    elif msg == VideoWriter.EXIT:
                        self.exitFlag = True
                        if lastState == VideoWriter.STOPPING:
                            nextState = VideoWriter.EXITING
                        else:
                            nextState = VideoWriter.STOPPING
                    else:
                        raise SyntaxError("Message \"" + msg + "\" not relevant to " + state + " state")
# ********************************* EXIT *********************************
                elif state == VideoWriter.EXITING:
                    break
                else:
                    raise KeyError("Unknown state: "+state)
            except KeyboardInterrupt:
                # Handle user using keyboard interrupt
                if self.verbose: syncPrint(self.ID + " - Keyboard interrupt received - exiting", buffer=self.stdoutBuffer)
                self.exitFlag = True
                nextState = VideoWriter.STOPPING
            except:
                # HANDLE UNKNOWN ERROR
                self.errorMessages.append("Error in "+state+" state\n\n"+traceback.format_exc())
                nextState = VideoWriter.ERROR

            if self.verbose:
                if msg != '' or self.exitFlag:
                    syncPrint("msg={msg}, exitFlag={exitFlag}".format(msg=msg, exitFlag=self.exitFlag), buffer=self.stdoutBuffer)
                syncPrint('*********************************** /\ ' + self.ID + ' ' + state + ' /\ ********************************************', buffer=self.stdoutBuffer)
            if len(self.stdoutBuffer) > 0: self.stdoutQueue.put(self.stdoutBuffer)
            self.stdoutBuffer = []

            # Prepare to advance to next state
            lastState = state
            state = nextState

        if self.verbose:
            syncPrint("Video write process STOPPED", buffer=self.stdoutBuffer)
        if self.verbose > 1:
            s = io.StringIO()
            ps = pstats.Stats(profiler, stream=s)
            ps.print_stats()
            syncPrint(s.getvalue(), buffer=self.stdoutBuffer)
        if len(self.stdoutBuffer) > 0: self.stdoutQueue.put(self.stdoutBuffer)
        self.stdoutBuffer = []

    def updateTriggers(self, triggers, trigger):
        if len(triggers) > 0 and trigger.id == triggers[-1].id:
            # This is an updated trigger, not a new trigger
            if self.verbose: syncPrint(self.ID + " - Updating trigger", buffer=self.stdoutBuffer)
            triggers[-1] = trigger
        else:
            # This is a new trigger
            if self.verbose: syncPrint(self.ID + " - Adding new trigger", buffer=self.stdoutBuffer)
            triggers.append(trigger)

def getCameraAttribute(nodemap, attributeName, attributeTypePtrFunction):
    nodeAttribute = attributeTypePtrFunction(nodemap.GetNode(attributeName))

    if not PySpin.IsAvailable(nodeAttribute) or not PySpin.IsReadable(nodeAttribute):
        print('Unable to retrieve '+attributeName+'. Aborting...')
        return None

    try:
        value = nodeAttribute.GetValue()
    except AttributeError:
        # Maybe it's an enum?
        valueEntry = nodeAttribute.GetCurrentEntry()
        value = (valueEntry.GetName(), valueEntry.GetDisplayName())
    return value

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

nodeAccessorFunctions = {
    PySpin.intfIString:('string', PySpin.CStringPtr),
    PySpin.intfIInteger:('integer', PySpin.CIntegerPtr),
    PySpin.intfIFloat:('float', PySpin.CFloatPtr),
    PySpin.intfIBoolean:('boolean', PySpin.CBooleanPtr),
    PySpin.intfICommand:('command', PySpin.CEnumerationPtr),
    PySpin.intfIEnumeration:('enum', PySpin.CEnumerationPtr),
    PySpin.intfICategory:('category', PySpin.CCategoryPtr)
}

nodeAccessorTypes = {
    'string':PySpin.CStringPtr,
    'integer':PySpin.CIntegerPtr,
    'float':PySpin.CFloatPtr,
    'boolean':PySpin.CBooleanPtr,
    'command':PySpin.CEnumerationPtr,
    'enum':PySpin.CEnumerationPtr,
    'category':PySpin.CCategoryPtr
}

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

class StreamType:
    AUDIO=0
    VIDEO=1

class EndpointType:
    ACQUIRE=0
    WRITE=1

class PyVAQ:
    lineStyles = [c+'-' for c in 'bgrcmyk']
    def __init__(self, master):
        self.master = master
        self.master.title("PyVAQ")
        self.master.protocol("WM_DELETE_WINDOW", self.cleanup)
        self.style = ttk.Style()
        self.style.theme_use('default')
        self.style.configure('TEntry', borderwidth=0, fieldbackground='white')
        self.style.configure('ValidDirectory.TEntry', fieldbackground='#C1FFC1')
        self.style.configure('InvalidDirectory.TEntry', fieldbackground='#FFC1C1')
#        self.style.map('Directory.TEntry.label', background=[(('!invalid',), 'green'),(('invalid',), 'red')])
        self.cameraAttributes = {}
        self.camSerials = discoverCameras()
        self.cameraSpeeds = dict([(camSerial, checkCameraSpeed(camSerial)) for camSerial in self.camSerials])
        # self.updateAllCamerasAttributes()
        # with open('attributes.txt', 'w') as f:
        #     pp = pprint.PrettyPrinter(stream=f, indent=2)
        #     pp.pprint(self.cameraAttributes)

        self.audioDAQChannels = ['Dev3/ai5']
        self.audioSyncSource = 'PFI4'
        self.audioSyncTerminal = 'Dev3/ctr0'
        self.videoSyncSource = 'PFI5'
        self.videoSyncTerminal = 'Dev3/ctr1'

        ########### GUI WIDGETS #####################
        self.settings = []  # A list of StringVar/IntVar/etc that are settings to save/restore

        self.monitorFrame = ttk.Frame(self.master)

        self.videoMonitorMasterFrame = ttk.Frame(self.monitorFrame)
        self.videoMonitorFrames = {}
        self.videoMonitors = {}
        self.imageIDs = {}
        self.currentImages = {}
        self.cameraAttributeBrowserButtons = {}
        self.videoStateWidgets = {}
        self.canvasSize = (400, 600)
        for camSerial in self.camSerials:
            self.videoMonitorFrames[camSerial] = vFrame = ttk.LabelFrame(self.videoMonitorMasterFrame, text="{serial} ({speed})".format(serial=camSerial, speed=self.cameraSpeeds[camSerial]))
            self.videoMonitors[camSerial] = tk.Canvas(vFrame, width=self.canvasSize[0], height=self.canvasSize[1], borderwidth=2, relief=tk.SUNKEN)
            self.imageIDs[camSerial] = None
            self.currentImages[camSerial] = None
            self.cameraAttributeBrowserButtons[camSerial] = ttk.Button(vFrame, text="Attribute browser", command=lambda:self.createCameraAttributeBrowser(camSerial))
            self.videoStateWidgets[camSerial] = ttk.Label(vFrame)

        # Audio stream monitoring widgets
        self.audioMonitorSampleSize = 44100*5
        self.audioDAQChannelWidgets = {}
        self.audioMonitorMasterFrame = ttk.Frame(self.monitorFrame)
        self.audioMonitorFrames = {}
        self.audioStateWidgets = {}
        for k, channel in enumerate(self.audioDAQChannels):
            self.audioMonitorFrames[channel] = aFrame = ttk.LabelFrame(self.audioMonitorMasterFrame, text=channel)
            self.audioStateWidgets[channel] = ttk.Label(aFrame)
            self.createAudioMonitor(channel, k)

        self.controlFrame = ttk.Frame(self.master)

        self.acquisitionFrame = ttk.LabelFrame(self.controlFrame, text="Acquisition")
        self.startAcquisitionButton = ttk.Button(self.acquisitionFrame, text="Start acquisition", command=self.acquireButtonClick)

        self.audioFrequencyFrame =  ttk.LabelFrame(self.acquisitionFrame, text="Audio freq. (Hz)")
        self.audioFrequencyVar =    tk.StringVar(); self.audioFrequencyVar.set("22010"); self.settings.append('audioFrequencyVar')
        self.audioFrequencyEntry =  ttk.Entry(self.audioFrequencyFrame, width=15, textvariable=self.audioFrequencyVar);

        self.videoFrequencyFrame =  ttk.LabelFrame(self.acquisitionFrame, text="Video freq (fps)")
        self.videoFrequencyVar =    tk.StringVar(); self.videoFrequencyVar.set("30"); self.settings.append('videoFrequencyVar')
        self.videoFrequencyEntry =  ttk.Entry(self.videoFrequencyFrame, width=15, textvariable=self.videoFrequencyVar)

        self.exposureTimeFrame =    ttk.LabelFrame(self.acquisitionFrame, text="Exposure time (us):")
        self.exposureVar =          tk.StringVar(); self.exposureVar.set("8000"); self.settings.append('exposureVar')
        self.exposureTimeEntry =    ttk.Entry(self.exposureTimeFrame, width=18, textvariable=self.exposureVar)

        self.preTriggerTimeFrame =  ttk.LabelFrame(self.acquisitionFrame, text="Pre-trigger record time (s)")
        self.preTriggerTimeVar =    tk.StringVar(); self.preTriggerTimeVar.set("2.0"); self.settings.append('preTriggerTimeVar')
        self.preTriggerTimeEntry =  ttk.Entry(self.preTriggerTimeFrame, width=26, textvariable=self.preTriggerTimeVar)

        self.recordTimeFrame =      ttk.LabelFrame(self.acquisitionFrame, text="Record time (s)")
        self.recordTimeVar =        tk.StringVar(); self.recordTimeVar.set("4.0"); self.settings.append('recordTimeVar')
        self.recordTimeEntry =      ttk.Entry(self.recordTimeFrame, width=14, textvariable=self.recordTimeVar)

        self.baseFileNameFrame =    ttk.LabelFrame(self.acquisitionFrame, text="Base write filename")
        self.baseFileNameVar =      tk.StringVar(); self.baseFileNameVar.set("recording"); self.settings.append("baseFileNameVar")
        self.baseFileNameEntry =    ttk.Entry(self.baseFileNameFrame, width=24, textvariable=self.baseFileNameVar)

        self.directoryFrame =       ttk.LabelFrame(self.acquisitionFrame, text="Write directory")
        self.directoryVar =         tk.StringVar(); self.directoryVar.set(""); self.settings.append("directoryVar")
        self.directoryEntry =       ttk.Entry(self.directoryFrame, width=48, textvariable=self.directoryVar, style='ValidDirectory.TEntry')
        self.directoryButton =      ttk.Button(self.directoryFrame, text="Select write directory", command=self.selectWriteDirectory)
        self.directoryEntry.bind('<FocusOut>', self.directoryChangeHandler)

        self.mergeFrame = ttk.LabelFrame(self.acquisitionFrame, text="AV File merging")

        self.mergeFilesVar =        tk.BooleanVar(); self.mergeFilesVar.set(True); self.settings.append('mergeFilesVar')
        self.mergeFilesCheckbutton = ttk.Checkbutton(self.mergeFrame, text="Merge audio/video", variable=self.mergeFilesVar, offvalue=False, onvalue=True)
        self.mergeFilesVar.trace('w', self.updateAVMergerState)

        self.deleteMergedFilesVar = tk.BooleanVar(); self.deleteMergedFilesVar.set(False); self.settings.append('deleteMergedFilesVar')
        self.deleteMergedFilesCheckbutton = ttk.Checkbutton(self.mergeFrame, text="Delete merged files", variable=self.deleteMergedFilesVar, offvalue=False, onvalue=True)
        self.deleteMergedFilesVar.trace('w', lambda *args: self.changeAVMergerParams(deleteMergedFiles=self.deleteMergedFilesVar.get()))

        self.montageMergeVar = tk.BooleanVar(); self.montageMergeVar.set(False); self.settings.append('montageMergeVar')
        self.montageMergeCheckbutton = ttk.Checkbutton(self.mergeFrame, text="Montage-merge videos", variable=self.montageMergeVar, offvalue=False, onvalue=True)
        self.montageMergeVar.trace('w', lambda *args: self.changeAVMergerParams(montage=self.montageMergeVar.get()))

        self.scheduleEnabledVar = tk.BooleanVar(); self.scheduleEnabledVar.set(False); self.settings.append('scheduleEnabledVar')
        self.scheduleEnabledCheckbutton = ttk.Checkbutton(self.acquisitionFrame, text="Enable schedule", variable=self.scheduleEnabledVar)

        self.scheduleStartVar = TimeVar(); self.settings.append('scheduleStartVar')
        self.scheduleStartTimeEntry = TimeEntry(self.acquisitionFrame, text="Start time", style=self.style)

        self.scheduleStopVar = TimeVar(); self.settings.append('scheduleStopVar')
        self.scheduleStopTimeEntry = TimeEntry(self.acquisitionFrame, text="Stop time")

        self.saveSettingsButton = ttk.Button(self.acquisitionFrame, text="Save settings", command=self.saveSettings)
        self.loadSettingsButton = ttk.Button(self.acquisitionFrame, text="Load settings", command=self.loadSettings)
        self.saveSettingsButton = ttk.Button(self.acquisitionFrame, text="Save defaults", command=lambda *args: self.saveSettings(path='default'))
        self.loadSettingsButton = ttk.Button(self.acquisitionFrame, text="Load defaults", command=lambda *args: self.loadSettings(path='default'))

        self.triggerFrame = ttk.LabelFrame(self.controlFrame, text='Triggering')
        self.triggerModes = ['Manual', 'Audio']
        self.triggerModeChooserFrame = ttk.Frame(self.triggerFrame)
        self.triggerModeVar = tk.StringVar(); self.triggerModeVar.set(self.triggerModes[0]); self.settings.append('triggerModeVar')
        self.triggerModeVar.trace('w', self.switchTriggerMode)
        self.triggerModeLabel = ttk.Label(self.triggerModeChooserFrame, text='Trigger mode')
        self.triggerModeRadioButtons = {}
        self.triggerModeControlGroupFrames = {}

        for mode in self.triggerModes:
            self.triggerModeRadioButtons[mode] = ttk.Radiobutton(self.triggerModeChooserFrame, text=mode, variable=self.triggerModeVar, value=mode)
            self.triggerModeControlGroupFrames[mode] = ttk.Frame(self.triggerFrame)

        # Manual controls
        self.manualWriteTriggerButton = ttk.Button(self.triggerModeControlGroupFrames['Manual'], text="Manual write trigger", command=self.writeButtonClick)

        # Audio trigger controls
        self.triggerHighLevelFrame = ttk.LabelFrame(self.triggerModeControlGroupFrames['Audio'], text="High vol. threshold")
        self.triggerHighLevelVar = tk.StringVar(); self.triggerHighLevelVar.set("1.5"); self.triggerHighLevelVar.trace('w', self.updateAudioTriggerSettings); self.settings.append('triggerHighLevelVar')
        self.triggerHighLevelEntry = ttk.Entry(self.triggerHighLevelFrame, textvariable=self.triggerHighLevelVar)

        self.triggerLowLevelFrame = ttk.LabelFrame(self.triggerModeControlGroupFrames['Audio'], text="Low vol. threshold")
        self.triggerLowLevelVar = tk.StringVar(); self.triggerLowLevelVar.set("0.5"); self.triggerLowLevelVar.trace('w', self.updateAudioTriggerSettings); self.settings.append('triggerLowLevelVar')
        self.triggerLowLevelEntry = ttk.Entry(self.triggerLowLevelFrame, textvariable=self.triggerLowLevelVar)

        self.triggerHighTimeFrame = ttk.LabelFrame(self.triggerModeControlGroupFrames['Audio'], text="High threshold time")
        self.triggerHighTimeVar = tk.StringVar(); self.triggerHighTimeVar.set("3.0"); self.triggerHighTimeVar.trace('w', self.updateAudioTriggerSettings); self.settings.append('triggerHighTimeVar')
        self.triggerHighTimeEntry = ttk.Entry(self.triggerHighTimeFrame, textvariable=self.triggerHighTimeVar)

        self.triggerLowTimeFrame = ttk.LabelFrame(self.triggerModeControlGroupFrames['Audio'], text="Low threshold time")
        self.triggerLowTimeVar = tk.StringVar(); self.triggerLowTimeVar.set("5.0"); self.triggerLowTimeVar.trace('w', self.updateAudioTriggerSettings); self.settings.append('triggerLowTimeVar')
        self.triggerLowTimeEntry = ttk.Entry(self.triggerLowTimeFrame, textvariable=self.triggerLowTimeVar)

        self.triggerHighFractionFrame = ttk.LabelFrame(self.triggerModeControlGroupFrames['Audio'], text="Frac. of time above high threshold")
        self.triggerHighFractionVar = tk.StringVar(); self.triggerHighFractionVar.set("0.7"); self.triggerHighFractionVar.trace('w', self.updateAudioTriggerSettings); self.settings.append('triggerHighFractionVar')
        self.triggerHighFractionEntry = ttk.Entry(self.triggerHighFractionFrame, textvariable=self.triggerHighFractionVar)

        self.triggerLowFractionFrame = ttk.LabelFrame(self.triggerModeControlGroupFrames['Audio'], text="Frac. of time below low threshold")
        self.triggerLowFractionVar = tk.StringVar(); self.triggerLowFractionVar.set("0.4"); self.triggerLowFractionVar.trace('w', self.updateAudioTriggerSettings); self.settings.append('triggerLowFractionVar')
        self.triggerLowFractionEntry = ttk.Entry(self.triggerLowFractionFrame, textvariable=self.triggerLowFractionVar)

        self.maxAudioTriggerTimeFrame = ttk.LabelFrame(self.triggerModeControlGroupFrames['Audio'], text="Max. audio trigger record time")
        self.maxAudioTriggerTimeVar = tk.StringVar(); self.maxAudioTriggerTimeVar.set("20.0"); self.maxAudioTriggerTimeVar.trace('w', self.updateAudioTriggerSettings); self.settings.append('maxAudioTriggerTimeVar')
        self.maxAudioTriggerTimeEntry = ttk.Entry(self.maxAudioTriggerTimeFrame, textvariable=self.maxAudioTriggerTimeVar)

        self.multiChannelStartBehaviorFrame = ttk.LabelFrame(self.triggerModeControlGroupFrames['Audio'], text="Start recording when...")
        self.multiChannelStartBehaviorVar = tk.StringVar(); self.multiChannelStartBehaviorVar.set("OR"); self.multiChannelStartBehaviorVar.trace('w', self.updateAudioTriggerSettings); self.settings.append('multiChannelStartBehaviorVar')
        self.multiChannelStartBehaviorOR = ttk.Radiobutton(self.multiChannelStartBehaviorFrame, text="...any channels stay above threshold", variable=self.multiChannelStartBehaviorVar, value="OR")
        self.multiChannelStartBehaviorAND = ttk.Radiobutton(self.multiChannelStartBehaviorFrame, text="...all channels stay above threshold", variable=self.multiChannelStartBehaviorVar, value="AND")

        self.multiChannelStopBehaviorFrame = ttk.LabelFrame(self.triggerModeControlGroupFrames['Audio'], text="Stop recording when...")
        self.multiChannelStopBehaviorVar = tk.StringVar(); self.multiChannelStopBehaviorVar.set("OR"); self.multiChannelStopBehaviorVar.trace('w', self.updateAudioTriggerSettings); self.settings.append('multiChannelStopBehaviorVar')
        self.multiChannelStopBehaviorOR = ttk.Radiobutton(self.multiChannelStopBehaviorFrame, text="...any channels stay below threshold", variable=self.multiChannelStopBehaviorVar, value="OR")
        self.multiChannelStopBehaviorAND = ttk.Radiobutton(self.multiChannelStopBehaviorFrame, text="...all channels stay below threshold", variable=self.multiChannelStopBehaviorVar, value="AND")

        ########### Child process objects #####################

        # Monitoring queues for collecting qudio and video data for user monitoring purposes
        self.videoMonitorQueues = None
        self.audioMonitorQueue = None
        self.audioAnalysisQueue = None
        self.mergeMessageQueue = None
        self.monitorFrameRate = 15
        self.audioMonitorData = np.zeros((len(self.audioDAQChannels), self.audioMonitorSampleSize))
        self.stdoutQueue = None

        # Message queues for sending commands to processes
        self.audioAcquireMessageQueue = None
        self.audioWriteMessageQueue = None
        self.syncMessageQueue = None
        self.videoAcquireMessageQueues = {}
        self.videoWriteMessageQueues = {}
        self.mergeMessageQueue = None
        self.audioTriggerMessageQueue = None

        # Pointers to processes
        self.videoWriteProcesses = {}
        self.videoAcquireProcesses = {}
        self.audioWriteProcess = None
        self.audioAcquireProcess = None
        self.syncProcess = None
        self.mergeProcess = None
        self.stdoutManager = None

        # Verbosity of child processes
        self.audioAcquireVerbose = False
        self.audioWriteVerbose = False
        self.videoAcquireVerbose = False
        self.videoWriteVerbose = False
        self.syncVerbose = False
        self.mergeVerbose = False
        self.audioTriggerVerbose = False

        self.profiler =  cProfile.Profile()

        self.update()

        # Start automatic updating of video and audio monitors
        self.audioMonitorUpdateJob = None
        self.videoMonitorUpdateJob = None
        self.autoUpdateVideoMonitors()
        self.autoUpdateAudioMonitors()

    # def createSetting(self, settingName, parent, varType, initialValue, labelText, width=None):
    #     # Creates a set of widgets (label, input widget, variable). Only good for Entry-type inputs
    #     newVar = varType()
    #     newVar.set(initialValue)
    #     newEntry = ttk.Entry(parent, width=width, )
    #     setattr(self, settingName+"Var")

    def saveSettings(self, *args, path=None):
        pass

    def loadSettings(self, *args, path=None):
        pass

    def cleanup(self):
        # Cancel automatic update jobs
        if self.audioMonitorUpdateJob is not None:
            self.master.after_cancel(self.audioMonitorUpdateJob)
        if self.videoMonitorUpdateJob is not None:
            self.master.after_cancel(self.videoMonitorUpdateJob)
        print("Stopping acquisition")
        self.stopAcquisition()
        print("Destroying master")
        self.master.destroy()
        self.master.quit()
        print("Everything should be closed now!")

    def updateAudioTriggerSettings(self, *args):
        if self.audioTriggerMessageQueue is not None:
            print("Updating audio triggerer settings")
            print("args are:", args)
            params = dict(
                triggerHighLevel=self.triggerHighLevelVar.get(),
                triggerLowLevel=self.triggerLowLevelVar.get(),
                triggerHighTime=self.triggerHighTimeVar.get(),
                triggerLowTime=self.triggerLowTimeVar.get(),
                triggerHighFraction=self.triggerHighFractionVar.get(),
                triggerLowFraction=self.triggerLowFractionVar.get(),
                maxAudioTriggerTime=self.maxAudioTriggerTimeVar.get(),
                multiChannelStartBehavior=self.multiChannelStartBehaviorVar.get(),
                multiChannelStopBehavior=self.multiChannelStopBehaviorVar.get()
            )
            self.audioTriggerMessageQueue.put((AudioTriggerer.PARAMSET, params))

    def updateAVMergerState(self, *args):
        merging = self.mergeFilesVar.get()
        if merging:
            self.deleteMergedFilesCheckbutton.config(state=tk.NORMAL)
            self.montageMergeCheckbutton.config(state=tk.NORMAL)
        else:
            self.deleteMergedFilesCheckbutton.config(state=tk.DISABLED)
            self.montageMergeCheckbutton.config(state=tk.DISABLED)

        if self.mergeMessageQueue is not None:
            if merging:
                self.mergeMessageQueue.put((AVMerger.START, None))
            else:
                self.mergeMessageQueue.put((AVMerger.CHILL, None))

    def changeAVMergerParams(self, **params):
        if self.mergeMessageQueue is not None:
            self.mergeMessageQueue.put((AVMerger.SETPARAMS, params))

    def directoryChangeHandler(self, *args):
        newDir = self.directoryVar.get()
        if len(newDir) == 0 or os.path.isdir(newDir):
            for camSerial in self.videoWriteMessageQueues:
                self.videoWriteMessageQueues[camSerial].put((VideoWriter.SETPARAMS, dict(videoDirectory=newDir)))
            if self.audioWriteMessageQueue is not None:
                self.audioWriteMessageQueue.put((AudioWriter.SETPARAMS, dict(audioDirectory=newDir)))
            if self.mergeMessageQueue is not None:
                self.mergeMessageQueue.put((AVMerger.SETPARAMS, dict(directory=newDir)))
            self.directoryEntry['style'] = 'ValidDirectory.TEntry'
        else:
            self.directoryEntry['style'] = 'InvalidDirectory.TEntry'


    def selectWriteDirectory(self, *args):
        directory = askdirectory(
#            initialdir = ,
#            message = "Choose a directory to write video and audio files to.",
            mustexist = False,
            title = "Choose a directory to write video and audio files to."
        )
        if len(directory) > 0:
            self.directoryVar.set(directory)
            self.directoryEntry.xview_moveto(0.5)
            self.directoryEntry.update_idletasks()
            self.directoryChangeHandler()

    def switchTriggerMode(self, *args):
        newMode = self.triggerModeVar.get()
        print("Switching trigger mode to ")
        self.update()

    def setAudioVideoState(self, streamType, ID, readState, writeState):
        if streamType == StreamType.AUDIO:
            widget = self.audioStateWidgets[ID]
        elif streamType == StreamType.VIDEO:
            widget = self.videoStateWidgets[ID]
        else:
            raise Exception("Unknown stream type")
        state = ('ACQUIRING' if readState else '') + '|' + ('WRITING' if writeState else '')
        widget.config(text=state)

    def createAudioMonitor(self, channel, index):
        self.audioDAQChannelWidgets[channel] = {}  # Change this to gracefully remove existing channel widgets under this channel name
        fig = Figure(figsize=(7, 0.75), dpi=100)
        t = np.arange(self.audioMonitorSampleSize)
        axes = fig.add_subplot(111)
        axes.autoscale(enable=True)
        axes.plot(t, 0 * t, PyVAQ.lineStyles[index], linewidth=1)
        axes.relim()
        axes.autoscale_view(True, True, True)
        axes.margins(x=0, y=0)
        #fig.tight_layout()
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0, hspace=0)
        canvas = FigureCanvasTkAgg(fig, master=self.audioMonitorFrames[channel])  # A tk.DrawingArea.
        canvas.draw()

        # Set up matplotlib figure callbacks
        # toolbar = NavigationToolbar2Tk(canvas, self.audioMonitorFrames[channel])
        # toolbar.update()
#         def figureKeyPressManager(event):
# #            syncPrint("you pressed {}".format(event.key))
#             key_press_handler(event, canvas, toolbar)
#         canvas.mpl_connect("key_press_event", figureKeyPressManager)

        self.audioDAQChannelWidgets[channel]['figure'] = fig
        self.audioDAQChannelWidgets[channel]['axes'] = axes
        self.audioDAQChannelWidgets[channel]['figureCanvas'] = canvas
        # self.audioDAQChannelWidgets[channel]['figureNavToolbar'] = toolbar
        # self.audioDAQChannelWidgets[channel]['figureLine'] = line

    def autoUpdateAudioMonitors(self, beginAuto=True):
        if self.audioMonitorQueue is not None:
            try:
                # Get audio data from monitor queue
                channels, audioData = self.audioMonitorQueue.get(True, 0.01)
                if self.audioMonitorData is None:
                    # Create monitor data
                    self.audioMonitorData = audioData
                else:
                    # Append new data
                    self.audioMonitorData = np.concatenate((self.audioMonitorData, audioData), axis=1)
                    # Trim monitor data to specified size
                    startTrim = self.audioMonitorData.shape[1] - self.audioMonitorSampleSize
                    if startTrim < 0:
                        startTrim = 0
                    self.audioMonitorData = self.audioMonitorData[:, startTrim:]
                for k, channel in enumerate(channels):
                    del self.audioDAQChannelWidgets[channel]['axes'].lines[0]
                    self.audioDAQChannelWidgets[channel]['axes'].plot(self.audioMonitorData[k, :].tolist(), PyVAQ.lineStyles[k], linewidth=1)
                    self.audioDAQChannelWidgets[channel]['axes'].relim()
                    self.audioDAQChannelWidgets[channel]['axes'].autoscale_view(True, True, True)
                    self.audioDAQChannelWidgets[channel]['axes'].margins(x=0, y=0)
                    self.audioDAQChannelWidgets[channel]['figure'].canvas.draw()
                    self.audioDAQChannelWidgets[channel]['figure'].canvas.flush_events()
            except queue.Empty:
                pass

        if beginAuto:
            self.audioMonitorUpdateJob = self.master.after(100, self.autoUpdateAudioMonitors)

    def autoUpdateVideoMonitors(self):
#        self.profiler.enable()
        if self.videoMonitorQueues is not None:
            availableImages = {}
            for camSerial in self.videoMonitorQueues:
                try:
                    pImage = self.videoMonitorQueues[camSerial].get(block=False)
                    availableImages[camSerial] = pImage  # Discard older images
                except queue.Empty:
                    pass

            for camSerial in availableImages:   # Display the most recent available image for each camera
                # syncPrint("Received frame for monitoring")
                pImage = availableImages[camSerial]
                imData = np.reshape(pImage.data, (pImage.height, pImage.width))
                im = Image.fromarray(imData).resize(self.canvasSize, resample=Image.BILINEAR)
                self.currentImages[camSerial] = ImageTk.PhotoImage(im)
                if self.imageIDs[camSerial] is None:
                    self.imageIDs[camSerial] = self.videoMonitors[camSerial].create_image((0, 0), image=self.currentImages[camSerial], anchor=tk.NW)
                else:
                    self.videoMonitors[camSerial].itemconfig(self.imageIDs[camSerial], image=self.currentImages[camSerial])
    #            self.videoMonitors[camSerial].update_idletasks()

#        self.profiler.disable()
        period = int(round(1000.0/(self.monitorFrameRate)))
        self.videoMonitorUpdateJob = self.master.after(period, self.autoUpdateVideoMonitors)

    def monitorProcesses(self): #, processList, processNameList, queueList, queueNameList):
        atLeastOneProcessAlive = False
        for camSerial in self.videoWriteProcesses:
            p_write = self.videoWriteProcesses[camSerial]
            p_acquire = self.videoAcquireProcesses[camSerial]
            writer_alive = p_write.is_alive()
            acquirer_alive = p_acquire.is_alive()
            atLeastOneProcessAlive = atLeastOneProcessAlive or writer_alive or acquirer_alive
            self.setAudioVideoState(StreamType.VIDEO, camSerial, acquirer_alive, writer_alive)
        if self.audioWriteProcess is not None or self.audioAcquireProcess is not None:
            p_write = self.audioWriteProcess
            p_acquire = self.audioAcquireProcess
            writer_alive = p_write.is_alive()
            acquirer_alive = p_acquire.is_alive()
            atLeastOneProcessAlive = atLeastOneProcessAlive or writer_alive or acquirer_alive
            print("AUDIO", "\tAQ:", acquirer_alive, "\tWR:", writer_alive)
            for channel in self.audioDAQChannels:
                self.setAudioVideoState(StreamType.AUDIO, channel, acquirer_alive, writer_alive)

        if atLeastOneProcessAlive:
            self.master.after(2000, self.monitorProcesses)
        else:
            print("Acquiring and writing audio and video complete!")

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
        # pp.psyncPrint(attributeNode)
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

    def acquisitionActive(self):
        # We should implement a shared-state multiprocessingl.Value object to publish state machine current states
        # For now we'll just check if processes are alive
        for camSerial in self.videoAcquireProcesses:
            if self.videoAcquireProcesses[camSerial] is not None and self.videoAcquireProcesses[camSerial].is_alive():
                return True
        if self.audioAcquireProcess is not None and self.audioAcquireProcess.is_alive():
            return True
        # No acquire processes alive
        return False

    def acquireButtonClick(self):
        print('acquisition is currently active? ', self.acquisitionActive())
        if self.acquisitionActive():
            print('stopping acqusition, and setting up button to start')
            self.stopAcquisition()
            self.startAcquisitionButton.config(text="Start acquisition")
        else:
            print('starting acqusition, and setting up button to stop')
            self.startAcquisition()
            self.startAcquisitionButton.config(text= "Stop acquisition")

    def writeButtonClick(self):
        self.sendWriteTrigger()

    def startAcquisition(self):
        audioFrequency = int(self.audioFrequencyEntry.get())
        videoFrequency = int(self.videoFrequencyEntry.get())
        exposureTime = float(self.exposureTimeEntry.get())
        preTriggerTime = float(self.preTriggerTimeVar.get())
        recordTime = float(self.recordTimeVar.get())
        baseFileName = self.baseFileNameVar.get()
        directory = self.directoryVar.get()
        mergeFiles = self.mergeFilesVar.get()
        deleteMergedFiles = self.deleteMergedFilesVar.get()
        scheduleEnabled = self.scheduleEnabledVar.get()
        scheduleStart = self.scheduleStartVar.get()
        scheduleStop = self.scheduleStopVar.get()
        triggerMode = self.triggerModeVar.get()
        triggerHighLevel = float(self.triggerHighLevelVar.get())
        triggerLowLevel = float(self.triggerLowLevelVar.get())
        triggerHighTime = float(self.triggerHighTimeVar.get())
        triggerLowTime = float(self.triggerLowTimeVar.get())
        triggerHighFraction = float(self.triggerHighFractionVar.get())
        triggerLowFraction = float(self.triggerLowFractionVar.get())
        maxAudioTriggerTime = float(self.maxAudioTriggerTimeVar.get())
        multiChannelStartBehavior = self.multiChannelStartBehaviorVar.get()
        multiChannelStopBehavior = self.multiChannelStopBehaviorVar.get()


        numStreams = (len(self.audioDAQChannels)>0) + len(self.camSerials)
        numProcesses = (len(self.audioDAQChannels)>0) + len(self.camSerials)*2 + 2
        numSyncedProcesses = (len(self.audioDAQChannels)>0) + len(self.camSerials) + 1  # 0 or 1 audio acquire processes, N video acquire processes, and 1 sync process

        print("Num synced processes:", numSyncedProcesses)
        ready = mp.Barrier(numSyncedProcesses)
        # manualTriggerProcess = spawnManualTriggerProcess(channelName='Dev3/port1/line0')

        self.stdoutQueue = mp.Queue()
        self.stdoutManager = stdoutManager(self.stdoutQueue)
        self.stdoutManager.start()

        self.videoMonitorQueues = {}
        self.audioMonitorQueue = mp.Queue()
        self.audioAnalysisQueue = mp.Queue()
        self.mergeMessageQueue = mp.Queue()
        self.audioTriggerMessageQueue = mp.Queue()

        startTime = mp.Value('d', -1)

        if exposureTime >= 1000000 * 0.95/videoFrequency:
            oldExposureTime = exposureTime
            exposureTime = 1000000*0.95/videoFrequency
            print()
            print("******WARNING*******")
            print()
            print("Exposure time is too long to achieve requested frame rate!")
            print("Shortening exposure time from {a}us to {b}us".format(a=oldExposureTime, b=exposureTime))
            print()
            print("********************")
            print()

        bufferSizeSeconds = preTriggerTime * 2 + 1   # Twice the pretrigger time to make sure we don't miss stuff, plus one second for good measure
        chunkSize = 1000
        bufferSizeAudioChunks = bufferSizeSeconds * audioFrequency / chunkSize   # Will be rounded up to nearest integer
        if len(self.audioDAQChannels) > 0:
            audioQueue = mp.Queue()
            self.audioAcquireMessageQueue = mp.Queue()
            self.audioWriteMessageQueue = mp.Queue()
            audioBaseFilename = slugify(baseFileName+'_'+','.join(self.audioDAQChannels))
            self.audioWriteProcess = AudioWriter(
                audioDirectory=directory,
                audioBaseFileName=audioBaseFilename,
                audioQueue=audioQueue,
                messageQueue=self.audioWriteMessageQueue,
                mergeMessageQueue=self.mergeMessageQueue,
                chunkSize=chunkSize,
                bufferSizeSeconds=bufferSizeSeconds,
                audioFrequency=audioFrequency,
                numChannels=len(self.audioDAQChannels),
                verbose=self.audioWriteVerbose,
                stdoutQueue=self.stdoutQueue)
            self.audioAcquireProcess = AudioAcquirer(
                startTime=startTime,
                audioQueue=audioQueue,
                audioMonitorQueue=self.audioMonitorQueue,
                audioAnalysisQueue=self.audioAnalysisQueue,
                messageQueue=self.audioAcquireMessageQueue,
                chunkSize=chunkSize,
                samplingRate=audioFrequency,
                bufferSize=None,
                channelNames=self.audioDAQChannels,
                syncChannel=self.audioSyncSource,
                verbose=self.audioAcquireVerbose,
                ready=ready,
                stdoutQueue=self.stdoutQueue)

        # Create all camera
        for camSerial in self.camSerials:
            print("Starting acquisition for camera", camSerial)
            acquireSettings = [
                ('AcquisitionMode', 'Continuous', 'enum'),
                ('TriggerMode', 'Off', 'enum'),
                ('TriggerSelector', 'FrameStart', 'enum'),
                ('TriggerSource', 'Line0', 'enum'),
                ('TriggerActivation', 'RisingEdge', 'enum'),
                # ('ExposureMode', 'TriggerWidth'),
                # ('Width', 800, 'integer'),
                # ('Height', 800, 'integer'),
                ('TriggerMode', 'On', 'enum'),
                ('ExposureAuto', 'Off', 'enum'),
                ('ExposureMode', 'Timed', 'enum'),
                ('ExposureTime', exposureTime, 'float')]   # List of attribute/value pairs to be applied to the camera in the given order
            imageQueue = mp.Queue()
            self.videoMonitorQueues[camSerial] = mp.Queue(1)
            self.videoAcquireMessageQueues[camSerial] = mp.Queue()
            self.videoWriteMessageQueues[camSerial] = mp.Queue()
            baseVideoFilename = slugify(baseFileName + '_' + camSerial)
            processes = {}

            videoAcquireProcess = VideoAcquirer(
                startTime=startTime,
                camSerial=camSerial,
                imageQueue=imageQueue,
                monitorImageQueue=self.videoMonitorQueues[camSerial],
                acquireSettings=acquireSettings,
                frameRate=videoFrequency,
                monitorFrameRate=self.monitorFrameRate,
                messageQueue=self.videoAcquireMessageQueues[camSerial],
                verbose=self.videoAcquireVerbose,
                ready=ready,
                stdoutQueue=self.stdoutQueue)
            videoWriteProcess = VideoWriter(
                camSerial=camSerial,
                videoDirectory=directory,
                videoBaseFilename=baseVideoFilename,
                imageQueue=imageQueue,
                frameRate=videoFrequency,
                messageQueue=self.videoWriteMessageQueues[camSerial],
                mergeMessageQueue=self.mergeMessageQueue,
                bufferSizeSeconds=bufferSizeSeconds,
                verbose=self.videoWriteVerbose,
                stdoutQueue=self.stdoutQueue
                )
            self.videoAcquireProcesses[camSerial] = videoAcquireProcess
            self.videoWriteProcesses[camSerial] = videoWriteProcess

        print("Video processes started:")
        print(self.videoAcquireProcesses)
        print(self.videoWriteProcesses)

        # Create sync process
        self.syncMessageQueue = mp.Queue()
        self.syncProcess = Synchronizer(
            startTime=startTime,
            audioSyncChannel=self.audioSyncTerminal,
            videoSyncChannel=self.videoSyncTerminal,
            audioFrequency=audioFrequency,
            videoFrequency=videoFrequency,
            messageQueue=self.syncMessageQueue,
            verbose=self.syncVerbose,
            ready=ready,
            stdoutQueue=self.stdoutQueue)

        # Create merge process
        self.mergeProcess = AVMerger(
            directory=directory,
            numFilesPerTrigger=numStreams,
            messageQueue=self.mergeMessageQueue,
            verbose=self.mergeVerbose,
            stdoutQueue=self.stdoutQueue,
            baseFileName=baseFileName,
            montage=True,
            deleteMergedFiles=deleteMergedFiles
            )

        self.audioTriggerProcess = AudioTriggerer(
            audioQueue=self.audioAnalysisQueue,
            audioFrequency=44100,
            chunkSize=chunkSize,
            triggerHighLevel=triggerHighLevel,
            triggerLowLevel=triggerLowLevel,
            triggerHighTime=triggerHighTime,
            triggerLowTime=triggerLowTime,
            triggerHighFraction=triggerHighFraction,
            triggerLowFraction=triggerLowFraction,
            maxAudioTriggerTime=maxAudioTriggerTime,
            preTriggerTime=preTriggerTime,
            multiChannelStartBehavior=multiChannelStartBehavior,
            multiChannelStopBehavior=multiChannelStopBehavior,
            verbose=self.audioTriggerVerbose,
            audioMessageQueue=self.audioWriteMessageQueue,
            videoMessageQueues=self.videoWriteMessageQueues,
            messageQueue=self.audioTriggerMessageQueue,
            stdoutQueue=None
            )

        # Start audio trigger process
        self.audioTriggerProcess.start()
        self.audioTriggerMessageQueue.put((AudioTriggerer.START, None))
        if triggerMode == "Audio":
            self.audioTriggerMessageQueue.put((AudioTriggerer.STARTANALYZE, None))

        # Start audioWriter
        self.audioWriteProcess.start()
        self.audioWriteMessageQueue.put((AudioWriter.START, None))

        # Start audioAcquirer
        self.audioAcquireProcess.start()
        self.audioAcquireMessageQueue.put((AudioAcquirer.START, None))

        # For each camera
        for camSerial in self.camSerials:
            # Start VideoWriter
            self.videoWriteProcesses[camSerial].start()
            self.videoWriteMessageQueues[camSerial].put((VideoWriter.START, None))
            # Start videoAcquirer
            self.videoAcquireProcesses[camSerial].start()
            self.videoAcquireMessageQueues[camSerial].put((VideoAcquirer.START, None))

        # Start sync process
        self.syncProcess.start()
        self.syncMessageQueue.put((Synchronizer.START, None))

        # Start merge process
        self.mergeProcess.start()
        self.updateAVMergerState()

    def stopAcquisition(self):

        try:
            s = io.StringIO()
            ps = pstats.Stats(self.profiler, stream=s)
            ps.print_stats()
            print(s.getvalue())
        except:
            print('Error printing profiler stats')

        print("stopping acquisition")
        try:
            if self.audioTriggerProcess is not None:
                self.audioTriggerMessageQueue.put((AudioTriggerer.STOP, None))
                self.audioTriggerMessageQueue.put((AudioTriggerer.EXIT, None))
                self.audioTriggerProcess = None
                clearQueue(self.audioTriggerMessageQueue)
                print('0/7 audio trigger done')
            if self.audioAcquireProcess is not None:
                self.audioAcquireMessageQueue.put((AudioAcquirer.STOP, None))
                self.audioAcquireMessageQueue.put((AudioAcquirer.EXIT, None))
#                self.audioAcquireProcess.join()
                self.audioAcquireProcess = None
                clearQueue(self.audioMonitorQueue)
                clearQueue(self.audioAcquireMessageQueue)
                print('1/7 audio acquire done')
            if self.audioWriteProcess is not None:
                self.audioWriteMessageQueue.put((AudioWriter.STOP, None))
                self.audioWriteMessageQueue.put((AudioWriter.EXIT, None))
#                self.audioWriteProcess.join()
                self.audioWriteProcess = None
                clearQueue(self.audioWriteMessageQueue)
                print('2/7 audio write done')
            for camSerial in self.camSerials:
                self.videoAcquireMessageQueues[camSerial].put((VideoAcquirer.STOP, None))
                self.videoAcquireMessageQueues[camSerial].put((VideoAcquirer.EXIT, None))
#                self.videoAcquireProcesses[camSerial].join()
#                self.videoWriteProcesses[camSerial].join()
                self.videoAcquireProcesses = {}
                self.videoWriteProcesses = {}
                clearQueue(self.videoMonitorQueues[camSerial])
                clearQueue(self.videoAcquireMessageQueues[camSerial])
                clearQueue(self.videoWriteMessageQueues[camSerial])
                print('3/7 video write/acquire done for', camSerial)
            if self.mergeProcess is not None:
#                self.mergeMessageQueue.put((AVMerger.STOP, None))
                self.mergeMessageQueue.put((AVMerger.EXIT, None))
#                self.mergeProcess.join()
                clearQueue(self.mergeMessageQueue)
                print('4/7 merge done')
            if self.syncProcess is not None:
                self.syncMessageQueue.put((Synchronizer.STOP, None))
                self.syncMessageQueue.put((Synchronizer.EXIT, None))
#                self.syncProcess.join()
                self.syncProcess = None
                clearQueue(self.syncMessageQueue)
                print('5/7 sync done')

            # Clear/destroy monitoring queues
            if self.videoMonitorQueues is not None:
                for camSerial in self.videoMonitorQueues:
                    clearQueue(self.videoMonitorQueues[camSerial])
                self.videoMonitorQueues = {}
            if self.audioMonitorQueue is not None:
                clearQueue(self.audioMonitorQueue)
                self.audioMonitorQueue = None
            if self.mergeMessageQueue is not None:
                clearQueue(self.mergeMessageQueue)
                self.mergeMessageQueue = None
            print('6/7 monitoring queues done')
            if self.stdoutManager is not None:
                self.stdoutQueue.put(stdoutManager.STOP)
                print('clearing queue')
                clearQueue(self.stdoutQueue)
                print('done clearing queue')
                self.stdoutQueue = None
                self.stdoutManager = None
            print('7/7 stdout manager done')
        except:
            traceback.print_exc()

    def sendWriteTrigger(self, t=None):
        if t is None:
            t = time.time_ns()/1000000000
        trig = Trigger(t-2, t, t+2)
        for camSerial in self.camSerials:
            self.videoWriteMessageQueues[camSerial].put((VideoWriter.TRIGGER, trig))
        if self.audioWriteProcess is not None:
            self.audioWriteMessageQueue.put((AudioWriter.TRIGGER, trig))

    def update(self):
        # root window
        #   monitorFrame
        #       videoMonitorMasterFrame
        #       audioMonitorMasterFrame
        #   controlFrame
        #       acquisitionFrame
        #       triggerFrame
        #           triggerModeChooserFrame
        #           triggerModeControlGroupFrame (only the active one is gridded)
        #

        self.master.columnconfigure(0, weight=1)
        self.master.columnconfigure(1, weight=1)
        self.master.rowconfigure(0, weight=1)
        self.master.rowconfigure(1, weight=1)

        self.monitorFrame.grid(row=0, column=0)

        self.videoMonitorMasterFrame.grid(row=0, column=0, sticky=tk.NSEW)
        wV, hV = getOptimalMonitorGrid(len(self.camSerials))
        for k, camSerial in enumerate(self.camSerials):
            self.videoMonitorFrames[camSerial].grid(row=2*(k // wV), column = k % wV)
            self.videoMonitors[camSerial].grid(row=0, column=0, columnspan=2)
            self.cameraAttributeBrowserButtons[camSerial].grid(row=1, column=0)
            self.videoStateWidgets[camSerial].grid(row=1, column=1)

        self.audioMonitorMasterFrame.grid(row=1, column=0, sticky=tk.NSEW)
        wA, hA = getOptimalMonitorGrid(len(self.audioDAQChannels))
        for k, channel in enumerate(self.audioDAQChannels):
            self.audioMonitorFrames[channel].pack()
            self.audioDAQChannelWidgets[channel]['figureCanvas'].get_tk_widget().pack(side=tk.LEFT, fill=tk.BOTH, expand=1)
            self.audioStateWidgets[channel].pack(side=tk.BOTTOM)

        self.controlFrame.grid(row=0, column=1, sticky=tk.NSEW)
        self.controlFrame.columnconfigure(0, weight=1)
        self.controlFrame.columnconfigure(1, weight=1)
        self.controlFrame.rowconfigure(0, weight=1)
        self.controlFrame.rowconfigure(1, weight=1)

        self.acquisitionFrame.grid(row=0, column=0, sticky=tk.NSEW)
        # for c in range(3):
        #     self.acquisitionFrame.columnconfigure(c, weight=1)
        # for r in range(4):
        #     self.acquisitionFrame.rowconfigure(r, weight=1)
        self.startAcquisitionButton.grid(row=0, column=0, columnspan=5, sticky=tk.NSEW)
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
        self.baseFileNameFrame.grid(row=2, column=2, columnspan=2, sticky=tk.EW)
        self.baseFileNameEntry.grid()
        self.directoryFrame.grid(row=3, column=2, sticky=tk.EW)
        self.directoryEntry.grid(row=0, column=0)
        self.directoryButton.grid(row=1, column=0, sticky=tk.EW)

        self.mergeFrame.grid(row=5, column=0, rowspan=3)
        self.mergeFilesCheckbutton.grid(row=1, column=0, sticky=tk.NW)
        self.deleteMergedFilesCheckbutton.grid(row=2, column=0, sticky=tk.NW)
        self.montageMergeCheckbutton.grid(row=3, column=0, stick=tk.NW)

        self.scheduleEnabledCheckbutton.grid(row=5, column=1, columnspan=2, sticky=tk.NW)
        self.scheduleStartTimeEntry.grid(row=6, column=1, columnspan=2, sticky=tk.NW)
        self.scheduleStopTimeEntry.grid(row=7, column=1, columnspan=2, sticky=tk.NW)

        self.triggerFrame.grid(row=1, column=0, sticky=tk.NSEW)
        self.triggerModeChooserFrame.grid(row=0, column=0, sticky=tk.NW)
        self.triggerModeLabel.grid(row=0, column=0, columnspan=2)
        for k, mode in enumerate(self.triggerModes):
            self.triggerModeRadioButtons[mode].grid(row=1, column=k)
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
        self.maxAudioTriggerTimeFrame.grid(row=2, column=0)
        self.maxAudioTriggerTimeEntry.grid()
        self.multiChannelStartBehaviorFrame.grid(row=2, column=1, rowspan=2)
        self.multiChannelStartBehaviorOR.grid(row=0)
        self.multiChannelStartBehaviorAND.grid(row=1)
        self.multiChannelStopBehaviorFrame.grid(row=2, column=2, rowspan=2)
        self.multiChannelStopBehaviorOR.grid(row=0)
        self.multiChannelStopBehaviorAND.grid(row=1)

def clearQueue(q):
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


#
# def manualTriggerProcess(channelName, stdinFileno):
#     sys.stdin = os.fdopen(stdinFileno)  #open stdin in this process
#
#     with nidaqmx.Task() as task:
#         task.do_channels.add_do_chan(channelName)
#
#         state = task.read()
#         syncPrint("Current state is: ", state)
#
#         msg = ''
#         while True:
#             msg = input('t to make a train of triggers, or anything else to exit: ')
#             if msg == 't':
#                 for k in range(20):
#                     state = not state
#                     time.sleep(0.1)
#                     task.write(state)
#             else:
#                 break
#     syncPrint("Manual process done")
#
### Process spawn functions

# def spawnManualTriggerProcess(channelName=None):
#     stdinFileno = sys.stdin.fileno()
#     p = mp.Process(
#         target=manualTriggerProcess,
#         args=(channelName,stdinFileno))
#     p.start()
#     return p

### Utility functions
