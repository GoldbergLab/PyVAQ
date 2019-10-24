import re
# import sys
# import os
import struct
import time
import unicodedata
import wave
import numpy as np
import multiprocessing as mp
import nidaqmx
from nidaqmx.stream_readers import AnalogSingleChannelReader
import PySpin
import tkinter as tk
import tkinter.ttk as ttk
import queue
from PIL import Image, ImageTk
import pprint
import traceback

# For audio monitor graph embedding:
from matplotlib.backends.backend_tkagg import (
    FigureCanvasTkAgg, NavigationToolbar2Tk)
# Implement the default Matplotlib key bindings.
from matplotlib.backend_bases import key_press_handler
from matplotlib.figure import Figure

# Todo:
#  - Figure out why video task isn't closing down properly
#  - Add video framerate indicator
#  - Make attributes settable
#  - Separate acquire and write modes so it's possible to monitor w/o writing
#  - Add filename entry for each stream
#  - Camera commands are not being collected properly
#  - Fix acquire/write indicator positioing
#  - Make saved avis not gigantic (maybe switch to opencv for video writing?)
#  - Add buffering capability
#  - Add external record triggering
#  - Add volume-based triggering
# Done
#  - Fix camera monitor
#  - Add video & audio frequency controls
#  - Fix audio

'''
cd "Dropbox\Documents\Work\Cornell Lab Tech\Projects\Video VI\PyVAQ"
python PyVAQ.py
'''

### Main recording and writing functions

class Synchronizer(mp.Process):
    # Class for generating two synchronization signals at the same time
    #   - one for video (send via cable to the camera GPIO)
    #   - one for audio (used internally to trigger analog input of microphone
    #     signals)
    # This class inherits from multiprocessing.Process so it can be run in a
    #   separate process, allowing a single script to generate the sync pulses
    #   and also accomplish other tasks.
    def __init__(self,
        videoFrequency=120,                     # The frequency in Hz of the video sync signal
        audioFrequency=44100,                   # The frequency in Hz of the audio sync signal
        videoSyncChannel="Dev3/ctr0",           # The counter channel on which to generate the video sync signal
        audioSyncChannel="Dev3/ctr1",           # The counter channel on which to generate the audio sync signal
        readyToStop = None,                     # Synchronization barrier to ensure everyone's ready to stop before stopping
        ready=None):                            # Synchronization barrier to ensure everyone's ready before beginning
        mp.Process.__init__(self)
        # Store inputs in instance variables for later access
        self.videoFrequency = videoFrequency
        self.audioFrequency = audioFrequency
        self.videoSyncChannel = videoSyncChannel
        self.audioSyncChannel = audioSyncChannel
        self.stop = mp.Event()                      # An event to gracefully halt this process
        self.ready = ready
        self.readyToStop = readyToStop

    def stopProcess(self):
        # Method to shut down process gracefully
        print('Stopping synchronization signal process')
        self.stop.set()

    def run(self):
        # Configure and generate synchronization signal
        with nidaqmx.Task() as trigTask:                       # Create task
            trigTask.co_channels.add_co_pulse_chan_freq(
                counter=self.videoSyncChannel,
                name_to_assign_to_channel="videoSync",
                units=nidaqmx.constants.FrequencyUnits.HZ,
                initial_delay=0.0,
                freq=self.videoFrequency,
                duty_cycle=0.5)     # Prepare a counter output channel for the video sync signal
            trigTask.co_channels.add_co_pulse_chan_freq(
                counter=self.audioSyncChannel,
                name_to_assign_to_channel="audioSync",
                units=nidaqmx.constants.FrequencyUnits.HZ,
                initial_delay=0.0,
                freq=self.audioFrequency,
                duty_cycle=0.5)     # Prepare a counter output channel for the audio sync signal
            trigTask.timing.cfg_implicit_timing(sample_mode=nidaqmx.constants.AcquisitionType.CONTINUOUS)
            if self.ready is not None:
                self.ready.wait()
            trigTask.start()
            print("Synchronization process STARTED")

            while True:  # Exit signal received
                if self.stop.is_set():
                    print("S - Ready and waiting to stop")
                    self.readyToStop.wait()
                    print("S - stopping")
                    break
                time.sleep(0.5)  # May be necessary to prevent the daqmx tasks from terminating
        print("Synchronization process STOPPED")

class AudioAcquirer(mp.Process):
    # Class for acquiring an audio signal (or any analog signal) at a rate that
    #   is synchronized to the rising edges on the specified synchronization
    #   channel.
    def __init__(self,
                audioQueue = None,                  # A multiprocessing queue to send data to another proces for writing to disk
                audioMonitorQueue = None,           # A multiprocessing queue to send data to the UI to monitor the audio
                sampleChunkSize = 44100,            # Size of the read chunk in samples
                maxExpectedSamplingRate = 44100,    # Maximum expected rate of the specified synchronization channel
                bufferSize = None,                  # Size of device buffer. Defaults to 1 second's worth of data
                channelName = None,                 # Channel name for analog input (microphone signal)
                syncChannel = None,                 # Channel name for synchronization source
                readyToStop = None,                 # Synchronization barrier to ensure everyone's ready to stop before stopping
                ready=None):                        # Synchronization barrier to ensure everyone's ready before beginning
        mp.Process.__init__(self, daemon=True)
        # Store inputs in instance variables for later access
        if bufferSize is None:
            self.bufferSize = maxExpectedSamplingRate  # Device buffer size defaults to One second's worth of buffer
        else:
            self.bufferSize = bufferSize
        self.audioQueue = audioQueue
        self.audioQueue.cancel_join_thread()
        self.audioMonitorQueue = audioMonitorQueue
        self.audioMonitorQueue.cancel_join_thread()
        self.sampleChunkSize = sampleChunkSize
        self.maxExpectedSamplingRate = maxExpectedSamplingRate
        self.inputChannel = channelName
        self.syncChannel = syncChannel
        self.stop = mp.Event()
        self.ready = ready
        self.readyToStop = readyToStop

    def stopProcess(self):
        # Method to shut down process gracefully
        print('Stopping audio acquire process')
        self.stop.set()

    def rescaleAudio(data, maxV=10, minV=-10, maxD=32767, minD=-32767):
        return (data * ((maxD-minD)/(maxV-minV))).astype('int16')

    def run(self):
        # Configure analog acquisition and begin acquisition
        data = np.zeros(self.sampleChunkSize, dtype='float')        # A pre-allocated array to receive audio data
        with nidaqmx.Task(new_task_name="audioTask!") as readTask:                            # Create task
            print("setting up task")
            print("timing channel:", self.syncChannel)
            print("audio channel:", self.inputChannel)
            readTask.ai_channels.add_ai_voltage_chan(               # Set up analog input channel
                self.inputChannel,
                terminal_config=nidaqmx.constants.TerminalConfiguration.DIFFERENTIAL,
                max_val=10,
                min_val=-10)
            readTask.timing.cfg_samp_clk_timing(                    # Configure clock source for triggering each analog read
                rate=self.maxExpectedSamplingRate,
                source=self.syncChannel,                            # Specify a timing source!
                active_edge=nidaqmx.constants.Edge.RISING,
                sample_mode=nidaqmx.constants.AcquisitionType.CONTINUOUS,
                samps_per_chan=self.bufferSize)

            reader = AnalogSingleChannelReader(readTask.in_stream)  # Set up an analog stream reader

            if self.ready is not None:
                self.ready.wait()
                # print("Audio acquire process STARTED")
            while True:  # Exit signal received
                if self.stop.is_set():
                    print("AA - ready and waiting to stop")
                    self.readyToStop.wait()
                    print("AA - stopping")
                    break
                try:
                    reader.read_many_sample(                            # Read a chunk of audio data
                        data,
                        number_of_samples_per_channel=self.sampleChunkSize,
                        timeout=10.0)
                    print("Acquired audio chunk")
                    processedData = AudioAcquirer.rescaleAudio(data).tolist()
                    if self.audioQueue is not None:
                        self.audioQueue.put(processedData)              # If a data queue is provided, queue up the new data
                    else:
                        print(processedData)

                    if self.audioMonitorQueue is not None:
                        self.audioMonitorQueue.put((self.inputChannel, processedData))      # If a monitoring queue is provided, queue up the data

                except nidaqmx.errors.DaqError:
                    pass  # Timeout
                    print("Audio chunk read timeout")
                    traceback.print_exc()
        print("Audio acquire process STOPPED")

class AudioWriter(mp.Process):
    def __init__(self,
                wavFilename='audioFile.wav',
                audioQueue=None,
                audioFrequency=44100,
                readyToStop = None,                 # Synchronization barrier to ensure everyone's ready to stop before stopping
                ready=None):                        # Synchronization barrier to ensure everyone's ready before beginning):
        mp.Process.__init__(self, daemon=True)
        self.wavFilename = wavFilename
        self.audioQueue = audioQueue
        self.audioQueue.cancel_join_thread()
        self.stop = mp.Event()
        self.ready = ready
        self.audioFrequency = audioFrequency
        self.readyToStop = readyToStop

    def stopProcess(self):
        print('Stopping audio write process')
        self.stop.set()

    def run(self):
        audioFile = wave.open(self.wavFilename, 'w')
        # setparams: (nchannels, sampwidth, framerate, nframes, comptype, compname)
        audioFile.setparams((1, 2, self.audioFrequency, 0, 'NONE', 'not compressed'))
        if self.ready is not None:
            self.ready.wait()
        # print("Audio acquisition begins now!")
        while True:
            if self.stop.is_set():
                print("AW - ready and waiting to stop")
                self.readyToStop.wait()
                print("AW - stopping")
                break
            try:
                audioChunk = self.audioQueue.get(True, 0.1)
                audioChunkBytes = b''.join(map(lambda x:struct.pack('h', x), audioChunk))
                audioFile.writeframes(audioChunkBytes)
            except queue.Empty:
                pass # Nothing in queue right now
        audioFile.close()
        print("Audio write process STOPPED")
        while True:
            print("AW - Clearing audio queue")
            try:
                self.audioQueue.get(True, 1)
            except:
                break

class VideoAcquirer(mp.Process):
    def __init__(self,
                camSerial,
                imageQueue,
                monitorImageQueue,
                acquireSettings={},
                monitorFrameRate=15,
                readyToStop = None,                 # Synchronization barrier to ensure everyone's ready to stop before stopping
                ready=None):                        # Synchronization barrier to ensure everyone's ready before beginning
        mp.Process.__init__(self, daemon=True)
        self.camSerial = camSerial
        self.acquireSettings = acquireSettings
        self.imageQueue = imageQueue
        self.imageQueue.cancel_join_thread()
        self.monitorImageQueue = monitorImageQueue
        self.monitorImageQueue.cancel_join_thread()
        self.monitorFrameRate = monitorFrameRate
        self.stop = mp.Event()
        self.ready = ready
        self.readyToStop = readyToStop

    def stopProcess(self):
        print('Stopping video acquire process')
        self.stop.set()

    def run(self):
        system = PySpin.System.GetInstance()
        camList = system.GetCameras()
        cam = camList.GetBySerial(self.camSerial)
        cam.Init()
        nodemap = cam.GetNodeMap()
        setCameraAttributes(nodemap, self.acquireSettings)
        cam.BeginAcquisition()

        monitorFramePeriod = 1.0/self.monitorFrameRate
        print("Video monitor frame period:", monitorFramePeriod)
        lastTime = time.time()
        k = 0
        im = imp = imageResult = None
        if self.ready is not None:
            self.ready.wait()
        print("Image acquisition begins now!")
        while True:  # Exit signal received
            if self.stop.is_set():
                print("VA - ready and waiting to stop")
                self.readyToStop.wait()
                print("VA - stopping")
                break
            try:
                #  Retrieve next received image
                imageResult = cam.GetNextImage(100) # Timeout of 100 ms to allow for stopping process
                #  Ensure image completion
                if imageResult.IsIncomplete():
                    print('Image incomplete with image status %d...' % imageResult.GetImageStatus())
                else:
                    #  Print image information; height and width recorded in pixels
                    width = imageResult.GetWidth()
                    height = imageResult.GetHeight()
                    k = k + 1
#                    print('Grabbed Image %d, width = %d, height = %d' % (k, width, height))
                    im = imageResult.Convert(PySpin.PixelFormat_Mono8, PySpin.HQ_LINEAR)
                    imp = PickleableImage(im.GetWidth(), im.GetHeight(), 0, 0, im.GetPixelFormat(), im.GetData())
                    self.imageQueue.put(imp)

                    # Put the occasional image in the monitor queue for the UI
                    thisTime = time.time()
                    if (thisTime - lastTime) >= monitorFramePeriod:
                        # print("Sent frame for monitoring")
                        self.monitorImageQueue.put((self.camSerial, imp))
                        lastTime = thisTime

                    imageResult.Release()
            except PySpin.SpinnakerException as ex:
                pass # Hopefully this is just because there was no image in camera buffer
                # print('Error: %s' % ex)
                # traceback.print_exc()
                # return False

        # Send stop signal to write process
        self.imageQueue.put(None)

        camList.Clear()
        cam.EndAcquisition()
        cam.DeInit()
        del cam
        system.ReleaseInstance()
        del nodemap
        del imageResult
        del im
        del imp
        del camList
        del system
        print("Video acquire process STOPPED")
        return 0

class VideoWriter(mp.Process):
    def __init__(self,
                aviFilename,
                imageQueue,
                framerate,
                readyToStop = None,                 # Synchronization barrier to ensure everyone's ready to stop before stopping
                ready=None):                        # Synchronization barrier to ensure everyone's ready before beginning
        mp.Process.__init__(self, daemon=True)
        self.aviFilename = aviFilename
        self.imageQueue = imageQueue
        self.imageQueue.cancel_join_thread()
        self.framerate = framerate
        self.stop = mp.Event()
        self.ready = ready
        self.readyToStop = readyToStop

    def stopProcess(self):
        print('Stopping video write process')
        self.stop.set()

    def run(self):
        aviRecorder = PySpin.SpinVideo()
        option = PySpin.AVIOption()
        option.frameRate = self.framerate
        aviRecorder.Open(self.aviFilename, option)
        imageCount = 0
        if self.ready is not None:
            self.ready.wait()
        while True:  # Exit signal received
            if self.stop.is_set():
                while True:
                    try:
                        self.imageQueue.get(False, 1)
                    except:
                        break
                print("VW - ready and waiting to stop")
                self.readyToStop.wait()
                print("VW - stopping")
                break
            try:
                imp = self.imageQueue.get(True, 0.1)
                if imp is None:
                    print("Stop signal received from video acquire process.")
                    break
                im = PySpin.Image.Create(imp.width, imp.height, imp.offsetX, imp.offsetY, imp.pixelFormat, imp.data)
                aviRecorder.Append(im)
        #        im.Release()
                imageCount+=1
                if imageCount % 10 == 0:
                    print("Wrote", imageCount, "images to file")
            except queue.Empty:
                pass  # Nothing in queue right now
        aviRecorder.Close()
        print("Video write process STOPPED")
        return

def processStatus(processes, queues, processNames=None, queueNames=None, processWidgets=None, queueWidgets=None):
    if processNames is None:
        processNames = [str(k) for k in range(len(processes))]
    if queueNames is None:
        queueNames = [str(k) for k in range(len(queues))]

    processesAlive = []
    for process, processName in zip(processes, processNames):
        alive = process.is_alive()
        processesAlive.append(alive)
        print("P_"+processName+"_alive="+str(int(alive)), end=" ")
    for queue, queueName in zip(queues, queueNames):
        print("Q_"+queueName+"="+str(queue.qsize()), end=" ")
    print()
    return processesAlive

class PickleableImage():
    def __init__(self, width, height, offsetX, offsetY, pixelFormat, data):
        self.width = width
        self.height = height
        self.offsetX = offsetX
        self.offsetY = offsetY
        self.pixelFormat = pixelFormat
        self.data = data

def setCameraAttribute(nodemap, attributeName, attributeValue, attributePtrType='CEnumerationPtr'):
    # Set camera attribute. Retrusn True if successful, False otherwise.
    print('Setting', attributeName, 'to', attributeValue)
    nodeAttribute = getattr(PySpin, attributePtrType)(nodemap.GetNode(attributeName))
    if not PySpin.IsAvailable(nodeAttribute) or not PySpin.IsWritable(nodeAttribute):
        print('Unable to set '+attributeName+' to '+attributeValue+' (enum retrieval). Aborting...')
        return False

    # Retrieve entry node from enumeration node
    nodeAttributeValue = nodeAttribute.GetEntryByName(attributeValue)
    if not PySpin.IsAvailable(nodeAttributeValue) or not PySpin.IsReadable(nodeAttributeValue):
        print('Unable to set '+attributeName+' to '+attributeValue+' (entry retrieval). Aborting...')
        return False

    # Set value
    attributeValueCode = nodeAttributeValue.GetValue()
    nodeAttribute.SetIntValue(attributeValueCode)
    return True

def setCameraAttributes(nodemap, attributeValuePairs):
#    print('Setting attributes')
    for attribute, value in attributeValuePairs:
        result = setCameraAttribute(nodemap, attribute, value)
        if not result:
            print("Failed to set", attribute, " to ", attributes[attribute])
#    print('Done')

def getCameraAttribute(nodemap, attributeName, attributeTypePtrFunction):
    nodeAttribute = attributeTypePtrFunction(nodemap.GetNode(attributeName))

    if not PySpin.IsAvailable(nodeAttribute) or not PySpin.IsReadable(nodeAttribute):
        print('Unable to retrieve '+attributeName+'. Aborting...')
        return None

    value = nodeAttribute.GetValue()
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

class StreamType:
    AUDIO=0
    VIDEO=1

class EndpointType:
    ACQUIRE=0
    WRITE=1

class PyVAQ:
    def __init__(self, master, audioSyncSource='PFI4', videoSyncSource='PFI5'):
        self.master = master
        self.master.title("PyVAQ")
        self.master.protocol("WM_DELETE_WINDOW", self.cleanup)
        self.cameraAttributes = {}
        self.camSerials = self.discoverCameras()
        self.updateAllCamerasAttributes()
        with open('attributes.txt', 'w') as f:
            pp = pprint.PrettyPrinter(stream=f, indent=2)
            pp.pprint(self.cameraAttributes)

        # Create widgets for monitoring video streams
        self.videoMonitorMasterFrame = ttk.Frame(self.master)
        self.videoMonitorFrames = {}
        self.videoMonitors = {}
        self.imageIDs = {}
        self.currentImages = {}
        self.cameraAttributeBrowserButtons = {}
        self.videoStateWidgets = {}
        for camSerial in self.camSerials:
            self.videoMonitorFrames[camSerial] = vFrame = ttk.LabelFrame(self.videoMonitorMasterFrame, text=camSerial)
            self.videoMonitors[camSerial] = tk.Canvas(vFrame, width=400, height=600, borderwidth=2, relief=tk.SUNKEN)
            self.imageIDs[camSerial] = None
            self.currentImages[camSerial] = None
            self.cameraAttributeBrowserButtons[camSerial] = ttk.Button(vFrame, text="Attribute browser", command=lambda:self.createCameraAttributeBrowser(camSerial))
            self.videoStateWidgets[camSerial] = ttk.Label(vFrame)

        self.audioDAQChannels = ['Dev3/ai5']
        self.audioSyncSource = audioSyncSource
        self.videoSyncSource = videoSyncSource

        # Create widgets for monitoring audio streams
        self.audioDAQChannelWidgets = {}
        self.audioMonitorMasterFrame = ttk.Frame(self.master)
        self.audioMonitorFrames = {}
        self.audioStateWidgets = {}
        for channel in self.audioDAQChannels:
            self.audioMonitorFrames[channel] = aFrame = ttk.LabelFrame(self.audioMonitorMasterFrame, text=channel)
            self.audioStateWidgets[channel] = ttk.Label(aFrame)
            self.createAudioMonitor(channel)

        self.controlFrame = tk.Canvas(self.master)
        self.startAcquisitionButton = ttk.Button(self.controlFrame, text="Start acquire", command=self.recordAudioVideo)
        self.stopAcquisitionButton = ttk.Button(self.controlFrame, text="Stop acquire", command=self.stopAudioVideo)
        self.audioFrequencyEntry = ttk.Entry(self.controlFrame)
        self.audioFrequencyEntry.insert(0, "22010")
        self.audioFrequencyLabel = ttk.Label(self.controlFrame, text="Audio frequency (Hz):")
        self.videoFrequencyEntry = ttk.Entry(self.controlFrame)
        self.videoFrequencyEntry.insert(0, "2")
        self.videoFrequencyLabel = ttk.Label(self.controlFrame, text="Video frequency (fps):")

        # Monitoring queues for collecting qudio and video data for user monitoring purposes
        self.videoMonitorQueue = mp.Queue()
        self.monitorFrameRate = 15

        self.audioMonitorQueue = mp.Queue()
        self.audioMonitorQueue.cancel_join_thread()

        self.videoWriteProcesses = {}
        self.videoAcquireProcesses = {}
        self.audioWriteProcesses = {}
        self.audioAcquireProcesses = {}
        self.syncProcess = None

        self.update()
        self.autoUpdateVideoMonitors()
        self.autoUpdateAudioMonitors()
        self.master.after(100, self.autoUpdateVideoMonitors)

    def cleanup(self):

        self.master.destroy()

    def setAudioVideoState(self, streamType, ID, readState, writeState):
        if streamType == StreamType.AUDIO:
            widget = self.audioStateWidgets[ID]
        elif streamType == StreamType.VIDEO:
            widget = self.videoStateWidgets[ID]
        else:
            raise Exception("Unknown stream type")
        state = ('ACQUIRING' if readState else '') + '|' + ('WRITING' if writeState else '')
        widget.config(text=state)

    def createAudioMonitor(self, channel):
        self.audioDAQChannelWidgets[channel] = {}  # Change this to gracefully remove existing channel widgets under this channel name
        fig = Figure(figsize=(5, 4), dpi=100)
        t = np.arange(0, 3, .01)
        line = fig.add_subplot(111).plot(t, 0.0 * t)
        canvas = FigureCanvasTkAgg(fig, master=self.audioMonitorFrames[channel])  # A tk.DrawingArea.
        canvas.draw()
        toolbar = NavigationToolbar2Tk(canvas, self.audioMonitorFrames[channel])
        toolbar.update()
        def figureKeyPressManager(event):
            print("you pressed {}".format(event.key))
            key_press_handler(event, canvas, toolbar)
        canvas.mpl_connect("key_press_event", figureKeyPressManager)

        self.audioDAQChannelWidgets[channel]['figure'] = fig
        self.audioDAQChannelWidgets[channel]['figureCanvas'] = canvas
        self.audioDAQChannelWidgets[channel]['figureNavToolbar'] = toolbar
        self.audioDAQChannelWidgets[channel]['figureLine'] = line

    def autoUpdateAudioMonitors(self):
        try:
            channel, audioData = self.audioMonitorQueue.get(True, 0.01)
            print("Got audio chunk for monitoring")
            line = self.audioDAQChannelWidgets[channel]['figureLine'][0]
            newY = np.random.rand(300)
            line.set_ydata(newY)
            self.audioDAQChannelWidgets[channel]['figure'].canvas.draw()
            self.audioDAQChannelWidgets[channel]['figure'].canvas.flush_events()
        except queue.Empty:
            pass

        self.master.after(100, self.autoUpdateAudioMonitors)

    def autoUpdateVideoMonitors(self):
        try:
            camSerial, pImage = self.videoMonitorQueue.get(True, 0.01)
            # print("Received frame for monitoring")
            imData = np.reshape(pImage.data, (pImage.height, pImage.width))
            im = Image.fromarray(imData)
            self.currentImages[camSerial] = ImageTk.PhotoImage(im)
            if self.imageIDs[camSerial] is None:
                self.imageIDs[camSerial] = self.videoMonitors[camSerial].create_image((0, 0), image=self.currentImages[camSerial])
            else:
                self.videoMonitors[camSerial].itemconfig(self.imageIDs[camSerial], image=self.currentImages[camSerial])
            self.videoMonitors[camSerial].update_idletasks()
        except queue.Empty:
            pass

        self.master.after(int(round(1000.0/(self.monitorFrameRate))), self.autoUpdateVideoMonitors)

    def monitorProcesses(self): #, processList, processNameList, queueList, queueNameList):
        atLeastOneProcessAlive = False
        for camSerial in self.camSerials:
            p_write = self.videoWriteProcesses[camSerial]
            p_acquire = self.videoAcquireProcesses[camSerial]
            writer_alive = p_write.is_alive()
            acquirer_alive = p_acquire.is_alive()
            atLeastOneProcessAlive = atLeastOneProcessAlive or writer_alive or acquirer_alive
            print(camSerial, "AQ:", acquirer_alive, "WR:", writer_alive)
            self.setAudioVideoState(StreamType.VIDEO, camSerial, acquirer_alive, writer_alive)
        for channel in self.audioDAQChannels:
            p_write = self.audioWriteProcesses[channel]
            p_acquire = self.audioAcquireProcesses[channel]
            writer_alive = p_write.is_alive()
            acquirer_alive = p_acquire.is_alive()
            atLeastOneProcessAlive = atLeastOneProcessAlive or writer_alive or acquirer_alive
            print(channel, "AQ:", acquirer_alive, "WR:", writer_alive)
            self.setAudioVideoState(StreamType.AUDIO, channel, acquirer_alive, writer_alive)

        if atLeastOneProcessAlive:
            self.master.after(2000, self.monitorProcesses)
        else:
            print("Acquiring and writing audio and video complete!")

    def initiateVideoRecord(self, cam, acquireSettings, ready, readyToStop):
        queue = mp.Queue()
        cam.Init()
        camSerial = getCameraAttribute(cam.GetTLDeviceNodeMap(), 'DeviceSerialNumber', PySpin.CStringPtr)
        print("starting camera: ", camSerial)
        filename = slugify('camera_recording_'+camSerial+'.avi')
        processes = {}
        processes['videoWriteProcess'] = VideoWriter(
            filename,
            queue,
            int(self.videoFrequencyEntry.get()),
            ready=ready,
            readyToStop=readyToStop)
        processes['videoAcquireProcess'] = VideoAcquirer(
            camSerial,
            queue,
            self.videoMonitorQueue,
            acquireSettings=acquireSettings,
            monitorFrameRate=self.monitorFrameRate,
            ready=ready,
            readyToStop=readyToStop)
        processes['videoWriteProcess'].start()
        processes['videoAcquireProcess'].start()
        return processes, queue

    def initiateAudioRecord(self, audioChannel, ready, readyToStop):
        audioQueue = mp.Queue()
        audioQueue.cancel_join_thread()
        audioFilename = slugify('audio_recording_'+audioChannel)+'.wav'
        processes = {}
        processes['audioWriteProcess'] = AudioWriter(
            audioFilename,
            audioQueue,
            ready=ready,
            readyToStop=readyToStop)
        processes['audioAcquireProcess'] = AudioAcquirer(
            audioQueue=audioQueue,
            audioMonitorQueue=self.audioMonitorQueue,
            sampleChunkSize=44100,
            maxExpectedSamplingRate=int(self.audioFrequencyEntry.get()),
            bufferSize=None,
            channelName=audioChannel,
            syncChannel=self.audioSyncSource,
            readyToStop=readyToStop,
            ready=ready)
        processes['audioWriteProcess'].start()
        processes['audioAcquireProcess'].start()
        return processes, audioQueue

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

        # print()
        # pp = pprint.PrettyPrinter(indent=1, depth=1)
        # pp.pprint(attributeNode)
        # print()

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

    def discoverCameras(self, numFakeCameras=0):
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

    def stopAudioVideo(self):
        for camSerial in self.videoWriteProcesses:
            self.videoWriteProcesses[camSerial].stopProcess()
            self.videoAcquireProcesses[camSerial].stopProcess()
        for channel in self.audioWriteProcesses:
            self.audioWriteProcesses[channel].stopProcess()
            self.audioAcquireProcesses[channel].stopProcess()
        self.syncProcess.stopProcess()
        clearQueue(self.videoMonitorQueue)
        for camSerial in self.videoWriteProcesses:
            print(1)
            self.videoAcquireProcesses[camSerial].join()
            print(2)
            self.videoWriteProcesses[camSerial].join()
            print(3)

        clearQueue(self.audioMonitorQueue)
        for channel in self.audioWriteProcesses:
            print(4)
            self.audioAcquireProcesses[channel].join()
            print(5)
            self.audioWriteProcesses[channel].join()
            print(6)

    def recordAudioVideo(self):
        print("starting acquisition")
        system = PySpin.System.GetInstance()
        camList = system.GetCameras()
        print("Cameras found:", camList.GetSize())
        imageQueues = []
        audioQueues = []
        numProcesses = 2*len(self.audioDAQChannels) + 2*len(self.camSerials) + 1
        ready = mp.Barrier(numProcesses)
        readyToStop = mp.Barrier(numProcesses)
        # manualTriggerProcess = spawnManualTriggerProcess(channelName='Dev3/port1/line0')

        for audioChannel in self.audioDAQChannels:
            processes, queue = self.initiateAudioRecord(audioChannel, ready, readyToStop)
            self.audioAcquireProcesses[audioChannel] = processes['audioAcquireProcess']
            self.audioWriteProcesses[audioChannel] = processes['audioWriteProcess']
            audioQueues.append(queue)

        acquireSettings = [('AcquisitionMode', 'Continuous'), ('TriggerMode', 'Off'), ('TriggerSelector', 'FrameStart'), ('TriggerSource', 'Line0'), ('TriggerMode', 'On')]   # List of attribute/value pairs to be applied to the camera in the given order
        for camSerial in self.camSerials:
            cam = camList.GetBySerial(camSerial)
            processes, queue = self.initiateVideoRecord(cam, acquireSettings, ready, readyToStop)
            imageQueues.append(queue)
            self.videoAcquireProcesses[camSerial] = processes['videoAcquireProcess']
            self.videoWriteProcesses[camSerial] = processes['videoWriteProcess']

        self.syncProcess = Synchronizer(audioSyncChannel='Dev3/ctr0',
                                        videoSyncChannel='Dev3/ctr1',
                                        audioFrequency=int(self.audioFrequencyEntry.get()),
                                        videoFrequency=int(self.videoFrequencyEntry.get()),
                                        ready=ready,
                                        readyToStop=readyToStop)


        # processList = self.videoAcquireProcesses+self.videoWriteProcesses+self.audioAcquireProcesses+self.audioWriteProcesses
        # processNameList = [camSerial+"-a" for camSerial in self.camSerials] + [camSerial+"-w" for camSerial in self.camSerials] + [channel+"-a" for channel in self.audioDAQChannels] + [channel+"-w" for channel in self.audioDAQChannels]
        # queueList = imageQueues+audioQueues
        # queueNameList = [camSerial+"-q" for camSerial in self.camSerials] + [channel+"-q" for channel in self.audioDAQChannels]
        time.sleep(1)
        self.syncProcess.start()
        self.monitorProcesses() #processList, processNameList, queueList, queueNameList)

        for cam in camList:
            cam.DeInit()
            del cam

        camList.Clear()

        system.ReleaseInstance()

    def update(self):
        self.videoMonitorMasterFrame.grid(row=0, column=0)
        wV, hV = getOptimalMonitorGrid(len(self.camSerials))
        for k, camSerial in enumerate(self.camSerials):
            self.videoMonitorFrames[camSerial].grid(row=2*(k // wV), column = k % wV)
            self.videoMonitors[camSerial].grid(row=0, column=0, columnspan=2)
            self.cameraAttributeBrowserButtons[camSerial].grid(row=1, column=0)
            self.videoStateWidgets[camSerial].grid(row=1, column=1)

        self.audioMonitorMasterFrame.grid(row=0, column=1, sticky=tk.N)
        wA, hA = getOptimalMonitorGrid(len(self.audioDAQChannels))
        for k, channel in enumerate(self.audioDAQChannels):
            self.audioMonitorFrames[channel].pack()
            self.audioDAQChannelWidgets[channel]['figureCanvas'].get_tk_widget().pack(side=tk.LEFT, fill=tk.BOTH, expand=1)
            self.audioStateWidgets[channel].pack(side=tk.BOTTOM)

        self.controlFrame.grid(row=1, columnspan=2)
        self.startAcquisitionButton.grid(row=0, column=0)
        self.stopAcquisitionButton.grid(row=0, column=1)
        self.audioFrequencyEntry.grid(row=1, column=1)
        self.audioFrequencyLabel.grid(row=1, column=0)
        self.videoFrequencyEntry.grid(row=1, column=3)
        self.videoFrequencyLabel.grid(row=1, column=2)

def clearQueue(q):
    while True:
        try:
            q.get(True, 0.1)
        except queue.Empty:
            break


if __name__ == "__main__":
    root = tk.Tk()
    p = PyVAQ(root)
    root.mainloop()

'''
cd "Dropbox\Documents\Work\Cornell Lab Tech\Projects\Video VI\PyVAQ"
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
#         print("Current state is: ", state)
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
#     print("Manual process done")
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
