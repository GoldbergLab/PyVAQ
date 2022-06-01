import random as r
import math
import PIL
import multiprocessing as mp
import time
from ctypes import c_bool
from pathlib import Path
import numpy as np
from SharedImageQueue import SharedImageSender, SharedImageReceiver
from pathlib import Path

DEFAULT_SYSTEM_RECORD_FILENAME = 'simCamSystem.tmp'
DEFAULT_SIM_CAMERA_FEED = 'simCameraFeed.gif'

class System:
    def __init__(self, systemRecordFilename=None, n=2, regenerateSystemRecord=False):
        # If no system record filename exists, create a new one with n cameras
        if systemRecordFilename is None:
            systemRecordFilename = DEFAULT_SYSTEM_RECORD_FILENAME

        self.systemRecord = Path(__file__).parents[0] / systemRecordFilename
        if regenerateSystemRecord or not self.systemRecord.exists():
            # No system record file found
            # Generate random camera serials
            serials = ['sim'+str(s+1000000) for s in r.sample(range(1000000), n)]
            # Create system record file
            with open(self.systemRecord, 'w') as f:
                camData = ['{s} {v}'.format(s=serial, v=DEFAULT_SIM_CAMERA_FEED) for serial in serials]
                for camDatum in camData:
                    f.write(camDatum + '\n')

        # Read in system record file
        with open(self.systemRecord, 'r') as f:
            camData = f.readlines()
        serials, videos = zip(*[camDatum.strip().split() for camDatum in camData])

        self.cameraList = CameraList(self, serials=serials)

    @classmethod
    def GetInstance(cls):
        return cls()
    def GetCameras(self):
        return self.cameraList
    def ReleaseInstance(self):
        self.cameraList = None

class CameraList:
    def __init__(self, system, serials):
        self.system = system
        self.cameras = {}
        self.serials = serials
        for serial in self.serials:
            self.cameras[serial] = Camera(serial)
        self.currentCameraIndex = 0
    def __iter__(self):
        return self
    def __next__(self):
        self.currentCameraIndex += 1
        if self.currentCameraIndex - 1 >= len(self.serials):
            raise StopIteration()
        return self.GetBySerial(self.serials[self.currentCameraIndex-1])
    def GetBySerial(self, camSerial):
        return self.cameras[camSerial]
    def Clear(self):
        self.cameras = {}

class Camera:
    DEFAULT_VIDEO_FILE = Path(__file__).parents[0] / 'simCameraFeed.gif'
    def __init__(self, serial, videoFile=DEFAULT_VIDEO_FILE, buffered=True):
        self.serial = serial
        self.frameRate = 30
        self.videoFile = videoFile
        self.images = None
        self.buffered = buffered
        self.Width = None
        self.Height = None
        self.simCamProcess = None
        self.simCamReceiver = None
        self.imageID = None
    def GetTLDeviceNodeMap(self):
        nm = NodeMap()
        nm.AddNode('DeviceSerialNumber', self.serial)
        return nm
    def GetNodeMap(self):
        return NodeMap()
    def Init(self):
        image = PIL.Image.open(self.videoFile)
        width = image.size[0]
        height = image.size[1]
        self.Width = NodeAttribute('Width', width)
        self.Height = NodeAttribute('Height', height)
        self.NumFrames = image.n_frames

        print('Opened image with size {w}x{h}x{c}x{n}'.format(w=width, h=height, c=3, n=self.NumFrames))

        self.images = np.zeros([height, width, 3, self.NumFrames], dtype='uint8')
        for frameNum in range(self.NumFrames):
            image.seek(frameNum)
            self.images[:, :, :, frameNum] = np.asarray(image.convert('RGB'), dtype='uint8')
        image.close()

    def BeginAcquisition(self):
        self.imageID = 0
    def GetNextImage(self):
        if self.imageID is None or self.images is None:
            raise RuntimeError('You must call Init and BeginAcquistion on camera before getting images.')
        frameNum = self.imageID % self.NumFrames
        image = self.images[:, :, :, frameNum]
        imageID = self.imageID
        self.imageID = self.imageID + 1
        return Image(image, imageID)
    def EndAcquisition(self):
        self.imageID = None
    def DeInit(self):
        self.images = None

# class CameraProcess(mp.Process):
#     def __init__(self,
#                 *args,
#                 camSerial='',
#                 frameRate=None,
#                 imageStack=None,
#                 verbose=False,
#                 daemon=True,
#                 **kwargs):
#         mp.Process.__init__(self, *args, daemon=daemon, **kwargs)
#         self.verbose = verbose
#         self.camSerial = camSerial
#         self.frameRate = None
#         self.bufferSize = 10
#         self.imageStack = imageStack
#         self.videoHeight, self.videoWidth, self.nChannels, self.numFrames = imageStack.shape
#         self.kill = mp.Value(c_bool, False)
#         self.imageQueue = SharedImageSender(
#             width=self.videoWidth,
#             height=self.videoHeight,
#             verbose=False,
#             outputType='numpy',
#             outputCopy=False,
#             lockForOutput=False,
#             maxBufferSize=self.bufferSize,
#             channels=1,
#             name=self.camSerial+'____main',
#             allowOverflow=True
#         )
#         self.imageQueueReceiver = self.imageQueue.getReceiver()
#
#     def EndAcquisition(self):
#         self.kill.value = True
#
#     def run(self):
#         print('Initiating simulated acquisition')
#         self.imageQueue.setupBuffers()
#
#         imageID = 0
#         sleepTime = 1/self.frameRate - 0.0001
#
#         while True:
#             frameNum = imageID % self.numFrames
#             self.imageQueue.put(self.images[:, :, :, frameNum], metadata={'imageID':imageID, 'frameNum':frameNum})
#             imageID += 1
#             if self.kill.value:
#                 print('Ending simulated acquisition...')
#                 break
#             time.sleep(sleepTime)
#         print('Simulated acquisition terminated.')

class Image:
    def __init__(self, imArr, imID):
        self.imArr = imArr
        self.imHeight, self.imWidth, self.imChannels = self.imArr.shape
        self.imID = imID
    def GetNDArray(self):
        return self.imArr
    def Release(self):
        self.imArr = None
        self.imID = None
        self.imHeight = None
        self.imWidth = None
        self.imChannels = None
    def IsIncomplete(self):
        return False
    def GetImageStatus(self):
        return 0
    def GetFrameID(self):
        return self.imID
    def GetWidth(self):
        return self.imWidth
    def GetHeight(self):
        return self.imHeight
    def GetPixelFormat(self):
        return 'simulated'
    def GetData(self):
        return self.imArr.tobytes()

class NodeMap:
    def __init__(self, attributes={}):
        self.nodes = {}
        for attributeName in attributes:
            self.AddNode(attributeName, attributes[attributeName])
    def GetNode(self, attributeName):
        if attributeName not in self.nodes:
            self.nodes[attributeName] = Node(attributeName)
        return self.nodes[attributeName]
    def AddNode(self, name, value):
        self.nodes[name] = Node(name, value)

class Node:
    def __init__(self, name, value=None):
        self.attribute = NodeAttribute(name, value)

class NodeAttribute:
    def __init__(self, name, value=None):
        self.name = name
        self.value = value
    def GetEntryByName(self, attributeValue):
        # If it's an enum
        return NodeAttributeValue(attributeValue)
    def GetValue(self):
        return self.value
    def SetIntValue(self, value):
        self.value = value
    def SetValue(self, value):
        self.value = value

class NodeAttributeValue:
    def __init__(self, attributeValue):
        self.value = attributeValue
    def GetValue(self):
        return self.value

def CStringPtr(node):
    return node.attribute
def CIntegerPtr(node):
    return node.attribute
def CFloatPtr(node):
    return node.attribute
def CBooleanPtr(node):
    return node.attribute
def CEnumerationPtr(node):
    return node.attribute
def CEnumerationPtr(node):
    return node.attribute
def CCategoryPtr(node):
    return node.attribute

intfIString = 'intfIString'
intfIInteger = 'intfIInteger'
intfIFloat = 'intfIFloat'
intfIBoolean = 'intfIBoolean'
intfICommand = 'intfICommand'
intfIEnumeration = 'intfIEnumeration'
intfICategory = 'intfICategory'

def IsAvailable(nodeAttribute):
    return True
def IsWritable(nodeAttribute):
    return True
def IsReadable(nodeAttribute):
    return True

# system = PySpin.System.GetInstance()
# camList = system.GetCameras()
# cam = camList.GetBySerial(self.camSerial)
# cam.Init()
#
# nodemap = cam.GetNodeMap()
# self.setCameraAttributes(nodemap, self.acquireSettings)
# cam.BeginAcquisition()
# imageResult = cam.GetNextImage()
# camList.Clear()
# cam.EndAcquisition()
# cam.DeInit()
# system.ReleaseInstance()
