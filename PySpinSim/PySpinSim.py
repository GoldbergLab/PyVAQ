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

pixelSizes = {
    1:'Bpp1',
    2:'Bpp2',
    4:'Bpp4',
    8:'Bpp8',
    10:'Bpp10',
    12:'Bpp12',
    14:'Bpp14',
    16:'Bpp16',
    20:'Bpp20',
    24:'Bpp24',
    30:'Bpp30',
    32:'Bpp32',
    36:'Bpp36',
    48:'Bpp48',
    64:'Bpp64',
    96:'Bpp96',
}

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
        self.startTime = None
        self.TLDeviceNodeMap = None
        self.NodeMap = None
    def IsInitialized(self):
        return self.images is not None
    def IsAcquisitionStarted(self):
        return self.imageID is not None
    def GetTLDeviceNodeMap(self):
        if not self.IsInitialized():
            raise RuntimeError('You must call Init and BeginAcquistion on camera before getting images.')
        if self.TLDeviceNodeMap is None:
            self.TLDeviceNodeMap = NodeMap()
            self.TLDeviceNodeMap.AddNode('DeviceSerialNumber', self.serial, 'string')
        return self.TLDeviceNodeMap
    def GetNodeMap(self):
        if not self.IsInitialized():
            raise RuntimeError('You must call Init and BeginAcquistion on camera before getting images.')
        if self.NodeMap is None:
            self.NodeMap = NodeMap()
        return self.NodeMap
    def Init(self):
        image = PIL.Image.open(self.videoFile)
        imageShape = np.asarray(image.convert('RGB'), dtype='uint8').shape
        width = imageShape[1]
        height = imageShape[0]
        if len(imageShape) < 3:
            nChannels = 1
        else:
            nChannels = imageShape[2]
        bitsPerChannel = 8
        pixelSizeName = pixelSizes[bitsPerChannel * nChannels]

        self.Width = NodeAttribute('Width', width, type='int')
        self.Height = NodeAttribute('Height', height, type='int')
        self.NumFrames = image.n_frames

        print('Opened image with size {w}x{h}x{c}x{n}'.format(w=width, h=height, c=nChannels, n=self.NumFrames))
        self.images = np.zeros([height, width, nChannels, self.NumFrames], dtype='uint8')
        for frameNum in range(self.NumFrames):
            image.seek(frameNum)
            self.images[:, :, :, frameNum] = np.asarray(image.convert('RGB'), dtype='uint8')
        image.close()

        nm = self.GetNodeMap()
        nm.AddNode('PixelSize', pixelSizeName, 'enum', readOnly=True)
        nm.AddNode('PixelDynamicRangeMax', np.iinfo(self.images.dtype).max, type='int', readOnly=True)
        nm.AddNode('PixelDynamicRangeMin', np.iinfo(self.images.dtype).min, type='int', readOnly=True)
        nm.AddNode('PixelFormat', 'RGB8', type='enum', readOnly=True)

    def GetCurrentImageID(self):
        return int((time.time()-self.startTime) * self.frameRate)

    def BeginAcquisition(self):
        self.startTime = time.time()
        self.imageID = self.GetCurrentImageID()

    def GetNextImage(self):
        if not self.IsInitialized() or not self.IsAcquisitionStarted():
            raise RuntimeError('You must call Init and BeginAcquistion on camera before getting images.')
        currentImageID = self.imageID
        while currentImageID == self.imageID:
            # Wait until it's time to release a new image
            time.sleep(1/(4*self.frameRate))
            currentImageID = self.GetCurrentImageID()
        self.imageID = currentImageID

        frameNum = self.imageID % self.NumFrames
        image = self.images[:, :, :, frameNum]
        imageID = self.imageID
        return Image(image, imageID)

    def EndAcquisition(self):
        self.imageID = None

    def DeInit(self):
        self.images = None

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
            self.AddNode(attributeName, value=attributes[attributeName][0], type=attributes[attributeName][1])
    def GetNode(self, attributeName):
        if attributeName not in self.nodes:
            self.nodes[attributeName] = Node(attributeName)
        return self.nodes[attributeName]
    def AddNode(self, name, value, type, readOnly=False):
        self.nodes[name] = Node(name, value, type, readOnly=readOnly)

class Node:
    def __init__(self, name, value=None, type=None, readOnly=False):
        self.attribute = NodeAttribute(name, value=value, type=type, readOnly=readOnly)

class NodeAttribute:
    def __init__(self, name, value=None, type=None, readOnly=False):
        self.name = name
        self.value = value
        self.type = type
        self.readOnly = readOnly
    def GetEntryByName(self, attributeValue):
        # If it's an enum
        return NodeAttributeValue(attributeValue)
    def GetCurrentEntry(self):
        return NodeAttributeValue(self.value)
    def GetValue(self):
        if self.type == 'enum':
            raise AttributeError('I guess you can\' just get the value of an enum or whatever. Try GetCurrentEntry instead.')
        return self.value
    def SetIntValue(self, value):
        if self.readOnly:
            print('Warning, tried to set {n} to {v}, but it was marked as read only. No changes made.'.format(n=self.name, v=self.value))
        else:
            self.value = value
    def SetValue(self, value):
        if self.readOnly:
            print('Warning, tried to set {n} to {v}, but it was marked as read only. No changes made.'.format(n=self.name, v=self.value))
        else:
            self.value = value

class NodeAttributeValue:
    def __init__(self, attributeValue):
        self.value = attributeValue
    def GetValue(self):
        return self.value
    def GetName(self):
        return self.value
    def GetDisplayName(self):
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

class SpinnakerException(Exception):
    pass
