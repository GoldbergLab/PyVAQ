import random as r
import PIL
import multiprocessing as mp
import time
from pathlib import Path
import numpy as np
import queue

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

PixelFormat_Mono8 = None
PixelFormat_Mono16 = None
PixelFormat_RGB8Packed = None
PixelFormat_BayerGR8 = None
PixelFormat_BayerRG8 = None
PixelFormat_BayerGB8 = None
PixelFormat_BayerBG8 = None
PixelFormat_BayerGR16 = None
PixelFormat_BayerRG16 = None
PixelFormat_BayerGB16 = None
PixelFormat_BayerBG16 = None
PixelFormat_Mono12Packed = None
PixelFormat_BayerGR12Packed = None
PixelFormat_BayerRG12Packed = None
PixelFormat_BayerGB12Packed = None
PixelFormat_BayerBG12Packed = None
PixelFormat_YUV411Packed = None
PixelFormat_YUV422Packed = None
PixelFormat_YUV444Packed = None
PixelFormat_Mono12p = None
PixelFormat_BayerGR12p = None
PixelFormat_BayerRG12p = None
PixelFormat_BayerGB12p = None
PixelFormat_BayerBG12p = None
PixelFormat_YCbCr8 = None
PixelFormat_YCbCr422_8 = None
PixelFormat_YCbCr411_8 = None
PixelFormat_BGR8 = None
PixelFormat_BGRa8 = None
PixelFormat_Mono10Packed = None
PixelFormat_BayerGR10Packed = None
PixelFormat_BayerRG10Packed = None
PixelFormat_BayerGB10Packed = None
PixelFormat_BayerBG10Packed = None
PixelFormat_Mono10p = None
PixelFormat_BayerGR10p = None
PixelFormat_BayerRG10p = None
PixelFormat_BayerGB10p = None
PixelFormat_BayerBG10p = None
PixelFormat_Mono1p = None
PixelFormat_Mono2p = None
PixelFormat_Mono4p = None
PixelFormat_Mono8s = None
PixelFormat_Mono10 = None
PixelFormat_Mono12 = None
PixelFormat_Mono14 = None
PixelFormat_Mono16s = None
PixelFormat_Mono32f = None
PixelFormat_BayerBG10 = None
PixelFormat_BayerBG12 = None
PixelFormat_BayerGB10 = None
PixelFormat_BayerGB12 = None
PixelFormat_BayerGR10 = None
PixelFormat_BayerGR12 = None
PixelFormat_BayerRG10 = None
PixelFormat_BayerRG12 = None
PixelFormat_RGBa8 = None
PixelFormat_RGBa10 = None
PixelFormat_RGBa10p = None
PixelFormat_RGBa12 = None
PixelFormat_RGBa12p = None
PixelFormat_RGBa14 = None
PixelFormat_RGBa16 = None
PixelFormat_RGB8 = None
PixelFormat_RGB8_Planar = None
PixelFormat_RGB10 = None
PixelFormat_RGB10_Planar = None
PixelFormat_RGB10p = None
PixelFormat_RGB10p32 = None
PixelFormat_RGB12 = None
PixelFormat_RGB12_Planar = None
PixelFormat_RGB12p = None
PixelFormat_RGB14 = None
PixelFormat_RGB16 = None
PixelFormat_RGB16s = None
PixelFormat_RGB32f = None
PixelFormat_RGB16_Planar = None
PixelFormat_RGB565p = None
PixelFormat_BGRa10 = None
PixelFormat_BGRa10p = None
PixelFormat_BGRa12 = None
PixelFormat_BGRa12p = None
PixelFormat_BGRa14 = None
PixelFormat_BGRa16 = None
PixelFormat_RGBa32f = None
PixelFormat_BGR10 = None
PixelFormat_BGR10p = None
PixelFormat_BGR12 = None
PixelFormat_BGR12p = None
PixelFormat_BGR14 = None
PixelFormat_BGR16 = None
PixelFormat_BGR565p = None
PixelFormat_R8 = None
PixelFormat_R10 = None
PixelFormat_R12 = None
PixelFormat_R16 = None
PixelFormat_G8 = None
PixelFormat_G10 = None
PixelFormat_G12 = None
PixelFormat_G16 = None
PixelFormat_B8 = None
PixelFormat_B10 = None
PixelFormat_B12 = None
PixelFormat_B16 = None
PixelFormat_Coord3D_ABC8 = None
PixelFormat_Coord3D_ABC8_Planar = None
PixelFormat_Coord3D_ABC10p = None
PixelFormat_Coord3D_ABC10p_Planar = None
PixelFormat_Coord3D_ABC12p = None
PixelFormat_Coord3D_ABC12p_Planar = None
PixelFormat_Coord3D_ABC16 = None
PixelFormat_Coord3D_ABC16_Planar = None
PixelFormat_Coord3D_ABC32f = None
PixelFormat_Coord3D_ABC32f_Planar = None
PixelFormat_Coord3D_AC8 = None
PixelFormat_Coord3D_AC8_Planar = None
PixelFormat_Coord3D_AC10p = None
PixelFormat_Coord3D_AC10p_Planar = None
PixelFormat_Coord3D_AC12p = None
PixelFormat_Coord3D_AC12p_Planar = None
PixelFormat_Coord3D_AC16 = None
PixelFormat_Coord3D_AC16_Planar = None
PixelFormat_Coord3D_AC32f = None
PixelFormat_Coord3D_AC32f_Planar = None
PixelFormat_Coord3D_A8 = None
PixelFormat_Coord3D_A10p = None
PixelFormat_Coord3D_A12p = None
PixelFormat_Coord3D_A16 = None
PixelFormat_Coord3D_A32f = None
PixelFormat_Coord3D_B8 = None
PixelFormat_Coord3D_B10p = None
PixelFormat_Coord3D_B12p = None
PixelFormat_Coord3D_B16 = None
PixelFormat_Coord3D_B32f = None
PixelFormat_Coord3D_C8 = None
PixelFormat_Coord3D_C10p = None
PixelFormat_Coord3D_C12p = None
PixelFormat_Coord3D_C16 = None
PixelFormat_Coord3D_C32f = None
PixelFormat_Confidence1 = None
PixelFormat_Confidence1p = None
PixelFormat_Confidence8 = None
PixelFormat_Confidence16 = None
PixelFormat_Confidence32f = None
PixelFormat_BiColorBGRG8 = None
PixelFormat_BiColorBGRG10 = None
PixelFormat_BiColorBGRG10p = None
PixelFormat_BiColorBGRG12 = None
PixelFormat_BiColorBGRG12p = None
PixelFormat_BiColorRGBG8 = None
PixelFormat_BiColorRGBG10 = None
PixelFormat_BiColorRGBG10p = None
PixelFormat_BiColorRGBG12 = None
PixelFormat_BiColorRGBG12p = None
PixelFormat_SCF1WBWG8 = None
PixelFormat_SCF1WBWG10 = None
PixelFormat_SCF1WBWG10p = None
PixelFormat_SCF1WBWG12 = None
PixelFormat_SCF1WBWG12p = None
PixelFormat_SCF1WBWG14 = None
PixelFormat_SCF1WBWG16 = None
PixelFormat_SCF1WGWB8 = None
PixelFormat_SCF1WGWB10 = None
PixelFormat_SCF1WGWB10p = None
PixelFormat_SCF1WGWB12 = None
PixelFormat_SCF1WGWB12p = None
PixelFormat_SCF1WGWB14 = None
PixelFormat_SCF1WGWB16 = None
PixelFormat_SCF1WGWR8 = None
PixelFormat_SCF1WGWR10 = None
PixelFormat_SCF1WGWR10p = None
PixelFormat_SCF1WGWR12 = None
PixelFormat_SCF1WGWR12p = None
PixelFormat_SCF1WGWR14 = None
PixelFormat_SCF1WGWR16 = None
PixelFormat_SCF1WRWG8 = None
PixelFormat_SCF1WRWG10 = None
PixelFormat_SCF1WRWG10p = None
PixelFormat_SCF1WRWG12 = None
PixelFormat_SCF1WRWG12p = None
PixelFormat_SCF1WRWG14 = None
PixelFormat_SCF1WRWG16 = None
PixelFormat_YCbCr8_CbYCr = None
PixelFormat_YCbCr10_CbYCr = None
PixelFormat_YCbCr10p_CbYCr = None
PixelFormat_YCbCr12_CbYCr = None
PixelFormat_YCbCr12p_CbYCr = None
PixelFormat_YCbCr411_8_CbYYCrYY = None
PixelFormat_YCbCr422_8_CbYCrY = None
PixelFormat_YCbCr422_10 = None
PixelFormat_YCbCr422_10_CbYCrY = None
PixelFormat_YCbCr422_10p = None
PixelFormat_YCbCr422_10p_CbYCrY = None
PixelFormat_YCbCr422_12 = None
PixelFormat_YCbCr422_12_CbYCrY = None
PixelFormat_YCbCr422_12p = None
PixelFormat_YCbCr422_12p_CbYCrY = None
PixelFormat_YCbCr601_8_CbYCr = None
PixelFormat_YCbCr601_10_CbYCr = None
PixelFormat_YCbCr601_10p_CbYCr = None
PixelFormat_YCbCr601_12_CbYCr = None
PixelFormat_YCbCr601_12p_CbYCr = None
PixelFormat_YCbCr601_411_8_CbYYCrYY = None
PixelFormat_YCbCr601_422_8 = None
PixelFormat_YCbCr601_422_8_CbYCrY = None
PixelFormat_YCbCr601_422_10 = None
PixelFormat_YCbCr601_422_10_CbYCrY = None
PixelFormat_YCbCr601_422_10p = None
PixelFormat_YCbCr601_422_10p_CbYCrY = None
PixelFormat_YCbCr601_422_12 = None
PixelFormat_YCbCr601_422_12_CbYCrY = None
PixelFormat_YCbCr601_422_12p = None
PixelFormat_YCbCr601_422_12p_CbYCrY = None
PixelFormat_YCbCr709_8_CbYCr = None
PixelFormat_YCbCr709_10_CbYCr = None
PixelFormat_YCbCr709_10p_CbYCr = None
PixelFormat_YCbCr709_12_CbYCr = None
PixelFormat_YCbCr709_12p_CbYCr = None
PixelFormat_YCbCr709_411_8_CbYYCrYY = None
PixelFormat_YCbCr709_422_8 = None
PixelFormat_YCbCr709_422_8_CbYCrY = None
PixelFormat_YCbCr709_422_10 = None
PixelFormat_YCbCr709_422_10_CbYCrY = None
PixelFormat_YCbCr709_422_10p = None
PixelFormat_YCbCr709_422_10p_CbYCrY = None
PixelFormat_YCbCr709_422_12 = None
PixelFormat_YCbCr709_422_12_CbYCrY = None
PixelFormat_YCbCr709_422_12p = None
PixelFormat_YCbCr709_422_12p_CbYCrY = None
PixelFormat_YUV8_UYV = None
PixelFormat_YUV411_8_UYYVYY = None
PixelFormat_YUV422_8 = None
PixelFormat_YUV422_8_UYVY = None
PixelFormat_Polarized8 = None
PixelFormat_Polarized10p = None
PixelFormat_Polarized12p = None
PixelFormat_Polarized16 = None
PixelFormat_BayerRGPolarized8 = None
PixelFormat_BayerRGPolarized10p = None
PixelFormat_BayerRGPolarized12p = None
PixelFormat_BayerRGPolarized16 = None
PixelFormat_LLCMono8 = None
PixelFormat_LLCBayerRG8 = None
PixelFormat_JPEGMono8 = None
PixelFormat_JPEGColor8 = None
PixelFormat_Raw16 = None
PixelFormat_Raw8 = None
PixelFormat_R12_Jpeg = None
PixelFormat_GR12_Jpeg = None
PixelFormat_GB12_Jpeg = None
PixelFormat_B12_Jpeg = None
PixelFormat_Mono8 = None
PixelFormat_Mono12Packed = None
PixelFormat_Mono12Packed = None
PixelFormat_Mono16 = None
PixelFormat_BayerGR8 = None
PixelFormat_BayerRG8 = None
PixelFormat_BayerGB8 = None
PixelFormat_BayerBG8 = None
PixelFormat_BayerGR12Packed = None
PixelFormat_BayerRG12Packed = None
PixelFormat_BayerGB12Packed = None
PixelFormat_BayerBG12Packed = None
PixelFormat_BayerGR12Packed = None
PixelFormat_BayerRG12Packed = None
PixelFormat_BayerGB12Packed = None
PixelFormat_BayerBG12Packed = None
PixelFormat_BayerGR16 = None
PixelFormat_BayerRG16 = None
PixelFormat_BayerGB16 = None
PixelFormat_BayerBG16 = None
PixelFormat_YCbCr411_8 = None
PixelFormat_YCbCr422_8 = None
PixelFormat_YCbCr8_CbYCr = None
PixelFormat_RGB8 = None

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
        self.imageIDQueue = None
        self.lastImageID = None
        self.startTime = None
        self.TLDeviceNodeMap = None
        self.TLStreamNodeMap = None
        self.TLNodeMap = None
        self.NodeMap = None

    def IsInitialized(self):
        return self.images is not None
    def IsAcquisitionStarted(self):
        return self.imageIDQueue is not None
    def IsStreaming(self):
        # Not sure exactly what the difference is,
        #   but this is probably good enough.
        return self.IsAcquisitionStarted()

    def GetTLDeviceNodeMap(self):
        if not self.IsInitialized():
            raise RuntimeError('You must call Init and BeginAcquistion on camera before getting images.')
        if self.TLDeviceNodeMap is None:
            self.TLDeviceNodeMap = NodeMap()
            self.TLDeviceNodeMap.AddNode('DeviceSerialNumber', self.serial, 'string')
        return self.TLDeviceNodeMap
    def GetTLStreamNodeMap(self):
        if not self.IsInitialized():
            raise RuntimeError('You must call Init and BeginAcquistion on camera before getting images.')
        if self.TLStreamNodeMap is None:
            self.TLStreamNodeMap = NodeMap()
            # self.TLStreamNodeMap.AddNode('DeviceSerialNumber', self.serial, 'string')
        return self.TLStreamNodeMap
    def GetTLNodeMap(self):
        if not self.IsInitialized():
            raise RuntimeError('You must call Init and BeginAcquistion on camera before getting images.')
        if self.TLNodeMap is None:
            self.TLNodeMap = NodeMap()
            # self.TLNodeMap.AddNode('DeviceSerialNumber', self.serial, 'string')
        return self.TLNodeMap
    def GetNodeMap(self):
        if not self.IsInitialized():
            raise RuntimeError('You must call Init and BeginAcquistion on camera before getting images.')
        if self.NodeMap is None:
            self.NodeMap = NodeMap()
            # self.NodeMap.AddNode('DeviceSerialNumber', self.serial, 'string')
        return self.NodeMap

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

    def UpdateImageIDBuffer(self):
        if self.startTime is None:
            self.startTime = time.time()
        framesSinceStart = int((time.time()-self.startTime) * self.frameRate)
        latentIDs = list(range(self.lastImageID+1, framesSinceStart))
        print('latentIDs: ', latentIDs)
        if len(latentIDs) > 0:
            self.lastImageID = latentIDs[-1]
            for id in latentIDs:
                print('putting id in queue:', id)
                while True:
                    try:
                        self.imageIDQueue.put(id, block=False)
                        break
                    except queue.Full:
                        id_drop = self.imageIDQueue.get(block=False)
                        print('image buffer full - dropping frame #', id_drop)

    def BeginAcquisition(self):
        self.startTime = None
        self.imageIDQueue = queue.Queue(maxsize=30*20)
        self.lastImageID = -1

    def GetNextImage(self, timeout):
        if not self.IsInitialized() or not self.IsAcquisitionStarted():
            raise RuntimeError('You must call Init and BeginAcquistion on camera before getting images.')

        while True:
            try:
                imageID = self.imageIDQueue.get(block=False)
                print('camera produced image #', imageID)
                break
            except queue.Empty:
                print('no image ids in buffer - repopulate')
                time.sleep(1/(4*self.frameRate))
                self.UpdateImageIDBuffer()

        frameNum = imageID % self.NumFrames
        image = self.images[:, :, :, frameNum]
        return Image(image, imageID)

    def EndAcquisition(self):
        self.imageIDQueue = None

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
def CValuePtr(node):
    return node.attribute
def CBasePtr(node):
    return node.attribute
def CRegisterPtr(node):
    return node.attribute
def CEnumEntryPtr(node):
    return node.attribute


intfIString = 'intfIString'
intfIInteger = 'intfIInteger'
intfIFloat = 'intfIFloat'
intfIBoolean = 'intfIBoolean'
intfICommand = 'intfICommand'
intfIEnumeration = 'intfIEnumeration'
intfICategory = 'intfICategory'
intfIValue = 'value',
intfIBase = 'base',
intfIRegister = 'register',
intfIEnumEntry = 'enumEntry',

def IsAvailable(nodeAttribute):
    return True
def IsWritable(nodeAttribute):
    return True
def IsReadable(nodeAttribute):
    return True

class SpinnakerException(Exception):
    pass
