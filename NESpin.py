import numpy as np, cv2, threading, time, sys, os, queue
import clr
from System import IntPtr
from System.Runtime.InteropServices import Marshal
from System.Collections.Generic import List
from pathlib import Path

root = Path(__file__).parent.resolve()
SDK_DIR = root / "lib" / "NanEye"                  # folder containing the Awaiba/ams-OSRAM DLLs
FW_IMG  = root / "lib" / "NanEye" / "firmware" / "fx3_fw_2EP.img"     # same as C# Form1
FPGA_BIN = root / "lib" / "NanEye" / "firmware" / "fob_fpga_v08.bin"  # same as C# Form1

# Make sure .NET can load the assemblies
sys.path.append(str(SDK_DIR))

# Load as importable assemblies
clr.AddReference(str(SDK_DIR / "Awaiba.dll"))
clr.AddReference(str(SDK_DIR / "AwSensors.dll"))
clr.AddReference(str(SDK_DIR / "NanEyeUSB3Provider.dll"))
clr.AddReference(str(SDK_DIR / "CyUSB"))

# Import .NET types after references are loaded
from Awaiba.Drivers.Grabbers.NanEye2D.FobUsb3 import NanEyeFobProvider

# A module designed to be a partial drop-in replacement for PySpin, so
#   ams-OSRAM "NanEye" cameras with the "Fiber Optic Box 2.0" receiver can be
#   used with mostly the same class/function call signature.

CameraAttributes = dict(
    # POS_MSEC='POS_MSEC',
    # POS_FRAMES='POS_FRAMES',
    # POS_AVI_RATIO='POS_AVI_RATIO',
    FRAME_WIDTH='FRAME_WIDTH',
    FRAME_HEIGHT='FRAME_HEIGHT',
    BIT_DEPTH='BIT_DEPTH',
    FPS='FPS',
    # FOURCC='FOURCC',
    # FRAME_COUNT='FRAME_COUNT',
    # FORMAT='FORMAT',
    # MODE='MODE',
    # BRIGHTNESS='BRIGHTNESS',
    # CONTRAST='CONTRAST',
    # SATURATION='SATURATION',
    # HUE='HUE',
    # GAIN='GAIN',
    # EXPOSURE='EXPOSURE',
    # CONVERT_RGB='CONVERT_RGB',
    # WHITE_BALANCE_BLUE_U='WHITE_BALANCE_BLUE_U',
    # RECTIFICATION='RECTIFICATION',
    # MONOCHROME='MONOCHROME',
    # SHARPNESS='SHARPNESS',
    # AUTO_EXPOSURE='AUTO_EXPOSURE',
    # GAMMA='GAMMA',
    # TEMPERATURE='TEMPERATURE',
    # TRIGGER='TRIGGER',
    # TRIGGER_DELAY='TRIGGER_DELAY',
    # WHITE_BALANCE_RED_V='WHITE_BALANCE_RED_V',
    # ZOOM='ZOOM',
    # FOCUS='FOCUS',
    # GUID='GUID',
    # ISO_SPEED='ISO_SPEED',
    # BACKLIGHT='BACKLIGHT',
    # PAN='PAN',
    # TILT='TILT',
    # ROLL='ROLL',
    # IRIS='IRIS',
    # SETTINGS='SETTINGS',
    # BUFFERSIZE='BUFFERSIZE',
    # AUTOFOCUS='AUTOFOCUS',
    # SAR_NUM='SAR_NUM',
    # SAR_DEN='SAR_DEN',
    # BACKEND='BACKEND',
    CHANNEL='CHANNEL',
    # AUTO_WB='AUTO_WB',
    # WB_TEMPERATURE='WB_TEMPERATURE',
    # CODEC_PIXEL_FORMAT='CODEC_PIXEL_FORMAT',
    # BITRATE='BITRATE',
    # ORIENTATION_META='ORIENTATION_META',
    # ORIENTATION_AUTO='ORIENTATION_AUTO',
    # OPEN_TIMEOUT_MSEC='OPEN_TIMEOUT_MSEC',
    # READ_TIMEOUT_MSEC='READ_TIMEOUT_MSEC'
)

CameraAttributeAccessMode = dict(
#     POS_MSEC='RW',
#     POS_FRAMES='RW',
#     POS_AVI_RATIO='RW',
    FRAME_WIDTH='R',
    FRAME_HEIGHT='R',
    BIT_DEPTH='R',
    FPS='R',
#     FOURCC='RW',
#     FRAME_COUNT='RW',
#     FORMAT='RW',
#     MODE='RW',
#     BRIGHTNESS='RW',
#     CONTRAST='RW',
#     SATURATION='RW',
#     HUE='RW',
#     GAIN='RW',
#     EXPOSURE='RW',
#     CONVERT_RGB='RW',
#     WHITE_BALANCE_BLUE_U='RW',
#     RECTIFICATION='RW',
#     MONOCHROME='RW',
#     SHARPNESS='RW',
#     AUTO_EXPOSURE='RW',
#     GAMMA='RW',
#     TEMPERATURE='RW',
#     TRIGGER='RW',
#     TRIGGER_DELAY='RW',
#     WHITE_BALANCE_RED_V='RW',
#     ZOOM='RW',
#     FOCUS='RW',
#     GUID='RW',
#     ISO_SPEED='RW',
#     BACKLIGHT='RW',
#     PAN='RW',
#     TILT='RW',
#     ROLL='RW',
#     IRIS='RW',
#     SETTINGS='RW',
#     BUFFERSIZE='RW',
#     AUTOFOCUS='RW',
#     SAR_NUM='RW',
#     SAR_DEN='RW',
#     BACKEND='RO',
    CHANNEL='R',
#     AUTO_WB='RW',
#     WB_TEMPERATURE='RW',
#     CODEC_PIXEL_FORMAT='RO',
#     BITRATE='RO',
#     ORIENTATION_META='RO',
#     ORIENTATION_AUTO='RW',
#     OPEN_TIMEOUT_MSEC='RW',
#     READ_TIMEOUT_MSEC='RW',
)

# For compatibility with PySpin
AlternateCameraAttributeNames = dict(
    AcquisitionFrameRate='FPS',
)

def GetAttributeCode(attributeName):
    """Attempt to translate a human-readable attribute name into a  code

    This takes a human-readable attribute name and attempts to translate it into
        a valid OpenCV VideoCaptureProperty code, using the CameraAttributes and
        AlternateCameraAttributeNames dictionaries.

    Args:
        attributeName (str): The attribute name to translate into a code

    Returns:
        int: OpenCV VideoCaptureProperty code

    """
    # Attempt to translate the attributeName into a valid OpenCV VideoCaptureProperty code
    try:
        attributeCode = CameraAttributes[attributeName]
        return attributeCode
    except KeyError:
        try:
            # Perhaps this is an alternate attribute name?
            attributeName = AlternateCameraAttributeNames[attributeName]
            attributeCode = CameraAttributes[attributeName]
        except KeyError:
            raise NameError('Attribute name {n} not recognized.'.format(n=attributeName))
        return attributeCode

class System:
    """
    The system object is used to retrieve the list of interfaces and
    cameras available.

    C++ includes: System.h
    """

    _reserved = False

    def __init__(self, *args, sensor1=True, sensor2=True, _safe=False, **kwargs):
        # Create provider
        if not _safe:
            raise RuntimeError('Use GetInstance to create System')

        self._camera_list = None

        self._capture_started = False

        self._connection_successful = False

        self._provider = NanEyeFobProvider()

        # Point to firmware/FPGA (same as in C#)
        self._provider.SetFWFile(str(FW_IMG))
        self._provider.SetFpgaFile(str(FPGA_BIN))

        self._sensor1 = sensor1
        self._sensor2 = sensor2

        # Enable specified sensors
        sensors = List[bool]()
        sensors.Add(self._sensor1)
        sensors.Add(self._sensor2)
        self._provider.Sensors = sensors

        self._connection_successful = True

        self._queue_size = 100              # number of prealloc slots

        # Preallocate rings of numpy buffers
        self._empty_buffer = queue.Queue(maxsize=self._queue_size)
        self._image_buffer = queue.Queue(maxsize=self._queue_size)

    def _start_capture(self):
        # Register frame ready callback
        self._provider.ImageProcessed += self._frame_ready_callback
        # Initiate camera capture
        if not self._capture_started:
            self._provider.StartCapture()              # :contentReference[oaicite:14]{index=14}
        self._capture_started = True

    def _stop_capture(self):
        if self._capture_started:
            self._provider.StopCapture()
        self._capture_started = False

    def _frame_ready_callback(self, sender, e):
        # FAST path: copy raw bytes, enqueue index, return
        # choose one raw source:
        # src = e.GetImageData.GetRawPixels1Byte  # fast 8-bit
        # src = e.GetImageData.GetRawPixels2Byte # 10-bit expanded (2 bytes/px)
        src = e.PixelData                      # RGB processed (3 bytes/px)

        # Other available fields for "e", the new image:
        # e.GetImageData = Awaiba.FrameProcessing.ImageData
        # e.ProcessingTime = 2182.0
        # e.FramesTime = 2598330880.0
        # e.BoardId = 0
        # e.FrameCount = 218
        # e.Width = 250
        # e.Height = 250
        # e.BitsPerPixel = 10
        # e.TimeStamp = 2598330910
        # e.PixelData = System.Byte[]
        # e.SensorID = 0
        # e.GetImageData.GetProcessedDataRGBByte = System.Byte[]
        # e.GetImageData.GetProcessedDataARGBByte = System.Byte[]
        # e.GetImageData.GetRawPixelsUShort = System.UInt16[]
        # e.GetImageData.GetRawPixels2Byte = System.Byte[]
        # e.GetImageData.GetRawPixels1Byte = System.Byte[]

        # Determine which NanEye camera this came from
        sensorID = e.SensorID

        # Pull a recycled buffer to write the new image data to.
        #   If none are availbale, a queue.Empty exception will be raised
        try:
            buf = self._empty_buffer.get(block=False)
        except queue.Empty:
            print('No empty buffers available to write new image to!')
            return

        # Copy the new image data into the free buffer
        Marshal.Copy(src, 0, IntPtr(buf.ctypes.data), src.Length)

        # Put the buffer containing the new image data into the appropriate image queue
        #   queue. If that queue is full, a queue.Full exception will be raised
        try:
            self._image_buffer.put((buf, e.SensorID, e.FrameCount, e.TimeStamp), block=False)
        except queue.Empty:
            print('No room in image buffers available to write new image to!')
            return


    def _get_width(self):
        return self._provider.Width

    def _get_height(self):
        return self._provider.Height

    def GetInstance():
        """GetInstance() -> SystemPtr"""
        if System._reserved:
            raise IOError("Only one NanEye System class may exist at once")
        System._reserved = True

        system = System(_safe=True)
        return system

    def ReleaseInstance(self):
        """
        ReleaseInstance(self)

        Parameters
        ----------
        self: System

        void
        Spinnaker::System::ReleaseInstance()

        This call releases the instance of the System Singleton for this
        process. After successfully releasing the System instance the pointer
        returned by GetInstance() will be invalid. Calling ReleaseInstance
        while a camera reference is still held will throw an error of type
        SPINNAKER_ERR_RESOURCE_IN_USE.

        See:  Error

        See:   GetInstance()
        """
        self._camera_list.Clear()
        self._provider.Dispose()
        System._reserved = False
        return

    def GetInterfaces(self, updateInterface=True):
        """
        GetInterfaces(self, updateInterface=True) -> InterfaceList

        Parameters
        ----------
        updateInterface: bool

        GetInterfaces(self) -> InterfaceList

        Parameters
        ----------
        self: Spinnaker::System *

        InterfaceList Spinnaker::System::GetInterfaces(bool
        updateInterface=true)

        Returns a list of interfaces available on the system. This call
        returns GigE and Usb2 and Usb3 interfaces.

        Parameters:
        -----------

        updateInterface:  Determines whether or not UpdateInterfaceList() is
        called before getting available interfaces

        An InterfaceList object that contains a list of all interfaces.
        """
        raise NotImplementedError()
        return

    def UpdateInterfaceList(self):
        """
        UpdateInterfaceList(self)

        Parameters
        ----------
        self: Spinnaker::System *

        """
        raise NotImplementedError()
        return

    def GetCameras(self, updateInterfaces=True, updateCameras=True):
        """
        GetCameras(self, updateInterfaces=True, updateCameras=True) -> CameraList

        Parameters
        ----------
        updateInterfaces: bool
        updateCameras: bool

        GetCameras(self, updateInterfaces=True) -> CameraList

        Parameters
        ----------
        updateInterfaces: bool

        GetCameras(self) -> CameraList

        Parameters
        ----------
        self: Spinnaker::System *

        CameraList
        Spinnaker::System::GetCameras(bool updateInterfaces=true, bool
        updateCameras=true)

        Returns a list of cameras that are available on the system. This call
        returns both GigE Vision and Usb3 Vision cameras from all interfaces.
        The camera list object will reference count the cameras it returns. It
        is important that the camera list is destroyed or is cleared before
        calling system-> ReleaseInstance() or else the call to system->
        ReleaseInstance() will result in an error message thrown that a
        reference to the camera is still held.

        See:   ReleaseInstance()

        See:   CameraList::Clear()

        Parameters:
        -----------

        updateInterfaces:  Determines whether or not updateInterfaceList() is
        called before getting cameras from available interfaces on the system

        updateCameras:  Determines whether or not UpdateCameras() is called
        before getting cameras from available interfaces on the system

        An CameraList object that contains a list of all cameras.
        """

        self._camera_list = CameraList(np.array([0, 1])[[self._sensor1, self._sensor2]], self._empty_buffer, self._image_buffer, system=self)
        return self._camera_list

    def UpdateCameras(self, updateInterfaces=True):
        """
        UpdateCameras(self, updateInterfaces=True) -> bool

        Parameters
        ----------
        updateInterfaces: bool

        UpdateCameras(self) -> bool

        Parameters
        ----------
        self: Spinnaker::System *

        bool
        Spinnaker::System::UpdateCameras(bool updateInterfaces=true)

        Updates the list of cameras on the system. Note that
        System::GetCameras() internally calls UpdateCameras() for each
        interface it enumerates. If the list changed between this call and the
        last time UpdateCameras was called then the return value will be true,
        otherwise it is false.

        See:   GetCameras()

        Parameters:
        -----------

        updateInterfaces:  Determines whether or not UpdateInterfaceList() is
        called before updating cameras for available interfaces on the system

        True if cameras changed on interface and false otherwise.
        """
        raise NotImplementedError()
        return

    def RegisterEventHandler(self, evtHandlerToRegister):
        """
        RegisterEventHandler(self, evtHandlerToRegister)

        Parameters
        ----------
        evtHandlerToRegister: Spinnaker::EventHandler &

        """
        raise NotImplementedError()
        return

    def UnregisterEventHandler(self, evtHandlerToUnregister):
        """
        UnregisterEventHandler(self, evtHandlerToUnregister)

        Parameters
        ----------
        evtHandlerToUnregister: Spinnaker::EventHandler &

        """
        raise NotImplementedError()
        return

    def RegisterInterfaceEventHandler(self, evtHandlerToRegister, updateInterface=True):
        """
        RegisterInterfaceEventHandler(self, evtHandlerToRegister, updateInterface=True)

        Parameters
        ----------
        evtHandlerToRegister: Spinnaker::EventHandler &
        updateInterface: bool

        RegisterInterfaceEventHandler(self, evtHandlerToRegister)

        Parameters
        ----------
        evtHandlerToRegister: Spinnaker::EventHandler &

        """
        raise NotImplementedError()
        return

    def UnregisterInterfaceEventHandler(self, evtHandlerToUnregister):
        """
        UnregisterInterfaceEventHandler(self, evtHandlerToUnregister)

        Parameters
        ----------
        evtHandlerToUnregister: Spinnaker::EventHandler &

        """
        raise NotImplementedError()
        return

    def RegisterLoggingEventHandler(self, handler):
        """
        RegisterLoggingEventHandler(self, handler)

        Parameters
        ----------
        handler: Spinnaker::LoggingEventHandler &

        """
        raise NotImplementedError()
        return

    def UnregisterAllLoggingEventHandlers(self):
        """
        UnregisterAllLoggingEventHandlers(self)

        Parameters
        ----------
        self: Spinnaker::System *

        """
        raise NotImplementedError()
        return

    def UnregisterLoggingEventHandler(self, handler):
        """
        UnregisterLoggingEventHandler(self, handler)

        Parameters
        ----------
        handler: Spinnaker::LoggingEventHandler &

        """
        raise NotImplementedError()
        return

    def SetLoggingEventPriorityLevel(self, level):
        """
        SetLoggingEventPriorityLevel(self, level)

        Parameters
        ----------
        level: enum Spinnaker::SpinnakerLogLevel

        void
        Spinnaker::System::SetLoggingEventPriorityLevel(SpinnakerLogLevel
        level)

        Sets a threshold priority level for logging event. Logging events
        below such level will not trigger callbacks.

        Spinnaker uses five levels of logging: Error - failures that are non-
        recoverable without user intervention.

        Warning - failures that are recoverable without user intervention.

        Notice - information about events such as camera arrival and removal,
        initialization and deinitialization, starting and stopping image
        acquisition, and feature modification.

        Info - information about recurring events that are generated regularly
        such as information on individual images.

        Debug - information that can be used to troubleshoot the system.

        See:  SpinnakerLogLevel

        Parameters:
        -----------

        level:  The threshold level
        """
        raise NotImplementedError()
        return

    def GetLoggingEventPriorityLevel(self):
        """
        GetLoggingEventPriorityLevel(self) -> Spinnaker::SpinnakerLogLevel

        Parameters
        ----------
        self: Spinnaker::System *

        SpinnakerLogLevel Spinnaker::System::GetLoggingEventPriorityLevel()

        Retrieves the current logging event priority level.

        Spinnaker uses five levels of logging: Error - failures that are non-
        recoverable without user intervention.

        Warning - failures that are recoverable without user intervention.

        Notice - information about events such as camera arrival and removal,
        initialization and deinitialization, starting and stopping image
        acquisition, and feature modification.

        Info - information about recurring events that are generated regularly
        such as information on individual images.

        Debug - information that can be used to troubleshoot the system.

        See:  SpinnakerLogLevel

        Level The threshold level
        """
        raise NotImplementedError()
        return

    def IsInUse(self):
        """
        IsInUse(self) -> bool

        Parameters
        ----------
        self: Spinnaker::System *

        bool
        Spinnaker::System::IsInUse()

        Checks if the system is in use by any interface or camera objects.

        Returns true if the system is in use and false otherwise.
        """
        return self._provider is not None and self._provider.IsCapturing

    def SendActionCommand(self, deviceKey, groupKey, groupMask, actionTime=0, pResultSize=None, results=0):
        """
        SendActionCommand(self, deviceKey, groupKey, groupMask, actionTime=0, pResultSize=None, results=0)

        Parameters
        ----------
        deviceKey: unsigned int
        groupKey: unsigned int
        groupMask: unsigned int
        actionTime: unsigned long long
        pResultSize: unsigned int *
        results: Spinnaker::ActionCommandResult []

        SendActionCommand(self, deviceKey, groupKey, groupMask, actionTime=0, pResultSize=None)

        Parameters
        ----------
        deviceKey: unsigned int
        groupKey: unsigned int
        groupMask: unsigned int
        actionTime: unsigned long long
        pResultSize: unsigned int *

        SendActionCommand(self, deviceKey, groupKey, groupMask, actionTime=0)

        Parameters
        ----------
        deviceKey: unsigned int
        groupKey: unsigned int
        groupMask: unsigned int
        actionTime: unsigned long long

        SendActionCommand(self, deviceKey, groupKey, groupMask)

        Parameters
        ----------
        deviceKey: unsigned int
        groupKey: unsigned int
        groupMask: unsigned int

        void
        Spinnaker::System::SendActionCommand(unsigned int deviceKey, unsigned
        int groupKey, unsigned int groupMask, unsigned long long actionTime=0,
        unsigned int *pResultSize=0, ActionCommandResult results[]=NULL)

        Broadcast an Action Command to all devices on system

        Parameters:
        -----------

        deviceKey:  The Action Command's device key

        groupKey:  The Action Command's group key

        groupMask:  The Action Command's group mask

        actionTime:  (Optional) Time when to assert a future action. Zero
        means immediate action.

        pResultSize:  (Optional) The number of results in the results array.
        The value passed should be equal to the expected number of devices
        that acknowledge the command. Returns the number of received results.

        results:  (Optional) An Array with *pResultSize elements to hold the
        action command result status. The buffer is filled starting from index
        0. If received results are less than expected number of devices that
        acknowledge the command, remaining results are not changed. If
        received results are more than expected number of devices that
        acknowledge the command, extra results are ignored and not appended to
        array. This parameter is ignored if pResultSize is 0. Thus this
        parameter can be NULL if pResultSize is 0 or NULL.
        """
        raise NotImplementedError()
        return

    def GetLibraryVersion(self):
        """
        GetLibraryVersion(self) -> LibraryVersion

        Parameters
        ----------
        self: Spinnaker::System *

        """
        raise NotImplementedError()
        return

    def GetTLNodeMap(self):
        """
        GetTLNodeMap(self) -> INodeMap

        Parameters
        ----------
        self: Spinnaker::System const *

        """
        raise NotImplementedError()
        return

class CameraList:
    """

    Used to hold a list of camera objects.

    C++ includes: CameraList.h
    """

    def __init__(self, valid_ports, empty_buffer, image_buffer, system=None):
        """
        __init__(self) -> CameraList
        __init__(self, iface) -> CameraList

        Parameters
        ----------
        iface: Spinnaker::CameraList const &

        Spinnaker::CameraList::CameraList(const CameraList &iface)

        Copy constructor
        """

        # For the NanEye FOB, the only valid ports are 0 and/or 1; it can handle
        #   up to two NanEye cams simultaneously.
        self._valid_ports = valid_ports
        self._system = system
        self._cameras = [Camera(valid_ports, empty_buffer, image_buffer, system=self._system)]
        self._iteration_number = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self._iteration_number >= len(self._cameras):
            self._iteration_number = 0
            raise StopIteration
        else:
            camera = self._cameras[self._iteration_number]
            self._iteration_number += 1
            return camera

    def GetSize(self):
        """
        GetSize(self) -> unsigned int

        Parameters
        ----------
        self: Spinnaker::CameraList const *

        int
        Spinnaker::CameraList::GetSize() const

        Returns the size of the camera list. The size is the number of Camera
        objects stored in the list.

        An integer that represents the list size.
        """

        return len(self._valid_ports)

    def GetByIndex(self, index):
        """
        GetByIndex(self, index) -> CameraPtr

        Parameters
        ----------
        index: unsigned int

        CameraPtr
        Spinnaker::CameraList::GetByIndex(int index) const

        Returns a pointer to a camera object at the "index".

        Parameters:
        -----------

        index:  The index at which to retrieve the camera object

        A pointer to an camera object.
        """

        return self._cameras[index]

    def GetBySerial(self, serialNumber):
        """
        GetBySerial(self, serialNumber) -> CameraPtr

        Parameters
        ----------
        serialNumber: std::string

        CameraPtr
        Spinnaker::CameraList::GetBySerial(std::string serialNumber) const

        Returns a pointer to a camera object with the specified serial number.

        Parameters:
        -----------

        serialNumber:  The serial number of the camera object to retrieve

        A pointer to an camera object.
        """

        for camera in self._cameras:
            if camera.Serial == serialNumber:
                return camera
        raise IOError('Camera with serial {s} not found.'.format(s=serialNumber))

    def GetByDeviceID(self, deviceID):
        """
        GetByDeviceID(self, deviceID) -> CameraPtr

        Parameters
        ----------
        deviceID: std::string

        """
        raise NotImplementedError()
        return

    def Clear(self):
        """
        Clear(self)

        Parameters
        ----------
        self: Spinnaker::CameraList *

        void
        Spinnaker::CameraList::Clear()

        Clears the list of cameras and destroys their corresponding reference
        counted objects. This is necessary in order to clean up the parent
        interface. It is important that the camera list is destroyed or is
        cleared before calling system->ReleaseInstance() or else the call to
        system->ReleaseInstance() will result in an error message thrown that
        a reference to the camera is still held.

        See:   System:ReleaseInstance()
        """
        for camera in self._cameras:
            camera.DeInit()
        return

    def RemoveByIndex(self, index):
        """
        RemoveByIndex(self, index)

        Parameters
        ----------
        index: unsigned int

        void
        Spinnaker::CameraList::RemoveByIndex(int index)

        Removes a camera at "index" and destroys its corresponding reference
        counted object.

        Parameters:
        -----------

        index:  The index at which to remove the Camera object
        """

        del self._valid_ports[index]

    def RemoveBySerial(self, serialNumber):
        """
        RemoveBySerial(self, serialNumber)

        Parameters
        ----------
        serialNumber: std::string

        void
        Spinnaker::CameraList::RemoveBySerial(std::string serialNumber)

        Removes a camera using its serial number and destroys its
        corresponding reference counted object.

        Parameters:
        -----------

        serialNumber:  The serial number of the Camera object to remove
        """

        self._valid_ports.remove(serialNumber)

    def RemoveByDeviceID(self, deviceID):
        """
        RemoveByDeviceID(self, deviceID)

        Parameters
        ----------
        deviceID: std::string

        """
        raise NotImplementedError()
        return

    def Append(self, list):
        """
        Append(self, list)

        Parameters
        ----------
        list: Spinnaker::CameraList const &

        void
        Spinnaker::CameraList::Append(CameraList &otherList)

        Appends a camera list to the current list.

        Parameters:
        -----------

        otherList:  The other list to append to this list
        """
        raise NotImplementedError()
        return

class Value:
    def __init__(self, getFcn):
        self.GetValue = getFcn

class Camera:
    """

    The base class for the camera object.

    """

    def __init__(self, valid_ports, empty_buffer, image_buffer, *args, system=None, **kwargs):
        self._system = system
        self._valid_ports = valid_ports

        self._width, self._height = self._system._get_width(), self._system._get_height()       # or read once from first event
        self._BPP = 3               # use 1 for 8-bit raw, 2 for 10/16-bit, 3 for RGB
        self._buflen = self._width*self._height*self._BPP

        self._empty_buffer = empty_buffer
        self._image_buffer = image_buffer

        self.Width =  Value(self.GetFrameWidth)
        self.Height = Value(self.GetFrameHeight) # Note that this will be the height of the stacked frames

        self.Serial = 'NanEye2D'
        self._initialized = False

        self._pending_images = {}
        self._pending_image_IDs = {}
        self._stacked_buf = self._create_stacked_buf()

    def _create_buf(self):
        return np.empty(self._buflen, dtype=np.uint8)

    def _create_stacked_buf(self):
        return np.empty(len(self._valid_ports)*self._buflen, dtype=np.uint8)

    def GetFrameWidth(self):
        """Get the width of the frames the camera acquires.

        Returns:
            int: Width of the camera frames in pixels

        """
        return self._width

    def GetFrameHeight(self):
        """Get the height of the frames the camera acquires.

        Returns:
            int: Height of the camera frames in pixels

        """
        return self._height * len(self._valid_ports) # Frames will be stacked if there are multiple NanEye cams

    def GetFrameDepth(self):
        """Get the pixel bit depth of the frames the camera acquires.

        Returns:
            int: Height of the camera frames in pixels

        """
        return 1

    def GetAttribute(self, attributeName):
        """Get a camera attribute.

        Ideally this would mirror the PySpin nodemap system, but I was lazy.

        Args:
            attributeName (str): An attribute name, corresponding to keys of
                CameraAttributes or AlternateCameraAttributeNames.

        Returns:
            *: Value corresponding to the given attribute name

        """
        # Throw error if camera has not been initialized
        if not self.IsInitialized():
            raise IOError('Camera must be initialized before getting attribute')

        attributeCode = GetAttributeCode(attributeName)

        if attributeCode == 'FRAME_WIDTH':
            return self.GetFrameWidth()
        elif attributeCode == 'FRAME_HEIGHT':
            return self.GetFrameHeight()
        elif attributeCode == 'BIT_DEPTH':
            return self.GetFrameDepth()
        elif attributeCode == 'FPS':
            print('WARNING GIVING DUMMY ACQUISITION FRAME RATE')
            return 45
        elif attributeCode == 'CHANNEL':
            return 3
        else:
            raise NameError('Unknown attribute name: {name}'.format(name=attributeName))

        raise NotImplementedError()

    def SetAttribute(self, attributeName, attributeValue):
        """Set a camera attribute.

        Ideally this would mirror the PySpin nodemap system, but I was lazy.

        Args:
            attributeName (str): An attribute name, corresponding to keys of
                CameraAttributes or AlternateCameraAttributeNames.
            attributeValue (*): Value to set for the given attribute name

        Returns:
            None

        """
        # Throw error if camera has not been initialized
        if not self.IsInitialized():
            raise IOError('Camera must be initialized before setting attribute')

        # Attempt to translate the attributeName into a valid OpenCV VideoCaptureProperty code
        # attributeCode = GetAttributeCode(attributeName)
        #
        # self._camera_pointer.set(attributeCode, attributeValue)
        raise NotImplementedError

    def Init(self):
        """
        Init(self)

        Parameters
        ----------
        self: Spinnaker::CameraBase *

        void
        Spinnaker::CameraBase::Init()

        Connect to camera, retrieve XML and generate node map. This function
        needs to be called before any camera related API calls such as
        BeginAcquisition(), EndAcquisition(), GetNodeMap(), GetNextImage().

        See:   BeginAcquisition()

        See:   EndAcquisition()

        See:   GetNodeMap()

        See:   GetNextImage()
        """

        self._initialized = True

        return

    def DeInit(self):
        """
        DeInit(self)

        Parameters
        ----------
        self: Spinnaker::CameraBase *

        void
        Spinnaker::CameraBase::DeInit()

        Disconnect camera port and free GenICam node map and GUI XML. Do not
        call more functions that access the remote device such as
        WritePort/ReadPort after calling DeInit(); Events should also be
        unregistered before calling camera DeInit(). Otherwise an exception
        will be thrown in the DeInit() call and require the user to unregister
        events before the camera can be re-initialized again.

        See:   Init()

        See:   UnregisterEvent(Event & evtToUnregister)
        """

        # De-allocate image buffers
        with self._empty_buffer.mutex:
            self._empty_buffer.queue.clear()
        with self._image_buffer.mutex:
            self._image_buffer.queue.clear()

        self._initialized = False

    def IsInitialized(self):
        """
        IsInitialized(self) -> bool

        Parameters
        ----------
        self: Spinnaker::CameraBase *

        bool
        Spinnaker::CameraBase::IsInitialized()

        Checks if camera is initialized. This function needs to return true in
        order to retrieve a valid NodeMap from the GetNodeMap() call.

        See:   GetNodeMap()

        If camera is initialized or not
        """

        return self._initialized

    def IsValid(self):
        """
        IsValid(self) -> bool

        Parameters
        ----------
        self: Spinnaker::CameraBase *

        bool
        Spinnaker::CameraBase::IsValid()

        Checks a flag to determine if camera is still valid for use.

        If camera is valid or not

        Note that CameraPtr and CameraBase both define an IsValid() function.
        In order to determine the validity of the camera using a CameraPtr,
        user must first call get() to retrieve the CameraBase object.
        """
        return self._initialized

    def GetNodeMap(self):
        """
        GetNodeMap(self) -> INodeMap

        Parameters
        ----------
        self: Spinnaker::CameraBase const *

        GenApi::INodeMap& Spinnaker::CameraBase::GetNodeMap() const

        Gets a reference to the node map that is generated from a GenICam XML
        file. The camera must be initialized by a call to Init() first before
        a node map reference can be successfully acquired.

        See:   Init()

        A reference to the INodeMap.
        """
        return None

    def GetTLDeviceNodeMap(self):
        """
        GetTLDeviceNodeMap(self) -> INodeMap

        Parameters
        ----------
        self: Spinnaker::CameraBase const *

        GenApi::INodeMap& Spinnaker::CameraBase::GetTLDeviceNodeMap() const

        Gets a reference to the node map that is generated from a GenICam XML
        file for the GenTL Device module. The camera does not need to be
        initialized before acquiring this node map.

        A reference to the INodeMap.
        """
        raise NotImplementedError()
        return

    def GetTLStreamNodeMap(self):
        """
        GetTLStreamNodeMap(self) -> INodeMap

        Parameters
        ----------
        self: Spinnaker::CameraBase const *

        GenApi::INodeMap& Spinnaker::CameraBase::GetTLStreamNodeMap() const

        Gets a reference to the node map that is generated from a GenICam XML
        file for the GenTL Stream module. The camera does not need to be
        initialized before acquiring this node map.

        A reference to the INodeMap.
        """
        raise NotImplementedError()
        return

    def GetAccessMode(self):
        """
        GetAccessMode(self) -> Spinnaker::GenApi::EAccessMode

        Parameters
        ----------
        self: Spinnaker::CameraBase const *

        GenApi::EAccessMode Spinnaker::CameraBase::GetAccessMode() const

        Returns the access mode that the software has on the Camera. The
        camera does not need to be initialized before calling this function.

        See:   Init()

        An enumeration value indicating the access mode
        """
        raise NotImplementedError()
        return

    def BeginAcquisition(self):
        """
        BeginAcquisition(self)

        Parameters
        ----------
        self: Spinnaker::CameraBase *

        void
        Spinnaker::CameraBase::BeginAcquisition()

        Starts the image acquisition engine. The camera must be initialized
        via a call to Init() before starting an acquisition.

        See:   Init()
        """
        if not self._system.IsInUse():

            # Preallocate empty image buffers
            for _ in range(self._empty_buffer.maxsize):
                self._empty_buffer.put(
                    self._create_buf()
                )

            self._system._start_capture()
        return

    def EndAcquisition(self):
        """
        EndAcquisition(self)

        Parameters
        ----------
        self: Spinnaker::CameraBase *

        void
        Spinnaker::CameraBase::EndAcquisition()

        Stops the image acquisition engine. If EndAcquisition() is called
        without a prior call to BeginAcquisition() an error message "Camera
        is not started" will be thrown. All Images that were acquired using
        GetNextImage() need to be released first using image->Release() before
        calling EndAcquisition(). All buffers in the input pool and output
        queue will be discarded when EndAcquisition() is called.

        See:   Init()

        See:   BeginAcquisition()

        See:  GetNextImage( grabTimeout )

        See:  Image::Release()
        """
        if self._system.IsInUse():
            self._system._stop_capture()
        return

    def GetBufferOwnership(self):
        """
        GetBufferOwnership(self) -> Spinnaker::BufferOwnership

        Parameters
        ----------
        self: Spinnaker::CameraBase const *

        """
        raise NotImplementedError()
        return

    def SetBufferOwnership(self, mode):
        """
        SetBufferOwnership(self, mode)

        Parameters
        ----------
        mode: enum Spinnaker::BufferOwnership const

        """
        raise NotImplementedError()
        return

    def GetUserBufferCount(self):
        """
        GetUserBufferCount(self) -> uint64_t

        Parameters
        ----------
        self: Spinnaker::CameraBase const *

        """
        raise NotImplementedError()
        return

    def GetUserBufferSize(self):
        """
        GetUserBufferSize(self) -> uint64_t

        Parameters
        ----------
        self: Spinnaker::CameraBase const *

        """
        raise NotImplementedError()
        return

    def GetUserBufferTotalSize(self):
        """
        GetUserBufferTotalSize(self) -> uint64_t

        Parameters
        ----------
        self: Spinnaker::CameraBase const *

        """
        raise NotImplementedError()
        return

    def SetUserBuffers(self, *args):
        """
        SetUserBuffers(self, pMemBuffers, totalSize)

        Parameters
        ----------
        pMemBuffers: void *const
        totalSize: uint64_t

        SetUserBuffers(self, ppMemBuffers, bufferCount, bufferSize)

        Parameters
        ----------
        ppMemBuffers: void **const
        bufferCount: uint64_t const
        bufferSize: uint64_t const

        """
        raise NotImplementedError()
        return

    def GetNextImage(self, *args):
        """
        GetNextImage(self, grabTimeout, streamID=0) -> ImagePtr

        Parameters
        ----------
        grabTimeout: uint64_t
        streamID: uint64_t

        GetNextImage(self, grabTimeout) -> ImagePtr

        Parameters
        ----------
        grabTimeout: uint64_t

        GetNextImage(self) -> ImagePtr

        Parameters
        ----------
        self: Spinnaker::CameraBase *

        ImagePtr
        Spinnaker::CameraBase::GetNextImage(uint64_t
        grabTimeout=EVENT_TIMEOUT_INFINITE, uint64_t streamID=0)

        Gets the next image that was received by the transport layer. This
        function will block indefinitely until an image arrives. Most cameras
        support one stream so the default streamID is 0 but if a camera
        supports multiple streams the user can input the streamID to select
        from which stream to grab images

        See:   Init()

        See:   BeginAcquisition()

        See:   EndAcquisition()

        Parameters:
        -----------

        grabTimeout:  a 64bit value that represents a timeout in milliseconds

        streamID:  The stream to grab the image.

        pointer to an Image object
        """

        # Check if user wants to specify timeout
        if len(args) > 0:
            timeout = int(args[0]/1000)
        else:
            timeout = None

        # Determine if blocking should happen based on timeout value
        if timeout is None:
            block = False
        else:
            block = True

        while True:
            # Loop until we have a pair of images from the two naneye cams
            #   with matching timestamps and stack them.

            # Get the latest image data
            buf, sensor_id, frame_idx, frame_time = self._image_buffer.get(timeout=timeout, block=block)

            # Check if we already have a stored image with this frame_time
            if frame_time in self._pending_images:
                # Frame time matches stored image - get the image pair container
                image_pair = self._pending_images[frame_time]

                # Check if we've already gotten an image from this sensor with the frame_time
                if image_pair[sensor_id] is not None:
                    raise IOError('Got two naneye images with the same frame_time and sensor_id')
                # Add new image to the frame pair container to complete the pair
                image_pair[sensor_id] = buf

                # Get the image ID pair container
                image_id_pair = self._pending_image_IDs[frame_time]
                # Add the new image ID
                image_id_pair[sensor_id] = frame_idx

                # Check that the two image IDs match
                if image_id_pair[0] != image_id_pair[1]:
                    raise IOError('Matched two naneye images that have the same frame_time but different IDs')

                # Stack the images
                self._stacked_buf[:self._buflen] = image_pair[0]
                self._stacked_buf[self._buflen:] = image_pair[1]

                # Recycle the used buffers
                self._empty_buffer.put(image_pair[0], block=block, timeout=timeout)
                self._empty_buffer.put(image_pair[1], block=block, timeout=timeout)

                # Remove the completed image and image ID pairs from storage
                del image_pair
                del self._pending_images[frame_time]
                del self._pending_image_IDs[frame_time]
                break
            else:
                # This image does not have a pair in storage - start a new pair and store it
                self._pending_images[frame_time] = [None, None]
                self._pending_images[frame_time][sensor_id] = buf
                # Same goes for the image ID
                self._pending_image_IDs[frame_time] = [None, None]
                self._pending_image_IDs[frame_time][sensor_id] = frame_idx

        # Copy the buffer data into an ImagePtr object
        img = ImagePtr(
                self._stacked_buf.copy().reshape(
                    (self.Height.GetValue(), self.Width.GetValue(), 3)
                ),
                frame_id=frame_idx,
                timestamp=frame_time
            )

        # Return the new ImagePtr object
        return img

    def GetUniqueID(self):
        """
        GetUniqueID(self) -> gcstring

        Parameters
        ----------
        self: Spinnaker::CameraBase *

        GenICam::gcstring Spinnaker::CameraBase::GetUniqueID()

        This returns a unique id string that identifies the camera. This is
        the camera serial number.

        string that uniquely identifies the camera (serial number)
        """
        raise NotImplementedError()
        return

    def IsStreaming(self):
        """
        IsStreaming(self) -> bool

        Parameters
        ----------
        self: Spinnaker::CameraBase const *

        bool
        Spinnaker::CameraBase::IsStreaming() const

        Returns true if the camera is currently streaming or false if it is
        not.

        See:   Init()

        returns true if camera is streaming and false otherwise.
        """

        return self._system.IsInUse()

    def GetGuiXml(self):
        """
        GetGuiXml(self) -> gcstring

        Parameters
        ----------
        self: Spinnaker::CameraBase const *

        GenICam::gcstring Spinnaker::CameraBase::GetGuiXml() const

        Returns the GUI XML that can be passed into the Spinnaker GUI
        framework

        GenICam::gcstring that represents the uncompressed GUI XML file
        """
        raise NotImplementedError()
        return

    def RegisterEventHandler(self, *args):
        """
        RegisterEventHandler(self, evtHandlerToRegister)

        Parameters
        ----------
        evtHandlerToRegister: Spinnaker::EventHandler &

        RegisterEventHandler(self, evtHandlerToRegister, eventName)

        Parameters
        ----------
        evtHandlerToRegister: Spinnaker::EventHandler &
        eventName: Spinnaker::GenICam::gcstring const &

        """
        raise NotImplementedError()
        return

    def UnregisterEventHandler(self, evtHandlerToUnregister):
        """
        UnregisterEventHandler(self, evtHandlerToUnregister)

        Parameters
        ----------
        evtHandlerToUnregister: Spinnaker::EventHandler &

        """
        raise NotImplementedError()
        return

    def GetNumImagesInUse(self):
        """
        GetNumImagesInUse(self) -> unsigned int

        Parameters
        ----------
        self: Spinnaker::CameraBase *

        unsigned int Spinnaker::CameraBase::GetNumImagesInUse()

        Returns the number of images that are currently in use. Each of the
        images that are currently in use must be cleaned up with a call to
        image->Release() before calling system->ReleaseInstance().

        The number of images that needs to be cleaned up.
        """
        raise NotImplementedError()
        return

    def GetNumDataStreams(self):
        """
        GetNumDataStreams(self) -> unsigned int

        Parameters
        ----------
        self: Spinnaker::CameraBase *

        unsigned int Spinnaker::CameraBase::GetNumDataStreams()

        Returns the number of streams that a device supports.

        The number of data streams
        """
        raise NotImplementedError()
        return

    def DiscoverMaxPacketSize(self):
        """
        DiscoverMaxPacketSize(self) -> unsigned int

        Parameters
        ----------
        self: Spinnaker::CameraBase *

        unsigned int Spinnaker::CameraBase::DiscoverMaxPacketSize()

        Returns the largest packet size that can be safely used on the
        interface that device is connected to

        The maximum packet size returned.
        """
        raise NotImplementedError()
        return

    def ForceIP(self):
        """
        ForceIP(self)

        Parameters
        ----------
        self: Spinnaker::CameraBase *

        """
        raise NotImplementedError()
        return

class ImagePtr(object):
    """

    The base class of the SystemPtr, CameraPtr, InterfacePtr, ImagePtr and
    LoggingEventDataPtr objects.

    C++ includes: BasePtr.h
    """

    def __init__(self, image_array, frame_id=0, timestamp=0, *args):
        """
        __init__(self) -> _SWIG_ImgPtr
        __init__(self, other) -> _SWIG_ImgPtr

        Parameters
        ----------
        other: Spinnaker::BasePtr< Spinnaker::IImage > const &

        Spinnaker::BasePtr< T, B >::BasePtr(const BasePtr &other)  throw ()
        """
        self._image_array = image_array
        self._frame_id = frame_id
        self._timestamp = timestamp

    def __deref__(self):
        """
        __deref__(self) -> IImage

        Parameters
        ----------
        self: Spinnaker::BasePtr< IImage > const *

        """
        raise NotImplementedError()
        return

    def IsValid(self):
        """
        IsValid(self) -> bool

        Parameters
        ----------
        self: Spinnaker::BasePtr< IImage > const *

        virtual bool
        Spinnaker::BasePtr< T, B >::IsValid() const  throw () true if the
        pointer is valid
        """

        return type(self._image_array) == ndarray

    def __nonzero__(self):
        raise NotImplementedError()
        return

    def __eq__(self, *args):
        """
        __eq__(self, rT) -> bool

        Parameters
        ----------
        rT: Spinnaker::BasePtr< Spinnaker::IImage > const &

        __eq__(self, arg2) -> bool

        Parameters
        ----------
        arg2: std::nullptr_t

        __eq__(self, nMustBeNull) -> bool

        Parameters
        ----------
        nMustBeNull: int

        __eq__(self, nMustBeNull) -> bool

        Parameters
        ----------
        nMustBeNull: long

        """
        raise NotImplementedError()
        return

    def get(self):
        """
        get(self) -> IImage

        Parameters
        ----------
        self: Spinnaker::BasePtr< IImage > const *

        """
        raise NotImplementedError()
        return

    def GetColorProcessing(self):
        """
        GetColorProcessing(self) -> Spinnaker::ColorProcessingAlgorithm

        Parameters
        ----------
        self: Spinnaker::BasePtr< IImage > const *

        """
        raise NotImplementedError()
        return

    def Convert(self, *args):
        """
        Convert(self, format, colorAlgorithm) -> ImagePtr

        Parameters
        ----------
        format: enum Spinnaker::PixelFormatEnums
        colorAlgorithm: enum Spinnaker::ColorProcessingAlgorithm

        Convert(self, format) -> ImagePtr

        Parameters
        ----------
        format: enum Spinnaker::PixelFormatEnums

        Convert(self, destinationImage, format, colorAlgorithm)

        Parameters
        ----------
        destinationImage: Spinnaker::ImagePtr
        format: enum Spinnaker::PixelFormatEnums
        colorAlgorithm: enum Spinnaker::ColorProcessingAlgorithm

        Convert(self, destinationImage, format)

        Parameters
        ----------
        destinationImage: Spinnaker::ImagePtr
        format: enum Spinnaker::PixelFormatEnums

        """
        raise NotImplementedError()
        return

    def ResetImage(self, *args):
        """
        ResetImage(self, width, height, offsetX, offsetY, pixelFormat)

        Parameters
        ----------
        width: size_t
        height: size_t
        offsetX: size_t
        offsetY: size_t
        pixelFormat: enum Spinnaker::PixelFormatEnums

        ResetImage(self, width, height, offsetX, offsetY, pixelFormat, pData)

        Parameters
        ----------
        width: size_t
        height: size_t
        offsetX: size_t
        offsetY: size_t
        pixelFormat: enum Spinnaker::PixelFormatEnums
        pData: void *

        ResetImage(self, width, height, offsetX, offsetY, pixelFormat, pData, dataPayloadType, dataSize)

        Parameters
        ----------
        width: size_t
        height: size_t
        offsetX: size_t
        offsetY: size_t
        pixelFormat: enum Spinnaker::PixelFormatEnums
        pData: void *
        dataPayloadType: enum Spinnaker::PayloadTypeInfoIDs
        dataSize: size_t

        """
        raise NotImplementedError()
        return

    def Release(self):
        """
        Release(self)

        Parameters
        ----------
        self: Spinnaker::BasePtr< IImage > *

        """

        self._image_array = None

    def GetID(self):
        """
        GetID(self) -> uint64_t

        Parameters
        ----------
        self: Spinnaker::BasePtr< IImage > const *

        """
        raise NotImplementedError()
        return

    def GetData(self, *args):
        """
        GetData(self)
        GetData(self) -> PyObject *

        Parameters
        ----------
        self: Spinnaker::BasePtr< IImage > *

        """
        raise NotImplementedError()
        return

    def GetPrivateData(self):
        """
        GetPrivateData(self) -> void *

        Parameters
        ----------
        self: Spinnaker::BasePtr< IImage > const *

        """
        raise NotImplementedError()
        return

    def GetDataAbsoluteMax(self):
        """
        GetDataAbsoluteMax(self) -> float

        Parameters
        ----------
        self: Spinnaker::BasePtr< IImage > const *

        """
        raise NotImplementedError()
        return

    def GetDataAbsoluteMin(self):
        """
        GetDataAbsoluteMin(self) -> float

        Parameters
        ----------
        self: Spinnaker::BasePtr< IImage > const *

        """
        raise NotImplementedError()
        return

    def GetBufferSize(self):
        """
        GetBufferSize(self) -> size_t

        Parameters
        ----------
        self: Spinnaker::BasePtr< IImage > const *

        """
        raise NotImplementedError()
        return

    def DeepCopy(self, pSrcImage):
        """
        DeepCopy(self, pSrcImage)

        Parameters
        ----------
        pSrcImage: Spinnaker::ImagePtr const

        """

        return ImagePtr(self._image_array.copy())

    def GetWidth(self):
        """
        GetWidth(self) -> size_t

        Parameters
        ----------
        self: Spinnaker::BasePtr< IImage > const *

        """

        return self._image_array.shape[1]

    def GetHeight(self):
        """
        GetHeight(self) -> size_t

        Parameters
        ----------
        self: Spinnaker::BasePtr< IImage > const *

        """

        return self._image_array.shape[0]

    def GetStride(self):
        """
        GetStride(self) -> size_t

        Parameters
        ----------
        self: Spinnaker::BasePtr< IImage > const *

        """
        raise NotImplementedError()
        return

    def GetBitsPerPixel(self):
        """
        GetBitsPerPixel(self) -> size_t

        Parameters
        ----------
        self: Spinnaker::BasePtr< IImage > const *

        """

        bitsPerChannel = self._image_array.dtype.itemsize * 8
        numChannels = self.GetNumChannels

        return bitsPerChannel * numChannels

    def GetNumChannels(self):
        """
        GetNumChannels(self) -> size_t

        Parameters
        ----------
        self: Spinnaker::BasePtr< IImage > const *

        """

        if len(self._image_array.shape) == 2:
            return 1
        else:
            return self._image_array.shape[3]

    def GetXOffset(self):
        """
        GetXOffset(self) -> size_t

        Parameters
        ----------
        self: Spinnaker::BasePtr< IImage > const *

        """
        raise NotImplementedError()
        return

    def GetYOffset(self):
        """
        GetYOffset(self) -> size_t

        Parameters
        ----------
        self: Spinnaker::BasePtr< IImage > const *

        """
        raise NotImplementedError()
        return

    def GetXPadding(self):
        """
        GetXPadding(self) -> size_t

        Parameters
        ----------
        self: Spinnaker::BasePtr< IImage > const *

        """
        raise NotImplementedError()
        return

    def GetYPadding(self):
        """
        GetYPadding(self) -> size_t

        Parameters
        ----------
        self: Spinnaker::BasePtr< IImage > const *

        """
        raise NotImplementedError()
        return

    def GetFrameID(self):
        """
        GetFrameID(self) -> uint64_t

        Parameters
        ----------
        self: Spinnaker::BasePtr< IImage > const *

        """

        return self._frame_id

    def GetPayloadType(self):
        """
        GetPayloadType(self) -> size_t

        Parameters
        ----------
        self: Spinnaker::BasePtr< IImage > const *

        """
        raise NotImplementedError()
        return

    def GetTLPayloadType(self):
        """
        GetTLPayloadType(self) -> Spinnaker::PayloadTypeInfoIDs

        Parameters
        ----------
        self: Spinnaker::BasePtr< IImage > const *

        """
        raise NotImplementedError()
        return

    def GetTLPixelFormat(self):
        """
        GetTLPixelFormat(self) -> uint64_t

        Parameters
        ----------
        self: Spinnaker::BasePtr< IImage > const *

        """
        raise NotImplementedError()
        return

    def GetTLPixelFormatNamespace(self):
        """
        GetTLPixelFormatNamespace(self) -> Spinnaker::PixelFormatNamespaceID

        Parameters
        ----------
        self: Spinnaker::BasePtr< IImage > const *

        """
        raise NotImplementedError()
        return

    def GetPixelFormatName(self):
        """
        GetPixelFormatName(self) -> gcstring

        Parameters
        ----------
        self: Spinnaker::BasePtr< IImage > const *

        """
        raise NotImplementedError()
        return

    def GetPixelFormat(self):
        """
        GetPixelFormat(self) -> Spinnaker::PixelFormatEnums

        Parameters
        ----------
        self: Spinnaker::BasePtr< IImage > const *

        """
        raise NotImplementedError()
        return

    def GetPixelFormatIntType(self):
        """
        GetPixelFormatIntType(self) -> Spinnaker::PixelFormatIntType

        Parameters
        ----------
        self: Spinnaker::BasePtr< IImage > const *

        """
        raise NotImplementedError()
        return

    def IsIncomplete(self):
        """
        IsIncomplete(self) -> bool

        Parameters
        ----------
        self: Spinnaker::BasePtr< IImage > const *

        """

        return type(self._image_array) != np.ndarray

    def GetValidPayloadSize(self):
        """
        GetValidPayloadSize(self) -> size_t

        Parameters
        ----------
        self: Spinnaker::BasePtr< IImage > const *

        """
        raise NotImplementedError()
        return

    def GetChunkLayoutId(self):
        """
        GetChunkLayoutId(self) -> uint64_t

        Parameters
        ----------
        self: Spinnaker::BasePtr< IImage > const *

        """
        raise NotImplementedError()
        return

    def GetTimeStamp(self):
        """
        GetTimeStamp(self) -> uint64_t

        Parameters
        ----------
        self: Spinnaker::BasePtr< IImage > const *

        """

        return self._timestamp

    def Save(self, filename, *args):
        """
        Save(self, pFilename, format)

        Parameters
        ----------
        pFilename: char const *
        format: enum Spinnaker::ImageFileFormat

        Save(self, pFilename)

        Parameters
        ----------
        pFilename: char const *

        Save(self, pFilename, pOption)

        Parameters
        ----------
        pFilename: char const *
        pOption: Spinnaker::PNGOption &

        Save(self, pFilename, pOption)

        Parameters
        ----------
        pFilename: char const *
        pOption: Spinnaker::PPMOption &

        Save(self, pFilename, pOption)

        Parameters
        ----------
        pFilename: char const *
        pOption: Spinnaker::PGMOption &

        Save(self, pFilename, pOption)

        Parameters
        ----------
        pFilename: char const *
        pOption: Spinnaker::TIFFOption &

        Save(self, pFilename, pOption)

        Parameters
        ----------
        pFilename: char const *
        pOption: Spinnaker::JPEGOption &

        Save(self, pFilename, pOption)

        Parameters
        ----------
        pFilename: char const *
        pOption: Spinnaker::JPG2Option &

        Save(self, pFilename, pOption)

        Parameters
        ----------
        pFilename: char const *
        pOption: Spinnaker::BMPOption &

        """

        Image.fromarray(self._image_array).save(filename)

    def GetChunkData(self):
        """
        GetChunkData(self) -> ChunkData

        Parameters
        ----------
        self: Spinnaker::BasePtr< IImage > const *

        """
        raise NotImplementedError()
        return

    def CalculateStatistics(self, pStatistics):
        """
        CalculateStatistics(self, pStatistics)

        Parameters
        ----------
        pStatistics: Spinnaker::ImageStatistics &

        """
        raise NotImplementedError()
        return

    def HasCRC(self):
        """
        HasCRC(self) -> bool

        Parameters
        ----------
        self: Spinnaker::BasePtr< IImage > const *

        """
        raise NotImplementedError()
        return

    def CheckCRC(self):
        """
        CheckCRC(self) -> bool

        Parameters
        ----------
        self: Spinnaker::BasePtr< IImage > const *

        """
        raise NotImplementedError()
        return

    def GetImageSize(self):
        """
        GetImageSize(self) -> size_t

        Parameters
        ----------
        self: Spinnaker::BasePtr< IImage > const *

        """
        raise NotImplementedError()
        return

    def IsInUse(self):
        """
        IsInUse(self) -> bool

        Parameters
        ----------
        self: Spinnaker::BasePtr< IImage > *

        """
        raise NotImplementedError()
        return

    def GetImageStatus(self):
        """
        GetImageStatus(self) -> Spinnaker::ImageStatus

        Parameters
        ----------
        self: Spinnaker::BasePtr< IImage > const *

        """
        return None

    def IsCompressed(self):
        """
        IsCompressed(self) -> bool

        Parameters
        ----------
        self: Spinnaker::BasePtr< IImage > const *

        """
        raise NotImplementedError()
        return

    def CalculateChannelStatistics(self, channel):
        """
        CalculateChannelStatistics(self, channel) -> ChannelStatistics

        Parameters
        ----------
        channel: enum Spinnaker::StatisticsChannel

        """
        raise NotImplementedError()
        return

    def GetDefaultColorProcessing(self):
        """
        GetDefaultColorProcessing(self) -> Spinnaker::ColorProcessingAlgorithm

        Parameters
        ----------
        self: Spinnaker::BasePtr< IImage > *

        """
        raise NotImplementedError()
        return

    def SetDefaultColorProcessing(self, defaultMethod):
        """
        SetDefaultColorProcessing(self, defaultMethod)

        Parameters
        ----------
        defaultMethod: enum Spinnaker::ColorProcessingAlgorithm

        """
        raise NotImplementedError()
        return

    def GetNDArray(self):
        """
        GetNDArray(self) -> PyObject *

        Parameters
        ----------
        self: Spinnaker::BasePtr< IImage > *

        """
        return self._image_array


if __name__ == "__main__":
    s = System.GetInstance()
    cs = s.GetCameras()


#
# PixelFormat_Mono8 = cv2.
# PixelFormat_Mono16 = cv2.
# PixelFormat_RGB8Packed = cv2.
# PixelFormat_BayerGR8 = cv2.
# PixelFormat_BayerRG8 = cv2.
# PixelFormat_BayerGB8 = cv2.
# PixelFormat_BayerBG8 = cv2.
# PixelFormat_BayerGR16 = cv2.
# PixelFormat_BayerRG16 = cv2.
# PixelFormat_BayerGB16 = cv2.
# PixelFormat_BayerBG16 = cv2.
# PixelFormat_Mono12Packed = cv2.
# PixelFormat_BayerGR12Packed = cv2.
# PixelFormat_BayerRG12Packed = cv2.
# PixelFormat_BayerGB12Packed = cv2.
# PixelFormat_BayerBG12Packed = cv2.
# PixelFormat_YUV411Packed = cv2.
# PixelFormat_YUV422Packed = cv2.
# PixelFormat_YUV444Packed = cv2.
# PixelFormat_Mono12p = cv2.
# PixelFormat_BayerGR12p = cv2.
# PixelFormat_BayerRG12p = cv2.
# PixelFormat_BayerGB12p = cv2.
# PixelFormat_BayerBG12p = cv2.
# PixelFormat_YCbCr8 = cv2.
# PixelFormat_YCbCr422_8 = cv2.
# PixelFormat_YCbCr411_8 = cv2.
# PixelFormat_BGR8 = cv2.
# PixelFormat_BGRa8 = cv2.
# PixelFormat_Mono10Packed = cv2.
# PixelFormat_BayerGR10Packed = cv2.
# PixelFormat_BayerRG10Packed = cv2.
# PixelFormat_BayerGB10Packed = cv2.
# PixelFormat_BayerBG10Packed = cv2.
# PixelFormat_Mono10p = cv2.
# PixelFormat_BayerGR10p = cv2.
# PixelFormat_BayerRG10p = cv2.
# PixelFormat_BayerGB10p = cv2.
# PixelFormat_BayerBG10p = cv2.
# PixelFormat_Mono1p = cv2.
# PixelFormat_Mono2p = cv2.
# PixelFormat_Mono4p = cv2.
# PixelFormat_Mono8s = cv2.
# PixelFormat_Mono10 = cv2.
# PixelFormat_Mono12 = cv2.
# PixelFormat_Mono14 = cv2.
# PixelFormat_Mono16s = cv2.
# PixelFormat_Mono32f = cv2.
# PixelFormat_BayerBG10 = cv2.
# PixelFormat_BayerBG12 = cv2.
# PixelFormat_BayerGB10 = cv2.
# PixelFormat_BayerGB12 = cv2.
# PixelFormat_BayerGR10 = cv2.
# PixelFormat_BayerGR12 = cv2.
# PixelFormat_BayerRG10 = cv2.
# PixelFormat_BayerRG12 = cv2.
# PixelFormat_RGBa8 = cv2.
# PixelFormat_RGBa10 = cv2.
# PixelFormat_RGBa10p = cv2.
# PixelFormat_RGBa12 = cv2.
# PixelFormat_RGBa12p = cv2.
# PixelFormat_RGBa14 = cv2.
# PixelFormat_RGBa16 = cv2.
# PixelFormat_RGB8 = cv2.
# PixelFormat_RGB8_Planar = cv2.
# PixelFormat_RGB10 = cv2.
# PixelFormat_RGB10_Planar = cv2.
# PixelFormat_RGB10p = cv2.
# PixelFormat_RGB10p32 = cv2.
# PixelFormat_RGB12 = cv2.
# PixelFormat_RGB12_Planar = cv2.
# PixelFormat_RGB12p = cv2.
# PixelFormat_RGB14 = cv2.
# PixelFormat_RGB16 = cv2.
# PixelFormat_RGB16s = cv2.
# PixelFormat_RGB32f = cv2.
# PixelFormat_RGB16_Planar = cv2.
# PixelFormat_RGB565p = cv2.
# PixelFormat_BGRa10 = cv2.
# PixelFormat_BGRa10p = cv2.
# PixelFormat_BGRa12 = cv2.
# PixelFormat_BGRa12p = cv2.
# PixelFormat_BGRa14 = cv2.
# PixelFormat_BGRa16 = cv2.
# PixelFormat_RGBa32f = cv2.
# PixelFormat_BGR10 = cv2.
# PixelFormat_BGR10p = cv2.
# PixelFormat_BGR12 = cv2.
# PixelFormat_BGR12p = cv2.
# PixelFormat_BGR14 = cv2.
# PixelFormat_BGR16 = cv2.
# PixelFormat_BGR565p = cv2.
# PixelFormat_R8 = cv2.
# PixelFormat_R10 = cv2.
# PixelFormat_R12 = cv2.
# PixelFormat_R16 = cv2.
# PixelFormat_G8 = cv2.
# PixelFormat_G10 = cv2.
# PixelFormat_G12 = cv2.
# PixelFormat_G16 = cv2.
# PixelFormat_B8 = cv2.
# PixelFormat_B10 = cv2.
# PixelFormat_B12 = cv2.
# PixelFormat_B16 = cv2.
# PixelFormat_Coord3D_ABC8 = cv2.
# PixelFormat_Coord3D_ABC8_Planar = cv2.
# PixelFormat_Coord3D_ABC10p = cv2.
# PixelFormat_Coord3D_ABC10p_Planar = cv2.
# PixelFormat_Coord3D_ABC12p = cv2.
# PixelFormat_Coord3D_ABC12p_Planar = cv2.
# PixelFormat_Coord3D_ABC16 = cv2.
# PixelFormat_Coord3D_ABC16_Planar = cv2.
# PixelFormat_Coord3D_ABC32f = cv2.
# PixelFormat_Coord3D_ABC32f_Planar = cv2.
# PixelFormat_Coord3D_AC8 = cv2.
# PixelFormat_Coord3D_AC8_Planar = cv2.
# PixelFormat_Coord3D_AC10p = cv2.
# PixelFormat_Coord3D_AC10p_Planar = cv2.
# PixelFormat_Coord3D_AC12p = cv2.
# PixelFormat_Coord3D_AC12p_Planar = cv2.
# PixelFormat_Coord3D_AC16 = cv2.
# PixelFormat_Coord3D_AC16_Planar = cv2.
# PixelFormat_Coord3D_AC32f = cv2.
# PixelFormat_Coord3D_AC32f_Planar = cv2.
# PixelFormat_Coord3D_A8 = cv2.
# PixelFormat_Coord3D_A10p = cv2.
# PixelFormat_Coord3D_A12p = cv2.
# PixelFormat_Coord3D_A16 = cv2.
# PixelFormat_Coord3D_A32f = cv2.
# PixelFormat_Coord3D_B8 = cv2.
# PixelFormat_Coord3D_B10p = cv2.
# PixelFormat_Coord3D_B12p = cv2.
# PixelFormat_Coord3D_B16 = cv2.
# PixelFormat_Coord3D_B32f = cv2.
# PixelFormat_Coord3D_C8 = cv2.
# PixelFormat_Coord3D_C10p = cv2.
# PixelFormat_Coord3D_C12p = cv2.
# PixelFormat_Coord3D_C16 = cv2.
# PixelFormat_Coord3D_C32f = cv2.
# PixelFormat_Confidence1 = cv2.
# PixelFormat_Confidence1p = cv2.
# PixelFormat_Confidence8 = cv2.
# PixelFormat_Confidence16 = cv2.
# PixelFormat_Confidence32f = cv2.
# PixelFormat_BiColorBGRG8 = cv2.
# PixelFormat_BiColorBGRG10 = cv2.
# PixelFormat_BiColorBGRG10p = cv2.
# PixelFormat_BiColorBGRG12 = cv2.
# PixelFormat_BiColorBGRG12p = cv2.
# PixelFormat_BiColorRGBG8 = cv2.
# PixelFormat_BiColorRGBG10 = cv2.
# PixelFormat_BiColorRGBG10p = cv2.
# PixelFormat_BiColorRGBG12 = cv2.
# PixelFormat_BiColorRGBG12p = cv2.
# PixelFormat_SCF1WBWG8 = cv2.
# PixelFormat_SCF1WBWG10 = cv2.
# PixelFormat_SCF1WBWG10p = cv2.
# PixelFormat_SCF1WBWG12 = cv2.
# PixelFormat_SCF1WBWG12p = cv2.
# PixelFormat_SCF1WBWG14 = cv2.
# PixelFormat_SCF1WBWG16 = cv2.
# PixelFormat_SCF1WGWB8 = cv2.
# PixelFormat_SCF1WGWB10 = cv2.
# PixelFormat_SCF1WGWB10p = cv2.
# PixelFormat_SCF1WGWB12 = cv2.
# PixelFormat_SCF1WGWB12p = cv2.
# PixelFormat_SCF1WGWB14 = cv2.
# PixelFormat_SCF1WGWB16 = cv2.
# PixelFormat_SCF1WGWR8 = cv2.
# PixelFormat_SCF1WGWR10 = cv2.
# PixelFormat_SCF1WGWR10p = cv2.
# PixelFormat_SCF1WGWR12 = cv2.
# PixelFormat_SCF1WGWR12p = cv2.
# PixelFormat_SCF1WGWR14 = cv2.
# PixelFormat_SCF1WGWR16 = cv2.
# PixelFormat_SCF1WRWG8 = cv2.
# PixelFormat_SCF1WRWG10 = cv2.
# PixelFormat_SCF1WRWG10p = cv2.
# PixelFormat_SCF1WRWG12 = cv2.
# PixelFormat_SCF1WRWG12p = cv2.
# PixelFormat_SCF1WRWG14 = cv2.
# PixelFormat_SCF1WRWG16 = cv2.
# PixelFormat_YCbCr8_CbYCr = cv2.
# PixelFormat_YCbCr10_CbYCr = cv2.
# PixelFormat_YCbCr10p_CbYCr = cv2.
# PixelFormat_YCbCr12_CbYCr = cv2.
# PixelFormat_YCbCr12p_CbYCr = cv2.
# PixelFormat_YCbCr411_8_CbYYCrYY = cv2.
# PixelFormat_YCbCr422_8_CbYCrY = cv2.
# PixelFormat_YCbCr422_10 = cv2.
# PixelFormat_YCbCr422_10_CbYCrY = cv2.
# PixelFormat_YCbCr422_10p = cv2.
# PixelFormat_YCbCr422_10p_CbYCrY = cv2.
# PixelFormat_YCbCr422_12 = cv2.
# PixelFormat_YCbCr422_12_CbYCrY = cv2.
# PixelFormat_YCbCr422_12p = cv2.
# PixelFormat_YCbCr422_12p_CbYCrY = cv2.
# PixelFormat_YCbCr601_8_CbYCr = cv2.
# PixelFormat_YCbCr601_10_CbYCr = cv2.
# PixelFormat_YCbCr601_10p_CbYCr = cv2.
# PixelFormat_YCbCr601_12_CbYCr = cv2.
# PixelFormat_YCbCr601_12p_CbYCr = cv2.
# PixelFormat_YCbCr601_411_8_CbYYCrYY = cv2.
# PixelFormat_YCbCr601_422_8 = cv2.
# PixelFormat_YCbCr601_422_8_CbYCrY = cv2.
# PixelFormat_YCbCr601_422_10 = cv2.
# PixelFormat_YCbCr601_422_10_CbYCrY = cv2.
# PixelFormat_YCbCr601_422_10p = cv2.
# PixelFormat_YCbCr601_422_10p_CbYCrY = cv2.
# PixelFormat_YCbCr601_422_12 = cv2.
# PixelFormat_YCbCr601_422_12_CbYCrY = cv2.
# PixelFormat_YCbCr601_422_12p = cv2.
# PixelFormat_YCbCr601_422_12p_CbYCrY = cv2.
# PixelFormat_YCbCr709_8_CbYCr = cv2.
# PixelFormat_YCbCr709_10_CbYCr = cv2.
# PixelFormat_YCbCr709_10p_CbYCr = cv2.
# PixelFormat_YCbCr709_12_CbYCr = cv2.
# PixelFormat_YCbCr709_12p_CbYCr = cv2.
# PixelFormat_YCbCr709_411_8_CbYYCrYY = cv2.
# PixelFormat_YCbCr709_422_8 = cv2.
# PixelFormat_YCbCr709_422_8_CbYCrY = cv2.
# PixelFormat_YCbCr709_422_10 = cv2.
# PixelFormat_YCbCr709_422_10_CbYCrY = cv2.
# PixelFormat_YCbCr709_422_10p = cv2.
# PixelFormat_YCbCr709_422_10p_CbYCrY = cv2.
# PixelFormat_YCbCr709_422_12 = cv2.
# PixelFormat_YCbCr709_422_12_CbYCrY = cv2.
# PixelFormat_YCbCr709_422_12p = cv2.
# PixelFormat_YCbCr709_422_12p_CbYCrY = cv2.
# PixelFormat_YUV8_UYV = cv2.
# PixelFormat_YUV411_8_UYYVYY = cv2.
# PixelFormat_YUV422_8 = cv2.
# PixelFormat_YUV422_8_UYVY = cv2.
# PixelFormat_Polarized8 = cv2.
# PixelFormat_Polarized10p = cv2.
# PixelFormat_Polarized12p = cv2.
# PixelFormat_Polarized16 = cv2.
# PixelFormat_BayerRGPolarized8 = cv2.
# PixelFormat_BayerRGPolarized10p = cv2.
# PixelFormat_BayerRGPolarized12p = cv2.
# PixelFormat_BayerRGPolarized16 = cv2.
# PixelFormat_LLCMono8 = cv2.
# PixelFormat_LLCBayerRG8 = cv2.
# PixelFormat_JPEGMono8 = cv2.
# PixelFormat_JPEGColor8 = cv2.
# PixelFormat_Raw16 = cv2.
# PixelFormat_Raw8 = cv2.
# PixelFormat_R12_Jpeg = cv2.
# PixelFormat_GR12_Jpeg = cv2.
# PixelFormat_GB12_Jpeg = cv2.
# PixelFormat_B12_Jpeg = cv2.
