import ctypes
from ctypes import c_char_p, c_void_p, c_uint32, c_int, c_size_t, POINTER, byref, create_string_buffer, c_char_p, c_ubyte
import numpy as np
import cv2
import time
from pathlib import Path

# Load the apbase DLL from Aptina (onsemi)
p = Path('C:\\')
aptinaRoots = list(p.glob('*Aptina*'))
if len(aptinaRoots) == 0:
    raise OSError('Could not find aptina.dll - please make sure Aptina API is installed in the default location.')
aptinaRoot = aptinaRoots[0]
dllPath = aptinaRoot / 'bin' / 'apbase.dll'
if not dllPath.exists():
    raise OSError('Could not find aptina.dll - please make sure Aptina API is installed in the default location.')
apbase_dll = ctypes.WinDLL(str(dllPath))

# Define some return codes and constants if not already defined
MI_CAMERA_SUCCESS = 0
MI_INI_SUCCESS = 256

# Define custom ctypes types
ap_handle = c_void_p
ap_s32 = c_int
ap_u32 = c_uint32

# Set up argument and return types for the functions we need:
apbase_dll.ap_GetFrameData.argtypes = [c_void_p, c_char_p]
apbase_dll.ap_GetFrameData.restype  = c_int

apbase_dll.ap_Create.argtypes = [c_int]
apbase_dll.ap_Create.restype = ap_handle

apbase_dll.ap_Destroy.argtypes = [ap_handle]
apbase_dll.ap_Destroy.restype = None

apbase_dll.ap_Finalize.argtypes = []
apbase_dll.ap_Finalize.restype = None

apbase_dll.ap_LoadIniPreset.argtypes = [ap_handle, c_char_p, c_char_p]
apbase_dll.ap_LoadIniPreset.restype = ap_s32

apbase_dll.ap_CheckSensorState.argtypes = [ap_handle, c_uint32]
apbase_dll.ap_CheckSensorState.restype = ap_s32

apbase_dll.ap_GrabFrame.argtypes = [ap_handle, c_char_p, ap_u32]
apbase_dll.ap_GrabFrame.restype = ap_u32

apbase_dll.ap_ColorPipe.argtypes = [ap_handle, c_char_p, ap_u32, POINTER(ap_u32), POINTER(ap_u32), POINTER(ap_u32)]
apbase_dll.ap_ColorPipe.restype = POINTER(c_ubyte)  # returns unsigned char *

apbase_dll.ap_GetLastError.argtypes = []
apbase_dll.ap_GetLastError.restype = ap_s32

apbase_dll.ap_DeviceProbe.argtypes = [c_char_p]
apbase_dll.ap_DeviceProbe.restype = ap_s32

apbase_dll.ap_NumCameras.argtypes = []
apbase_dll.ap_NumCameras.restype = c_int

# Now translate the provided C code into Python:
device_dir = c_char_p(r'C:\Aptina Imaging\sensor_data\MT9V024-REV4.xsdat'.encode('utf-8'))
err = apbase_dll.ap_DeviceProbe(device_dir)

numCameras = apbase_dll.ap_NumCameras()
print('Num cameras:', numCameras)

# Create a device handle for the first camera
camera_handle = apbase_dll.ap_Create(0)
if not camera_handle:
    print("Failed to create device handle.")
    apbase_dll.ap_Finalize()
    raise SystemExit(1)

# Load the default INI preset (NULL, NULL)
ini = c_char_p(r'D:\Dropbox\Documents\Work\Cornell Lab Tech\Projects\Budgies\Pupillometry\PupilCam\Kerr Lab Eye Camera\[Docs] Dual-eye cam\MT9V024-REV4.ini'.encode('utf-8'));

# err = apbase_dll.ap_LoadIniPreset(camera_handle, ini, b'EyeCAM-Triggered')
err = apbase_dll.ap_LoadIniPreset(camera_handle, ini, b'EyeCAM')
if err != MI_INI_SUCCESS:
    print("Failed to load default INI preset. Error code:", err)

# Re-read sensor state
err = apbase_dll.ap_CheckSensorState(camera_handle, 0)
if err != MI_CAMERA_SUCCESS:
    print("Failed to check sensor state. Error code:", err)
key = -1
dts = []
startTime = None
stopTime = None
historyLength = 10

# First, call ap_GrabFrame with NULL to get the required buffer size
buf_size = apbase_dll.ap_GrabFrame(camera_handle, None, 0)
if buf_size == 0:
    last_err = apbase_dll.ap_GetLastError()
    print("Failed to get buffer size. Error code:", last_err)
    apbase_dll.ap_Destroy(camera_handle)
    apbase_dll.ap_Finalize()
    raise SystemExit(1)

# Allocate a buffer of the required size
pBuffer = create_string_buffer(buf_size)

# Prepare variables for ap_ColorPipe
rgbWidth = ap_u32(0)
rgbHeight = ap_u32(0)
rgbBitDepth = ap_u32(0)

while key == -1:
    startTime = time.time()

    # Now grab the frame into pBuffer
    bytes_returned = apbase_dll.ap_GrabFrame(camera_handle, pBuffer, buf_size)
    if bytes_returned == 0:
        last_err = apbase_dll.ap_GetLastError()
        print("Failed to grab frame. Error code:", last_err)
        apbase_dll.ap_Destroy(camera_handle)
        apbase_dll.ap_Finalize()
        raise SystemExit(1)

    frame_idx = apbase_dll.ap_GetFrameData(camera_handle, b"frameNumber")
    print("Board frame counter:", frame_idx)

    # Only call ap_ColorPipe if we had success grabbing the frame
    if apbase_dll.ap_GetLastError() == MI_CAMERA_SUCCESS:
        pRGB = apbase_dll.ap_ColorPipe(camera_handle,
                                   pBuffer,               # Input image
                                   buf_size,
                                   byref(rgbWidth),
                                   byref(rgbHeight),
                                   byref(rgbBitDepth))
        if pRGB:
            # pRGB points to an internal buffer managed by apbase_dll.
            # If needed, copy it out now because subsequent calls may overwrite it.
            width = rgbWidth.value
            height = rgbHeight.value
            depth = rgbBitDepth.value

            # Calculate how many bytes in the returned image?
            # For example, if depth=32 bits per pixel, and image size is width*height
            # total bytes = width * height * (depth/8)
            # In a real scenario, you might copy out the data from pRGB into another buffer.
            # print(f"Got an RGB image: {width}x{height}, depth={depth} bits per pixel")
        else:
            print("ap_ColorPipe returned NULL.")
    else:
        print("Error occurred before calling ap_ColorPipe.")
    total_bytes = width * height * depth // 8
    c_array_type = c_ubyte * total_bytes
    c_array = ctypes.cast(pRGB, POINTER(c_ubyte * total_bytes)).contents
    image = np.array(c_array).reshape([height, width, 4])
    stopTime = time.time()
    dts.append(stopTime - startTime)
    if len(dts) > historyLength:
        dts.pop(0)
    print('Framerate:', 1/np.mean(dts))
    cv2.imshow('screen', image[:, :, :3]);
    key = cv2.waitKey(10)
    # print('key = ', key)

# Cleanup
apbase_dll.ap_Destroy(camera_handle)
apbase_dll.ap_Finalize()


# closing all open windows
cv2.destroyAllWindows()
