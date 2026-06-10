# -*- coding: utf-8 -*-
# Minimal NanEye2D (FOB 2.0) frame grabber using pythonnet + OpenCV
# Press ESC to quit.

import numpy as np, cv2, threading, time, sys, os
from pathlib import Path
from collections import deque
import clr
from System import IntPtr
from System.Runtime.InteropServices import Marshal
from System.Collections.Generic import List

root = Path(__file__).resolve().parent.parent.parent
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

# Create provider (same class as in your C# Form1)
provider = NanEyeFobProvider()

# Point to firmware/FPGA (same as in C#)
provider.SetFWFile(str(FW_IMG))
provider.SetFpgaFile(str(FPGA_BIN))

# Enable only Sensor 1 (index 0 = True, index 1 = False), like your C#
sensors = List[bool]()
sensors.Add(True)   # Sensor 1
sensors.Add(False)  # Sensor 2
provider.Sensors = sensors

print('Created provider:')
print('\t', provider.Width, 'x', provider.Height)
print('\t', provider.GetFrameRate(0), 'fps')
for prop in dir(provider):
    try:
        print('\t'+prop+':', getattr(provider, prop))
    except:
        print('\t'+prop+':', '<not readable>')

print('provider props:')
for prop in dir(provider):
    try:
        print(prop + ':', getattr(provider, prop))
    except:
        print(prop + ':', '<not readable>')

# (Optional) If you want to adjust auto-exposure etc., that’s available too,
# but we’ll keep the PoC minimal.

w, h = 250, 250       # or read once from first event
BPP = 3               # use 1 for 8-bit raw, 2 for 10/16-bit, 3 for RGB
buflen = w*h*BPP
RING = 10              # number of prealloc slots

# Preallocate ring of numpy buffers
ring = [np.empty(buflen, dtype=np.uint8) for _ in range(RING)]
q = deque(maxlen=RING)
qlock = threading.Lock()
lastFrameCount = None
lastTimeStamp = 0

timetampDeque = deque(maxlen=10)

def on_image(sender, e):
    global lastFrameCount, lastTimeStamp
    # FAST path: copy raw bytes, enqueue index, return
    try:
        # choose one raw source:
        # src = e.GetImageData.GetRawPixels1Byte  # fast 8-bit
        # src = e.GetImageData.GetRawPixels2Byte # 10-bit expanded (2 bytes/px)
        src = e.PixelData                      # RGB processed (3 bytes/px)
        frameCount = e.FrameCount
        if lastFrameCount is not None and frameCount != lastFrameCount + 1:
            print('Dropped frame! {k} => {j}'.format(k=lastFrameCount, j=frameCount))
        lastFrame = frameCount

        timeStamp = e.TimeStamp / 1000000.0

        timetampDeque.append(timeStamp - lastTimeStamp)
        if len(timetampDeque) > 0:
            fps = sum([1/t for t in timetampDeque]) / len(timetampDeque)

        print('fps: ', fps)
        lastTimeStamp = timeStamp

        with qlock:
            # overwrite the oldest slot (bounded queue)
            slot = ring[len(q) % RING]
        Marshal.Copy(src, 0, IntPtr(slot.ctypes.data), src.Length)
        with qlock:
            q.append((slot, int(e.Width), int(e.Height)))
    except Exception as ex:
        print(ex)
        pass  # keep it lean; log elsewhere

def ui_loop():
    while True:
        item = None
        with qlock:
            if q:
                item = q.pop()
                q.clear()  # drop older ones
        if item:
            buf, W, H = item
            if BPP == 1:
                frame = buf.reshape(H, W)
                cv2.imshow("NanEye RAW", frame)
            elif BPP == 2:
                # visualize 10/16-bit as 8-bit for display
                frame16 = buf.view(np.uint16).reshape(H, W)
                cv2.imshow("NanEye RAW", (frame16 >> 2).astype(np.uint8))
            else:
                frame = buf.reshape(H, W, 3)  # RGB
                cv2.imshow("NanEye RGB", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        if (cv2.waitKey(1) & 0xFF) == 27:
            print('cv waitkey stopping')
            break
        time.sleep(0.001)

provider.ImageProcessed += on_image  # (same event as C#) :contentReference[oaicite:13]{index=13}
print('pre  capturing:', provider.IsCapturing)
provider.StartCapture()              # :contentReference[oaicite:14]{index=14}
print('post capturing:', provider.IsCapturing)
ui_loop()
provider.StopCapture()


    # Available image info:
    # e.GetImageData = Awaiba.FrameProcessing.ImageData
    # e.ProcessingTime = 2197.0
    # e.FramesTime = 2598132480.0
    # e.BoardId = 0
    # e.FrameCount = 209
    # e.Width = 250
    # e.Height = 250
    # e.BitsPerPixel = 10
    # e.TimeStamp = 2598132507
    # e.PixelData = System.Byte[]
    # e.SensorID = 0
    # e.GetImageData.GetProcessedDataRGBByte = System.Byte[]
    # e.GetImageData.GetProcessedDataARGBByte = System.Byte[]
    # e.GetImageData.GetRawPixelsUShort = System.UInt16[]
    # e.GetImageData.GetRawPixels2Byte = System.Byte[]
    # e.GetImageData.GetRawPixels1Byte = System.Byte[]
