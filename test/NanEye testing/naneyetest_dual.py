# -*- coding: utf-8 -*-
# Minimal NanEye2D (FOB 2.0) frame grabber using pythonnet + OpenCV
# Press ESC to quit.
import sys

sys.path.append(r'D:\Dropbox\Documents\Work\Cornell Lab Tech\Projects\Video VI\PyVAQ\Source')

import numpy as np, threading, time, os, queue
from pathlib import Path
from collections import deque
import clr
from System import IntPtr
from System.Runtime.InteropServices import Marshal
from System.Collections.Generic import List
from ffplayViewer import ffplayer

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
from Awaiba.Drivers.Grabbers.NanEye2D import NanEye2DUSB3
from Awaiba.Drivers.Grabbers import INanEye2D

# Create provider (same class as in your C# Form1)
provider = NanEyeFobProvider()
handler = NanEye2DUSB3()

# Point to firmware/FPGA (same as in C#)
provider.SetFWFile(str(FW_IMG))
provider.SetFpgaFile(str(FPGA_BIN))

# Enable only Sensor 1 (index 0 = True, index 1 = False), like your C#
sensors = List[bool]()
sensors.Add(True)   # Sensor 1
sensors.Add(True)  # Sensor 2
provider.Sensors = sensors

provider.SetSensor(0)
sen0 = INanEye2D(provider.CurrentSen)            # ISensor instance for the active port
provider.SetSensor(1)
sen1 = INanEye2D(provider.CurrentSen)            # ISensor instance for the active port

supply_voltage = 2200  # Default is 1800 mV, range is 1600 - 2400
print('sen0 before:', provider.CurrentSen.ToString)
sen0.set_Digipot(supply_voltage)
sen1.set_Digipot(supply_voltage)
print('sen0 after:', provider.CurrentSen.ToString)

breakpoint()

w, h = 250, 250       # or read once from first event
BPP = 3               # use 1 for 8-bit raw, 2 for 10/16-bit, 3 for RGB
buflen = w*h*BPP
RING = 20              # number of prealloc slots

recycled_images = queue.Queue(maxsize=RING)
new_images = queue.Queue(maxsize=RING)

# Preallocate ring of numpy buffers
for _ in range(RING):
    recycled_images.put(np.empty(buflen, dtype=np.uint8))

lastFrameCount = [None, None]
lastFramesTime = [None, None]

framesTimeHistory = [deque(maxlen=20), deque(maxlen=20)]

def on_image(sender, e):
    global lastFrameCount, new_images, recycled_images, lastFramesTime, framesTimeHistory
    # FAST path: copy raw bytes, enqueue index, return
    try:
        # choose one raw source:
        # src = e.GetImageData.GetRawPixels1Byte  # fast 8-bit
        # src = e.GetImageData.GetRawPixels2Byte # 10-bit expanded (2 bytes/px)
        src = e.PixelData                      # RGB processed (3 bytes/px)
        sensorID = e.SensorID
        frameCount = e.FrameCount  # FrameCount does not seem to be reliable.
        framesTime = e.TimeStamp
        dt_avg = np.mean(np.diff(framesTimeHistory[sensorID]))
        print('framerate: {fps:0.1f}'.format(fps=1000000/dt_avg))
        framesTimeHistory[sensorID].append(framesTime)
        if lastFrameCount[sensorID] is not None and frameCount != lastFrameCount[sensorID] + 1 and not (frameCount == 0 and lastFrameCount[sensorID] == 255):
            if lastFramesTime[sensorID] is None:
                dt = None
            else:
                dt = framesTime - lastFramesTime[sensorID]
            print('Sensor {s} dropped frame! {k} => {j}, dt={dt}/{dtavg}={df}'.format(
                s=sensorID,
                k=lastFrameCount[sensorID],
                j=frameCount,
                dt=int(round(dt/1000)),
                dtavg=int(round(dt_avg/1000)),
                df=int(round(dt/dt_avg))
            ))
        lastFramesTime[sensorID] = framesTime
        lastFrameCount[sensorID] = frameCount

        buf = recycled_images.get(block=False)
        Marshal.Copy(src, 0, IntPtr(buf.ctypes.data), src.Length)
        new_images.put((buf, sensorID, int(e.Width), int(e.Height), framesTime), block=False)
    except queue.Empty:
        print('empty recycled image queue!')
    except queue.Full:
        print('full new image queue!')

viewer = ffplayer(1000, 'sensor 0', pixelFormat='rgb24')

def ui_loop():
    pending_images = {}
    dual_buf = np.empty(buflen*2, dtype=np.uint8)
    pairs_received = 0
    while True:
        # print('new images:', new_images.qsize())
        # print('rcy images:', recycled_images.qsize())
        buf, sensorID, W, H, T = new_images.get(block=True)
        if T in pending_images:
            image_pair = pending_images[T]
            image_pair[sensorID] = buf
            dual_buf[:buflen] = image_pair[0]
            dual_buf[buflen:] = image_pair[1]
            recycled_images.put(image_pair[0], block=True)
            recycled_images.put(image_pair[1], block=True)
            pairs_received += 1
            del image_pair
            del pending_images[T]
            viewer.showFrame(dual_buf.reshape(H*2, W, 3))
        else:
            pending_images[T] = [None, None]
            pending_images[T][sensorID] = buf

        if len(pending_images) > 1:
            print('unpaired frames:', len(pending_images) - 1)

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
