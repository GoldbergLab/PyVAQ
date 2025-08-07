# -*- coding: utf-8 -*-
# Minimal NanEye2D (FOB 2.0) frame grabber using pythonnet + OpenCV
# Press ESC to quit.

import numpy as np, cv2, threading, time, sys, os
from collections import deque
import clr
from System import Array, Byte
from System.Runtime.InteropServices import Marshal
from System.Collections.Generic import List

root = r"C:\Users\briankardon\ams\NanEye_EvalSW_API_FiberOpticBox_csharp_pWin_v2-4-3-1"
SDK_DIR = os.path.join(root, r"NanEyeFobUsb3\bin\x64\Release")        # folder containing the Awaiba/ams-OSRAM DLLs
FW_IMG  = os.path.join(root, r"firmware\fx3_fw_2EP.img")     # same as C# Form1
FPGA_BIN = os.path.join(root, r"firmware\fob_fpga_v08.bin")  # same as C# Form1

# Make sure .NET can load the assemblies
sys.path.append(SDK_DIR)

# Load as importable assemblies
clr.AddReference(os.path.join(SDK_DIR, "Awaiba.dll"))
clr.AddReference(os.path.join(SDK_DIR, "AwSensors.dll"))
# clr.AddReference(os.path.join(SDK_DIR, "awcorecs.dll"))
clr.AddReference(os.path.join(SDK_DIR, "NanEyeUSB3Provider.dll"))
clr.AddReference(os.path.join(SDK_DIR, "CyUSB"))

# Import .NET types after references are loaded
from Awaiba.Drivers.Grabbers.NanEye2D.FobUsb3 import NanEyeFobProvider

# Create provider (same class as in your C# Form1)
provider = NanEyeFobProvider()

# Point to firmware/FPGA (same as in C#)
provider.SetFWFile(FW_IMG)
provider.SetFpgaFile(FPGA_BIN)

# Enable only Sensor 1 (index 0 = True, index 1 = False), like your C#
sensors = List[bool]()
sensors.Add(True)   # Sensor 1
sensors.Add(False)  # Sensor 2
provider.Sensors = sensors

print('Created provider:')
print('\t', provider.Width, 'x', provider.Height)
print('\t', provider.GetFrameRate(0), 'fps')
# (Optional) If you want to adjust auto-exposure etc., that’s available too,
# but we’ll keep the PoC minimal.

w, h = 250, 250       # or read once from first event
BPP = 1               # use 1 for 8-bit raw, 2 for 10/16-bit, 3 for RGB
buflen = w*h*BPP
RING = 4              # number of prealloc slots

# Preallocate ring of numpy buffers
ring = [np.empty(buflen, dtype=np.uint8) for _ in range(RING)]
q = deque(maxlen=RING)
qlock = threading.Lock()

def on_image(sender, e):
    # FAST path: copy raw bytes, enqueue index, return
    try:
        # choose one raw source:
        src = e.GetImageData.GetRawPixels1Byte  # fast 8-bit
        # src = e.GetImageData.GetRawPixels2Byte # 10-bit expanded (2 bytes/px)
        # src = e.PixelData                      # RGB processed (3 bytes/px)

        with qlock:
            # overwrite the oldest slot (bounded queue)
            slot = ring[len(q) % RING]
        Marshal.Copy(Array[Byte](src), 0, slot.__array_interface__['data'][0], len(slot))
        with qlock:
            q.append((slot, int(e.Width), int(e.Height)))
    except Exception as ex:
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
            break
        time.sleep(0.001)

provider.ImageProcessed += on_image  # (same event as C#) :contentReference[oaicite:13]{index=13}
provider.StartCapture()              # :contentReference[oaicite:14]{index=14}
threading.Thread(target=ui_loop, daemon=True).start()
provider.StopCapture()
