import multiprocessing as mp
import ctypes

imageDataType = ctypes.c_uint8
imageDataSize = 1024*1280*3  # 3,932,160 bytes
maxBufferSize = 300
buffers = []
for k in range(maxBufferSize):
    print("Creating buffer #", k)
    buffers.append(mp.RawArray(imageDataType, imageDataSize))
