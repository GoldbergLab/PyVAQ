import multiprocessing as mp
import numpy as np
import ctypes
width=10
height=12
channels=3
x = mp.RawArray(ctypes.c_uint8, width*height*channels)
y = np.frombuffer(x, dtype=ctypes.c_uint8).reshape((height, width, channels))
print('shared mem:', x[0])
print('numpy:', y[0, 0, 0])

new = np.random.randint(100, size=(height, width, channels), dtype=ctypes.c_uint8)
print('random:', new[0, 0, 0])
print()
np.copyto(y, new)

print('shared mem:', x[0])
print('numpy: ', y[0, 0, 0])
