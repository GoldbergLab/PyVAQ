import multiprocessing as mp
import time
import numpy as np
import ctypes

class testProcess(mp.Process):
    def __init__(self, *args, **kwargs):
        mp.Process.__init__(self, *args, **kwargs)
        self.width=10
        self.height=12
        self.channels=3
        self.x = mp.RawArray(ctypes.c_uint8, 10*12*3)

    def run(self):
        print("shared mem addr:", self.x)
        self.y = np.frombuffer(self.x, dtype=ctypes.c_uint8).reshape((self.height, self.width, self.channels))
        print('numpy in child:', self.y[0, 0, 0])
        print('shared mem in child:', self.x[0])
        new = np.random.randint(100, size=(self.height, self.width, self.channels), dtype=ctypes.c_uint8)
        print('random in child:', new[0, 0, 0])
        print()
        np.copyto(self.y, new)
        print('numpy in child after copy: ', self.y[0, 0, 0])
        print('shared mem in child after copy:', self.x[0])
        pass

if __name__ == "__main__":


    p = testProcess()
    p.start()





    input("Hit a key to continue\n")


    print('shared mem in main:', p.x[0])
