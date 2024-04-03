import numpy as np
import sys
import time
from scipy.ndimage import uniform_filter
import ffmpegWriter as fw

# Initialize variables
numFrames = 100
w = 1200
h = 1080
filename = sys.argv[1]
shapeArg = '{w}x{h}'.format(w=w, h=h)

dummyFrames = np.random.randint(0, 255, [numFrames, h, w, 3], dtype='uint8')
for k in range(numFrames):
    a = 5*k
    b = 5*k + 200
    dummyFrames[k, a:b, a:b, 0] = 255
    dummyFrames[k, a:b, a:b, 1] = 255
    dummyFrames[k, a:b, a:b, 2] = 0
print('Created dummy frames')
dummyFrames = uniform_filter(dummyFrames, size=3)
print('Smoothed')

repeatCount = 5

repeatTimes = []

for r in range(repeatCount):
    print('repeat #', r)
    startTime = time.time()
    videoFileInterface = fw.ffmpegVideoWriter(filename, "numpy", gpuVEnc=True)
    for f in range(numFrames):
        videoFileInterface.write(dummyFrames[f, :, :, :]);
    videoFileInterface.close()
    endTime = time.time()
    repeatTimes.append(endTime - startTime)

print('Elapsed times:')
print(repeatTimes)
