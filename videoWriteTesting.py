import numpy as np
import sys
import time
from scipy.ndimage import uniform_filter
import ffmpegWriter as fw
from SharedImageQueue import SharedImageSender
import multiprocessing as mp

# Initialize variables
numFrames = 100
w = 1200
h = 1080
filename = sys.argv[1]
shapeArg = '{w}x{h}'.format(w=w, h=h)

pipeName = 'write_test'

imageQueue = SharedImageSender(
    width=w,
    height=h,
    verbose=1,
    pixelFormat="bayer_rggb8",
    outputType='bytes',
    channels=3,
    name=pipeName,
)
imageQueueReceiver = imageQueue.getReceiver()

def dummyAcquirer(w, h, numFrames, imageQueue):
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
    for f in range(numFrames):
        imageQueue.put(dummyFrames[f, :, :, :])

def dummyWriter(imageReceiver):
    while True:
        imageReceiver.get()
        videoFileInterface = fw.ffmpegVideoWriter(filename, "numpy", gpuVEnc=True)
        videoFileInterface.write(dummyFrames[f, :, :, :]);
        videoFileInterface.close()

if __name__ == '__main__':
    p = mp.Process(target=dummyAcquirer, args=(w, h, numFrames, imageQueue))
    p.start()
    startTime = time.time()
    dummyWriter(imageQueueReceiver)
    print('Elapsed time:')
    print(time.time() - startTime)
