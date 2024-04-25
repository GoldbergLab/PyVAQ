import numpy as np
import sys
import time
from scipy.ndimage import uniform_filter
import ffmpegIO as fIO
from SharedImageQueue import SharedImageSender
import multiprocessing as mp
from PIL import Image
from pathlib import Path
import queue
from subprocess import TimeoutExpired
import random as r
from PIL import Image

def dummyAcquirer(testFrames, imageQueue):
    numFrames = testFrames.shape[0]
    print('ACQUIRER: Ready to send {n} frames'.format(n=numFrames))
    for f in range(numFrames):
        print("ACQUIRER: \"Acquiring\" frame #", f)
        # time.sleep(1/10)
        imageQueue.put(imarray=testFrames[f, :, :, :], frameIndex=f)
        time.sleep(0.01)
    print('ACQUIRER DONE')
    imageQueue.close()

def dummyWriter(filename, imageReceiver, shape, numFrames):
    randomFrame = r.choice(range(numFrames))
    k = 0
    while True:
        print('Getting frame #', k)
        image = imageReceiver.get()
        if k == randomFrame:
            im = Image.frombytes('RGB', [imageReceiver.width, imageReceiver.height], image)
            im.show()
        time.sleep(0.1)
        k += 1

if __name__ == '__main__':
    # Initialize variables

    debugMode = 'squirrel'

    if debugMode == 'squirrel':
        ffr = fIO.ffmpegVideoReader(r'C:\Users\briankardon\Downloads\squirrel.avi')
        testFrames = ffr.read(startFrame=1, endFrame=75)

        # from PIL import Image
        # frame = videoData[5, :, :, :]
        # im = Image.fromarray(frame)
        # im.show()
        w = testFrames.shape[2]
        h = testFrames.shape[1]
        numFrames = testFrames.shape[0]
    elif debugMode == 'dummy':
        numFrames = 10
        w = 1200
        h = 1080
        testFrames = np.random.randint(0, 255, [numFrames, h, w, 3], dtype='uint8')
        for k in range(numFrames):
            a = 5*k
            b = 5*k + 200
            testFrames[k, a:b, a:b, 0] = 255
            testFrames[k, a:b, a:b, 1] = 255
            testFrames[k, a:b, a:b, 2] = 0

        testFrames = uniform_filter(testFrames, size=3)

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
        pipeBaseName=pipeName,
        includeMetadata=False,
        chunkFrameCount=25,
        createReceiver=True
    )
    imageReceiver = imageQueue.receiver

    print('Writer starting')
    p = mp.Process(target=dummyAcquirer, args=(testFrames, imageQueue))
    p.start()
    print('Writer started')
    startTime = time.time()
    print('Acquirer starting')
    dummyWriter(filename, imageReceiver, [h, w], numFrames)
    print('Elapsed time:')
    print(time.time() - startTime)
