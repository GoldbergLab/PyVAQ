import numpy as np
import sys
import time
from scipy.ndimage import uniform_filter
import ffmpegIO as fIO
from SharedImageQueue import SharedImageSender
import multiprocessing as mp
from PIL import Image
from pathlib import Path

def dummyAcquirer(testFrames, imageQueue):
    numFrames = testFrames.shape[0]
    for f in range(numFrames):
        print("ACQUIRER: \"Acquiring\" frame #", f)
        time.sleep(1/10)
        imageQueue.put(imarray=testFrames[f, :, :, :])
    imageQueue.close()

def dummyWriter(filename, pipePath, pipeReady, shape, chunkSize):
    k = 1
    filePath = Path(filename)
    suffix = filePath.suffix
    originalStem = filePath.stem
    while True:
        if k > 1:
            filePath = filePath.with_name(originalStem + '_' + str(k) + suffix)
        print('Writing to:', filePath)
        while not pipeReady.is_set():
            print('Pipe not ready...trying again...')
            isReady = pipeReady.wait(timeout=1)
            print('pipe is ready?', isReady, pipeReady.is_set())
        videoFileInterface = fIO.ffmpegPipedVideoWriter(filePath, pipePath, gpuVEnc=True, verbose=1, input_pixel_format='rgb24', numFrames=chunkSize)
        # if k > 0:
        #     with open(pipePath, 'rb') as np:
        #         print('opened it')
        #         time.sleep(10)
        #         print('slept 10')
        #         breakpoint()
        #         print('breakpoint done')
        # time.sleep(0.1)
        videoFileInterface.initializeFFMPEG(shape[::-1])
        videoFileInterface.wait()
        videoFileInterface.close()
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
        pipeName=pipeName,
        includeMetadata=False,
        # createReceiver=False
    )
    # imageQueueReceiver = imageQueue.getReceiver()

    print('Writer starting')
    p = mp.Process(target=dummyAcquirer, args=(testFrames, imageQueue))
    p.start()
    print('Writer started')
    startTime = time.time()
    print('Acquirer starting')
    dummyWriter(filename, imageQueue.pipePath, imageQueue.pipeReady, [h, w], 25)
    print('Elapsed time:')
    print(time.time() - startTime)
