import numpy as np
import sys
import time
from scipy.ndimage import uniform_filter
import ffmpegIO as fw
from SharedImageQueue import SharedImageSender
import multiprocessing as mp

def dummyAcquirer(dummyFrames, imageQueue):
    numFrames = dummyFrames.shape[0]
    for f in range(numFrames):
        print("ACQUIRER: \"Acquiring\" frame #", f)
        imageQueue.put(imarray=dummyFrames[f, :, :, :])

def dummyWriter(imageReceiver, shape):
    time.sleep(3)
    videoFileInterface = fw.ffmpegVideoWriter(filename, "bytes", gpuVEnc=True, verbose=2)
    while True:
        print('WRITER: Getting')
        [frameBytes, metadata] = imageReceiver.get(includeMetadata=True)
        print('WRITER: Got')
        print('WRITER: Writing')
        print('WRITER: len=', len(frameBytes), 'shape=', shape)
        print('WRITER: class=', type(frameBytes))
        videoFileInterface.write(frameBytes, shape=shape);
        print('WRITER: Wrote')
        videoFileInterface.close()

if __name__ == '__main__':
    # Initialize variables

    ffr = fw.ffmpegReader(r'C:\Users\briankardon\Downloads\squirrel.avi')
    videoData = ffr.read()

    breakpoint()

    numFrames = 10
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
        pipeName=pipeName,
    )
    imageQueueReceiver = imageQueue.getReceiver()

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

    print('Writer starting')
    p = mp.Process(target=dummyAcquirer, args=(dummyFrames, imageQueue))
    p.start()
    print('Writer started')
    startTime = time.time()
    print('Acquirer starting')
    dummyWriter(imageQueueReceiver, [w, h])
    print('Elapsed time:')
    print(time.time() - startTime)
