from StateMachineProcesses import Stopwatch # StateMachineProcess
import multiprocessing as mp
import numpy as np
import traceback
from SharedImageQueue import SharedImageSender
from queue import Empty as qEmpty
import ffmpegWriter as fw
import rawWriter as rw
import time
import sys
try:
    import PySpin
except ModuleNotFoundError:
    # pip seems to install PySpin as pyspin sometimes...
    import pyspin as PySpin

def generateImageMap(width, height, maxVal=18):
    xVals = ((np.array(range(width))-width/2)*(maxVal*2/width)).astype('int16')
    yVals = ((np.array(range(height))-height/2)*(maxVal*2/height)).astype('int16')
    [xGrid, yGrid] = np.meshgrid(xVals, yVals)
    x = (np.sqrt(np.power(xGrid, 2) + np.power(yGrid, 2))).astype('uint8')
    return x

def generateNumberedImage_ring(imMap, index):
    binIndex = [int(i) for i in list('{0:0b}'.format(index))]
    binIndex.reverse()
    y = np.zeros(imMap.shape, dtype='uint8')
    for k, d in enumerate(binIndex):
        if d:
            y[imMap==k]=255
    return np.dstack((y, y, y))

def generateNumberedImage(arr, size, index):
    binIndex = [int(i) for i in list('{0:0b}'.format(index))]
    binIndex.reverse()
    w = 100
    for k, d in enumerate(binIndex):
        color = [d*255*int(i) for i in list('{0:03b}'.format((k % 7)+1))]
        arr[k*w:(k+1)*w, :, 0] = color[0]
        arr[k*w:(k+1)*w, :, 1] = color[1]
        arr[k*w:(k+1)*w, :, 2] = color[2]

def generateNumberedImage_mono(arr, size, index):
    binIndex = [int(i) for i in list('{0:0b}'.format(index))]
    binIndex.reverse()
    w = 100
    for k, d in enumerate(binIndex):
        arr[k*w:(k+1)*w, :] = d*255;

def generateRandomImage(size):
    pass

class ImageProcessor(mp.Process):
    def __init__(self, receiver, ready, camSerial='none'):
        super().__init__()
        self.receiver = receiver
        self.logBuffer = []
        self.ready = ready
        self.camSerial = camSerial

    def print(self, msg):
        self.logBuffer.append(msg)

    def flushLog(self, msg):
        for line in self.logBuffer:
            print(line)
        self.logBuffer = []

    def run(self):
        maxFrames = 1000
        videoCount = 0
        framesDropped = 0
        lastID = None
        writeStopwatch = Stopwatch(history=10)
        writeRaw = False
        self.ready.wait()
        print('\t\t\t\tRECEIVE: Passed barrier')
        while True:
            frameCount = 0
            videoPath = r"videoWriteTest_{s}_{k:03d}.avi".format(s=self.camSerial, k=videoCount)
            print("\t\t\t\tRECEIVE: Starting new video")
            videoFileInterface = fw.ffmpegWriter(videoPath, "bytes", fps=30)
#            videoFileInterface = rw.rawWriter(videoPath, "bytes", fps=30)
            while True:
                if frameCount >= maxFrames:
                    print('\t\t\t\tRECEIVE: Reached max frames={fc}!'.format(fc=frameCount))
                    print('\t\t\t\tRECEIVE: Closing video {k}, starting new one.'.format(k=videoCount))
                    break
#                print("\t\t\t\tRECEIVE: waiting for image")
                while True:
                    try:
                        im, metadata = self.receiver.get(includeMetadata=True)
                        frameCount += 1
                        break
                    except qEmpty:
                        pass
                # if lastID is not None:
                #     frameChange = metadata["ID"] - lastID
                #     lastID = metadata["ID"]
                #     if frameChange != 1:
                #         framesDropped += (frameChange - 1)
                if len(self.receiver.frameShape) == 3:
                    width, height, channels = self.receiver.frameShape
                else:
                    width, height = self.receiver.frameShape
                    channels = 1
                videoFileInterface.write(im, shape=(height, width))
#                print("\t\t\t\tRECEIVE: got image at address {addr}!".format(addr=im.data))
#                print("\t\t\t\tRECEIVE: image size = {w}x{h}!".format(w=width, h=height))
                writeStopwatch.click()
                print("\t\t\t\tRECEIVE: image ID = {id}".format(id=metadata["ID"]))
                print("\t\t\t\tRECEIVE: frames dropped = {fd}".format(fd=framesDropped))
                print("\t\t\t\tRECEIVE: processing backlog = {qs}".format(qs=self.receiver.qsize()))
                print("\t\t\t\tRECEIVE: wrote {k} of {n} to video".format(k=frameCount, n=maxFrames))
                print('\t\t\t\tRECEIVE: Write frequency = {f}'.format(f=writeStopwatch.frequency()))
#                print("\t\t\t\tRECEIVE: {m}".format(m=im.mean()))
            videoFileInterface.close()
            videoCount += 1

if __name__ == "__main__":
    acquireStopwatch = Stopwatch(history=10)
    useCamera = True

    if useCamera:
        camSerial = sys.argv[1]
        print('SEND:    Initializing camera, serial={s}'.format(s=camSerial))
        system = PySpin.System.GetInstance()
        camList = system.GetCameras()
        cam = camList.GetBySerial(camSerial)
        cam.Init()
        width = cam.Width.GetValue()
        height = cam.Height.GetValue()
    else:
        print('SEND:    Using synthetic images - no cameras')
        camSerial='synthetic'
        width=3208
        height=2200
        imArr = np.zeros([height, width], dtype='uint8')

    sis = SharedImageSender(
        width=width,
        height=height,
        verbose=False,
#        outputType='numpy',
        outputType='bytes',
        outputCopy=False,
        lockForOutput=False,
        maxBufferSize=80,
        channels=1
    )
    sis.setupBuffers()
    sir = sis.getReceiver()

    ready = mp.Barrier(2)
    ip = ImageProcessor(receiver=sir, ready=ready, camSerial=camSerial)
    ip.start()
#    nodemap = cam.GetNodeMap()
#    self.setCameraAttributes(nodemap, self.acquireSettings)
    if useCamera:
        print('SEND:    Beginning acquisition')
        cam.BeginAcquisition()
    else:
        print('SEND:    Begin synthetic generation')
    imID = 0
    lastImID = None
    framesDropped = 0
    framesGrabbed = 0
    ready.wait()
    print('SEND:    Passed barrier')
    startTime = time.time()
    try:
        while True:
            print('SEND: ***************************************************')
            if useCamera:
                print('SEND:    Fetching image...')
                while True:
                    try:
                        im = cam.GetNextImage(3000)
                        break
                    except PySpin.SpinnakerException:
                        print('SEND:    Image grab timeout. Trying again.')
                        framesGrabbed = 0
                imArr = im.GetNDArray()
                lastImID = imID
                imID = im.GetFrameID()
                im.Release()
                del im
            else:
                print('SEND:    Generating synthetic image...')
                lastImID = imID
                imID += 1
                generateNumberedImage(imArr, [height, width], imID)
                for k in range(300000):
                    # Wait a bit
                    x = 1000*1000;
#                time.sleep(1/1000)
            framesGrabbed += 1
            if lastImID is not None:
                if imID != lastImID + 1:
                    print('SEND:    DROPPED FRAMES')
                    framesDropped += (imID - lastImID - 1)
            print('SEND:    fetched image!')
            # print('SEND:    ', imArr.mean())
#            print('SEND:    sending data at address {addr}'.format(addr=imArr.data))
            print('SEND:    sending image id {id}...'.format(id=imID))
            sis.put(imarray=imArr, metadata={"ID":imID})
            print('SEND:    sent image!')
            acquireStopwatch.click()
            print('SEND:    Acquire frequency = {f}'.format(f=acquireStopwatch.frequency()))
            print('SEND:    Frames dropped = {f}'.format(f=framesDropped))
            print('SEND:    Frames grabbed in this burst = {f}'.format(f=framesGrabbed))
    except:
        print('SEND:    error!')
        print(traceback.format_exc())
        print('SEND:    Closing down...')
        if useCamera:
            cam.EndAcquisition()
            cam.DeInit()
            del cam
            camList.Clear()
            del camList
            system.ReleaseInstance()
        print('SEND:    EXITING')
    print('Total time: {t}'.format(t=time.time() - startTime))
