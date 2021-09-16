from StateMachineProcesses import Stopwatch # StateMachineProcess
import multiprocessing as mp
import numpy as np
import traceback
from SharedImageQueue import SharedImageSender
from queue import Empty as qEmpty
try:
    import PySpin
except ModuleNotFoundError:
    # pip seems to install PySpin as pyspin sometimes...
    import pyspin as PySpin

class ImageProcessor(mp.Process):
    def __init__(self, receiver):
        super().__init__()
        self.receiver = receiver
        self.logBuffer = []

    def print(self, msg):
        self.logBuffer.append(msg)

    def flushLog(self, msg):
        for line in self.logBuffer:
            print(line)
        self.logBuffer = []

    def run(self):
        while True:
            print("RECEIVE: waiting for image")
            while True:
                try:
                    im, metadata = self.receiver.get(includeMetadata=True)
                    break
                except qEmpty:
                    pass
            print("RECEIVE: got image at address {addr}!".format(addr=im.data))
            print("RECEIVE: image ID = {id}".format(id=metadata["ID"]))
            print("RECEIVE: {m}".format(m=im.mean()))

if __name__ == "__main__":
    acquireStopwatch = Stopwatch()
    sis = SharedImageSender(
        width=160,
        height=160,
        verbose=False,
        outputType='numpy',
        outputCopy=False,
        lockForOutput=False,
        maxBufferSize=10
    )
    sis.setupBuffers()
    sir = sis.getReceiver()

    ip = ImageProcessor(receiver=sir)
    ip.start()
    print('SEND:    Initializing camera')
    system = PySpin.System.GetInstance()
    camList = system.GetCameras()
    cam = camList.GetBySerial("19281923")
    cam.Init()
#    nodemap = cam.GetNodeMap()
#    self.setCameraAttributes(nodemap, self.acquireSettings)
    print('SEND:    Beginning acquisition')
    cam.BeginAcquisition()
    while True:
        try:
            print('SEND: ***************************************************')
            print('SEND:    Fetching image...')
            im = cam.GetNextImage()
            imArr = im.GetNDArray()
            imID = im.GetFrameID()
            print('SEND:    fetched image!')
            print('SEND:    ', imArr.mean())
            print('SEND:    sending data at address {addr}'.format(addr=imArr.data))
            print('SEND:    sending image...')
            sis.put(imarray=imArr, metadata={"ID":imID})
            print('SEND:    sent image!')
            acquireStopwatch.click()
            print('SEND:    Acquire frequency = {f}'.format(f=acquireStopwatch.frequency()))
        except:
            print('SEND:    error!')
            print(traceback.format_exc())
            break
