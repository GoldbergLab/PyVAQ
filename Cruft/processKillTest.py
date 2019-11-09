import time
import multiprocessing as mp
import PySpin
import queue
from PIL import Image, ImageTk

class PickleableImage():
    def __init__(self, width, height, offsetX, offsetY, pixelFormat, data):
        self.width = width
        self.height = height
        self.offsetX = offsetX
        self.offsetY = offsetY
        self.pixelFormat = pixelFormat
        self.data = data


def setCameraAttribute(nodemap, attributeName, attributeValue, attributePtrType='CEnumerationPtr'):
    # Set camera attribute. Retrusn True if successful, False otherwise.
    nodeAttribute = getattr(PySpin, attributePtrType)(nodemap.GetNode(attributeName))
    if not PySpin.IsAvailable(nodeAttribute) or not PySpin.IsWritable(nodeAttribute):
        print('Unable to set '+attributeName+' to '+attributeValue+' (enum retrieval). Aborting...')
        return False

    # Retrieve entry node from enumeration node
    nodeAttributeValue = nodeAttribute.GetEntryByName(attributeValue)
    if not PySpin.IsAvailable(nodeAttributeValue) or not PySpin.IsReadable(nodeAttributeValue):
        print('Unable to set '+attributeName+' to '+attributeValue+' (entry retrieval). Aborting...')
        return False

    # Set value
    attributeValueCode = nodeAttributeValue.GetValue()
    nodeAttribute.SetIntValue(attributeValueCode)
    return True

def setCameraAttributes(nodemap, attributeValuePairs):
#    print('Setting attributes')
    for attribute, value in attributeValuePairs:
        result = setCameraAttribute(nodemap, attribute, value)
        if not result:
            print("Failed to set", attribute, " to ", attributes[attribute])
#    print('Done')

class VideoAcquirer(mp.Process):
    def __init__(self, camSerial, imageQueue, monitorImageQueue, acquireSettings={}, monitorFrameRate=15):
        mp.Process.__init__(self, daemon=True)
        self.camSerial = camSerial
        self.acquireSettings = acquireSettings
        self.imageQueue = imageQueue
        self.monitorImageQueue = monitorImageQueue
        self.monitorFrameRate = monitorFrameRate
        self.stop = mp.Event()

    def stopProcess(self):
        print('Stopping video acquire process')
        self.stop.set()

    def run(self):
        system = PySpin.System.GetInstance()
        camList = system.GetCameras()
        cam = camList.GetBySerial(self.camSerial)
        cam.Init()
        nodemap = cam.GetNodeMap()
        setCameraAttributes(nodemap, self.acquireSettings)
        cam.BeginAcquisition()

        monitorFramePeriod = 1.0/self.monitorFrameRate
        print("Video monitor frame period:", monitorFramePeriod)
        lastTime = time.time()
        k = 0
        imageResult = im = imp = None
        print("Image acquisition begins now!")
        while not self.stop.is_set():
            try:
                #  Retrieve next received image
                print(1)
                imageResult = cam.GetNextImage(100) # Timeout of 100 ms to allow for stopping process
                print(2)
                #  Ensure image completion
                if imageResult.IsIncomplete():
                    print('Image incomplete with image status %d...' % imageResult.GetImageStatus())
                else:
                    #  Print image information; height and width recorded in pixels
                    width = imageResult.GetWidth()
                    height = imageResult.GetHeight()
                    k = k + 1
#                    print('Grabbed Image %d, width = %d, height = %d' % (k, width, height))
                    im = imageResult.Convert(PySpin.PixelFormat_Mono8, PySpin.HQ_LINEAR)
                    imp = PickleableImage(im.GetWidth(), im.GetHeight(), 0, 0, im.GetPixelFormat(), im.GetData())
                    self.imageQueue.put(imp)

                    # Put the occasional image in the monitor queue for the UI
                    thisTime = time.time()
                    if (thisTime - lastTime) >= monitorFramePeriod:
                        # print("Sent frame for monitoring")
                        self.monitorImageQueue.put((self.camSerial, imp))
                        lastTime = thisTime

                    imageResult.Release()
                print(3)
            except PySpin.SpinnakerException as ex:
                pass # Hopefully this is just because there was no image in camera buffer
                # print('Error: %s' % ex)
                # traceback.print_exc()
                # return False

        # Send stop signal to write process
        print(4)
        self.imageQueue.put(None)

        camList.Clear()
        cam.EndAcquisition()
        cam.DeInit()
        print(5)
        del cam
        system.ReleaseInstance()
        del nodemap
        del imageResult
        del im
        del imp
        del camList
        del system
        print("Video acquire process STOPPED")



if __name__ == '__main__':
    q = mp.Queue()
    mq = mp.Queue()
    acquireSettings = [('AcquisitionMode', 'Continuous'), ('TriggerMode', 'Off'), ('TriggerSource', 'Line0'), ('TriggerMode', 'On')]   # List of attribute/value pairs to be applied to the camera in the given order
    p = VideoAcquirer('19129078', q, mq, acquireSettings=acquireSettings, monitorFrameRate=15)
    p.start()
    input("Press any key to stop.\n")
    p.stopProcess()
    time.sleep(2)
