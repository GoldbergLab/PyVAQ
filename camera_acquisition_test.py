import PySpin
import multiprocessing as mp

def getCameraAttribute(nodemap, attributeName):
    node_acquisition_framerate = PySpin.CFloatPtr(nodemap.GetNode(attributeName))

    if not PySpin.IsAvailable(node_acquisition_framerate) and not PySpin.IsReadable(node_acquisition_framerate):
        print('Unable to retrieve frame rate. Aborting...')
        return False

    framerate_to_set = node_acquisition_framerate.GetValue()

system = PySpin.System.GetInstance()
cam_list = system.GetCameras()
for k, cam in enumerate(cam_list):
    cam.Init()

    nodemap_tldevice = cam.GetTLDeviceNodeMap()
    node_serial = PySpin.CStringPtr(nodemap_tldevice.GetNode('DeviceSerialNumber'))

    if PySpin.IsAvailable(node_serial) and PySpin.IsReadable(node_serial):
        device_serial_number = node_serial.GetValue()
        print('Device serial number retrieved as %s...' % device_serial_number)


    cam_serial = getCameraAttribute(cam.GetTLDeviceNodeMap(), 'DeviceSerialNumber')
    acquireSettings = {}
    imageQueue = mp.Queue()
    filename = 'camera_recording_'+cam_serial+'.avi'
    writeProcess = spawnWriteProcess(filename, imageQueue)
    acquireProcess = spawnAcquireProcess(cam_serial, acquireSettings, imageQueue)
    cam.DeInit()
cam_list.Clear()

system.ReleaseInstance()
