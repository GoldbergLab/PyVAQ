import PySpin

def getCameraAttribute(nodemap, attributeName, attributeTypePtrFunction):
    nodeAttribute = attributeTypePtrFunction(nodemap.GetNode(attributeName))

    if not PySpin.IsAvailable(nodeAttribute) or not PySpin.IsReadable(nodeAttribute):
        print('Unable to retrieve '+attributeName+'. Aborting...')
        return None

    value = nodeAttribute.GetValue()
    return value


# Initialize system and find a camera
system = PySpin.System.GetInstance()
camList = system.GetCameras()
camSerials = []
cam = camList[0]
cam.Init()

nodemap = cam.GetNodeMap()
PFNode = PySpin.CEnumerationPtr(nodemap.GetNode('PixelFormat'))
entryPtrs = PFNode.GetEntries()
for entryPtr in entryPtrs:
    print(entryPtr)
    print(entryPtr.GetName())

# Clean up
cam.DeInit()
del cam
camList.Clear()
system.ReleaseInstance()
