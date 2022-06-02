import sys
import math

simulatedHardware = False
for arg in sys.argv[1:]:
    if arg == '-s' or arg == '--sim':
        # Use simulated harddware instead of physical cameras and DAQs
        simulatedHardware = True

if simulatedHardware:
    # Use simulated harddware instead of physical cameras and DAQs
    import PySpinSim.PySpinSim as PySpin
else:
    # Use physical cameras/DAQs
    try:
        import PySpin
    except ModuleNotFoundError:
        # pip seems to install PySpin as pyspin sometimes...
        import pyspin as PySpin

pixelFormats = [
"PixelFormat_Mono8",
"PixelFormat_Mono16",
"PixelFormat_RGB8Packed",
"PixelFormat_BayerGR8",
"PixelFormat_BayerRG8",
"PixelFormat_BayerGB8",
"PixelFormat_BayerBG8",
"PixelFormat_BayerGR16",
"PixelFormat_BayerRG16",
"PixelFormat_BayerGB16",
"PixelFormat_BayerBG16",
"PixelFormat_Mono12Packed",
"PixelFormat_BayerGR12Packed",
"PixelFormat_BayerRG12Packed",
"PixelFormat_BayerGB12Packed",
"PixelFormat_BayerBG12Packed",
"PixelFormat_YUV411Packed",
"PixelFormat_YUV422Packed",
"PixelFormat_YUV444Packed",
"PixelFormat_Mono12p",
"PixelFormat_BayerGR12p",
"PixelFormat_BayerRG12p",
"PixelFormat_BayerGB12p",
"PixelFormat_BayerBG12p",
"PixelFormat_YCbCr8",
"PixelFormat_YCbCr422_8",
"PixelFormat_YCbCr411_8",
"PixelFormat_BGR8",
"PixelFormat_BGRa8",
"PixelFormat_Mono10Packed",
"PixelFormat_BayerGR10Packed",
"PixelFormat_BayerRG10Packed",
"PixelFormat_BayerGB10Packed",
"PixelFormat_BayerBG10Packed",
"PixelFormat_Mono10p",
"PixelFormat_BayerGR10p",
"PixelFormat_BayerRG10p",
"PixelFormat_BayerGB10p",
"PixelFormat_BayerBG10p",
"PixelFormat_Mono1p",
"PixelFormat_Mono2p",
"PixelFormat_Mono4p",
"PixelFormat_Mono8s",
"PixelFormat_Mono10",
"PixelFormat_Mono12",
"PixelFormat_Mono14",
"PixelFormat_Mono16s",
"PixelFormat_Mono32f",
"PixelFormat_BayerBG10",
"PixelFormat_BayerBG12",
"PixelFormat_BayerGB10",
"PixelFormat_BayerGB12",
"PixelFormat_BayerGR10",
"PixelFormat_BayerGR12",
"PixelFormat_BayerRG10",
"PixelFormat_BayerRG12",
"PixelFormat_RGBa8",
"PixelFormat_RGBa10",
"PixelFormat_RGBa10p",
"PixelFormat_RGBa12",
"PixelFormat_RGBa12p",
"PixelFormat_RGBa14",
"PixelFormat_RGBa16",
"PixelFormat_RGB8",
"PixelFormat_RGB8_Planar",
"PixelFormat_RGB10",
"PixelFormat_RGB10_Planar",
"PixelFormat_RGB10p",
"PixelFormat_RGB10p32",
"PixelFormat_RGB12",
"PixelFormat_RGB12_Planar",
"PixelFormat_RGB12p",
"PixelFormat_RGB14",
"PixelFormat_RGB16",
"PixelFormat_RGB16s",
"PixelFormat_RGB32f",
"PixelFormat_RGB16_Planar",
"PixelFormat_RGB565p",
"PixelFormat_BGRa10",
"PixelFormat_BGRa10p",
"PixelFormat_BGRa12",
"PixelFormat_BGRa12p",
"PixelFormat_BGRa14",
"PixelFormat_BGRa16",
"PixelFormat_RGBa32f",
"PixelFormat_BGR10",
"PixelFormat_BGR10p",
"PixelFormat_BGR12",
"PixelFormat_BGR12p",
"PixelFormat_BGR14",
"PixelFormat_BGR16",
"PixelFormat_BGR565p",
"PixelFormat_R8",
"PixelFormat_R10",
"PixelFormat_R12",
"PixelFormat_R16",
"PixelFormat_G8",
"PixelFormat_G10",
"PixelFormat_G12",
"PixelFormat_G16",
"PixelFormat_B8",
"PixelFormat_B10",
"PixelFormat_B12",
"PixelFormat_B16",
"PixelFormat_Coord3D_ABC8",
"PixelFormat_Coord3D_ABC8_Planar",
"PixelFormat_Coord3D_ABC10p",
"PixelFormat_Coord3D_ABC10p_Planar",
"PixelFormat_Coord3D_ABC12p",
"PixelFormat_Coord3D_ABC12p_Planar",
"PixelFormat_Coord3D_ABC16",
"PixelFormat_Coord3D_ABC16_Planar",
"PixelFormat_Coord3D_ABC32f",
"PixelFormat_Coord3D_ABC32f_Planar",
"PixelFormat_Coord3D_AC8",
"PixelFormat_Coord3D_AC8_Planar",
"PixelFormat_Coord3D_AC10p",
"PixelFormat_Coord3D_AC10p_Planar",
"PixelFormat_Coord3D_AC12p",
"PixelFormat_Coord3D_AC12p_Planar",
"PixelFormat_Coord3D_AC16",
"PixelFormat_Coord3D_AC16_Planar",
"PixelFormat_Coord3D_AC32f",
"PixelFormat_Coord3D_AC32f_Planar",
"PixelFormat_Coord3D_A8",
"PixelFormat_Coord3D_A10p",
"PixelFormat_Coord3D_A12p",
"PixelFormat_Coord3D_A16",
"PixelFormat_Coord3D_A32f",
"PixelFormat_Coord3D_B8",
"PixelFormat_Coord3D_B10p",
"PixelFormat_Coord3D_B12p",
"PixelFormat_Coord3D_B16",
"PixelFormat_Coord3D_B32f",
"PixelFormat_Coord3D_C8",
"PixelFormat_Coord3D_C10p",
"PixelFormat_Coord3D_C12p",
"PixelFormat_Coord3D_C16",
"PixelFormat_Coord3D_C32f",
"PixelFormat_Confidence1",
"PixelFormat_Confidence1p",
"PixelFormat_Confidence8",
"PixelFormat_Confidence16",
"PixelFormat_Confidence32f",
"PixelFormat_BiColorBGRG8",
"PixelFormat_BiColorBGRG10",
"PixelFormat_BiColorBGRG10p",
"PixelFormat_BiColorBGRG12",
"PixelFormat_BiColorBGRG12p",
"PixelFormat_BiColorRGBG8",
"PixelFormat_BiColorRGBG10",
"PixelFormat_BiColorRGBG10p",
"PixelFormat_BiColorRGBG12",
"PixelFormat_BiColorRGBG12p",
"PixelFormat_SCF1WBWG8",
"PixelFormat_SCF1WBWG10",
"PixelFormat_SCF1WBWG10p",
"PixelFormat_SCF1WBWG12",
"PixelFormat_SCF1WBWG12p",
"PixelFormat_SCF1WBWG14",
"PixelFormat_SCF1WBWG16",
"PixelFormat_SCF1WGWB8",
"PixelFormat_SCF1WGWB10",
"PixelFormat_SCF1WGWB10p",
"PixelFormat_SCF1WGWB12",
"PixelFormat_SCF1WGWB12p",
"PixelFormat_SCF1WGWB14",
"PixelFormat_SCF1WGWB16",
"PixelFormat_SCF1WGWR8",
"PixelFormat_SCF1WGWR10",
"PixelFormat_SCF1WGWR10p",
"PixelFormat_SCF1WGWR12",
"PixelFormat_SCF1WGWR12p",
"PixelFormat_SCF1WGWR14",
"PixelFormat_SCF1WGWR16",
"PixelFormat_SCF1WRWG8",
"PixelFormat_SCF1WRWG10",
"PixelFormat_SCF1WRWG10p",
"PixelFormat_SCF1WRWG12",
"PixelFormat_SCF1WRWG12p",
"PixelFormat_SCF1WRWG14",
"PixelFormat_SCF1WRWG16",
"PixelFormat_YCbCr8_CbYCr",
"PixelFormat_YCbCr10_CbYCr",
"PixelFormat_YCbCr10p_CbYCr",
"PixelFormat_YCbCr12_CbYCr",
"PixelFormat_YCbCr12p_CbYCr",
"PixelFormat_YCbCr411_8_CbYYCrYY",
"PixelFormat_YCbCr422_8_CbYCrY",
"PixelFormat_YCbCr422_10",
"PixelFormat_YCbCr422_10_CbYCrY",
"PixelFormat_YCbCr422_10p",
"PixelFormat_YCbCr422_10p_CbYCrY",
"PixelFormat_YCbCr422_12",
"PixelFormat_YCbCr422_12_CbYCrY",
"PixelFormat_YCbCr422_12p",
"PixelFormat_YCbCr422_12p_CbYCrY",
"PixelFormat_YCbCr601_8_CbYCr",
"PixelFormat_YCbCr601_10_CbYCr",
"PixelFormat_YCbCr601_10p_CbYCr",
"PixelFormat_YCbCr601_12_CbYCr",
"PixelFormat_YCbCr601_12p_CbYCr",
"PixelFormat_YCbCr601_411_8_CbYYCrYY",
"PixelFormat_YCbCr601_422_8",
"PixelFormat_YCbCr601_422_8_CbYCrY",
"PixelFormat_YCbCr601_422_10",
"PixelFormat_YCbCr601_422_10_CbYCrY",
"PixelFormat_YCbCr601_422_10p",
"PixelFormat_YCbCr601_422_10p_CbYCrY",
"PixelFormat_YCbCr601_422_12",
"PixelFormat_YCbCr601_422_12_CbYCrY",
"PixelFormat_YCbCr601_422_12p",
"PixelFormat_YCbCr601_422_12p_CbYCrY",
"PixelFormat_YCbCr709_8_CbYCr",
"PixelFormat_YCbCr709_10_CbYCr",
"PixelFormat_YCbCr709_10p_CbYCr",
"PixelFormat_YCbCr709_12_CbYCr",
"PixelFormat_YCbCr709_12p_CbYCr",
"PixelFormat_YCbCr709_411_8_CbYYCrYY",
"PixelFormat_YCbCr709_422_8",
"PixelFormat_YCbCr709_422_8_CbYCrY",
"PixelFormat_YCbCr709_422_10",
"PixelFormat_YCbCr709_422_10_CbYCrY",
"PixelFormat_YCbCr709_422_10p",
"PixelFormat_YCbCr709_422_10p_CbYCrY",
"PixelFormat_YCbCr709_422_12",
"PixelFormat_YCbCr709_422_12_CbYCrY",
"PixelFormat_YCbCr709_422_12p",
"PixelFormat_YCbCr709_422_12p_CbYCrY",
"PixelFormat_YUV8_UYV",
"PixelFormat_YUV411_8_UYYVYY",
"PixelFormat_YUV422_8",
"PixelFormat_YUV422_8_UYVY",
"PixelFormat_Polarized8",
"PixelFormat_Polarized10p",
"PixelFormat_Polarized12p",
"PixelFormat_Polarized16",
"PixelFormat_BayerRGPolarized8",
"PixelFormat_BayerRGPolarized10p",
"PixelFormat_BayerRGPolarized12p",
"PixelFormat_BayerRGPolarized16",
"PixelFormat_LLCMono8",
"PixelFormat_LLCBayerRG8",
"PixelFormat_JPEGMono8",
"PixelFormat_JPEGColor8",
"PixelFormat_Raw16",
"PixelFormat_Raw8",
"PixelFormat_R12_Jpeg",
"PixelFormat_GR12_Jpeg",
"PixelFormat_GB12_Jpeg",
"PixelFormat_B12_Jpeg"
]

pixelSizes = {
    'Bpp1':1,
    'Bpp2':2,
    'Bpp4':4,
    'Bpp8':8,
    'Bpp10':10,
    'Bpp12':12,
    'Bpp14':14,
    'Bpp16':16,
    'Bpp20':20,
    'Bpp24':24,
    'Bpp30':30,
    'Bpp32':32,
    'Bpp36':36,
    'Bpp48':48,
    'Bpp64':64,
    'Bpp96':96,
}

nodeAccessorFunctions = {
    PySpin.intfIString:('string', PySpin.CStringPtr),
    PySpin.intfIInteger:('integer', PySpin.CIntegerPtr),
    PySpin.intfIFloat:('float', PySpin.CFloatPtr),
    PySpin.intfIBoolean:('boolean', PySpin.CBooleanPtr),
    PySpin.intfICommand:('command', PySpin.CEnumerationPtr),
    PySpin.intfIEnumeration:('enum', PySpin.CEnumerationPtr),
    PySpin.intfICategory:('category', PySpin.CCategoryPtr)
}

nodeAccessorTypes = {
    'string':PySpin.CStringPtr,
    'integer':PySpin.CIntegerPtr,
    'float':PySpin.CFloatPtr,
    'boolean':PySpin.CBooleanPtr,
    'command':PySpin.CEnumerationPtr,
    'enum':PySpin.CEnumerationPtr,
    'category':PySpin.CCategoryPtr
}

def discoverCameras(numFakeCameras=0):
    system = PySpin.System.GetInstance()
    camList = system.GetCameras()
    camSerials = []
    for cam in camList:
        cam.Init()
        camSerials.append(getCameraAttribute(cam.GetTLDeviceNodeMap(), 'DeviceSerialNumber', PySpin.CStringPtr))
        cam.DeInit()
        del cam
    for k in range(numFakeCameras):
        camSerials.append('fake_camera_'+str(k))
    camList.Clear()
    system.ReleaseInstance()
    return camSerials

def initCam(camSerial, camList=None, system=None):
    if system is None and camList is None:
        system = PySpin.System.GetInstance()
    if camList is None:
        camList = system.GetCameras()
    cam = camList.GetBySerial(camSerial)
    cam.Init()
    return cam, camList, system

def initCams(camSerials=None, camList=None, system=None):
    if system is None and camList is None:
        system = PySpin.System.GetInstance()
    if camList is None:
        camList = system.GetCameras()
    if camSerials is None:
        # Init and return all cameras
        cams = []
        for cam in camList:
            cams.append(cam)
    else:
        cams = [camList.GetBySerial(camSerial) for camSerial in camSerials]
    for cam in cams:
        cam.Init()
    return cams, camList, system

def getColorChannelCount(cam=None, camSerial=None):
    if cam is None:
        cleanup = True
        cam, camList, system = initCam(camSerial)
    else:
        cleanup = False
    nm = cam.GetNodeMap()
    # Get pixel size (indicating total # of bits per pixel)
    pixelSizeName, pixelSize = getCameraAttribute(nm, 'PixelSize', PySpin.CEnumerationPtr)
    # Convert enum value to an integer
    pixelSize = pixelSizes[pixelSize]
    # Get max dynamic range, which indicates the maximum value a single color channel can take
    maxPixelValue = getCameraAttribute(nm, 'PixelDynamicRangeMax', PySpin.CIntegerPtr);
    # Convert max value to # of bits
    channelSize = round(math.log(maxPixelValue + 1)/math.log(2))
    # Infer # of color channels
    numChannels = pixelSize / channelSize
    if abs(numChannels - round(numChannels)) > 0.0001:
        raise ValueError('Calculated # of color channels for camera {s} was not an integer ({bpp} bpp, {mdr} max channel value)'.format(s=camSerial, bpp=pixelSize, mdr=channelSize))
    if cleanup:
        cam.DeInit()
        del cam
        camList.Clear()
        del camList
        system.ReleaseInstance()
    return round(numChannels)

def getCameraAttribute(nodemap, attributeName, attributeTypePtrFunction):
    nodeAttribute = attributeTypePtrFunction(nodemap.GetNode(attributeName))

    if not PySpin.IsAvailable(nodeAttribute) or not PySpin.IsReadable(nodeAttribute):
        raise AttributeError('Unable to retrieve '+attributeName+'. Aborting...')
        return None

    try:
        value = nodeAttribute.GetValue()
    except AttributeError:
        # Maybe it's an enum?
        valueEntry = nodeAttribute.GetCurrentEntry()
        value = (valueEntry.GetName(), valueEntry.GetDisplayName())
    return value

def checkCameraSpeed(camSerial):
    try:
        system = PySpin.System.GetInstance()
        camList = system.GetCameras()
        cam = camList.GetBySerial(camSerial)
        cam.Init()
        cameraSpeedValue, cameraSpeed = getCameraAttribute(cam.GetTLDeviceNodeMap(), 'DeviceCurrentSpeed', PySpin.CEnumerationPtr)
        # This causes weird crashes for one of our flea3 cameras...
        #cameraBaudValue, cameraBaud =   getCameraAttribute(cam.GetNodeMap(), 'SerialPortBaudRate', PySpin.CEnumerationPtr)
#        cameraSpeed = cameraSpeed + ' ' + cameraBaud
        cam.DeInit()
        del cam
        camList.Clear()
        system.ReleaseInstance()
        return cameraSpeed
    except:
        return "Unknown speed"

def queryAttributeNode(nodePtr, nodeType):
    """
    Retrieves and prints the display name and value of any node.
    """
    try:
        # Create string node
        (nodeTypeName, nodeAccessorFunction) = nodeAccessorFunctions[nodeType]
        node = nodeAccessorFunction(nodePtr)

        # Retrieve string node value
        try:
            display_name = node.GetDisplayName()
        except:
            display_name = None

        # Ensure that the value length is not excessive for printing
        try:
            value = node.GetValue()
        except AttributeError:
            try:
                valueEntry = node.GetCurrentEntry()
                value = (valueEntry.GetName(), valueEntry.GetDisplayName())
            except:
                value = None
        except:
            value = None

        try:
            symbolic  = node.GetSymbolic()
        except AttributeError:
            symbolic = None
        except:
            symbolic = None

        try:
            tooltip = node.GetToolTip()
        except AttributeError:
            tooltip = None
        except:
            tooltip = None

        try:
            accessMode = PySpin.EAccessModeClass_ToString(node.GetAccessMode())
        except AttributeError:
            accessMode = None
        except:
            accessMode = None

        try:
            options = {}
            optionsPtrs = node.GetEntries()
            for optionsPtr in optionsPtrs:
                options[optionsPtr.GetName()] = optionsPtr.GetDisplayName()
        except:
            if nodeTypeName == "enum":
                print("Failed to get options from enum!")
                traceback.print_exc()
            options = {}

        try:
            subcategories = []
            children = []
            for childNode in node.GetFeatures():
                # Ensure node is available and readable
                if not PySpin.IsAvailable(childNode) or not PySpin.IsReadable(childNode):
                    continue
                nodeType = childNode.GetPrincipalInterfaceType()
                if nodeType not in nodeAccessorFunctions:
                    print("Unknown node type:", nodeType)
                    continue
                (childNodeTypeName, nodeAccessorFunction) = nodeAccessorFunctions[nodeType]
                if childNodeTypeName == "category":
                    subcategories.append(queryAttributeNode(childNode, nodeType))
                else:
                    children.append(queryAttributeNode(childNode, nodeType))
        except AttributeError:
            # Not a category node
            pass
        except:
            pass

        try:
            name = node.GetName()
        except:
            name = None

        return {'type':nodeTypeName, 'name':name, 'symbolic':symbolic, 'displayName':display_name, 'value':value, 'tooltip':tooltip, 'accessMode':accessMode, 'options':options, 'subcategories':subcategories, 'children':children}

    except PySpin.SpinnakerException as ex:
        print('Error: %s' % ex)
        traceback.print_exc()
        return None

def getAllCameraAttributes(cam):
    # cam must be initialized before being passed to this function
    try:
        nodeData = {'type':'category', 'name':'Master', 'symbolic':'Master', 'displayName':'Master', 'value':None, 'tooltip':'Camera attributes', 'accessMode':'RO', 'options':{}, 'subcategories':[], 'children':[]}

        nodemap_gentl = cam.GetTLDeviceNodeMap()

        nodeDataTL = queryAttributeNode(nodemap_gentl.GetNode('Root'), PySpin.intfICategory)
        nodeDataTL['displayName'] = "Transport layer settings"
        nodeData['subcategories'].append(nodeDataTL)

        nodemap_tlstream = cam.GetTLStreamNodeMap()

        nodeDataTLStream = queryAttributeNode(nodemap_tlstream.GetNode('Root'), PySpin.intfICategory)
        nodeDataTLStream['displayName'] = "Transport layer stream settings"
        nodeData['subcategories'].append(nodeDataTLStream)

        nodemap_applayer = cam.GetNodeMap()

        nodeDataAppLayer = queryAttributeNode(nodemap_applayer.GetNode('Root'), PySpin.intfICategory)
        nodeDataAppLayer['displayName'] = "Camera settings"
        nodeData['subcategories'].append(nodeDataAppLayer)

        return nodeData

    except PySpin.SpinnakerException as ex:
        print('Error: %s' % ex)
        traceback.print_exc()
        return None

# For debugging purposes
if __name__ == "__main__":
    s = discoverCameras()[0]
    cam, camList, system = initCam(s)
    print(getColorChannelCount(cam=cam))
