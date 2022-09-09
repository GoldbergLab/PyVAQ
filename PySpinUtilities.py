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

pixelFormats = {
None:                     {'bayer':False},
"Mono8":                  {'bayer':False},
"Mono16":                 {'bayer':False},
"RGB8Packed":             {'bayer':False},
"BayerGR8":               {'bayer':True},
"BayerRG8":               {'bayer':True},
"BayerGB8":               {'bayer':True},
"BayerBG8":               {'bayer':True},
"BayerGR16":              {'bayer':True},
"BayerRG16":              {'bayer':True},
"BayerGB16":              {'bayer':True},
"BayerBG16":              {'bayer':True},
"Mono12Packed":           {'bayer':False},
"BayerGR12Packed":        {'bayer':True},
"BayerRG12Packed":        {'bayer':True},
"BayerGB12Packed":        {'bayer':True},
"BayerBG12Packed":        {'bayer':True},
"YUV411Packed":           {'bayer':False},
"YUV422Packed":           {'bayer':False},
"YUV444Packed":           {'bayer':False},
"Mono12p":                {'bayer':False},
"BayerGR12p":             {'bayer':True},
"BayerRG12p":             {'bayer':True},
"BayerGB12p":             {'bayer':True},
"BayerBG12p":             {'bayer':True},
"YCbCr8":                 {'bayer':False},
"YCbCr422_8":             {'bayer':False},
"YCbCr411_8":             {'bayer':False},
"BGR8":                   {'bayer':False},
"BGRa8":                  {'bayer':False},
"Mono10Packed":           {'bayer':False},
"BayerGR10Packed":        {'bayer':True},
"BayerRG10Packed":        {'bayer':True},
"BayerGB10Packed":        {'bayer':True},
"BayerBG10Packed":        {'bayer':True},
"Mono10p":                {'bayer':False},
"BayerGR10p":             {'bayer':True},
"BayerRG10p":             {'bayer':True},
"BayerGB10p":             {'bayer':True},
"BayerBG10p":             {'bayer':True},
"Mono1p":                 {'bayer':False},
"Mono2p":                 {'bayer':False},
"Mono4p":                 {'bayer':False},
"Mono8s":                 {'bayer':False},
"Mono10":                 {'bayer':False},
"Mono12":                 {'bayer':False},
"Mono14":                 {'bayer':False},
"Mono16s":                {'bayer':False},
"Mono32f":                {'bayer':False},
"BayerBG10":              {'bayer':True},
"BayerBG12":              {'bayer':True},
"BayerGB10":              {'bayer':True},
"BayerGB12":              {'bayer':True},
"BayerGR10":              {'bayer':True},
"BayerGR12":              {'bayer':True},
"BayerRG10":              {'bayer':True},
"BayerRG12":              {'bayer':True},
"RGBa8":                  {'bayer':False},
"RGBa10":                 {'bayer':False},
"RGBa10p":                {'bayer':False},
"RGBa12":                 {'bayer':False},
"RGBa12p":                {'bayer':False},
"RGBa14":                 {'bayer':False},
"RGBa16":                 {'bayer':False},
"RGB8":                   {'bayer':False},
"RGB8_Planar":            {'bayer':False},
"RGB10":                  {'bayer':False},
"RGB10_Planar":           {'bayer':False},
"RGB10p":                 {'bayer':False},
"RGB10p32":               {'bayer':False},
"RGB12":                  {'bayer':False},
"RGB12_Planar":           {'bayer':False},
"RGB12p":                 {'bayer':False},
"RGB14":                  {'bayer':False},
"RGB16":                  {'bayer':False},
"RGB16s":                 {'bayer':False},
"RGB32f":                 {'bayer':False},
"RGB16_Planar":           {'bayer':False},
"RGB565p":                {'bayer':False},
"BGRa10":                 {'bayer':False},
"BGRa10p":                {'bayer':False},
"BGRa12":                 {'bayer':False},
"BGRa12p":                {'bayer':False},
"BGRa14":                 {'bayer':False},
"BGRa16":                 {'bayer':False},
"RGBa32f":                {'bayer':False},
"BGR10":                  {'bayer':False},
"BGR10p":                 {'bayer':False},
"BGR12":                  {'bayer':False},
"BGR12p":                 {'bayer':False},
"BGR14":                  {'bayer':False},
"BGR16":                  {'bayer':False},
"BGR565p":                {'bayer':False},
"R8":                     {'bayer':False},
"R10":                    {'bayer':False},
"R12":                    {'bayer':False},
"R16":                    {'bayer':False},
"G8":                     {'bayer':False},
"G10":                    {'bayer':False},
"G12":                    {'bayer':False},
"G16":                    {'bayer':False},
"B8":                     {'bayer':False},
"B10":                    {'bayer':False},
"B12":                    {'bayer':False},
"B16":                    {'bayer':False},
"Coord3D_ABC8":           {'bayer':False},
"Coord3D_ABC8_Planar":    {'bayer':False},
"Coord3D_ABC10p":         {'bayer':False},
"Coord3D_ABC10p_Planar":  {'bayer':False},
"Coord3D_ABC12p":         {'bayer':False},
"Coord3D_ABC12p_Planar":  {'bayer':False},
"Coord3D_ABC16":          {'bayer':False},
"Coord3D_ABC16_Planar":   {'bayer':False},
"Coord3D_ABC32f":         {'bayer':False},
"Coord3D_ABC32f_Planar":  {'bayer':False},
"Coord3D_AC8":            {'bayer':False},
"Coord3D_AC8_Planar":     {'bayer':False},
"Coord3D_AC10p":          {'bayer':False},
"Coord3D_AC10p_Planar":   {'bayer':False},
"Coord3D_AC12p":          {'bayer':False},
"Coord3D_AC12p_Planar":   {'bayer':False},
"Coord3D_AC16":           {'bayer':False},
"Coord3D_AC16_Planar":    {'bayer':False},
"Coord3D_AC32f":          {'bayer':False},
"Coord3D_AC32f_Planar":   {'bayer':False},
"Coord3D_A8":             {'bayer':False},
"Coord3D_A10p":           {'bayer':False},
"Coord3D_A12p":           {'bayer':False},
"Coord3D_A16":            {'bayer':False},
"Coord3D_A32f":           {'bayer':False},
"Coord3D_B8":             {'bayer':False},
"Coord3D_B10p":           {'bayer':False},
"Coord3D_B12p":           {'bayer':False},
"Coord3D_B16":            {'bayer':False},
"Coord3D_B32f":           {'bayer':False},
"Coord3D_C8":             {'bayer':False},
"Coord3D_C10p":           {'bayer':False},
"Coord3D_C12p":           {'bayer':False},
"Coord3D_C16":            {'bayer':False},
"Coord3D_C32f":           {'bayer':False},
"Confidence1":            {'bayer':False},
"Confidence1p":           {'bayer':False},
"Confidence8":            {'bayer':False},
"Confidence16":           {'bayer':False},
"Confidence32f":          {'bayer':False},
"BiColorBGRG8":           {'bayer':False},
"BiColorBGRG10":          {'bayer':False},
"BiColorBGRG10p":         {'bayer':False},
"BiColorBGRG12":          {'bayer':False},
"BiColorBGRG12p":         {'bayer':False},
"BiColorRGBG8":           {'bayer':False},
"BiColorRGBG10":          {'bayer':False},
"BiColorRGBG10p":         {'bayer':False},
"BiColorRGBG12":          {'bayer':False},
"BiColorRGBG12p":         {'bayer':False},
"SCF1WBWG8":              {'bayer':False},
"SCF1WBWG10":             {'bayer':False},
"SCF1WBWG10p":            {'bayer':False},
"SCF1WBWG12":             {'bayer':False},
"SCF1WBWG12p":            {'bayer':False},
"SCF1WBWG14":             {'bayer':False},
"SCF1WBWG16":             {'bayer':False},
"SCF1WGWB8":              {'bayer':False},
"SCF1WGWB10":             {'bayer':False},
"SCF1WGWB10p":            {'bayer':False},
"SCF1WGWB12":             {'bayer':False},
"SCF1WGWB12p":            {'bayer':False},
"SCF1WGWB14":             {'bayer':False},
"SCF1WGWB16":             {'bayer':False},
"SCF1WGWR8":              {'bayer':False},
"SCF1WGWR10":             {'bayer':False},
"SCF1WGWR10p":            {'bayer':False},
"SCF1WGWR12":             {'bayer':False},
"SCF1WGWR12p":            {'bayer':False},
"SCF1WGWR14":             {'bayer':False},
"SCF1WGWR16":             {'bayer':False},
"SCF1WRWG8":              {'bayer':False},
"SCF1WRWG10":             {'bayer':False},
"SCF1WRWG10p":            {'bayer':False},
"SCF1WRWG12":             {'bayer':False},
"SCF1WRWG12p":            {'bayer':False},
"SCF1WRWG14":             {'bayer':False},
"SCF1WRWG16":             {'bayer':False},
"YCbCr8_CbYCr":           {'bayer':False},
"YCbCr10_CbYCr":          {'bayer':False},
"YCbCr10p_CbYCr":         {'bayer':False},
"YCbCr12_CbYCr":          {'bayer':False},
"YCbCr12p_CbYCr":         {'bayer':False},
"YCbCr411_8_CbYYCrYY":    {'bayer':False},
"YCbCr422_8_CbYCrY":      {'bayer':False},
"YCbCr422_10":            {'bayer':False},
"YCbCr422_10_CbYCrY":     {'bayer':False},
"YCbCr422_10p":           {'bayer':False},
"YCbCr422_10p_CbYCrY":    {'bayer':False},
"YCbCr422_12":            {'bayer':False},
"YCbCr422_12_CbYCrY":     {'bayer':False},
"YCbCr422_12p":           {'bayer':False},
"YCbCr422_12p_CbYCrY":    {'bayer':False},
"YCbCr601_8_CbYCr":       {'bayer':False},
"YCbCr601_10_CbYCr":      {'bayer':False},
"YCbCr601_10p_CbYCr":     {'bayer':False},
"YCbCr601_12_CbYCr":      {'bayer':False},
"YCbCr601_12p_CbYCr":     {'bayer':False},
"YCbCr601_411_8_CbYYCrYY":{'bayer':False},
"YCbCr601_422_8":         {'bayer':False},
"YCbCr601_422_8_CbYCrY":  {'bayer':False},
"YCbCr601_422_10":        {'bayer':False},
"YCbCr601_422_10_CbYCrY": {'bayer':False},
"YCbCr601_422_10p":       {'bayer':False},
"YCbCr601_422_10p_CbYCrY":{'bayer':False},
"YCbCr601_422_12":        {'bayer':False},
"YCbCr601_422_12_CbYCrY": {'bayer':False},
"YCbCr601_422_12p":       {'bayer':False},
"YCbCr601_422_12p_CbYCrY":{'bayer':False},
"YCbCr709_8_CbYCr":       {'bayer':False},
"YCbCr709_10_CbYCr":      {'bayer':False},
"YCbCr709_10p_CbYCr":     {'bayer':False},
"YCbCr709_12_CbYCr":      {'bayer':False},
"YCbCr709_12p_CbYCr":     {'bayer':False},
"YCbCr709_411_8_CbYYCrYY":{'bayer':False},
"YCbCr709_422_8":         {'bayer':False},
"YCbCr709_422_8_CbYCrY":  {'bayer':False},
"YCbCr709_422_10":        {'bayer':False},
"YCbCr709_422_10_CbYCrY": {'bayer':False},
"YCbCr709_422_10p":       {'bayer':False},
"YCbCr709_422_10p_CbYCrY":{'bayer':False},
"YCbCr709_422_12":        {'bayer':False},
"YCbCr709_422_12_CbYCrY": {'bayer':False},
"YCbCr709_422_12p":       {'bayer':False},
"YCbCr709_422_12p_CbYCrY":{'bayer':False},
"YUV8_UYV":               {'bayer':False},
"YUV411_8_UYYVYY":        {'bayer':False},
"YUV422_8":               {'bayer':False},
"YUV422_8_UYVY":          {'bayer':False},
"Polarized8":             {'bayer':False},
"Polarized10p":           {'bayer':False},
"Polarized12p":           {'bayer':False},
"Polarized16":            {'bayer':False},
"BayerRGPolarized8":      {'bayer':True},
"BayerRGPolarized10p":    {'bayer':True},
"BayerRGPolarized12p":    {'bayer':True},
"BayerRGPolarized16":     {'bayer':True},
"LLCMono8":               {'bayer':False},
"LLCBayerRG8":            {'bayer':True},
"JPEGMono8":              {'bayer':False},
"JPEGColor8":             {'bayer':False},
"Raw16":                  {'bayer':False},
"Raw8":                   {'bayer':False},
"R12_Jpeg":               {'bayer':False},
"GR12_Jpeg":              {'bayer':False},
"GB12_Jpeg":              {'bayer':False},
"B12_Jpeg":               {'bayer':False}
}

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
    '8 Bits/Pixel':8,
    '10 Bits/Pixel':10,
    '12 Bits/Pixel':12,
    '16 Bits/Pixel':16,
    '24 Bits/Pixel':24,
    '32 Bits/Pixel':32,
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

nodeMapAccessorFunctions = {
    'TLDeviceNodeMap': lambda cam:cam.GetTLDeviceNodeMap(),
    'TLStreamNodeMap': lambda cam:cam.GetTLStreamNodeMap(),
    'TLNodeMap':       lambda cam:cam.GetTLNodeMap(),
    'NodeMap':         lambda cam:cam.GetNodeMap()
}

def handleCam(func):
    # Decorator that automatically gracefully opens and closes camera-related
    #   references if camSerial keyword arg is passed in, and cam keyword
    #   arg is not passed in.
    # Decorated function should take a cam keyword argument, but can be called
    #   with a camSerial keyword argument instead.
    def wrapper(*args, cam=None, camSerial=None, **kwargs):
        if cam is None and camSerial is not None:
            cleanup = True
            cam, camList, system = initCam(camSerial)
        else:
            cleanup = False
        returnVal = func(*args, cam=cam, **kwargs)
        if cleanup:
            cam.DeInit()
            del cam
            camList.Clear()
            del camList
            system.ReleaseInstance()
        return returnVal
    return wrapper

def handleCamList(func):
    # Decorator that automatically gracefully opens and closes system and
    #   camlist objects.
    # Decorated function should take a camList keyword argument
    def wrapper(*args, **kwargs):
        system = PySpin.System.GetInstance()
        camList = system.GetCameras()

        returnVal = func(*args, camList=camList, **kwargs)

        camList.Clear()
        del camList
        system.ReleaseInstance()

        return returnVal
    return wrapper

@handleCamList
def discoverCameras(camList=None, numFakeCameras=0):
    camSerials = []
    for cam in camList:
        cam.Init()
        camSerials.append(getCameraAttribute('DeviceSerialNumber', PySpin.CStringPtr, nodemap=cam.GetTLDeviceNodeMap()))
        cam.DeInit()
        del cam
    for k in range(numFakeCameras):
        camSerials.append('fake_camera_'+str(k))
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

@handleCam
def getFrameSize(cam=None):
    width = cam.Width.GetValue()
    height = cam.Height.GetValue()
    return width, height

@handleCam
def isBayerFiltered(cam=None):
    name, displayName = getCameraAttribute('PixelFormat', PySpin.CEnumerationPtr, nodemap=cam.GetNodeMap())
    return pixelFormats[displayName]['bayer']

@handleCam
def getColorChannelCount(cam=None, camSerial=None):
    nm = cam.GetNodeMap()
    # Get pixel size (indicating total # of bits per pixel)
    pixelSizeName, pixelSize = getCameraAttribute('PixelSize', PySpin.CEnumerationPtr, nodemap=nm)
    # Convert enum value to an integer
    pixelSize = pixelSizes[pixelSize]
    # Get max dynamic range, which indicates the maximum value a single color channel can take
    maxPixelValue = getCameraAttribute('PixelDynamicRangeMax', PySpin.CIntegerPtr, nodemap=nm);
    # Convert max value to # of bits
    channelSize = round(math.log(maxPixelValue + 1)/math.log(2))
    # Infer # of color channels
    numChannels = pixelSize / channelSize
    if abs(numChannels - round(numChannels)) > 0.0001:
        raise ValueError('Calculated # of color channels for camera {s} was not an integer ({bpp} bpp, {mdr} max channel value)'.format(s=camSerial, bpp=pixelSize, mdr=channelSize))
    return round(numChannels)

@handleCam
def getCameraAttribute(attributeName, attributeTypePtrFunction, cam=None, camSerial=None, nodemap='NodeMap'):
    # Get an attribute from a camera
    #
    # Acceptable argument combinations:
    #   1.
    #   cam = PySpin.Camera instance,
    #   camSerial = None
    #   nodemap = string indicating type of nodemap to use
    #   attributeName = name of attribute
    #   attributeTypePtrFunction = whatever the hell this is
    #
    #   2.
    #   cam = None,
    #   camSerial = None
    #   nodemap = PySpin.INodeMap instance
    #   attributeName = name of attribute
    #   attributeTypePtrFunction = whatever the hell this is
    #
    #   3.
    #   cam = None
    #   camSerial = Valid serial # of a connected camera
    #   nodemap = string indicating type of nodemap to use
    #   attributeName = name of attribute
    #   attributeTypePtrFunction = whatever the hell this is


    if type(nodemap) == str:
        # nodemap is a string indicating whichy type of nodemap to get from cam
        nodemap = nodeMapAccessorFunctions[nodemap](cam)
    else:
        # nodemap is hopefully a PySpin.INodeMap instance
        pass

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

@handleCam
def checkCameraSpeed(cam=None):
    try:
        cameraSpeedValue, cameraSpeed = getCameraAttribute('DeviceCurrentSpeed', PySpin.CEnumerationPtr, nodemap=cam.GetTLDeviceNodeMap())
        # This causes weird crashes for one of our flea3 cameras...
        #cameraBaudValue, cameraBaud =   getCameraAttribute(cam.GetNodeMap(), 'SerialPortBaudRate', PySpin.CEnumerationPtr)
#        cameraSpeed = cameraSpeed + ' ' + cameraBaud
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

@handleCam
def getAllCameraAttributes(cam=None):
    # cam must be initialized before being passed to this function
    try:
        nodeData = {
            'type':'category',
            'name':'Master',
            'symbolic':'Master',
            'displayName':'Master',
            'value':None,
            'tooltip':'Camera attributes',
            'accessMode':'RO',
            'options':{},
            'subcategories':[],
            'children':[]}

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

def getAllCamerasAttributes():
    camSerials = discoverCameras()
    cameraAttributes = {}
    for camSerial in camSerials:
        cameraAttributes[camSerial] = getAllCameraAttributes(camSerial=camSerial)
    return cameraAttributes

# For debugging purposes
if __name__ == "__main__":
    s = discoverCameras()[0]
    attributes = ['PixelFormat', 'PixelFormatInfoID', 'PixelFormatInfoSelector']
    types = [PySpin.CEnumerationPtr, PySpin.CStringPtr, PySpin.CStringPtr]
    for attribute, type in zip(attributes, types):
        val = getCameraAttribute(attribute, type, camSerial=s, nodemap='NodeMap')
        print('{a}: {v}'.format(a=attribute, v=val))
