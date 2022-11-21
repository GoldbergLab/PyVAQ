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


# Information about PySpin pixel formats, with a partial mapping to common ffmpeg pixel formats
pixelFormats = {
None:                           dict(bayer=False, channelCount=None, ffmpeg=None),
"Mono8":                        dict(bayer=False, channelCount=1,    ffmpeg=['gray']),
"Mono16":                       dict(bayer=False, channelCount=1,    ffmpeg=['gray']),
"RGB8Packed":                   dict(bayer=False, channelCount=3,    ffmpeg=None),
"BayerGR8":                     dict(bayer=True,  channelCount=1,    ffmpeg=['bayer_grbg8']),
"BayerRG8":                     dict(bayer=True,  channelCount=1,    ffmpeg=['bayer_rggb8']),
"BayerGB8":                     dict(bayer=True,  channelCount=1,    ffmpeg=['bayer_gbrg8']),
"BayerBG8":                     dict(bayer=True,  channelCount=1,    ffmpeg=['bayer_bggr8']),
"BayerGR16":                    dict(bayer=True,  channelCount=1,    ffmpeg=['bayer_grbg16le', 'bayer_grbg16be']),
"BayerRG16":                    dict(bayer=True,  channelCount=1,    ffmpeg=['bayer_rggb16le', 'bayer_rggb16be']),
"BayerGB16":                    dict(bayer=True,  channelCount=1,    ffmpeg=['bayer_gbrg16le', 'bayer_gbrg16be']),
"BayerBG16":                    dict(bayer=True,  channelCount=1,    ffmpeg=['bayer_bggr16le', 'bayer_bggr16be']),
"Mono12Packed":                 dict(bayer=False, channelCount=1,    ffmpeg=['gray12be', 'gray12le']),
"BayerGR12Packed":              dict(bayer=True,  channelCount=1,    ffmpeg=None),
"BayerRG12Packed":              dict(bayer=True,  channelCount=1,    ffmpeg=None),
"BayerGB12Packed":              dict(bayer=True,  channelCount=1,    ffmpeg=None),
"BayerBG12Packed":              dict(bayer=True,  channelCount=1,    ffmpeg=None),
"YUV411Packed":                 dict(bayer=False, channelCount=3,    ffmpeg=['yuv411p']),
"YUV422Packed":                 dict(bayer=False, channelCount=3,    ffmpeg=['yuv422p']),
"YUV444Packed":                 dict(bayer=False, channelCount=3,    ffmpeg=['yuv444p']),
"Mono12p":                      dict(bayer=False, channelCount=3,    ffmpeg=['gray12be', 'gray12le']),
"BayerGR12p":                   dict(bayer=True,  channelCount=1,    ffmpeg=None),
"BayerRG12p":                   dict(bayer=True,  channelCount=1,    ffmpeg=None),
"BayerGB12p":                   dict(bayer=True,  channelCount=1,    ffmpeg=None),
"BayerBG12p":                   dict(bayer=True,  channelCount=1,    ffmpeg=None),
"YCbCr8":                       dict(bayer=False, channelCount=3,    ffmpeg=None),
"YCbCr422_8":                   dict(bayer=False, channelCount=3,    ffmpeg=None),
"YCbCr411_8":                   dict(bayer=False, channelCount=3,    ffmpeg=None),
"BGR8":                         dict(bayer=False, channelCount=3,    ffmpeg=['bgr8']),
"BGRa8":                        dict(bayer=False, channelCount=3,    ffmpeg=['bgra']),
"Mono10Packed":                 dict(bayer=False, channelCount=3,    ffmpeg=['gray10be', 'gray10le']),
"BayerGR10Packed":              dict(bayer=True,  channelCount=1,    ffmpeg=None),
"BayerRG10Packed":              dict(bayer=True,  channelCount=1,    ffmpeg=None),
"BayerGB10Packed":              dict(bayer=True,  channelCount=1,    ffmpeg=None),
"BayerBG10Packed":              dict(bayer=True,  channelCount=1,    ffmpeg=None),
"Mono10p":                      dict(bayer=False, channelCount=1,    ffmpeg=['gray10be', 'gray10le']),
"BayerGR10p":                   dict(bayer=True,  channelCount=1,    ffmpeg=None),
"BayerRG10p":                   dict(bayer=True,  channelCount=1,    ffmpeg=None),
"BayerGB10p":                   dict(bayer=True,  channelCount=1,    ffmpeg=None),
"BayerBG10p":                   dict(bayer=True,  channelCount=1,    ffmpeg=None),
"Mono1p":                       dict(bayer=False, channelCount=1,    ffmpeg=['monow', 'monob']),
"Mono2p":                       dict(bayer=False, channelCount=1,    ffmpeg=None),
"Mono4p":                       dict(bayer=False, channelCount=1,    ffmpeg=None),
"Mono8s":                       dict(bayer=False, channelCount=1,    ffmpeg=['gray']),
"Mono10":                       dict(bayer=False, channelCount=1,    ffmpeg=['gray10be', 'gray10le']),
"Mono12":                       dict(bayer=False, channelCount=1,    ffmpeg=['gray12be', 'gray12le']),
"Mono14":                       dict(bayer=False, channelCount=1,    ffmpeg=['gray14be', 'gray14le']),
"Mono16s":                      dict(bayer=False, channelCount=1,    ffmpeg=['gray16be', 'gray16le']),
"Mono32f":                      dict(bayer=False, channelCount=1,    ffmpeg=None),
"BayerBG10":                    dict(bayer=True,  channelCount=1,    ffmpeg=None),
"BayerBG12":                    dict(bayer=True,  channelCount=1,    ffmpeg=None),
"BayerGB10":                    dict(bayer=True,  channelCount=1,    ffmpeg=None),
"BayerGB12":                    dict(bayer=True,  channelCount=1,    ffmpeg=None),
"BayerGR10":                    dict(bayer=True,  channelCount=1,    ffmpeg=None),
"BayerGR12":                    dict(bayer=True,  channelCount=1,    ffmpeg=None),
"BayerRG10":                    dict(bayer=True,  channelCount=1,    ffmpeg=None),
"BayerRG12":                    dict(bayer=True,  channelCount=1,    ffmpeg=None),
"RGBa8":                        dict(bayer=False, channelCount=3,    ffmpeg=['rgba']),
"RGBa10":                       dict(bayer=False, channelCount=3,    ffmpeg=None),
"RGBa10p":                      dict(bayer=False, channelCount=3,    ffmpeg=None),
"RGBa12":                       dict(bayer=False, channelCount=3,    ffmpeg=None),
"RGBa12p":                      dict(bayer=False, channelCount=3,    ffmpeg=None),
"RGBa14":                       dict(bayer=False, channelCount=3,    ffmpeg=None),
"RGBa16":                       dict(bayer=False, channelCount=3,    ffmpeg=['rgba64be', 'rgba64le']),
"RGB8":                         dict(bayer=False, channelCount=3,    ffmpeg=['rgb24']),
"RGB8_Planar":                  dict(bayer=False, channelCount=3,    ffmpeg=None),
"RGB10":                        dict(bayer=False, channelCount=3,    ffmpeg=None),
"RGB10_Planar":                 dict(bayer=False, channelCount=3,    ffmpeg=None),
"RGB10p":                       dict(bayer=False, channelCount=3,    ffmpeg=None),
"RGB10p32":                     dict(bayer=False, channelCount=3,    ffmpeg=None),
"RGB12":                        dict(bayer=False, channelCount=3,    ffmpeg=None),
"RGB12_Planar":                 dict(bayer=False, channelCount=3,    ffmpeg=None),
"RGB12p":                       dict(bayer=False, channelCount=3,    ffmpeg=None),
"RGB14":                        dict(bayer=False, channelCount=3,    ffmpeg=None),
"RGB16":                        dict(bayer=False, channelCount=3,    ffmpeg=None),
"RGB16s":                       dict(bayer=False, channelCount=3,    ffmpeg=None),
"RGB32f":                       dict(bayer=False, channelCount=3,    ffmpeg=None),
"RGB16_Planar":                 dict(bayer=False, channelCount=3,    ffmpeg=None),
"RGB565p":                      dict(bayer=False, channelCount=3,    ffmpeg=None),
"BGRa10":                       dict(bayer=False, channelCount=3,    ffmpeg=None),
"BGRa10p":                      dict(bayer=False, channelCount=3,    ffmpeg=None),
"BGRa12":                       dict(bayer=False, channelCount=3,    ffmpeg=None),
"BGRa12p":                      dict(bayer=False, channelCount=3,    ffmpeg=None),
"BGRa14":                       dict(bayer=False, channelCount=3,    ffmpeg=None),
"BGRa16":                       dict(bayer=False, channelCount=3,    ffmpeg=None),
"RGBa32f":                      dict(bayer=False, channelCount=3,    ffmpeg=None),
"BGR10":                        dict(bayer=False, channelCount=3,    ffmpeg=None),
"BGR10p":                       dict(bayer=False, channelCount=3,    ffmpeg=None),
"BGR12":                        dict(bayer=False, channelCount=3,    ffmpeg=None),
"BGR12p":                       dict(bayer=False, channelCount=3,    ffmpeg=None),
"BGR14":                        dict(bayer=False, channelCount=3,    ffmpeg=None),
"BGR16":                        dict(bayer=False, channelCount=3,    ffmpeg=None),
"BGR565p":                      dict(bayer=False, channelCount=3,    ffmpeg=None),
"R8":                           dict(bayer=False, channelCount=None, ffmpeg=None),
"R10":                          dict(bayer=False, channelCount=None, ffmpeg=None),
"R12":                          dict(bayer=False, channelCount=None, ffmpeg=None),
"R16":                          dict(bayer=False, channelCount=None, ffmpeg=None),
"G8":                           dict(bayer=False, channelCount=None, ffmpeg=None),
"G10":                          dict(bayer=False, channelCount=None, ffmpeg=None),
"G12":                          dict(bayer=False, channelCount=None, ffmpeg=None),
"G16":                          dict(bayer=False, channelCount=None, ffmpeg=None),
"B8":                           dict(bayer=False, channelCount=None, ffmpeg=None),
"B10":                          dict(bayer=False, channelCount=None, ffmpeg=None),
"B12":                          dict(bayer=False, channelCount=None, ffmpeg=None),
"B16":                          dict(bayer=False, channelCount=None, ffmpeg=None),
"Coord3D_ABC8":                 dict(bayer=False, channelCount=None, ffmpeg=None),
"Coord3D_ABC8_Planar":          dict(bayer=False, channelCount=None, ffmpeg=None),
"Coord3D_ABC10p":               dict(bayer=False, channelCount=None, ffmpeg=None),
"Coord3D_ABC10p_Planar":        dict(bayer=False, channelCount=None, ffmpeg=None),
"Coord3D_ABC12p":               dict(bayer=False, channelCount=None, ffmpeg=None),
"Coord3D_ABC12p_Planar":        dict(bayer=False, channelCount=None, ffmpeg=None),
"Coord3D_ABC16":                dict(bayer=False, channelCount=None, ffmpeg=None),
"Coord3D_ABC16_Planar":         dict(bayer=False, channelCount=None, ffmpeg=None),
"Coord3D_ABC32f":               dict(bayer=False, channelCount=None, ffmpeg=None),
"Coord3D_ABC32f_Planar":        dict(bayer=False, channelCount=None, ffmpeg=None),
"Coord3D_AC8":                  dict(bayer=False, channelCount=None, ffmpeg=None),
"Coord3D_AC8_Planar":           dict(bayer=False, channelCount=None, ffmpeg=None),
"Coord3D_AC10p":                dict(bayer=False, channelCount=None, ffmpeg=None),
"Coord3D_AC10p_Planar":         dict(bayer=False, channelCount=None, ffmpeg=None),
"Coord3D_AC12p":                dict(bayer=False, channelCount=None, ffmpeg=None),
"Coord3D_AC12p_Planar":         dict(bayer=False, channelCount=None, ffmpeg=None),
"Coord3D_AC16":                 dict(bayer=False, channelCount=None, ffmpeg=None),
"Coord3D_AC16_Planar":          dict(bayer=False, channelCount=None, ffmpeg=None),
"Coord3D_AC32f":                dict(bayer=False, channelCount=None, ffmpeg=None),
"Coord3D_AC32f_Planar":         dict(bayer=False, channelCount=None, ffmpeg=None),
"Coord3D_A8":                   dict(bayer=False, channelCount=None, ffmpeg=None),
"Coord3D_A10p":                 dict(bayer=False, channelCount=None, ffmpeg=None),
"Coord3D_A12p":                 dict(bayer=False, channelCount=None, ffmpeg=None),
"Coord3D_A16":                  dict(bayer=False, channelCount=None, ffmpeg=None),
"Coord3D_A32f":                 dict(bayer=False, channelCount=None, ffmpeg=None),
"Coord3D_B8":                   dict(bayer=False, channelCount=None, ffmpeg=None),
"Coord3D_B10p":                 dict(bayer=False, channelCount=None, ffmpeg=None),
"Coord3D_B12p":                 dict(bayer=False, channelCount=None, ffmpeg=None),
"Coord3D_B16":                  dict(bayer=False, channelCount=None, ffmpeg=None),
"Coord3D_B32f":                 dict(bayer=False, channelCount=None, ffmpeg=None),
"Coord3D_C8":                   dict(bayer=False, channelCount=None, ffmpeg=None),
"Coord3D_C10p":                 dict(bayer=False, channelCount=None, ffmpeg=None),
"Coord3D_C12p":                 dict(bayer=False, channelCount=None, ffmpeg=None),
"Coord3D_C16":                  dict(bayer=False, channelCount=None, ffmpeg=None),
"Coord3D_C32f":                 dict(bayer=False, channelCount=None, ffmpeg=None),
"Confidence1":                  dict(bayer=False, channelCount=None, ffmpeg=None),
"Confidence1p":                 dict(bayer=False, channelCount=None, ffmpeg=None),
"Confidence8":                  dict(bayer=False, channelCount=None, ffmpeg=None),
"Confidence16":                 dict(bayer=False, channelCount=None, ffmpeg=None),
"Confidence32f":                dict(bayer=False, channelCount=None, ffmpeg=None),
"BiColorBGRG8":                 dict(bayer=False, channelCount=None, ffmpeg=None),
"BiColorBGRG10":                dict(bayer=False, channelCount=None, ffmpeg=None),
"BiColorBGRG10p":               dict(bayer=False, channelCount=None, ffmpeg=None),
"BiColorBGRG12":                dict(bayer=False, channelCount=None, ffmpeg=None),
"BiColorBGRG12p":               dict(bayer=False, channelCount=None, ffmpeg=None),
"BiColorRGBG8":                 dict(bayer=False, channelCount=None, ffmpeg=None),
"BiColorRGBG10":                dict(bayer=False, channelCount=None, ffmpeg=None),
"BiColorRGBG10p":               dict(bayer=False, channelCount=None, ffmpeg=None),
"BiColorRGBG12":                dict(bayer=False, channelCount=None, ffmpeg=None),
"BiColorRGBG12p":               dict(bayer=False, channelCount=None, ffmpeg=None),
"SCF1WBWG8":                    dict(bayer=False, channelCount=None, ffmpeg=None),
"SCF1WBWG10":                   dict(bayer=False, channelCount=None, ffmpeg=None),
"SCF1WBWG10p":                  dict(bayer=False, channelCount=None, ffmpeg=None),
"SCF1WBWG12":                   dict(bayer=False, channelCount=None, ffmpeg=None),
"SCF1WBWG12p":                  dict(bayer=False, channelCount=None, ffmpeg=None),
"SCF1WBWG14":                   dict(bayer=False, channelCount=None, ffmpeg=None),
"SCF1WBWG16":                   dict(bayer=False, channelCount=None, ffmpeg=None),
"SCF1WGWB8":                    dict(bayer=False, channelCount=None, ffmpeg=None),
"SCF1WGWB10":                   dict(bayer=False, channelCount=None, ffmpeg=None),
"SCF1WGWB10p":                  dict(bayer=False, channelCount=None, ffmpeg=None),
"SCF1WGWB12":                   dict(bayer=False, channelCount=None, ffmpeg=None),
"SCF1WGWB12p":                  dict(bayer=False, channelCount=None, ffmpeg=None),
"SCF1WGWB14":                   dict(bayer=False, channelCount=None, ffmpeg=None),
"SCF1WGWB16":                   dict(bayer=False, channelCount=None, ffmpeg=None),
"SCF1WGWR8":                    dict(bayer=False, channelCount=None, ffmpeg=None),
"SCF1WGWR10":                   dict(bayer=False, channelCount=None, ffmpeg=None),
"SCF1WGWR10p":                  dict(bayer=False, channelCount=None, ffmpeg=None),
"SCF1WGWR12":                   dict(bayer=False, channelCount=None, ffmpeg=None),
"SCF1WGWR12p":                  dict(bayer=False, channelCount=None, ffmpeg=None),
"SCF1WGWR14":                   dict(bayer=False, channelCount=None, ffmpeg=None),
"SCF1WGWR16":                   dict(bayer=False, channelCount=None, ffmpeg=None),
"SCF1WRWG8":                    dict(bayer=False, channelCount=None, ffmpeg=None),
"SCF1WRWG10":                   dict(bayer=False, channelCount=None, ffmpeg=None),
"SCF1WRWG10p":                  dict(bayer=False, channelCount=None, ffmpeg=None),
"SCF1WRWG12":                   dict(bayer=False, channelCount=None, ffmpeg=None),
"SCF1WRWG12p":                  dict(bayer=False, channelCount=None, ffmpeg=None),
"SCF1WRWG14":                   dict(bayer=False, channelCount=None, ffmpeg=None),
"SCF1WRWG16":                   dict(bayer=False, channelCount=None, ffmpeg=None),
"YCbCr8_CbYCr":                 dict(bayer=False, channelCount=3,    ffmpeg=None),
"YCbCr10_CbYCr":                dict(bayer=False, channelCount=3,    ffmpeg=None),
"YCbCr10p_CbYCr":               dict(bayer=False, channelCount=3,    ffmpeg=None),
"YCbCr12_CbYCr":                dict(bayer=False, channelCount=3,    ffmpeg=None),
"YCbCr12p_CbYCr":               dict(bayer=False, channelCount=3,    ffmpeg=None),
"YCbCr411_8_CbYYCrYY":          dict(bayer=False, channelCount=3,    ffmpeg=None),
"YCbCr422_8_CbYCrY":            dict(bayer=False, channelCount=3,    ffmpeg=None),
"YCbCr422_10":                  dict(bayer=False, channelCount=3,    ffmpeg=None),
"YCbCr422_10_CbYCrY":           dict(bayer=False, channelCount=3,    ffmpeg=None),
"YCbCr422_10p":                 dict(bayer=False, channelCount=3,    ffmpeg=None),
"YCbCr422_10p_CbYCrY":          dict(bayer=False, channelCount=3,    ffmpeg=None),
"YCbCr422_12":                  dict(bayer=False, channelCount=3,    ffmpeg=None),
"YCbCr422_12_CbYCrY":           dict(bayer=False, channelCount=3,    ffmpeg=None),
"YCbCr422_12p":                 dict(bayer=False, channelCount=3,    ffmpeg=None),
"YCbCr422_12p_CbYCrY":          dict(bayer=False, channelCount=3,    ffmpeg=None),
"YCbCr601_8_CbYCr":             dict(bayer=False, channelCount=3,    ffmpeg=None),
"YCbCr601_10_CbYCr":            dict(bayer=False, channelCount=3,    ffmpeg=None),
"YCbCr601_10p_CbYCr":           dict(bayer=False, channelCount=3,    ffmpeg=None),
"YCbCr601_12_CbYCr":            dict(bayer=False, channelCount=3,    ffmpeg=None),
"YCbCr601_12p_CbYCr":           dict(bayer=False, channelCount=3,    ffmpeg=None),
"YCbCr601_411_8_CbYYCrYY":      dict(bayer=False, channelCount=3,    ffmpeg=None),
"YCbCr601_422_8":               dict(bayer=False, channelCount=3,    ffmpeg=None),
"YCbCr601_422_8_CbYCrY":        dict(bayer=False, channelCount=3,    ffmpeg=None),
"YCbCr601_422_10":              dict(bayer=False, channelCount=3,    ffmpeg=None),
"YCbCr601_422_10_CbYCrY":       dict(bayer=False, channelCount=3,    ffmpeg=None),
"YCbCr601_422_10p":             dict(bayer=False, channelCount=3,    ffmpeg=None),
"YCbCr601_422_10p_CbYCrY":      dict(bayer=False, channelCount=3,    ffmpeg=None),
"YCbCr601_422_12":              dict(bayer=False, channelCount=3,    ffmpeg=None),
"YCbCr601_422_12_CbYCrY":       dict(bayer=False, channelCount=3,    ffmpeg=None),
"YCbCr601_422_12p":             dict(bayer=False, channelCount=3,    ffmpeg=None),
"YCbCr601_422_12p_CbYCrY":      dict(bayer=False, channelCount=3,    ffmpeg=None),
"YCbCr709_8_CbYCr":             dict(bayer=False, channelCount=3,    ffmpeg=None),
"YCbCr709_10_CbYCr":            dict(bayer=False, channelCount=3,    ffmpeg=None),
"YCbCr709_10p_CbYCr":           dict(bayer=False, channelCount=3,    ffmpeg=None),
"YCbCr709_12_CbYCr":            dict(bayer=False, channelCount=3,    ffmpeg=None),
"YCbCr709_12p_CbYCr":           dict(bayer=False, channelCount=3,    ffmpeg=None),
"YCbCr709_411_8_CbYYCrYY":      dict(bayer=False, channelCount=3,    ffmpeg=None),
"YCbCr709_422_8":               dict(bayer=False, channelCount=3,    ffmpeg=None),
"YCbCr709_422_8_CbYCrY":        dict(bayer=False, channelCount=3,    ffmpeg=None),
"YCbCr709_422_10":              dict(bayer=False, channelCount=3,    ffmpeg=None),
"YCbCr709_422_10_CbYCrY":       dict(bayer=False, channelCount=3,    ffmpeg=None),
"YCbCr709_422_10p":             dict(bayer=False, channelCount=3,    ffmpeg=None),
"YCbCr709_422_10p_CbYCrY":      dict(bayer=False, channelCount=3,    ffmpeg=None),
"YCbCr709_422_12":              dict(bayer=False, channelCount=3,    ffmpeg=None),
"YCbCr709_422_12_CbYCrY":       dict(bayer=False, channelCount=3,    ffmpeg=None),
"YCbCr709_422_12p":             dict(bayer=False, channelCount=3,    ffmpeg=None),
"YCbCr709_422_12p_CbYCrY":      dict(bayer=False, channelCount=3,    ffmpeg=None),
"YUV8_UYV":                     dict(bayer=False, channelCount=3,    ffmpeg=None),
"YUV411_8_UYYVYY":              dict(bayer=False, channelCount=3,    ffmpeg=None),
"YUV422_8":                     dict(bayer=False, channelCount=3,    ffmpeg=None),
"YUV422_8_UYVY":                dict(bayer=False, channelCount=3,    ffmpeg=None),
"Polarized8":                   dict(bayer=False, channelCount=None, ffmpeg=None),
"Polarized10p":                 dict(bayer=False, channelCount=None, ffmpeg=None),
"Polarized12p":                 dict(bayer=False, channelCount=None, ffmpeg=None),
"Polarized16":                  dict(bayer=False, channelCount=None, ffmpeg=None),
"BayerRGPolarized8":            dict(bayer=True,  channelCount=1,    ffmpeg=None),
"BayerRGPolarized10p":          dict(bayer=True,  channelCount=1,    ffmpeg=None),
"BayerRGPolarized12p":          dict(bayer=True,  channelCount=1,    ffmpeg=None),
"BayerRGPolarized16":           dict(bayer=True,  channelCount=1,    ffmpeg=None),
"LLCMono8":                     dict(bayer=False, channelCount=1,    ffmpeg=None),
"LLCBayerRG8":                  dict(bayer=True,  channelCount=1,    ffmpeg=None),
"JPEGMono8":                    dict(bayer=False, channelCount=None, ffmpeg=None),
"JPEGColor8":                   dict(bayer=False, channelCount=None, ffmpeg=None),
"Raw16":                        dict(bayer=False, channelCount=None, ffmpeg=None),
"Raw8":                         dict(bayer=False, channelCount=None, ffmpeg=None),
"R12_Jpeg":                     dict(bayer=False, channelCount=None, ffmpeg=None),
"GR12_Jpeg":                    dict(bayer=False, channelCount=None, ffmpeg=None),
"GB12_Jpeg":                    dict(bayer=False, channelCount=None, ffmpeg=None),
"B12_Jpeg":                     dict(bayer=False, channelCount=None, ffmpeg=None),
"Mono 8":                       dict(bayer=False, channelCount=1,    ffmpeg=['gray']),
"Mono 12 Packed (IIDC-msb)":    dict(bayer=False, channelCount=1,    ffmpeg=['gray12be', 'gray12le']),
"Mono 12 Packed":               dict(bayer=False, channelCount=1,    ffmpeg=['gray12be', 'gray12le']),
"Mono 16":                      dict(bayer=False, channelCount=1,    ffmpeg=['gray16be', 'gray16le']),
"Bayer GR 8":                   dict(bayer=True,  channelCount=1,    ffmpeg=['bayer_grbg8']),
"Bayer RG 8":                   dict(bayer=True,  channelCount=1,    ffmpeg=['bayer_rggb8']),
"Bayer GB 8":                   dict(bayer=True,  channelCount=1,    ffmpeg=['bayer_gbrg8']),
"Bayer BG 8":                   dict(bayer=True,  channelCount=1,    ffmpeg=['bayer_bggr8']),
"BayerGR 12 Packed":            dict(bayer=True,  channelCount=1,    ffmpeg=None),
"BayerRG 12 Packed":            dict(bayer=True,  channelCount=1,    ffmpeg=None),
"BayerGB 12 Packed":            dict(bayer=True,  channelCount=1,    ffmpeg=None),
"BayerBG 12 Packed":            dict(bayer=True,  channelCount=1,    ffmpeg=None),
"BayerGR 12 Packed (IIDC-msb)": dict(bayer=True,  channelCount=1,    ffmpeg=None),
"BayerRG 12 Packed (IIDC-msb)": dict(bayer=True,  channelCount=1,    ffmpeg=None),
"BayerGB 12 Packed (IIDC-msb)": dict(bayer=True,  channelCount=1,    ffmpeg=None),
"BayerBG 12 Packed (IIDC-msb)": dict(bayer=True,  channelCount=1,    ffmpeg=None),
"BayerGR 16":                   dict(bayer=True,  channelCount=1,    ffmpeg=['bayer_grbg16le', 'bayer_grbg16be']),
"BayerRG 16":                   dict(bayer=True,  channelCount=1,    ffmpeg=['bayer_rggb16le', 'bayer_rggb16be']),
"BayerGB 16":                   dict(bayer=True,  channelCount=1,    ffmpeg=['bayer_gbrg16le', 'bayer_gbrg16be']),
"BayerBG 16":                   dict(bayer=True,  channelCount=1,    ffmpeg=['bayer_bggr16le', 'bayer_bggr16be']),
"YCbCr 411 8 (CbYYCrYY)":       dict(bayer=False, channelCount=3,    ffmpeg=None),
"YCbCr 422 8 (CbYCrY)":         dict(bayer=False, channelCount=3,    ffmpeg=None),
"YCbCr 8 (CbYCr)":              dict(bayer=False, channelCount=3,    ffmpeg=None),
"RGB 8":                        dict(bayer=False, channelCount=3,    ffmpeg=['rgb24'])
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

nodeAccessorFunctionsToTypeName = {
    PySpin.intfIString:         'string',
    PySpin.intfIInteger:        'integer',
    PySpin.intfIFloat:          'float',
    PySpin.intfIBoolean:        'boolean',
    PySpin.intfICommand:        'command',
    PySpin.intfIEnumeration:    'enum',
    PySpin.intfICategory:       'category',
    PySpin.intfIValue:          'value',
    PySpin.intfIBase:           'base',
    PySpin.intfIRegister:       'register',
    PySpin.intfIEnumEntry:      'enumEntry',
}

typeNameToNodeAccessorFunctions = {
    'string':       PySpin.intfIString,
    'integer':      PySpin.intfIInteger,
    'float':        PySpin.intfIFloat,
    'boolean':      PySpin.intfIBoolean,
    'command':      PySpin.intfICommand,
    'enum':         PySpin.intfIEnumeration,
    'category':     PySpin.intfICategory,
    'value':        PySpin.intfIValue,
    'base':         PySpin.intfIBase,
    'register':     PySpin.intfIRegister,
    'enumEntry':    PySpin.intfIEnumEntry,
}

typeNameToNodeType = {
    'string':       PySpin.CStringPtr,
    'integer':      PySpin.CIntegerPtr,
    'float':        PySpin.CFloatPtr,
    'boolean':      PySpin.CBooleanPtr,
    'command':      PySpin.CEnumerationPtr,
    'enum':         PySpin.CEnumerationPtr,
    'category':     PySpin.CCategoryPtr,
    'value':        PySpin.CValuePtr,
    'base':         PySpin.CBasePtr,
    'register':     PySpin.CRegisterPtr,
    'enumEntry':    PySpin.CEnumEntryPtr,
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
        camSerials.append(getCameraAttribute('DeviceSerialNumber', 'string', nodemap=cam.GetTLDeviceNodeMap()))
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
def getPixelFormat(cam=None):
    return getCameraAttribute('PixelFormat', 'enum', cam=cam)[1]

@handleCam
def isBayerFiltered(cam=None):
    name, displayName = getCameraAttribute('PixelFormat', 'enum', nodemap=cam.GetNodeMap())
    return pixelFormats[displayName]['bayer']

@handleCam
def getColorChannelCount(cam=None):
    nm = cam.GetNodeMap()
    # Get max dynamic range, which indicates the maximum value a single color channel can take
    maxPixelValue = getCameraAttribute('PixelDynamicRangeMax', 'integer', nodemap=nm);
    if maxPixelValue == 0:
        # For some reason Blackfly USB (not Blackfly S USB3) cameras return zero for this property.
        # We'll try to use the pixel format to determine the # of channels.
        pixelFormat = getPixelFormat(cam=cam)
        return pixelFormats[pixelFormat]['channelCount']
    # Get pixel size (indicating total # of bits per pixel)
    pixelSizeName, pixelSize = getCameraAttribute('PixelSize', 'enum', nodemap=nm)
    # Convert enum value to an integer
    pixelSize = pixelSizes[pixelSize]
    # Convert max value to # of bits
    channelSize = round(math.log(maxPixelValue + 1)/math.log(2))
    # Infer # of color channels
    numChannels = pixelSize / channelSize
    if abs(numChannels - round(numChannels)) > 0.0001:
        raise ValueError('Calculated # of color channels for camera {s} was not an integer ({bpp} bpp, {mdr} max channel value)'.format(s=camSerial, bpp=pixelSize, mdr=channelSize))
    return round(numChannels)

@handleCam
def getCameraAttribute(attributeName, attributeType, cam=None, camSerial=None, nodemap='NodeMap'):
    # Get an attribute from a camera
    #
    # Acceptable argument combinations:
    #   1.
    #   attributeName = name of attribute
    #   attributeType = type of attribute
    #   cam = PySpin.Camera instance,
    #   camSerial = None
    #   nodemap = string indicating type of nodemap to use
    #
    #   2.
    #   attributeName = name of attribute
    #   attributeType = type of attribute
    #   cam = None,
    #   camSerial = None
    #   nodemap = PySpin.INodeMap instance
    #   attributeName = name of attribute
    #
    #   3.
    #   attributeName = name of attribute
    #   attributeType = type of attribute
    #   cam = None
    #   camSerial = Valid serial # of a connected camera
    #   nodemap = string indicating type of nodemap to use
    #   attributeName = name of attribute

    nodeType = typeNameToNodeType[attributeType]

    if type(nodemap) == str:
        # nodemap is a string indicating whichy type of nodemap to get from cam
        nodemap = nodeMapAccessorFunctions[nodemap](cam)
    else:
        # nodemap is hopefully a PySpin.INodeMap instance
        pass

    nodeAttribute = nodeType(nodemap.GetNode(attributeName))

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
def setCameraAttribute(attributeName, attributeValue, attributeType, cam=None, nodemap='NodeMap'):
    # Set camera attribute. Return True if successful, False otherwise.

    if type(nodemap) == str:
        # nodemap is a string indicating whichy type of nodemap to get from cam
        nodemap = nodeMapAccessorFunctions[nodemap](cam)
    else:
        # nodemap is hopefully a PySpin.INodeMap instance
        pass

    nodeAttribute = typeNameToNodeType[attributeType](nodemap.GetNode(attributeName))
    if not PySpin.IsAvailable(nodeAttribute) or not PySpin.IsWritable(nodeAttribute):
        # print('Unable to set '+str(attributeName)+' to '+str(attributeValue)+' (enum retrieval). Aborting...')
        return False

    if attributeType == 'enum':
        # Retrieve entry node from enumeration node
        nodeAttributeValue = nodeAttribute.GetEntryByName(attributeValue)
        if not PySpin.IsAvailable(nodeAttributeValue) or not PySpin.IsReadable(nodeAttributeValue):
            # print('Unable to set '+str(attributeName)+' to '+str(attributeValue)+' (entry retrieval). Aborting...')
            return False

        # Set value
        attributeValue = nodeAttributeValue.GetValue()
        nodeAttribute.SetIntValue(attributeValue)
    else:
        nodeAttribute.SetValue(attributeValue)
    return True

@handleCam
def setCameraAttributes(attributeValueTriplets, cam=None, nodemap='NodeMap'):
    if type(nodemap) == str:
        # nodemap is a string indicating which type of nodemap to get from cam
        nodemap = nodeMapAccessorFunctions[nodemap](cam)
    else:
        # nodemap is hopefully a PySpin.INodeMap instance
        pass

    results = {}

    print('triplets:')
    print(attributeValueTriplets)

    for attribute, value, attributeType in attributeValueTriplets:
        results[attribute] = setCameraAttribute(attribute, value, attributeType, cam=cam, nodemap=nodemap)
        # if not result:
            # print("Failed to set", str(attribute), " to ", str(value))
    return results

@handleCam
def checkCameraSpeed(cam=None):
    try:
        cameraSpeedValue, cameraSpeed = getCameraAttribute('DeviceCurrentSpeed', 'enum', nodemap=cam.GetTLDeviceNodeMap())
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
        nodeTypeName = nodeAccessorFunctionsToTypeName[nodeType]
        nodeAccessorFunction = typeNameToNodeType[nodeTypeName]
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

        if value is not None:
            if nodeTypeName == 'integer':
                value = int(value)
            elif nodeTypeName == 'float':
                value = float(value)
            elif nodeTypeName == 'boolean':
                value = bool(int(value))

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
                if nodeType not in nodeAccessorFunctionsToTypeName:
                    print("Unknown node type:", nodeType)
                    continue
                childNodeTypeName = nodeAccessorFunctionsToTypeName[nodeType]
                nodeAccessorFunction = typeNameToNodeAccessorFunctions[childNodeTypeName]
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

@handleCam
def applyCameraConfiguration(configuration, cam=None):
    # Apply configuration of the form:
    # odict(
    #     attributeName1:{name=attributeName1, value=attributeValue1, type=attributeType1},
    #     attributeNameN:{name=attributeName1, value=attributeValue1, type=attributeType1},
    #     ...
    #     attributeNameN:{name=attributeNameN, value=attributeValueN, type=attributeTypeN},
    # )

    # Reformat configuration to [(name1, value1, type1), (name2, value2, type2), ..., (nameN, valueN, typeN)]
    formattedConfiguration = []
    for attributeName in configuration:
        attributeValue = configuration[attributeName]['value']
        attributeType =  configuration[attributeName]['type']
        attributeValue = self.convertAttributeValue(attributeValue, attributeType)
        formattedConfiguration.append(
            (attributeName, attributeValue, attributeType)
        )

    results = setCameraAttributes(formattedConfiguration, cam=cam)
    return results

def getAllCamerasAttributes(camSerials=None):
    if camSerials is None:
        camSerials = discoverCameras()
    cameraAttributes = {}
    for camSerial in camSerials:
        cameraAttributes[camSerial] = getAllCameraAttributes(camSerial=camSerial)
    return cameraAttributes

def flattenCameraAttributes(attribute, path=[]):
    flatAttributes = []

    if attribute['type'] == 'category':
        for subAttribute in attribute['children'] + attribute['subcategories']:
            flatAttributes.extend(
                    flattenCameraAttributes(subAttribute, path=path + [attribute['displayName']])
                )
    else:
        flatAttributes.append(dict(
            name=attribute['name'],
            displayName=attribute['displayName'],
            value=attribute['value'],
            options=attribute['options'],
            accessMode=attribute['accessMode'],
            type=attribute['type'],
            path=path
        ))
    return flatAttributes

def createCameraAttributeBrowser(container, camSerial):
    """Create a window allowing user to browse camera settings.

    Args:
        camSerial (str): String representing the serial number of a camera
            currently attached to the computer

    Returns:
        None

    """
    nb = ttk.Notebook(container)
    nb.grid(row=0)
    tooltipLabel = ttk.Label(container, text="temp")
    tooltipLabel.grid(row=1)

    attributes = getAllCamerasAttribute(camSerial=camSerial);
    widgets = createAttributeBrowserNode(attributes, nb, tooltipLabel, 1)

def createAttributeBrowserNode(attributeNode, parent, tooltipLabel, gridRow):
    """Create widgets for one camera attribute node in the browser.

    See createCameraAttributeBrowser

    Args:
        attributeNode (type): Description of parameter `attributeNode`.
        parent (type): Description of parameter `parent`.
        tooltipLabel (type): Description of parameter `tooltipLabel`.
        gridRow (type): Description of parameter `gridRow`.

    Returns:
        dict: Dictionary containing widgets created, organized by type

    """
    frame = ttk.Frame(parent)
    frame.bind("<Enter>", lambda event: tooltipLabel.config(text=attributeNode["tooltip"]))  # Set tooltip rollover callback
    frame.grid(row=gridRow)

    # syncPrint()
    # pp = pprint.PrettyPrinter(indent=1, depth=1)
    # pp.pprint(attributeNode)
    # syncPrint.log()

    widgets = [frame]
    childWidgets = []
    childCategoryHolder = None
    childCategoryWidgets = []

    if attributeNode['type'] == "category":
        children = []
        parent.add(frame, text=attributeNode['displayName'])
        if len(attributeNode['subcategories']) > 0:
            # If this category has subcategories, create a notebook to hold them
            childCategoryHolder = ttk.Notebook(frame)
            childCategoryHolder.grid(row=0)
            widgets.append(childCategoryHolder)
            for subcategoryAttributeNode in attributeNode['subcategories']:
                childCategoryWidgets.append(createAttributeBrowserNode(subcategoryAttributeNode, childCategoryHolder, tooltipLabel, 0))
        for k, childAttributeNode in enumerate(attributeNode['children']):
            childWidgets.append(createAttributeBrowserNode(childAttributeNode, frame, tooltipLabel, k+1))
    else:
        if attributeNode['accessMode'] == "RW":
            # Read/write attribute
            accessState = 'normal'
        else:
            # Read only attribute
            accessState = 'readonly'
        if attributeNode['type'] == "command":
            commandButton = ttk.Button(frame, text=attributeNode['displayName'])
            commandButton.grid()
            widgets.append(commandButton)
        elif attributeNode['type'] == "enum":
            enumLabel = ttk.Label(frame, text=attributeNode['displayName'])
            enumLabel.grid(column=0, row=0)
            options = list(attributeNode['options'].values())
            enumSelector = ttk.Combobox(frame, state=accessState, values=options)
            enumSelector.set(attributeNode['value'][1])
            enumSelector.grid(column=1, row=0)
            widgets.append(enumLabel)
            widgets.append(enumSelector)
        else:
            entryLabel = ttk.Label(frame, text=attributeNode['displayName'])
            entryLabel.grid(column=0, row=0)
            entry = ttk.Entry(frame, state=accessState)
            entry.insert(0, attributeNode['value'])
            entry.grid(column=1, row=0)
            widgets.append(entryLabel)
            widgets.append(entry)

    return {'widgets':widgets, 'childWidgets':childWidgets, 'childCategoryWidgets':childCategoryWidgets, 'childCategoryHolder':childCategoryHolder}

# For debugging purposes
if __name__ == "__main__":
    s = discoverCameras()[0]
    aa = getAllCameraAttributes(camSerial=s)
    print(aa)
    # attributes = ['PixelFormat', 'ExposureTime', 'ExposureAuto']
    # attributeTypes = ['enum', 'float', 'enum']
    # for attribute, attributeType in zip(attributes, attributeTypes):
    #     val = getCameraAttribute(attribute, attributeType, camSerial=s, nodemap='NodeMap')
    #     print('{a}: {v}'.format(a=attribute, v=val))
    #
    # print('setting ExposureAuto to Continuous')
    # setCameraAttribute('ExposureAuto', 'Continuous', 'enum', camSerial=s)
    #
    # for attribute, attributeType in zip(attributes, attributeTypes):
    #     val = getCameraAttribute(attribute, attributeType, camSerial=s, nodemap='NodeMap')
    #     print('{a}: {v}'.format(a=attribute, v=val))
