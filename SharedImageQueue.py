import numpy as np
import multiprocessing as mp
from queue import Empty as qEmpty
from queue import Full as qFull
import win32pipe, win32file, pywintypes
import ctypes
import time

try:
    import PySpin
except ModuleNotFoundError:
    # pip seems to install PySpin as pyspin sometimes...
    import pyspin as PySpin
from PIL import Image

def getPipePath(pipeName):
    return r'\\.\pipe\{name}'.format(name=pipeName)

# Thanks to ChrisWue for the example implementation of the named pipe infrastructure: https://stackoverflow.com/a/51239081

class SharedImageSender():
    def __init__(self,
                width,
                height,
                pixelFormat=PySpin.PixelFormat_BGR8,  #PySpin.PixelFormat_BayerRG8,
                offsetX=0,
                offsetY=0,
                verbose=0,
                channels=3,                    # Number of color channels in images (for example, 1 for grayscale, 3 for RGB, 4 for RGBA)
                imageDataType=ctypes.c_uint8,       # ctype of data of each pixel's channel value
                imageBitsPerPixel=8,                # Number of bits per pixel
                outputType='PySpin',                # Specify how to return data at receiver endpoint. Options are PySpin, numpy, PIL, bytes
                fileWriter=None,                    # fileWriter must be either None or a function that takes the specified output type and writes it to a file.
                lockForOutput=True,                 # Should the shared memory buffer be locked during fileWriter call? If outputCopy=False and fileWriter is not None, it is recommended that lockForOutput=True
                maxBufferSize=1,                    # Maximum number of images to allocate. Attempting to allocate more than that will raise an index error
                name='unnamed_fifo',                # Name for named pipe
                allowOverflow=False,                # Should an error be raised if the queue is filled up, or should old entries be overwritten?
                ):

        self.pipeName = name
        self.pipePath = getPipePath(self.pipeName)
        self.verbose = verbose
        self.maxBufferSize = maxBufferSize
        if self.maxBufferSize < 1:
            raise ValueError("{name}: maxBufferSize must be greater than 1".format(name=self.pipeName))
        self.width = width
        self.height = height
        self.channels = channels
        self.allowOverflow = allowOverflow

        self.metadataQueue = mp.Queue(maxsize=maxBufferSize)

        self.pipe = None
        self.pipeConnected = False

        self.receiver = SharedImageReceiver(
            width=self.width,
            height=self.height,
            channels=self.channels,
            pixelFormat=pixelFormat,
            frameSize=self.width*self.height*self.channels*imageBitsPerPixel,
            verbose=self.verbose,
            offsetX=offsetX,
            offsetY=offsetY,
            outputType=outputType,
            fileWriter=fileWriter,
            imageDataType=imageDataType,
            lockForOutput=lockForOutput,
            maxBufferSize = self.maxBufferSize,
            metadataQueue=self.metadataQueue,
            name=self.pipeName)

    def qsize(self):
        return (0, 0) # (main queue, metadata queue)

    def setupNamedPipe(self):
        self.pipe = win32pipe.CreateNamedPipe(
                self.pipePath,
                win32pipe.PIPE_ACCESS_OUTBOUND,
                win32pipe.PIPE_TYPE_MESSAGE | win32pipe.PIPE_WAIT,
                1, 65536, 65536,
                300,
                None)

    def getReceiver(self):
        return self.receiver

    def connectPipe(self):
        win32pipe.ConnectNamedPipe(self.pipe, None)
        self.pipeConnected = True

    def put(self, image=None, imarray=None, metadata=None):
        # Puts an image in the shared memory queue.
        # Either pass a PySpin Image in the "image" argument
        #   or pass a numpy array in the "imarray" argument

        if not self.pipeConnected:
            # wait for pipe to be connected
            self.connectPipe()

        # image is a PySpin ImagePtr that must match the characteristics passed to the SharedImageSender constructor
        #   image = cam.GetNextImage()
        if image is not None:
            imarray = image.GetNDArray()

        # Write the numpy array data buffer to the pipe
        win32file.WriteFile(self.pipe, imarray.data)

        try:
            self.metadataQueue.put(metadata, block=False)
        except qFull:
            if not self.allowOverflow:
                raise qFull('{name}: Metadata queue overflow. qsize={qsize} max={maxsize}'.format(name=self.pipeName, qsize=self.qsize(), maxsize=self.maxBufferSize))
            elif self.verbose >= 2:
                print('{name}: Warning, metadata queue full. Overflow allowed - continuing...'.format(name=self.pipeName))

        # Write the numpy array data buffer to the pipe
        win32file.WriteFile(self.pipe, imarray.data)

    def close(self):
        win32file.CloseHandle(self.pipe)

class SharedImageReceiver():
    def __init__(self,
                width,
                height,
                channels,
                pixelFormat,
                frameSize,
                verbose=0,
                offsetX=0,
                offsetY=0,
                imageDataType=ctypes.c_uint8,
                outputType='PySpin',
                fileWriter=None,
                lockForOutput=True,
                maxBufferSize=1,
                metadataQueue=None,
                name=None
                ):
        self.width = width
        self.height = height
        self.pixelFormat = pixelFormat
        self.offsetX = offsetX
        self.offsetY = offsetY
        self.frameSize = frameSize
        self.maxBufferSize = maxBufferSize
        self.outputType = outputType
        self.fileWriter = fileWriter
        self.imageDataType = imageDataType
        self.channels = channels
        self.lockForOutput = lockForOutput
        self.metadataQueue = metadataQueue
        self.verbose = verbose
        self.nextID = 0
        self.pipeName = name
        self.pipePath = getPipePath(self.pipeName)
        self.pipeHandle = None
        self.pipeConnected = False

        if self.channels > 1:
            self.frameShape = (self.height, self.width, self.channels)
        else:
            self.frameShape = (self.height, self.width)

    def qsize(self):
        return (0, 0) # (main queue, metadata queue)

    def prepareOutput(self, data):
        # As specified, copies, converts, and/or writes data to a file
        if self.outputType == 'PySpin':
            imarray = np.frombuffer(data, dtype=self.imageDataType).reshape(self.frameShape)
            output = PySpin.Image.Create(self.width, self.height, self.offsetX, self.offsetY, self.pixelFormat, imarray)
            if self.verbose >= 3: print("Got image: ", imarray[0:5, 0, 0])
        elif self.outputType == 'PIL':
            if self.channels == 1:
                format = "L"
            elif self.channels == 3:
                format = "RGB"
            else:
                raise ValueError('{name}: Invalid channel count: {c}. Must be 1 or 3.'.format(name=self.pipeName, c=self.channels))
            # if self.outputCopy:
            #     output = Image.frombytes(format, (self.width, self.height), data, "raw", format, 0, 1)
            # else:
            output = Image.frombuffer(format, (self.width, self.height), data, "raw", format, 0, 1)
        elif self.outputType == 'numpy':
            output = np.frombuffer(data, dtype=self.imageDataType).reshape(self.frameShape)
            # if self.outputCopy:
            #     output = np.copy(output)
        elif self.outputType == 'bytes':
            output = data
            # if self.outputCopy:
            #     output = output[:]
        else:
            raise KeyError('{name}: Unrecognized output type: '.format(name=self.pipeName), self.outputType)

        # If given, apply the fileWriter function to the output
        if self.fileWriter:
            self.fileWriter(output)
            if self.verbose >= 3: print('Applied filewriter!')
        return output

    def connectPipe(self):
        success = False
        for k in range(5):
            print("attempt to connect to sender #", k+1)
            try:
                self.pipeHandle = open(self.pipePath, 'rb')
                success = True
                break
            except:
                success = False
                print('Failed to connect to pipe...')
                time.sleep(0.5)
        if not success:
            raise IOError("broken pipe, bye bye")
        else:
            self.pipeConnected = True

    def get(self, includeMetadata=False):
        # Returns a PySpin image, if one is available in the buffer, otherwise
        #   raise queue.Empty

        if not self.pipeConnected or self.pipeHandle is None:
            # Pipe not set up - do it now
            self.connectPipe()

        print('getting from queue')
        try:
            output = self.pipeHandle.read(size=self.frameSize)
        except:
            raise IOError('pipe read failed')

        metadata = self.metadataQueue.get(block=False)

        if includeMetadata:
            return output, metadata
        else:
            return output
