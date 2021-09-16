import numpy as np
import multiprocessing as mp
from queue import Empty as qEmpty
from queue import Full as qFull
import ctypes
try:
    import PySpin
except ModuleNotFoundError:
    # pip seems to install PySpin as pyspin sometimes...
    import pyspin as PySpin
from PIL import Image

class SharedImageSender():
    def __init__(self,
                width,
                height,
                pixelFormat=PySpin.PixelFormat_BGR8,  #PySpin.PixelFormat_BayerRG8,
                offsetX=0,
                offsetY=0,
                verbose=False,
                channels=3,                    # Number of color channels in images (for example, 1 for grayscale, 3 for RGB, 4 for RGBA)
                imageDataType=ctypes.c_uint8,       # ctype of data of each pixel's channel value
                imageBitsPerPixel=8,                # Number of bits per pixel
                outputType='PySpin',                # Specify how to return data at receiver endpoint. Options are PySpin, numpy, PIL, rawBuffer
                outputCopy=True,                    # If true, the output data type references a copied buffer, rather than the original. Caution - the original buffer is not synchronized, and may be overwritten at any time!
                fileWriter=None,                    # fileWriter must be either None or a function that takes the specified output type and writes it to a file.
                lockForOutput=True,                # Should the shared memory buffer be locked during fileWriter call? If outputCopy=False and fileWriter is not None, it is recommended that lockForOutput=True
                maxBufferSize=1                     # Maximum number of images to allocate. Attempting to allocate more than that will raise an index error
                ):

        self.verbose = verbose
        self.maxBufferSize = maxBufferSize
        if self.maxBufferSize < 1:
            raise ValueError("maxBufferSize must be greater than 1")
        self.width = width
        self.height = height
        self.channels = channels

        self.metadataQueue = mp.Queue(maxsize=maxBufferSize)

        imageDataSize = self.width * self.height * imageBitsPerPixel * self.channels // 8
        if imageDataSize > 8000000000:
            print("Warning, super big buffer! Do you have enough RAM? Buffer will be {f} frames and {b} GB".format(f=maxBufferSize, b=round(maxBufferSize*imageDataSize/8000000000, 2)))

        self.readLag = mp.Value(ctypes.c_uint32, 0)
        self.nextID = 0

        self.buffersReady = False
        self.buffers = []
        self.bufferLocks = []
        self.npBuffers = []
        for k in range(self.maxBufferSize):
            # Create a series of shared memory locations, one for each image buffer slot
            self.buffers.append(mp.RawArray(imageDataType, imageDataSize))
            # Create synchronization locks to ensure that any given buffer element
            #   is not being read from and written to at the same time
            self.bufferLocks.append(mp.Lock())

        self.receiver = SharedImageReceiver(
            width=self.width,
            height=self.height,
            channels=self.channels,
            pixelFormat=pixelFormat,
            verbose=self.verbose,
            offsetX=offsetX,
            offsetY=offsetY,
            buffers=self.buffers,
            bufferLocks=self.bufferLocks,
            outputType=outputType,
            outputCopy=outputCopy,
            fileWriter=fileWriter,
            imageDataType=imageDataType,
            lockForOutput=lockForOutput,
            metadataQueue=self.metadataQueue,
            readLag=self.readLag)

    def qsize(self):
        return self.readLag.value

    def setupBuffers(self):
        # Prepare numpy buffer views inside putter process
        for k in range(self.maxBufferSize):
            # Create a series of numpy array views into that shared memory location.
            #   The numpy arrays share an underlying memory buffer with the shared memory arrays,
            #   so changing the numpy array changes the shared memory buffer (and vice versa)
            self.npBuffers.append(np.frombuffer(self.buffers[k], dtype=ctypes.c_uint8).reshape((self.height, self.width, self.channels)))
        self.buffersReady = True

    def getReceiver(self):
        return self.receiver

    def getNextID(self):
        nextID = self.nextID
        self.nextID += 1
        return nextID

    def put(self, image=None, imarray=None, metadata=None):
        # Puts an image in the shared memory queue.
        # Either pass a PySpin Image in the "image" argument
        #   or pass a numpy array in the "imarray" argument
        if not self.buffersReady:
            raise IOError("setupBuffers must be called in the process where putting will happen before putting any images")
        # image is a PySpin ImagePtr that must match the characteristics passed to the SharedImageSender constructor
        #   image = cam.GetNextImage()
        if image is not None:
            imarray = image.GetNDArray()
        # Use the numpy array view of the shared memory buffer to copy the image data into shared memory
        with self.readLag.get_lock():
            readLag = self.readLag.value
            if readLag >= self.maxBufferSize:
                raise qFull('Reader too far behind writer')
            self.readLag.value = readLag + 1
        nextID = self.getNextID()
        with self.bufferLocks[nextID % self.maxBufferSize]:
            np.copyto(self.npBuffers[nextID % self.maxBufferSize], imarray)
        if self.verbose: print("PUT! buffer #", nextID % self.maxBufferSize, " readlag=", readLag+1, "data=", imarray[0:5, 0, 0])
        self.metadataQueue.put(metadata, block=False)

class SharedImageReceiver():
    def __init__(self,
                width,
                height,
                channels,
                pixelFormat,
                verbose=False,
                offsetX=0,
                offsetY=0,
                buffers=None,
                bufferLocks=None,
                imageDataType=ctypes.c_uint8,
                outputType='PySpin',
                fileWriter=None,
                outputCopy=True,
                lockForOutput=True,
                metadataQueue=None,
                readLag=None
                ):
        self.width = width
        self.height = height
        self.pixelFormat = pixelFormat
        self.offsetX = offsetX
        self.offsetY = offsetY
        self.buffers = buffers
        self.bufferLocks = bufferLocks
        self.readLag = readLag
        self.maxBufferSize = len(self.buffers)
        self.outputType = outputType
        self.outputCopy = outputCopy
        self.fileWriter = fileWriter
        self.imageDataType = imageDataType
        self.lockForOutput = lockForOutput
        self.metadataQueue = metadataQueue
        self.verbose = verbose
        self.nextID = 0

    def getNextID(self):
        nextID = self.nextID
        self.nextID += 1
        return nextID

    def qsize(self):
        return self.readLag.value

    def prepareOutput(self, data):
        # As specified, copies, converts, and/or writes data to a file
        if self.outputType == 'PySpin':
            imarray = np.frombuffer(data, dtype=self.imageDataType).reshape((self.height, self.width, 3))
            output = PySpin.Image.Create(self.width, self.height, self.offsetX, self.offsetY, self.pixelFormat, imarray)
            if self.verbose: print("Got image: ", imarray[0:5, 0, 0])
            if self.outputCopy:
                pass # Do we need to copy this? Does PySpin.Image.Create already copy the buffer? Dunno, need to test to find out
        elif self.outputType == 'PIL':
            if self.outputCopy:
                output = Image.frombytes("RGB", (self.width, self.height), data, "raw", 'RGB')
            else:
                output = Image.frombuffer("RGB", (self.width, self.height), data, "raw", 'RGB')
            if self.verbose: print("Got image: ", np.array(output)[0:5, 0, 0])
        elif self.outputType == 'numpy':
            output = np.frombuffer(data, dtype=self.imageDataType).reshape((self.height, self.width, 3))
            if self.outputCopy:
                output = np.copy(output)
        elif self.outputType == 'rawBuffer':
            output = data
            if self.outputCopy:
                output = output[:]

        # If given, apply the fileWriter function to the output
        if self.fileWriter:
            self.fileWriter(output)
        return output

    def get(self, includeMetadata=False):
        # Returns a PySpin image, if one is available in the buffer, otherwise
        #   raise queue.Empty
        with self.readLag.get_lock():
            readLag = self.readLag.value
            if readLag == 0:
                # if self.verbose: print("NO GET!")
                raise qEmpty('No images available')
            self.readLag.value = readLag - 1
        nextID = self.getNextID()
        output = None
        with self.bufferLocks[nextID % self.maxBufferSize]:
            data = self.buffers[nextID % self.maxBufferSize]
            if self.verbose: print("GET! buffer #", nextID % self.maxBufferSize, " readlag=", readLag-1)
            if self.lockForOutput:
                output = self.prepareOutput(data)
        if not self.lockForOutput:
            output = self.prepareOutput(data)

        metadata = self.metadataQueue.get(block=False)


        if includeMetadata:
            return output, metadata
        else:
            return output
