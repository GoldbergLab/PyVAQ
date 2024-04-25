import numpy as np
import multiprocessing as mp
from queue import Empty as qEmpty
from queue import Full as qFull
import win32pipe, win32file, pywintypes
import ctypes
import time
import queue
import threading as th

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
                width,                          # Height of video frame in pixels
                height,                         # Width of video frame in pixels
                pixelFormat=PySpin.PixelFormat_BGR8,  #PySpin.PixelFormat_BayerRG8,
                offsetX=0,                      # Parameter for PySpin output type
                offsetY=0,                      # Parameter for PySpin output type
                verbose=0,
                channels=3,                    # Number of color channels in images (for example, 1 for grayscale, 3 for RGB, 4 for RGBA)
                imageDataType=ctypes.c_uint8,       # ctype of data of each pixel's channel value
                imageBitsPerPixel=8,                # Number of bits per pixel
                outputType='PySpin',                # Specify how to return data at receiver endpoint. Options are PySpin, numpy, PIL, bytes
                fileWriter=None,                    # fileWriter must be either None or a function that takes the specified output type and writes it to a file.
                bufferSize=100,                  # Maximum size for metadata queue
                pipeBaseName=None,                  # Name for named pipe
                allowOverflow=False,                # Should an error be raised if the queue is filled up, or should old entries be overwritten?
                createReceiver=False,
                includeMetadata=True,
                chunkFrameCount=np.inf,                # Number of frames per named pipe, if inf, named pipe will never be closed by server, only one will be generated, unless autoReopen is true
                allowNonconsecutiveFrames=False,
                autoReopen=False                    # Automatically generate a new pipe if the pipe is broken?
                ):

        if pipeBaseName is None:
            pipeBaseName = 'PyVAQ_fifo_{t}'.format(t=int(time.time()*10))
        else:
            # Add timestamp to ensure the name is not in use
            pipeBaseName = pipeBaseName + '_' + str(int(time.time()*10))
        self.pipeBaseName = pipeBaseName
        self.pipeBasePath = getPipePath(self.pipeBaseName)
        self.verbose = verbose
        self.bufferSize = bufferSize
        if self.bufferSize < 1:
            raise ValueError("{name}: bufferSize must be greater than 1".format(name=self.pipeBaseName))
        self.width = width
        self.height = height
        self.channels = channels
        self.allowOverflow = allowOverflow

        self.currentPipeIndex = 0

        # Local list of pipe info - pipe handles stay in this list from the time
        #   they are created until they are marked as read by the reader and
        #   closed and cleaned up by the cleanupPipes function
        self.pipes = []
        # Queue to tell the reader which named pipes have been created and are ready for reading
        self.pipeReadyQueue = mp.Queue()
        # Queue to indicate which named pipes have been read and are ready for disposal
        self.pipeDoneQueue = mp.Queue()
        # The current pipe being written to
        self.currentPipe = None
        # Are we connected to the current pipe
        self.currentPipeConnected = False

        if includeMetadata:
            self.metadataQueue = mp.Queue(maxsize=bufferSize)
        else:
            self.metadataQueue = None

        self.chunkFrameCount = chunkFrameCount
        self.framesLeftInChunk = 0

        self.lastFrameIndex = None

        self.allowNonconsecutiveFrames = allowNonconsecutiveFrames

        self.frameSize = self.width*self.height*self.channels*(imageBitsPerPixel//8)
        self.bufferSize = bufferSize*self.frameSize

        if createReceiver:
            # If user requests a receiver object, create it.
            # Otherwise the user can use the named pipe directly
            self.receiver = SharedImageReceiver(
                width=self.width,
                height=self.height,
                channels=self.channels,
                pixelFormat=pixelFormat,
                frameSize=self.frameSize,
                chunkFrameCount=self.chunkFrameCount,
                verbose=self.verbose,
                pipeReadyQueue=self.pipeReadyQueue,
                pipeDoneQueue=self.pipeDoneQueue,
                offsetX=offsetX,
                offsetY=offsetY,
                outputType=outputType,
                fileWriter=fileWriter,
                imageDataType=imageDataType,
                bufferSize=self.bufferSize,
                metadataQueue=self.metadataQueue)
        else:
            self.receiver = None

    def initialize(self):
        # Optionally initialize the first pipe - this will happen automatically
        #   when put is first called, but this can be called first if the
        #   reader potentially needs the pipe to exist before adding data gets
        #   added
        if self.currentPipe is not None or len(self.pipes) > 0 or self.currentPipeConnected or not self.framesLeftInChunk == 0:
            raise RuntimeError('Error, initialize should only be called as the first call after the object has been constructed')
        self.addNewNamedPipe(frameRange=(0, self.chunkFrameCount))

    def qsize(self):
        return (-1, -1) # (main queue, metadata queue)

    def getNextPipeIndex(self):
        nextPipeIndex = self.currentPipeIndex
        self.currentPipeIndex += 1
        return nextPipeIndex

    def addNewNamedPipe(self, frameRange=(None, None)):
        pipeIndex = self.getNextPipeIndex()
        pipePath = self.pipeBasePath + '_' + str(pipeIndex)
        pipe = win32pipe.CreateNamedPipe(
                pipePath,                                                   # pipeName
                win32pipe.PIPE_ACCESS_OUTBOUND | win32file.GENERIC_WRITE,   # openMode (int)
                win32pipe.PIPE_TYPE_BYTE | win32pipe.PIPE_WAIT,             # pipeMode (int)
                1,                                                          # nMaxInstances
                self.bufferSize,                                                      # nOutBufferSize
                0,                                                      # nInBufferSize
                300,                                                        # nDefaultTimeOut
                None)                                                       # security attributes
        # Add new pipe handle to the end of the list
        self.pipes.append({'path':pipePath, 'index':pipeIndex, 'pipe':pipe, 'frameRange':frameRange})
        # Reset frame count for new pipe
        self.framesLeftInChunk = self.chunkFrameCount
        # Inform client of the new pipe path and index so it can use the pipe,
        #   then inform the server which pipe it's done reading from
        self.pipeReadyQueue.put({'path':pipePath, 'index':pipeIndex, 'frameRange':frameRange})
        # Set the current pipe for writing
        self.currentPipe = pipe
        # Haven't connected yet
        self.currentPipeConnected = False
        # Clean up any pipes the reader is done with
        self.cleanupPipes()

    def cleanupPipes(self, block=False, timeout=None):
        # Cleanup pipes that the reader is done with
        while True:
            try:
                pipePath = self.pipeDoneQueue.get(block=block, timeout=timeout)
                self.destroyPipe(pipePath)
            except queue.Empty:
                # No more done pipes
                break

    def getPipeInfoLocation(self, pipeIndexOrPath):
        # Get the location of the pipe in the self.pipes structure (note this is
        #   not the same as the "pipe index")
        if isinstance(pipeIndexOrPath, str):
            locations = [k for k, p in enumerate(self.pipes) if p['path'] == pipeIndexOrPath]
        elif isinstance(pipeIndexOrPath, int):
            locations = [k for k, p in enumerate(self.pipes) if p['index'] == pipeIndexOrPath]
        else:
            raise SyntaxError('pipeIndexOrPath must be an integer pipe index or a string pipe path')
        if len(locations) > 1:
            # Uh oh, duplicate pipe indices
            raise RuntimeError('Error, duplicate pipe indexes/paths for {i}!'.format(i=pipeIndexOrPath))
        elif len(locations) == 0:
            # Pipe index not found
            raise RunTimeError('Error, pipe index/path not found: {i}'.format(i=pipeIndexOrPath))
        else:
            # Only one match, which is as it should be
            return locations[0]

    def getPipe(self, pipeIndex):
        # Return the pipe handle matching the given pipe index
        pipeInfoLocation = self.getPipeInfoLocation(pipeIndex)
        return self.pipes[pipeInfoLocation]

    def destroyPipe(self, pipeIndexOrPath):
        pipeInfoLocation = self.getPipeInfoLocation(pipeIndexOrPath)
        pipeInfo = self.pipes[pipeInfoLocation]
        win32file.CloseHandle(pipeInfo['pipe'])
        self.pipes.pop(pipeInfoLocation)

    def connectPipe(self):
        if self.pipes is None:
            self.setupNamedPipe()
        if not self.pipeConnected:
            win32pipe.ConnectNamedPipe(self.pipe, None)
            self.pipeConnected = True

    def put(self, image=None, imarray=None, frameIndex=0, metadata=None):
        # Puts an image in the shared memory queue.
        # Either pass a PySpin Image in the "image" argument
        #   or pass a numpy array in the "imarray" argument
        if self.lastFrameIndex is not None and not frameIndex - self.lastFrameIndex == 1:
            msg = 'Frames are not consecutive! Last index: {i} Current index: {j}'.format(i=self.lastFrameIndex, j=frameIndex)
            if self.allowNonconsecutiveFrames:
                print(msg)
            else:
                raise IndexError(msg)
        self.lastFrameIndex = frameIndex

        if self.framesLeftInChunk < 0:
            raise RuntimeError('Something went wrong: self.framesLeftInChunk={flic}'.format(flic=self.framesLeftInChunk))
        if self.currentPipe is None or self.framesLeftInChunk == 0:
            # Either no pipes have been created yet, or the current pipe is done - set up another one
            self.addNewNamedPipe(frameRange=(frameIndex, frameIndex + self.chunkFrameCount))
        if not self.currentPipeConnected:
            # Wait for client to connect to pipe
            win32pipe.ConnectNamedPipe(self.currentPipe, None)

        # image is a PySpin ImagePtr that must match the characteristics passed to the SharedImageSender constructor
        #   image = cam.GetNextImage()
        if image is not None:
            imarray = image.GetNDArray()

        try:
            # Write the numpy array data buffer to the pipe
            win32file.WriteFile(self.currentPipe, bytes(imarray.data))
        except:
            pass
        self.framesLeftInChunk -= 1

        if self.metadataQueue is not None:
            try:
                self.metadataQueue.put(metadata, block=False)
            except qFull:
                if not self.allowOverflow:
                    raise qFull('{name}: Metadata queue overflow. qsize={qsize} max={maxsize}'.format(name=self.pipeName, qsize=self.qsize(), maxsize=self.bufferSize))
                elif self.verbose >= 2:
                    print('{name}: Warning, metadata queue full. Overflow allowed - continuing...'.format(name=self.pipeName))

    def close(self, force=False):
        if force:
            # Rude - close pipes regardless of whether they've been read or are
            #   being read or not
            for pipeHandle in self.pipes:
                win32file.CloseHandle(pipeHandle)
        else:
            # Wait for all pipes to be done, then close
            while len(self.pipes) > 0:
                self.pipeReadyQueue.close()
                self.pipeReadyQueue.join_thread()
                self.cleanupPipes(block=True, timeout=1)
            self.pipeDoneQueue.close()
            self.pipeDoneQueue.join_thread()

class SharedImageReceiver():
    def __init__(self,
                width,
                height,
                channels,
                pixelFormat,
                frameSize,
                chunkFrameCount,
                verbose=0,
                offsetX=0,
                offsetY=0,
                imageDataType=ctypes.c_uint8,
                outputType='PySpin',
                fileWriter=None,
                bufferSize=1,
                metadataQueue=None,
                pipeReadyQueue=None,
                pipeDoneQueue=None,
                ):
        self.width = width
        self.height = height
        self.pixelFormat = pixelFormat
        self.offsetX = offsetX
        self.offsetY = offsetY
        self.frameSize = frameSize
        self.chunkFrameCount = chunkFrameCount
        self.framesLeftInChunk = self.chunkFrameCount
        self.bufferSize = bufferSize
        self.outputType = outputType
        self.fileWriter = fileWriter
        self.imageDataType = imageDataType
        self.channels = channels
        self.metadataQueue = metadataQueue
        self.verbose = verbose
        self.nextID = 0
        self.pipePath = None
        self.pipeHandle = None
        self.pipeConnected = False
        self.pipeReadyQueue = pipeReadyQueue
        self.pipeDoneQueue = pipeDoneQueue

        if self.channels > 1:
            self.frameShape = (self.height, self.width, self.channels)
        else:
            self.frameShape = (self.height, self.width)

    def qsize(self):
        return (-1, -1) # (main queue, metadata queue)

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
            raise KeyError('Unrecognized output type: {ot}'.format(ot=self.outputType))

        # If given, apply the fileWriter function to the output
        if self.fileWriter:
            self.fileWriter(output)
            if self.verbose >= 3: print('Applied filewriter!')
        return output

    def connectPipe(self, block=False, timeout=None):
        if self.pipeHandle is not None:
            self.pipeHandle.close()
            self.pipeDoneQueue.put(self.pipePath)
            self.pipePath = None
            self.pipeHandle = None
        try:
            pipeInfo = self.pipeReadyQueue.get(block=block, timeout=timeout)
        except qEmpty:
            raise TimeoutError('No pipe available for connecting')
        self.pipePath = pipeInfo['path']
        self.pipeHandle = open(self.pipePath, 'rb')

    def get(self, includeMetadata=False, maxConnectAttempts=5, connectTimeout=1):
        # Returns a PySpin image, if one is available in the buffer, otherwise
        #   raise queue.Empty
        if self.framesLeftInChunk < 0:
            raise RunTimeError('Something went very wrong with image receiver')
        elif self.framesLeftInChunk == 0 or self.pipeHandle is None:
            # We need to get a new pipe to connect to
            remainingConnectionAttempts = maxConnectAttempts
            while remainingConnectionAttempts is not None and remainingConnectionAttempts > 0:
                try:
                    # Attempt to get a new pipe to connect to
                    self.connectPipe(block=True, timeout=connectTimeout)
                    # Attempt succeeded - stop trying and move on
                    break
                except TimeoutError:
                    # Attempt failed
                    if self.verbose > 2:
                        print('Attempt #{k} to get a new pipe to connect to failed'.format(k=k))
                    remainingConnectionAttempts -= 1
            if self.pipePath is None:
                # All attempts failed
                raise TimeoutError('Failed to get a new pipe to connect to')

            # Reset frame count for new pipe
            self.framesLeftInChunk = self.chunkFrameCount

        # Get next frame from open pipe
        output = self.pipeHandle.read(self.frameSize)
        # Decrement frame count for this pipe
        self.framesLeftInChunk -= 1

        if self.metadataQueue is not None:
            # Get the associated metadata
            metadata = self.metadataQueue.get(block=False)
        else:
            metadata = None

        if includeMetadata:
            if self.metadataQueue is None:
                raise IOError('Cannot include metadata, as metadata queue does not exist.')
            return output, metadata
        else:
            return output

    def closePipe(self):
        self.pipeConnected = False
        self.pipeHandle.close()

class ThreadedReader(th.Thread):
    def __init__(self, stream, *args, **kwargs):
        th.Thread.__init__(self, *args, **kwargs)
        self.stream = stream
        self.data = None

        self.dataSize = None
        self.exception = None
    def run(self):
        try:
            if self.dataSize is None:
                self.data = self.stream.read()
            else:
                self.data = self.stream.read(self.dataSize)
        except Exception as e:
            self.exception = e

    def read(self, size=None, block=False, timeout=1, abortOnTimeout=False):
        if self.data is None:
            self.dataSize = size
            self.start()
            if block:
                # User requests blocking call
                self.join(timeout=timeout)
                if self.is_alive():
                    # Read is not done, it timed out
                    if abortOnTimeout:
                        # Close the stream
                        self.stream.close()
                    raise TimeoutError('Read did not complete before timeout')
                else:
                    # Read finished, return the data
                    data = self.data
            else:
                # We did not wait for read, so return None
                data = None
        else:
            # Data already has been read - clear the field and return it
            data = self.data
            self.data = None

        if self.exception is not None:
            # Check if the thread encountered an exception
            exception = self.exception
            self.exception = None
            raise exception

        return data
