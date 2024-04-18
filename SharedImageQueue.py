import numpy as np
import multiprocessing as mp
from queue import Empty as qEmpty
from queue import Full as qFull
import win32pipe, win32file, pywintypes
import ctypes
import time
import queue

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
                maxBufferSize=100,                  # Maximum size for metadata queue
                pipeBaseName=None,                  # Name for named pipe
                allowOverflow=False,                # Should an error be raised if the queue is filled up, or should old entries be overwritten?
                createReceiver=False,
                includeMetadata=True,
                chunkFrameCount=300,                # Number of frames per named pipe
                allowNonconsecutiveFrames=False
                ):

        if pipeBaseName is None:
            pipeBaseName = 'PyVAQ_fifo_{t}'.format(t=int(time.time()*10))
        else:
            # Add timestamp to ensure the name is not in use
            pipeBaseName = pipeBaseName + '_' + str(int(time.time()*10))
        self.pipeBaseName = pipeBaseName
        self.pipeBasePath = getPipePath(self.pipeBaseName)
        self.verbose = verbose
        self.maxBufferSize = maxBufferSize
        if self.maxBufferSize < 1:
            raise ValueError("{name}: maxBufferSize must be greater than 1".format(name=self.pipeBaseName))
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

        if includeMetadata:
            self.metadataQueue = mp.Queue(maxsize=maxBufferSize)
        else:
            self.metadataQueue = None

        self.chunkFrameCount = chunkFrameCount
        self.framesLeftInChunk = 0

        self.lastFrameIndex = None

        self.allowNonconsecutiveFrames = allowNonconsecutiveFrames

        if createReceiver:
            # If user requests a receiver object, create it.
            # Otherwise the user can use the named pipe directly
            self.receiver = SharedImageReceiver(
                pipeBaseName=self.pipeBaseName,
                width=self.width,
                height=self.height,
                channels=self.channels,
                pixelFormat=pixelFormat,
                frameSize=self.width*self.height*self.channels*(imageBitsPerPixel//8),
                verbose=self.verbose,
                offsetX=offsetX,
                offsetY=offsetY,
                outputType=outputType,
                fileWriter=fileWriter,
                imageDataType=imageDataType,
                lockForOutput=lockForOutput,
                maxBufferSize = self.maxBufferSize,
                metadataQueue=self.metadataQueue,
                pipeReady=self.pipeReady)
        else:
            self.receiver = None

    def qsize(self):
        return (0, 0) # (main queue, metadata queue)

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
                65536,                                                      # nOutBufferSize
                65536,                                                      # nInBufferSize
                300,                                                        # nDefaultTimeOut
                None)                                                       # security attributes
        print('Created new pipe: {p}'.format(p=pipePath))
        # Add new pipe handle to the end of the list
        self.pipes.append({'path':pipePath, 'index':pipeIndex, 'pipe':pipe, 'frameRange':frameRange})
        # Reset frame count for new pipe
        self.framesLeftInChunk = self.chunkFrameCount
        # Inform client of the new pipe path and index so it can use the pipe,
        #   then inform the server which pipe it's done reading from
        self.pipeReadyQueue.put({'path':pipePath, 'index':pipeIndex, 'frameRange':frameRange})
        # Set the current pipe for writing
        self.currentPipe = pipe
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
        self.pipes.pop(pipeInfoLoction)

    def connectPipe(self):
        if self.pipes is None:
            print("SETUP NAMED PIPE...")
            self.setupNamedPipe()
            print("...DONE")
        if not self.pipeConnected:
            print('CONNECT NAMED PIPE')
            win32pipe.ConnectNamedPipe(self.pipe, None)
            print('...DONE')
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
            # Wait for client to connect to pipe
            win32pipe.ConnectNamedPipe(self.currentPipe, None)

        print('Sending image #{f}'.format(f=frameIndex))

        # image is a PySpin ImagePtr that must match the characteristics passed to the SharedImageSender constructor
        #   image = cam.GetNextImage()
        if image is not None:
            imarray = image.GetNDArray()

        print('Writing data: len=', len(bytes(imarray.data)), 'shape=', imarray.shape)
        # Write the numpy array data buffer to the pipe
        win32file.WriteFile(self.currentPipe, bytes(imarray.data))
        self.framesLeftInChunk -= 1

        if self.metadataQueue is not None:
            try:
                self.metadataQueue.put(metadata, block=False)
            except qFull:
                if not self.allowOverflow:
                    raise qFull('{name}: Metadata queue overflow. qsize={qsize} max={maxsize}'.format(name=self.pipeName, qsize=self.qsize(), maxsize=self.maxBufferSize))
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
            while True:
                self.pipeReadyQueue.close()
                self.pipeReadyQueue.join_thread()
                try:
                    self.cleanupPipes(block=True, timeout=1)
                except TimeoutError:
                    print('Still waiting for reader to consume data in {c} pipes'.format(c=len(self.pipes)))
            self.pipeDoneQueue.close()
            self.pipeDoneQueue.join_thread()
            print('Image queue closed')

class SharedImageReceiver():
    def __init__(self,
                pipeName,
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
                pipeReady=None
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
        self.pipeName = pipeName
        self.pipePath = getPipePath(self.pipeName)
        self.pipeHandle = None
        self.pipeConnected = False
        self.pipeReady = pipeReady

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

    def connectPipe(self, timeout=None):
        ready = self.pipeReady.wait(timeout=timeout)
        if not ready:
            # Sender did not indicate pipe was ready in time
            raise mp.TimeoutError('Pipe did not become available before the timeout period elapsed')
        success = False
        numAttempts = 10
        for k in range(numAttempts):
            try:
                self.pipeHandle = open(self.pipePath, 'rb')
                success = True
                break
            except Exception as e:
                print(e)
                success = False
                print('Failed to connect to pipe...')
                time.sleep(1)
        if not success:
            raise IOError("broken pipe, bye bye")
        else:
            self.pipeConnected = True

    def get(self, includeMetadata=False):
        # Returns a PySpin image, if one is available in the buffer, otherwise
        #   raise queue.Empty

        if not self.pipeConnected or self.pipeHandle is None or not self.pipeReady.is_set():
            # Pipe not set up - do it now
            self.connectPipe()
        try:
            print('framesize=', self.frameSize)
            output = self.pipeHandle.read(self.frameSize)
        except:
            raise IOError('pipe read failed')

        if self.metadataQueue is not None:
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
