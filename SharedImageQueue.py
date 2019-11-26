import numpy as np
import multiprocessing as mp
import queue

class SharedImageSender():
    def __init__(self,
                width,
                height,
                pixelFormat,
                offsetX=0,
                offsetY=0,
                imageChannels,      # Number of color channels in images (for example, 1 for grayscale, 3 for RGB, 4 for RGBA)
                imageDataType,      # ctype of data of each pixel's channel value
                imageBitsPerPixel,  # Number of bits per pixel
                maxBufferSize       # Maximum number of images to allocate. Attempting to allocate more than that will raise an index error
                ):
        # self.imageChannels = imageChannels
        # self.imageBytes = imageBytes
        self.maxBufferSize = maxBufferSize
        if self.maxBufferSize < 1:
            raise ValueError("maxBufferSize must be greater than 1")
        self.readLag = mp.Value(ctypes.uint32, 0)
        self.nextID =

        imageDataSize = imageDataType, width * height * imageBitsPerPixel // 8
        if imageDataSize > 8000000000:
            print("Warning, super big buffer! Do you have enough RAM? Buffer will be {b} bytes".format(b=imageDataSize))

        self.buffers = []
        for k in range(maxBufferSize):
            # Create a series of shared memory locations, one for each image buffer slot
            self.buffers.append(mp.RawArray(imageDataSize))
            # Create a series of numpy array views into that shared memory location.
            #   The numpy arrays share an underlying memory buffer with the shared memory arrays,
            #   so changing the numpy array changes the shared memory buffer (and vice versa)
            self.npBuffers = np.frombuffer(self.buffers[-1], dtype=ctypes.c_uint8).reshape(height, width)

        self.receiver = SharedImageReceiver(
                width=width,
                height=height,
                pixelFormat=pixelFormat,
                offsetX=offsetX,
                offsetY=offsetY,
                buffers=self.buffers,
                readLag=self.readLag)

    def getReceiver(self):
        return self.receiver

    def getNextID(self):
        nextID = self.nextID
        self.nextID += 1
        return nextID

    def putImage(self, image):
        # image is a PySpin ImagePtr that must match the characteristics passed to the SharedImageSender constructor
        #   image = cam.GetNextImage()
        imarray = image.GetNDArray()
        # Use the numpy array view of the shared memory buffer to copy the image data into shared memory
        with self.readLag.get_lock():
            readLag = self.readLag.value
            if readLag >= self.maxBufferSize:
                raise queue.Full('Reader too far behind writer')
            self.readLag.value = readLag + 1
        nextID = self.getNextID()
        np.copyto(self.npBuffers[nextID % self.maxBufferSize], imarray)


class SharedImageReceiver():
    def __init__(self,
                width,
                height,
                pixelFormat,
                offsetX=0,
                offsetY=0,
                buffers=None,
                readLag=None
                ):
        self.width = width
        self.height = height
        self.pixelFormat = pixelFormat
        self.offsetX = offsetX
        self.offsetY = offsetY
        self.buffers = buffers
        self.readLag = readLag
        self.maxBufferSize = len(self.buffers)
        self.nextID = 0

    def getNextID(self):
        nextID = self.nextID
        self.nextID += 1
        return nextID

    def getImage(self):
        with self.readLag.get_lock():
            readLag = self.readLag.value
            if readLag == 0:
                raise queue.Empty('No images available')
            self.readLag.value = readLag - 1
        nextID = self.getNextID()
        data = self.buffers[nextID % self.maxBufferSize]
        image = PySpin.Image.Create(self.width, self.height, self.offsetX, self.offsetY, self.pixelFormat, data)
        return image
