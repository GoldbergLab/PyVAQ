
class PickleableImage():
    def __init__(self, width, height, offsetX, offsetY, pixelFormat, data, frameTime):
        self.width = width
        self.height = height
        self.offsetX = offsetX
        self.offsetY = offsetY
        self.pixelFormat = pixelFormat
        self.data = data
        self.frameTime = frameTime


class SharedPickleableImage():
    def __init__(self, width, height, offsetX, offsetY, pixelFormat, data, frameTime):
        self.width = width
        self.height = height
        self.offsetX = offsetX
        self.offsetY = offsetY
        self.pixelFormat = pixelFormat
        self.data = data
        self.frameTime = frameTime
