from SharedImageQueue import SharedImageSender
import time
import numpy as np
import os

pipeName = 'pipe_test'

image = np.random.rand(1000, 1000)

imageQueue = SharedImageSender(
    width=1000,
    height=1000,
    verbose=1,
    pixelFormat="bayer_rggb8",
    outputType='bytes',
    channels=3,
    pipeName=pipeName,
    includeMetadata=False,
    # createReceiver=False
)

imageQueue.setupNamedPipe()
f = open(imageQueue.pipePath, 'rb')
print('pipe connected! #1')
imageQueue.connectPipe()
f.close()
try:
    print('Pipe exists?:', os.path.isfile(imageQueue.pipePath))
    f = open(imageQueue.pipePath, 'rb')
    print('Opened pipe #1')
    f.close()
except OSError as e:
    print('no pipe yet')
imageQueue.setupNamedPipe()
print('Pipe exists?:', os.path.isfile(imageQueue.pipePath))
f = open(imageQueue.pipePath, 'rb')

print('pipe connected! #2')
