import subprocess

class ffmpegWriter():
    def __init__(self, filename, frameType, fps=30, shape=None):
        # You can specify the image shape at initialization, or when you write
        #   the first frame (the shape parameter is ignored for subsequent
        #   frames), or not at all, and hope we can figure it out.
        # frameType should be one of 'numpy', 'image', or 'bytes'
        self.ffmpegProc = None
        self.fps = fps
        self.filename = filename
        self.shape = shape
        self.frameType = frameType

    def write(self, frame, shape=None):
        # frame should be an RGB PIL image
        # or a numpy array (of the format returned by calling np.asarray(image) on a RGB PIL image
        # All frames should be the same size and format
        # If shape is given (as a (width, height) tuple), it will be used. If not, we will try to figure out the image shape.
        if self.ffmpegProc is None:
            print("STARTING NEW FFMPEG PROCESS!")
            if shape is None and self.shape is None:
                if self.frameType == 'image':
                    w, h = frame.size
                elif self.frameType == 'numpy':
                    if len(frame.shape) == 1:
                        # Ok, this is flattened, can't really deduce the resolution
                        raise TypeError("For flattened arrays, the shape parameter must be passed in")
                    else:
                        w = frame.shape[1]
                        h = frame.shape[0]
                else:
                    raise TypeError("You must provide width and height for a bytearray frame format")
            else:
                if shape is None:
                    shape = self.shape
                w, h = shape
            shapeArg = '{w}x{h}'.format(w=w, h=h)
            self.ffmpegProc = subprocess.Popen(['ffmpeg', '-hide_banner', '-y', '-f', '-v', 'error', 'rawvideo', '-pix_fmt', 'rgb24', '-s', shapeArg, '-r', str(self.fps), '-i', 'pipe:', '-c:v', 'libx264', '-an', self.filename], stdin=subprocess.PIPE, stdout=subprocess.PIPE)
        if self.frameType == 'image':
            bytes = frame.tobytes()
        elif self.frameType == 'numpy':
            bytes = frame.tobytes()
        elif self.frameType == 'bytes':
            bytes = frame
        self.ffmpegProc.stdin.write(bytes)    #'raw', 'RGB'))

    def close(self):
        if self.ffmpegProc is not None:
            self.ffmpegProc.communicate()
