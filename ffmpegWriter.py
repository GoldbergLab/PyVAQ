import subprocess
import shutil
import warnings

FFMPEG_EXE = shutil.which('ffmpeg')

class ffmpegWriter():
    def __init__(self, filename, frameType, verbose=1, fps=30, shape=None, input_pixel_format="bayer_rggb8", output_pixel_format="rgb0", gpuVEnc=False):
        # You can specify the image shape at initialization, or when you write
        #   the first frame (the shape parameter is ignored for subsequent
        #   frames), or not at all, and hope we can figure it out.
        # frameType should be one of 'numpy', 'image', or 'bytes'
        self.ffmpegProc = None
        self.verbose = verbose
        self.fps = fps
        self.filename = filename
        self.shape = shape
        self.frameType = frameType
        self.input_pixel_format = input_pixel_format
        self.output_pixel_format = output_pixel_format
        self.gpuVEnc = gpuVEnc

    def write(self, frame, shape=None):
        # frame should be an RGB PIL image
        # or a numpy array (of the format returned by calling np.asarray(image) on a RGB PIL image
        # All frames should be the same size and format
        # If shape is given (as a (width, height) tuple), it will be used. If not, we will try to figure out the image shape.
        if self.ffmpegProc is None:
            if self.verbose >= 3:
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
#            self.ffmpegProc = subprocess.Popen([FFMPEG_EXE, '-hide_banner', '-y', '-v', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb8', '-s', shapeArg, '-r', str(self.fps), '-i', 'pipe:', '-c:v', 'libx264', '-tune', 'zerolatency', '-preset', 'ultrafast', '-an', self.filename], stdin=subprocess.PIPE, stdout=subprocess.DEVNULL)

            if self.verbose <= 0:
                ffmpegVerbosity = 'quiet'
            elif self.verbose == 1:
                ffmpegVerbosity = 'error'
            elif self.verbose == 2:
                ffmpegVerbosity = 'warning'
            elif self.verbose >= 3:
                ffmpegVerbosity = 'verbose'

            if self.gpuVEnc:
                # With GPU acceleration
                ffmpegCommand = [FFMPEG_EXE, '-y',
                    '-vsync', 'passthrough', '-hwaccel', 'cuda', '-hwaccel_output_format', 'cuda',
                    '-v', ffmpegVerbosity, '-f', 'rawvideo', '-c:v', 'rawvideo',
                    '-pix_fmt', self.input_pixel_format, '-s', shapeArg, '-thread_queue_size', '1024',
                    '-r', str(self.fps), '-i', '-', '-c:v', 'h264_nvenc', '-preset', 'fast',
                    '-cq', '32',
                    '-pix_fmt', self.output_pixel_format, '-an', self.filename]
            else:
                # Without GPU acceleration
                ffmpegCommand = [FFMPEG_EXE, '-y',
                    '-vsync', 'passthrough', '-v', ffmpegVerbosity, '-f', 'rawvideo',
                    '-c:v', 'rawvideo', '-pix_fmt', self.input_pixel_format,
                    '-s', shapeArg, '-r', str(self.fps), '-thread_queue_size', '1024',
                     '-i', '-',
                    '-c:v', 'libx264', '-preset', 'fast', '-crf', '23',
                    '-pix_fmt', self.output_pixel_format, '-an', self.filename]
            # print('Command:')
            # print(ffmpegCommand)
            if self.verbose >= 2:
                print('ffmpeg command:')
                print(ffmpegCommand)
            self.ffmpegProc = subprocess.Popen(ffmpegCommand, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL)
            # self.ffmpegProc = subprocess.Popen([FFMPEG_EXE, '-hide_banner', '-y',
            #     '-v', 'error', '-f', 'rawvideo', '-vcodec', 'rawvideo',
            #     '-pix_fmt', self.input_pixel_format, '-s', shapeArg,
            #     '-r', str(self.fps), '-i', '-', '-vcodec', 'mpeg4',
            #     '-pix_fmt', self.output_pixel_format, '-an', self.filename], stdin=subprocess.PIPE, stdout=subprocess.DEVNULL)
        if self.frameType == 'image':
            bytes = frame.tobytes()
        elif self.frameType == 'numpy':
            bytes = frame.tobytes()
        elif self.frameType == 'bytes':
            bytes = frame
        if self.verbose >= 3:
            print('Sending frame to ffmpeg!')
        self.ffmpegProc.stdin.write(bytes)    #'raw', 'RGB'))
        self.ffmpegProc.stdin.flush()

    def close(self):
        if self.ffmpegProc is not None:
            self.ffmpegProc.stdin.close()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.ffmpegProc = None
            if self.verbose >= 2:
                print('Closed pipe to ffmpeg')
#            self.ffmpegProc.communicate()



# nvenc lossless (~2.5 sec / 100 frames)
# ffmpeg -benchmark -f rawvideo -s 3208x2200 -pix_fmt bgr24 -i G:\testVideos\videoWriteTest_000.raw -c:v nvenc -lossless G:\testVideos\converted_lossless_nvenc.avi

# fastest libx264 (~4.5 sec / 100 frames)
# ffmpeg -benchmark -f rawvideo -s 3208x2200 -pix_fmt bgr24 -i G:\testVideos\videoWriteTest_000.raw -c:v libx264 -crf 0 -preset "ultrafast" G:\testVideos\converted_lossless.avi

# List GPUs:
# ffmpeg -f lavfi -i nullsrc -c:v h264_nvenc -gpu list -f null -

# List encoder options
# ffmpeg -hide_banner -h encoder=hevc_nvenc
