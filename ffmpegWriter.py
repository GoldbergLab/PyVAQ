import subprocess
import shutil
import warnings

FFMPEG_EXE = shutil.which('ffmpeg')

DEFAULT_CPU_COMPRESSION_ARGS = [
    '-c:v', 'libx264', '-preset', 'fast', '-crf', '23'
    ]
DEFAULT_GPU_COMPRESSION_ARGS = [
    '-c:v', 'h264_nvenc', '-preset', 'fast', '-cq', '32'
    ]

def buildCPUCompressionArgs(crf):
    """Build libx264 (CPU) ffmpeg compression args for a given CRF quality.

    Lower CRF = higher quality / larger files. Valid range is roughly 0-51.
    """
    return ['-c:v', 'libx264', '-preset', 'fast', '-crf', str(crf)]

def buildGPUCompressionArgs(cq):
    """Build h264_nvenc (GPU) ffmpeg compression args for a given CQ quality.

    Lower CQ = higher quality / larger files. Valid range is roughly 0-51.
    """
    return ['-c:v', 'h264_nvenc', '-preset', 'fast', '-cq', str(cq)]

def _defaultQualityValue(args, flag, fallback):
    # Extract the quality value following the given flag in a default args list,
    #   so callers can derive a default CQ/CRF without duplicating the literal.
    try:
        return int(args[args.index(flag) + 1])
    except (ValueError, IndexError):
        return fallback

def defaultCQ():
    """Default nvenc CQ quality, taken from DEFAULT_GPU_COMPRESSION_ARGS."""
    return _defaultQualityValue(DEFAULT_GPU_COMPRESSION_ARGS, '-cq', 23)

def defaultCRF():
    """Default libx264 CRF quality, taken from DEFAULT_CPU_COMPRESSION_ARGS."""
    return _defaultQualityValue(DEFAULT_CPU_COMPRESSION_ARGS, '-crf', 23)

class ffmpegWriter():
    def __init__(
        self,
        filename: str,
        frameType: str,
        verbose: int = 1,
        fps: int = 30,
        shape = None,
        input_pixel_format: str = "bayer_rggb8",
        output_pixel_format: str = "rgb0",
        gpuVEnc: bool = False,
        gpuCompressionArgs: list = DEFAULT_GPU_COMPRESSION_ARGS,
        cpuCompressionArgs: list = DEFAULT_CPU_COMPRESSION_ARGS
    ):
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
        # Fall back to the module defaults if no args are supplied, so callers
        #   can safely pass None.
        self.gpuCompressionArgs = gpuCompressionArgs if gpuCompressionArgs is not None else DEFAULT_GPU_COMPRESSION_ARGS
        self.cpuCompressionArgs = cpuCompressionArgs if cpuCompressionArgs is not None else DEFAULT_CPU_COMPRESSION_ARGS

    def write(self, frame, shape=None):
        # frame should be an RGB PIL image
        #   or a numpy array (of the format returned by calling
        #   np.asarray(image) on a RGB PIL image
        # All frames should be the same size and format
        # If shape is given (as a (width, height) tuple), it will be used. If
        #   not, we will try to figure out the image shape.
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
                    '-pix_fmt', self.input_pixel_format, '-s', shapeArg, '-thread_queue_size', '128',
                    '-r', str(self.fps), '-i', '-', *self.gpuCompressionArgs, '-pix_fmt', self.output_pixel_format, '-an',
                    self.filename]
            else:
                # Without GPU acceleration
                ffmpegCommand = [FFMPEG_EXE, '-y',
                    '-vsync', 'passthrough', '-v', ffmpegVerbosity, '-f', 'rawvideo',
                    '-c:v', 'rawvideo', '-pix_fmt', self.input_pixel_format,
                    '-s', shapeArg, '-r', str(self.fps), '-thread_queue_size', '128',
                     '-i', '-', *self.cpuCompressionArgs,
                    '-pix_fmt', self.output_pixel_format, '-an',
                    self.filename]

            if self.verbose >= 2:
                print('ffmpeg command:')
                print(ffmpegCommand)
            self.ffmpegProc = subprocess.Popen(ffmpegCommand, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL)

        if self.frameType == 'bytes':
            buf = frame
        elif self.frameType == 'image':
            buf = frame.tobytes()
        elif self.frameType == 'numpy':
            # buf = frame.tobytes()
            buf = memoryview(frame)
        if self.verbose >= 3:
            print('Sending frame to ffmpeg!')

        self.ffmpegProc.stdin.write(buf)    #'raw', 'RGB'))

        self.ffmpegProc.stdin.flush()

    def close(self):
        if self.ffmpegProc is not None:
            try:
                # Close stdin to signal end-of-stream to ffmpeg.
                self.ffmpegProc.stdin.close()
            except (BrokenPipeError, OSError):
                pass
            try:
                # Wait for ffmpeg to finish encoding buffered frames and write
                #   the output file trailer. Without this, killing the parent
                #   process (e.g. on a daemon shutdown) can orphan ffmpeg and
                #   leave the output file unfinalized/corrupt.
                self.ffmpegProc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                if self.verbose >= 0:
                    print('ffmpeg did not exit within timeout; killing it.')
                self.ffmpegProc.kill()
                self.ffmpegProc.wait()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.ffmpegProc = None
            if self.verbose >= 2:
                print('Closed pipe to ffmpeg')



# nvenc lossless (~2.5 sec / 100 frames)
# ffmpeg -benchmark -f rawvideo -s 3208x2200 -pix_fmt bgr24 -i G:\testVideos\videoWriteTest_000.raw -c:v nvenc -lossless G:\testVideos\converted_lossless_nvenc.avi

# fastest libx264 (~4.5 sec / 100 frames)
# ffmpeg -benchmark -f rawvideo -s 3208x2200 -pix_fmt bgr24 -i G:\testVideos\videoWriteTest_000.raw -c:v libx264 -crf 0 -preset "ultrafast" G:\testVideos\converted_lossless.avi

# List GPUs:
# ffmpeg -f lavfi -i nullsrc -c:v h264_nvenc -gpu list -f null -

# List encoder options
# ffmpeg -hide_banner -h encoder=hevc_nvenc
