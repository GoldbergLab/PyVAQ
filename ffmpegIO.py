import subprocess
import shutil
import warnings
from numpy import frombuffer, reshape

FFMPEG_EXE = shutil.which('ffmpeg')
FFPROBE_EXE = shutil.which('ffprobe')

DEFAULT_CPU_COMPRESSION_ARGS = [
    '-c:v', 'libx264', '-preset', 'fast', '-crf', '23'
    ]
DEFAULT_GPU_COMPRESSION_ARGS = [
    '-c:v', 'h264_nvenc', '-preset', 'fast', '-cq', '32'
    ]

class ffmpegVideoReader():
    def __init__(self, filename, verbose=1):
        self.filename = filename

        if verbose <= 0:
            self.ffmpegVerbosity = 'quiet'
        elif verbose == 1:
            self.ffmpegVerbosity = 'error'
        elif verbose == 2:
            self.ffmpegVerbosity = 'warning'
        elif verbose >= 3:
            self.ffmpegVerbosity = 'verbose'

        self.ffmpegProc = None

    def getVideoInfo(self):
        ffprobeCommand = [FFPROBE_EXE,
            '-v', 'error',
            '-select_streams', 'v:0',
            '-show_entries', 'stream',
            '-of', 'default=nokey=0:noprint_wrappers=1',
            self.filename]
        with subprocess.Popen(ffprobeCommand, stdout=subprocess.PIPE) as p:
            rawVideoInfo, err = p.communicate()
        rawVideoInfo = rawVideoInfo.decode('utf-8').strip().split('\r\n')
        videoInfo = dict([keyValue.split('=') for keyValue in rawVideoInfo])
        return videoInfo

    def getVideoSize(self):
        videoInfo = self.getVideoInfo()
        numFrames = int(videoInfo['nb_frames'])
        width = int(videoInfo['width'])
        height = int(videoInfo['height'])
        return [numFrames, height, width]

    def read(self, outputType='numpy', startFrame=None, endFrame=None):
        # outputType must be 'bytes' or 'numpy'
        # frame indices 1 indexed
        videoSize = self.getVideoSize()
        if startFrame is None:
            startFrame = 1
        if endFrame is None:
            endFrame = videoSize[0]

        frameByteCount = videoSize[1]*videoSize[2]*3
        startByte = (startFrame-1)*frameByteCount
        endByte = endFrame*frameByteCount
        numFrames = endFrame - startFrame + 1

        ffmpegCommand = [FFMPEG_EXE, '-y', '-v', self.ffmpegVerbosity,
            '-i', '"'+self.filename+'"', '-f', 'rawvideo', '-c:v', 'rawvideo', '-pix_fmt', 'rgb24', '-an', '-']
        with subprocess.Popen(' '.join(ffmpegCommand), stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=False) as self.ffmpegProc:
            self.ffmpegProc.stdout.seek(startByte)
            frameBytes = self.ffmpegProc.stdout.read(endByte-startByte)
            self.ffmpegProc.kill()
            err = self.ffmpegProc.stderr.read()

        if outputType == 'bytes':
            return frameBytes
        elif outputType == 'numpy':
            frames = frombuffer(frameBytes, dtype='uint8')
            outputVideoSize = [numFrames, videoSize[1], videoSize[2], 3]
            return reshape(frames, outputVideoSize)
        else:
            raise SyntaxError('Unknown output type: {ot}'.format(ot=outputType))

class ffmpegVideoWriter():
    def __init__(self, filename, frameType="bytes", verbose=1, fps=30, shape=None,
                input_pixel_format="bayer_rggb8", output_pixel_format="rgb0",
                gpuVEnc=False, gpuCompressionArgs=DEFAULT_GPU_COMPRESSION_ARGS,
                cpuCompressionArgs=DEFAULT_CPU_COMPRESSION_ARGS):
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
        self.gpuCompressionArgs = gpuCompressionArgs
        self.cpuCompressionArgs = cpuCompressionArgs

    def getFFMPEGVerbosity(self):
        if self.verbose <= 0:
            ffmpegVerbosity = 'quiet'
        elif self.verbose == 1:
            ffmpegVerbosity = 'error'
        elif self.verbose == 2:
            ffmpegVerbosity = 'warning'
        elif self.verbose >= 3:
            ffmpegVerbosity = 'verbose'
        return ffmpegVerbosity

    def initializeFFMPEG(self, shape):
        # Initialize a new ffmpeg process

        w, h = shape
        shapeArg = '{w}x{h}'.format(w=w, h=h)

        if self.verbose >= 3:
            print("STARTING NEW FFMPEG VIDEO PROCESS!")

        ffmpegVerbosity = self.getFFMPEGVerbosity()

        if self.gpuVEnc:
            # With GPU acceleration
            ffmpegCommand = [FFMPEG_EXE, '-y', '-probesize', '32', '-flush_packets', '1',
                '-hwaccel', 'cuda', '-hwaccel_output_format', 'cuda',
                '-v', ffmpegVerbosity, '-f', 'rawvideo', '-c:v', 'rawvideo',
                '-pix_fmt', self.input_pixel_format, '-s', shapeArg, '-thread_queue_size', '128',
                '-r', str(self.fps), '-i', '-', *self.gpuCompressionArgs,
                '-pix_fmt', self.output_pixel_format, '-fps_mode', 'passthrough', '-an',
                self.filename]
        else:
            # Without GPU acceleration
            ffmpegCommand = [FFMPEG_EXE, '-y', '-probesize', '32', '-flush_packets', '1',
                '-v', ffmpegVerbosity, '-f', 'rawvideo',
                '-c:v', 'rawvideo', '-pix_fmt', self.input_pixel_format,
                '-s', shapeArg, '-r', str(self.fps), '-thread_queue_size', '128',
                 '-i', '-', *self.cpuCompressionArgs,
                '-pix_fmt', self.output_pixel_format, '-fps_mode', 'passthrough', '-an',
                self.filename]

        if self.verbose >= 2:
            print('ffmpeg command:')
            print(ffmpegCommand)
        try:
            self.ffmpegProc = subprocess.Popen(ffmpegCommand, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL)
        except TypeError:
            raise OSError('Error starting ffmpeg process - check that ffmpeg is present on this system and included in the system PATH variable')

    def getFrameShape(self, frame, shape=None):
        # Ensure we know the shape of the frame
        if shape is None:
            # No shape given
            if self.shape is None:
                # Shape not set in constructor
                if self.frameType == 'image':
                    # Determine shape from PySpin object
                    shape = frame.size
                elif self.frameType == 'numpy':
                    if len(frame.shape) == 1:
                        # Ok, this is flattened, can't really deduce the resolution
                        raise TypeError("For flattened arrays, the shape parameter must be passed in")
                    else:
                        # Determine shape from numpy array
                        shape = [frame.shape[1], frame.shape[0]]
                else:
                    raise TypeError("You must provide width and height for a bytearray frame format")
            else:
                # Use shape passed in at constructor
                shape = self.shape
        if self.verbose >= 2: print('Frame shape: {s}'.format(s=str(shape)))
        return shape

    def write(self, frame, shape=None):
        # Send a video frame to an ffmpeg process. If a ffmpeg process does not
        #   currently exist, start a new one
        #
        # frame should be an RGB PIL image
        #   or a numpy array (of the format returned by calling
        #   np.asarray(image) on a RGB PIL image
        # All frames should be the same size and format
        # If shape is given (as a (width, height) tuple), it will be used. If
        #   not, we will try to figure out the image shape.

        if self.frameType == 'image':
            bytes = frame.tobytes()
        elif self.frameType == 'numpy':
            bytes = frame.data
        elif self.frameType == 'bytes':
            bytes = frame
        if self.verbose >= 3:
            print('Sending frame to ffmpeg!')

        if self.ffmpegProc is None:
            # Time to initialize a new ffmpeg process
            shape = self.getFrameShape(frame, shape=shape)
            self.initializeFFMPEG(shape)

        # Write this frame
        self.ffmpegProc.stdin.write(bytes)    #'raw', 'RGB'))
        self.ffmpegProc.stdin.flush()

    def close(self):
        if self.ffmpegProc is not None:
            self.ffmpegProc.kill()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.ffmpegProc = None
            if self.verbose >= 2:
                print('Closed pipe to ffmpeg')

class ffmpegPipedVideoWriter(ffmpegVideoWriter):
    def __init__(self, filename, pipePath, **kwargs):
        super().__init__(filename, **kwargs)
        self.pipePath = pipePath
        if not self.frameType == "bytes":
            raise ValueError('Piped video writers can only accomodate "bytes" frame type')

    def initializeFFMPEG(self, shape):
        # Initialize a new ffmpeg process

        w, h = shape
        shapeArg = '{w}x{h}'.format(w=w, h=h)

        if self.verbose >= 3:
            print("STARTING NEW FFMPEG VIDEO PROCESS!")

        ffmpegVerbosity = self.getFFMPEGVerbosity()

        if self.gpuVEnc:
            # With GPU acceleration
            ffmpegCommand = [FFMPEG_EXE, '-y', '-probesize', '32', '-flush_packets', '1',
                '-progress', '-', '-stats_period', '0.1',
                '-hwaccel', 'cuda', '-hwaccel_output_format', 'cuda',
                '-v', ffmpegVerbosity, '-f', 'rawvideo', '-c:v', 'rawvideo',
                '-pix_fmt', self.input_pixel_format, '-s', shapeArg, '-thread_queue_size', '128',
                '-r', str(self.fps), '-i', self.pipePath, *self.gpuCompressionArgs,
                '-pix_fmt', self.output_pixel_format, '-fps_mode', 'passthrough', '-an',
                self.filename]
        else:
            # Without GPU acceleration
            ffmpegCommand = [FFMPEG_EXE, '-y', '-probesize', '32', '-flush_packets', '1',
                '-progress', '-', '-stats_period', '0.1',
                '-v', ffmpegVerbosity, '-f', 'rawvideo',
                '-c:v', 'rawvideo', '-pix_fmt', self.input_pixel_format,
                '-s', shapeArg, '-r', str(self.fps), '-thread_queue_size', '128',
                 '-i', self.pipePath, *self.cpuCompressionArgs,
                '-pix_fmt', self.output_pixel_format, '-fps_mode', 'passthrough', '-an',
                self.filename]

        if self.verbose >= 2:
            print('ffmpeg command:')
            print(ffmpegCommand)
        try:
            self.ffmpegProc = subprocess.Popen(ffmpegCommand, stdin=None, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        except TypeError:
            raise OSError('Error starting ffmpeg process - check that ffmpeg is present on this system and included in the system PATH variable')

    def currentFrameNumber(self):
        # Check the frame number the writer has most recently received
        if self.ffmpegProc is None:
            raise IOError('ffmpeg process does not currently exist')

        print('Checking current frame #')
        self.ffmpegProc.stderr.flush()
        data = self.ffmpegProc.stderr.read(8)
        print("PROGRESS:")
        print(data)

    def checkProgress(self, timeout=1):
        # Doesn't work, ffmpeg buffers stderr until it exist :()
        try:
            out, err = self.ffmpegProc.communicate(timeout=timeout)
            print('out:')
            print(out)
            print('err:')
            print(err)
            print('return code:')
            print(self.ffmpegProc.returncode)
        except subprocess.TimeoutExpired:
            print('Communicate timeout expired after', timeout)

    def wait(self):
        if self.ffmpegProc is None:
            return

        out, err = self.ffmpegProc.communicate()
        print('out:')
        print(out)
        print('err:')
        print(err)
        print('return code:')
        print(self.ffmpegProc.returncode)

    def write(self, frame, shape=None):
        # This method has no purpose for a piped writer, as the data will flow
        #   directly from the source through the pipe to the ffmpeg process
        raise SyntaxError('write method should not be called for a piped video writer. Writing occurs automatically as data is made available on the named pipe')



class ffmpegAudioWriter():
    # Thanks to https://github.com/Zulko/moviepy/blob/master/moviepy/audio/io/ffmpeg_audiowriter.py
    # for the FFMPEG recipe
    def __init__(self, filename, verbose=1, sampleRate=40000):
        # You can specify the image shape at initialization, or when you write
        #   the first frame (the shape parameter is ignored for subsequent
        #   frames), or not at all, and hope we can figure it out.
        # dataType should be one of 'numpy', 'image', or 'bytes'
        self.ffmpegProc = None
        self.verbose = verbose
        self.sampleRate = sampleRate
        self.filename = filename
        # nBytes and nChannels will be determined by the first audio chunk provided.
        self.nBytes = None
        self.nChannels = None

    def write(self, data):
        # data should be a N x C numpy array, where N is the # of samples, and C is the # of channels
        # All data chunks should have the same number of channels
        if self.ffmpegProc is None:
            if self.verbose >= 3:
                print("STARTING NEW FFMPEG AUDIO PROCESS!")

            # Determine FFMPEG verbosity level
            if self.verbose <= 0:
                ffmpegVerbosity = 'quiet'
            elif self.verbose == 1:
                ffmpegVerbosity = 'error'
            elif self.verbose == 2:
                ffmpegVerbosity = 'warning'
            elif self.verbose >= 3:
                ffmpegVerbosity = 'verbose'

            # Gather info about data type
            self.nBytes = data.itemsize
            self.nChannels = data.shape[-1]

            # Generate FFMPEG command
            ffmpegCommand = [
                FFMPEG_EXE,
                '-y',
                '-v', ffmpegVerbosity,
                '-f', 's{b}le'.format(b=8*self.nBytes),
                '-c:a', 'pcm_s{b}le'.format(b=8*self.nBytes),
                '-ar', '{r}'.format(r=self.sampleRate),
                '-ac', '{c}'.format(c=self.nChannels),
                '-thread_queue_size', '128',
                '-i', '-',
                '-thread_queue_size', '128',
                self.filename
                ]

            if self.verbose >= 2:
                print('ffmpeg command:')
                print(ffmpegCommand)

            try:
                self.ffmpegProc = subprocess.Popen(ffmpegCommand, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL)
            except TypeError:
                raise OSError('Error starting ffmpeg process - check that ffmpeg is present on this system and included in the system PATH variable')

        # Convert array to bytes
        bytes = data.tobytes()
        if self.verbose >= 3:
            print('Sending frame to ffmpeg!')

        # Pipe data to ffmpeg
        self.ffmpegProc.stdin.write(bytes)
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
