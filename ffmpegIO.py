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
        """Create a new ffmpeg video reader object.

        Args:
            filename (str): Filename to read video from.
            verbose (verbose): Output verbosity level. Defaults to 1.
        """
        self.filename = str(filename)

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
        """Use ffprobe to get information about the video.

        Returns:
            dict: Dictionary of information about the video.

        """
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
        """Use ffprobe to get the size of the video.

        Returns:
            list of ints: number of frames x height x width
        """
        videoInfo = self.getVideoInfo()
        numFrames = int(videoInfo['nb_frames'])
        width = int(videoInfo['width'])
        height = int(videoInfo['height'])
        return [numFrames, height, width]

    def read(self, outputType='numpy', startFrame=None, endFrame=None):
        """Read the video from file.

        Args:
            outputType (str): Output format - either "bytes" or "numpy".
                Defaults to 'numpy'.
            startFrame (int or None): First frame to read. 1 if left as None
            endFrame (int or None): Last frame to read. Defaults to the entire
                video, if left as None.

        Returns:
            bytes or numpy.ndarray: Frame data, either as bytes or numpy array,
                depending on the "outputType" argument
        """
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
    def __init__(self, filename, frameType="bytes", verbose=1, fps=30.0, shape=None,
                input_pixel_format="bayer_rggb8", output_pixel_format="rgb0",
                gpuVEnc=False, gpuCompressionArgs=DEFAULT_GPU_COMPRESSION_ARGS,
                cpuCompressionArgs=DEFAULT_CPU_COMPRESSION_ARGS):
        """Construct a new ffmpegVideoWriter object.

        Args:
            filename (str): Filename where video should be written.
            frameType (str): A string indicating which frame type will be
                provided. Must be one of "image" (PIL.Image), "numpy"
                (numpy.ndarray) or "bytes" (bytes, the default)
            verbose (int): Verbosity output level. Defaults to 1.
            fps (float): Frames per second of the video Defaults to 30.0.
            shape (2-tuple or None): A tuple of two integers indicating the
                width and height of the video frames in pixels.
            input_pixel_format (str): The input pixel format to be used by
                ffmpeg to decode the incoming data. See ffmpeg for options.
                Defaults to "bayer_rggb8".
            output_pixel_format (str): The output pixel format to be used by
                ffmpeg to encode the video file. see ffmpeg for options.
                Defaults to "rgb0".
            gpuVEnc (bool): Whether or not to use GPU for encoding (nvenc).
                Defaults to False.
            gpuCompressionArgs (list of str): List of ffmpeg arguments
                describing desired compression algorithm/arguments, used only if
                gpeVEnc is True. Defaults to DEFAULT_GPU_COMPRESSION_ARGS.
            cpuCompressionArgs (list of str): List of ffmpeg arguments
                describing desired compression algorithm/arguments, used only if
                gpeVEnc is False. Defaults to DEFAULT_CPU_COMPRESSION_ARGS.
        """
        self.ffmpegProc = None
        self.verbose = verbose
        self.fps = fps
        self.filename = str(filename)
        self.shape = shape
        self.frameType = frameType
        self.input_pixel_format = input_pixel_format
        self.output_pixel_format = output_pixel_format
        self.gpuVEnc = gpuVEnc
        self.gpuCompressionArgs = gpuCompressionArgs
        self.cpuCompressionArgs = cpuCompressionArgs

    def getFFMPEGVerbosity(self):
        """Translate a numerical verbosity level to a verbosity level word
            understood by ffmpeg.

        Returns:
            str: ffmpeg verbosity level word

        """
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
        """Initialize a new ffmpeg process.

        Args:
            shape (2-tuple or None): A tuple of two integers indicating the
                width and height of the video frames in pixels.
        """

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
        """If shape is not provided, try to determine the shape from the frame
            data.

        Args:
            frame (PIL.Image, np.array, or bytes): A frame of video as the type
                specified by ffmpegWriter.frameType.
            shape (2-tuple or None): A tuple of two integers indicating the
                width and height of the video frames in pixels. If None, we will
                try to figure out the image shape. Defaults to None.
        """

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
        """Send a video frame to an ffmpeg process. If a ffmpeg process does not
            currently exist, start a new one

        Args:
            frame (PIL.Image, np.array, or bytes): A frame of video as the type
                specified by ffmpegWriter.frameType.
            shape (2-tuple or None): A tuple of two integers indicating the
                width and height of the video frames in pixels. If None, we will
                try to figure out the image shape. Defaults to None.
        """

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
        # with warnings.catch_warnings():
        #     warnings.simplefilter("ignore")
        self.ffmpegProc = None
        if self.verbose >= 2:
            print('Closed pipe to ffmpeg')

class ffmpegPipedVideoWriter(ffmpegVideoWriter):
    def __init__(self, filename, pipePath, numFrames=None, **kwargs):
        """Construct an ffmpegPipedVideoWriter.

        Args:
            filename (str): Filename where video should be written.
            pipePath (str): Path to windows named pipe
            numFrames (int or None): Number of frames to write before closing
                the writer. If None, wil continue to write until pipe is closed
                by the sender.
            **kwargs (dict): Arguments to pass to ffmpegVideoWriter parent class
        """
        super().__init__(filename, **kwargs)
        self.pipePath = pipePath
        self.numFrames = numFrames
        self.completed = False
        if not self.frameType == "bytes":
            raise ValueError('Piped video writers can only accomodate "bytes" frame type')

    def initializeFFMPEG(self, *args, **kwargs):
        self.dispatchFFMPEG(*args, **kwargs)

    def dispatchFFMPEG(self, shape): #, numTries=None, tryInverval=None, timeout=None):
        """Open an ffmpeg process and write video.

        Args:
            shape (2-tuple of ints): A tuple of two integers indicating the
                width and height of the video frames in pixels
            numTries (int or None): Number of times to attempt initializing the
                ffmpeg process. If None, will continue trying indefinitely, or
                until the given timeout expires.
            tryInterval (float or None): Amount of time to wait between tries.
                If None, will wait by default 1/fps, where fps is the given
                video framerate
            timeout (float or None): Maximum amount of time to keep trying. If
                None, will keep trying indefinitely or until it has tried
                numTries times.
        """

        if self.completed:
            raise RuntimeError('Process is already completed - create a new one')

        w, h = shape
        shapeArg = '{w}x{h}'.format(w=w, h=h)

        if self.verbose >= 3:
            print("STARTING NEW FFMPEG VIDEO PROCESS!")

        ffmpegVerbosity = self.getFFMPEGVerbosity()

        if self.numFrames is not None:
            # User requested specific # of frames to write
            frameCountArgs = ['-frames', str(self.numFrames)]
        else:
            # Continue writing until pipe is closed by sender
            frameCountArgs = []

        if self.gpuVEnc:
            # With GPU acceleration
            ffmpegCommand = [FFMPEG_EXE, '-y', '-probesize', '32', '-flush_packets', '1',
                '-progress', '-', '-stats_period', '0.1',
                '-hwaccel', 'cuda', '-hwaccel_output_format', 'cuda',
                '-v', ffmpegVerbosity, '-f', 'rawvideo', '-c:v', 'rawvideo',
                '-pix_fmt', self.input_pixel_format, '-s', shapeArg, '-thread_queue_size', '128',
                '-r', str(self.fps),
                '-i', self.pipePath,
                *self.gpuCompressionArgs,
                '-pix_fmt', self.output_pixel_format, '-fps_mode', 'passthrough', '-an'] + \
                frameCountArgs + \
                [self.filename]
        else:
            # Without GPU acceleration
            ffmpegCommand = [FFMPEG_EXE, '-y', '-probesize', '32', '-flush_packets', '1',
                '-progress', '-', '-stats_period', '0.1',
                '-v', ffmpegVerbosity, '-f', 'rawvideo',
                '-c:v', 'rawvideo', '-pix_fmt', self.input_pixel_format,
                '-s', shapeArg, '-r', str(self.fps), '-thread_queue_size', '128',
                 '-i', self.pipePath, *self.cpuCompressionArgs,
                '-pix_fmt', self.output_pixel_format, '-fps_mode', 'passthrough', '-an'] + \
                frameCountArgs + \
                [self.filename]

        if self.verbose >= 2:
            print('ffmpeg command:')
            print(ffmpegCommand)
        try:
            self.ffmpegProc = subprocess.Popen(ffmpegCommand, stdin=None, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        except FileNotFoundError:
            raise OSError('Error starting ffmpeg process - check that ffmpeg is present on this system and included in the system PATH variable')

    def getOutput(self, timeout=None):
        if self.ffmpegProc is None:
            raise RuntimeError('Process does not exist')
        # Get output from process - will raise TimeoutError if process does not finish before timeout (in seconds) expires
        outs, errs = self.ffmpegProc.communicate(timeout=timeout)
        returncode = self.ffmpegProc.returncode
        self.completed = True
        return [outs, errs, returncode]

    def getReturnCode(self):
        if self.ffmpegProc is None:
            raise RuntimeError('Process does not exist')
        if self.ffmpegProc.returncode is not None:
            self.completed = True
        return self.ffmpegProc.returncode

    def wait(self):
        """Wait for ffmpeg writer to finish.
        """
        if self.ffmpegProc is None:
            return

        [outs, errs, returncode] = self.getOutput(timeout=None)
        if not returncode == 0:
           raise IOError('ffmpeg error: {e}'.format(e=str(errs)))
        return [outs, errs, returncode]

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
        self.filename = str(filename)
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
