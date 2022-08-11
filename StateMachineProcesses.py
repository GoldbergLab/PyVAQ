import os
import struct
import time
import numpy as np
import multiprocessing as mp
import datetime as dt
from scipy.signal import butter, lfilter
import wave
import queue
from PIL import Image
from collections import defaultdict, deque
from threading import BrokenBarrierError
import itertools
import ffmpegWriter as fw
from SharedImageQueue import SharedImageSender
import traceback
import unicodedata
import re
from ctypes import c_wchar
import PySpinUtilities as psu
try:
    import PySpin
except ModuleNotFoundError:
    # pip seems to install PySpin as pyspin sometimes...
    import pyspin as PySpin
import sys

simulatedHardware = False
for arg in sys.argv[1:]:
    if arg == '-s' or arg == '--sim':
        # Use simulated harddware instead of physical cameras and DAQs
        simulatedHardware = True

if simulatedHardware:
    # Use simulated harddware instead of physical cameras and DAQs
    import PySpinSim.PySpinSim as PySpin
    import nidaqmxSim as nidaqmx
    from nidaqmxSim.stream_readers import AnalogMultiChannelReader, DigitalSingleChannelReader
    from nidaqmxSim.constants import Edge, TriggerType
else:
    # Use physical cameras/DAQs
    try:
        import PySpin
    except ModuleNotFoundError:
        # pip seems to install PySpin as pyspin sometimes...
        import pyspin as PySpin
    import nidaqmx
    from nidaqmx.stream_readers import AnalogMultiChannelReader, DigitalSingleChannelReader
    from nidaqmx.constants import Edge, TriggerType

def getFrameSize(camSerial):
    system = PySpin.System.GetInstance()
    camList = system.GetCameras()
    cam = camList.GetBySerial(camSerial)
    cam.Init()
    width = cam.Width.GetValue()
    height = cam.Height.GetValue()
    camList.Clear()
    cam.DeInit()
    cam = None
    system.ReleaseInstance()
    return width, height

def syncPrint(*args, sep=' ', end='\n', flush=True, buffer=None):
    kwargs = dict(sep=sep, end=end, flush=flush)
    buffer.append((args, kwargs))

def slugify(value, allow_unicode=False):
    """
    Convert to ASCII if 'allow_unicode' is False. Convert spaces to hyphens.
    Remove characters that aren't alphanumerics, underscores, or hyphens.
    Convert to lowercase. Also strip leading and trailing whitespace.

    Adapter from Django utils
    """
    value = str(value)
    if allow_unicode:
        value = unicodedata.normalize('NFKC', value)
    else:
        value = unicodedata.normalize('NFKD', value).encode('ascii', 'ignore').decode('ascii')
    value = re.sub(r'[^\w\s-]', '', value).strip().lower()
    return re.sub(r'[-\s]+', '-', value)

def clearQueue(q):
    if q is not None:
        while True:
            try:
                stuff = q.get(block=True, timeout=0.1)
            except queue.Empty:
                break

class Stopwatch:
    def __init__(self, history=2):
        self.t = [None for k in range(history)]

    def click(self):
        self.t.append(time.time())
        self.t.pop(0)

    def frequency(self):
        period = self.period()
        if period is None:
            return None
        return 1.0 / period

    def period(self):
        t0 = self.t[0]
        t1 = self.t[-1]
        if t1 is None or t0 is None:
            return None
        return (t1 - t0)/(len(self.t)-1)

class Trigger():
    # A class to represent a record trigger.
    # It maintains a unique ID, to allow processes to ensure they don't reuse
    #   triggers
    # It contains information about the start, end, and arbitrary "trigger" time
    # It also contains tags to allow processes to pass information about the
    #   trigger, typically to be added onto the filename of the recorded audio/video

    # Generator for new unique IDs
    newid = itertools.count().__next__   # Source of this clever little idea: https://stackoverflow.com/a/1045724/1460057
    # A class to represent a trigger object
    def __init__(self, startTime, triggerTime, endTime, tags=set(), id=None, idspace=None):
        # times in seconds
        if not (endTime >= triggerTime >= startTime):
            raise ValueError("Trigger times must satisfy startTime <= triggerTime <= endTime")
        if id is None:
            # No ID supplied. Generate a unique one.
            self.id = (Trigger.newid(), idspace)
        else:
            # ID given. Assign it.
            self.id = (id, idspace)
        self.startTime = startTime
        self.triggerTime = triggerTime
        self.endTime = endTime
        self.tags = tags

    def __str__(self):
        # Create string representation of trigger for logging/debug purposes
        return 'Trigger id {id}: {s}-->{t}-->{e} tags: {tags}'.format(id=self.id, s=self.startTime, t=self.triggerTime, e=self.endTime, tags=self.tags)

    def tagFilename(self, filename, separator='_'):
        # Add trigger tags onto filename in a standardized way
        if len(self.tags) == 0:
            return filename
        root, ext = os.path.splitext(filename)
        path, name = os.path.split(root)
        name = name + separator + separator.join(self.tags)
        taggedPath = os.path.join(path, name+ext)
        return taggedPath

    def isValid(self):
        # Sanity check the trigger values
        return self.startTime <= self.triggerTime and self.triggerTime <= self.endTime

    def state(self, time):
        # Given a frame/sample count and frame/sample rate:
        # If time is before trigger range
        #       return how far before in seconds
        # If time is after trigger range
        #       return how far after in seconds
        # If time is during trigger range
        #       return 0
        if time < self.startTime:
            return time - self.startTime
        if time > self.endTime:
            return time - self.endTime
        return 0

    def overlap(self, otherTrigger):
        # Given another trigger, return the overlap information for the other
        #   trigger and this trigger.
        #   [-, -] = other trigger is entirely before this trigger
        #   [-, 0] = other trigger overlaps the beginning of this trigger
        #   [-, +] = other trigger fully envelops this trigger
        #   [0, -] = this or the other trigger are invalid (or both)
        #   [0, 0] = these triggers have identical start and end times
        #   [0, +] = other trigger overlaps the end of this trigger
        #   [+, -] = this or the other trigger are invalid (or both)
        #   [+, 0] = this or the other trigger are invalid (or both)
        #   [+, +] = other trigger is entirely after this trigger
        return [self.state(otherTrigger.startTime), self.state(otherTrigger.endTime)]

    def overlaps(self, otherTrigger):
        # Returns a boolean indicating whether or not this trigger overlaps
        #   otherTrigger. Returns False if either trigger is invalid
        overlap = self.overlap(otherTrigger)
        return (overlap[0] * overlap[1]) <= 0 and self.isValid() and otherTrigger.isValid()

class AudioChunk():
    # A class to wrap a chunk of audio data.
    # It conveniently bundles the audio data with timing statistics about the data,
    # and functions for manipulating and querying time info about the data
    newid = itertools.count().__next__   # Source of this clever little idea: https://stackoverflow.com/a/1045724/1460057
    def __init__(self,
                chunkStartTime=None,    # Time of first sample, in seconds since something
                audioFrequency=None,      # Audio sampling rate, in Hz
                data=None,               # Audio data as a CxN numpy array, C=# of channels, N=# of samples
                idspace=None,           # A parameter designed to make IDs unique across processes. Pass a different idspace for calls from different processes.
                ):
        self.id = (AudioChunk.newid(), idspace)
        self.data = data
        self.chunkStartTime = chunkStartTime
        self.audioFrequency = audioFrequency
        self.channelNumber, self.chunkSize = self.data.shape
        self.chunkEndTime = self.calculateChunkEndTime()

    def __str__(self):
        return 'Audio chunk {id}: {start} ---- {samples} samp x {n} ch ----> {end} @ {freq} Hz'.format(start=self.chunkStartTime, end=self.chunkEndTime, samples=self.chunkSize, n=self.channelNumber, freq=self.audioFrequency, id=self.id)

    def calculateChunkEndTime(self):
        return self.chunkStartTime + (self.chunkSize / self.audioFrequency)

    def addChunkToEnd(self, nextChunk):
        # Add on another chunk to the end of this one.
        self.data = np.concatenate((self.data, nextChunk.data), axis=1)
        self.chunkEndTime = self.calculateChunkEndTime()

    def addChunkToStart(self, nextChunk):
        # Add on another chunk to the end of this one.
        self.data = np.concatenate((nextChunk.data, self.data), axis=1)
        self.chunkStartTime = nextChunk.chunkStartTime
        self.chunkEndTime = self.calculateChunkEndTime()

    def getChannelCount(self):
        return self.data.shape[0]

    def getSampleCount(self):
        return self.data.shape[1]

    def getTriggerState(self, trigger):
        # Check the manner in which this audio chunk overlaps, or does not
        #   overlap, with the given trigger
        chunkStartTriggerState = trigger.state(self.chunkStartTime)
        chunkEndTriggerState =   trigger.state(self.chunkEndTime)
        return chunkStartTriggerState, chunkEndTriggerState

    def splitAtSample(self, sampleSplitNum):
        # Split audio chunk into two so the first chunk has sampleSplitNum
        #   samples in it, and the second chunk has the rest.
        #   Returns the two resultant chunks in a tuple

        if sampleSplitNum < 0:
            sampleSplitNum = 0
        if sampleSplitNum > self.getSampleCount():
            sampleSplitNum = self.getSampleCount()

        # Construct the pre-chunk
        preChunk = AudioChunk(
            chunkStartTime=self.chunkStartTime,
            audioFrequency=self.audioFrequency,
            data=np.copy(self.data[:, :sampleSplitNum]),
            idspace=self.id[1])

        # Modify this chunk so it's the post-chunk
        self.data = self.data[:, sampleSplitNum:]
        self.chunkStartTime = self.chunkStartTime + (sampleSplitNum / self.audioFrequency)
        self.channelNumber, self.chunkSize = self.data.shape
        self.chunkEndTime = self.calculateChunkEndTime()
        return preChunk, self

    def trimToTrigger(self, trigger, returnOtherPieces=False): # padStart=False):
        # Trim audio chunk so it lies entirely within the trigger period, and update stats accordingly
        # If padStart == True, pad the audio chunk with enough data so it begins at the beginning of the trigger period
        # If returnOtherPieces == True, returns the pre-chunk before the trim and the post-chunk after the trim, or None for one or both if there is no trim before and/or after
        chunkStartTriggerState, chunkEndTriggerState = self.getTriggerState(trigger)

        # Trim chunk start:
        if chunkStartTriggerState < 0:
            # Start of chunk is before start of trigger - truncate start of chunk.
            startSample = abs(int(chunkStartTriggerState * self.audioFrequency))
            newChunkStartTime = trigger.startTime
        elif chunkStartTriggerState == 0:
            # Start of chunk is in trigger period, do not trim start of chunk, pad if padStart=True
            startSample = 0
            newChunkStartTime = self.chunkStartTime
        else:
            # Start of chunk is after trigger period...chunk must not be in trigger period at all
            startSample = self.chunkSize
            newChunkStartTime = trigger.endTime

        # Trim chunk end
        if chunkEndTriggerState < 0:
            # End of chunk is before start of trigger...chunk must be entirely before trigger period
            endSample = 0
            newChunkEndTime = trigger.startTime
        elif chunkEndTriggerState == 0:
            # End of chunk is in trigger period, do not trim end of chunk
            endSample = self.chunkSize
            newChunkEndTime = self.chunkEndTime
        else:
            # End of chunk is after trigger period - trim chunk to end of trigger period
            endSample = self.chunkSize - (chunkEndTriggerState * self.audioFrequency)
            newChunkEndTime = trigger.endTime

        startSample = round(startSample)
        endSample = round(endSample)
#        print("Trim samples: {first}|{start} --> {end}|{last}".format(start=startSample, end=endSample, first=0, last=self.chunkSize))
        if returnOtherPieces:
            if startSample > 0:
                preChunk = AudioChunk(
                    chunkStartTime=self.chunkStartTime,
                    audioFrequency=self.audioFrequency,
                    data=np.copy(self.data[:, :startSample]))
            else:
                preChunk = None
            if endSample < self.chunkSize:
                postChunk = AudioChunk(
                    chunkStartTime=self.chunkStartTime + (endSample / self.audioFrequency),
                    audioFrequency=self.audioFrequency,
                    data=np.copy(self.data[:, endSample:]))
            else:
                postChunk = None
            parts = [preChunk, postChunk]
        else:
            parts = None

        self.data = self.data[:, startSample:endSample]
        self.chunkStartTime = newChunkStartTime
        self.channelNumber, self.chunkSize = self.data.shape
        self.chunkEndTime = self.calculateChunkEndTime()
        return parts

        # if padStart is True and startSample == 0:
        #     padLength = round((self.chunkStartTime - trigger.startTime) * self.audioFrequency)
        #     pad = np.zeros((self.channelNumber, padLength), dtype='int16')
        #     self.data = np.concatenate((pad, self.data), axis=1)
        # self.chunkSize = self.data.shape[1]

    def getAsBytes(self):
        bytePackingPattern = 'h'*self.data.shape[0]
        packingFunc = lambda x:struct.pack(bytePackingPattern, *x)
#        print(b''.join(list(map(packingFunc, self.data.transpose().tolist())))[0:20])
        return b''.join(map(packingFunc, self.data.transpose().tolist()))

#  audioChunkBytes = b''.join(map(lambda x:struct.pack(bytePackingPattern, *x), audioChunk.transpose().tolist()))

DATE_FORMAT = '%Y-%m-%d'
TIME_FORMAT = '%Y-%m-%d-%H-%M-%S-%f'

def getDaySubfolder(root, trigger=None, timestamp=None):
    # Construct a standardized path for a subfolder representing the day in
    #   which the trigger falls. If a trigger is not provided, use the timestamp argument instead.
    if trigger is not None:
        timestamp = trigger.triggerTime
    dateString = dt.datetime.fromtimestamp(timestamp).strftime(DATE_FORMAT)
    return os.path.join(root, dateString)

def ensureDirectoryExists(directory):
    # Creates directory (and subdirectories if necessary) to ensure that the directory exists in the filesystem
    if len(directory) > 0:
        os.makedirs(directory, exist_ok=True)

def generateTimeString(trigger=None, timestamp=None):
    # Generate a time string from the trigger time.
    # If no trigger is given, the timestamp argument is used instead.
    if trigger is not None:
        timestamp = trigger.triggerTime
    return dt.datetime.fromtimestamp(timestamp).strftime(TIME_FORMAT)

def generateFileName(directory='.', baseName='unnamed', tags=[], extension=''):
    # Construct a standardized filename based on a root directory a base name,
    #   zero or more tags, and an extension.
    extension = '.' + slugify(extension)
    fileName = baseName
    fileName = '_'.join([fileName]+tags)
    fileName = slugify(fileName)
    fileName += extension
    return os.path.join(directory, fileName)

def generateButterBandpassCoeffs(lowcut, highcut, fs, order=5):
    # Set up audio bandpass filter coefficients
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

class StdoutManager(mp.Process):
    # A process for printing output to stdout from other processes.
    # Expects the following messageBundle format from queues:
    #   msgBundle = [msg1, msg2, msg3...],
    # Where each message is of the format
    #   msg = ((arg1, arg2, arg3...), {kw1:kwarg1, kw2:kwarg2...})

    EXIT = 'exit'

    def __init__(self, logFilePath=''):
        mp.Process.__init__(self, daemon=True)
        self.queue = mp.Queue()
        self.timeout = 0.1
        self.PID = mp.Value('i', -1)
        if logFilePath is not None:
            if logFilePath == '':
                self.logFilePath = './logs/PyVAQ_Log_'+dt.datetime.now().strftime(TIME_FORMAT)
            else:
                self.logFilePath = logFilePath
            path, name = os.path.split(self.logFilePath)
            ensureDirectoryExists(path)
        else:
            self.logFilePath = None


    def run(self):
        self.PID.value = os.getpid()
        if self.logFilePath is not None:
            try:
                self.logFile = open(self.logFilePath, 'w')
            except:
                self.logFile = None
                print('Failed to open log file.')
        if self.logFile is not None:
            print('Log file begin: '+dt.datetime.now().strftime(TIME_FORMAT), file=self.logFile)

        while True:
            msgBundles = []
            try:
                msgBundles.append(self.queue.get(block=True, timeout=self.timeout))
            except queue.Empty:
                pass

            for msgBundle in msgBundles:
                if msgBundle == StdoutManager.EXIT:
                    print("StdoutManager: Received stop signal!")
                    return 0
                for args, kwargs in msgBundle:
                    print(*args, **kwargs)
                    if self.logFile is not None:
                        print(*args, **kwargs, file=self.logFile)
                print()
        clearQueue(self.queue)
        if self.logFile is not None:
            try:
                self.logFile.close()
            except:
                print('Failed to close log file')

class States:
    # Dummy class to hold all states
    UNKNOWN =       -1
    STOPPED =       0
    INITIALIZING =  1
    READY =         2
    STOPPING =      3
    ERROR =         4
    EXITING =       5
    DEAD =          6
    IGNORING =      100
    MERGING =       102
    SYNCHRONIZING = 200
    WAITING =       300
    ANALYZING =     301
    AUDIOINIT =     501
    WRITING =       600
    BUFFERING =     601
    ACQUIRING =     700
    VIDEOINIT =     801
    TRIGGERING =    1000

class StateMachineProcess(mp.Process):
    # The base process that all other state machine processes inherit from.
    # It contains functionality for receiving messages from other processes,
    # and other conveniences.

    # Human-readable states
    stateList = {
        States.UNKNOWN :       'UNKNOWN',
        States.STOPPED :       'STOPPED',
        States.INITIALIZING :  'INITIALIZING',
        States.STOPPING :      'STOPPING',
        States.ERROR :         'ERROR',
        States.EXITING :       'EXITING',
        States.DEAD :          'DEAD'
    }

    def __init__(self, *args, stdoutQueue=None, daemon=True, **kwargs):
        mp.Process.__init__(self, *args, daemon=daemon, **kwargs)
        self.ID = "X"                                   # An ID for logging purposes to identify the source of log messages
        self.msgQueue = mp.Queue()                      # Queue for receiving messages/requests from other processes
        self.stdoutQueue = stdoutQueue                  # Queue for pushing output message groups to for printing
        self.publishedStateVar = mp.Value('i', -1)      # A thread-safe variable so other processes can query this process's state
        self.PID = mp.Value('i', -1)                    # A thread-safe variable so other processes can query this process's PID
        self.publishedInfoLength = 256
        self.publishedInfoVar = mp.Array(c_wchar, self.publishedInfoLength)  # A thread-safe variable so other processes can query this process's latest info
        self.exitFlag = False                           # A flag to set to ensure the process exits ASAP
        self.stdoutBuffer = []                          # A buffer to accumulate log messages before sending out
        self.state = None
        self.lastState = None
        self.nextState = None

    def run(self):
        # Start run by recording this process's PID
        self.PID.value = os.getpid()

    def updatePublishedInfo(self, info):
        if self.publishedInfoVar is not None:
            # Pad or truncate info as necessary to make it the correct length to fit in the shared array
            infoLength = len(info)
            if infoLength > self.publishedInfoLength:
                info = info[:self.publishedInfoLength]
            elif infoLength < self.publishedInfoLength:
                info = info + ' '*(self.publishedInfoLength - infoLength)
            L = self.publishedInfoVar.get_lock()
            locked = L.acquire(block=False)
            if locked:
                self.publishedInfoVar[:] = info
                L.release()

    def updatePublishedState(self, newState=None):
        # Update the thread-safe variable holding the current state info
        if newState is not None:
            self.state = newState
        # Update the thread-safe variable holding the current state info
        if self.publishedStateVar is not None:
            L = self.publishedStateVar.get_lock()
            locked = L.acquire(block=False)
            if locked:
                self.publishedStateVar.value = self.state
                L.release()

    def log(self, msg, *args, **kwargs):
        # Queue up another ID-tagged log message
        syncPrint('|| {ID} - {msg}'.format(ID=self.ID, msg=msg), *args, buffer=self.stdoutBuffer, **kwargs)

    def logTime(self, *args, **kwargs):
        timestamp = dt.datetime.now().strftime(TIME_FORMAT)
        self.log('| {timestamp} | '.format(timestamp=timestamp), *args, **kwargs)

    def logEnd(self):
        timestamp = dt.datetime.now().strftime(TIME_FORMAT)
        self.log(r'*** {timestamp} *** lastState={lastState}, state={state}, nextState={nextState} *** exitFlag={exitFlag}'.format(timestamp=timestamp, exitFlag=self.exitFlag, lastState=self.stateList[self.lastState], state=self.stateList[self.state], nextState=self.stateList[self.nextState]))
        self.log(r'*********************************** /\ {ID} {state} /\ ********************************************'.format(ID=self.ID, state=self.stateList[self.state]))

    def flushStdout(self):
        # Send current accumulated log buffer out, clear buffer.
        if len(self.stdoutBuffer) > 0:
            self.stdoutQueue.put(self.stdoutBuffer)
        self.stdoutBuffer = []

class AVMerger(StateMachineProcess):
    # Class for merging audio and video files using ffmpeg

    # Human-readable states
    stateList = {
        States.IGNORING :'IGNORING',
        States.WAITING :'WAITING',
        States.MERGING :'MERGING',
    }

    # Include common states from parent class
    stateList.update(StateMachineProcess.stateList)

    # Recognized message types:
    START = 'msg_start'
    MERGE = 'msg_merge'
    STOP = 'msg_stop'
    EXIT = 'msg_exit'
    CHILL = 'msg_chill'
    SETPARAMS = 'msg_setParams'

    # Stream types
    VIDEO = 'video'
    AUDIO = 'audio'

    NO_REENCODING = "No reencoding"
    COMPRESSION_OPTIONS = [NO_REENCODING]+[str(k) for k in range(52)]


    # List of params that can be set externally with the 'msg_setParams' message
    settableParams = [
        'verbose',
        'directory',
        'numFilesPerTrigger',
        'baseFileName',
        'montage',
        'deleteMergedAudioFiles',
        'deleteMergedVideoFiles',
        'compression',
        'daySubfolders',
        'baseFileName'
    ]

    def __init__(self,
        verbose=False,
        numFilesPerTrigger=2,           # Number of files expected per trigger event (audio + video)
        directory='.',                  # Directory for writing merged files
        baseFileName='',                # Base filename (sans extension) for writing merged files
        deleteMergedAudioFiles=False,   # After merging, delete unmerged audio originals
        deleteMergedVideoFiles=False,   # After merging, delete unmerged video originals
        montage=False,                  # Combine videos side by side
        compression='0',                # CRF factor for libx264 compression. '0'=lossless '23'=default '51'=terrible
        daySubfolders=True,             # Put output files into separate day folders?
        **kwargs):
        StateMachineProcess.__init__(self, **kwargs)
        # Store inputs in instance variables for later access
        self.ID = "M"
        self.verbose = verbose
        self.ignoreFlag = True
        self.errorMessages = []
        self.numFilesPerTrigger = numFilesPerTrigger
        self.directory = directory
        self.baseFileName = baseFileName
        self.montage = montage
        self.deleteMergedAudioFiles = deleteMergedAudioFiles
        self.deleteMergedVideoFiles = deleteMergedVideoFiles
        self.compression = compression
        if self.numFilesPerTrigger < 2:
            if self.verbose >= 0: self.log("Warning! AVMerger can't merge less than 2 files!")
        self.daySubfolders = daySubfolders

    def setParams(self, **params):
        for key in params:
            if key in AVMerger.settableParams:
                setattr(self, key, params[key])
                if self.verbose >= 1: self.log("Param set: {key}={val}".format(key=key, val=params[key]))
            else:
                if self.verbose >= 0: self.log("Param not settable: {key}={val}".format(key=key, val=params[key]))

    def run(self):
        self.PID.value = os.getpid()
        if self.verbose >= 1: self.log("PID={pid}".format(pid=os.getpid()))
        self.state = States.STOPPED
        self.nextState = States.STOPPED
        self.lastState = -1
        msg = ''; arg = None

        while True:
            # Publish updated state
            if self.state != self.lastState:
                self.updatePublishedState()

            try:
# AVMerger: *********************** STOPPPED *********************************
                if self.state == States.STOPPED:
                    # DO STUFF

                    # CHECK FOR MESSAGES
                    try:
                        msg, arg = self.msgQueue.get(block=True)
                        if msg == AVMerger.SETPARAMS: self.setParams(**arg); msg = ''; arg=None
                    except queue.Empty: msg = ''; arg = None

                    # CHOOSE NEXT STATE
                    if self.exitFlag:
                        self.nextState = States.EXITING
                    elif msg == '':
                        self.nextState = self.state
                    elif msg == AVMerger.STOP:
                        self.nextState = States.STOPPED
                    elif msg == AVMerger.CHILL:
                        self.ignoreFlag = True
                        self.nextState = States.INITIALIZING
                    elif msg == AVMerger.START:
                        self.ignoreFlag = False
                        self.nextState = States.INITIALIZING
                    elif msg == AVMerger.STOP:
                        self.nextState = States.STOPPED
                    elif msg == AVMerger.EXIT:
                        self.exitFlag = True
                        self.nextState = States.EXITING
                    else:
                        raise SyntaxError("Message \"" + msg + "\" not relevant to " + self.stateList[self.state] + " state")
# AVMerger: *********************** INITIALIZING *********************************
                elif self.state == States.INITIALIZING:
                    # DO STUFF
                    if self.numFilesPerTrigger < 2:
                        raise IOError("Can't merge less than two files at a time!")

                    receivedFileEventList = []
                    groupedFileEventList = []

                    # CHECK FOR MESSAGES
                    try:
                        msg, arg = self.msgQueue.get(block=False)
                        if msg == AVMerger.SETPARAMS: self.setParams(**arg); msg = ''; arg=None
                    except queue.Empty: msg = ''; arg = None

                    # CHOOSE NEXT STATE
                    if self.exitFlag:
                        self.nextState = States.STOPPING
                    elif msg == AVMerger.CHILL or self.ignoreFlag:
                        self.ignoreFlag = True
                        self.nextState = States.IGNORING
                    elif msg == AVMerger.START:
                        self.ignoreFlag = False
                        self.nextState = States.WAITING
                    elif msg in '':
                        self.nextState = States.WAITING
                    elif msg == AVMerger.STOP:
                        self.nextState = States.STOPPING
                    elif msg == AVMerger.EXIT:
                        self.exitFlag = True
                        self.nextState = States.STOPPING
                    else:
                        raise SyntaxError("Message \"" + msg + "\" not relevant to " + self.stateList[self.state] + " state")
# AVMerger: *********************** IGNORING *********************************
                elif self.state == States.IGNORING:    # ignoring merge requests
                    # DO STUFF
                    if self.numFilesPerTrigger < 2:
                        raise IOError("Can't merge less than two files at a time!")

                    # Clear any file events already received
                    receivedFileEventList = []
                    groupedFileEventList = []

                    # Reset ignore flag
                    self.ignoreFlag = False

                    # CHECK FOR MESSAGES
                    try:
                        msg, arg = self.msgQueue.get(block=True, timeout=0.1)
                        if msg == AVMerger.SETPARAMS: self.setParams(**arg); msg = ''; arg=None
                    except queue.Empty: msg = ''; arg = None

                    # CHOOSE NEXT STATE
                    if msg == AVMerger.MERGE and arg is not None:
                        if self.verbose >= 1: self.log("Ignoring {streamType} file event for merging at {time}: {file}".format(file=arg['filePath'], streamType=arg['streamType'], time=arg['trigger'].triggerTime))
                        self.nextState = States.IGNORING
                    elif msg == AVMerger.CHILL:
                        self.ignoreFlag = True
                        self.nextState = States.IGNORING
                    elif msg == AVMerger.START:
                        self.ignoreFlag = False
                        self.nextState = States.WAITING
                    elif msg == '':
                        self.nextState = States.IGNORING
                    elif msg == AVMerger.STOP:
                        self.nextState = States.STOPPING
                    elif msg == AVMerger.EXIT:
                        self.exitFlag = True
                        self.nextState = AVMergers.STOPPING
                    else:
                        raise SyntaxError("Message \"" + msg + "\" not relevant to " + self.stateList[self.state] + " state")
# AVMerger: *********************** WAITING *********************************
                elif self.state == States.WAITING:    # Waiting for files to merge
                    # DO STUFF
                    if self.numFilesPerTrigger < 2:
                        raise IOError("Can't merge less than two files at a time!")

                    # Waiting for message of type:
                    #   (AVMerger.MERGE, {'filePath':filePath,
                    #                     'streamType':streamType,
                    #                     'trigger':trigger,
                    #                     'streamID':streamID}
                    # Where
                    #   filepath is a string representing an audio or video file to merge
                    #   streamType is one of the defined streamtypes in AVMerger
                    #   trigger is a trigger object
                    #   streamID is a stream ID to use for file naming (only relevant for video files)
                    # If a new file has been received, add it to the list

                    # Check if any two of the files received share a matching trigger
                    IDCounts = defaultdict(lambda:0)
                    foundID = None
                    for fileEvent in receivedFileEventList:
                        IDCounts[fileEvent['trigger'].id] += 1
                        if IDCounts[fileEvent['trigger'].id] >= self.numFilesPerTrigger:
                            foundID = fileEvent['trigger'].id
                            break

                    # If one of the trigger IDs appears enough times, separate out those two files and prepare to merge them.
                    if foundID is not None:
                        groupedFileEventList.append(tuple(filter(lambda fileEvent:fileEvent['trigger'].id == foundID, receivedFileEventList)))
                        for fileEventGroup in groupedFileEventList:
                            for fileEvent in fileEventGroup:
                                try:
                                    receivedFileEventList.remove(fileEvent)
                                except ValueError:
                                    pass  # Already removed

                    if self.verbose > 3:
                        if len(receivedFileEventList) > 0 or len(groupedFileEventList) > 0:
                            self.log("Received: ", [p['filePath'] for p in receivedFileEventList])
                            self.log("Ready:    ", [tuple([p['filePath'] for p in fileEvent]) for fileEvent in groupedFileEventList])

                    # CHECK FOR MESSAGES
                    try:
                        msg, arg = self.msgQueue.get(block=True, timeout=0.1)
                        if msg == AVMerger.SETPARAMS: self.setParams(**arg); msg = ''; arg=None
                    except queue.Empty: msg = ''; arg = None

                    # CHOOSE NEXT STATE
                    if msg == AVMerger.MERGE and arg is not None:
                        receivedFileEventList.append(arg)
                        if self.verbose >= 1: self.log("Received {streamType} file event for merging at {time}: {file}".format(file=arg['filePath'], streamType=arg['streamType'], time=arg['trigger'].triggerTime))
                        self.nextState = States.WAITING
                    elif msg == AVMerger.CHILL or self.ignoreFlag:
                        self.ignoreFlag = True
                        self.nextState = States.IGNORING
                    elif len(groupedFileEventList) > 0:
                        # At least one group of unmerged matching files - go to merge
                        self.nextState = States.MERGING
                    elif msg in ['', AVMerger.START]:
                        self.nextState = States.WAITING
                    elif msg == AVMerger.STOP:
                        self.nextState = States.STOPPING
                    elif msg == AVMerger.EXIT:
                        self.exitFlag = True
                        self.nextState = AVMergers.STOPPING
                    else:
                        raise SyntaxError("Message \"" + msg + "\" not relevant to " + self.stateList[self.state] + " state")
# AVMerger: *********************** MERGING *********************************
                elif self.state == States.MERGING:
                    # DO STUFF
                    if self.numFilesPerTrigger < 2:
                        raise IOError("Can't merge less than two files at a time!")
                    # If a new file has been received, add it to the list

                    for fileEventGroup in groupedFileEventList:
                        mergeSuccess = True
                         # Merge all audio streams with each video stream individually
                        audioFileEvents = tuple(filter(lambda fileEvent:fileEvent['streamType'] == AVMerger.AUDIO, fileEventGroup))
                        videoFileEvents = tuple(filter(lambda fileEvent:fileEvent['streamType'] == AVMerger.VIDEO, fileEventGroup))
                        # Construct the audio part of the ffmpeg command template
                        audioFileInputText = ' '.join(['-i "{{audioFile{k}}}"'.format(k=k) for k in range(len(audioFileEvents))])
                        if self.daySubfolders:
                            mergeDirectory = getDaySubfolder(self.directory, fileEventGroup[0]['trigger'])
                        else:
                            mergeDirectory = self.directory
                        ensureDirectoryExists(mergeDirectory)
                        if self.verbose >= 1: self.log('Merging into directory: {d}, daySubfolders={dsf}'.format(d=mergeDirectory, dsf=self.daySubfolders))
                        if not self.montage:  # Make a separate file for each video stream
                            # Construct command template
                            if self.compression == AVMerger.NO_REENCODING:
                                videoEncoding = '-c:v copy'
                            else:
                                videoEncoding = '-c:v libx264 -preset veryfast -crf {compression}'.format(compression=self.compression)
                            mergeCommandTemplate = 'ffmpeg -i "{videoFile}" ' + audioFileInputText + ' ' + videoEncoding + ' -shortest -nostdin -y "{outputFile}"'
                            # Set up dictionary of strings to substitute into command template
                            kwargs = dict([('audioFile{k}'.format(k=k), audioFileEvents[k]['filePath']) for k in range(len(audioFileEvents))])
                            for videoFileEvent in videoFileEvents:
                                # Add/update dictionary to reflect this video file
                                kwargs['videoFile'] = videoFileEvent['filePath']
                                fileNameTags = [videoFileEvent['streamID'], 'merged', generateTimeString(videoFileEvent['trigger'])] + list(videoFileEvent['trigger'].tags)
                                kwargs['outputFile'] = generateFileName(directory=mergeDirectory, baseName=self.baseFileName, extension='.avi', tags=fileNameTags)
                                # Substitute strings into command template
                                mergeCommand = mergeCommandTemplate.format(**kwargs)
                                if self.verbose >= 1:
                                    self.log("Merging with kwargs: "+str(kwargs))
                                    self.log("Merging with command:")
                                    self.log("{command}".format(command=mergeCommand))
                                # Execute constructed merge command
                                status = os.system(mergeCommand)
                                mergeSuccess = mergeSuccess and (status == 0)
                                if self.verbose >= 1: self.log("Merge exit status: {status}".format(status=status))
                        else:   # Montage the video streams into one file
                            # Construct the video part of the ffmpeg command template
                            videoFileInputText = ' '.join(['-i "{{videoFile{k}}}"'.format(k=k) for k in range(len(videoFileEvents))])
                            if self.compression == AVMerger.NO_REENCODING:
                                videoEncoding = '-c:v copy'
                            else:
                                videoEncoding = '-c:v libx264 -preset veryfast -crf {compression}'.format(compression=self.compression)
                            # Construct command template
                            mergeCommandTemplate = "ffmpeg " + videoFileInputText + " " + audioFileInputText + ' ' + videoEncoding + ' -shortest -nostdin -y -filter_complex hstack "{outputFile}"'
                            # Set up dictionary of strings to substitute into command template
                            kwargs = dict(
                                [('audioFile{k}'.format(k=k), audioFileEvents[k]['filePath']) for k in range(len(audioFileEvents))] + \
                                [('videoFile{k}'.format(k=k), videoFileEvents[k]['filePath']) for k in range(len(videoFileEvents))])
                            fileNameTags = ['_'.join([videoFileEvent['streamID'] for videoFileEvent in videoFileEvents]),
                                            'montage',
                                            generateTimeString(videoFileEvent['trigger'])
                                            ] + list(videoFileEvent['trigger'].tags)
                            kwargs['outputFile'] = generateFileName(directory=mergeDirectory, baseName=self.baseFileName, extension='.avi', tags=fileNameTags)
                            kwargs['compression'] = self.compression
                            mergeCommand = mergeCommandTemplate.format(**kwargs)
                            if self.verbose >= 1:
                                self.log("Merging with kwargs: "+str(kwargs))
                                self.log("Merging with command:")
                                self.log("{command}".format(command=mergeCommand))
                            # Execute constructed merge command
                            status = os.system(mergeCommand)
                            mergeSuccess = mergeSuccess and (status == 0)
                            if self.verbose >= 1: self.log("Merge exit status: {status}".format(status=status))
                        if mergeSuccess:
                            if self.deleteMergedAudioFiles:
                                for fileEvent in audioFileEvents:
                                    if self.verbose >= 1: self.log('deleting source audio file: {file}'.format(file=fileEvent['filePath']))
                                    os.remove(fileEvent['filePath'])
                            if self.deleteMergedVideoFiles:
                                for fileEvent in videoFileEvents:
                                    if self.verbose >= 1: self.log('deleting source video file: {file}'.format(file=fileEvent['filePath']))
                                    os.remove(fileEvent['filePath'])
                        else:
                            if self.verbose >= 0: self.log('Merge failure - keeping all source files in place!')

                    # Clear merged files
                    groupedFileEventList = []

                    # CHECK FOR MESSAGES
                    try:
                        msg, arg = self.msgQueue.get(block=True, timeout=0.1)
                        if msg == AVMerger.SETPARAMS: self.setParams(**arg); msg = ''; arg=None
                    except queue.Empty: msg = ''; arg = None

                    # CHOOSE NEXT STATE
                    if self.exitFlag:
                        self.nextState = States.STOPPING
                    elif msg == AVMerger.MERGE:
                        if arg is not None:
                            receivedFileEventList.append(arg)
                        self.nextState = States.WAITING
                    elif msg in ['', AVMerger.START]:
                        self.nextState = States.WAITING
                    elif msg == AVMerger.CHILL:
                        self.ignoreFlag = True
                        self.nextState = States.IGNORING
                    elif msg == AVMerger.STOP:
                        self.nextState = States.STOPPING
                    elif msg == AVMerger.EXIT:
                        self.exitFlag = True
                        self.nextState = States.STOPPING
                    else:
                        raise SyntaxError("Message \"" + msg + "\" not relevant to " + self.stateList[self.state] + " state")
# AVMerger: *********************** STOPPING *********************************
                elif self.state == States.STOPPING:
                    # DO STUFF

                    # CHECK FOR MESSAGES
                    try:
                        msg, arg = self.msgQueue.get(block=False)
                        if msg == AVMerger.SETPARAMS: self.setParams(**arg); msg = ''; arg=None
                    except queue.Empty: msg = ''; arg = None

                    # CHOOSE NEXT STATE
                    if self.exitFlag:
                        self.nextState = States.STOPPED
                    elif msg == '':
                        self.nextState = States.STOPPED
                    elif msg == AVMerger.START:
                        self.nextState = States.INITIALIZING
                    elif msg == AVMerger.STOP:
                        self.nextState = States.STOPPED
                    elif msg == AVMerger.EXIT:
                        self.exitFlag = True
                        self.nextState = States.STOPPED
                    else:
                        raise SyntaxError("Message \"" + msg + "\" not relevant to " + self.stateList[self.state] + " state")
# AVMerger: *********************** ERROR *********************************
                elif self.state == States.ERROR:
                    # DO STUFF
                    if self.verbose >= 0:
                        self.log("ERROR STATE. Error messages:\n\n")
                        self.log("\n\n".join(self.errorMessages))

                    self.updatePublishedInfo("\n".join(self.errorMessages))

                    self.errorMessages = []

                    # CHECK FOR MESSAGES
                    try:
                        msg, arg = self.msgQueue.get(block=False)
                        if msg == AVMerger.SETPARAMS: self.setParams(**arg); msg = ''; arg=None
                    except queue.Empty: msg = ''; arg = None

                    # CHOOSE NEXT STATE
                    if self.lastState == States.ERROR:
                        # Error ==> Error, let's just exit
                        self.nextState = States.EXITING
                    elif msg == '':
                        if self.lastState == States.STOPPING:
                            # We got an error in stopping state? Better just stop.
                            self.nextState = States.STOPPED
                        elif self.lastState ==States.STOPPED:
                            # We got an error in the stopped state? Better just exit.
                            self.nextState = States.EXITING
                        else:
                            self.nextState = States.STOPPING
                    elif msg == AVMerger.STOP:
                        self.nextState = States.STOPPED
                    elif msg == AVMerger.EXIT:
                        self.exitFlag = True
                        if self.lastState == States.STOPPING:
                            self.nextState = States.EXITING
                        else:
                            self.nextState = States.STOPPING
                    else:
                        raise SyntaxError("Message \"" + msg + "\" not relevant to " + self.stateList[self.state] + " state")
# AVMerger: *********************** EXIT *********************************
                elif self.state == States.EXITING:
                    break
                else:
                    raise KeyError("Unknown state: "+self.stateList[self.state])
            except KeyboardInterrupt:
                # Handle user using keyboard interrupt
                if self.verbose >= 0: self.log("Keyboard interrupt received - exiting")
                self.exitFlag = True
                self.nextState = States.STOPPING
            except:
                # HANDLE UNKNOWN ERROR
                self.errorMessages.append("Error in "+self.stateList[self.state]+" state\n\n"+traceback.format_exc())
                self.nextState = States.ERROR

            if (self.verbose >= 1 and (len(msg) > 0 or self.exitFlag)) or len(self.stdoutBuffer) > 0 or self.verbose >= 3:
                self.log("msg={msg}".format(msg=msg))
                self.logEnd()

            # Prepare to advance to next state
            self.lastState = self.state
            self.state = self.nextState
            self.flushStdout()

        clearQueue(self.msgQueue)
        if self.verbose >= 1: self.log("AVMerger process STOPPED")

        self.flushStdout()
        self.updatePublishedState(States.DEAD)

class Synchronizer(StateMachineProcess):
    # Class for generating two synchronization signals at the same time
    #   - one for video (send via cable to the camera GPIO)
    #   - one for audio (used internally to trigger analog input of microphone
    #     signals)
    # This class inherits from multiprocessing.Process so it can be run in a
    #   separate process, allowing a single script to generate the sync pulses
    #   and also accomplish other tasks.

    # Human-readable states
    stateList = {
         States.SYNCHRONIZING:  'SYNCHRONIZING',
         States.READY:          'SYNC_READY',
         States.WAITING:        'WAITING'
    }

    # Include common states from parent class
    stateList.update(StateMachineProcess.stateList)

    # Recognized message types:
    START = 'msg_start'
    STOP = 'msg_stop'
    EXIT = 'msg_exit'
    SETPARAMS = 'msg_setParams'
    SYNC = 'msg_sync'

    # List of params that can be set externally with the 'msg_setParams' message
    settableParams = [
        'verbose',
    ]

    def __init__(self,
        actualVideoFrequency=None,          # A shared value for publishing the actual video frequencies obtained from DAQ
        actualAudioFrequency=None,          # A shared value for publishing the actual audio frequencies obtained from DAQ
        requestedVideoFrequency=120,                     # The frequency in Hz of the video sync signal
        requestedAudioFrequency=44100,                   # The frequency in Hz of the audio sync signal
        videoSyncChannel=None,           # The counter channel on which to generate the video sync signal Dev3/ctr0
        videoDutyCycle=0.5,
        audioSyncChannel=None,           # The counter channel on which to generate the audio sync signal Dev3/ctr1
        audioDutyCycle=0.5,
        startTriggerChannel=None,             # A digital channel on which to wait for a start trigger signal. If this is none, sync process starts ASAP.
        startTime=None,                         # Shared value that is set when sync starts, used as start time by all processes (relevant for manual triggers)
        verbose=False,
        ready=None,                             # Synchronization barrier to ensure everyone's ready before beginning
        **kwargs):
        StateMachineProcess.__init__(self, **kwargs)
        # Store inputs in instance variables for later access
        self.ID = "S"
        self.actualAudioFrequency = actualAudioFrequency
        self.actualVideoFrequency = actualVideoFrequency
        self.startTime = startTime
        self.videoFrequency = requestedVideoFrequency
        self.audioFrequency = requestedAudioFrequency
        self.videoSyncChannel = videoSyncChannel
        self.audioSyncChannel = audioSyncChannel
        self.videoDutyCycle = videoDutyCycle
        self.audioDutyCycle = audioDutyCycle
        self.startTriggerChannel = startTriggerChannel
        self.ready = ready
        self.errorMessages = []
        self.verbose = verbose

    def setParams(self, **params):
        for key in params:
            if key in Synchronizer.settableParams:
                setattr(self, key, params[key])
                if self.verbose >= 1: self.log("Param set: {key}={val}".format(key=key, val=params[key]))
            else:
                if self.verbose >= 0: self.log("Param not settable: {key}={val}".format(key=key, val=params[key]))

    def run(self):
        self.PID.value = os.getpid()
        if self.verbose >= 1: self.log("PID={pid}".format(pid=os.getpid()))
        self.state = States.STOPPED
        self.nextState = States.STOPPED
        self.lastState = -1
        msg = ''; arg = None

        while True:
            # Publish updated state
            if self.state != self.lastState:
                self.updatePublishedState()

            try:
# ContinuousTriggerer: ************ STOPPPED *********************************
                if self.state == States.STOPPED:
                    # DO STUFF

                    # CHECK FOR MESSAGES
                    try:
                        msg, arg = self.msgQueue.get(block=True)
                        if msg == Synchronizer.SETPARAMS: self.setParams(**arg); msg = ''; arg=None
                    except queue.Empty: msg = ''; arg = None

                    # CHOOSE NEXT STATE
                    if self.exitFlag:
                        self.nextState = States.EXITING
                    elif msg == '':
                        self.nextState = self.state
                    elif msg == Synchronizer.STOP:
                        self.nextState = States.STOPPED
                    elif msg == Synchronizer.START:
                        self.nextState = States.INITIALIZING
                    elif msg == Synchronizer.EXIT:
                        self.exitFlag = True
                        self.nextState = States.EXITING
                    else:
                        raise SyntaxError("Message \"" + msg + "\" not relevant to " + self.stateList[self.state] + " state")
# Synchronizer: ******************* INITIALIZING *********************************
                elif self.state == States.INITIALIZING:
                    # DO STUFF

                    if self.ready.broken:
                        # Someone has already tried and failed to pass through the Barrier - reset it for everyone.
                        self.ready.reset()

                    # Configure and generate synchronization signal
                    if self.audioSyncChannel is None and self.videoSyncChannel is None:
                        trigTask = None
                        startTask = None
                        raise IOError("At least one audio or video sync channel must be specified.")
                    else:
                        trigTask = nidaqmx.Task()                       # Create task
                        if self.startTriggerChannel is not None:
                            startTask = nidaqmx.Task()
                        else:
                            startTask = None

                    if self.videoSyncChannel is not None:
                        trigTask.co_channels.add_co_pulse_chan_freq(
                            counter=self.videoSyncChannel,
                            name_to_assign_to_channel="videoFrequency",
                            units=nidaqmx.constants.FrequencyUnits.HZ,
                            initial_delay=0.0,
                            freq=self.videoFrequency,
                            duty_cycle=self.videoDutyCycle)     # Prepare a counter output channel for the video sync signal
                        if self.verbose >= 2:
                            self.log('Added video sync channel to task')
                    if self.audioSyncChannel is not None:
                        trigTask.co_channels.add_co_pulse_chan_freq(
                            counter=self.audioSyncChannel,
                            name_to_assign_to_channel="audioFrequency",
                            units=nidaqmx.constants.FrequencyUnits.HZ,
                            initial_delay=0.0,
                            freq=self.audioFrequency,
                            duty_cycle=self.audioDutyCycle)     # Prepare a counter output channel for the audio sync signal
                        if self.verbose >= 2:
                            self.log('Added audio sync channel to task')
                    # if (self.startTriggerChannel is not None) and ((self.videoSyncChannel is not None) or (self.audioSyncChannel is not None)):
                    #     # Configure task to wait for a digital pulse on the specified channel.
                    #     trigTask.triggers.arm_start_trigger.dig_edge_src=self.startTriggerChannel
                    #     trigTask.triggers.arm_start_trigger.trig_type=TriggerType.DIGITAL_EDGE
                    #     trigTask.triggers.arm_start_trigger.dig_edge_edge=Edge.RISING
                    trigTask.timing.cfg_implicit_timing(sample_mode=nidaqmx.constants.AcquisitionType.CONTINUOUS)

                    # Set shared values so other processes can get actual a/v frequencies
                    if self.audioSyncChannel is not None and self.actualAudioFrequency is not None:
                        self.actualAudioFrequency.value = trigTask.co_channels['audioFrequency'].co_pulse_freq
                        if self.verbose > 0: self.log('Requested audio frequency: ', self.audioFrequency, ' | actual audio frequency: ', self.actualAudioFrequency.value);
                    if self.videoSyncChannel is not None and self.actualVideoFrequency is not None:
                        self.actualVideoFrequency.value = trigTask.co_channels['videoFrequency'].co_pulse_freq
                        if self.verbose > 0: self.log('Requested video frequency: ', self.videoFrequency, ' | actual video frequency: ', self.actualVideoFrequency.value);

                    if startTask is not None:
                        # Add dummy write channel to force execute to block until task gets start trigger
#                        startTask.do_channels.add_do_chan(lines=self.startTriggerChannel)
                        # startTask.timing.cfg_samp_clk_timing(rate=10000, sample_mode=nidaqmx.constants.AcquisitionType.FINITE, samps_per_chan=1)
                        # # Configure task to wait for a digital pulse on the specified channel.
                        # startTask.triggers.arm_start_trigger.dig_edge_src=self.startTriggerChannel
                        # startTask.triggers.arm_start_trigger.trig_type=TriggerType.DIGITAL_EDGE
                        # startTask.triggers.arm_start_trigger.dig_edge_edge=Edge.RISING
                        startReader = DigitalSingleChannelReader(startTask.in_stream)  # Set up an analog stream reader
                        startTask.di_channels.add_di_chan(self.startTriggerChannel)
                        if self.verbose >= 2:
                            self.log('Added digital trigger channel to start task')

                        # startTask.timing.cfg_samp_clk_timing(                    # Configure clock source for triggering each analog read
                        #     rate=1000,
                        #     active_edge=nidaqmx.constants.Edge.RISING,
                        #     sample_mode=nidaqmx.constants.AcquisitionType.FINITE,
                        #     samps_per_chan=1)

                    # CHECK FOR MESSAGES
                    try:
                        msg, arg = self.msgQueue.get(block=False)
                        if msg == Synchronizer.SETPARAMS: self.setParams(**arg); msg = ''; arg=None
                    except queue.Empty: msg = ''; arg = None

                    # CHOOSE NEXT STATE
                    if self.exitFlag:
                        self.nextState = States.STOPPING
                    elif msg in ['', Synchronizer.START]:
                        self.nextState = States.WAITING
                    elif msg == Synchronizer.SYNC:
                        # Skip waiting state, go straight to READY
                        self.nextState = States.READY
                    elif msg == Synchronizer.STOP:
                        self.nextState = States.STOPPING
                    elif msg == Synchronizer.EXIT:
                        self.exitFlag = True
                        self.nextState = States.STOPPING
                    else:
                        raise SyntaxError("Message \"" + msg + "\" not relevant to " + self.stateList[self.state] + " state")
# Synchronizer: ******************* WAITING *********************************
                elif self.state == States.WAITING:
                    # DO STUFF
                    time.sleep(0.1)

                    # CHECK FOR MESSAGES
                    try:
                        msg, arg = self.msgQueue.get(block=False)
                        if msg == Synchronizer.SETPARAMS: self.setParams(**arg); msg = ''; arg=None
                    except queue.Empty: msg = ''; arg = None

                    # CHOOSE NEXT STATE
                    if msg in ['', Synchronizer.START]:
                        self.nextState = States.WAITING
                    elif msg == Synchronizer.SYNC:
                        self.nextState = States.READY
                    elif msg == Synchronizer.STOP:
                        self.nextState = States.STOPPING
                    elif msg == Synchronizer.EXIT:
                        self.exitFlag = True
                        self.nextState = States.STOPPING
                    else:
                        raise SyntaxError("Message \"" + msg + "\" not relevant to " + self.stateList[self.state] + " state")
# Synchronizer: ******************* READY *********************************
                elif self.state == States.READY:
                    # DO STUFF
                    try:
                        if self.verbose >= 2:
                            self.log('Ready to start task')

                        if self.ready is not None:
                            if self.verbose >= 2: self.log('Barrier: {n} others waiting, broken={b}'.format(n=self.ready.n_waiting, b=self.ready.broken))
                            if self.ready.broken:
                                self.ready.reset()
                            self.ready.wait()
                        passedBarrier = True

                        # To give audio and video processes a chance to get totally set up for acquiring, wait a second.
                        time.sleep(1)

                        if startTask is not None:
                            startTask.start()
                            while not startReader.read_one_sample_one_line():
                                pass
                            if self.verbose >= 2:
                                self.log("Got sync start trigger!")

                        preTime = time.time_ns()
                        trigTask.start()
                        postTime = time.time_ns()
                        self.startTime.value = (preTime + postTime) / 2000000000
                        if self.verbose >= 1: self.log("Sync task started at {time} s".format(time=self.startTime.value))
                        if self.verbose >= 1: self.log("Sync task startup took {time} s".format(time=(postTime - preTime)/1000000000))
                        if startTask is not None:
                            startTask.stop()
                    except BrokenBarrierError:
                        passedBarrier = False
                        if self.verbose >= 2: self.log("No simultaneous start - retrying")
                        time.sleep(0.1)

                    # CHECK FOR MESSAGES
                    try:
                        msg, arg = self.msgQueue.get(block=False)
                        if msg == Synchronizer.SETPARAMS: self.setParams(**arg); msg = ''; arg=None
                    except queue.Empty: msg = ''; arg = None

                    # CHOOSE NEXT STATE
                    if not passedBarrier:
                        self.nextState = States.READY
                    elif msg in ['', Synchronizer.START, Synchronizer.SYNC]:
                        self.nextState = States.SYNCHRONIZING
                    elif msg == Synchronizer.STOP:
                        self.nextState = States.STOPPING
                    elif msg == Synchronizer.EXIT:
                        self.exitFlag = True
                        self.nextState = States.STOPPING
                    else:
                        raise SyntaxError("Message \"" + msg + "\" not relevant to " + self.stateList[self.state] + " state")

# Synchronizer: ******************* SYNCHRONIZING *********************************
                elif self.state == States.SYNCHRONIZING:
                    # DO STUFF
                    if trigTask.is_task_done():
                        raise RuntimeError('Warning, synchronizer trigger task stopped unexpectedly.')

                    # CHECK FOR MESSAGES
                    try:
                        msg, arg = self.msgQueue.get(block=True, timeout=0.1)
                        if msg == Synchronizer.SETPARAMS: self.setParams(**arg); msg = ''; arg=None
                    except queue.Empty: msg = ''; arg = None

                    # CHOOSE NEXT STATE
                    if self.exitFlag:
                        self.nextState = States.STOPPING
                    elif msg in ['', Synchronizer.START, Synchronizer.SYNC]:
                        self.nextState = self.state
                    elif msg == Synchronizer.STOP:
                        self.nextState = States.STOPPING
                    elif msg == Synchronizer.EXIT:
                        self.exitFlag = True
                        self.nextState = States.STOPPING
                    else:
                        raise SyntaxError("Message \"" + msg + "\" not relevant to " + self.stateList[self.state] + " state")
# Synchronizer: ******************* STOPPING *********************************
                elif self.state == States.STOPPING:
                    # DO STUFF
                    if trigTask is not None:
                        trigTask.close()
                    if self.actualAudioFrequency is not None:
                        self.actualAudioFrequency.value = -1
                    if self.actualVideoFrequency is not None:
                        self.actualVideoFrequency.value = -1

                    # CHECK FOR MESSAGES
                    try:
                        msg, arg = self.msgQueue.get(block=False)
                        if msg == Synchronizer.SETPARAMS: self.setParams(**arg); msg = ''; arg=None
                    except queue.Empty: msg = ''; arg = None

                    # CHOOSE NEXT STATE
                    if self.exitFlag:
                        self.nextState = States.STOPPED
                    elif msg == '':
                        self.nextState = States.STOPPED
                    elif msg == Synchronizer.START:
                        self.nextState = States.INITIALIZING
                    elif msg == Synchronizer.STOP:
                        self.nextState = States.STOPPED
                    elif msg == Synchronizer.EXIT:
                        self.exitFlag = True
                        self.nextState = States.STOPPED
                    else:
                        raise SyntaxError("Message \"" + msg + "\" not relevant to " + self.stateList[self.state] + " state")
# Synchronizer: ******************* ERROR *********************************
                elif self.state == States.ERROR:
                    # DO STUFF
                    if self.verbose >= 0:
                        self.log("ERROR STATE. Error messages:\n\n")
                        self.log("\n\n".join(self.errorMessages))

                    self.updatePublishedInfo("\n".join(self.errorMessages))

                    self.errorMessages = []

                    # CHECK FOR MESSAGES
                    try:
                        msg, arg = self.msgQueue.get(block=False)
                        if msg == Synchronizer.SETPARAMS: self.setParams(**arg); msg = ''; arg=None
                    except queue.Empty: msg = ''; arg = None

                    # CHOOSE NEXT STATE
                    if self.lastState == States.ERROR:
                        # Error ==> Error, let's just exit
                        self.nextState = States.EXITING
                    elif msg == '':
                        if self.lastState == States.STOPPING:
                            # We got an error in stopping state? Better just stop.
                            self.nextState = States.STOPPED
                        elif self.lastState ==States.STOPPED:
                            # We got an error in the stopped state? Better just exit.
                            self.nextState = States.EXITING
                        else:
                            self.nextState = States.STOPPING
                    elif msg == Synchronizer.STOP:
                        self.nextState = States.STOPPED
                    elif msg == Synchronizer.EXIT:
                        self.exitFlag = True
                        if self.lastState == States.STOPPING:
                            self.nextState = States.EXITING
                        else:
                            self.nextState = States.STOPPING
                    else:
                        raise SyntaxError("Message \"" + msg + "\" not relevant to " + self.stateList[self.state] + " state")
# Synchronizer: ******************* EXIT *********************************
                elif self.state == States.EXITING:
                    break
                else:
                    raise KeyError("Unknown state: "+self.stateList[self.state])
            except KeyboardInterrupt:
                # Handle user using keyboard interrupt
                if self.verbose >= 0: self.log("Keyboard interrupt received - exiting")
                self.exitFlag = True
                self.nextState = States.STOPPING
            except:
                # HANDLE UNKNOWN ERROR
                self.errorMessages.append("Error in "+self.stateList[self.state]+" state\n\n"+traceback.format_exc())
                self.nextState = States.ERROR

            if (self.verbose >= 1 and (len(msg) > 0 or self.exitFlag)) or len(self.stdoutBuffer) > 0 or self.verbose >= 3:
                self.log("msg={msg}".format(msg=msg))
                self.logEnd()

            self.flushStdout()

            # Prepare to advance to next state
            self.lastState = self.state
            self.state = self.nextState

        clearQueue(self.msgQueue)
        if self.verbose >= 1: self.log("Synchronization process STOPPED")
        self.flushStdout()
        self.updatePublishedState(States.DEAD)

class AudioTriggerer(StateMachineProcess):
    '''
    AudioTriggerer: A self.state machine class to generate audio-based for both audio
        and video writer processes.
    '''

    # Human-readable states
    stateList = {
        States.WAITING :'WAITING',
        States.ANALYZING :'ANALYZING',
    }

    # Include common states from parent class
    stateList.update(StateMachineProcess.stateList)

    # Recognized message types:
    START = 'msg_start'
    STARTANALYZE = "msg_startanalyze"
    STOPANALYZE = "msg_stopanalyze"
    STOP = 'msg_stop'
    EXIT = 'msg_exit'
    SETPARAMS = 'msg_setParams'

    multiChannelBehaviors = ['OR', 'AND']

    # List of params that can be set externally with the 'msg_setParams' message
    settableParams = [
        'chunkSize',
        'triggerHighLevel',
        'triggerLowLevel',
        'triggerHighTime',
        'triggerLowTime',
        'triggerHighFraction',
        'triggerLowFraction',
        'maxAudioTriggerTime',
        'bandpassFrequencies',
        'butterworthOrder',
        'multiChannelStartBehavior',
        'multiChannelStopBehavior',
        'verbose',
        'scheduleEnabled',
        'scheduleStart',
        'scheduleStop',
        'tagTriggerEnabled',
        'writeTriggerEnabled'
        ]

    def __init__(self,
                audioQueue=None,
                audioFrequency=None,                # Shared var: Number of audio samples per second
                chunkSize=1000,                     # Number of audio samples per audio chunk
                triggerHighLevel=0.5,               # Volume level above which the audio must stay for triggerHighTime seconds to generate a start trigger
                triggerLowLevel=0.1,                # Volume level below which the audio must stay for triggerLowTime seconds to generate an updated (stop) trigger
                triggerHighTime=2,                  # Length of time that volume must stay above triggerHigh
                triggerLowTime=1,                   # Length of time that volume must stay below triggerLowLevel
                maxAudioTriggerTime=20,                  # Maximum length of trigger regardless of volume levels
                preTriggerTime=2,                   # Time before trigger to record
                triggerHighFraction=0.3,            # Fraction of high time that audio must be above high threshold for trigger start
                triggerLowFraction=0.3,             # Fraction of low time that audio must be below low threshold for trigger stop
                multiChannelStartBehavior='OR',     # How to handle multiple channels of audio. Either 'OR' (start when any channel goes higher than high threshold) or 'AND' (start when all channels go higher than high threshold)
                multiChannelStopBehavior='AND',     # How to handle multiple channels of audio. Either 'OR' (stop when any channel goes lower than low threshold) or 'AND' (stop when all channels go lower than low threshold)
                bandpassFrequencies = (100, 4000),  # A tuple of (lowfreq, highfreq) cutoff frequencies for bandpass filter
                butterworthOrder=6,
                scheduleEnabled=False,
                scheduleStartTime=None,
                scheduleStopTime=None,
                verbose=False,
                audioMessageQueue=None,             # Queue to send triggers to audio writers
                videoMessageQueues={},              # Queues to send triggers to video writers
                taggerQueues=None,
                tagTriggerEnabled=False,            # Enable sending triggers to writer queues?
                writeTriggerEnabled=False,          # Enable sending triggers to tagger queues?
                **kwargs):
        StateMachineProcess.__init__(self, **kwargs)
        self.audioQueue = audioQueue
        self.ID = "AT"
        if self.audioQueue is not None:
            self.audioQueue.cancel_join_thread()
        self.analysisMonitorQueue = mp.Queue()  # A queue to send analysis results to GUI for monitoring
        self.audioMessageQueue = audioMessageQueue
        self.videoMessageQueues = videoMessageQueues

        self.tagTriggerEnabled = tagTriggerEnabled      # Send triggers to tagger queues?
        self.writeTriggerEnabled = writeTriggerEnabled  # Send triggers to writer queues?
        self.taggerQueues = taggerQueues            # List of queues to processes that will use the audio triggers for tagging purposes

        self.audioFrequencyVar = audioFrequency
        self.audioFrequency = None
        self.chunkSize = chunkSize
        self.triggerHighLevel = triggerHighLevel
        self.triggerLowLevel = triggerLowLevel
        self.maxAudioTriggerTime = maxAudioTriggerTime
        self.preTriggerTime = preTriggerTime
        self.bandpassFrequencies = bandpassFrequencies
        self.butterworthOrder = butterworthOrder
        self.multiChannelStartBehavior = multiChannelStartBehavior
        self.multiChannelStopBehavior = multiChannelStopBehavior
        self.triggerHighTime = triggerHighTime
        self.triggerLowTime = triggerLowTime

        self.triggerHighChunks = None
        self.triggerLowChunks = None
        self.triggerHighFraction = triggerHighFraction
        self.triggerLowFraction = triggerLowFraction

        self.scheduleEnabled = scheduleEnabled
        self.scheduleStartTime = scheduleStartTime
        self.scheduleStopTime = scheduleStopTime

        # Generate butterworth filter coefficients...or something...
        self.filter = None

        self.errorMessages = []
        self.analyzeFlag = False
        self.verbose = verbose

        self.highLevelBuffer = None
        self.lowLevelBuffer = None

    def updateHighBuffer(self):
        self.triggerHighChunks = int(self.triggerHighTime * self.audioFrequency / self.chunkSize)
        if self.highLevelBuffer is not None:
            previousHighLevelBuffer = list(self.highLevelBuffer)
        else:
            previousHighLevelBuffer = []
        self.highLevelBuffer = deque(maxlen=self.triggerHighChunks)
        self.highLevelBuffer.extend(previousHighLevelBuffer)

    def updateLowBuffer(self):
        self.triggerLowChunks = int(self.triggerLowTime * self.audioFrequency / self.chunkSize)
        if self.lowLevelBuffer is not None:
            previousLowLevelBuffer = list(self.lowLevelBuffer)
        else:
            previousLowLevelBuffer = []
        self.lowLevelBuffer = deque(maxlen=self.triggerLowChunks)
        self.lowLevelBuffer.extend(previousLowLevelBuffer)

    def setParams(self, **params):
        for key in params:
            if key in AudioTriggerer.settableParams:
                setattr(self, key, params[key])
                if key in ["triggerHighTime", 'chunkSize']:
                    self.updateHighBuffer()
                if key in ["triggerLowTime", 'chunkSize']:
                    self.updateLowBuffer()
                if key == 'bandpassFrequencies' or key == 'butterworthOrder':
                    self.updateFilter()
                if self.verbose >= 1: self.log("Param set: {key}={val}".format(key=key, val=params[key]))
            else:
                if self.verbose >= 0: self.log("Param not settable: {key}={val}".format(key=key, val=params[key]))

    def run(self):
        self.PID.value = os.getpid()
        if self.verbose >= 1: self.log("PID={pid}".format(pid=os.getpid()))
        self.state = States.STOPPED
        self.nextState = States.STOPPED
        self.lastState = -1
        msg = ''; arg = None

        while True:
            # Publish updated state
            if self.state != self.lastState:
                self.updatePublishedState()

            try:
# AudioTriggerer: ************ STOPPPED *********************************
                if self.state == States.STOPPED:
                    # DO STUFF

                    # CHECK FOR MESSAGES
                    try:
                        msg, arg = self.msgQueue.get(block=True)
                        if msg == AudioTriggerer.SETPARAMS: self.setParams(**arg); msg = ''; arg=None
                    except queue.Empty: msg = ''; arg = None

                    # CHOOSE NEXT STATE
                    if msg == AudioTriggerer.EXIT or self.exitFlag:
                        self.exitFlag = True
                        self.nextState = States.EXITING
                    elif msg == '':
                        self.nextState = States.STOPPED
                    elif msg == AudioTriggerer.STOP:
                        self.nextState = States.STOPPED
                    elif msg == AudioTriggerer.START:
                        self.nextState = States.INITIALIZING
                    elif msg == AudioTriggerer.STARTANALYZE:
                        self.analyzeFlag = True
                        self.nextState = States.INITIALIZING
                    else:
                        raise SyntaxError("Message \"" + msg + "\" not relevant to " + self.stateList[self.state] + " state")
# AudioTriggerer: ***************** INITIALIZING *****************************
                elif self.state == States.INITIALIZING:
                    # DO STUFF
                    activeTrigger = None
                    self.audioFrequency = None

                    if self.audioFrequencyVar.value == -1:
                        # Wait for shared value audioFrequency to be set by the Synchronizer process
                        time.sleep(0.1)
                    else:
                        self.audioFrequency = self.audioFrequencyVar.value

                        self.updateFilter()
                        self.updateHighBuffer()
                        self.updateLowBuffer()

                    # CHECK FOR MESSAGES
                    try:
                        msg, arg = self.msgQueue.get(block=False)
                        if msg == AudioTriggerer.SETPARAMS: self.setParams(**arg); msg = ''; arg=None
                    except queue.Empty: msg = ''; arg = None

                    # CHOOSE NEXT STATE
                    if msg == AudioTriggerer.EXIT or self.exitFlag:
                        self.exitFlag = True
                        self.nextState = States.STOPPING
                    elif msg == AudioTriggerer.STOP:
                        self.nextState = States.STOPPING
                    elif msg in ['', AudioTriggerer.START, AudioTriggerer.STOPANALYZE]:
                        if self.audioFrequency is None:
                            self.nextState = States.INITIALIZING
                        else:
                            self.nextState = States.WAITING
                    elif msg == AudioTriggerer.STARTANALYZE or self.analyzeFlag:
                        self.nextState = States.ANALYZING
                    else:
                        raise SyntaxError("Message \"" + msg + "\" not relevant to " + self.stateList[self.state] + " state")
# AudioTriggerer: ***************** WAITING ********************************
                elif self.state == States.WAITING:
                    # DO STUFF

                    # Throw away received audio
                    try:
                        self.audioQueue.get(block=True, timeout=None)
                    except queue.Empty:
                        # No audio data available, no prob.
                        pass

                    # CHECK FOR MESSAGES
                    try:
                        msg, arg = self.msgQueue.get(block=False)
                        if msg == AudioTriggerer.SETPARAMS: self.setParams(**arg); msg = ''; arg=None
                    except queue.Empty: msg = ''; arg = None

                    # CHOOSE NEXT STATE
                    if msg == AudioTriggerer.STOP:
                        self.nextState = States.STOPPING
                    elif msg == AudioTriggerer.EXIT:
                        self.exitFlag = True
                        self.nextState = States.STOPPING
                    elif msg == AudioTriggerer.STARTANALYZE or self.analyzeFlag:
                        self.nextState = States.ANALYZING
                    elif msg in ['', AudioTriggerer.STOPANALYZE, AudioTriggerer.START]:
                        self.nextState = States.WAITING
                    else:
                        raise SyntaxError("Message \"" + msg + "\" not relevant to " + self.stateList[self.state] + " state")
# AudioTriggerer: ***************** ANALYZING *********************************
                elif self.state == States.ANALYZING:
                    # DO STUFF
                    self.analyzeFlag = False

                    # n = number of time steps, c = number of channels
                    # Get audio chunk from audio acquirer (c x n)
                    try:
                        chunkStartTime, audioChunk = self.audioQueue.get(block=True, timeout=0.1)
                        chunkEndTime = chunkStartTime + self.chunkSize / self.audioFrequency

                        currentTimeOfDay = dt.datetime.now()
                        if not self.scheduleEnabled or (self.scheduleStartTime <= currentTimeOfDay and self.scheduleStopTime <= currentTimeOfDay):
                            # If the scheduling feature is disabled, or it's enabled and we're between start/stop times, then:

                            # Bandpass filter audio (c x n)
                            filteredAudioChunk = self.bandpass(audioChunk)
                            # Center audio - remove mean (c x n)
                            filteredCenteredAudioChunk = (filteredAudioChunk.transpose() - filteredAudioChunk.mean(axis=1)).transpose()
                            # RMS audio (c x 1)
                            volume = np.sqrt((filteredCenteredAudioChunk ** 2).mean(axis=1))
                            # Threshold
                            high = volume > self.triggerHighLevel
                            low = volume < self.triggerLowLevel
                            # Enqueue new high/low indication
                            self.highLevelBuffer.append(high)
                            self.lowLevelBuffer.append(low)
                            # Calculate fraction of high/low monitoring time signals have been higher than high level or lower than low level
                            highChunks = sum(self.highLevelBuffer)
                            lowChunks = sum(self.lowLevelBuffer)
                            # Calculate fraction of high/low monitoring time audio has been above/below high/low level
                            highFrac = highChunks / self.triggerHighChunks
                            lowFrac = lowChunks / self.triggerLowChunks
                            # Check if levels have been high/low for long enough
                            highTrigger = highFrac >= self.triggerHighFraction
                            lowTrigger = lowFrac >= self.triggerLowFraction
                            # Combine channel outcomes into a single trigger outcome using specified behavior
                            if self.multiChannelStartBehavior == "OR":
                                highTrigger = highTrigger.any()
                            elif self.multiChannelStartBehavior == "AND":
                                highTrigger = highTrigger.all()
                            if self.multiChannelStopBehavior == "OR":
                                lowTrigger = lowTrigger.any()
                            elif self.multiChannelStopBehavior == "AND":
                                lowTrigger = lowTrigger.all()

                            # print("highTrigger = ", highTrigger, "lowTrigger = ", lowTrigger)

                            # Check if current active trigger is expired, and delete it if it is
                            # print("Chunk bounds:  ", chunkStartTime, '-', chunkEndTime)
                            if activeTrigger is not None:
                                # print("Trigger bounds:", activeTrigger.startTime, '-', activeTrigger.endTime)
                                if activeTrigger.state(chunkStartTime) > 0 and activeTrigger.state(chunkEndTime) > 0:
                                    # Entire chunk is after the end of the trigger period
                                    # print("Deleting active trigger (it's out of range)")
                                    activeTrigger = None
                            if activeTrigger is None and highTrigger:
                                # Send new trigger! Set to record preTriggerTime before the chunk start, and end maxAudioTriggerTime later.
                                #   If volumes go low enough for long enough, we will send an updated trigger with a new stop time
                                # print("Sending new trigger")
                                activeTrigger = Trigger(
                                    startTime = chunkStartTime - self.preTriggerTime,
                                    triggerTime = chunkStartTime,
                                    endTime = chunkStartTime - self.preTriggerTime + self.maxAudioTriggerTime,
                                    tags = set(['A']),
                                    idspace = self.ID)
                                self.sendTrigger(activeTrigger)
                                if self.verbose >= 1: self.log("Send new trigger: {t}".format(t=activeTrigger))
                            elif activeTrigger is not None and lowTrigger:
                                # Send updated trigger
                                # print("Sending updated stop trigger")
                                activeTrigger.endTime = chunkStartTime
                                # print("Setting trigger stop time to", activeTrigger.endTime)
                                self.sendTrigger(activeTrigger)
                                if self.verbose >= 1: self.log("Update trigger to stop now: {t}".format(t=activeTrigger))

                            # Send analysis summary of this chunk to the GUI
                            summary = dict(
                                volume=volume,
                                triggerHighLevel=self.triggerHighLevel,
                                triggerLowLevel=self.triggerLowLevel,
                                low=low,
                                high=high,
                                lowChunks=lowChunks,
                                highChunks=highChunks,
                                triggerLowChunks=self.triggerLowChunks,
                                triggerHighChunks=self.triggerHighChunks,
                                highFrac=highFrac,
                                lowFrac=lowFrac,
                                lowTrigger=lowTrigger,
                                highTrigger=highTrigger,
                                triggerLowFrac=self.triggerLowFraction,
                                triggerHighFrac=self.triggerHighFraction,
                                chunkStartTime=chunkStartTime,
                                chunkSize = self.chunkSize,
                                audioFrequency = self.audioFrequency,
                                activeTrigger = activeTrigger
                            )
                            self.analysisMonitorQueue.put(summary)

                            if self.verbose >= 3: self.log('h% =', highFrac, 'l% =', lowFrac, 'hT =', highTrigger, 'lT =', lowTrigger)

                    except queue.Empty:
                        pass # No audio data to analyze

                    # CHECK FOR MESSAGES
                    try:
                        msg, arg = self.msgQueue.get(block=False)
                        if msg == AudioTriggerer.SETPARAMS: self.setParams(**arg); msg = ''; arg=None
                    except queue.Empty: msg = ''; arg = None

                    # CHOOSE NEXT STATE
#                    if self.verbose >= 3: self.log("|{startState} ---- {endState}|".format(startState=chunkStartTriggerState, endState=chunkEndTriggerState))
                    if msg == AudioTriggerer.EXIT or self.exitFlag:
                        self.exitFlag = True
                        self.nextState = States.STOPPING
                    elif msg == AudioTriggerer.STOP:
                        self.nextState = States.STOPPING
                    elif msg == AudioTriggerer.STOPANALYZE:
                        self.nextState = States.WAITING
                    elif msg in ['', AudioTriggerer.STARTANALYZE, AudioTriggerer.START]:
                        self.nextState = self.state
                    else:
                        raise SyntaxError("Message \"" + msg + "\" not relevant to " + self.stateList[self.state] + " state")
# AudioTriggerer: ***************** STOPPING *********************************
                elif self.state == States.STOPPING:
                    # DO STUFF

                    # CHECK FOR MESSAGES
                    try:
                        msg, arg = self.msgQueue.get(block=False)
                        if msg == AudioTriggerer.SETPARAMS: self.setParams(**arg); msg = ''; arg=None
                    except queue.Empty: msg = ''; arg = None

                    # CHOOSE NEXT STATE
                    if self.exitFlag:
                        self.nextState = States.STOPPED
                    elif msg == '':
                        self.nextState = States.STOPPED
                    elif msg == AudioTriggerer.STOP:
                        self.nextState = States.STOPPED
                    elif msg == AudioTriggerer.EXIT:
                        self.exitFlag = True
                        self.nextState = States.STOPPED
                    elif msg == AudioTriggerer.START:
                        self.nextState = States.INITIALIZING
                    else:
                        raise SyntaxError("Message \"" + msg + "\" not relevant to " + self.stateList[self.state] + " state")
# AudioTriggerer: ***************** ERROR *********************************
                elif self.state == States.ERROR:
                    # DO STUFF
                    if self.verbose >= 0:
                        self.log("ERROR STATE. Error messages:\n\n")
                        self.log("\n\n".join(self.errorMessages))

                    self.updatePublishedInfo("\n".join(self.errorMessages))

                    self.errorMessages = []

                    # CHECK FOR MESSAGES
                    try:
                        msg, arg = self.msgQueue.get(block=False)
                        if msg == AudioTriggerer.SETPARAMS: self.setParams(**arg); msg = ''; arg=None
                    except queue.Empty: msg = ''; arg = None

                    # CHOOSE NEXT STATE
                    if self.lastState == States.ERROR:
                        # Error ==> Error, let's just exit
                        self.nextState = States.EXITING
                    elif msg == '':
                        if self.lastState == States.STOPPING:
                            # We got an error in stopping state? Better just stop.
                            self.nextState = States.STOPPED
                        elif self.lastState ==States.STOPPED:
                            # We got an error in the stopped state? Better just exit.
                            self.nextState = States.EXITING
                        else:
                            self.nextState = States.STOPPING
                    elif msg == AudioTriggerer.STOP:
                        self.nextState = States.STOPPED
                    elif msg == AudioTriggerer.EXIT:
                        self.exitFlag = True
                        if self.lastState == States.STOPPING:
                            self.nextState = States.EXITING
                        else:
                            self.nextState = States.STOPPING
                    else:
                        raise SyntaxError("Message \"" + msg + "\" not relevant to " + self.stateList[self.state] + " state")
# AudioTriggerer: ***************** EXIT *********************************
                elif self.state == States.EXITING:
                    break
                else:
                    raise KeyError("Unknown state: "+self.stateList[self.state])
            except KeyboardInterrupt:
                # Handle user using keyboard interrupt
                if self.verbose >= 0: self.log("Keyboard interrupt received - exiting")
                self.exitFlag = True
                self.nextState = States.STOPPING
            except:
                # HANDLE UNKNOWN ERROR
                self.errorMessages.append("Error in "+self.stateList[self.state]+" state\n\n"+traceback.format_exc())
                self.nextState = States.ERROR

            if (self.verbose >= 1 and (len(msg) > 0 or self.exitFlag)) or len(self.stdoutBuffer) > 0 or self.verbose >= 3:
                self.log("msg={msg}".format(msg=msg))
                self.logEnd()

            self.flushStdout()

            # Prepare to advance to next state
            self.lastState = self.state
            self.state = self.nextState

        clearQueue(self.msgQueue)
        clearQueue(self.analysisMonitorQueue)
        if self.verbose >= 1: self.log("Audio write process STOPPED")

        self.flushStdout()
        self.updatePublishedState(States.DEAD)

    def updateFilter(self):
        self.filter = generateButterBandpassCoeffs(self.bandpassFrequencies[0], self.bandpassFrequencies[1], self.audioFrequency, order=self.butterworthOrder)

    def thresholds(self, audioChunk):
        # Return (belowLow, belowHigh) where belowLow is true if the audioChunk
        # is below the low level, and aboveHigh is true if the audioChunk is
        # above the high level
        pass

    def bandpass(self, audioChunk):
        b, a = self.filter
        y = lfilter(b, a, audioChunk, axis=1)  # axis=
        return y

    def sendTrigger(self, trigger):
        if self.writeTriggerEnabled:
            self.audioMessageQueue.put((AudioWriter.TRIGGER, trigger))
            for camSerial in self.videoMessageQueues:
                self.videoMessageQueues[camSerial].put((VideoWriter.TRIGGER, trigger))
        if self.tagTriggerEnabled:
            for queue in self.taggerQueues:
                queue.put((ContinuousTriggerer.TAGTRIGGER, trigger))

class AudioAcquirer(StateMachineProcess):
    # Class for acquiring an audio signal (or any analog signal) at a rate that
    #   is synchronized to the rising edges on the specified synchronization
    #   channel.

    # Human-readable states
    stateList = {
        States.ACQUIRING :'ACQUIRING',
        States.READY :'ACQUIRE_READY',
    }

    # Include common states from parent class
    stateList.update(StateMachineProcess.stateList)

    # Recognized message types:
    START = 'msg_start'
    STOP = 'msg_stop'
    EXIT = 'msg_exit'
    SETPARAMS = 'msg_setParams'

    # List of params that can be set externally with the 'msg_setParams' message
    settableParams = [
        'verbose',
        'copyToMonitoringQueue',
        'copyToAnalysisQueue'
    ]

    def __init__(self,
                startTime=None,
                audioQueue = None,                  # A multiprocessing queue to send data to another proces for writing to disk
                chunkSize = 4410,                   # Size of the read chunk in samples
                audioFrequency = 44100,               # Maximum expected rate of the specified synchronization channel
                bufferSize = None,                  # Size of device buffer. Defaults to 1 second's worth of data
                channelNames = [],                  # Channel name for analog input (microphone signal)
                channelConfig = "DEFAULT",
                syncChannel = None,                 # Channel name for synchronization source
                verbose = False,
                ready=None,                         # Synchronization barrier to ensure everyone's ready before beginning
                copyToMonitoringQueue=True,         # Should images be also sent to the monitoring queue?
                copyToAnalysisQueue=True,           # Should images be also sent to the analysis queue?
                **kwargs):
        StateMachineProcess.__init__(self, **kwargs)
        # Store inputs in instance variables for later access
        self.ID = "AA"
        self.copyToMonitoringQueue = copyToMonitoringQueue
        self.copyToAnalysisQueue = copyToAnalysisQueue
        self.startTimeSharedValue = startTime
        self.audioFrequencyVar = audioFrequency
        self.audioFrequency = None
        self.acquireTimeout = 1 #2*chunkSize / self.audioFrequency
        self.audioQueue = audioQueue
        if self.audioQueue is not None:
            self.audioQueue.cancel_join_thread()
        self.monitorQueue = mp.Queue()      # A multiprocessing queue to send data to the UI to monitor the audio
        self.analysisQueue = mp.Queue()    # A multiprocessing queue to send data to the audio triggerer process for analysis
        # if len(self.monitorQueue) > 0:
        #     self.monitorQueue.cancel_join_thread()
        self.chunkSize = chunkSize
        self.inputChannels = channelNames
        if channelConfig == "DEFAULT":
            self.channelConfig = nidaqmx.constants.TerminalConfiguration.DEFAULT
        elif channelConfig == "DIFFERENTIAL":
            self.channelConfig = nidaqmx.constants.TerminalConfiguration.DIFFERENTIAL
        elif channelConfig == "NRSE":
            self.channelConfig = nidaqmx.constants.TerminalConfiguration.NRSE
        elif channelConfig == "PSEUDODIFFERENTIAL":
            self.channelConfig = nidaqmx.constants.TerminalConfiguration.PSEUDODIFFERENTIAL
        elif channelConfig == "RSE":
            self.channelConfig = nidaqmx.constants.TerminalConfiguration.RSE
        self.syncChannel = syncChannel
        self.ready = ready
        self.errorMessages = []
        self.verbose = verbose
        self.exitFlag = False

    def setParams(self, **params):
        for key in params:
            if key in AudioAcquirer.settableParams:
                setattr(self, key, params[key])
                if self.verbose >= 1: self.log("Param set: {key}={val}".format(key=key, val=params[key]))
            else:
                if self.verbose >= 0: self.log("Param not settable: {key}={val}".format(key=key, val=params[key]))

    def rescaleAudio(data, maxV=10, minV=-10, maxD=32767, minD=-32767):
        return (data * ((maxD-minD)/(maxV-minV))).astype('int16')

    def run(self):
        self.PID.value = os.getpid()
        if self.verbose >= 1: self.log("PID={pid}".format(pid=os.getpid()))
        self.state = States.STOPPED
        self.nextState = States.STOPPED
        self.lastState = -1
        msg = ''; arg = None

        while True:
            # Publish updated state
            if self.state != self.lastState:
                self.updatePublishedState()

            try:
# AudioAcquirer: ***************** STOPPPED *********************************
                if self.state == States.STOPPED:
                    # DO STUFF

                    # CHECK FOR MESSAGES
                    try:
                        msg, arg = self.msgQueue.get(block=True)
                        if msg == AudioAcquirer.SETPARAMS: self.setParams(**arg); msg = ''; arg=None
                    except queue.Empty: msg = ''; arg = None

                    # CHOOSE NEXT STATE
                    if self.exitFlag:
                        self.nextState = States.EXITING
                    elif msg == '':
                        self.nextState = self.state
                    elif msg == AudioAcquirer.STOP:
                        self.nextState = States.STOPPED
                    elif msg == AudioAcquirer.START:
                        self.nextState = States.INITIALIZING
                    elif msg == AudioAcquirer.EXIT:
                        self.exitFlag = True
                        self.nextState = States.EXITING
                    else:
                        raise SyntaxError("Message \"" + msg + "\" not relevant to " + self.stateList[self.state] + " state")
# AudioAcquirer: ****************** INITIALIZING *********************************
                elif self.state == States.INITIALIZING:
                    # DO STUFF
                    self.audioFrequency = None

                    # Read actual audio frequency from the Synchronizer process
                    if self.audioFrequencyVar.value == -1:
                        # Wait for shared value audioFrequency to be set by the Synchronizer process
                        time.sleep(0.1)
                    else:
                        self.audioFrequency = self.audioFrequencyVar.value

                        data = np.zeros((len(self.inputChannels), self.chunkSize), dtype='float')   # A pre-allocated array to receive audio data

                        processedData = data.copy()
                        readTask = nidaqmx.Task(new_task_name="audioTask")                            # Create task
                        reader = AnalogMultiChannelReader(readTask.in_stream)  # Set up an analog stream reader
                        for inputChannel in self.inputChannels:
                            readTask.ai_channels.add_ai_voltage_chan(               # Set up analog input channel
                                inputChannel,
                                terminal_config=self.channelConfig,
                                max_val=10,
                                min_val=-10)
                        readTask.timing.cfg_samp_clk_timing(                    # Configure clock source for triggering each analog read
                            rate=self.audioFrequency,
                            source=self.syncChannel,                            # Specify a timing source!
                            active_edge=nidaqmx.constants.Edge.RISING,
                            sample_mode=nidaqmx.constants.AcquisitionType.CONTINUOUS,
                            samps_per_chan=self.chunkSize)
                        startTime = None
                        sampleCount = 0

                    # CHECK FOR MESSAGES
                    try:
                        msg, arg = self.msgQueue.get(block=False)
                        if msg == AudioAcquirer.SETPARAMS: self.setParams(**arg); msg = ''; arg=None
                    except queue.Empty: msg = ''; arg = None

                    # CHOOSE NEXT STATE
                    if self.exitFlag:
                        self.nextState = States.STOPPING
                    elif msg in ['', AudioAcquirer.START]:
                        if self.audioFrequency is None:
                            self.nextState = States.INITIALIZING
                        else:
                            self.nextState = States.READY
                    elif msg == AudioAcquirer.STOP:
                        self.nextState = States.STOPPING
                    elif msg == AudioAcquirer.EXIT:
                        self.exitFlag = True
                        self.nextState = States.STOPPING
                    else:
                        raise SyntaxError("AA - Message \"" + msg + "\" not relevant to " + self.stateList[self.state] + " state")
# AudioAcquirer: ****************** READY *********************************
                elif self.state == States.READY:
                    # DO STUFF
                    try:
                        if self.ready is not None:
                            self.ready.wait()
                        passedBarrier = True
                    except BrokenBarrierError:
                        passedBarrier = False
                        if self.verbose >= 2: self.log("No simultaneous start - retrying")
                        time.sleep(0.1)

#                    if self.verbose >= 1: self.log('passed barrier')

                    # CHECK FOR MESSAGES
                    try:
                        msg, arg = self.msgQueue.get(block=False)
                        if msg == AudioAcquirer.SETPARAMS: self.setParams(**arg); msg = ''; arg=None
                    except queue.Empty: msg = ''; arg = None

                    # CHOOSE NEXT STATE
                    if not passedBarrier:
                        self.nextState = States.READY
                    elif msg in ['', AudioAcquirer.START]:
                        self.nextState = States.ACQUIRING
                    elif msg == AudioAcquirer.STOP:
                        self.nextState = States.STOPPING
                    elif msg == AudioAcquirer.EXIT:
                        self.exitFlag = True
                        self.nextState = States.STOPPING
                    else:
                        raise SyntaxError("Message \"" + msg + "\" not relevant to " + self.stateList[self.state] + " state")
# AudioAcquirer: ****************** ACQUIRING *********************************
                elif self.state == States.ACQUIRING:
                    # DO STUFF
                    try:
                        reader.read_many_sample(                            # Read a chunk of audio data
                            data,
                            number_of_samples_per_channel=self.chunkSize,
                            timeout=self.acquireTimeout)

                        # Get timestamp of first audio chunk acquisition
                        if startTime is None:
                            if self.verbose >= 1: self.log("Getting start time from sync process...")
                            while startTime == -1 or startTime is None:
                                startTime = self.startTimeSharedValue.value
                            if self.verbose >= 1: self.log("Got start time from sync process: "+str(startTime))
#                            startTime = time.time_ns() / 1000000000 - self.chunkSize / self.audioFrequency

                        chunkStartTime = startTime + sampleCount / self.audioFrequency
                        sampleCount += self.chunkSize
                        if self.verbose >= 3: self.log('# samples:'+str(sampleCount))
                        processedData = AudioAcquirer.rescaleAudio(data)
                        audioChunk = AudioChunk(chunkStartTime = chunkStartTime, audioFrequency = self.audioFrequency, data = processedData, idspace=self.ID)
                        if self.audioQueue is not None:
                            self.audioQueue.put(audioChunk)              # If a data queue is provided, queue up the new data
                        else:
                            if self.verbose >= 2: self.log('' + processedData)

                        # Copy audio data for monitoring queues
                        monitorDataCopy = np.copy(data)

                        if self.copyToMonitoringQueue and self.monitorQueue is not None:
                            self.monitorQueue.put((self.inputChannels, chunkStartTime, monitorDataCopy))      # If a monitoring queue is provided, queue up the data
                        if self.copyToAnalysisQueue and self.analysisQueue is not None:
                            self.analysisQueue.put((chunkStartTime, monitorDataCopy))

                        if self.verbose >= 3:
                            self.log('Queue sizes:')
                            self.log('        Main:', self.audioQueue.qsize())
                            self.log('  Monitoring:', self.monitorQueue.qsize())
                            self.log('    Analysis:', self.analysisQueue.qsize())
                    except nidaqmx.errors.DaqError:
#                        traceback.print_exc()
                        if self.verbose >= 0: self.log("Audio Chunk acquisition timed out.")

                    # CHECK FOR MESSAGES
                    try:
                        msg, arg = self.msgQueue.get(block=False)
                        if msg == AudioAcquirer.SETPARAMS: self.setParams(**arg); msg = ''; arg=None
                    except queue.Empty: msg = ''; arg = None

                    # CHOOSE NEXT STATE
                    if self.exitFlag:
                        self.nextState = States.STOPPING
                    elif msg in ['', AudioAcquirer.START]:
                        self.nextState = States.ACQUIRING
                    elif msg == AudioAcquirer.STOP:
                        self.nextState = States.STOPPING
                    elif msg == AudioAcquirer.EXIT:
                        self.exitFlag = True
                        self.nextState = States.STOPPING
                    else:
                        raise SyntaxError("Message \"" + msg + "\" not relevant to " + self.stateList[self.state] + " state")
# AudioAcquirer: ****************** STOPPING *********************************
                elif self.state == States.STOPPING:
                    # DO STUFF
                    if readTask is not None:
                        readTask.close()

                    # CHECK FOR MESSAGES
                    try:
                        msg, arg = self.msgQueue.get(block=False)
                        if msg == AudioAcquirer.SETPARAMS: self.setParams(**arg); msg = ''; arg=None
                    except queue.Empty: msg = ''; arg = None

                    # CHOOSE NEXT STATE
                    if self.exitFlag:
                        self.nextState = States.STOPPED
                    elif msg == '':
                        self.nextState = States.STOPPED
                    elif msg == AudioAcquirer.STOP:
                        self.nextState = States.STOPPED
                    elif msg == AudioAcquirer.EXIT:
                        self.exitFlag = True
                        self.nextState = States.STOPPED
                    elif msg == AudioAcquirer.START:
                        self.nextState = States.INITIALIZING
                    else:
                        raise SyntaxError("Message \"" + msg + "\" not relevant to " + self.stateList[self.state] + " state")
# AudioAcquirer: ****************** ERROR *********************************
                elif self.state == States.ERROR:
                    # DO STUFF
                    if self.verbose >= 0:
                        self.log("ERROR STATE. Error messages:\n\n")
                        self.log("\n\n".join(self.errorMessages))

                    self.updatePublishedInfo("\n".join(self.errorMessages))

                    self.errorMessages = []

                    # CHECK FOR MESSAGES
                    try:
                        msg, arg = self.msgQueue.get(block=False)
                        if msg == AudioAcquirer.SETPARAMS: self.setParams(**arg); msg = ''; arg=None
                    except queue.Empty: msg = ''; arg = None

                    # CHOOSE NEXT STATE
                    if self.lastState == States.ERROR:
                        # Error ==> Error, let's just exit
                        self.nextState = States.EXITING
                    elif msg == '':
                        if self.lastState == States.STOPPING:
                            # We got an error in stopping state? Better just stop.
                            self.nextState = States.STOPPED
                        elif self.lastState == States.STOPPED:
                            # We got an error in the stopped state? Better just exit.
                            self.nextState = States.EXITING
                        else:
                            self.nextState = States.STOPPING
                    elif msg == AudioAcquirer.STOP:
                        self.nextState = States.STOPPING
                    elif msg == AudioAcquirer.EXIT:
                        self.exitFlag = True
                        if self.lastState == States.STOPPING:
                            self.nextState = States.EXITING
                        else:
                            self.nextState = States.STOPPING
                    else:
                        raise SyntaxError("Message \"" + msg + "\" not relevant to " + self.stateList[self.state] + " state")
# AudioAcquirer: ****************** EXIT *********************************
                elif self.state == States.EXITING:
                    if self.verbose >= 1: self.log('Exiting!')
                    break
                else:
                    raise KeyError("Unknown state: "+self.stateList[self.state])
            except KeyboardInterrupt:
                # Handle user using keyboard interrupt
                if self.verbose >= 1: self.log("Keyboard interrupt received - exiting")
                self.exitFlag = True
                self.nextState = States.STOPPING
            except:
                # HANDLE UNKNOWN ERROR
                self.errorMessages.append("Error in "+self.stateList[self.state]+" state\n\n"+traceback.format_exc())
                self.nextState = States.ERROR

            if (self.verbose >= 1 and (len(msg) > 0 or self.exitFlag)) or len(self.stdoutBuffer) > 0 or self.verbose >= 3:
                self.log("msg={msg}".format(msg=msg))
                self.logEnd()

            self.flushStdout()

            # Prepare to advance to next state
            self.lastState = self.state
            self.state = self.nextState

        clearQueue(self.msgQueue)
        clearQueue(self.monitorQueue)
        clearQueue(self.analysisQueue)
        if self.verbose >= 1: self.log("Audio acquire process STOPPED")

        self.flushStdout()
        self.updatePublishedState(States.DEAD)

class SimpleAudioWriter(StateMachineProcess):
    # Human-readable states
    stateList = {
        States.WRITING :'WRITING',
        States.AUDIOINIT:'AUDIOINIT',
    }

    # Include common states from parent class
    stateList.update(StateMachineProcess.stateList)

    # Recognized message types:
    START = 'msg_start'
    STOP = 'msg_stop'
    EXIT = 'msg_exit'
    SETPARAMS = 'msg_setParams'

    # List of params that can be set externally with the 'msg_setParams' message
    settableParams = [
        'verbose',
        'audioBaseFileName',
        'audioDirectory',
        'daySubfolders',
        'enableWrite',
        'scheduleEnabled',
        'scheduleStart',
        'scheduleStop'
        ]

    def __init__(self,
                audioDirectory='.',
                audioBaseFileName='audioFile',
                channelNames=[],
                audioQueue=None,
                audioFrequency=None,    # A shared variable for audioFrequency
                frameRate=None,         # A shared variable for video framerate (needed to ensure audio sync)
                numChannels=1,
                videoLength=None,       # Requested time in seconds of each video.
                verbose=False,
                audioDepthBytes=2,
                mergeMessageQueue=None, # Queue to put (filename, trigger) in for merging
                daySubfolders=True,         # Create and write to subfolders labeled by day?
                enableWrite=True,
                scheduleEnabled=False,
                scheduleStartTime=None,
                scheduleStopTime=None,
                **kwargs):
        StateMachineProcess.__init__(self, **kwargs)
        self.ID = "SAW"
        self.audioDirectory = audioDirectory
        self.audioBaseFileName = audioBaseFileName
        self.channelNames = channelNames
        self.audioQueue = audioQueue
        if self.audioQueue is not None:
            self.audioQueue.cancel_join_thread()
        self.audioFrequencyVar = audioFrequency
        self.audioFrequency = None
        self.frameRateVar = frameRate
        self.frameRate = None
        self.numChannels = numChannels
        self.videoLength = videoLength
        self.mergeMessageQueue = mergeMessageQueue
        self.errorMessages = []
        self.verbose = verbose
        self.audioDepthBytes = audioDepthBytes
        self.daySubfolders = daySubfolders
        self.enableWrite = enableWrite
        self.scheduleEnabled = scheduleEnabled
        self.scheduleStartTime = scheduleStartTime
        self.scheduleStopTime = scheduleStopTime

    def setParams(self, **params):
        for key in params:
            if key in SimpleAudioWriter.settableParams:
                setattr(self, key, params[key])
                if self.verbose >= 1: self.log("Param set: {key}={val}".format(key=key, val=params[key]))
            else:
                if self.verbose >= 0: self.log("Param not settable: {key}={val}".format(key=key, val=params[key]))

    def run(self):
        self.PID.value = os.getpid()
        if self.verbose >= 1: self.log("PID={pid}".format(pid=os.getpid()))
        self.state = States.STOPPED
        self.nextState = States.STOPPED
        self.lastState = -1
        msg = ''; arg = None

        while True:
            # Publish updated state
            if self.state != self.lastState:
                self.updatePublishedState()

            try:
# SimpleAudioWriter: **************** STOPPPED *********************************
                if self.state == States.STOPPED:
                    # DO STUFF

                    # CHECK FOR MESSAGES
                    try:
                        msg, arg = self.msgQueue.get(block=True)
                        if msg == SimpleAudioWriter.SETPARAMS: self.setParams(**arg); msg = ''; arg=None
                    except queue.Empty: msg = ''; arg = None

                    # CHOOSE NEXT STATE
                    if self.exitFlag:
                        self.nextState = States.EXITING
                    elif msg == '':
                        self.nextState = States.STOPPED
                    elif msg == SimpleAudioWriter.STOP:
                        self.nextState = States.STOPPED
                    elif msg == SimpleAudioWriter.START:
                        self.nextState = States.INITIALIZING
                    elif msg == SimpleAudioWriter.EXIT:
                        self.exitFlag = True
                        self.nextState = States.EXITING
                    else:
                        raise SyntaxError("Message \"" + msg + "\" not relevant to " + self.stateList[self.state] + " state")
# SimpleAudioWriter: ************** INITIALIZING *****************************
                elif self.state == States.INITIALIZING:
                    # DO STUFF
                    audioFile = None
                    seriesStartTime = time.time()   # Record approximate start time (in seconds since epoch) of series for filenaming purposes
                    audioFileCount = 0
                    numSamplesInCurrentSeries = 0
                    audioChunk = None
                    audioChunkLeftover = None
                    timeWrote = 0
                    writeEnabledPrevious = True
                    writeEnabled = True
                    self.audioFrequency = None

                    # Read actual audio frequency from the Synchronizer process
                    if self.audioFrequencyVar.value == -1 or self.frameRateVar.value == -1:
                        # Wait for shared value audioFrequency & frameRate to be set by the Synchronizer process
                        # Wait for shared value frameRate to be set by the Synchronizer process
                        time.sleep(0.1)
                    else:
                        self.audioFrequency = self.audioFrequencyVar.value
                        self.frameRate = self.frameRateVar.value

                        # Calculate actual exact # of frames per video that SimpleVideoWriter will be recording
                        actualFramesPerVideo = round(self.videoLength * self.frameRate)
                        # Actual video length that SimpleVideoWriter will be using
                        actualVideoLength = actualFramesPerVideo / self.frameRate
                        # Actual # of samples per video we should record. Note thta this will have an accuracy of +/- 0.5 audio samples.
                        #   At 44100 Hz, and 30 fps, the audio may be as much as whole frame de-synced after 2900 videos. This is acceptable for now.
                        numSamplesPerFile = round(actualVideoLength * self.audioFrequency)

                        if self.verbose >= 1:
                            self.log('Audio writer initialized:')
                            self.log('\tAudiofreq = {af} Hz'.format(af=self.audioFrequency))
                            self.log('\tSamples per file = {spf}'.format(spf=numSamplesPerFile))
                            self.log('\tTime per file = {t} s'.format(t=actualVideoLength))

                    # CHECK FOR MESSAGES
                    try:
                        msg, arg = self.msgQueue.get(block=False)
                        if msg == SimpleAudioWriter.SETPARAMS: self.setParams(**arg); msg = ''; arg=None
                    except queue.Empty: msg = ''; arg = None

                    # CHOOSE NEXT STATE
                    if self.exitFlag:
                        self.nextState = States.STOPPING
                    elif msg in ['', SimpleAudioWriter.START]:
                        if self.audioFrequency is None or self.frameRate is None:
                            # Haven't received audio frequency or frame rate from synchronizer - continue waiting
                            self.nextState = States.INITIALIZING
                        else:
                            # Ready to go
                            self.nextState = States.AUDIOINIT
                    elif msg == SimpleAudioWriter.STOP:
                        self.nextState = States.STOPPING
                    elif msg == SimpleAudioWriter.EXIT:
                        self.exitFlag = True
                        self.nextState = States.STOPPING
                    else:
                        raise SyntaxError("Message \"" + msg + "\" not relevant to " + self.stateList[self.state] + " state")
# SimpleAudioWriter: ************** AUDIOINIT ********************************
                elif self.state == States.AUDIOINIT:
                    # Start a new audio file
                    # DO STUFF
                    currentTimeOfDay = dt.datetime.now()
                    writeEnabledPrevious = writeEnabled
                    writeEnabled = (self.enableWrite and
                                        (not self.scheduleEnabled or
                                            (self.scheduleStartTime <= currentTimeOfDay and
                                             self.scheduleStopTime <= currentTimeOfDay)))

                    if self.verbose >= 1:
                        if writeEnabled and not writeEnabledPrevious:
                            self.logTime('Audio write now enabled.')
                        elif not writeEnabled and writeEnabledPrevious:
                            self.logTime('Audio write now disabled')

                    if writeEnabled:
                        audioFileStartTime = seriesStartTime + numSamplesInCurrentSeries / self.audioFrequency
                        numSamplesInCurrentFile = 0

                        if audioFile is not None:
                            # Close file
                            audioFile.writeframes(b'')  # Causes recompute of header info?
                            audioFile.close()
                            audioFile = None

                        # Generate new audio file path
                        audioFileNameTags = [','.join(self.channelNames), generateTimeString(timestamp=seriesStartTime), '{audioFileCount:03d}'.format(audioFileCount=audioFileCount)]
                        if self.daySubfolders:
                            audioDirectory = getDaySubfolder(self.audioDirectory, timestamp=audioFileStartTime)
                        else:
                            audioDirectory = self.audioDirectory
                        audioFileName = generateFileName(directory=audioDirectory, baseName=self.audioBaseFileName, extension='.wav', tags=audioFileNameTags)
                        ensureDirectoryExists(audioDirectory)

                        # Open and initialize audio file
                        audioFile = wave.open(audioFileName, 'w')
                        audioFile.audioFileName = audioFileName
                        # setParams: (nchannels, sampwidth, frameRate, nframes, comptype, compname)
                        audioFile.setparams((self.numChannels, self.audioDepthBytes, self.audioFrequency, 0, 'NONE', 'not compressed'))

                        newFileInfo = 'Opened audio file {name} ({n} channels, {b} bytes, {f:.2f} Hz sample rate)'.format(name=audioFileName, n=self.numChannels, b=self.audioDepthBytes, f=self.audioFrequency);
                        self.updatePublishedInfo(newFileInfo)

                        if self.verbose >= 2:
                            self.log(newFileInfo)

                    audioFileCount += 1

                    # CHECK FOR MESSAGES
                    try:
                        msg, arg = self.msgQueue.get(block=False)
                        if msg == SimpleAudioWriter.SETPARAMS: self.setParams(**arg); msg = ''; arg=None
                    except queue.Empty: msg = ''; arg = None

                    # CHOOSE NEXT STATE
                    if self.exitFlag:
                        self.nextState = States.STOPPING
                    elif msg == SimpleAudioWriter.STOP:
                        self.nextState = States.STOPPING
                    elif msg == SimpleAudioWriter.EXIT:
                        self.exitFlag = True
                        self.nextState = States.STOPPING
                    elif msg in ['', SimpleAudioWriter.START]:
                        self.nextState = States.WRITING
                    else:
                        raise SyntaxError("Message \"" + msg + "\" not relevant to " + self.stateList[self.state] + " state")
# SimpleAudioWriter: ************** WRITING *********************************
                elif self.state == States.WRITING:
                    # DO STUFF
                    if self.verbose >= 3:
                        self.log("Audio queue size: ", self.audioQueue.qsize())

                    # Write all or part of last audio chunk to file
                    if audioChunk is None:
                        # No audio chunk yet
                        pass
                    else:
                        # We have an audio chunk to write.

                        # Calculate how many more samples needed to complete the file
                        samplesUntilEOF = numSamplesPerFile - numSamplesInCurrentFile

                        # Split chunk to part before end of file, and part after end of file.
                        [audioChunk, audioChunkLeftover] = audioChunk.splitAtSample(samplesUntilEOF)
                        if self.verbose >= 3:
                            self.log("Pre chunk:", audioChunk)
                            self.log("Post chunk:", audioChunkLeftover)

                        # Write chunk of audio to file that was previously retrieved from the buffer
                        audioFile.writeframes(audioChunk.getAsBytes())
                        numSamplesInCurrentFile += audioChunk.getSampleCount()
                        numSamplesInCurrentSeries += audioChunk.getSampleCount()

                        if self.verbose >= 3:
                            self.log("Wrote audio chunk {id}".format(id=audioChunk.id))
                            timeWrote += (audioChunk.getSampleCount() / audioChunk.audioFrequency)
                            self.log("Audio time wrote: {time}".format(time=timeWrote))

                    audioChunk = self.getNextChunk()
                    if audioChunkLeftover is not None:
                        if audioChunkLeftover.getSampleCount() != 0:
                            # There are leftover samples. Include those at the start of this chunk.
                            if self.verbose >= 3:
                                self.log('Prepending leftover chunk to new chunk:')
                                self.log('Leftover: ' + audioChunkLeftover)
                                self.log('New:      ' + audioChunk)
                            audioChunk.addChunkToStart(audioChunkLeftover)
                            audioChunkLeftover = None

                    # CHECK FOR MESSAGES (and consume certain messages that don't trigger state transitions)
                    try:
                        msg, arg = self.msgQueue.get(block=False)
                        if msg == SimpleAudioWriter.SETPARAMS: self.setParams(**arg); msg = ''; arg=None
                    except queue.Empty: msg = ''; arg = None

                    # CHOOSE NEXT STATE
                    if self.exitFlag:
                        self.nextState = States.STOPPING
                    elif msg == SimpleAudioWriter.STOP:
                        self.nextState = States.STOPPING
                    elif msg == SimpleAudioWriter.EXIT:
                        self.exitFlag = True
                        self.nextState = States.STOPPING
                    elif msg in ['', SimpleAudioWriter.START]:
                        if numSamplesInCurrentFile == numSamplesPerFile:
                            # We've reached the desired sample count. Start a new audio file.
                            self.nextState = States.AUDIOINIT
                            # If requested, merge with video
                            if self.mergeMessageQueue is not None:
                                # Send file for AV merging:
                                fileEvent = dict(
                                    filePath=audioFile.audioFileName,
                                    streamType=AVMerger.AUDIO,
                                    trigger=Trigger(audioFileStartTime,
                                        audioFileStartTime,
                                        audioFileStartTime+actualVideoLength,
                                        id=audioFileCount, idspace='SimpleAVFiles'), #triggers[0],
                                    streamID='audio',
                                    startTime=audioFileStartTime,
                                    tags=['{audioFileCount:03d}'.format(audioFileCount=audioFileCount)]
                                    )
                                if self.verbose >= 1: self.log("Sending audio filename to merger")
                                self.mergeMessageQueue.put((AVMerger.MERGE, fileEvent))
                            else:
                                if self.verbose >= 3:
                                    self.log('No merge message queue available, cannot send to AVMerger')
                        elif numSamplesInCurrentFile < numSamplesPerFile:
                            # Not enough audio samples written to this file yet. Keep writing.
                            self.nextState = States.WRITING
                        else:
                            # Uh oh, too many audio samples in this file? Something went wrong.
                            raise IOError('More audio samples ({k}) than requested ({n}) in file!'.format(k=numSamplesInCurrentFile, n=numSamplesPerFile))
                    else:
                        raise SyntaxError("Message \"" + msg + "\" not relevant to " + self.stateList[self.state] + " state")
# SimpleAudioWriter: ************** STOPPING *********************************
                elif self.state == States.STOPPING:
                    # DO STUFF
                    if audioFile is not None:
                        audioFile.writeframes(b'')  # Recompute header info?
                        audioFile.close()
                        if self.mergeMessageQueue is not None:
                            # Send file for AV merging:
                            fileEvent = dict(
                                filePath=audioFile.audioFileName,
                                streamType=AVMerger.AUDIO,
                                trigger=Trigger(audioFileStartTime,
                                    audioFileStartTime,
                                    audioFileStartTime+actualVideoLength,
                                    id=audioFileCount, idspace='SimpleAVFiles'), #triggers[0],
                                streamID='audio',
                                startTime=audioFileStartTime,
                                tags=['{audioFileCount:03d}'.format(audioFileCount=audioFileCount)]
                            )
                            self.mergeMessageQueue.put((AVMerger.MERGE, fileEvent))
                        audioFile = None
                    else:
                        if self.verbose >= 3:
                            self.log('No merge message queue available, cannot send to AVMerger')

                    # CHECK FOR MESSAGES
                    try:
                        msg, arg = self.msgQueue.get(block=False)
                        if msg == SimpleAudioWriter.SETPARAMS: self.setParams(**arg); msg = ''; arg=None
                    except queue.Empty: msg = ''; arg = None

                    # CHOOSE NEXT STATE
                    if self.exitFlag:
                        self.nextState = States.STOPPED
                    elif msg == '':
                        self.nextState = States.STOPPED
                    elif msg == SimpleAudioWriter.STOP:
                        self.nextState = States.STOPPED
                    elif msg == SimpleAudioWriter.EXIT:
                        self.exitFlag = True
                        self.nextState = States.STOPPED
                    elif msg == SimpleAudioWriter.START:
                        self.nextState = States.INITIALIZING
                    else:
                        raise SyntaxError("Message \"" + msg + "\" not relevant to " + self.stateList[self.state] + " state")
# SimpleAudioWriter: ************** ERROR *********************************
                elif self.state == States.ERROR:
                    # DO STUFF
                    if self.verbose >= 0:
                        self.log("ERROR STATE. Error messages:\n\n")
                        self.log("\n\n".join(self.errorMessages))

                    self.updatePublishedInfo("\n".join(self.errorMessages))

                    self.errorMessages = []

                    # CHECK FOR MESSAGES
                    try:
                        msg, arg = self.msgQueue.get(block=False)
                        if msg == SimpleAudioWriter.SETPARAMS: self.setParams(**arg); msg = ''; arg=None
                    except queue.Empty: msg = ''; arg = None

                    # CHOOSE NEXT STATE
                    if self.lastState == States.ERROR:
                        # Error ==> Error, let's just exit
                        self.nextState = States.EXITING
                    elif msg == '':
                        if self.lastState == States.STOPPING:
                            # We got an error in stopping state? Better just stop.
                            self.nextState = States.STOPPED
                        elif self.lastState ==States.STOPPED:
                            # We got an error in the stopped state? Better just exit.
                            self.nextState = States.EXITING
                        else:
                            self.nextState = States.STOPPING
                    elif msg == SimpleAudioWriter.STOP:
                        self.nextState = States.STOPPED
                    elif msg == SimpleAudioWriter.EXIT:
                        self.exitFlag = True
                        if self.lastState == States.STOPPING:
                            self.nextState = States.EXITING
                        else:
                            self.nextState = States.STOPPING
                    else:
                        raise SyntaxError("Message \"" + msg + "\" not relevant to " + self.stateList[self.state] + " state")
# SimpleAudioWriter: ************** EXIT *********************************
                elif self.state == States.EXITING:
                    break
                else:
                    raise KeyError("Unknown state: "+self.stateList[self.state])
            except KeyboardInterrupt:
                # Handle user using keyboard interrupt
                if self.verbose >= 0: self.log("Keyboard interrupt received - exiting")
                self.exitFlag = True
                self.nextState = States.STOPPING
            except:
                # HANDLE UNKNOWN ERROR
                self.errorMessages.append("Error in "+self.stateList[self.state]+" state\n\n"+traceback.format_exc())
                self.nextState = States.ERROR

            if (self.verbose >= 1 and (len(msg) > 0 or self.exitFlag)) or len(self.stdoutBuffer) > 0 or self.verbose >= 3:
                self.log("msg={msg}".format(msg=msg))
                self.logEnd()

            self.flushStdout()

            # Prepare to advance to next state
            self.lastState = self.state
            self.state = self.nextState

        clearQueue(self.msgQueue)
        if self.verbose >= 1: self.log("Audio write process STOPPED")

        self.flushStdout()
        self.updatePublishedState(States.DEAD)

    def getNextChunk(self):
        try:
            # Get new audio chunk and return it
            newAudioChunk = self.audioQueue.get(block=True, timeout=0.1)
            if self.verbose >= 3: self.log("Got audio chunk {id} from acquirer. Pushing into the buffer.".format(id=newAudioChunk.id))
        except queue.Empty: # None available
            newAudioChunk = None
        return newAudioChunk

class AudioWriter(StateMachineProcess):
    # Human-readable states
    stateList = {
        States.WRITING :'WRITING',
        States.BUFFERING:'BUFFERING',
    }

    # Include common states from parent class
    stateList.update(StateMachineProcess.stateList)

    # Recognized message types:
    START = 'msg_start'
    STOP = 'msg_stop'
    EXIT = 'msg_exit'
    TRIGGER = 'msg_trigger'
    SETPARAMS = 'msg_setParams'

    # List of params that can be set externally with the 'msg_setParams' message
    settableParams = [
        'verbose',
        'audioBaseFileName',
        'audioDirectory',
        'daySubfolders'
        ]

    def __init__(self,
                audioBaseFileName='audioFile',
                channelNames=[],
                audioDirectory='.',
                audioQueue=None,
                audioFrequency=None,
                numChannels=1,
                bufferSizeSeconds=4,     # Buffer size in chunks - must be equal to the buffer size of associated videowriters, and equal to an integer # of audio chunks
                chunkSize=None,
                verbose=False,
                audioDepthBytes=2,
                mergeMessageQueue=None, # Queue to put (filename, trigger) in for merging
                daySubfolders=True,         # Create and write to subfolders labeled by day?
                **kwargs):
        StateMachineProcess.__init__(self, **kwargs)
        self.ID = "AW"
        self.audioDirectory = audioDirectory
        self.audioBaseFileName = audioBaseFileName
        self.channelNames = channelNames
        self.audioQueue = audioQueue
        if self.audioQueue is not None:
            self.audioQueue.cancel_join_thread()
        self.audioFrequencyVar = audioFrequency
        self.audioFrequency = None
        self.numChannels = numChannels
        self.requestedBufferSizeSeconds = bufferSizeSeconds
        self.chunkSize = chunkSize
        self.mergeMessageQueue = mergeMessageQueue
        self.bufferSize = None
        self.buffer = None
        self.errorMessages = []
        self.verbose = verbose
        self.audioDepthBytes = audioDepthBytes
        self.daySubfolders = daySubfolders

    def setParams(self, **params):
        for key in params:
            if key in AudioWriter.settableParams:
                setattr(self, key, params[key])
                if self.verbose >= 1: self.log("Param set: {key}={val}".format(key=key, val=params[key]))
            else:
                if self.verbose >= 0: self.log("Param not settable: {key}={val}".format(key=key, val=params[key]))

    def run(self):
        self.PID.value = os.getpid()
        if self.verbose >= 1: self.log("PID={pid}".format(pid=os.getpid()))
        self.state = States.STOPPED
        self.nextState = States.STOPPED
        self.lastState = -1
        msg = ''; arg = None

        while True:
            # Publish updated state
            if self.state != self.lastState:
                self.updatePublishedState()

            try:
# AudioWriter: ******************* STOPPPED *********************************
                if self.state == States.STOPPED:
                    # DO STUFF

                    # CHECK FOR MESSAGES
                    try:
                        msg, arg = self.msgQueue.get(block=True)
                        if msg == AudioWriter.SETPARAMS: self.setParams(**arg); msg = ''; arg=None
                    except queue.Empty: msg = ''; arg = None

                    # CHOOSE NEXT STATE
                    if self.exitFlag:
                        self.nextState = States.EXITING
                    elif msg == '':
                        self.nextState = States.STOPPED
                    elif msg == AudioWriter.STOP:
                        self.nextState = States.STOPPED
                    elif msg == AudioWriter.START:
                        self.nextState = States.INITIALIZING
                    elif msg == AudioWriter.EXIT:
                        self.exitFlag = True
                        self.nextState = States.EXITING
                    else:
                        raise SyntaxError("Message \"" + msg + "\" not relevant to " + self.stateList[self.state] + " state")
# AudioWriter: ******************** INITIALIZING *****************************
                elif self.state == States.INITIALIZING:
                    # DO STUFF
                    triggers = []
                    audioChunk = None
                    audioFile = None
                    audioFileStartTime = 0;
                    timeWrote = 0
                    self.audioFrequency = None

                    # Read actual audio frequency from the Synchronizer process
                    if self.audioFrequencyVar.value == -1:
                        # Wait for shared value audioFrequency to be set by the Synchronizer process
                        time.sleep(0.1)
                    else:
                        self.audioFrequency = self.audioFrequencyVar.value

                        # Calculate buffer size and create buffer
                        self.bufferSize = int(2*(self.requestedBufferSizeSeconds * self.audioFrequency / self.chunkSize))
                        self.buffer = deque() #maxlen=self.bufferSize)

                    # CHECK FOR MESSAGES
                    try:
                        msg, arg = self.msgQueue.get(block=False)
                        if msg == AudioWriter.SETPARAMS: self.setParams(**arg); msg = ''; arg=None
                    except queue.Empty: msg = ''; arg = None

                    # CHOOSE NEXT STATE
                    if self.exitFlag:
                        self.nextState = States.STOPPING
                    elif msg in ['', AudioWriter.START]:
                        if self.audioFrequency is None:
                            self.nextState = States.INITIALIZING
                        else:
                            self.nextState = States.BUFFERING
                    elif msg == AudioWriter.STOP:
                        self.nextState = States.STOPPING
                    elif msg == AudioWriter.EXIT:
                        self.exitFlag = True
                        self.nextState = States.STOPPING
                    else:
                        raise SyntaxError("Message \"" + msg + "\" not relevant to " + self.stateList[self.state] + " state")
# AudioWriter: ******************** BUFFERING ********************************
                elif self.state == States.BUFFERING:
                    # DO STUFF
                    if self.verbose >= 3:
                        self.log("Audio queue size: ", self.audioQueue.qsize())
                        self.log("Audio chunks in buffer: ", len(self.buffer))
                        self.log([c.id for c in self.buffer])

                    if len(self.buffer) >= self.bufferSize:
                        # If buffer is full, pull oldest audio chunk from buffer
                        audioChunk = self.buffer.popleft()
                        if self.verbose >= 0:
                            if len(self.buffer) >= self.bufferSize + 3:
                                # Buffer is getting overful for some reason
                                self.log("Warning, audio buffer is overfull: {curlen} > {maxlen}".format(curlen=len(self.buffer), maxlen=self.bufferSize))
                        if self.verbose >= 3: self.log("Pulled audio chunk {id} from buffer (buffer: {len}/{maxlen})".format(len=len(self.buffer), maxlen=self.bufferSize, id=audioChunk.id))

                    try:
                        if len(self.buffer) < self.bufferSize:
                            # There is room in the buffer for a new chunk
                            # Get new audio chunk and push it into the buffer
                            newAudioChunk = self.audioQueue.get(block=True, timeout=0.1)
                            if self.verbose >= 3: self.log("Got audio chunk {id} from acquirer. Pushing into the buffer.".format(id=newAudioChunk.id))
                            self.buffer.append(newAudioChunk)
                        else:
                            newAudioChunk = None
                    except queue.Empty: # None available
                        newAudioChunk = None

                    # CHECK FOR MESSAGES
                    try:
                        msg, arg = self.msgQueue.get(block=False)
                        if msg == AudioWriter.SETPARAMS: self.setParams(**arg); msg = ''; arg=None
                        elif msg == AudioWriter.TRIGGER: self.updateTriggers(triggers, arg); msg = ''; arg=None
                    except queue.Empty: msg = ''; arg = None

                    # CHOOSE NEXT STATE
                    if msg == AudioWriter.STOP:
                        self.nextState = States.STOPPING
                    elif msg == AudioWriter.EXIT:
                        self.exitFlag = True
                        self.nextState = States.STOPPING
                    elif len(triggers) > 0:
                        # We have triggers - next state will depend on them
                        if self.verbose >= 2: self.log("{N} triggers in line".format(N=len(triggers)))
                        if audioChunk is not None:
                            # At least one audio chunk has been received - we can check if trigger period has begun
                            chunkStartTriggerState, chunkEndTriggerState = audioChunk.getTriggerState(triggers[0])
                            delta = audioChunk.chunkStartTime - triggers[0].startTime
                            if self.verbose >= 3: self.log("Chunk {cid} trigger {tid} state: |{startState} ---- {endState}|".format(startState=chunkStartTriggerState, endState=chunkEndTriggerState, tid=triggers[0].id, cid=audioChunk.id))
                            if chunkStartTriggerState < 0 and chunkEndTriggerState < 0:
                                # Entire chunk is before trigger range. Continue buffering until we get to trigger start time.
                                if self.verbose >= 2: self.log("Active trigger {id}, but haven't gotten to start time yet, continue buffering.".format(id=triggers[0].id))
                                self.nextState = States.BUFFERING
                            elif chunkEndTriggerState >= 0 and ((chunkStartTriggerState < 0) or (delta < (1/self.audioFrequency))):
                                # Chunk overlaps start of trigger, or starts within one sample duration of the start of the trigger
                                if self.verbose >= 1: self.log("Got trigger {id} start!".format(id=triggers[0].id))
                                timeWrote = 0
                                self.nextState = States.WRITING
                            elif chunkStartTriggerState == 0 or (chunkStartTriggerState < 0 and chunkStartTriggerState > 0):
                                # Chunk overlaps trigger, but not the start of the trigger
                                if self.verbose >= 0:
                                    self.log("Partially missed audio trigger {id} by {t} seconds, which is {s} samples and {c} chunks!".format(t=delta, s=delta * self.audioFrequency, c=delta * self.audioFrequency / self.chunkSize, id=triggers[0].id))
                                timeWrote = 0
                                self.nextState = States.WRITING
                            else:
                                # Time is after trigger range...
                                if self.verbose >= 0: self.log("Warning, completely missed entire audio trigger {id}!".format(id=triggers[0].id))
                                timeWrote = 0
                                self.nextState = States.BUFFERING
                                triggers.pop(0)   # Pop off trigger that we missed
                        else:
                            # No audio chunks have been received yet, can't evaluate if trigger time has begun yet
                            if self.verbose >= 1: self.log("No audio chunks yet, can't begin trigger yet (buffer: {len}/{maxlen})".format(len=len(self.buffer), maxlen=self.bufferSize))
                            self.nextState = States.BUFFERING
                    elif msg in ['', AudioWriter.START]:
                        self.nextState = States.BUFFERING
                    else:
                        raise SyntaxError("Message \"" + msg + "\" not relevant to " + self.stateList[self.state] + " state")
# AudioWriter: ******************** WRITING *********************************
                elif self.state == States.WRITING:
                    # DO STUFF
                    if self.verbose >= 3:
                        self.log("Audio queue size: ", self.audioQueue.qsize())
                        self.log("Audio chunks in buffer: ", len(self.buffer))
                        self.log([c.id for c in self.buffer])
                    if audioChunk is not None:
                        #     padStart = True  # Because this is the first chunk, pad the start of the chunk if it starts after the trigger period start
                        # else:
                        #     padStart = False
                        [preChunk, postChunk] = audioChunk.trimToTrigger(triggers[0], returnOtherPieces=True) #, padStart=padStart)
                        if self.verbose >= 3:
                            self.log("Trimmed chunk:", audioChunk)
                            self.log("Pre chunk:", preChunk)
                            self.log("Post chunk:", postChunk)
                        # preChunk can be discarded, as the trigger is past it, but postChunk needs to be put back in the buffer

                        if audioFile is None:
                            # Start new audio file
                            audioFileStartTime = audioChunk.chunkStartTime
                            audioFileNameTags = [','.join(self.channelNames), generateTimeString(triggers[0])] + list(triggers[0].tags)
                            if self.daySubfolders:
                                audioDirectory = getDaySubfolder(self.audioDirectory, triggers[0])
                            else:
                                audioDirectory = self.audioDirectory
                            audioFileName = generateFileName(directory=audioDirectory, baseName=self.audioBaseFileName, extension='.wav', tags=audioFileNameTags)
                            ensureDirectoryExists(audioDirectory)
                            audioFile = wave.open(audioFileName, 'w')
                            audioFile.audioFileName = audioFileName
                            # setParams: (nchannels, sampwidth, frameRate, nframes, comptype, compname)
                            audioFile.setparams((self.numChannels, self.audioDepthBytes, self.audioFrequency, 0, 'NONE', 'not compressed'))

                        # Write chunk of audio to file that was previously retrieved from the buffer
                        audioFile.writeframes(audioChunk.getAsBytes())
                        if self.verbose >= 3:
                            self.log("Wrote audio chunk {id}".format(id=audioChunk.id))
                            timeWrote += (audioChunk.data.shape[1] / audioChunk.audioFrequency)
                            self.log("Audio time wrote: {time}".format(time=timeWrote))
                    else:
                        preChunk = None
                        postChunk = None
                    try:
                        # Pop the oldest buffered audio chunk from the back of the buffer.
                        audioChunk = self.buffer.popleft()
                        if self.verbose >= 3: self.log("Pulled audio chunk {id} from buffer".format(id=audioChunk.id))
                    except IndexError:
                        if self.verbose >= 0: self.log("No audio chunks in buffer")
                        audioChunk = None  # Buffer was empty

                    # Pull new audio chunk from AudioAcquirer and add to the front of the buffer.
                    try:
                        if len(self.buffer) < self.bufferSize:
                            newAudioChunk = self.audioQueue.get(True, 0.05)
                            if self.verbose >= 3: self.log("Got audio chunk {id} from acquirer. Pushing into buffer.".format(id=newAudioChunk.id))
                            self.buffer.append(newAudioChunk)
                        else:
                            newAudioChunk = None
                    except queue.Empty:
                        # No data in audio queue yet - pass.
                        if self.verbose >= 3: self.log("No audio chunks available from acquirer")
                        newAudioChunk = None

                    # CHECK FOR MESSAGES (and consume certain messages that don't trigger state transitions)
                    try:
                        msg, arg = self.msgQueue.get(block=False)
                        if msg == AudioWriter.SETPARAMS: self.setParams(**arg); msg = ''; arg=None
                        elif msg == AudioWriter.TRIGGER: self.updateTriggers(triggers, arg); msg = ''; arg=None
                    except queue.Empty: msg = ''; arg = None

                    # CHOOSE NEXT STATE
                    if self.exitFlag:
                        self.nextState = States.STOPPING
                    elif msg == AudioWriter.STOP:
                        self.nextState = States.STOPPING
                    elif msg == AudioWriter.EXIT:
                        self.exitFlag = True
                        self.nextState = States.STOPPING
                    elif msg in ['', AudioWriter.START]:
                        self.nextState = States.WRITING
                        if len(triggers) > 0 and audioChunk is not None:
                            chunkStartTriggerState, chunkEndTriggerState = audioChunk.getTriggerState(triggers[0])
                            if self.verbose >= 3: self.log("Chunk {cid} trigger {tid} state: |{startState} ---- {endState}|".format(startState=chunkStartTriggerState, endState=chunkEndTriggerState, tid=triggers[0].id, cid=audioChunk.id))
                            if chunkStartTriggerState * chunkEndTriggerState > 0:
                                # Trigger period does not overlap the chunk at all - return to buffering
                                if self.verbose >= 2: self.log("Audio chunk {cid} does not overlap trigger {tid}. Switching to buffering.".format(cid=audioChunk.id, tid=triggers[0].id))
                                self.nextState = States.BUFFERING
                                if audioFile is not None:
                                    # Done with trigger, close file and clear audioFile
                                    audioFile.writeframes(b'')  # Causes recompute of header info?
                                    audioFile.close()
                                    if self.mergeMessageQueue is not None:
                                        # Send file for AV merging:
                                        fileEvent = dict(
                                            filePath=audioFile.audioFileName,
                                            streamType=AVMerger.AUDIO,
                                            trigger=triggers[0],
                                            streamID='audio',
                                            startTime=audioFileStartTime
                                        )
                                        if self.verbose >= 1: self.log("Sending audio filename to merger")
                                        self.mergeMessageQueue.put((AVMerger.MERGE, fileEvent))
                                    audioFile = None
                                # Remove current trigger
                                oldTrigger = triggers.pop(0)

                                # Last chunk wasn't part of this trigger at all, so it didn't get written. Put it back in buffer.
                                self.buffer.appendleft(audioChunk)
                                if self.verbose >= 2:
                                    self.log("Trigger id {id} is over - putting back unused chunk.".format(id=oldTrigger.id))
                                    self.log(str(audioChunk))
                                # Last part of previous chunk wasn't part of this trigger, so put it back.
                                if postChunk is not None:
                                    # Put remainder of previous chunk back in the buffer, since it didn't get written
                                    if self.verbose >= 2:
                                        self.log("Trigger id {id} is over - putting back remaining {s} samples of current chunk in buffer.".format(s=postChunk.chunkSize, id=oldTrigger.id))
                                        self.log(str(postChunk))
                                    self.buffer.appendleft(postChunk)

                            else:
                                # Audio chunk does overlap with trigger period. Continue writing.
                                pass
                    else:
                        raise SyntaxError("Message \"" + msg + "\" not relevant to " + self.stateList[self.state] + " state")
# AudioWriter: ******************** STOPPING *********************************
                elif self.state == States.STOPPING:
                    # DO STUFF
                    if audioFile is not None:
                        audioFile.writeframes(b'')  # Recompute header info?
                        audioFile.close()
                        if self.mergeMessageQueue is not None:
                            # Send file for AV merging:
                            fileEvent = dict(
                                filePath=audioFile.audioFileName,
                                streamType=AVMerger.AUDIO,
                                trigger=triggers[0],
                                streamID='audio',
                                startTime=audioFileStartTime
                            )
                            self.mergeMessageQueue.put((AVMerger.MERGE, fileEvent))
                        audioFile = None
                    triggers = []

                    # CHECK FOR MESSAGES
                    try:
                        msg, arg = self.msgQueue.get(block=False)
                        if msg == AudioWriter.SETPARAMS: self.setParams(**arg); msg = ''; arg=None
                    except queue.Empty: msg = ''; arg = None

                    # CHOOSE NEXT STATE
                    if self.exitFlag:
                        self.nextState = States.STOPPED
                    elif msg == '':
                        self.nextState = States.STOPPED
                    elif msg == AudioWriter.STOP:
                        self.nextState = States.STOPPED
                    elif msg == AudioWriter.EXIT:
                        self.exitFlag = True
                        self.nextState = States.STOPPED
                    elif msg == AudioWriter.START:
                        self.nextState = States.INITIALIZING
                    else:
                        raise SyntaxError("Message \"" + msg + "\" not relevant to " + self.stateList[self.state] + " state")
# AudioWriter: ******************** ERROR *********************************
                elif self.state == States.ERROR:
                    # DO STUFF
                    if self.verbose >= 0:
                        self.log("ERROR STATE. Error messages:\n\n")
                        self.log("\n\n".join(self.errorMessages))

                    self.updatePublishedInfo("\n".join(self.errorMessages))

                    self.errorMessages = []

                    # CHECK FOR MESSAGES
                    try:
                        msg, arg = self.msgQueue.get(block=False)
                        if msg == AudioWriter.SETPARAMS: self.setParams(**arg); msg = ''; arg=None
                    except queue.Empty: msg = ''; arg = None

                    # CHOOSE NEXT STATE
                    if self.lastState == States.ERROR:
                        # Error ==> Error, let's just exit
                        self.nextState = States.EXITING
                    elif msg == '':
                        if self.lastState == States.STOPPING:
                            # We got an error in stopping state? Better just stop.
                            self.nextState = States.STOPPED
                        elif self.lastState ==States.STOPPED:
                            # We got an error in the stopped state? Better just exit.
                            self.nextState = States.EXITING
                        else:
                            self.nextState = States.STOPPING
                    elif msg == AudioWriter.STOP:
                        self.nextState = States.STOPPED
                    elif msg == AudioWriter.EXIT:
                        self.exitFlag = True
                        if self.lastState == States.STOPPING:
                            self.nextState = States.EXITING
                        else:
                            self.nextState = States.STOPPING
                    else:
                        raise SyntaxError("Message \"" + msg + "\" not relevant to " + self.stateList[self.state] + " state")
# AudioWriter: ******************** EXIT *********************************
                elif self.state == States.EXITING:
                    break
                else:
                    raise KeyError("Unknown state: "+self.stateList[self.state])
            except KeyboardInterrupt:
                # Handle user using keyboard interrupt
                if self.verbose >= 0: self.log("Keyboard interrupt received - exiting")
                self.exitFlag = True
                self.nextState = States.STOPPING
            except:
                # HANDLE UNKNOWN ERROR
                self.errorMessages.append("Error in "+self.stateList[self.state]+" state\n\n"+traceback.format_exc())
                self.nextState = States.ERROR

            if (self.verbose >= 1 and (len(msg) > 0 or self.exitFlag)) or len(self.stdoutBuffer) > 0 or self.verbose >= 3:
                self.log("msg={msg}".format(msg=msg))
                self.logEnd()

            self.flushStdout()

            # Prepare to advance to next state
            self.lastState = self.state
            self.state = self.nextState

        clearQueue(self.msgQueue)
        if self.verbose >= 1: self.log("Audio write process STOPPED")

        self.flushStdout()
        self.updatePublishedState(States.DEAD)

    def updateTriggers(self, triggers, newTrigger):
        try:
            triggerIndex = [trigger.id for trigger in triggers].index(newTrigger.id)
            # This is an updated trigger, not a new trigger
            if self.verbose >= 2:
                self.log("Updating trigger:")
                self.log(newTrigger)
            if triggerIndex > 0 and newTrigger.startTime > newTrigger.endTime:
                # End time has been set before start time, and this is not the active trigger, so delete this trigger.
                del triggers[triggerIndex]
                if self.verbose >= 2: self.log("Deleting invalidated trigger")
            else:
                triggers[triggerIndex] = newTrigger
        except ValueError:
            # This is a new trigger
            if self.verbose >= 2:
                self.log("Adding new trigger:")
                self.log(newTrigger)
            triggers.append(newTrigger)

class VideoAcquirer(StateMachineProcess):
    '''
    VideoAcquirer: A self.state machine class to pull frames from a camera and pass
        to a VideoWriter process, when a received trigger becomes active.
    '''

    # Human-readable states
    stateList = {
        States.ACQUIRING :'ACQUIRING',
        States.READY :'ACQUIRE_READY',
        }

    # Include common states from parent class
    stateList.update(StateMachineProcess.stateList)

    # Recognized message types:
    START = 'msg_start'
    STOP = 'msg_stop'
    EXIT = 'msg_exit'
    SETPARAMS = 'msg_setParams'

    # List of params that can be set externally with the 'msg_setParams' message
    settableParams = [
        'verbose',
    ]

    def __init__(self,
                startTime=None,
                camSerial='',
                acquireSettings={},
                frameRate=None,
                requestedFrameRate=None,
                # acquisitionBufferSize=100,
                bufferSizeSeconds=2.2,
                monitorFrameRate=15,
                verbose=False,
                ready=None,                        # Synchronization barrier to ensure everyone's ready before beginning
                **kwargs):
        StateMachineProcess.__init__(self, **kwargs)
        self.verbose = verbose
        self.startTimeSharedValue = startTime
        self.camSerial = camSerial
        self.ID = 'VA_'+self.camSerial
        self.acquireSettings = acquireSettings
        self.requestedFrameRate = requestedFrameRate
        self.frameRateVar = frameRate
        self.frameRate = None
        self.pixelFormat = None
        # self.imageQueue = mp.Queue()
        # self.imageQueue.cancel_join_thread()
        self.bufferSize = int(2*bufferSizeSeconds * self.requestedFrameRate)

        self.nChannels = psu.getColorChannelCount(camSerial=self.camSerial)

        if self.verbose >= 3: self.log("Temporarily initializing camera to get image size...")
        videoWidth, videoHeight = getFrameSize(self.camSerial)
        self.imageQueue = SharedImageSender(
            width=videoWidth,
            height=videoHeight,
            verbose=self.verbose,
            outputType='bytes',
            outputCopy=False,
            lockForOutput=False,
            maxBufferSize=self.bufferSize,
            channels=self.nChannels,
            name=self.camSerial+'____main',
            allowOverflow=False
        )
        if self.verbose >= 2: self.log("Creating shared image sender with max buffer size:", self.bufferSize)
        self.imageQueueReceiver = self.imageQueue.getReceiver()

        self.monitorImageSender = SharedImageSender(
            width=videoWidth,
            height=videoHeight,
            verbose=self.verbose,
            outputType='PIL',
            outputCopy=False,
            lockForOutput=False,
            maxBufferSize=1,
            channels=self.nChannels,
            name=self.camSerial+'_monitor',
            allowOverflow=True
        )
        self.monitorImageReceiver = self.monitorImageSender.getReceiver()
#        self.monitorImageQueue.cancel_join_thread()
        self.monitorMasterFrameRate = monitorFrameRate
        self.ready = ready
        self.frameStopwatch = Stopwatch()
        self.monitorStopwatch = Stopwatch()
        self.acquireStopwatch = Stopwatch()
        self.exitFlag = False
        self.errorMessages = []

    def setParams(self, **params):
        for key in params:
            if key in VideoAcquirer.settableParams:
                setattr(self, key, params[key])
                if self.verbose >= 1: self.log("Param set: {key}={val}".format(key=key, val=params[key]))
            else:
                if self.verbose >= 0: self.log("Param not settable: {key}={val}".format(key=key, val=params[key]))

    def run(self):
        self.PID.value = os.getpid()
#        if self.verbose >= 1: profiler = cProfile.Profile()
        if self.verbose >= 1: self.log("PID={pid}".format(pid=os.getpid()))

        self.imageQueue.setupBuffers()
        self.monitorImageSender.setupBuffers()

        self.state = States.STOPPED
        self.nextState = States.STOPPED
        self.lastState = -1
        msg = ''; arg = None

        while True:
            # Publish updated state
            if self.state != self.lastState:
                self.updatePublishedState()

            try:
# VideoAcquirer: ******************** STOPPPED *********************************
                if self.state == States.STOPPED:
                    # DO STUFF

                    # CHECK FOR MESSAGES
                    try:
                        msg, arg = self.msgQueue.get(block=True)
                        if msg == VideoAcquirer.SETPARAMS: self.setParams(**arg); msg = ''; arg=None
                    except queue.Empty: msg = ''; arg = None

                    # CHOOSE NEXT STATE
                    if self.exitFlag:
                        self.nextState = States.EXITING
                    elif msg == '':
                        self.nextState = self.state
                    elif msg == VideoAcquirer.STOP:
                        self.nextState = States.STOPPED
                    elif msg == VideoAcquirer.START:
                        self.nextState = States.INITIALIZING
                    elif msg == VideoAcquirer.EXIT:
                        self.exitFlag = True
                        self.nextState = States.EXITING
                    else:
                        raise SyntaxError("Message \"" + msg + "\" not relevant to " + self.stateList[self.state] + " state")
# VideoAcquirer: ****************** INITIALIZING *********************************
                elif self.state == States.INITIALIZING:
                    # DO STUFF
                    self.frameRate = None

                    # Read actual frame rate from the Synchronizer process
                    if self.frameRateVar.value == -1:
                        # Wait for shared value frameRate to be set by the Synchronizer process
                        time.sleep(0.1)
                    else:
                        self.frameRate = self.frameRateVar.value
                        if self.verbose > 2: self.log("Initializing camera...")
                        system = PySpin.System.GetInstance()
                        camList = system.GetCameras()
                        cam = camList.GetBySerial(self.camSerial)
                        cam.Init()

                        nodemap = cam.GetNodeMap()
                        self.setCameraAttributes(nodemap, self.acquireSettings)
                        if self.verbose > 2: self.log("...camera initialization complete")

                        # Get current camera pixel format
                        self.pixelFormat = psu.getCameraAttribute('PixelFormat', PySpin.CEnumerationPtr, nodemap=nodemap)[1]
                        if self.verbose >= 2: print('Camera pixel format is:', self.pixelFormat)

                        monitorFramePeriod = 1.0/self.monitorMasterFrameRate
                        if self.verbose >= 1: self.log("Monitoring with period", monitorFramePeriod)
                        thisTime = 0
                        lastTime = time.time()
                        imageCount = 0
                        im = imp = imageResult = None
                        startTime = None
                        frameTime = None
                        imageID = None
                        lastImageID = None
                        droppedFrameCount = 0

                    # CHECK FOR MESSAGES
                    try:
                        msg, arg = self.msgQueue.get(block=False)
                        if msg == VideoAcquirer.SETPARAMS: self.setParams(**arg); msg = ''; arg=None
                    except queue.Empty: msg = ''; arg = None

                    # CHOOSE NEXT STATE
                    if self.exitFlag:
                        self.nextState = States.STOPPING
                    elif msg in ['', VideoAcquirer.START]:
                        if self.frameRate is None:
                            self.nextState = States.INITIALIZING
                        else:
                            self.nextState = States.READY
                    elif msg == VideoAcquirer.STOP:
                        self.nextState = States.STOPPING
                    elif msg == VideoAcquirer.EXIT:
                        self.exitFlag = True
                        self.nextState = States.STOPPING
                    else:
                        raise SyntaxError(self.ID + " Message \"" + msg + "\" not relevant to " + self.stateList[self.state] + " state")
# VideoAcquirer: ****************** READY *********************************
                elif self.state == States.READY:
                    # DO STUFF
                    if not cam.IsStreaming():
                        cam.BeginAcquisition()

                    try:
                        if self.ready is not None:
                            self.ready.wait()
                        passedBarrier = True
                    except BrokenBarrierError:
                        passedBarrier = False
                        if self.verbose >= 2: self.log("No simultaneous start - retrying")
                        time.sleep(0.1)

                    if self.verbose >= 3: self.log('{ID} passed barrier'.format(ID=self.ID))

                    # CHECK FOR MESSAGES
                    try:
                        msg, arg = self.msgQueue.get(block=False)
                        if msg == VideoAcquirer.SETPARAMS: self.setParams(**arg); msg = ''; arg=None
                    except queue.Empty: msg = ''; arg = None

                    # CHOOSE NEXT STATE
                    if not passedBarrier:
                        self.nextState = States.READY
                    elif msg in ['', VideoAcquirer.START]:
                        self.nextState = States.ACQUIRING
                    elif msg == VideoAcquirer.STOP:
                        self.nextState = States.STOPPING
                    elif msg == VideoAcquirer.EXIT:
                        self.exitFlag = True
                        self.nextState = States.STOPPING
                    else:
                        raise SyntaxError("Message \"" + msg + "\" not relevant to " + self.stateList[self.state] + " state")
# VideoAcquirer: ****************** ACQUIRING *********************************
                elif self.state == States.ACQUIRING:
#                    if self.verbose > 1: profiler.enable()
                    # DO STUFF
                    try:
                        #  Retrieve next received image
                        if self.verbose >= 3:
                            self.acquireStopwatch.click()
                        try:
                            imageResult = cam.GetNextImage(1000)  # Timeout in ms

                            if self.verbose >= 3:
                                self.acquireStopwatch.click()
                                self.log("Get image from camera time: {t}".format(t=self.acquireStopwatch.period()))

                            # Get timestamp of first image acquisition
                            if startTime is None:
                                if self.verbose >= 1: self.log("Getting start time from sync process...")
                                while startTime == -1 or startTime is None:
                                    startTime = self.startTimeSharedValue.value
                                if self.verbose >= 1: self.log("Got start time from sync process: "+str(startTime))
    #                            startTime = time.time_ns() / 1000000000

                            if self.verbose >= 3:
                                # Time frames, as an extra check
                                self.frameStopwatch.click()
                                self.log("Video freq: ", self.frameStopwatch.frequency())
                        except PySpin.SpinnakerException as e:
                            self.log('Image grab timeout:', str(e))
                            imageResult = None

                        #  Ensure image completion
                        if imageResult is None:
                            pass
                        elif imageResult.IsIncomplete():
                            if self.verbose >= 0: self.log('Image incomplete with image status %d...' % imageResult.GetImageStatus())
                        else:
#                            imageConverted = imageResult.Convert(PySpin.PixelFormat_BGR8)
                            imageCount += 1
                            lastImageID = imageID
                            imageID = imageResult.GetFrameID()
                            if lastImageID is not None and imageID != lastImageID + 1 and self.verbose >= 0:
                                droppedFrameCount += 1
                                self.log('WARNING - DROPPED FRAMES! Image ID {a} was followed by image ID {b}. {k} dropped frames total'.format(a=lastImageID, b=imageID, k=droppedFrameCount))
                                raise IOError('DROPPED FRAMES!!!')
                            if self.verbose >= 3:
                                self.log('# frames:'+str(imageCount))
                                self.log('Frame ID:'+str(imageID))
                            frameTime = startTime + imageCount / self.frameRate

                            if self.verbose >= 3: self.log("Got image from camera, t="+str(frameTime))

                            # imp = PickleableImage(imageResult.GetWidth(), imageResult.GetHeight(), 0, 0, imageResult.GetPixelFormat(), imageResult.GetData(), frameTime)

                            # Put image into image queue
                            if self.verbose >= 3: self.log("bytes = "+str(imageResult.GetNDArray()[0:10, 0]))
                            self.imageQueue.put(imarray=imageResult.GetNDArray(), metadata={'frameTime':frameTime, 'imageID':imageID})
                            if self.verbose >= 3:
                                self.log("Pushed image into buffer")
                                self.log('Queue size={qsize}, maxsize={maxsize}'.format(qsize=self.imageQueue.qsize(), maxsize=self.imageQueue.maxBufferSize))

                            if self.monitorImageSender is not None:
                                # Put the occasional image in the monitor queue for the UI
                                thisTime = time.time()
                                actualMonitorFramePeriod = thisTime - lastTime
                                if (thisTime - lastTime) >= monitorFramePeriod:
                                    try:
                                        self.monitorImageSender.put(imageResult, metadata={'pixelFormat':self.pixelFormat})
                                        if self.verbose >= 3: self.log("Sent frame for monitoring")
                                        lastTime = thisTime
                                    except queue.Full:
                                        if self.verbose >= 3: self.log("Can't put frame in for monitoring - no room")
                                        pass

                        if imageResult is not None:
                            imageResult.Release()
                    except PySpin.SpinnakerException:
                        self.log('' + traceback.format_exc())
                        if self.verbose >= 0: self.log("Video frame acquisition timed out.")

                    # CHECK FOR MESSAGES
                    try:
                        msg, arg = self.msgQueue.get(block=False)
                        if msg == VideoAcquirer.SETPARAMS: self.setParams(**arg); msg = ''; arg=None
                    except queue.Empty: msg = ''; arg = None

                    # CHOOSE NEXT STATE
                    if self.exitFlag:
                        self.nextState = States.STOPPING
                    elif msg in ['', VideoAcquirer.START]:
                        self.nextState = States.ACQUIRING
                    elif msg == VideoAcquirer.STOP:
                        self.nextState = States.STOPPING
                    elif msg == VideoAcquirer.EXIT:
                        self.exitFlag = True
                        self.nextState = States.STOPPING
                    else:
                        raise SyntaxError("Message \"" + msg + "\" not relevant to " + self.stateList[self.state] + " state")
#                    if self.verbose > 1: profiler.disable()
# VideoAcquirer: ****************** STOPPING *********************************
                elif self.state == States.STOPPING:
                    # DO STUFF
                    if cam is not None:
                        camList.Clear()
                        cam.EndAcquisition()
                        cam.DeInit()
                        cam = None
                    if system is not None:
                        system.ReleaseInstance()

                    # Inform image monitors that we're done sending images for now
                    if self.monitorImageSender is not None:
                        self.monitorImageSender.put(None, metadata={'done':True})

                    # CHECK FOR MESSAGES
                    try:
                        msg, arg = self.msgQueue.get(block=False)
                        if msg == VideoAcquirer.SETPARAMS: self.setParams(**arg); msg = ''; arg=None
                    except queue.Empty: msg = ''; arg = None

                    # CHOOSE NEXT STATE
                    if self.exitFlag:
                        self.nextState = States.STOPPED
                    elif msg == '':
                        self.nextState = States.STOPPED
                    elif msg == AudioWriter.STOP:
                        self.nextState = States.STOPPED
                    elif msg == VideoAcquirer.EXIT:
                        self.exitFlag = True
                        self.nextState = States.STOPPED
                    elif msg == VideoAcquirer.START:
                        self.nextState = States.INITIALIZING
                    else:
                        raise SyntaxError("Message \"" + msg + "\" not relevant to " + self.stateList[self.state] + " state")
# VideoAcquirer: ****************** ERROR *********************************
                elif self.state == States.ERROR:
                    # DO STUFF
                    if self.verbose >= 0:
                        self.log("ERROR STATE. Error messages:\n\n")
                        self.log("\n\n".join(self.errorMessages))

                    self.updatePublishedInfo("\n".join(self.errorMessages))

                    self.errorMessages = []

                    # CHECK FOR MESSAGES
                    try:
                        msg, arg = self.msgQueue.get(block=False)
                        if msg == VideoAcquirer.SETPARAMS: self.setParams(**arg); msg = ''; arg=None
                    except queue.Empty: msg = ''; arg = None

                    # CHOOSE NEXT STATE
                    if self.lastState == States.ERROR:
                        # Error ==> Error, let's just exit
                        self.nextState = States.EXITING
                    elif msg == '':
                        if self.lastState == States.STOPPING:
                            # We got an error in stopping state? Better just stop.
                            self.nextState = States.STOPPED
                        elif self.lastState ==States.STOPPED:
                            # We got an error in the stopped state? Better just exit.
                            self.nextState = States.EXITING
                        else:
                            self.nextState = States.STOPPING
                    elif msg == VideoAcquirer.STOP:
                        self.nextState = States.STOPPING
                    elif msg == VideoAcquirer.EXIT:
                        self.exitFlag = True
                        if self.lastState == States.STOPPING:
                            self.nextState = States.EXITING
                        else:
                            self.nextState = States.STOPPING
                    else:
                        raise SyntaxError("Message \"" + msg + "\" not relevant to " + self.stateList[self.state] + " state")
# VideoAcquirer: ****************** EXIT *********************************
                elif self.state == States.EXITING:
                    break
                else:
                    raise KeyError("Unknown state: "+self.stateList[self.state])
            except KeyboardInterrupt:
                # Handle user using keyboard interrupt
                if self.verbose >= 0: self.log("Keyboard interrupt received - exiting")
                self.exitFlag = True
                self.nextState = States.STOPPING
            except:
                # HANDLE UNKNOWN ERROR
                self.errorMessages.append("Error in "+self.stateList[self.state]+" state\n\n"+traceback.format_exc())
                self.nextState = States.ERROR

            if (self.verbose >= 1 and (len(msg) > 0 or self.exitFlag)) or len(self.stdoutBuffer) > 0 or self.verbose >= 3:
                self.log("msg={msg}".format(msg=msg))
                self.logEnd()

            self.flushStdout()

            # Prepare to advance to next state
            self.lastState = self.state
            self.state = self.nextState

        clearQueue(self.msgQueue)
        # clearQueue(self.imageQueue)
        # clearQueue(self.monitorImageQueue)
        if self.verbose >= 1: self.log("Video acquire process STOPPED")
        # if self.verbose > 1:
        #     s = io.StringIO()
        #     ps = pstats.Stats(profiler, stream=s)
        #     ps.print_stats()
        #     self.log(s.getvalue())

        self.flushStdout()
        self.updatePublishedState(States.DEAD)

    def setCameraAttribute(self, nodemap, attributeName, attributeValue, type='enum'):
        # Set camera attribute. ReturnRetrusn True if successful, False otherwise.
        if self.verbose >= 1: self.log('Setting', attributeName, 'to', attributeValue, 'as', type)
        nodeAttribute = psu.nodeAccessorTypes[type](nodemap.GetNode(attributeName))
        if not PySpin.IsAvailable(nodeAttribute) or not PySpin.IsWritable(nodeAttribute):
            if self.verbose >= 0: self.log('Unable to set '+str(attributeName)+' to '+str(attributeValue)+' (enum retrieval). Aborting...')
            return False

        if type == 'enum':
            # Retrieve entry node from enumeration node
            nodeAttributeValue = nodeAttribute.GetEntryByName(attributeValue)
            if not PySpin.IsAvailable(nodeAttributeValue) or not PySpin.IsReadable(nodeAttributeValue):
                if self.verbose >= 0: self.log('Unable to set '+str(attributeName)+' to '+str(attributeValue)+' (entry retrieval). Aborting...')
                return False

            # Set value
            attributeValue = nodeAttributeValue.GetValue()
            nodeAttribute.SetIntValue(attributeValue)
        else:
            nodeAttribute.SetValue(attributeValue)
        return True

    def setCameraAttributes(self, nodemap, attributeValueTriplets):
    #    print('Setting attributes')
        for attribute, value, type in attributeValueTriplets:
            result = self.setCameraAttribute(nodemap, attribute, value, type=type)
            if not result:
                self.log("Failed to set", str(attribute), " to ", str(value))

class SimpleVideoWriter(StateMachineProcess):
    # Human-readable states
    stateList = {
        States.WRITING :'WRITING',
        States.VIDEOINIT : 'VIDEOINIT',
    }

    # Include common states from parent class
    stateList.update(StateMachineProcess.stateList)

    # Recognized message types:
    START = 'msg_start'
    STOP = 'msg_stop'
    EXIT = 'msg_exit'
    SETPARAMS = 'msg_setParams'

    # List of params that can be set externally with the 'msg_setParams' message
    settableParams = [
        'verbose',
        'videoBaseFileName',
        'videoDirectory',
        'daySubfolders',
        'gpuVEnc',
        'enableWrite',
        'scheduleEnabled',
        'scheduleStart',
        'scheduleStop'
        ]

    def __init__(self,
                videoDirectory='.',
                videoBaseFileName='videoFile',
                imageQueue=None,
                requestedFrameRate=None,
                frameRate=None,
                mergeMessageQueue=None,            # Queue to put (filename, trigger) in for merging
                camSerial='',
                verbose=False,
                daySubfolders=True,
                videoLength=2,   # Video length in seconds
                gpuVEnc=False,   # Should we use GPU acceleration
                enableWrite=True,
                scheduleEnabled=False,
                scheduleStartTime=None,
                scheduleStopTime=None,
                **kwargs):
        StateMachineProcess.__init__(self, **kwargs)
        self.camSerial = camSerial
        self.ID = 'SVW_' + self.camSerial
        self.videoDirectory=videoDirectory
        self.videoBaseFileName = videoBaseFileName
        self.imageQueue = imageQueue
        # if self.imageQueue is not None:
        #     self.imageQueue.cancel_join_thread()
        self.requestedFrameRate = requestedFrameRate
        self.frameRateVar = frameRate
        self.frameRate = None
        self.mergeMessageQueue = mergeMessageQueue
        self.errorMessages = []
        self.verbose = verbose
        self.videoWriteMethod = 'ffmpeg'   # options are ffmpeg, PySpin, OpenCV
        self.daySubfolders = daySubfolders
        self.videoLength = videoLength
        self.videoFrameCount = None   # Number of frames to save to each video. Wait until we get actual framerate from synchronizer
        self.gpuVEnc = gpuVEnc
        self.enableWrite = enableWrite
        self.scheduleEnabled = scheduleEnabled
        self.scheduleStartTime = scheduleStartTime
        self.scheduleStopTime = scheduleStopTime

    def setParams(self, **params):
        for key in params:
            if key in SimpleVideoWriter.settableParams:
                setattr(self, key, params[key])
                if self.verbose >= 1: self.log("Param set: {key}={val}".format(key=key, val=params[key]))
            else:
                if self.verbose >= 0: self.log("Param not settable: {key}={val}".format(key=key, val=params[key]))

    def run(self):
        self.PID.value = os.getpid()
#        if self.verbose >= 1: profiler = cProfile.Profile()
        if self.verbose >= 1: self.log("PID={pid}".format(pid=os.getpid()))

        self.state = States.STOPPED
        self.nextState = States.STOPPED
        self.lastState = -1
        msg = ''; arg = None

        while True:
            # Publish updated state
            if self.state != self.lastState:
                self.updatePublishedState()

            try:
# SimpleVideoWriter: ***************** STOPPPED *********************************
                if self.state == States.STOPPED:
                    # DO STUFF

                    # CHECK FOR MESSAGES
                    try:
                        msg, arg = self.msgQueue.get(block=True)
                        if msg == SimpleVideoWriter.SETPARAMS: self.setParams(**arg); msg = ''; arg=None
                    except queue.Empty: msg = ''; arg = None

                    # CHOOSE NEXT STATE
                    if self.exitFlag:
                        self.nextState = States.EXITING
                    elif msg == '':
                        self.nextState = States.STOPPED
                    elif msg == SimpleVideoWriter.STOP:
                        self.nextState = States.STOPPED
                    elif msg == SimpleVideoWriter.START:
                        self.nextState = States.INITIALIZING
                    elif msg == SimpleVideoWriter.EXIT:
                        self.exitFlag = True
                        self.nextState = States.EXITING
                    else:
                        raise SyntaxError("Message \"" + msg + "\" not relevant to " + self.stateList[self.state] + " state")
# SimpleVideoWriter: ************** INITIALIZING *********************************
                elif self.state == States.INITIALIZING:
                    # DO STUFF
                    videoFileInterface = None
                    seriesStartTime = time.time()   # Record approximate start time (in seconds since epoch) of series for filenaming purposes
                    videoCount = 0                  # Initialize video count, used to number video files
                    numFramesInCurrentSeries = 0    # Initialize series-wide frame count, for estimating subsequent video times
                    writeEnabledPrevious = True
                    writeEnabled = True

                    self.frameRate = self.frameRateVar.value
                    if self.frameRate == -1:
                        # Frame rate var still hasn't been set
                        # Wait for shared value frameRate to be set by the Synchronizer process
                        time.sleep(0.1)
                    else:
                        # Frame rate has been set by the synchronizer process - continue on
                        self.frameRate = self.frameRateVar.value
                        self.videoFrameCount = round(self.videoLength * self.frameRate)
                        self.log("Video framerate = {f}".format(f=self.frameRate))
                        self.log("Video length = {L}".format(L=self.videoLength))
                        self.log("Video frame count = {n}".format(n=self.videoFrameCount))

                    # CHECK FOR MESSAGES
                    try:
                        msg, arg = self.msgQueue.get(block=False)
                        if msg == SimpleVideoWriter.SETPARAMS: self.setParams(**arg); msg = ''; arg=None
                    except queue.Empty: msg = ''; arg = None

                    # CHOOSE NEXT STATE
                    if self.exitFlag:
                        self.nextState = States.STOPPING
                    elif msg in ['', SimpleVideoWriter.START]:
                        if self.frameRate == -1:
                            # Frame rate hasn't been set by synchronizer yet
                            self.nextState = States.INITIALIZING
                        else:
                            # Frame rate has been set by synchronizer
                            self.nextState = States.VIDEOINIT
                    elif msg == SimpleVideoWriter.STOP:
                        self.nextState = States.STOPPING
                    elif msg == SimpleVideoWriter.EXIT:
                        self.exitFlag = True
                        self.nextState = States.STOPPING
                    else:
                        raise SyntaxError("Message \"" + msg + "\" not relevant to " + self.stateList[self.state] + " state")
# SimpleVideoWriter: ************** VIDEOINIT *********************************
                elif self.state == States.VIDEOINIT:
                    # Start a new video file
                    # DO STUFF

                    # Check if
                    #   1. Video write is manually enabled or not
                    #   2. Video write is scheduled to be on or off
                    numFramesInCurrentVideo = 0

                    currentTimeOfDay = dt.datetime.now()
                    writeEnabledPrevious = writeEnabled
                    writeEnabled = (self.enableWrite and
                                        (not self.scheduleEnabled or
                                            (self.scheduleStartTime <= currentTimeOfDay and
                                             self.scheduleStopTime <= currentTimeOfDay)))

                    if self.verbose >= 1:
                        if writeEnabled and not writeEnabledPrevious:
                            self.logTime('Video write now enabled.')
                        elif not writeEnabled and writeEnabledPrevious:
                            self.logTime('Video write now disabled')

                    if writeEnabled:
                        im = None
                        videoFileStartTime = seriesStartTime + numFramesInCurrentSeries / self.frameRate

                        if videoFileInterface is not None:
                            if self.verbose >= 2: self.log('Closing pre-existing video file interface.')
                            # Close file
                            if self.videoWriteMethod == "PySpin":
                                videoFileInterface.Close()
                            elif self.videoWriteMethod == "ffmpeg":
                                videoFileInterface.close()
                            videoFileInterface = None

                        # Generate new video file path
                        videoFileNameTags = [self.camSerial, generateTimeString(timestamp=seriesStartTime), '{videoCount:03d}'.format(videoCount=videoCount)]
                        if self.daySubfolders:
                            videoDirectory = getDaySubfolder(self.videoDirectory, timestamp=videoFileStartTime)
                        else:
                            videoDirectory = self.videoDirectory
                        videoFileName = generateFileName(directory=videoDirectory, baseName=self.videoBaseFileName, extension='.avi', tags=videoFileNameTags)
                        if self.verbose >= 2: self.log('New filename:', videoFileName)
                        if self.verbose >= 3: self.log('Ensuring directory exists:', videoDirectory)
                        ensureDirectoryExists(videoDirectory)

                        if self.verbose >= 3: self.log('Opening new file writing interface...')

                        # Initialize video writer interface
                        if self.videoWriteMethod == "PySpin":
                            if videoFileInterface is not None:
                                videoFileInterface.Close()

                            videoFileInterface = PySpin.SpinVideo()
                            option = PySpin.AVIOption()
                            option.frameRate = self.frameRate
                            if self.verbose >= 2: self.log("Opening file to save video with frameRate ", option.frameRate)
                            videoFileInterface.Open(videoFileName, option)
                            stupidChangedVideoNameThanksABunchFLIR = videoFileName + '-0000.avi'
                            videoFileInterface.videoFileName = stupidChangedVideoNameThanksABunchFLIR
                        elif self.videoWriteMethod == "ffmpeg":
                            if self.verbose >= 3: self.log('Using ffmpeg writer')
                            if videoFileInterface is not None:
                                if self.verbose >= 3: self.log('Closing previous file interface')
                                videoFileInterface.close()
                            videoFileInterface = fw.ffmpegWriter(videoFileName, "bytes", fps=self.frameRate, gpuVEnc=self.gpuVEnc)

                        newFileInfo = 'Opened video file {name} ({f:.2f} fps, gpu encoding={gpu})'.format(name=videoFileName, f=self.frameRate, gpu=self.gpuVEnc);
                        self.updatePublishedInfo(newFileInfo)

                        if self.verbose >= 3: self.log('...opened new file writing interface')

                    videoCount += 1

                    # CHECK FOR MESSAGES (and consume certain messages that don't trigger state transitions)
                    try:
                        msg, arg = self.msgQueue.get(block=False)
                        if msg == SimpleVideoWriter.SETPARAMS: self.setParams(**arg); msg = ''; arg=None
                    except queue.Empty: msg = ''; arg = None

                    # CHOOSE NEXT STATE
                    if self.exitFlag:
                        self.nextState = States.STOPPING
                    elif msg == SimpleVideoWriter.STOP:
                        self.nextState = States.STOPPING
                    elif msg == SimpleVideoWriter.EXIT:
                        self.exitFlag = True
                        self.nextState = States.STOPPING
                    elif msg in ['', SimpleVideoWriter.START]:
                        self.nextState = States.WRITING
                    else:
                        raise SyntaxError("Message \"" + msg + "\" not relevant to " + self.stateList[self.state] + " state")
                    # if self.verbose >= 1: profiler.disable()
# SimpleVideoWriter: ************** WRITING *********************************
                elif self.state == States.WRITING:
                    # if self.verbose >= 1: profiler.enable()
                    # DO STUFF
                    if self.verbose >= 3:
                        self.logTime("Image queue size: ", self.imageQueue.qsize(), ". Getting next image...")

                    im, frameTime, imageID, frameShape = self.getNextimage()

                    if im is None:
                        # No images available. To avoid hosing the processor, sleep a bit before continuing
                        if self.verbose >= 3:
                            self.logTime("...no image yet. Waiting...")

                        time.sleep(0.5/self.requestedFrameRate)
                    else:
                        if writeEnabled:
                            if videoFileInterface is None:
                                raise IOError('Attempted to write but writer interface does not exist')

                            if self.verbose >= 3:
                                self.logTime("...got image. Sending to writer...")

                            if len(self.imageQueue.frameShape) == 3:
                                width, height, channels = frameShape
                            else:
                                width, height = frameShape
                                channels = 1

                            # Write video frame from queue to file
                            if self.videoWriteMethod == "PySpin":
                                # Reconstitute PySpin image from PickleableImage
                                # im = PySpin.Image.Create(imp.width, imp.height, imp.offsetX, imp.offsetY, imp.pixelFormat, imp.data)
                                # Convert image to desired format
                                videoFileInterface.Append(im.Convert(PySpin.PixelFormat_RGB8, PySpin.HQ_LINEAR))
                                if self.verbose >= 2: self.logTime("wrote frame using PySpin!")
                                # try:
                                #     im.Release()
                                # except PySpin.SpinnakerException:
                                #     if self.verbose >= 0:
                                #         self.log("Error releasing unconverted PySpin image after appending to AVI.")
                                #         self.log(traceback.format_exc())
                                del im
                            elif self.videoWriteMethod == "ffmpeg":
                                videoFileInterface.write(im, shape=(height, width))
                                if self.verbose >= 2: self.logTime("wrote frame using ffmpeg!")
                                if self.verbose >= 3: self.log("bytes=", str(im[0:10]))
                            if self.verbose >= 3:
                                self.logTime("...wrote image ID " + str(imageID))
                        elif self.verbose >= 3:
                            self.log('Skipped writing a frame because video write is disabled.')

                        numFramesInCurrentVideo += 1
                        numFramesInCurrentSeries += 1

                    # CHECK FOR MESSAGES (and consume certain messages that don't trigger state transitions)
                    try:
                        msg, arg = self.msgQueue.get(block=False)
                        if msg == SimpleVideoWriter.SETPARAMS: self.setParams(**arg); msg = ''; arg=None
                    except queue.Empty: msg = ''; arg = None

                    # CHOOSE NEXT STATE
                    if self.exitFlag:
                        self.nextState = States.STOPPING
                    elif msg == SimpleVideoWriter.STOP:
                        self.nextState = States.STOPPING
                    elif msg == SimpleVideoWriter.EXIT:
                        self.exitFlag = True
                        self.nextState = States.STOPPING
                    elif msg in ['', SimpleVideoWriter.START]:
                        if numFramesInCurrentVideo == self.videoFrameCount:
                            # We've reached desired video frame count. Start a new video.
                            self.nextState = States.VIDEOINIT
                            # If requested, merge with audio.
                            #   This doesn't really work with SimpleVideoWriter right now. For potential future use.
                            if self.mergeMessageQueue is not None and videoFileInterface is not None:
                                # Send file for AV merging:
                                if self.videoWriteMethod == "PySpin":
                                    fileEvent = dict(
                                        filePath=videoFileInterface.videoFileName,
                                        streamType=AVMerger.VIDEO,
                                        trigger=None, #triggers[0],
                                        streamID=self.camSerial,
                                        startTime=videoFileStartTime,
                                        tags=['{videoCount:03d}'.format(videoCount=videoCount)]
                                    )
                                else:
                                    fileEvent = dict(
                                        filePath=videoFileName,
                                        streamType=AVMerger.VIDEO,
                                        trigger=Trigger(videoFileStartTime,
                                            videoFileStartTime,
                                            videoFileStartTime+self.videoFrameCount/self.frameRate,
                                            id=videoCount, idspace='SimpleAVFiles'), #triggers[0],
                                        streamID=self.camSerial,
                                        startTime=videoFileStartTime,
                                        tags=['{videoCount:03d}'.format(videoCount=videoCount)]
                                    )
                                if self.verbose >= 2: self.log("Sending video filename to merger")
                                self.mergeMessageQueue.put((AVMerger.MERGE, fileEvent))
                            else:
                                if self.verbose >= 3:
                                    self.log('No merge message queue available, cannot send to AVMerger')
                        elif numFramesInCurrentVideo < self.videoFrameCount:
                            # Not enough frames yet. Keep writing.
                            self.nextState = States.WRITING
                        else:
                            # Uh oh, too many frames? Something went wrong.
                            raise IOError('More frames ({k}) than requested ({n}) in video!'.format(k=numFramesInCurrentVideo, n=self.videoFrameCount))
                    else:
                        raise SyntaxError("Message \"" + msg + "\" not relevant to " + self.stateList[self.state] + " state")
                    # if self.verbose >= 1: profiler.disable()
# SimpleVideoWriter: ************** STOPPING *********************************
                elif self.state == States.STOPPING:
                    # DO STUFF
                    if videoFileInterface is not None:
                        if self.videoWriteMethod == "PySpin":
                            videoFileInterface.Close()
                        elif self.videoWriteMethod == "ffmpeg":
                            videoFileInterface.close()
                        if self.mergeMessageQueue is not None:
                            # Send file for AV merging:
                            if self.videoWriteMethod == "PySpin":
                                fileEvent = dict(
                                    filePath=videoFileInterface.videoFileName,
                                    streamType=AVMerger.VIDEO,
                                    trigger=None, #triggers[0],
                                    streamID=self.camSerial,
                                    startTime=videoFileStartTime,
                                    tags=['{videoCount:03d}'.format(videoCount=videoCount)]
                                )
                            else:
                                fileEvent = dict(
                                    filePath=videoFileName,
                                    streamType=AVMerger.VIDEO,
                                    trigger=Trigger(videoFileStartTime,
                                        videoFileStartTime,
                                        videoFileStartTime+self.videoFrameCount/self.frameRate,
                                        id=videoCount, idspace='SimpleAVFiles'), #triggers[0],
                                    streamID=self.camSerial,
                                    startTime=videoFileStartTime,
                                    tags=['{videoCount:03d}'.format(videoCount=videoCount)]
                                )
                            self.mergeMessageQueue.put((AVMerger.MERGE, fileEvent))
                        videoFileInterface = None

                    # CHECK FOR MESSAGES
                    try:
                        msg, arg = self.msgQueue.get(block=False)
                        if msg == SimpleVideoWriter.SETPARAMS: self.setParams(**arg); msg = ''; arg=None
                    except queue.Empty: msg = ''; arg = None

                    # CHOOSE NEXT STATE
                    if self.exitFlag:
                        self.nextState = States.STOPPED
                    elif msg == '':
                        self.nextState = States.STOPPED
                    elif msg == SimpleVideoWriter.STOP:
                        self.nextState = States.STOPPED
                    elif msg == SimpleVideoWriter.EXIT:
                        self.exitFlag = True
                        self.nextState = States.STOPPED
                    elif msg == SimpleVideoWriter.START:
                        self.nextState = States.INITIALIZING
                    else:
                        raise SyntaxError("Message \"" + msg + "\" not relevant to " + self.stateList[self.state] + " state")
# SimpleVideoWriter: ************** ERROR *********************************
                elif self.state == States.ERROR:
                    # DO STUFF
                    if self.verbose >= 0:
                        self.log("ERROR STATE. Error messages:\n\n")
                        self.log("\n\n".join(self.errorMessages))

                    self.updatePublishedInfo("\n".join(self.errorMessages))

                    self.errorMessages = []

                    # CHECK FOR MESSAGES
                    try:
                        msg, arg = self.msgQueue.get(block=False)
                        if msg == SimpleVideoWriter.SETPARAMS: self.setParams(**arg); msg = ''; arg=None
                    except queue.Empty: msg = ''; arg = None

                    # CHOOSE NEXT STATE
                    if self.lastState == States.ERROR:
                        # Error ==> Error, let's just exit
                        self.nextState = States.EXITING
                    elif msg == '':
                        if self.lastState == States.STOPPING:
                            # We got an error in stopping state? Better just stop.
                            self.nextState = States.STOPPED
                        elif self.lastState ==States.STOPPED:
                            # We got an error in the stopped state? Better just exit.
                            self.nextState = States.EXITING
                        else:
                            self.nextState = States.STOPPING
                    elif msg == SimpleVideoWriter.STOP:
                        self.nextState = States.STOPPED
                    elif msg == SimpleVideoWriter.EXIT:
                        self.exitFlag = True
                        if self.lastState == States.STOPPING:
                            self.nextState = States.EXITING
                        else:
                            self.nextState = States.STOPPING
                    else:
                        raise SyntaxError("Message \"" + msg + "\" not relevant to " + self.stateList[self.state] + " state")
# SimpleVideoWriter: ************** EXIT *********************************
                elif self.state == States.EXITING:
                    break
                else:
                    raise KeyError("Unknown state: "+self.stateList[self.state])
            except KeyboardInterrupt:
                # Handle user using keyboard interrupt
                if self.verbose >= 0: self.log("Keyboard interrupt received - exiting")
                self.exitFlag = True
                self.nextState = States.STOPPING
            except:
                # HANDLE UNKNOWN ERROR
                self.errorMessages.append("Error in "+self.stateList[self.state]+" state\n\n"+traceback.format_exc())
                self.nextState = States.ERROR

            if (self.verbose >= 1 and (len(msg) > 0 or self.exitFlag)) or len(self.stdoutBuffer) > 0 or self.verbose >= 3:
                self.log("msg={msg}".format(msg=msg))
                self.logEnd()

            self.flushStdout()

            # Prepare to advance to next state
            self.lastState = self.state
            self.state = self.nextState

        if self.verbose >= 1: self.log("Video write process STOPPED")
        # if self.verbose > 1:
        #     s = io.StringIO()
        #     ps = pstats.Stats(profiler, stream=s)
        #     ps.print_stats()
        #     self.log(s.getvalue())
        self.flushStdout()
        self.updatePublishedState(States.DEAD)

    def getNextimage(self):
        # Pull image and metadata from acquirer queue
        try:
            # Get new video frame from acquirer and push it into the buffer
            im, metadata = self.imageQueue.get(includeMetadata=True) #block=True, timeout=0.1)
            frameTime = metadata['frameTime']
            imageID = metadata['imageID']
            frameShape = self.imageQueue.frameShape;
            if self.verbose >= 3: self.log("Got video frame from acquirer. ID={ID}, t={t}".format(t=metadata['frameTime'], ID=imageID))
        except queue.Empty:
            # No frames available from acquirer
            if self.verbose >= 3: self.log("No images available from acquirer")
            im = None
            frameTime = None
            imageID = None
            frameShape = None
        return im, frameTime, imageID, frameShape

class VideoWriter(StateMachineProcess):
    # Human-readable states
    stateList = {
        States.WRITING :'WRITING',
        States.BUFFERING :'BUFFERING',
    }

    # Include common states from parent class
    stateList.update(StateMachineProcess.stateList)

    # Recognized message types:
    START = 'msg_start'
    STOP = 'msg_stop'
    EXIT = 'msg_exit'
    TRIGGER = 'msg_trigger'
    SETPARAMS = 'msg_setParams'

    # List of params that can be set externally with the 'msg_setParams' message
    settableParams = [
        'verbose',
        'videoBaseFileName',
        'videoDirectory',
        'daySubfolders'
        ]

    def __init__(self,
                videoDirectory='.',
                videoBaseFileName='videoFile',
                imageQueue=None,
                requestedFrameRate=None,
                frameRate=None,
                mergeMessageQueue=None,            # Queue to put (filename, trigger) in for merging
                bufferSizeSeconds=2.2,
                camSerial='',
                verbose=False,
                daySubfolders=True,
                **kwargs):
        StateMachineProcess.__init__(self, **kwargs)
        self.camSerial = camSerial
        self.ID = 'VW_' + self.camSerial
        self.videoDirectory=videoDirectory
        self.videoBaseFileName = videoBaseFileName
        self.imageQueue = imageQueue
        # if self.imageQueue is not None:
        #     self.imageQueue.cancel_join_thread()
        self.requestedFrameRate = requestedFrameRate
        self.frameRateVar = frameRate
        self.frameRate = None
        self.mergeMessageQueue = mergeMessageQueue
        self.bufferSize = int(1.6*bufferSizeSeconds * self.requestedFrameRate)
        self.buffer = deque() #maxlen=self.bufferSize)
        self.errorMessages = []
        self.verbose = verbose
        self.videoWriteMethod = 'PySpin'   # options are ffmpeg, PySpin, OpenCV
        self.daySubfolders = daySubfolders

    def setParams(self, **params):
        for key in params:
            if key in VideoWriter.settableParams:
                setattr(self, key, params[key])
                if self.verbose >= 1: self.log("Param set: {key}={val}".format(key=key, val=params[key]))
            else:
                if self.verbose >= 0: self.log("Param not settable: {key}={val}".format(key=key, val=params[key]))

    def run(self):
        self.PID.value = os.getpid()
#        if self.verbose >= 1: profiler = cProfile.Profile()
        if self.verbose >= 1: self.log("PID={pid}".format(pid=os.getpid()))

        self.state = States.STOPPED
        self.nextState = States.STOPPED
        self.lastState = -1
        msg = ''; arg = None

        while True:
            # Publish updated state
            if self.state != self.lastState:
                self.updatePublishedState()

            try:
# VideoWriter: ******************** STOPPPED *********************************
                if self.state == States.STOPPED:
                    # DO STUFF

                    # CHECK FOR MESSAGES
                    try:
                        msg, arg = self.msgQueue.get(block=True)
                        if msg == VideoWriter.SETPARAMS: self.setParams(**arg); msg = ''; arg=None
                    except queue.Empty: msg = ''; arg = None

                    # CHOOSE NEXT STATE
                    if self.exitFlag:
                        self.nextState = States.EXITING
                    elif msg == '':
                        self.nextState = States.STOPPED
                    elif msg == VideoWriter.STOP:
                        self.nextState = States.STOPPED
                    elif msg == VideoWriter.START:
                        self.nextState = States.INITIALIZING
                    elif msg == VideoWriter.EXIT:
                        self.exitFlag = True
                        self.nextState = States.EXITING
                    else:
                        raise SyntaxError("Message \"" + msg + "\" not relevant to " + self.stateList[self.state] + " state")
# VideoWriter: ******************** INITIALIZING *********************************
                elif self.state == States.INITIALIZING:
                    # DO STUFF
                    triggers = []
                    im = None
                    videoFileStartTime = 0
                    videoFileInterface = None
                    timeWrote = 0
                    self.frameRate = None

                    if self.frameRateVar.value == -1:
                        # Wait for shared value frameRate to be set by the Synchronizer process
                        time.sleep(0.1)
                    else:
                        self.frameRate = self.frameRateVar.value

                    # CHECK FOR MESSAGES
                    try:
                        msg, arg = self.msgQueue.get(block=False)
                        if msg == VideoWriter.SETPARAMS: self.setParams(**arg); msg = ''; arg=None
                    except queue.Empty: msg = ''; arg = None

                    # CHOOSE NEXT STATE
                    if self.exitFlag:
                        self.nextState = States.STOPPING
                    elif msg in ['', VideoWriter.START]:
                        if self.frameRate is None:
                            self.nextState = States.INITIALIZING
                        else:
                            self.nextState = States.BUFFERING
                    elif msg == VideoWriter.STOP:
                        self.nextState = States.STOPPING
                    elif msg == VideoWriter.EXIT:
                        self.exitFlag = True
                        self.nextState = States.STOPPING
                    else:
                        raise SyntaxError("Message \"" + msg + "\" not relevant to " + self.stateList[self.state] + " state")
# VideoWriter: ******************** BUFFERING *********************************
                elif self.state == States.BUFFERING:
                    # DO STUFF
                    if self.verbose >= 3:
                        self.log("Image queue size: ", self.imageQueue.qsize())
                        self.log("Images in buffer: ", len(self.buffer))

                    im, frameTime, imageID = self.rotateImageBuffer()

                    # CHECK FOR MESSAGES
                    try:
                        msg, arg = self.msgQueue.get(block=False)
                        if msg == VideoWriter.SETPARAMS: self.setParams(**arg); msg = ''; arg=None
                        elif msg == VideoWriter.TRIGGER: self.updateTriggers(triggers, arg); msg = ''; arg=None
                    except queue.Empty: msg = ''; arg = None

                    # CHOOSE NEXT STATE
                    if msg == VideoWriter.STOP:
                        self.nextState = States.STOPPING
                    elif msg == VideoWriter.EXIT:
                        self.exitFlag = True
                        self.nextState = States.STOPPING
                    elif len(triggers) > 0:
                        # We have triggers - next state will depend on them
                        if self.verbose >= 2: self.log("" + str(len(triggers)) + " trigger(s) exist:")
                        if im is not None:
                            # At least one video frame has been received - we can check if trigger period has begun
                            triggerState = triggers[0].state(frameTime)
                            if self.verbose >= 2: self.log("Trigger state: {state}".format(state=triggerState))
                            if triggerState < 0:        # Time is before trigger range
                                if self.verbose >= 2: self.log("Active trigger, but haven't gotten to start time yet, continue buffering.")
                                self.nextState = States.BUFFERING
                            elif triggerState == 0:     # Time is now in trigger range
                                if self.verbose >= 0:
                                    delta = frameTime - triggers[0].startTime
                                    if delta <= 1/self.frameRate:
                                        # Within one frame of trigger start
                                        self.log("Got trigger start!")
                                    else:
                                        # More than one frame after trigger start - we missed some
                                        self.log("Partially missed video trigger start by {t} seconds, which is {f} frames!".format(t=delta, f=delta * self.frameRate))
                                timeWrote = 0
                                self.nextState = States.WRITING
                            else:                       # Time is after trigger range
                                if self.verbose >= 0: self.log("Missed entire trigger by {triggerState} seconds!".format(triggerState=triggerState))
                                timeWrote = 0
                                self.nextState = States.BUFFERING
                                triggers.pop(0)
                        else:
                            # No video frames have been received yet, can't evaluate if trigger time has begun yet
                            if self.verbose >= 2: self.log("No frames at the moment, can't begin trigger yet (buffer: {len}/{maxlen})".format(len=len(self.buffer), maxlen=self.bufferSize))
                            self.nextState = States.BUFFERING

                    elif msg in ['', VideoWriter.START]:
                        self.nextState = States.BUFFERING
                    else:
                        raise SyntaxError("Message \"" + msg + "\" not relevant to " + self.stateList[self.state] + " state")
# VideoWriter: ******************** WRITING *********************************
                elif self.state == States.WRITING:
                    # if self.verbose >= 1: profiler.enable()
                    # DO STUFF
                    if self.verbose >= 3:
                        self.log("Image queue size: ", self.imageQueue.qsize())
                        self.log("Images in buffer: ", len(self.buffer))
                    if im is not None:
                        if videoFileInterface is None:
                            # Start new video file
                            videoFileStartTime = frameTime
                            videoFileNameTags = [self.camSerial, generateTimeString(triggers[0])] + list(triggers[0].tags)
                            if self.daySubfolders:
                                videoDirectory = getDaySubfolder(self.videoDirectory, triggers[0])
                            else:
                                videoDirectory = self.videoDirectory
                            videoFileName = generateFileName(directory=videoDirectory, baseName=self.videoBaseFileName, extension='.avi', tags=videoFileNameTags)
                            ensureDirectoryExists(videoDirectory)
                            if self.videoWriteMethod == "PySpin":
                                videoFileInterface = PySpin.SpinVideo()
                                option = PySpin.AVIOption()
                                option.frameRate = self.frameRate
                                if self.verbose >= 2: self.log("Opening file to save video with frameRate ", option.frameRate)
                                videoFileInterface.Open(videoFileName, option)
                                stupidChangedVideoNameThanksABunchFLIR = videoFileName + '-0000.avi'
                                videoFileInterface.videoFileName = stupidChangedVideoNameThanksABunchFLIR
                            elif self.videoWriteMethod == "ffmpeg":
                                videoFileInterface = fw.ffmpegWriter(videoFileName+'.avi', fps=self.frameRate)

                        # Write video frame to file that was previously retrieved from the buffer
                        if self.verbose >= 3:
                            self.log("Wrote image ID " + str(imageID))
                            timeWrote += 1/self.frameRate
                            self.log("Video time wrote ="+str(timeWrote))

                        if self.videoWriteMethod == "PySpin":
                            # Reconstitute PySpin image from PickleableImage
                            # im = PySpin.Image.Create(imp.width, imp.height, imp.offsetX, imp.offsetY, imp.pixelFormat, imp.data)
                            # Convert image to desired format
                            videoFileInterface.Append(im.Convert(PySpin.PixelFormat_RGB8, PySpin.HQ_LINEAR))
                            if self.verbose >= 2: self.log("wrote frame!")
                            # try:
                            #     im.Release()
                            # except PySpin.SpinnakerException:
                            #     if self.verbose >= 0:
                            #         self.log("Error releasing unconverted PySpin image after appending to AVI.")
                            #         self.log(traceback.format_exc())
                            del im
                        elif self.videoWriteMethod == "ffmpeg":
                            videoFileInterface.write(imp.data, shape=(imp.width, imp.height))
                            if self.verbose >= 2: self.log("wrote frame!")

                    im, frameTime, imageID = self.rotateImageBuffer(fillBuffer=False)

                    # CHECK FOR MESSAGES (and consume certain messages that don't trigger state transitions)
                    try:
                        msg, arg = self.msgQueue.get(block=False)
                        if msg == VideoWriter.SETPARAMS: self.setParams(**arg); msg = ''; arg=None
                        elif msg == VideoWriter.TRIGGER: self.updateTriggers(triggers, arg); msg = ''; arg=None
                    except queue.Empty: msg = ''; arg = None

                    # CHOOSE NEXT STATE
                    if self.exitFlag:
                        self.nextState = States.STOPPING
                    elif msg == VideoWriter.STOP:
                        self.nextState = States.STOPPING
                    elif msg == VideoWriter.EXIT:
                        self.exitFlag = True
                        self.nextState = States.STOPPING
                    elif msg in ['', VideoWriter.START]:
                        self.nextState = States.WRITING
                        if len(triggers) > 0 and im is not None:
                            triggerState = triggers[0].state(frameTime)
                            if triggerState != 0:
                                # Frame is not in trigger period - return to buffering
                                if self.verbose >= 2: self.log("Frame does not overlap trigger. Switching to buffering.")
                                self.nextState = States.BUFFERING
                                # Remove current trigger
                                if videoFileInterface is not None:
                                    # Done with trigger, close file and clear video file
                                    if self.videoWriteMethod == "PySpin":
                                        videoFileInterface.Close()
                                    elif self.videoWriteMethod == "ffmpeg":
                                        videoFileInterface.close()
                                    if self.mergeMessageQueue is not None:
                                        # Send file for AV merging:
                                        if self.videoWriteMethod == "PySpin":
                                            fileEvent = dict(
                                                filePath=videoFileInterface.videoFileName,
                                                streamType=AVMerger.VIDEO,
                                                trigger=triggers[0],
                                                streamID=self.camSerial,
                                                startTime=videoFileStartTime
                                            )
                                        else:
                                            fileEvent = dict(
                                                filePath=videoFileName+'.avi',
                                                streamType=AVMerger.VIDEO,
                                                trigger=triggers[0],
                                                streamID=self.camSerial,
                                                startTime=videoFileStartTime
                                            )
                                        if self.verbose >= 2: self.log("Sending video filename to merger")
                                        self.mergeMessageQueue.put((AVMerger.MERGE, fileEvent))
                                    videoFileInterface = None
                                triggers.pop(0)
                                # Last image wasn't part of this trigger, so it didn't get written. Put it back in buffer.
                                self.rotateImageBufferBack(im, frameTime, imageID)
                            else:
                                # Frame is in trigger period - continue writing
                                pass
                    else:
                        raise SyntaxError("Message \"" + msg + "\" not relevant to " + self.stateList[self.state] + " state")
                    # if self.verbose >= 1: profiler.disable()
# VideoWriter: ******************** STOPPING *********************************
                elif self.state == States.STOPPING:
                    # DO STUFF
                    if videoFileInterface is not None:
                        if self.videoWriteMethod == "PySpin":
                            videoFileInterface.Close()
                        elif self.videoWriteMethod == "ffmpeg":
                            videoFileInterface.close()
                        if self.mergeMessageQueue is not None:
                            # Send file for AV merging:
                            if self.videoWriteMethod == "PySpin":
                                fileEvent = dict(
                                    filePath=videoFileInterface.videoFileName,
                                    streamType=AVMerger.VIDEO,
                                    trigger=triggers[0],
                                    streamID=self.camSerial,
                                    startTime=videoFileStartTime
                                )
                            else:
                                fileEvent = dict(
                                    filePath=videoFileName+'.avi',
                                    streamType=AVMerger.VIDEO,
                                    trigger=triggers[0],
                                    streamID=self.camSerial,
                                    startTime=videoFileStartTime
                                )
                            self.mergeMessageQueue.put((AVMerger.MERGE, fileEvent))
                        videoFileInterface = None
                    triggers = []

                    # CHECK FOR MESSAGES
                    try:
                        msg, arg = self.msgQueue.get(block=False)
                        if msg == VideoWriter.SETPARAMS: self.setParams(**arg); msg = ''; arg=None
                    except queue.Empty: msg = ''; arg = None

                    # CHOOSE NEXT STATE
                    if self.exitFlag:
                        self.nextState = States.STOPPED
                    elif msg == '':
                        self.nextState = States.BUFFERING
                    elif msg == VideoWriter.STOP:
                        self.nextState = States.STOPPED
                    elif msg == VideoWriter.EXIT:
                        self.exitFlag = True
                        self.nextState = States.STOPPED
                    elif msg == VideoWriter.START:
                        self.nextState = States.INITIALIZING
                    else:
                        raise SyntaxError("Message \"" + msg + "\" not relevant to " + self.stateList[self.state] + " state")
# VideoWriter: ******************** ERROR *********************************
                elif self.state == States.ERROR:
                    # DO STUFF
                    if self.verbose >= 0:
                        self.log("ERROR STATE. Error messages:\n\n")
                        self.log("\n\n".join(self.errorMessages))

                    self.updatePublishedInfo("\n".join(self.errorMessages))

                    self.errorMessages = []

                    # CHECK FOR MESSAGES
                    try:
                        msg, arg = self.msgQueue.get(block=False)
                        if msg == VideoWriter.SETPARAMS: self.setParams(**arg); msg = ''; arg=None
                    except queue.Empty: msg = ''; arg = None

                    # CHOOSE NEXT STATE
                    if self.lastState == States.ERROR:
                        # Error ==> Error, let's just exit
                        self.nextState = States.EXITING
                    elif msg == '':
                        if self.lastState == States.STOPPING:
                            # We got an error in stopping state? Better just stop.
                            self.nextState = States.STOPPED
                        elif self.lastState ==States.STOPPED:
                            # We got an error in the stopped state? Better just exit.
                            self.nextState = States.EXITING
                        else:
                            self.nextState = States.STOPPING
                    elif msg == VideoWriter.STOP:
                        self.nextState = States.STOPPED
                    elif msg == VideoWriter.EXIT:
                        self.exitFlag = True
                        if self.lastState == States.STOPPING:
                            self.nextState = States.EXITING
                        else:
                            self.nextState = States.STOPPING
                    else:
                        raise SyntaxError("Message \"" + msg + "\" not relevant to " + self.stateList[self.state] + " state")
# VideoWriter: ******************** EXIT *********************************
                elif self.state == States.EXITING:
                    break
                else:
                    raise KeyError("Unknown state: "+self.stateList[self.state])
            except KeyboardInterrupt:
                # Handle user using keyboard interrupt
                if self.verbose >= 0: self.log("Keyboard interrupt received - exiting")
                self.exitFlag = True
                self.nextState = States.STOPPING
            except:
                # HANDLE UNKNOWN ERROR
                self.errorMessages.append("Error in "+self.stateList[self.state]+" state\n\n"+traceback.format_exc())
                self.nextState = States.ERROR

            if (self.verbose >= 1 and (len(msg) > 0 or self.exitFlag)) or len(self.stdoutBuffer) > 0 or self.verbose >= 3:
                self.log("msg={msg}".format(msg=msg))
                self.logEnd()

            self.flushStdout()

            # Prepare to advance to next state
            self.lastState = self.state
            self.state = self.nextState

        if self.verbose >= 1: self.log("Video write process STOPPED")
        # if self.verbose > 1:
        #     s = io.StringIO()
        #     ps = pstats.Stats(profiler, stream=s)
        #     ps.print_stats()
        #     self.log(s.getvalue())
        self.flushStdout()
        self.updatePublishedState(States.DEAD)

    def rotateImageBufferBack(self, im, frameTime, imageID):
        self.buffer.appendleft((im, frameTime, imageID))

    def rotateImageBuffer(self, fillBuffer=True):
        # Pull image from acquirer queue, push to buffer
        # Pull image from buffer, return it

        if len(self.buffer) < self.bufferSize:
            # There is room in the buffer for a new image
            try:
                # Get new video frame from acquirer and push it into the buffer
                newIm, newMetadata = self.imageQueue.get(includeMetadata=True) #block=True, timeout=0.1)
                newFrameTime = newMetadata['frameTime']
                newImageID = newMetadata['imageID']

                if self.verbose >= 3: self.log("Got video frame from acquirer. Pushing into the buffer. ID={ID}, t={t}".format(t=newMetadata['frameTime'], ID=newImageID))
                self.buffer.append((newIm, newFrameTime, newImageID))
                if self.verbose >= 0:
                    if len(self.buffer) >= self.bufferSize + 3:
                        self.log("Warning, video buffer is overfull: {curlen} > {maxlen}".format(curlen=len(self.buffer), maxlen=self.bufferSize))
            except queue.Empty:
                # No frames available from acquirer
                if self.verbose >= 3: self.log("No images available from acquirer")
                time.sleep(0.5/self.requestedFrameRate)
                newIm = None
                newFrameTime = None
                newImageID = None

        if (fillBuffer and (len(self.buffer) >= self.bufferSize)) or ((not fillBuffer) and (len(self.buffer) > 0)):
            # Pop the oldest image frame from the back of the buffer.
            im, frameTime, imageID = self.buffer.popleft()
            if self.verbose >= 3: self.log("Pulling video frame (ID {ID}) from buffer (buffer: {len}/{maxlen})".format(len=len(self.buffer), maxlen=self.buffer.maxlen, ID=imageID))
        else:
            # Do not pop any off until it's either full or not empty, depending on fillBuffer).
            if self.verbose >= 3:
                self.log('Fill buffer is set to: ', fillBuffer)
                self.log('Buffer has ', len(self.buffer), ' frames of ', self.bufferSize)
                if fillBuffer:
                    self.log("buffer not full yet, declining to pull frame from buffer")
                else:
                    self.log("buffer empty, no frame to pull")
            im = None
            frameTime = None
            imageID = None

        return im, frameTime, imageID

    def updateTriggers(self, triggers, newTrigger):
        try:
            triggerIndex = [trigger.id for trigger in triggers].index(newTrigger.id)
            # This is an updated trigger, not a new trigger
            if self.verbose >= 2: self.log("Updating trigger")
            if triggerIndex > 0 and newTrigger.startTime > newTrigger.endTime:
                # End time has been set before start time, and this is not the active trigger, so delete this trigger.
                del triggers[triggerIndex]
                if self.verbose >= 2: self.log("Deleting invalidated trigger")
            else:
                triggers[triggerIndex] = newTrigger
        except ValueError:
            # This is a new trigger
            if self.verbose >= 2: self.log("Adding new trigger")
            triggers.append(newTrigger)

class ContinuousTriggerer(StateMachineProcess):
    '''
    ContinuousTriggerer: A self.state machine class to automatically generate a
        continuous train of triggers for both audio and video writer processes.
    '''

    # Human-readable states
    stateList = {
        States.TRIGGERING :'TRIGGERING',
    }

    # Include common states from parent class
    stateList.update(StateMachineProcess.stateList)

    # Recognized message types:
    START = 'msg_start'
    STOP = 'msg_stop'
    EXIT = 'msg_exit'
    SETPARAMS = 'msg_setParams'
    TAGTRIGGER = 'tag_trigger'      # Send a trigger that indicates the videos that overlap with that trigger should be tagged with trigger's tags

    # List of params that can be set externally with the 'msg_setParams' message
    settableParams = [
        'continuousTriggerPeriod',
        'scheduleEnabled',
        'scheduleStart',
        'scheduleStop'
        ]

    def __init__(self,
                startTime=None,
                recordPeriod=1,                   # Length each trigger
                scheduleEnabled=False,
                scheduleStartTime=None,
                scheduleStopTime=None,
                verbose=False,
                audioMessageQueue=None,             # Queue to send triggers to audio writers
                videoMessageQueues={},              # Queues to send triggers to video writers
                **kwargs):
        StateMachineProcess.__init__(self, **kwargs)
        self.ID = 'CT'

        self.startTimeSharedValue = startTime

        self.audioMessageQueue = audioMessageQueue
        self.videoMessageQueues = videoMessageQueues
        self.recordPeriod = recordPeriod

        self.updatePeriod = None
        self.updateUpdatePeriod()

        self.scheduleEnabled = scheduleEnabled
        self.scheduleStartTime = scheduleStartTime
        self.scheduleStopTime = scheduleStopTime

        self.errorMessages = []
        self.verbose = verbose

    def setParams(self, **params):
        for key in params:
            if key in ContinuousTriggerer.settableParams:
                setattr(self, key, params[key])
                if key == 'continuousTriggerPeriod':
                    if params[key] > 0:
                        self.recordPeriod = params[key]
                        self.updateUpdatePeriod()
                    else:
                        raise AttributeError('Record period must be greater than zero')
                if self.verbose >= 1: self.log("Param set: {key}={val}".format(key=key, val=params[key]))
            else:
                if self.verbose >= 0: self.log("Param not settable: {key}={val}".format(key=key, val=params[key]))

    def updateUpdatePeriod(self):
        self.updatePeriod = min(self.recordPeriod/20, 0.1)

    def run(self):
        self.PID.value = os.getpid()
        if self.verbose >= 1: self.log("PID={pid}".format(pid=os.getpid()))
        self.state = States.STOPPED
        self.nextState = States.STOPPED
        self.lastState = -1
        tagTriggers = []   # A list of triggers received from other processes that will be used to tag overlapping audio/video files
        msg = ''; arg = None

        while True:
            # Publish updated state
            if self.state != self.lastState:
                self.updatePublishedState()

            try:
# ContinuousTriggerer: ************* STOPPPED *********************************
                if self.state == States.STOPPED:
                    # DO STUFF

                    # CHECK FOR MESSAGES
                    try:
                        msg, arg = self.msgQueue.get(block=True)
                        if msg == ContinuousTriggerer.SETPARAMS: self.setParams(**arg); msg = ''; arg=None
                        elif msg == ContinuousTriggerer.TAGTRIGGER: msg = ''; arg=None  # Ignore tag triggers if we haven't started yet
                    except queue.Empty: msg = ''; arg = None

                    # CHOOSE NEXT STATE
                    if msg == ContinuousTriggerer.EXIT or self.exitFlag:
                        self.exitFlag = True
                        self.nextState = States.EXITING
                    elif msg == '':
                        self.nextState = States.STOPPED
                    elif msg == ContinuousTriggerer.STOP:
                        self.nextState = States.STOPPED
                    elif msg == ContinuousTriggerer.START:
                        self.nextState = States.INITIALIZING
                    else:
                        raise SyntaxError("Message \"" + msg + "\" not relevant to " + self.stateList[self.state] + " state")
# ContinuousTriggerer: ************ INITIALIZING *****************************
                elif self.state == States.INITIALIZING:
                    # DO STUFF
                    triggerBufferSize = int(max([3,1/self.recordPeriod]))        # Number of triggers to send ahead of time, in case of latency, up to 1s
                    activeTriggers = deque(maxlen=triggerBufferSize)
                    startTime = None
                    lastTriggerTime = None

                    if self.verbose >= 1: self.log("Getting start time from sync process...")
                    while startTime == -1 or startTime is None:
                        # Wait until Synchronizer process has a start time
                        startTime = self.startTimeSharedValue.value
                    if self.verbose >= 1: self.log("Got start time from sync process: "+str(startTime))

                    # CHECK FOR MESSAGES
                    try:
                        msg, arg = self.msgQueue.get(block=False)
                        if msg == ContinuousTriggerer.SETPARAMS: self.setParams(**arg); msg = ''; arg=None
                        elif msg == ContinuousTriggerer.TAGTRIGGER: msg = ''; arg=None  # Ignore tag triggers if we haven't started yet
                    except queue.Empty: msg = ''; arg = None

                    # CHOOSE NEXT STATE
                    if msg == ContinuousTriggerer.EXIT or self.exitFlag:
                        self.exitFlag = True
                        self.nextState = States.STOPPING
                    elif msg == ContinuousTriggerer.STOP:
                        self.nextState = States.STOPPING
                    elif msg in ['', ContinuousTriggerer.START]:
                        self.nextState = States.TRIGGERING
                    else:
                        raise SyntaxError("Message \"" + msg + "\" not relevant to " + self.stateList[self.state] + " state")
# ContinuousTriggerer: ************ TRIGGERING *********************************
                elif self.state == States.TRIGGERING:
                    # DO STUFF

                    currentTimeOfDay = dt.datetime.now()
                    if not self.scheduleEnabled or (self.scheduleStartTime <= currentTimeOfDay and self.scheduleStopTime <= currentTimeOfDay):
                        # If the scheduling feature is disabled, or it's enabled and we're between start/stop times, then:
                        currentTime = time.time_ns()/1000000000
                        activeTriggersChanged = False
                        # Purge triggers that are entirely in the past
                        while len(activeTriggers) > 0 and activeTriggers[0].state(currentTime) > 0:
                            oldTrigger = activeTriggers.popleft()
                            activeTriggersChanged = True
                            if self.verbose >= 2:
                                self.log('Removing old trigger:')
                                self.log('\t{t}'.format(t=oldTrigger))
                        # Create new triggers if any are needed
                        while len(activeTriggers) < activeTriggers.maxlen:
                            if lastTriggerTime is None:
                                # The first trigger will start the largest number of recordPeriods after startTime that is before or at the current time
                                newTriggerTime = startTime + self.recordPeriod * int((currentTime - startTime) / self.recordPeriod)
                            else:
                                newTriggerTime = lastTriggerTime + self.recordPeriod
                            newTrigger = Trigger(
                                startTime = newTriggerTime,
                                triggerTime = newTriggerTime,
                                endTime = newTriggerTime + self.recordPeriod,
                                idspace = self.ID)
                            lastTriggerTime = newTriggerTime
                            self.sendTrigger(newTrigger)
                            activeTriggers.append(newTrigger)
                            activeTriggersChanged = True
                            if self.verbose >= 1:
                                self.log("Sent new trigger:")
                                self.log("\t{t}".format(t=newTrigger))

                        self.updateTriggerTags(activeTriggers, tagTriggers)

                        self.purgeOldTagTriggers(tagTriggers, activeTriggers)

                        if activeTriggersChanged:
                            if self.verbose >= 2:
                                self.log('Current active triggers:')
                                for at in activeTriggers:
                                    self.log('\t{t}'.format(t=at))

                    # CHECK FOR MESSAGES
                    try:
                        msg, arg = self.msgQueue.get(block=True, timeout=self.updatePeriod)
                        if msg == ContinuousTriggerer.SETPARAMS: self.setParams(**arg); msg = ''; arg=None
                        elif msg == ContinuousTriggerer.TAGTRIGGER: self.updateTagTriggers(tagTriggers, arg); msg = ''; arg=None
                    except queue.Empty: msg = ''; arg = None

                    # CHOOSE NEXT STATE
#                    if self.verbose >= 3: self.log("|{startState} ---- {endState}|".format(startState=chunkStartTriggerState, endState=chunkEndTriggerState))
                    if msg == ContinuousTriggerer.EXIT or self.exitFlag:
                        self.exitFlag = True
                        self.nextState = States.STOPPING
                    elif msg == ContinuousTriggerer.STOP:
                        self.nextState = States.STOPPING
                    elif msg in ['', ContinuousTriggerer.START]:
                        self.nextState = self.state
                    else:
                        raise SyntaxError("Message \"" + msg + "\" not relevant to " + self.stateList[self.state] + " state")
# ContinuousTriggerer: ************ STOPPING *********************************
                elif self.state == States.STOPPING:
                    # DO STUFF
                    self.cancelTriggers(activeTriggers)

                    # CHECK FOR MESSAGES
                    try:
                        msg, arg = self.msgQueue.get(block=False)
                        if msg == ContinuousTriggerer.SETPARAMS: self.setParams(**arg); msg = ''; arg=None
                        elif msg == ContinuousTriggerer.TAGTRIGGER: msg = ''; arg=None  # Ignore tag triggers if we are stopping
                    except queue.Empty: msg = ''; arg = None

                    # CHOOSE NEXT STATE
                    if self.exitFlag:
                        self.nextState = States.STOPPED
                    elif msg == '':
                        self.nextState = States.STOPPED
                    elif msg == ContinuousTriggerer.STOP:
                        self.nextState = States.STOPPED
                    elif msg == ContinuousTriggerer.EXIT:
                        self.exitFlag = True
                        self.nextState = States.STOPPED
                    elif msg == ContinuousTriggerer.START:
                        self.nextState = States.INITIALIZING
                    else:
                        raise SyntaxError("Message \"" + msg + "\" not relevant to " + self.stateList[self.state] + " state")
# ContinuousTriggerer: ************ ERROR *********************************
                elif self.state == States.ERROR:
                    # DO STUFF
                    if self.verbose >= 0:
                        self.log("ERROR STATE. Error messages:\n\n")
                        self.log("\n\n".join(self.errorMessages))

                    self.updatePublishedInfo("\n".join(self.errorMessages))

                    self.errorMessages = []

                    # CHECK FOR MESSAGES
                    try:
                        msg, arg = self.msgQueue.get(block=False)
                        if msg == ContinuousTriggerer.SETPARAMS: self.setParams(**arg); msg = ''; arg=None
                        elif msg == ContinuousTriggerer.TAGTRIGGER: msg = ''; arg=None  # Ignore tag triggers if we are in an error state
                    except queue.Empty: msg = ''; arg = None

                    # CHOOSE NEXT STATE
                    if self.lastState == States.ERROR:
                        # Error ==> Error, let's just exit
                        self.nextState = States.EXITING
                    elif msg == '':
                        if self.lastState == States.STOPPING:
                            # We got an error in stopping state? Better just stop.
                            self.nextState = States.STOPPED
                        elif self.lastState ==States.STOPPED:
                            # We got an error in the stopped state? Better just exit.
                            self.nextState = States.EXITING
                        else:
                            self.nextState = States.STOPPING
                    elif msg == ContinuousTriggerer.STOP:
                        self.nextState = States.STOPPED
                    elif msg == ContinuousTriggerer.EXIT:
                        self.exitFlag = True
                        if self.lastState == States.STOPPING:
                            self.nextState = States.EXITING
                        else:
                            self.nextState = States.STOPPING
                    else:
                        raise SyntaxError("Message \"" + msg + "\" not relevant to " + self.stateList[self.state] + " state")
# ContinuousTriggerer: **************** EXIT *********************************
                elif self.state == States.EXITING:
                    break
                else:
                    raise KeyError("Unknown state: "+self.stateList[self.state])
            except KeyboardInterrupt:
                # Handle user using keyboard interrupt
                if self.verbose >= 0: self.log("Keyboard interrupt received - exiting")
                self.exitFlag = True
                self.nextState = States.STOPPING
            except:
                # HANDLE UNKNOWN ERROR
                self.errorMessages.append("Error in "+self.stateList[self.state]+" state\n\n"+traceback.format_exc())
                self.nextState = States.ERROR

            if (self.verbose >= 1 and (len(msg) > 0 or self.exitFlag)) or len(self.stdoutBuffer) > 0 or self.verbose >= 3:
                self.log("msg={msg}".format(msg=msg))
                self.logEnd()

            self.flushStdout()

            # Prepare to advance to next state
            self.lastState = self.state
            self.state = self.nextState

        clearQueue(self.msgQueue)
        if self.verbose >= 1: self.log("Audio write process STOPPED")

        self.flushStdout()
        self.updatePublishedState(States.DEAD)

    def cancelTriggers(self, triggers):
        for trigger in triggers:
            trigger.endtime = trigger.startTime-1
            self.sendTrigger(trigger)

    def sendTrigger(self, trigger):
        if self.audioMessageQueue:
            # If an audio message queue exists, send the trigger through it.
            self.audioMessageQueue.put((AudioWriter.TRIGGER, trigger))
        for camSerial in self.videoMessageQueues:
            # Send the trigger through any and all video message queues
            self.videoMessageQueues[camSerial].put((VideoWriter.TRIGGER, trigger))

    def purgeOldTagTriggers(self, tagTriggers, activeTriggers):
        # Get earliest trigger time
        earliestTime = min([trigger.startTime for trigger in activeTriggers])
        # Purge tagTriggers that are entirely before the earliest trigger start time
        oldTagTriggers = [tagTrigger for tagTrigger in tagTriggers if tagTrigger.state(earliestTime) > 0]
        for oldTagTrigger in oldTagTriggers:
            if self.verbose >= 2:
                self.log("Removing tag trigger earlier than {et}: {t}".format(et=earliestTime, t=oldTagTrigger))
            tagTriggers.remove(oldTagTrigger)

    def updateTagTriggers(self, tagTriggers, newTagTrigger):
        # Update the list of tag triggers with the newly arrived tag trigger
        if self.verbose >= 2: self.log("Updating tag triggers with: {t}".format(t=newTagTrigger))
        try:
            triggerIndex = [tagTrigger.id for tagTrigger in tagTriggers].index(newTagTrigger.id)
            # This is an updated trigger, not a new trigger
            if not newTagTrigger.isValid():
                # Delete this invalid trigger.
                if self.verbose >= 2: self.log("Deleting invalidated tag trigger")
                del tagTriggers[triggerIndex]
            else:
                # This is a valid updated trigger
                if self.verbose >= 2: self.log("Updating tag trigger")
                tagTriggers[triggerIndex] = newTagTrigger
        except ValueError:
            # This is a new trigger
            if self.verbose >= 2: self.log("Adding new tag trigger")
            tagTriggers.append(newTagTrigger)

    def updateTriggerTags(self, activeTriggers, tagTriggers):
        # Update tags for currently active triggers
        for activeTrigger in activeTriggers:
            tags = set()
            for tagTrigger in tagTriggers:
                if activeTrigger.overlaps(tagTrigger):
                    tags |= tagTrigger.tags
            if tags != activeTrigger.tags:
                # Tags for this trigger have changed
                if self.verbose >= 2: self.log("Applying tags to trigger: " + ','.join(tags))
                activeTrigger.tags = tags
                # Resend updated trigger because its tags have changed
                if self.verbose >= 2:
                    self.log("Resending trigger {t}".format(t=activeTrigger))
                    self.log("  with updated tags: " + ','.join(tags))
                self.sendTrigger(activeTrigger)
