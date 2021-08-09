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
import nidaqmx
from nidaqmx.stream_readers import AnalogMultiChannelReader, DigitalSingleChannelReader
from nidaqmx.constants import Edge, TriggerType
from SharedImageQueue import SharedImageSender
import traceback
import unicodedata
import re
try:
    import PySpin
except ModuleNotFoundError:
    # pip seems to install PySpin as pyspin sometimes...
    import pyspin as PySpin

nodeAccessorFunctions = {
    PySpin.intfIString:('string', PySpin.CStringPtr),
    PySpin.intfIInteger:('integer', PySpin.CIntegerPtr),
    PySpin.intfIFloat:('float', PySpin.CFloatPtr),
    PySpin.intfIBoolean:('boolean', PySpin.CBooleanPtr),
    PySpin.intfICommand:('command', PySpin.CEnumerationPtr),
    PySpin.intfIEnumeration:('enum', PySpin.CEnumerationPtr),
    PySpin.intfICategory:('category', PySpin.CCategoryPtr)
}

nodeAccessorTypes = {
    'string':PySpin.CStringPtr,
    'integer':PySpin.CIntegerPtr,
    'float':PySpin.CFloatPtr,
    'boolean':PySpin.CBooleanPtr,
    'command':PySpin.CEnumerationPtr,
    'enum':PySpin.CEnumerationPtr,
    'category':PySpin.CCategoryPtr
}

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
    def __init__(self):
        self.t0 = None
        self.t1 = None

    def click(self):
        self.t0 = self.t1
        self.t1 = time.time()

    def frequency(self):
        period = self.period()
        if period is None:
            return None
        return 1.0 / period

    def period(self):
        if self.t1 is None or self.t0 is None:
            return None
        return (self.t1 - self.t0)

class Trigger():
    newid = itertools.count().__next__   # Source of this clever little idea: https://stackoverflow.com/a/1045724/1460057
    # A class to represent a trigger object
    def __init__(self, startTime, triggerTime, endTime, tags=set(), idspace=None):
        # times in seconds
        if not (endTime >= triggerTime >= startTime):
            raise ValueError("Trigger times must satisfy startTime <= triggerTime <= endTime")
        self.id = (Trigger.newid(), idspace)
        self.startTime = startTime
        self.triggerTime = triggerTime
        self.endTime = endTime
        self.tags = tags

    def __str__(self):
        return 'Trigger id {id}: {s}-->{t}-->{e} tags: {tags}'.format(id=self.id, s=self.startTime, t=self.triggerTime, e=self.endTime, tags=self.tags)

    def tagFilename(self, filename, separator='_'):
        if len(self.tags) == 0:
            return filename
        root, ext = os.path.splitext(filename)
        path, name = os.path.split(root)
        name = name + separator + separator.join(self.tags)
        taggedPath = os.path.join(path, name+ext)
        return taggedPath

    def isValid(self):
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
        self.chunkEndTime = chunkStartTime + (self.chunkSize / self.audioFrequency)

    def __str__(self):
        return 'Audio chunk {id}: {start} ---- {samples} samp x {n} ch ----> {end} @ {freq} Hz'.format(start=self.chunkStartTime, end=self.chunkEndTime, samples=self.chunkSize, n=self.channelNumber, freq=self.audioFrequency, id=self.id)

    def getTriggerState(self, trigger):
        chunkStartTriggerState = trigger.state(self.chunkStartTime)
        chunkEndTriggerState =   trigger.state(self.chunkEndTime)
        return chunkStartTriggerState, chunkEndTriggerState

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
        self.chunkEndTime = self.chunkStartTime + (self.chunkSize / self.audioFrequency)
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

def getDaySubfolder(root, trigger):
    dateString = dt.datetime.fromtimestamp(trigger.triggerTime).strftime(DATE_FORMAT)
    return os.path.join(root, dateString)

def ensureDirectoryExists(directory):
    # Creates directory (and subdirectories if necessary) to ensure that the directory exists in the filesystem
    if len(directory) > 0:
        os.makedirs(directory, exist_ok=True)

def generateTimeString(trigger):
    return dt.datetime.fromtimestamp(trigger.triggerTime).strftime(TIME_FORMAT)

def generateFileName(directory='.', baseName='unnamed', tags=[], extension=''):
    extension = '.' + slugify(extension)
    fileName = baseName
    for tag in tags:
        if len(tag) > 0:
            fileName += '_' + tag
    fileName = slugify(fileName)
    fileName += extension
    return os.path.join(directory, fileName)

def generateButterBandpassCoeffs(lowcut, highcut, fs, order=5):
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

class StateMachineProcess(mp.Process):
    def __init__(self, *args, stdoutQueue=None, daemon=True, **kwargs):
        mp.Process.__init__(self, *args, daemon=daemon, **kwargs)
        self.ID = "X"
        self.msgQueue = mp.Queue()
        self.stdoutQueue = stdoutQueue               # Queue for pushing output message groups to for printing
        self.publishedStateVar = mp.Value('i', -1)
        self.PID = mp.Value('i', -1)
        self.exitFlag = False
        self.stdoutBuffer = []

    def run(self):
        self.PID.value = os.getpid()

    def updatePublishedState(self, state):
        if self.publishedStateVar is not None:
            L = self.publishedStateVar.get_lock()
            locked = L.acquire(block=False)
            if locked:
                self.publishedStateVar.value = state
                L.release()

    def log(self, msg, *args, **kwargs):
        syncPrint('|| {ID} - {msg}'.format(ID=self.ID, msg=msg), *args, buffer=self.stdoutBuffer, **kwargs)

    def flushStdout(self):
        if len(self.stdoutBuffer) > 0:
            self.stdoutQueue.put(self.stdoutBuffer)
        self.stdoutBuffer = []

class AVMerger(StateMachineProcess):
# Class for merging audio and video files using ffmpeg

    # States:
    STOPPED = 0
    INITIALIZING = 1
    IGNORING = 2
    WAITING = 3
    MERGING = 4
    STOPPING = 5
    ERROR = 6
    EXITING = 7
    DEAD = 100

    stateList = {
        -1:'UNKNOWN',
        STOPPED :'STOPPED',
        INITIALIZING :'INITIALIZING',
        IGNORING :'IGNORING',
        WAITING :'WAITING',
        MERGING :'MERGING',
        STOPPING :'STOPPING',
        ERROR :'ERROR',
        EXITING :'EXITING',
        DEAD :'DEAD'
    }

    #messages:
    START = 'msg_start'
    MERGE = 'msg_merge'
    STOP = 'msg_stop'
    EXIT = 'msg_exit'
    CHILL = 'msg_chill'
    SETPARAMS = 'msg_setParams'

    #Stream types
    VIDEO = 'video'
    AUDIO = 'audio'

    settableParams = [
        'verbose',
        'directory',
        'numFilesPerTrigger',
        'baseFileName',
        'montage',
        'deleteMergedAudioFiles',
        'deleteMergedVideoFiles',
        'compression',
        'daySubfolders'
    ]

    def __init__(self,
        verbose=False,
        numFilesPerTrigger=2,       # Number of files expected per trigger event (audio + video)
        directory='.',              # Directory for writing merged files
        baseFileName='',            # Base filename (sans extension) for writing merged files
        deleteMergedAudioFiles=False,    # After merging, delete unmerged audio originals
        deleteMergedVideoFiles=False,    # After merging, delete unmerged video originals
        montage=False,              # Combine videos side by side
        compression='0',            # CRF factor for libx264 compression. '0'=lossless '23'=default '51'=terrible
        daySubfolders=True,
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
        state = AVMerger.STOPPED
        nextState = AVMerger.STOPPED
        lastState = AVMerger.STOPPED
        msg = ''; arg = None

        while True:
            # Publish updated state
            if state != lastState:
                self.updatePublishedState(state)

            try:
# ********************************* STOPPPED *********************************
                if state == AVMerger.STOPPED:
                    # DO STUFF

                    # CHECK FOR MESSAGES
                    try:
                        msg, arg = self.msgQueue.get(block=True)
                        if msg == AVMerger.SETPARAMS: self.setParams(**arg); msg = ''; arg=None
                    except queue.Empty: msg = ''; arg = None

                    # CHOOSE NEXT STATE
                    if self.exitFlag:
                        nextState = AVMerger.EXITING
                    elif msg == '':
                        nextState = state
                    elif msg == AVMerger.STOP:
                        nextState = AVMerger.STOPPED
                    elif msg == AVMerger.CHILL:
                        self.ignoreFlag = True
                        nextState = AVMerger.INITIALIZING
                    elif msg == AVMerger.START:
                        self.ignoreFlag = False
                        nextState = AVMerger.INITIALIZING
                    elif msg == AVMerger.STOP:
                        nextState = AVMerger.STOPPED
                    elif msg == AVMerger.EXIT:
                        self.exitFlag = True
                        nextState = AVMerger.EXITING
                    else:
                        raise SyntaxError("Message \"" + msg + "\" not relevant to " + self.stateList[state] + " state")
# ********************************* INITIALIZING *********************************
                elif state == AVMerger.INITIALIZING:
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
                        nextState = AVMerger.STOPPING
                    elif msg == AVMerger.CHILL or self.ignoreFlag:
                        self.ignoreFlag = True
                        nextState = AVMerger.IGNORING
                    elif msg == AVMerger.START:
                        self.ignoreFlag = False
                        nextState = AVMerger.WAITING
                    elif msg in '':
                        nextState = AVMerger.WAITING
                    elif msg == AVMerger.STOP:
                        nextState = AVMerger.STOPPING
                    elif msg == AVMerger.EXIT:
                        self.exitFlag = True
                        nextState = AVMerger.STOPPING
                    else:
                        raise SyntaxError("Message \"" + msg + "\" not relevant to " + self.stateList[state] + " state")
# ********************************* IGNORING *********************************
                elif state == AVMerger.IGNORING:    # ignoring merge requests
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
                        nextState = AVMerger.IGNORING
                    elif msg == AVMerger.CHILL:
                        self.ignoreFlag = True
                        nextState = AVMerger.IGNORING
                    elif msg == AVMerger.START:
                        self.ignoreFlag = False
                        nextState = AVMerger.WAITING
                    elif msg == '':
                        nextState = AVMerger.IGNORING
                    elif msg == AVMerger.STOP:
                        nextState = AVMerger.STOPPING
                    elif msg == AVMerger.EXIT:
                        self.exitFlag = True
                        nextState = AVMergers.STOPPING
                    else:
                        raise SyntaxError("Message \"" + msg + "\" not relevant to " + self.stateList[state] + " state")
# ********************************* WAITING *********************************
                elif state == AVMerger.WAITING:    # Waiting for files to merge
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
                        nextState = AVMerger.WAITING
                    elif msg == AVMerger.CHILL or self.ignoreFlag:
                        self.ignoreFlag = True
                        nextState = AVMerger.IGNORING
                    elif len(groupedFileEventList) > 0:
                        # At least one group of unmerged matching files - go to merge
                        nextState = AVMerger.MERGING
                    elif msg in ['', AVMerger.START]:
                        nextState = AVMerger.WAITING
                    elif msg == AVMerger.STOP:
                        nextState = AVMerger.STOPPING
                    elif msg == AVMerger.EXIT:
                        self.exitFlag = True
                        nextState = AVMergers.STOPPING
                    else:
                        raise SyntaxError("Message \"" + msg + "\" not relevant to " + self.stateList[state] + " state")
# ********************************* MERGING *********************************
                elif state == AVMerger.MERGING:
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
                            mergeCommandTemplate = 'ffmpeg -i "{videoFile}" ' + audioFileInputText + ' -c:v libx264 -preset veryfast -crf {compression} -shortest -nostdin -y "{outputFile}"'
                            # Set up dictionary of strings to substitute into command template
                            kwargs = dict([('audioFile{k}'.format(k=k), audioFileEvents[k]['filePath']) for k in range(len(audioFileEvents))])
                            for videoFileEvent in videoFileEvents:
                                # Add/update dictionary to reflect this video file
                                kwargs['videoFile'] = videoFileEvent['filePath']
                                fileNameTags = [videoFileEvent['streamID'], 'merged', generateTimeString(videoFileEvent['trigger'])] + list(videoFileEvent['trigger'].tags)
                                kwargs['outputFile'] = generateFileName(directory=mergeDirectory, baseName=self.baseFileName, extension='.avi', tags=fileNameTags)
                                kwargs['compression'] = self.compression
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
                            # Construct command template
                            mergeCommandTemplate = "ffmpeg " + videoFileInputText + " " + audioFileInputText + ' -c:v libx264 -preset veryfast -crf {compression} -shortest -nostdin -y -filter_complex hstack "{outputFile}"'
                            # Set up dictionary of strings to substitute into command template
                            kwargs = dict(
                                [('audioFile{k}'.format(k=k), audioFileEvents[k]['filePath']) for k in range(len(audioFileEvents))] + \
                                [('videoFile{k}'.format(k=k), videoFileEvents[k]['filePath']) for k in range(len(videoFileEvents))])
                            fileNameTags = [videoFileEvent['streamID'], 'montage', generateTimeString(videoFileEvent['trigger'])] + list(videoFileEvent['trigger'].tags)
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
                        nextState = AVMerger.STOPPING
                    elif msg == AVMerger.MERGE:
                        if arg is not None:
                            receivedFileEventList.append(arg)
                        nextState = AVMerger.WAITING
                    elif msg in ['', AVMerger.START]:
                        nextState = AVMerger.WAITING
                    elif msg == AVMerger.CHILL:
                        self.ignoreFlag = True
                        nextState = AVMerger.IGNORING
                    elif msg == AVMerger.STOP:
                        nextState = AVMerger.STOPPING
                    elif msg == AVMerger.EXIT:
                        self.exitFlag = True
                        nextState = AVMerger.STOPPING
                    else:
                        raise SyntaxError("Message \"" + msg + "\" not relevant to " + self.stateList[state] + " state")
# ********************************* STOPPING *********************************
                elif state == AVMerger.STOPPING:
                    # DO STUFF

                    # CHECK FOR MESSAGES
                    try:
                        msg, arg = self.msgQueue.get(block=False)
                        if msg == AVMerger.SETPARAMS: self.setParams(**arg); msg = ''; arg=None
                    except queue.Empty: msg = ''; arg = None

                    # CHOOSE NEXT STATE
                    if self.exitFlag:
                        nextState = AVMerger.STOPPED
                    elif msg == '':
                        nextState = AVMerger.STOPPED
                    elif msg == AVMerger.START:
                        nextState = AVMerger.INITIALIZING
                    elif msg == AVMerger.STOP:
                        nextState = AVMerger.STOPPED
                    elif msg == AVMerger.EXIT:
                        self.exitFlag = True
                        nextState = AVMerger.STOPPED
                    else:
                        raise SyntaxError("Message \"" + msg + "\" not relevant to " + self.stateList[state] + " state")
# ********************************* ERROR *********************************
                elif state == AVMerger.ERROR:
                    # DO STUFF
                    if self.verbose >= 0:
                        self.log("ERROR STATE. Error messages:\n\n")
                        self.log("\n\n".join(self.errorMessages))
                    self.errorMessages = []

                    # CHECK FOR MESSAGES
                    try:
                        msg, arg = self.msgQueue.get(block=False)
                        if msg == AVMerger.SETPARAMS: self.setParams(**arg); msg = ''; arg=None
                    except queue.Empty: msg = ''; arg = None

                    # CHOOSE NEXT STATE
                    if lastState == AVMerger.ERROR:
                        # Error ==> Error, let's just exit
                        nextState = AVMerger.EXITING
                    elif msg == '':
                        if lastState == AVMerger.STOPPING:
                            # We got an error in stopping state? Better just stop.
                            nextState = AVMerger.STOPPED
                        elif lastState ==AVMerger.STOPPED:
                            # We got an error in the stopped state? Better just exit.
                            nextState = AVMerger.EXITING
                        else:
                            nextState = AVMerger.STOPPING
                    elif msg == AVMerger.STOP:
                        nextState = AVMerger.STOPPED
                    elif msg == AVMerger.EXIT:
                        self.exitFlag = True
                        if lastState == AVMerger.STOPPING:
                            nextState = AVMerger.EXITING
                        else:
                            nextState = AVMerger.STOPPING
                    else:
                        raise SyntaxError("Message \"" + msg + "\" not relevant to " + self.stateList[state] + " state")
# ********************************* EXIT *********************************
                elif state == AVMerger.EXITING:
                    break
                else:
                    raise KeyError("Unknown state: "+self.stateList[state])
            except KeyboardInterrupt:
                # Handle user using keyboard interrupt
                if self.verbose >= 0: self.log("Keyboard interrupt received - exiting")
                self.exitFlag = True
                nextState = AVMerger.STOPPING
            except:
                # HANDLE UNKNOWN ERROR
                self.errorMessages.append("Error in "+self.stateList[state]+" state\n\n"+traceback.format_exc())
                nextState = AVMerger.ERROR

            if (self.verbose >= 1 and (len(msg) > 0 or self.exitFlag)) or len(self.stdoutBuffer) > 0 or self.verbose >= 3:
                self.log("msg={msg}, exitFlag={exitFlag}".format(msg=msg, exitFlag=self.exitFlag))
                self.log(r'*********************************** /\ {ID} {state} /\ ********************************************'.format(ID=self.ID, state=self.stateList[state]))

            # Prepare to advance to next state
            lastState = state
            state = nextState
            self.flushStdout()

        clearQueue(self.msgQueue)
        if self.verbose >= 1: self.log("AVMerger process STOPPED")

        self.flushStdout()
        self.updatePublishedState(self.DEAD)

class Synchronizer(StateMachineProcess):
    # Class for generating two synchronization signals at the same time
    #   - one for video (send via cable to the camera GPIO)
    #   - one for audio (used internally to trigger analog input of microphone
    #     signals)
    # This class inherits from multiprocessing.Process so it can be run in a
    #   separate process, allowing a single script to generate the sync pulses
    #   and also accomplish other tasks.

    # States:
    STOPPED = 0
    INITIALIZING = 1
    SYNCHRONIZING = 2
    STOPPING = 3
    SYNC_READY = 4
    ERROR = 5
    EXITING = 6
    DEAD = 100

    stateList = {
        -1:'UNKNOWN',
         STOPPED:'STOPPED',
         INITIALIZING:'INITIALIZING',
         SYNCHRONIZING:'SYNCHRONIZING',
         STOPPING:'STOPPING',
         SYNC_READY:'SYNC_READY',
         ERROR:'ERROR',
         EXITING:'EXITING',
         DEAD:'DEAD'
    }

    #messages:
    START = 'msg_start'
    STOP = 'msg_stop'
    EXIT = 'msg_exit'
    SETPARAMS = 'msg_setParams'

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
        state = Synchronizer.STOPPED
        nextState = Synchronizer.STOPPED
        lastState = Synchronizer.STOPPED
        msg = ''; arg = None

        while True:
            # Publish updated state
            if state != lastState:
                self.updatePublishedState(state)

            try:
# ********************************* STOPPPED *********************************
                if state == Synchronizer.STOPPED:
                    # DO STUFF

                    # CHECK FOR MESSAGES
                    try:
                        msg, arg = self.msgQueue.get(block=True)
                        if msg == Synchronizer.SETPARAMS: self.setParams(**arg); msg = ''; arg=None
                    except queue.Empty: msg = ''; arg = None

                    # CHOOSE NEXT STATE
                    if self.exitFlag:
                        nextState = Synchronizer.EXITING
                    elif msg == '':
                        nextState = state
                    elif msg == Synchronizer.STOP:
                        nextState = Synchronizer.STOPPED
                    elif msg == Synchronizer.START:
                        nextState = Synchronizer.INITIALIZING
                    elif msg == Synchronizer.EXIT:
                        self.exitFlag = True
                        nextState = Synchronizer.EXITING
                    else:
                        raise SyntaxError("Message \"" + msg + "\" not relevant to " + self.stateList[state] + " state")
# ********************************* INITIALIZING *********************************
                elif state == Synchronizer.INITIALIZING:
                    # DO STUFF

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
                    if self.audioSyncChannel is not None:
                        trigTask.co_channels.add_co_pulse_chan_freq(
                            counter=self.audioSyncChannel,
                            name_to_assign_to_channel="audioFrequency",
                            units=nidaqmx.constants.FrequencyUnits.HZ,
                            initial_delay=0.0,
                            freq=self.audioFrequency,
                            duty_cycle=self.audioDutyCycle)     # Prepare a counter output channel for the audio sync signal
                    # if (self.startTriggerChannel is not None) and ((self.videoSyncChannel is not None) or (self.audioSyncChannel is not None)):
                    #     # Configure task to wait for a digital pulse on the specified channel.
                    #     trigTask.triggers.arm_start_trigger.dig_edge_src=self.startTriggerChannel
                    #     trigTask.triggers.arm_start_trigger.trig_type=TriggerType.DIGITAL_EDGE
                    #     trigTask.triggers.arm_start_trigger.dig_edge_edge=Edge.RISING
                    trigTask.timing.cfg_implicit_timing(sample_mode=nidaqmx.constants.AcquisitionType.CONTINUOUS)

                    # Set shared values so other processes can get actual a/v frequencies
                    if self.actualAudioFrequency is not None:
                        self.actualAudioFrequency.value = trigTask.co_channels['audioFrequency'].co_pulse_freq
                        if self.verbose > 0: self.log('Requested audio frequency: ', self.audioFrequency, ' | actual audio frequency: ', self.actualAudioFrequency.value);
                    if self.actualVideoFrequency is not None:
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
                        nextState = Synchronizer.STOPPING
                    elif msg in ['', Synchronizer.START]:
                        nextState = Synchronizer.SYNC_READY
                    elif msg == Synchronizer.STOP:
                        nextState = Synchronizer.STOPPING
                    elif msg == Synchronizer.EXIT:
                        self.exitFlag = True
                        nextState = Synchronizer.STOPPING
                    else:
                        raise SyntaxError("Message \"" + msg + "\" not relevant to " + self.stateList[state] + " state")
# ********************************* SYNC_READY *********************************
                elif state == Synchronizer.SYNC_READY:
                    # DO STUFF
                    try:
                        if self.ready is not None:
                            self.ready.wait()
                        # To give audio and video processes a chance to get totally set up for acquiring, wait a second.
                        time.sleep(1)

                        if startTask is not None:
                            startTask.start()
                            while not startReader.read_one_sample_one_line():
                                pass
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
                        if self.verbose >= 0: self.log("Simultaneous start failure")
                        nextState = Synchronizer.STOPPING

                    # CHECK FOR MESSAGES
                    try:
                        msg, arg = self.msgQueue.get(block=False)
                        if msg == Synchronizer.SETPARAMS: self.setParams(**arg); msg = ''; arg=None
                    except queue.Empty: msg = ''; arg = None

                    # CHOOSE NEXT STATE
                    if msg in ['', Synchronizer.START]:
                        nextState = Synchronizer.SYNCHRONIZING
                    elif msg == Synchronizer.STOP:
                        nextState = Synchronizer.STOPPING
                    elif msg == Synchronizer.EXIT:
                        self.exitFlag = True
                        nextState = Synchronizer.STOPPING
                    else:
                        raise SyntaxError("Message \"" + msg + "\" not relevant to " + self.stateList[state] + " state")

# ********************************* SYNCHRONIZING *********************************
                elif state == Synchronizer.SYNCHRONIZING:
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
                        nextState = Synchronizer.STOPPING
                    elif msg in ['', Synchronizer.START]:
                        nextState = state
                    elif msg == Synchronizer.STOP:
                        nextState = Synchronizer.STOPPING
                    elif msg == Synchronizer.EXIT:
                        self.exitFlag = True
                        nextState = Synchronizer.STOPPING
                    else:
                        raise SyntaxError("Message \"" + msg + "\" not relevant to " + self.stateList[state] + " state")
# ********************************* STOPPING *********************************
                elif state == Synchronizer.STOPPING:
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
                        nextState = Synchronizer.STOPPED
                    elif msg == '':
                        nextState = Synchronizer.STOPPED
                    elif msg == Synchronizer.START:
                        nextState = Synchronizer.INITIALIZING
                    elif msg == Synchronizer.STOP:
                        nextState = Synchronizer.STOPPED
                    elif msg == Synchronizer.EXIT:
                        self.exitFlag = True
                        nextState = Synchronizer.STOPPED
                    else:
                        raise SyntaxError("Message \"" + msg + "\" not relevant to " + self.stateList[state] + " state")
# ********************************* ERROR *********************************
                elif state == Synchronizer.ERROR:
                    # DO STUFF
                    if self.verbose >= 0:
                        self.log("ERROR STATE. Error messages:\n\n")
                        self.log("\n\n".join(self.errorMessages))
                    self.errorMessages = []

                    # CHECK FOR MESSAGES
                    try:
                        msg, arg = self.msgQueue.get(block=False)
                        if msg == Synchronizer.SETPARAMS: self.setParams(**arg); msg = ''; arg=None
                    except queue.Empty: msg = ''; arg = None

                    # CHOOSE NEXT STATE
                    if lastState == Synchronizer.ERROR:
                        # Error ==> Error, let's just exit
                        nextState = Synchronizer.EXITING
                    elif msg == '':
                        if lastState == Synchronizer.STOPPING:
                            # We got an error in stopping state? Better just stop.
                            nextState = Synchronizer.STOPPED
                        elif lastState ==Synchronizer.STOPPED:
                            # We got an error in the stopped state? Better just exit.
                            nextState = Synchronizer.EXITING
                        else:
                            nextState = Synchronizer.STOPPING
                    elif msg == Synchronizer.STOP:
                        nextState = Synchronizer.STOPPED
                    elif msg == Synchronizer.EXIT:
                        self.exitFlag = True
                        if lastState == Synchronizer.STOPPING:
                            nextState = Synchronizer.EXITING
                        else:
                            nextState = Synchronizer.STOPPING
                    else:
                        raise SyntaxError("Message \"" + msg + "\" not relevant to " + self.stateList[state] + " state")
# ********************************* EXIT *********************************
                elif state == Synchronizer.EXITING:
                    break
                else:
                    raise KeyError("Unknown state: "+self.stateList[state])
            except KeyboardInterrupt:
                # Handle user using keyboard interrupt
                if self.verbose >= 0: self.log("Keyboard interrupt received - exiting")
                self.exitFlag = True
                nextState = Synchronizer.STOPPING
            except:
                # HANDLE UNKNOWN ERROR
                self.errorMessages.append("Error in "+self.stateList[state]+" state\n\n"+traceback.format_exc())
                nextState = Synchronizer.ERROR

            if (self.verbose >= 1 and (len(msg) > 0 or self.exitFlag)) or len(self.stdoutBuffer) > 0 or self.verbose >= 3:
                self.log("msg={msg}, exitFlag={exitFlag}".format(msg=msg, exitFlag=self.exitFlag))
                self.log(r'*********************************** /\ {ID} {state} /\ ********************************************'.format(ID=self.ID, state=self.stateList[state]))

            self.flushStdout()

            # Prepare to advance to next state
            lastState = state
            state = nextState

        clearQueue(self.msgQueue)
        if self.verbose >= 1: self.log("Synchronization process STOPPED")
        self.flushStdout()
        self.updatePublishedState(self.DEAD)

class AudioTriggerer(StateMachineProcess):
    # States:
    STOPPED = 0
    INITIALIZING = 1
    WAITING = 2
    ANALYZING = 3
    STOPPING = 4
    ERROR = 5
    EXITING = 6
    DEAD = 100

    stateList = {
        -1:'UNKNOWN',
        STOPPED :'STOPPED',
        INITIALIZING :'INITIALIZING',
        WAITING :'WAITING',
        ANALYZING :'ANALYZING',
        STOPPING :'STOPPING',
        ERROR :'ERROR',
        EXITING :'EXITING',
        DEAD :'DEAD'
    }

    #messages:
    START = 'msg_start'
    STARTANALYZE = "msg_startanalyze"
    STOPANALYZE = "msg_stopanalyze"
    STOP = 'msg_stop'
    EXIT = 'msg_exit'
    SETPARAMS = 'msg_setParams'

    multiChannelBehaviors = ['OR', 'AND']

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
        'scheduleStartTime',
        'scheduleStopTime',
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
        state = AudioTriggerer.STOPPED
        nextState = AudioTriggerer.STOPPED
        lastState = AudioTriggerer.STOPPED
        msg = ''; arg = None

        while True:
            # Publish updated state
            if state != lastState:
                self.updatePublishedState(state)

            try:
# ********************************* STOPPPED *********************************
                if state == AudioTriggerer.STOPPED:
                    # DO STUFF

                    # CHECK FOR MESSAGES
                    try:
                        msg, arg = self.msgQueue.get(block=True)
                        if msg == AudioTriggerer.SETPARAMS: self.setParams(**arg); msg = ''; arg=None
                    except queue.Empty: msg = ''; arg = None

                    # CHOOSE NEXT STATE
                    if msg == AudioTriggerer.EXIT or self.exitFlag:
                        self.exitFlag = True
                        nextState = AudioTriggerer.EXITING
                    elif msg == '':
                        nextState = AudioTriggerer.STOPPED
                    elif msg == AudioTriggerer.STOP:
                        nextState = AudioTriggerer.STOPPED
                    elif msg == AudioTriggerer.START:
                        nextState = AudioTriggerer.INITIALIZING
                    elif msg == AudioTriggerer.STARTANALYZE:
                        self.analyzeFlag = True
                        nextState = AudioTriggerer.INITIALIZING
                    else:
                        raise SyntaxError("Message \"" + msg + "\" not relevant to " + self.stateList[state] + " state")
# ********************************* INITIALIZING *****************************
                elif state == AudioTriggerer.INITIALIZING:
                    # DO STUFF
                    activeTrigger = None
                    while self.audioFrequencyVar.value == -1:
                        # Wait for shared value audioFrequency to be set by the Synchronizer process
                        time.sleep(0.1)
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
                        nextState = AudioTriggerer.STOPPING
                    elif msg == AudioTriggerer.STOP:
                        nextState = AudioTriggerer.STOPPING
                    elif msg in ['', AudioTriggerer.START, AudioTriggerer.STOPANALYZE]:
                        nextState = AudioTriggerer.WAITING
                    elif msg == AudioTriggerer.STARTANALYZE or self.analyzeFlag:
                        nextState = AudioTriggerer.ANALYZING
                    else:
                        raise SyntaxError("Message \"" + msg + "\" not relevant to " + self.stateList[state] + " state")
# ********************************* WAITING ********************************
                elif state == AudioTriggerer.WAITING:
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
                        nextState = AudioTriggerer.STOPPING
                    elif msg == AudioTriggerer.EXIT:
                        self.exitFlag = True
                        nextState = AudioTriggerer.STOPPING
                    elif msg == AudioTriggerer.STARTANALYZE or self.analyzeFlag:
                        nextState = AudioTriggerer.ANALYZING
                    elif msg in ['', AudioTriggerer.STOPANALYZE, AudioTriggerer.START]:
                        nextState = AudioTriggerer.WAITING
                    else:
                        raise SyntaxError("Message \"" + msg + "\" not relevant to " + self.stateList[state] + " state")
# ********************************* ANALYZING *********************************
                elif state == AudioTriggerer.ANALYZING:
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
                        nextState = AudioTriggerer.STOPPING
                    elif msg == AudioTriggerer.STOP:
                        nextState = AudioTriggerer.STOPPING
                    elif msg == AudioTriggerer.STOPANALYZE:
                        nextState = AudioTriggerer.WAITING
                    elif msg in ['', AudioTriggerer.STARTANALYZE, AudioTriggerer.START]:
                        nextState = state
                    else:
                        raise SyntaxError("Message \"" + msg + "\" not relevant to " + self.stateList[state] + " state")
# ********************************* STOPPING *********************************
                elif state == AudioTriggerer.STOPPING:
                    # DO STUFF

                    # CHECK FOR MESSAGES
                    try:
                        msg, arg = self.msgQueue.get(block=False)
                        if msg == AudioTriggerer.SETPARAMS: self.setParams(**arg); msg = ''; arg=None
                    except queue.Empty: msg = ''; arg = None

                    # CHOOSE NEXT STATE
                    if self.exitFlag:
                        nextState = AudioTriggerer.STOPPED
                    elif msg == '':
                        nextState = AudioTriggerer.STOPPED
                    elif msg == AudioTriggerer.STOP:
                        nextState = AudioTriggerer.STOPPED
                    elif msg == AudioTriggerer.EXIT:
                        self.exitFlag = True
                        nextState = AudioTriggerer.STOPPED
                    elif msg == AudioTriggerer.START:
                        nextState = AudioTriggerer.INITIALIZING
                    else:
                        raise SyntaxError("Message \"" + msg + "\" not relevant to " + self.stateList[state] + " state")
# ********************************* ERROR *********************************
                elif state == AudioTriggerer.ERROR:
                    # DO STUFF
                    if self.verbose >= 0:
                        self.log("ERROR STATE. Error messages:\n\n")
                        self.log("\n\n".join(self.errorMessages))
                    self.errorMessages = []

                    # CHECK FOR MESSAGES
                    try:
                        msg, arg = self.msgQueue.get(block=False)
                        if msg == AudioTriggerer.SETPARAMS: self.setParams(**arg); msg = ''; arg=None
                    except queue.Empty: msg = ''; arg = None

                    # CHOOSE NEXT STATE
                    if lastState == AudioTriggerer.ERROR:
                        # Error ==> Error, let's just exit
                        nextState = AudioTriggerer.EXITING
                    elif msg == '':
                        if lastState == AudioTriggerer.STOPPING:
                            # We got an error in stopping state? Better just stop.
                            nextState = AudioTriggerer.STOPPED
                        elif lastState ==AudioTriggerer.STOPPED:
                            # We got an error in the stopped state? Better just exit.
                            nextState = AudioTriggerer.EXITING
                        else:
                            nextState = AudioTriggerer.STOPPING
                    elif msg == AudioTriggerer.STOP:
                        nextState = AudioTriggerer.STOPPED
                    elif msg == AudioTriggerer.EXIT:
                        self.exitFlag = True
                        if lastState == AudioTriggerer.STOPPING:
                            nextState = AudioTriggerer.EXITING
                        else:
                            nextState = AudioTriggerer.STOPPING
                    else:
                        raise SyntaxError("Message \"" + msg + "\" not relevant to " + self.stateList[state] + " state")
# ********************************* EXIT *********************************
                elif state == AudioTriggerer.EXITING:
                    break
                else:
                    raise KeyError("Unknown state: "+self.stateList[state])
            except KeyboardInterrupt:
                # Handle user using keyboard interrupt
                if self.verbose >= 0: self.log("Keyboard interrupt received - exiting")
                self.exitFlag = True
                nextState = AudioTriggerer.STOPPING
            except:
                # HANDLE UNKNOWN ERROR
                self.errorMessages.append("Error in "+self.stateList[state]+" state\n\n"+traceback.format_exc())
                nextState = AudioTriggerer.ERROR

            if (self.verbose >= 1 and (len(msg) > 0 or self.exitFlag)) or len(self.stdoutBuffer) > 0 or self.verbose >= 3:
                self.log("msg={msg}, exitFlag={exitFlag}".format(msg=msg, exitFlag=self.exitFlag))
                self.log(r'*********************************** /\ {ID} {state} /\ ********************************************'.format(ID=self.ID, state=self.stateList[state]))

            self.flushStdout()

            # Prepare to advance to next state
            lastState = state
            state = nextState

        clearQueue(self.msgQueue)
        clearQueue(self.analysisMonitorQueue)
        if self.verbose >= 1: self.log("Audio write process STOPPED")

        self.flushStdout()
        self.updatePublishedState(self.DEAD)

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

    # States:
    STOPPED = 0
    INITIALIZING = 1
    ACQUIRING = 2
    STOPPING = 3
    ACQUIRE_READY = 4
    ERROR = 5
    EXITING = 6
    DEAD = 100

    stateList = {
        -1:'UNKNOWN',
        STOPPED :'STOPPED',
        INITIALIZING :'INITIALIZING',
        ACQUIRING :'ACQUIRING',
        STOPPING :'STOPPING',
        ACQUIRE_READY :'ACQUIRE_READY',
        ERROR :'ERROR',
        EXITING :'EXITING',
        DEAD :'DEAD'
    }

    #messages:
    START = 'msg_start'
    STOP = 'msg_stop'
    EXIT = 'msg_exit'
    SETPARAMS = 'msg_setParams'

    settableParams = [
        'verbose'
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
                **kwargs):
        StateMachineProcess.__init__(self, **kwargs)
        # Store inputs in instance variables for later access
        self.ID = "AA"
        self.startTimeSharedValue = startTime
        self.audioFrequencyVar = audioFrequency
        self.audioFrequency = None
        self.acquireTimeout = 10 #2*chunkSize / self.audioFrequency
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
        state = AudioAcquirer.STOPPED
        nextState = AudioAcquirer.STOPPED
        lastState = AudioAcquirer.STOPPED
        msg = ''; arg = None

        while True:
            # Publish updated state
            if state != lastState:
                self.updatePublishedState(state)

            try:
# ********************************* STOPPPED *********************************
                if state == AudioAcquirer.STOPPED:
                    # DO STUFF

                    # CHECK FOR MESSAGES
                    try:
                        msg, arg = self.msgQueue.get(block=True)
                        if msg == AudioAcquirer.SETPARAMS: self.setParams(**arg); msg = ''; arg=None
                    except queue.Empty: msg = ''; arg = None

                    # CHOOSE NEXT STATE
                    if self.exitFlag:
                        nextState = AudioAcquirer.EXITING
                    elif msg == '':
                        nextState = state
                    elif msg == AudioAcquirer.STOP:
                        nextState = AudioAcquirer.STOPPED
                    elif msg == AudioAcquirer.START:
                        nextState = AudioAcquirer.INITIALIZING
                    elif msg == AudioAcquirer.EXIT:
                        self.exitFlag = True
                        nextState = AudioAcquirer.EXITING
                    else:
                        raise SyntaxError("Message \"" + msg + "\" not relevant to " + self.stateList[state] + " state")
# ********************************* INITIALIZING *********************************
                elif state == AudioAcquirer.INITIALIZING:
                    # DO STUFF
                    # Read actual audio frequency from the Synchronizer process
                    while self.audioFrequencyVar.value == -1:
                        # Wait for shared value audioFrequency to be set by the Synchronizer process
                        time.sleep(0.1)
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
                        nextState = AudioAcquirer.STOPPING
                    elif msg in ['', AudioAcquirer.START]:
                        nextState = AudioAcquirer.ACQUIRE_READY
                    elif msg == AudioAcquirer.STOP:
                        nextState = AudioAcquirer.STOPPING
                    elif msg == AudioAcquirer.EXIT:
                        self.exitFlag = True
                        nextState = AudioAcquirer.STOPPING
                    else:
                        raise SyntaxError("AA - Message \"" + msg + "\" not relevant to " + self.stateList[state] + " state")
# ********************************* ACQUIRE_READY *********************************
                elif state == AudioAcquirer.ACQUIRE_READY:
                    # DO STUFF
                    try:
                        if self.ready is not None:
#                            if self.verbose >= 1: self.log('ready: {parties} {n_waiting}'.format(parties=self.ready.parties, n_waiting=self.ready.n_waiting))
                            self.ready.wait()
                    except BrokenBarrierError:
                        if self.verbose >= 0: self.log("Simultaneous start failure")
                        nextState = AudioAcquirer.STOPPING

#                    if self.verbose >= 1: self.log('passed barrier')

                    # CHECK FOR MESSAGES
                    try:
                        msg, arg = self.msgQueue.get(block=False)
                        if msg == AudioAcquirer.SETPARAMS: self.setParams(**arg); msg = ''; arg=None
                    except queue.Empty: msg = ''; arg = None

                    # CHOOSE NEXT STATE
                    if msg in ['', AudioAcquirer.START]:
                        nextState = AudioAcquirer.ACQUIRING
                    elif msg == AudioAcquirer.STOP:
                        nextState = AudioAcquirer.STOPPING
                    elif msg == AudioAcquirer.EXIT:
                        self.exitFlag = True
                        nextState = AudioAcquirer.STOPPING
                    else:
                        raise SyntaxError("Message \"" + msg + "\" not relevant to " + self.stateList[state] + " state")
# ********************************* ACQUIRING *********************************
                elif state == AudioAcquirer.ACQUIRING:
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

                        if self.monitorQueue is not None:
                            self.monitorQueue.put((self.inputChannels, chunkStartTime, monitorDataCopy))      # If a monitoring queue is provided, queue up the data
                        if self.analysisQueue is not None:
                            self.analysisQueue.put((chunkStartTime, monitorDataCopy))
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
                        nextState = AudioAcquirer.STOPPING
                    elif msg in ['', AudioAcquirer.START]:
                        nextState = AudioAcquirer.ACQUIRING
                    elif msg == AudioAcquirer.STOP:
                        nextState = AudioAcquirer.STOPPING
                    elif msg == AudioAcquirer.EXIT:
                        self.exitFlag = True
                        nextState = AudioAcquirer.STOPPING
                    else:
                        raise SyntaxError("Message \"" + msg + "\" not relevant to " + self.stateList[state] + " state")
# ********************************* STOPPING *********************************
                elif state == AudioAcquirer.STOPPING:
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
                        nextState = AudioAcquirer.STOPPED
                    elif msg == '':
                        nextState = AudioAcquirer.STOPPED
                    elif msg == AudioAcquirer.STOP:
                        nextState = AudioAcquirer.STOPPED
                    elif msg == AudioAcquirer.EXIT:
                        self.exitFlag = True
                        nextState = AudioAcquirer.STOPPED
                    elif msg == AudioAcquirer.START:
                        nextState = AudioAcquirer.INITIALIZING
                    else:
                        raise SyntaxError("Message \"" + msg + "\" not relevant to " + self.stateList[state] + " state")
# ********************************* ERROR *********************************
                elif state == AudioAcquirer.ERROR:
                    # DO STUFF
                    if self.verbose >= 0:
                        self.log("ERROR STATE. Error messages:\n\n")
                        self.log("\n\n".join(self.errorMessages))
                    self.errorMessages = []

                    # CHECK FOR MESSAGES
                    try:
                        msg, arg = self.msgQueue.get(block=False)
                        if msg == AudioAcquirer.SETPARAMS: self.setParams(**arg); msg = ''; arg=None
                    except queue.Empty: msg = ''; arg = None

                    # CHOOSE NEXT STATE
                    if lastState == AudioAcquirer.ERROR:
                        # Error ==> Error, let's just exit
                        nextState = AudioAcquirer.EXITING
                    elif msg == '':
                        if lastState == AudioAcquirer.STOPPING:
                            # We got an error in stopping state? Better just stop.
                            nextState = AudioAcquirer.STOPPED
                        elif lastState ==AudioAcquirer.STOPPED:
                            # We got an error in the stopped state? Better just exit.
                            nextState = AudioAcquirer.EXITING
                        else:
                            nextState = AudioAcquirer.STOPPING
                    elif msg == AudioAcquirer.STOP:
                        nextState = AudioAcquirer.STOPPING
                    elif msg == AudioAcquirer.EXIT:
                        self.exitFlag = True
                        if lastState == AudioAcquirer.STOPPING:
                            nextState = AudioAcquirer.EXITING
                        else:
                            nextState = AudioAcquirer.STOPPING
                    else:
                        raise SyntaxError("Message \"" + msg + "\" not relevant to " + self.stateList[state] + " state")
# ********************************* EXIT *********************************
                elif state == AudioAcquirer.EXITING:
                    if self.verbose >= 1: self.log('Exiting!')
                    break
                else:
                    raise KeyError("Unknown state: "+self.stateList[state])
            except KeyboardInterrupt:
                # Handle user using keyboard interrupt
                if self.verbose >= 1: self.log("Keyboard interrupt received - exiting")
                self.exitFlag = True
                nextState = AudioAcquirer.STOPPING
            except:
                # HANDLE UNKNOWN ERROR
                self.errorMessages.append("Error in "+self.stateList[state]+" state\n\n"+traceback.format_exc())
                nextState = AudioAcquirer.ERROR

            if (self.verbose >= 1 and (len(msg) > 0 or self.exitFlag)) or len(self.stdoutBuffer) > 0 or self.verbose >= 3:
                self.log("msg={msg}, exitFlag={exitFlag}".format(msg=msg, exitFlag=self.exitFlag))
                self.log(r'*********************************** /\ {ID} {state} /\ ********************************************'.format(ID=self.ID, state=self.stateList[state]))

            self.flushStdout()

            # Prepare to advance to next state
            lastState = state
            state = nextState

        clearQueue(self.msgQueue)
        clearQueue(self.monitorQueue)
        clearQueue(self.analysisQueue)
        if self.verbose >= 1: self.log("Audio acquire process STOPPED")

        self.flushStdout()
        self.updatePublishedState(self.DEAD)

class AudioWriter(StateMachineProcess):
    # States:
    STOPPED = 0
    INITIALIZING = 1
    WRITING = 2
    BUFFERING = 3
    STOPPING = 4
    ERROR = 5
    EXITING = 6
    DEAD = 100

    stateList = {
        -1:'UNKNOWN',
        STOPPED :'STOPPED',
        INITIALIZING :'INITIALIZING',
        WRITING :'WRITING',
        BUFFERING:'BUFFERING',
        STOPPING :'STOPPING',
        ERROR :'ERROR',
        EXITING :'EXITING',
        DEAD :'DEAD'
    }

    #messages:
    START = 'msg_start'
    STOP = 'msg_stop'
    EXIT = 'msg_exit'
    TRIGGER = 'msg_trigger'
    SETPARAMS = 'msg_setParams'

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
        state = AudioWriter.STOPPED
        nextState = AudioWriter.STOPPED
        lastState = AudioWriter.STOPPED
        msg = ''; arg = None

        while True:
            # Publish updated state
            if state != lastState:
                self.updatePublishedState(state)

            try:
# ********************************* STOPPPED *********************************
                if state == AudioWriter.STOPPED:
                    # DO STUFF

                    # CHECK FOR MESSAGES
                    try:
                        msg, arg = self.msgQueue.get(block=True)
                        if msg == AudioWriter.SETPARAMS: self.setParams(**arg); msg = ''; arg=None
                    except queue.Empty: msg = ''; arg = None

                    # CHOOSE NEXT STATE
                    if self.exitFlag:
                        nextState = AudioWriter.EXITING
                    elif msg == '':
                        nextState = AudioWriter.STOPPED
                    elif msg == AudioWriter.STOP:
                        nextState = AudioWriter.STOPPED
                    elif msg == AudioWriter.START:
                        nextState = AudioWriter.INITIALIZING
                    elif msg == AudioWriter.EXIT:
                        self.exitFlag = True
                        nextState = AudioWriter.EXITING
                    else:
                        raise SyntaxError("Message \"" + msg + "\" not relevant to " + self.stateList[state] + " state")
# ********************************* INITIALIZING *****************************
                elif state == AudioWriter.INITIALIZING:
                    # DO STUFF
                    triggers = []
                    audioChunk = None
                    audioFile = None
                    audioFileStartTime = 0;
                    timeWrote = 0

                    # Read actual audio frequency from the Synchronizer process
                    while self.audioFrequencyVar.value == -1:
                        # Wait for shared value audioFrequency to be set by the Synchronizer process
                        time.sleep(0.1)
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
                        nextState = AudioWriter.STOPPING
                    elif msg in ['', AudioWriter.START]:
                        nextState = AudioWriter.BUFFERING
                    elif msg == AudioWriter.STOP:
                        nextState = AudioWriter.STOPPING
                    elif msg == AudioWriter.EXIT:
                        self.exitFlag = True
                        nextState = AudioWriter.STOPPING
                    else:
                        raise SyntaxError("Message \"" + msg + "\" not relevant to " + self.stateList[state] + " state")
# ********************************* BUFFERING ********************************
                elif state == AudioWriter.BUFFERING:
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
                        nextState = AudioWriter.STOPPING
                    elif msg == AudioWriter.EXIT:
                        self.exitFlag = True
                        nextState = AudioWriter.STOPPING
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
                                nextState = AudioWriter.BUFFERING
                            elif chunkEndTriggerState >= 0 and ((chunkStartTriggerState < 0) or (delta < (1/self.audioFrequency))):
                                # Chunk overlaps start of trigger, or starts within one sample duration of the start of the trigger
                                if self.verbose >= 1: self.log("Got trigger {id} start!".format(id=triggers[0].id))
                                timeWrote = 0
                                nextState = AudioWriter.WRITING
                            elif chunkStartTriggerState == 0 or (chunkStartTriggerState < 0 and chunkStartTriggerState > 0):
                                # Chunk overlaps trigger, but not the start of the trigger
                                if self.verbose >= 0:
                                    self.log("Partially missed audio trigger {id} by {t} seconds, which is {s} samples and {c} chunks!".format(t=delta, s=delta * self.audioFrequency, c=delta * self.audioFrequency / self.chunkSize, id=triggers[0].id))
                                timeWrote = 0
                                nextState = AudioWriter.WRITING
                            else:
                                # Time is after trigger range...
                                if self.verbose >= 0: self.log("Warning, completely missed entire audio trigger {id}!".format(id=triggers[0].id))
                                timeWrote = 0
                                nextState = AudioWriter.BUFFERING
                                triggers.pop(0)   # Pop off trigger that we missed
                        else:
                            # No audio chunks have been received yet, can't evaluate if trigger time has begun yet
                            if self.verbose >= 1: self.log("No audio chunks yet, can't begin trigger yet (buffer: {len}/{maxlen})".format(len=len(self.buffer), maxlen=self.bufferSize))
                            nextState = AudioWriter.BUFFERING
                    elif msg in ['', AudioWriter.START]:
                        nextState = AudioWriter.BUFFERING
                    else:
                        raise SyntaxError("Message \"" + msg + "\" not relevant to " + self.stateList[state] + " state")
# ********************************* WRITING *********************************
                elif state == AudioWriter.WRITING:
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
                        nextState = AudioWriter.STOPPING
                    elif msg == AudioWriter.STOP:
                        nextState = AudioWriter.STOPPING
                    elif msg == AudioWriter.EXIT:
                        self.exitFlag = True
                        nextState = AudioWriter.STOPPING
                    elif msg in ['', AudioWriter.START]:
                        nextState = AudioWriter.WRITING
                        if len(triggers) > 0 and audioChunk is not None:
                            chunkStartTriggerState, chunkEndTriggerState = audioChunk.getTriggerState(triggers[0])
                            if self.verbose >= 3: self.log("Chunk {cid} trigger {tid} state: |{startState} ---- {endState}|".format(startState=chunkStartTriggerState, endState=chunkEndTriggerState, tid=triggers[0].id, cid=audioChunk.id))
                            if chunkStartTriggerState * chunkEndTriggerState > 0:
                                # Trigger period does not overlap the chunk at all - return to buffering
                                if self.verbose >= 2: self.log("Audio chunk {cid} does not overlap trigger {tid}. Switching to buffering.".format(cid=audioChunk.id, tid=triggers[0].id))
                                nextState = AudioWriter.BUFFERING
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
                        raise SyntaxError("Message \"" + msg + "\" not relevant to " + self.stateList[state] + " state")
# ********************************* STOPPING *********************************
                elif state == AudioWriter.STOPPING:
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
                        nextState = AudioWriter.STOPPED
                    elif msg == '':
                        nextState = AudioWriter.STOPPED
                    elif msg == AudioWriter.STOP:
                        nextState = AudioWriter.STOPPED
                    elif msg == AudioWriter.EXIT:
                        self.exitFlag = True
                        nextState = AudioWriter.STOPPED
                    elif msg == AudioWriter.START:
                        nextState = AudioWriter.INITIALIZING
                    else:
                        raise SyntaxError("Message \"" + msg + "\" not relevant to " + self.stateList[state] + " state")
# ********************************* ERROR *********************************
                elif state == AudioWriter.ERROR:
                    # DO STUFF

                    if self.verbose >= 0:
                        self.log("ERROR STATE. Error messages:\n\n")
                        self.log("\n\n".join(self.errorMessages))
                    self.errorMessages = []

                    # CHECK FOR MESSAGES
                    try:
                        msg, arg = self.msgQueue.get(block=False)
                        if msg == AudioWriter.SETPARAMS: self.setParams(**arg); msg = ''; arg=None
                    except queue.Empty: msg = ''; arg = None

                    # CHOOSE NEXT STATE
                    if lastState == AudioWriter.ERROR:
                        # Error ==> Error, let's just exit
                        nextState = AudioWriter.EXITING
                    elif msg == '':
                        if lastState == AudioWriter.STOPPING:
                            # We got an error in stopping state? Better just stop.
                            nextState = AudioWriter.STOPPED
                        elif lastState ==AudioWriter.STOPPED:
                            # We got an error in the stopped state? Better just exit.
                            nextState = AudioWriter.EXITING
                        else:
                            nextState = AudioWriter.STOPPING
                    elif msg == AudioWriter.STOP:
                        nextState = AudioWriter.STOPPED
                    elif msg == AudioWriter.EXIT:
                        self.exitFlag = True
                        if lastState == AudioWriter.STOPPING:
                            nextState = AudioWriter.EXITING
                        else:
                            nextState = AudioWriter.STOPPING
                    else:
                        raise SyntaxError("Message \"" + msg + "\" not relevant to " + self.stateList[state] + " state")
# ********************************* EXIT *********************************
                elif state == AudioWriter.EXITING:
                    break
                else:
                    raise KeyError("Unknown state: "+self.stateList[state])
            except KeyboardInterrupt:
                # Handle user using keyboard interrupt
                if self.verbose >= 0: self.log("Keyboard interrupt received - exiting")
                self.exitFlag = True
                nextState = AudioWriter.STOPPING
            except:
                # HANDLE UNKNOWN ERROR
                self.errorMessages.append("Error in "+self.stateList[state]+" state\n\n"+traceback.format_exc())
                nextState = AudioWriter.ERROR

            if (self.verbose >= 1 and (len(msg) > 0 or self.exitFlag)) or len(self.stdoutBuffer) > 0 or self.verbose >= 3:
                self.log("msg={msg}, exitFlag={exitFlag}".format(msg=msg, exitFlag=self.exitFlag))
                self.log(r'*********************************** /\ {ID} {state} /\ ********************************************'.format(ID=self.ID, state=self.stateList[state]))

            self.flushStdout()

            # Prepare to advance to next state
            lastState = state
            state = nextState

        clearQueue(self.msgQueue)
        if self.verbose >= 1: self.log("Audio write process STOPPED")

        self.flushStdout()
        self.updatePublishedState(self.DEAD)

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
    # States:
    STOPPED = 0
    INITIALIZING = 1
    ACQUIRING = 2
    STOPPING = 3
    ACQUIRE_READY = 4
    ERROR = 5
    EXITING = 6
    DEAD = 100

    stateList = {
        -1:'UNKNOWN',
        STOPPED :'STOPPED',
        INITIALIZING :'INITIALIZING',
        ACQUIRING :'ACQUIRING',
        STOPPING :'STOPPING',
        ACQUIRE_READY :'ACQUIRE_READY',
        ERROR :'ERROR',
        EXITING :'EXITING',
        DEAD :'DEAD'
        }

    #messages:
    START = 'msg_start'
    STOP = 'msg_stop'
    EXIT = 'msg_exit'
    SETPARAMS = 'msg_setParams'

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
                videoWidth=1280,
                videoHeight=1024,
                ready=None,                        # Synchronization barrier to ensure everyone's ready before beginning
                **kwargs):
        StateMachineProcess.__init__(self, **kwargs)
        self.startTimeSharedValue = startTime
        self.camSerial = camSerial
        self.ID = 'VA_'+self.camSerial
        self.acquireSettings = acquireSettings
        self.requestedFrameRate = requestedFrameRate
        self.frameRateVar = frameRate
        self.frameRate = None
        # self.imageQueue = mp.Queue()
        # self.imageQueue.cancel_join_thread()
        self.bufferSize = int(2*bufferSizeSeconds * self.requestedFrameRate)
        print("Creating shared image sender with max buffer size:", self.bufferSize)
        self.imageQueue = SharedImageSender(
            width=videoWidth,
            height=videoHeight,
            verbose=False,
            outputType='PySpin',
            outputCopy=True,
            lockForOutput=False,
            maxBufferSize=self.bufferSize
        )
        self.imageQueueReceiver = self.imageQueue.getReceiver()

        self.monitorImageSender = SharedImageSender(
            width=videoWidth,
            height=videoHeight,
            outputType='PIL',
            outputCopy=False,
            verbose=False,
            lockForOutput=False,
            maxBufferSize=1
        )
        self.monitorImageReceiver = self.monitorImageSender.getReceiver()
#        self.monitorImageQueue.cancel_join_thread()
        self.monitorMasterFrameRate = monitorFrameRate
        self.ready = ready
        self.frameStopwatch = Stopwatch()
        self.monitorStopwatch = Stopwatch()
        self.exitFlag = False
        self.errorMessages = []
        self.verbose = verbose

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

        self.monitorImageSender.setupBuffers()
        self.imageQueue.setupBuffers()

        state = VideoAcquirer.STOPPED
        nextState = VideoAcquirer.STOPPED
        lastState = VideoAcquirer.STOPPED
        msg = ''; arg = None

        while True:
            # Publish updated state
            if state != lastState:
                self.updatePublishedState(state)

            try:
# ********************************* STOPPPED *********************************
                if state == VideoAcquirer.STOPPED:
                    # DO STUFF

                    # CHECK FOR MESSAGES
                    try:
                        msg, arg = self.msgQueue.get(block=True)
                        if msg == VideoAcquirer.SETPARAMS: self.setParams(**arg); msg = ''; arg=None
                    except queue.Empty: msg = ''; arg = None

                    # CHOOSE NEXT STATE
                    if self.exitFlag:
                        nextState = VideoAcquirer.EXITING
                    elif msg == '':
                        nextState = state
                    elif msg == VideoAcquirer.STOP:
                        nextState = VideoAcquirer.STOPPED
                    elif msg == VideoAcquirer.START:
                        nextState = VideoAcquirer.INITIALIZING
                    elif msg == VideoAcquirer.EXIT:
                        self.exitFlag = True
                        nextState = VideoAcquirer.EXITING
                    else:
                        raise SyntaxError("Message \"" + msg + "\" not relevant to " + self.stateList[state] + " state")
# ********************************* INITIALIZING *********************************
                elif state == VideoAcquirer.INITIALIZING:
                    # DO STUFF
                    system = PySpin.System.GetInstance()
                    camList = system.GetCameras()
                    cam = camList.GetBySerial(self.camSerial)
                    cam.Init()
                    nodemap = cam.GetNodeMap()
                    self.setCameraAttributes(nodemap, self.acquireSettings)

                    monitorFramePeriod = 1.0/self.monitorMasterFrameRate
                    if self.verbose >= 1: self.log("Monitoring with period", monitorFramePeriod)
                    thisTime = 0
                    lastTime = time.time()
                    imageCount = 0
                    im = imp = imageResult = None
                    startTime = None
                    frameTime = None

                    while self.frameRateVar.value == -1:
                        # Wait for shared value frameRate to be set by the Synchronizer process
                        time.sleep(0.1)
                    self.frameRate = self.frameRateVar.value

                    # CHECK FOR MESSAGES
                    try:
                        msg, arg = self.msgQueue.get(block=False)
                        if msg == VideoAcquirer.SETPARAMS: self.setParams(**arg); msg = ''; arg=None
                    except queue.Empty: msg = ''; arg = None

                    # CHOOSE NEXT STATE
                    if self.exitFlag:
                        nextState = VideoAcquirer.STOPPING
                    elif msg in ['', VideoAcquirer.START]:
                        nextState = VideoAcquirer.ACQUIRE_READY
                    elif msg == VideoAcquirer.STOP:
                        nextState = VideoAcquirer.STOPPING
                    elif msg == VideoAcquirer.EXIT:
                        self.exitFlag = True
                        nextState = VideoAcquirer.STOPPING
                    else:
                        raise SyntaxError(self.ID + " Message \"" + msg + "\" not relevant to " + self.stateList[state] + " state")
# ********************************* ACQUIRE_READY *********************************
                elif state == VideoAcquirer.ACQUIRE_READY:
                    # DO STUFF
                    cam.BeginAcquisition()
                    try:
                        if self.ready is not None:
#                            if self.verbose >= 1: self.log('{ID} ready: {parties} {n_waiting}'.format(ID=self.ID, parties=self.ready.parties, n_waiting=self.ready.n_waiting))
                            self.ready.wait()
                    except BrokenBarrierError:
                        if self.verbose >= 0: self.log("Simultaneous start failure")
                        nextState = VideoAcquirer.STOPPING

#                    if self.verbose >= 1: self.log('{ID} passed barrier'.format(ID=self.ID))

                    # CHECK FOR MESSAGES
                    try:
                        msg, arg = self.msgQueue.get(block=False)
                        if msg == VideoAcquirer.SETPARAMS: self.setParams(**arg); msg = ''; arg=None
                    except queue.Empty: msg = ''; arg = None

                    # CHOOSE NEXT STATE
                    if msg in ['', VideoAcquirer.START]:
                        nextState = VideoAcquirer.ACQUIRING
                    elif msg == VideoAcquirer.STOP:
                        nextState = VideoAcquirer.STOPPING
                    elif msg == VideoAcquirer.EXIT:
                        self.exitFlag = True
                        nextState = VideoAcquirer.STOPPING
                    else:
                        raise SyntaxError("Message \"" + msg + "\" not relevant to " + self.stateList[state] + " state")
# ********************************* ACQUIRING *********************************
                elif state == VideoAcquirer.ACQUIRING:
#                    if self.verbose > 1: profiler.enable()
                    # DO STUFF
                    try:
                        #  Retrieve next received image
                        imageResult = cam.GetNextImage()
                        # Get timestamp of first image acquisition
                        if startTime is None:
                            if self.verbose >= 1: self.log("Getting start time from sync process...")
                            while startTime == -1 or startTime is None:
                                startTime = self.startTimeSharedValue.value
                            if self.verbose >= 1: self.log("Got start time from sync process: "+str(startTime))
#                            startTime = time.time_ns() / 1000000000

                        # Time frames, as an extra check
                        self.frameStopwatch.click()
                        if self.verbose >= 3: self.log("Video freq: ", self.frameStopwatch.frequency())

                        #  Ensure image completion
                        if imageResult.IsIncomplete():
                            if self.verbose >= 0: self.log('Image incomplete with image status %d...' % imageResult.GetImageStatus())
                        else:
#                            imageConverted = imageResult.Convert(PySpin.PixelFormat_BGR8)
                            imageCount += 1
                            imageID = imageResult.GetFrameID()
                            if self.verbose >= 3:
                                self.log('# frames:'+str(imageCount))
                                self.log('Image ID:'+str(imageID))
                            frameTime = startTime + imageCount / self.frameRate

                            if self.verbose >= 3: self.log("Got image from camera, t="+str(frameTime))

                            # imp = PickleableImage(imageResult.GetWidth(), imageResult.GetHeight(), 0, 0, imageResult.GetPixelFormat(), imageResult.GetData(), frameTime)

                            # Put image into image queue
                            self.imageQueue.put(imageResult, metadata={'frameTime':frameTime, 'imageID':imageID})
                            if self.verbose >= 3: self.log("Pushed image into buffer")

                            if self.monitorImageSender is not None:
                                # Put the occasional image in the monitor queue for the UI
                                thisTime = time.time()
                                actualMonitorFramePeriod = thisTime - lastTime
                                if (thisTime - lastTime) >= monitorFramePeriod:
                                    try:
                                        self.monitorImageSender.put(imageResult)
                                        if self.verbose >= 3: self.log("Sent frame for monitoring")
                                        lastTime = thisTime
                                    except queue.Full:
                                        if self.verbose >= 3: self.log("Can't put frame in for monitoring - no room")
                                        pass

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
                        nextState = VideoAcquirer.STOPPING
                    elif msg in ['', VideoAcquirer.START]:
                        nextState = VideoAcquirer.ACQUIRING
                    elif msg == VideoAcquirer.STOP:
                        nextState = VideoAcquirer.STOPPING
                    elif msg == VideoAcquirer.EXIT:
                        self.exitFlag = True
                        nextState = VideoAcquirer.STOPPING
                    else:
                        raise SyntaxError("Message \"" + msg + "\" not relevant to " + self.stateList[state] + " state")
#                    if self.verbose > 1: profiler.disable()
# ********************************* STOPPING *********************************
                elif state == VideoAcquirer.STOPPING:
                    # DO STUFF
                    if cam is not None:
                        camList.Clear()
                        cam.EndAcquisition()
                        cam.DeInit()
                        cam = None
                    if system is not None:
                        system.ReleaseInstance()

                    # CHECK FOR MESSAGES
                    try:
                        msg, arg = self.msgQueue.get(block=False)
                        if msg == VideoAcquirer.SETPARAMS: self.setParams(**arg); msg = ''; arg=None
                    except queue.Empty: msg = ''; arg = None

                    # CHOOSE NEXT STATE
                    if self.exitFlag:
                        nextState = VideoAcquirer.STOPPED
                    elif msg == '':
                        nextState = VideoAcquirer.STOPPED
                    elif msg == AudioWriter.STOP:
                        nextState = VideoAcquirer.STOPPED
                    elif msg == VideoAcquirer.EXIT:
                        self.exitFlag = True
                        nextState = VideoAcquirer.STOPPED
                    elif msg == VideoAcquirer.START:
                        nextState = VideoAcquirer.INITIALIZING
                    else:
                        raise SyntaxError("Message \"" + msg + "\" not relevant to " + self.stateList[state] + " state")
# ********************************* ERROR *********************************
                elif state == VideoAcquirer.ERROR:
                    # DO STUFF
                    if self.verbose >= 0:
                        self.log("ERROR STATE. Error messages:\n\n")
                        self.log("\n\n".join(self.errorMessages))
                    self.errorMessages = []

                    # CHECK FOR MESSAGES
                    try:
                        msg, arg = self.msgQueue.get(block=False)
                        if msg == VideoAcquirer.SETPARAMS: self.setParams(**arg); msg = ''; arg=None
                    except queue.Empty: msg = ''; arg = None

                    # CHOOSE NEXT STATE
                    if lastState == VideoAcquirer.ERROR:
                        # Error ==> Error, let's just exit
                        nextState = VideoAcquirer.EXITING
                    elif msg == '':
                        if lastState == VideoAcquirer.STOPPING:
                            # We got an error in stopping state? Better just stop.
                            nextState = VideoAcquirer.STOPPED
                        elif lastState ==VideoAcquirer.STOPPED:
                            # We got an error in the stopped state? Better just exit.
                            nextState = VideoAcquirer.EXITING
                        else:
                            nextState = VideoAcquirer.STOPPING
                    elif msg == VideoAcquirer.STOP:
                        nextState = VideoAcquirer.STOPPING
                    elif msg == VideoAcquirer.EXIT:
                        self.exitFlag = True
                        if lastState == VideoAcquirer.STOPPING:
                            nextState = VideoAcquirer.EXITING
                        else:
                            nextState = VideoAcquirer.STOPPING
                    else:
                        raise SyntaxError("Message \"" + msg + "\" not relevant to " + self.stateList[state] + " state")
# ********************************* EXIT *********************************
                elif state == VideoAcquirer.EXITING:
                    break
                else:
                    raise KeyError("Unknown state: "+self.stateList[state])
            except KeyboardInterrupt:
                # Handle user using keyboard interrupt
                if self.verbose >= 0: self.log("Keyboard interrupt received - exiting")
                self.exitFlag = True
                nextState = VideoAcquirer.STOPPING
            except:
                # HANDLE UNKNOWN ERROR
                self.errorMessages.append("Error in "+self.stateList[state]+" state\n\n"+traceback.format_exc())
                nextState = VideoAcquirer.ERROR

            if (self.verbose >= 1 and (len(msg) > 0 or self.exitFlag)) or len(self.stdoutBuffer) > 0 or self.verbose >= 3:
                self.log("msg={msg}, exitFlag={exitFlag}".format(msg=msg, exitFlag=self.exitFlag))
                self.log(r'*********************************** /\ {ID} {state} /\ ********************************************'.format(ID=self.ID, state=self.stateList[state]))

            self.flushStdout()

            # Prepare to advance to next state
            lastState = state
            state = nextState

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
        self.updatePublishedState(self.DEAD)

    def setCameraAttribute(self, nodemap, attributeName, attributeValue, type='enum'):
        # Set camera attribute. ReturnRetrusn True if successful, False otherwise.
        if self.verbose >= 1: self.log('Setting', attributeName, 'to', attributeValue, 'as', type)
        nodeAttribute = nodeAccessorTypes[type](nodemap.GetNode(attributeName))
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

class VideoWriter(StateMachineProcess):
    # States:
    STOPPED = 0
    INITIALIZING = 1
    WRITING = 2
    BUFFERING = 3
    STOPPING = 4
    ERROR = 5
    EXITING = 6
    DEAD = 100

    stateList = {
        -1:'UNKNOWN',
        STOPPED :'STOPPED',
        INITIALIZING :'INITIALIZING',
        WRITING :'WRITING',
        BUFFERING :'BUFFERING',
        STOPPING :'STOPPING',
        ERROR :'ERROR',
        EXITING :'EXITING',
        DEAD :'DEAD'
    }

    #messages:
    START = 'msg_start'
    STOP = 'msg_stop'
    EXIT = 'msg_exit'
    TRIGGER = 'msg_trigger'
    SETPARAMS = 'msg_setParams'

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

        state = VideoWriter.STOPPED
        nextState = VideoWriter.STOPPED
        lastState = VideoWriter.STOPPED
        msg = ''; arg = None

        while True:
            # Publish updated state
            if state != lastState:
                self.updatePublishedState(state)

            try:
# ********************************* STOPPPED *********************************
                if state == VideoWriter.STOPPED:
                    # DO STUFF

                    # CHECK FOR MESSAGES
                    try:
                        msg, arg = self.msgQueue.get(block=True)
                        if msg == VideoWriter.SETPARAMS: self.setParams(**arg); msg = ''; arg=None
                    except queue.Empty: msg = ''; arg = None

                    # CHOOSE NEXT STATE
                    if self.exitFlag:
                        nextState = VideoWriter.EXITING
                    elif msg == '':
                        nextState = VideoWriter.STOPPED
                    elif msg == VideoWriter.STOP:
                        nextState = VideoWriter.STOPPED
                    elif msg == VideoWriter.START:
                        nextState = VideoWriter.INITIALIZING
                    elif msg == VideoWriter.EXIT:
                        self.exitFlag = True
                        nextState = VideoWriter.EXITING
                    else:
                        raise SyntaxError("Message \"" + msg + "\" not relevant to " + self.stateList[state] + " state")
# ********************************* INITIALIZING *********************************
                elif state == VideoWriter.INITIALIZING:
                    # DO STUFF
                    triggers = []
                    im = None
                    videoFileStartTime = 0
                    videoFileInterface = None
                    timeWrote = 0

                    while self.frameRateVar.value == -1:
                        # Wait for shared value frameRate to be set by the Synchronizer process
                        time.sleep(0.1)
                    self.frameRate = self.frameRateVar.value

                    # CHECK FOR MESSAGES
                    try:
                        msg, arg = self.msgQueue.get(block=False)
                        if msg == VideoWriter.SETPARAMS: self.setParams(**arg); msg = ''; arg=None
                    except queue.Empty: msg = ''; arg = None

                    # CHOOSE NEXT STATE
                    if self.exitFlag:
                        nextState = VideoWriter.STOPPING
                    elif msg in ['', VideoWriter.START]:
                        nextState = VideoWriter.BUFFERING
                    elif msg == VideoWriter.STOP:
                        nextState = VideoWriter.STOPPING
                    elif msg == VideoWriter.EXIT:
                        self.exitFlag = True
                        nextState = VideoWriter.STOPPING
                    else:
                        raise SyntaxError("Message \"" + msg + "\" not relevant to " + self.stateList[state] + " state")
# ********************************* BUFFERING *********************************
                elif state == VideoWriter.BUFFERING:
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
                        nextState = VideoWriter.STOPPING
                    elif msg == VideoWriter.EXIT:
                        self.exitFlag = True
                        nextState = VideoWriter.STOPPING
                    elif len(triggers) > 0:
                        # We have triggers - next state will depend on them
                        if self.verbose >= 2: self.log("" + str(len(triggers)) + " trigger(s) exist:")
                        if im is not None:
                            # At least one video frame has been received - we can check if trigger period has begun
                            triggerState = triggers[0].state(frameTime)
                            if self.verbose >= 2: self.log("Trigger state: {state}".format(state=triggerState))
                            if triggerState < 0:        # Time is before trigger range
                                if self.verbose >= 2: self.log("Active trigger, but haven't gotten to start time yet, continue buffering.")
                                nextState = VideoWriter.BUFFERING
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
                                nextState = VideoWriter.WRITING
                            else:                       # Time is after trigger range
                                if self.verbose >= 0: self.log("Missed entire trigger by {triggerState} seconds!".format(triggerState=triggerState))
                                timeWrote = 0
                                nextState = VideoWriter.BUFFERING
                                triggers.pop(0)
                        else:
                            # No video frames have been received yet, can't evaluate if trigger time has begun yet
                            if self.verbose >= 2: self.log("No frames at the moment, can't begin trigger yet (buffer: {len}/{maxlen})".format(len=len(self.buffer), maxlen=self.bufferSize))
                            nextState = VideoWriter.BUFFERING

                    elif msg in ['', VideoWriter.START]:
                        nextState = VideoWriter.BUFFERING
                    else:
                        raise SyntaxError("Message \"" + msg + "\" not relevant to " + self.stateList[state] + " state")
# ********************************* WRITING *********************************
                elif state == VideoWriter.WRITING:
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
                        nextState = VideoWriter.STOPPING
                    elif msg == VideoWriter.STOP:
                        nextState = VideoWriter.STOPPING
                    elif msg == VideoWriter.EXIT:
                        self.exitFlag = True
                        nextState = VideoWriter.STOPPING
                    elif msg in ['', VideoWriter.START]:
                        nextState = VideoWriter.WRITING
                        if len(triggers) > 0 and im is not None:
                            triggerState = triggers[0].state(frameTime)
                            if triggerState != 0:
                                # Frame is not in trigger period - return to buffering
                                if self.verbose >= 2: self.log("Frame does not overlap trigger. Switching to buffering.")
                                nextState = VideoWriter.BUFFERING
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
                        raise SyntaxError("Message \"" + msg + "\" not relevant to " + self.stateList[state] + " state")
                    # if self.verbose >= 1: profiler.disable()
# ********************************* STOPPING *********************************
                elif state == VideoWriter.STOPPING:
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
                        nextState = VideoWriter.STOPPED
                    elif msg == '':
                        nextState = VideoWriter.BUFFERING
                    elif msg == VideoWriter.STOP:
                        nextState = VideoWriter.STOPPED
                    elif msg == VideoWriter.EXIT:
                        self.exitFlag = True
                        nextState = VideoWriter.STOPPED
                    elif msg == VideoWriter.START:
                        nextState = VideoWriter.INITIALIZING
                    else:
                        raise SyntaxError("Message \"" + msg + "\" not relevant to " + self.stateList[state] + " state")
# ********************************* ERROR *********************************
                elif state == VideoWriter.ERROR:
                    # DO STUFF
                    if self.verbose >= 0:
                        self.log("ERROR STATE. Error messages:\n\n")
                        self.log("\n\n".join(self.errorMessages))
                    self.errorMessages = []

                    # CHECK FOR MESSAGES
                    try:
                        msg, arg = self.msgQueue.get(block=False)
                        if msg == VideoWriter.SETPARAMS: self.setParams(**arg); msg = ''; arg=None
                    except queue.Empty: msg = ''; arg = None

                    # CHOOSE NEXT STATE
                    if lastState == VideoWriter.ERROR:
                        # Error ==> Error, let's just exit
                        nextState = VideoWriter.EXITING
                    elif msg == '':
                        if lastState == VideoWriter.STOPPING:
                            # We got an error in stopping state? Better just stop.
                            nextState = VideoWriter.STOPPED
                        elif lastState ==VideoWriter.STOPPED:
                            # We got an error in the stopped state? Better just exit.
                            nextState = VideoWriter.EXITING
                        else:
                            nextState = VideoWriter.STOPPING
                    elif msg == VideoWriter.STOP:
                        nextState = VideoWriter.STOPPED
                    elif msg == VideoWriter.EXIT:
                        self.exitFlag = True
                        if lastState == VideoWriter.STOPPING:
                            nextState = VideoWriter.EXITING
                        else:
                            nextState = VideoWriter.STOPPING
                    else:
                        raise SyntaxError("Message \"" + msg + "\" not relevant to " + self.stateList[state] + " state")
# ********************************* EXIT *********************************
                elif state == VideoWriter.EXITING:
                    break
                else:
                    raise KeyError("Unknown state: "+self.stateList[state])
            except KeyboardInterrupt:
                # Handle user using keyboard interrupt
                if self.verbose >= 0: self.log("Keyboard interrupt received - exiting")
                self.exitFlag = True
                nextState = VideoWriter.STOPPING
            except:
                # HANDLE UNKNOWN ERROR
                self.errorMessages.append("Error in "+self.stateList[state]+" state\n\n"+traceback.format_exc())
                nextState = VideoWriter.ERROR

            if (self.verbose >= 1 and (len(msg) > 0 or self.exitFlag)) or len(self.stdoutBuffer) > 0 or self.verbose >= 3:
                self.log("msg={msg}, exitFlag={exitFlag}".format(msg=msg, exitFlag=self.exitFlag))
                self.log(r'*********************************** /\ {ID} {state} /\ ********************************************'.format(ID=self.ID, state=self.stateList[state]))

            self.flushStdout()

            # Prepare to advance to next state
            lastState = state
            state = nextState

        if self.verbose >= 1: self.log("Video write process STOPPED")
        # if self.verbose > 1:
        #     s = io.StringIO()
        #     ps = pstats.Stats(profiler, stream=s)
        #     ps.print_stats()
        #     self.log(s.getvalue())
        self.flushStdout()
        self.updatePublishedState(self.DEAD)

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
    ContinuousTriggerer: A state machine class to automatically generate a
        continuous train of triggers for both audio and video writer processes.
    '''
    # States:
    STOPPED = 0
    INITIALIZING = 1
    TRIGGERING = 3
    STOPPING = 4
    ERROR = 5
    EXITING = 6
    DEAD = 100

    stateList = {
        -1:'UNKNOWN',
        STOPPED :'STOPPED',
        INITIALIZING :'INITIALIZING',
        TRIGGERING :'TRIGGERING',
        STOPPING :'STOPPING',
        ERROR :'ERROR',
        EXITING :'EXITING',
        DEAD :'DEAD'
    }

    #messages:
    START = 'msg_start'
    STOP = 'msg_stop'
    EXIT = 'msg_exit'
    SETPARAMS = 'msg_setParams'
    TAGTRIGGER = 'tag_trigger'      # Send a trigger that indicates the videos that overlap with that trigger should be tagged with trigger's tags

    settableParams = [
        'continuousTriggerPeriod',
        'scheduleEnabled',
        'scheduleStartTime',
        'scheduleStopTime'
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
        state = ContinuousTriggerer.STOPPED
        nextState = ContinuousTriggerer.STOPPED
        lastState = ContinuousTriggerer.STOPPED
        tagTriggers = []   # A list of triggers received from other processes that will be used to tag overlapping audio/video files
        msg = ''; arg = None

        while True:
            # Publish updated state
            if state != lastState:
                self.updatePublishedState(state)

            try:
# ********************************* STOPPPED *********************************
                if state == ContinuousTriggerer.STOPPED:
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
                        nextState = ContinuousTriggerer.EXITING
                    elif msg == '':
                        nextState = ContinuousTriggerer.STOPPED
                    elif msg == ContinuousTriggerer.STOP:
                        nextState = ContinuousTriggerer.STOPPED
                    elif msg == ContinuousTriggerer.START:
                        nextState = ContinuousTriggerer.INITIALIZING
                    else:
                        raise SyntaxError("Message \"" + msg + "\" not relevant to " + self.stateList[state] + " state")
# ********************************* INITIALIZING *****************************
                elif state == ContinuousTriggerer.INITIALIZING:
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
                        nextState = ContinuousTriggerer.STOPPING
                    elif msg == ContinuousTriggerer.STOP:
                        nextState = ContinuousTriggerer.STOPPING
                    elif msg in ['', ContinuousTriggerer.START]:
                        nextState = ContinuousTriggerer.TRIGGERING
                    else:
                        raise SyntaxError("Message \"" + msg + "\" not relevant to " + self.stateList[state] + " state")
# ********************************* TRIGGERING *********************************
                elif state == ContinuousTriggerer.TRIGGERING:
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
                        nextState = ContinuousTriggerer.STOPPING
                    elif msg == ContinuousTriggerer.STOP:
                        nextState = ContinuousTriggerer.STOPPING
                    elif msg in ['', ContinuousTriggerer.START]:
                        nextState = state
                    else:
                        raise SyntaxError("Message \"" + msg + "\" not relevant to " + self.stateList[state] + " state")
# ********************************* STOPPING *********************************
                elif state == ContinuousTriggerer.STOPPING:
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
                        nextState = ContinuousTriggerer.STOPPED
                    elif msg == '':
                        nextState = ContinuousTriggerer.STOPPED
                    elif msg == ContinuousTriggerer.STOP:
                        nextState = ContinuousTriggerer.STOPPED
                    elif msg == ContinuousTriggerer.EXIT:
                        self.exitFlag = True
                        nextState = ContinuousTriggerer.STOPPED
                    elif msg == ContinuousTriggerer.START:
                        nextState = ContinuousTriggerer.INITIALIZING
                    else:
                        raise SyntaxError("Message \"" + msg + "\" not relevant to " + self.stateList[state] + " state")
# ********************************* ERROR *********************************
                elif state == ContinuousTriggerer.ERROR:
                    # DO STUFF
                    if self.verbose >= 0:
                        self.log("ERROR STATE. Error messages:\n\n")
                        self.log("\n\n".join(self.errorMessages))
                    self.errorMessages = []

                    # CHECK FOR MESSAGES
                    try:
                        msg, arg = self.msgQueue.get(block=False)
                        if msg == ContinuousTriggerer.SETPARAMS: self.setParams(**arg); msg = ''; arg=None
                        elif msg == ContinuousTriggerer.TAGTRIGGER: msg = ''; arg=None  # Ignore tag triggers if we are in an error state
                    except queue.Empty: msg = ''; arg = None

                    # CHOOSE NEXT STATE
                    if lastState == ContinuousTriggerer.ERROR:
                        # Error ==> Error, let's just exit
                        nextState = ContinuousTriggerer.EXITING
                    elif msg == '':
                        if lastState == ContinuousTriggerer.STOPPING:
                            # We got an error in stopping state? Better just stop.
                            nextState = ContinuousTriggerer.STOPPED
                        elif lastState ==ContinuousTriggerer.STOPPED:
                            # We got an error in the stopped state? Better just exit.
                            nextState = ContinuousTriggerer.EXITING
                        else:
                            nextState = ContinuousTriggerer.STOPPING
                    elif msg == ContinuousTriggerer.STOP:
                        nextState = ContinuousTriggerer.STOPPED
                    elif msg == ContinuousTriggerer.EXIT:
                        self.exitFlag = True
                        if lastState == ContinuousTriggerer.STOPPING:
                            nextState = ContinuousTriggerer.EXITING
                        else:
                            nextState = ContinuousTriggerer.STOPPING
                    else:
                        raise SyntaxError("Message \"" + msg + "\" not relevant to " + self.stateList[state] + " state")
# ********************************* EXIT *********************************
                elif state == ContinuousTriggerer.EXITING:
                    break
                else:
                    raise KeyError("Unknown state: "+self.stateList[state])
            except KeyboardInterrupt:
                # Handle user using keyboard interrupt
                if self.verbose >= 0: self.log("Keyboard interrupt received - exiting")
                self.exitFlag = True
                nextState = ContinuousTriggerer.STOPPING
            except:
                # HANDLE UNKNOWN ERROR
                self.errorMessages.append("Error in "+self.stateList[state]+" state\n\n"+traceback.format_exc())
                nextState = ContinuousTriggerer.ERROR

            if (self.verbose >= 1 and (len(msg) > 0 or self.exitFlag)) or len(self.stdoutBuffer) > 0 or self.verbose >= 3:
                self.log("msg={msg}, exitFlag={exitFlag}".format(msg=msg, exitFlag=self.exitFlag))
                self.log(r'*********************************** /\ {ID} {state} /\ ********************************************'.format(ID=self.ID, state=self.stateList[state]))

            self.flushStdout()

            # Prepare to advance to next state
            lastState = state
            state = nextState

        clearQueue(self.msgQueue)
        if self.verbose >= 1: self.log("Audio write process STOPPED")

        self.flushStdout()
        self.updatePublishedState(self.DEAD)

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
