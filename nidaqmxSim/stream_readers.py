from pathlib import Path
from scipy.io import wavfile
from time import time, sleep
import numpy as np

DEFAULT_AUDIO_FILENAMES = [
    'simAudioFeed0.wav',
    'simAudioFeed1.wav',
    'simAudioFeed2.wav'
]

class DigitalSingleChannelReader:
    def __init__(self, in_stream):
        self.in_stream = in_stream

    def read_one_sample_one_line(self):
        return True

class DigitalMultiChannelReader:
    """Simulated stand-in for nidaqmx.stream_readers.DigitalMultiChannelReader.

    Paces reads to the task's sample clock rate (like the analog simulator)
    and fills the buffer with a simple alternating square-wave pattern so the
    digital monitor has something to display.
    """
    def __init__(self, in_stream):
        self.in_stream = in_stream
        self.sampleCount = 0
        self.sampleRate = None
        self.nextReadTime = None

    def read_many_sample_port_uint32(self, buffer, number_of_samples_per_channel, timeout=10.0):
        if self.sampleRate is None:
            # Taken from the task's sample clock configuration if available
            timing = getattr(self.in_stream.parent_task, 'timing', None)
            self.sampleRate = getattr(timing, 'rate', None) or 1000.0
            self.nextReadTime = time()
        framePeriod = number_of_samples_per_channel / self.sampleRate
        while time() < self.nextReadTime:
            sleep(framePeriod / 5)
        self.nextReadTime += framePeriod
        # Square wave toggling every sampleRate/10 samples on each line/port
        idx = self.sampleCount + np.arange(number_of_samples_per_channel)
        pattern = ((idx // max(1, int(self.sampleRate // 10))) % 2).astype('uint32')
        buffer[:, :] = pattern[np.newaxis, :]
        self.sampleCount += number_of_samples_per_channel
        return number_of_samples_per_channel

class AnalogMultiChannelReader:
    def __init__(self, in_stream, audio_filenames=DEFAULT_AUDIO_FILENAMES):
        self.in_stream = in_stream
        self.data = []
        self.sampleRate = None
        self.filePointers = []
        self.numSamples = []
        self.num_channels = len(self.in_stream.parent_task.ai_channels)
        rootPath = Path(__file__).parents[0]
        for filename in audio_filenames:
            sampleRate, data = wavfile.read(rootPath / filename)
            self.data.append(data)
            if self.sampleRate is None:
                self.sampleRate = sampleRate
            else:
                if self.sampleRate != sampleRate:
                    print('Warning, not all loaded simulated audio feeds have the same sample rate.')
            self.filePointers.append(0)
            self.numSamples.append(data.size)
        startTime = time()
        self.lastReadTime = startTime
        self.nextReadTime = startTime

    def read_many_sample(self, buffer, number_of_samples_per_channel, timeout):
        framePeriod = number_of_samples_per_channel/self.sampleRate
        while True:
            currentTime = time()
            if currentTime >= self.nextReadTime:
                break
            else:
                sleep(framePeriod/5)
        self.lastReadTime = self.nextReadTime
        self.nextReadTime += framePeriod
        for chan in range(self.num_channels):
            fileNum = chan % length(self.data)
            p = self.filePointers[fileNum]
            n = self.numSamples[fileNum]
            if p + s >= n:
                # Wrap around
                buffer[chan, :(n-p)] = self.data[fileNum][p:]
                buffer[chan, (n-p):] = self.data[fileNum][:(p+s-n)]
            else:
                buffer[chan, :] = self.data[fileNum][p:(p+s)]
            self.filePointers[fileNum] = (self.filePointers[fileNum] + s) % n
