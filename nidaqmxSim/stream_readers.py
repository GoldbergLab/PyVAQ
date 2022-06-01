from pathlib import Path
from scipy.io import wavfile

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

class AnalogMultiChannelReader:
    def __init__(self, in_stream, audio_filenames=DEFAULT_AUDIO_FILENAMES):
        self.in_stream = in_stream
        self.data = []
        self.framerates = []
        self.filePointers = []
        self.numSamples = []
        rootPath = Path(__file__).parents[0]
        for filename in audio_filenames:
            framerate, data = wavfile.read(rootPath / filename)
        self.data.append(data)
        self.framerates.append(data)
        self.filePointers.append(0)
        self.numSamples.append(data.size)

    def read_many_sample(self, buffer, number_of_samples_per_channel, timeout):
        s = number_of_samples_per_channel
        num_channels = len(self.in_stream.parent_task.ai_channels)
        for chan in range(num_channels):
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
