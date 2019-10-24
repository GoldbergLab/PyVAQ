import struct
import time
import numpy as np
import multiprocessing as mp
import nidaqmx
from nidaqmx.stream_readers import AnalogSingleChannelReader
from nidaqmx.stream_writers import CounterWriter

class Synchronizer(mp.Process):
    # Class for generating two synchronization signals at the same time
    #   - one for video (send via cable to the camera GPIO)
    #   - one for audio (used internally to trigger analog input of microphone
    #     signals)
    # This class inherits from multiprocessing.Process so it can be run in a
    #   separate process, allowing a single script to generate the sync pulses
    #   and also accomplish other tasks.
    def __init__(self,
        videoFrequency=120,                     # The frequency in Hz of the video sync signal
        audioFrequency=44100,                   # The frequency in Hz of the audio sync signal
        videoSyncChannel="Dev3/ctr0",           # The counter channel on which to generate the video sync signal
        audioSyncChannel="Dev3/ctr1"):          # The counter channel on which to generate the audio sync signal
        mp.Process.__init__(self)
        # Store inputs in instance variables for later access
        self.videoFrequency = videoFrequency
        self.audioFrequency = audioFrequency
        self.videoSyncChannel = videoSyncChannel
        self.audioSyncChannel = audioSyncChannel
        self.stop = mp.Event()                      # An event to gracefully halt this process

    def stopProcess(self):
        # Method to shut down process gracefully
        print('Stopping synchronization signal process')
        self.stop.set()

    def run(self):
        # Configure and generate synchronization signal
        with nidaqmx.Task() as trigTask:                       # Create task
            trigTask.co_channels.add_co_pulse_chan_freq(
                counter=self.videoSyncChannel,
                name_to_assign_to_channel="videoSync",
                units=nidaqmx.constants.FrequencyUnits.HZ,
                initial_delay=0.0,
                freq=self.videoFrequency,
                duty_cycle=0.5)     # Prepare a counter output channel for the video sync signal
            trigTask.co_channels.add_co_pulse_chan_freq(
                counter=self.audioSyncChannel,
                name_to_assign_to_channel="audioSync",
                units=nidaqmx.constants.FrequencyUnits.HZ,
                initial_delay=0.0,
                freq=self.audioFrequency,
                duty_cycle=0.5)     # Prepare a counter output channel for the audio sync signal
            trigTask.timing.cfg_implicit_timing(sample_mode=nidaqmx.constants.AcquisitionType.CONTINUOUS)
            trigTask.start()
            print("Synchronization process STARTED")

            while not self.stop.is_set():   # Continue process until stop signal is received
                time.sleep(0.5)  # May be necessary to prevent the daqmx tasks from terminating
        print("Synchronization process STOPPED")

class AudioAcquirer(mp.Process):
    # Class for acquiring an audio signal (or any analog signal) at a rate that
    #   is synchronized to the rising edges on the specified synchronization
    #   channel.
    def __init__(self,
                audioQueue = None,                  # A multiprocessing queue to send data to another proces for writing to disk
                audioMonitorQueue = None,           # A multiprocessing queue to send data to the UI to monitor the audio
                sampleChunkSize = 44100,            # Size of the read chunk in samples
                maxExpectedSamplingRate = 44100,    # Maximum expected rate of the specified synchronization channel
                bufferSize = None,                  # Size of device buffer. Defaults to 1 second's worth of data
                channelName = None,                 # Channel name for analog input (microphone signal)
                syncChannel = None):                # Channel name for synchronization source
        mp.Process.__init__(self)
        # Store inputs in instance variables for later access
        if bufferSize is None:
            self.bufferSize = maxExpectedSamplingRate  # Device buffer size defaults to One second's worth of buffer
        else:
            self.bufferSize = bufferSize
        self.audioQueue = audioQueue
        self.audioMonitorQueue = audioMonitorQueue
        self.sampleChunkSize = sampleChunkSize
        self.maxExpectedSamplingRate = maxExpectedSamplingRate
        self.inputChannel = channelName
        self.syncChannel = syncChannel
        self.stop = mp.Event()

    def stopProcess(self):
        # Method to shut down process gracefully
        print('Stopping audio acquire process')
        self.stop.set()

    def rescaleAudio(data, maxV=10, minV=-10, maxD=32767, minD=-32767):
        return (data * ((maxD-minD)/(maxV-minV))).astype('int16')

    def run(self):
        # Configure analog acquisition and begin acquisition
        data = np.zeros(self.sampleChunkSize, dtype='float')        # A pre-allocated array to receive audio data
        with nidaqmx.Task() as readTask:                            # Create task
            print("setting up task")
            print("timing channel:", self.syncChannel)
            print("audio channel:", self.inputChannel)
            readTask.ai_channels.add_ai_voltage_chan(               # Set up analog input channel
                self.inputChannel,
                terminal_config=nidaqmx.constants.TerminalConfiguration.DIFFERENTIAL,
                max_val=10,
                min_val=-10)
            readTask.timing.cfg_samp_clk_timing(                    # Configure clock source for triggering each analog read
                rate=self.maxExpectedSamplingRate,
                source=self.syncChannel,                            # Specify a timing source!
                active_edge=nidaqmx.constants.Edge.RISING,
                sample_mode=nidaqmx.constants.AcquisitionType.CONTINUOUS,
                samps_per_chan=self.bufferSize)

            reader = AnalogSingleChannelReader(readTask.in_stream)  # Set up an analog stream reader



            print("Audio acquire process STARTED")
            while not self.stop.is_set():  # Exit signal received
                reader.read_many_sample(                            # Read a chunk of audio data
                    data,
                    number_of_samples_per_channel=self.sampleChunkSize,
                    timeout=10.0)
                processedData = AudioAcquirer.rescaleAudio(data).tolist()
                print(processedData)
                if self.audioQueue is not None:
                    self.audioQueue.put(processedData)              # If a data queue is provided, queue up the new data
                else:
                    print(processedData)
                if self.audioMonitorQueue is not None:
                    self.audioMonitorQueue.put((self.inputChannel, processedData))      # If a monitoring queue is provided, queue up the data
            # Send stop signal to write process
            self.audioQueue.put(None)
        print("Audio acquire process STOPPED")

if __name__ == '__main__':
    # Create synchronizer process
    s = Synchronizer(audioSyncChannel='Dev3/ctr0', videoSyncChannel='Dev3/ctr1', audioFrequency=22050)
    # Create audio acquisition process
    a = AudioAcquirer(channelName='Dev3/ai5', syncChannel='PFI4')   # Also tried 'Dev3/Ctr0InternalOutput', same error.
    # Start audio acquisition (no acquisition should actually occur until sync signal starts up)
    a.start()
    # Wait until audio acquire process starts up
    x = input("Press any key to begin sending sync pulses! \n ")
    s.start()
