import nidaqmx
from nidaqmx.types import CtrFreq

import time

with nidaqmx.Task() as task:
    task.co_channels.add_co_pulse_chan_freq(counter="Dev3/ctr0", units=nidaqmx.constants.FrequencyUnits.HZ, initial_delay=0.0, freq=120.0, duty_cycle=0.5)

    # sample = CtrTime(high_time=0.001, low_time=0.002)

    task.timing.cfg_implicit_timing(sample_mode=nidaqmx.constants.AcquisitionType.CONTINUOUS) # , samps_per_chan=10)
    task.start()
    time.sleep(60)

    # print('1 Channel 1 Sample Write: ')
    # print(task.write(sample))

    # print(task.write([CtrFreq() for x in range(10)], auto_start=True))
