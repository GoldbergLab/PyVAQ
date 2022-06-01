class Task:
    def __init__(self, new_task_name=""):
        self.name = new_task_name
        self.in_stream = InStream(parent_task=self)
        self.ai_channels = AIChannelCollection()
        self.di_channels = DIChannelCollection()
        self.co_channels = COChannelCollection()
        self.timing = Timing()

    def start(self):
        pass

    def stop(self):
        pass

    def is_task_done(self):
        return False

    def close(self):
        pass

class COChannel:
    def __init__(
        self,
        channel,
        units=None,
        initial_delay=None,
        freq=None,
        duty_cycle=None
        ):
        self.name = channel
        self.channel = channel
        self.units = units
        self.initial_delay = initial_delay
        self.co_pulse_freq = freq
        self.duty_cycle = duty_cycle

    def __str__(self):
        return self.channel

class AIChannel:
    def __init__(self, channel, terminal_config=None, max_val=None, min_val=None):
        self.name = channel
        self.channel = channel
        self.terminal_config = terminal_config
        self.max_val = max_val
        self.min_val = min_val

    def __str__(self):
        return self.channel

class DIChannel:
    def __init__(self, channel):
        self.name = channel
        self.channel = channel

    def __str__(self):
        return self.channel

class ChannelCollection:
    def __init__(self):
        self.channels = {}

    def get_new_name(self):
        c = 0
        while True:
            newName = 'chan{c}'.format(c=c)
            if newName not in self.channels:
                return newName
            c = c + 1

    def __getitem__(self, item):
        return self.channels[item]

    def __len__(self):
        return len(self.channels)

class COChannelCollection(ChannelCollection):
    def __init__(self):
        super().__init__()

    def add_co_pulse_chan_freq(
        self,
        counter,
        name_to_assign_to_channel,
        units,
        initial_delay,
        freq,
        duty_cycle
        ):

        self.channels[name_to_assign_to_channel] = COChannel(
            channel=counter,
            units=units,
            initial_delay=initial_delay,
            freq=freq,
            duty_cycle=duty_cycle
            )
        return self.channels[name_to_assign_to_channel]

class DIChannelCollection(ChannelCollection):
    def __init__(self):
        super().__init__()

    def add_di_chan(
        self,
        chan
        ):
        self.channels[chan] = DIChannel(chan)
        return self.channels[chan]

class AIChannelCollection(ChannelCollection):
    def __init__(self):
        super().__init__()

    def add_ai_voltage_chan(
        self,
        input_channel,
        name_to_assign_to_channel="",
        terminal_config=None,
        max_val=10,
        min_val=-10):

        if len(name_to_assign_to_channel) == 0:
            name_to_assign_to_channel = self.get_new_name()

        newChannel = AIChannel(
            input_channel,
            terminal_config,
            max_val,
            min_val
        )
        self.channels = []

class Timing:
    def __init__(self):
        self.rate = None
        self.source = None
        self.active_edge = None
        self.sample_mode = None
        self.samps_per_chan = None

    def cfg_implicit_timing(
        self,
        rate=None,
        source=None,
        active_edge=None,
        sample_mode=None,
        samps_per_chan=None
        ):

        if rate is not None:
            self.rate = rate
        if source is not None:
            self.source = source
        if active_edge is not None:
            self.active_edge = active_edge
        if sample_mode is not None:
            self.sample_mode = sample_mode
        if samps_per_chan is not None:
            self.samps_per_chan = samps_per_chan

    def cfg_samp_clk_timing(
        self,
        rate=None,
        source=None,
        active_edge=None,
        sample_mode=None,
        samps_per_chan=None
        ):

        if rate is not None:
            self.rate = rate
        if source is not None:
            self.source = source
        if active_edge is not None:
            self.active_edge = active_edge
        if sample_mode is not None:
            self.sample_mode = sample_mode
        if samps_per_chan is not None:
            self.samps_per_chan = samps_per_chan

class InStream:
    def __init__(self, parent_task):
        self.parent_task = parent_task
