from nidaqmxSim.task import DIChannel, AIChannel, COChannel

DEFAULT_DEVICE_NAMES = [
    'SimDev1',
    'SimDev2'
]

class System:
    def __init__(self, device_names=DEFAULT_DEVICE_NAMES):
        self.devices = [Device(name) for name in DEFAULT_DEVICE_NAMES]
    @classmethod
    def local(cls):
        return cls()

DEFAULT_AI_CHAN_NAMES = [
    'ai0',
    'ai1',
    'ai2',
    'ai3'
]
DEFAULT_DI_CHAN_NAMES = [
    'di0',
    'di1',
    'di2',
    'di3'
]
DEFAULT_CO_CHAN_NAMES = [
    'co0',
    'co1'
]

class Device:
    def __init__(self, name, ai_chan_names=DEFAULT_AI_CHAN_NAMES, di_chan_names=DEFAULT_DI_CHAN_NAMES, co_chan_names=DEFAULT_CO_CHAN_NAMES):
        self.name = name
        self.ai_physical_chans = []
        self.terminals = []
        self.co_physical_chans = []
        ai_chan_names = ['{dev}/{chan}'.format(dev=self.name, chan=chan) for chan in ai_chan_names]
        di_chan_names = ['{dev}/{chan}'.format(dev=self.name, chan=chan) for chan in di_chan_names]
        co_chan_names = ['{dev}/{chan}'.format(dev=self.name, chan=chan) for chan in co_chan_names]
        for chan in ai_chan_names:
            self.ai_physical_chans.append(AIChannel(channel=chan))
        for chan in di_chan_names:
            self.terminals.append(DIChannel(channel=chan))
        for chan in co_chan_names:
            self.co_physical_chans.append(COChannel(channel=chan))
