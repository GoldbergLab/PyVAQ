class AudioTriggerer(mp.Process):
    # States:
    STOPPED = 'STOPPED'
    INITIALIZING = 'INITIALIZING'
    WAITING = 'WAITING'
    ANALYZING = 'ANALYZING'
    STOPPING = 'STOPPING'
    ERROR = 'ERROR'
    EXITING = 'EXITING'

    #messages:
    START = 'msg_start'
    STARTNALYZE = "msg_startanalyze"
    STOPANALYZE = "msg_stopanalyze"
    STOP = 'msg_stop'
    EXIT = 'msg_exit'
    SETPARAMS = 'msg_setParams'

    settableParams = [
        'audioFrequency',
        'triggerHighLevel',
        'triggerLowLevel',
        'triggerHighTime',
        'triggerLowTime',
        'maxTriggerTime',
        'multiChannelStartBehavior',
        'multiChannelStopBehavior',
        'verbose'
        ]

    def __init__(self,
                audioQueue=None,
                audioFrequency=44100,
                triggerHighLevel=0.5,               # Volume level above which the audio must stay for triggerHighTime seconds to generate a start trigger
                triggerLowLevel=0.1,                # Volume level below which the audio must stay for triggerLowTime seconds to generate an updated (stop) trigger
                triggerHighTime=2,                  # Length of time that volume must stay above triggerHigh
                triggerLowTime=1,                   # Length of time that volume must stay below triggerLowLevel
                maxTriggerTime=20,                  # Maximum length of trigger regardless of volume levels
                multiChannelStartBehavior='OR',     # How to handle multiple channels of audio. Either 'OR' (start when any channel goes higher than high threshold) or 'AND' (start when all channels go higher than high threshold)
                multiChannelStopBehavior='OR',      # How to handle multiple channels of audio. Either 'OR' (stop when any channel goes lower than low threshold) or 'AND' (stop when all channels go lower than low threshold)
                verbose=False,
                audioMessageQueue=None,             # Queue to send triggers to audio writers
                videoMessageQueues=[],              # Queues to send triggers to video writers
                messageQueue=None,                  # Queue for getting commands to change state
                stdoutQueue=None):
        mp.Process.__init__(self, daemon=True)
        self.audioQueue = audioQueue
        self.audioQueue.cancel_join_thread()
        self.audioFrequency = audioFrequency
        self.triggerHighLevel = triggerHighLevel
        self.triggerLowLevel = triggerLowLevel
        self.triggerHighTime = triggerHighTime
        self.triggerLowTime = triggerLowTime
        self.maxTriggerTime = maxTriggerTime
        self.multiChannelStartBehavior = multiChannelStartBehavior
        self.multiChannelStopBehavior = multiChannelStopBehavior
        self.messageQueue = messageQueue

        self.errorMessages = []
        self.exitFlag = False
        self.analyzeFlag = False
        self.verbose = verbose
        self.stdoutQueue = stdoutQueue
        self.stdoutBuffer = []

    def setParams(self, **params):
        for key in params:
            if key in AudioTriggerer.settableParams:
                setattr(self, key, params[key])
                if self.verbose: syncPrint("AT - Param set: {key}={val}".format(key=key, val=params[key]), buffer=self.stdoutBuffer)
            else:
                syncPrint("AT - Param not settable: {key}={val}".format(key=key, val=params[key]), buffer=self.stdoutBuffer)

    def run(self):
        syncPrint("AT - PID={pid}".format(pid=os.getpid()), buffer=self.stdoutBuffer)
        state = AudioTriggerer.STOPPED
        nextState = AudioTriggerer.STOPPED
        lastState = AudioTriggerer.STOPPED
        while True:
            try:
# ********************************* STOPPPED *********************************
                if state == AudioTriggerer.STOPPED:
                    # DO STUFF

                    # CHECK FOR MESSAGES
                    try:
                        msg, arg = self.messageQueue.get(block=True, timeout=0.1)
                        if msg == AudioTriggerer.SETPARAMS: self.setParams(**arg); msg = ''; arg=None
                    except queue.Empty: msg = ''; arg = None

                    # CHOOSE NEXT STATE
                    if msg == AudioTriggerer.EXIT or self.exitFlag:
                        self.exitFlag = True
                        nextState = AudioTriggerer.EXITING
                    elif msg == '':
                        nextState = AudioTriggerer.STOPPED
                    elif msg == AudioTriggerer.START:
                        nextState = AudioTriggerer.INITIALIZING
                    elif msg == AudioTriggerer.STARTANALYZE:
                        self.analyzeFlag = True
                        nextState = AudioTriggerer.INITIALIZING
                    else:
                        raise SyntaxError("Message \"" + msg + "\" not relevant to " + state + " state")
# ********************************* INITIALIZING *****************************
                elif state == AudioTriggerer.INITIALIZING:
                    # DO STUFF
                    activeTrigger = None

                    # CHECK FOR MESSAGES
                    try:
                        msg, arg = self.messageQueue.get(block=False)
                        if msg == AudioTriggerer.SETPARAMS: self.setParams(**arg); msg = ''; arg=None
                    except queue.Empty: msg = ''; arg = None

                    # CHOOSE NEXT STATE
                    if msg == AudioTriggerer.EXIT or self.exitFlag:
                        self.exitFlag = True
                        nextState = AudioTriggerer.STOPPING
                    elif msg == AudioTriggerer.STOP:
                        nextState = AudioTriggerer.STOPPING
                    elif msg == '' or msg == AudioTriggerer.STOPANALYZE:
                        nextState = AudioTriggerer.WAITING
                    elif msg == AudioTriggerer.STARTANALYZE or self.analyzeFlag:
                        nextState = AudioTriggerer.ANALYZING
                    else:
                        raise SyntaxError("Message \"" + msg + "\" not relevant to " + state + " state")
# ********************************* WAITING ********************************
                elif state == AudioTriggerer.WAITING:
                    # DO STUFF

                    # CHECK FOR MESSAGES
                    try:
                        msg, arg = self.messageQueue.get(block=False)
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
                    elif msg == '' or msg == AudioTriggerer.STOPANALYZE:
                        nextState = AudioTriggerer.WAITING
                    else:
                        raise SyntaxError("Message \"" + msg + "\" not relevant to " + state + " state")
# ********************************* ANALYZING *********************************
                elif state == AudioTriggerer.ANALYZING:
                    # DO STUFF
                    self.analyzeFlag = False

                    # CHECK FOR MESSAGES
                    try:
                        msg, arg = self.messageQueue.get(block=False)
                        if msg == AudioTriggerer.SETPARAMS: self.setParams(**arg); msg = ''; arg=None
                    except queue.Empty: msg = ''; arg = None

                    # CHOOSE NEXT STATE
                    if self.verbose: syncPrint("AT - |{startState} ---- {endState}|".format(startState=chunkStartTriggerState, endState=chunkEndTriggerState), buffer=self.stdoutBuffer)
                    if msg == AudioTriggerer.EXIT or self.exitFlag:
                        self.exitFlag = True
                        nextState = AudioTriggerer.STOPPING
                    elif msg == AudioTriggerer.STOP:
                        nextState = AudioTriggerer.STOPPING
                    elif msg == AudioTriggerer.STOPANALYZE:
                        nextState = AudioTriggerer.WAITING
                    elif msg == '' or msg == AudioTriggerer.STARTANALYE:
                        nextState = state
                    else:
                        raise SyntaxError("Message \"" + msg + "\" not relevant to " + state + " state")
# ********************************* STOPPING *********************************
                elif state == AudioTriggerer.STOPPING:
                    # DO STUFF

                    # CHECK FOR MESSAGES
                    try:
                        msg, arg = self.messageQueue.get(block=False)
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
                    else:
                        raise SyntaxError("Message \"" + msg + "\" not relevant to " + state + " state")
# ********************************* ERROR *********************************
                elif state == AudioTriggerer.ERROR:
                    # DO STUFF
                    syncPrint("AT - ERROR STATE. Error messages:\n\n", buffer=self.stdoutBuffer)
                    syncPrint("\n\n".join(self.errorMessages), buffer=self.stdoutBuffer)
                    self.errorMessages = []

                    # CHECK FOR MESSAGES
                    try:
                        msg, arg = self.messageQueue.get(block=False)
                        if msg == AudioTriggerer.SETPARAMS: self.setParams(**arg); msg = ''; arg=None
                    except queue.Empty: msg = ''; arg = None

                    # CHOOSE NEXT STATE
                    if msg == '':
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
                        raise SyntaxError("Message \"" + msg + "\" not relevant to " + state + " state")
# ********************************* EXIT *********************************
                elif state == AudioTriggerer.EXITING:
                    break
                else:
                    raise KeyError("Unknown state: "+state)
            except KeyboardInterrupt:
                # Handle user using keyboard interrupt
                if self.verbose: syncPrint("AT - Keyboard interrupt received - exiting", buffer=self.stdoutBuffer)
                self.exitFlag = True
                nextState = AudioTriggerer.STOPPING
            except:
                # HANDLE UNKNOWN ERROR
                self.errorMessages.append("Error in "+state+" state\n\n"+traceback.format_exc())
                nextState = AudioTriggerer.ERROR

            if self.verbose:
                if msg != '' or self.exitFlag:
                    syncPrint("msg={msg}, exitFlag={exitFlag}".format(msg=msg, exitFlag=self.exitFlag), buffer=self.stdoutBuffer)
                syncPrint('*********************************** /\ AW ' + state + ' /\ ********************************************', buffer=self.stdoutBuffer)
            if len(self.stdoutBuffer) > 0: self.stdoutQueue.put(self.stdoutBuffer)
            self.stdoutBuffer = []

            # Prepare to advance to next state
            lastState = state
            state = nextState

        if self.verbose: syncPrint("Audio write process STOPPED", buffer=self.stdoutBuffer)
        if len(self.stdoutBuffer) > 0: self.stdoutQueue.put(self.stdoutBuffer)
        self.stdoutBuffer = []
