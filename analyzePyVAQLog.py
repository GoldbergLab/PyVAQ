import os, re
import os.path
import pprint
import numpy as np
import copy

def parseLog(logText):
    rawLogEntries = re.split(r'\|\|\ ([a-zA-Z0-9]*\ \-\ )?\*+\ \/\\ ([a-zA-Z0-9\ \_]*) \/\\ \**', logText, flags=re.MULTILINE)
    logEntries = {}
    for k in range(len(rawLogEntries)//2):
        index = k
        state = rawLogEntries[2*k+1].strip()
        content = rawLogEntries[2*k].strip().splitlines()
        if state not in logEntries:
            logEntries[state] = []
        logEntries[state].append((index, content))
    return logEntries

def summarizeLog(logEntries):
    print("SUMMARY OF LOG MESSAGES:")
    for key in logEntries.keys():
        print(key + ": " + str(len(logEntries[key])))

def printLog(logEntries, abridge=False, ordered=False):
    if ordered:
        flattenedLogEntries = []
        for key in logEntries:
            flattenedLogEntries = flattenedLogEntries + logEntries[key]
        logEntries = sorted(flattenedLogEntries, key=lambda entry:entry[0])
        if abridge:
            threshold = 50
            logEntries = logEntries[:threshold//2-1] + ['...'] + logEntries[-threshold//2:]
    else:
        if abridge:
            threshold = 10
            logEntries = copy.deepcopy(logEntries)
            logEntryKeys = logEntries.keys()
            for key in logEntryKeys:
                if len(logEntries[key]) > threshold:
                    logEntries[key] = logEntries[key][:threshold//2-1] + ['...'] + logEntries[key][-threshold//2:]
    pp.pprint(logEntries)


pp = pprint.PrettyPrinter(width=240)
root, thisScript = os.path.split(os.path.realpath(__file__))

#root = r'C:\Users\Goldberg\Documents\PyVAQ'
logFolder = os.path.join(root, 'logs')
logs = sorted(os.listdir(logFolder))
lastLog = os.path.join(logFolder, logs[-1])
print('Analyzing log:', lastLog)
with open(lastLog, 'r') as f:
    logText = f.read();

afreq = float(re.findall('actual audio frequency\: *([0-9\.]*)', logText)[0])
vfreq = float(re.findall('actual video frequency\: *([0-9\.]*)', logText)[0])
sampList = [int(sampNum) for sampNum in re.findall('\# samples\:([0-9]*)', logText)]
frameList = [int(frameNum) for frameNum in re.findall('\# frames\:([0-9]*)', logText)]
imageIDList = [int(imageID) for imageID in re.findall('Image ID\:([0-9]*)', logText)]
if len(sampList) * len(frameList) > 0:
    chunkSize = 1000
    sampT = np.array(range(len(sampList))) / (afreq / chunkSize)
    frameT = np.array(range(len(frameList))) / (vfreq)

    frameInterp = np.interp(sampT, frameT, frameList)
    print('Max audio/video chunk/frame discrepancy:')
    print('Expected: <', max([chunkSize/afreq, 1/vfreq]))
    print('Actual:    ', max((np.array(sampList) * vfreq / afreq)  - frameInterp) * (1/vfreq))

    print('Dropped frames:', max(abs((1 + np.array(imageIDList)) - np.array(frameList))))

logEntries = parseLog(logText)
summarizeLog(logEntries)

while True:
    filt = input("Enter a regex keyword to filter the log entries by or 'i' to search by index: ")
    filteredLogEntries = {}
    filteredCount = 0
    if filt == 'i':
        ordered=True
        cnt, rad = [int(i) for i in input("Enter center and radius indices separated by a space: ").split(' ')]
        entries = []
        for key in logEntries:
            for entry in logEntries[key]:
                index, content = entry
                entry = (index, key, content) # Add type into entry
                if abs(index - cnt) <= rad:
                    if key not in filteredLogEntries:
                        filteredLogEntries[key] = [];
                    filteredLogEntries[key].append(entry)
                    filteredCount += 1
    else:
        negate = False
        if filt.split(' ')[0] == 'not':
            filt = filt.split(' ', maxsplit=1)[1]
            negate = True
        ordered=False
        filtRegex = re.compile(filt, flags=re.MULTILINE)
        for key in logEntries.keys():
            for entry in logEntries[key]:
                index, content = entry
                if negate ^ bool(re.search(filtRegex, '\n'.join(content))):
                    if key not in filteredLogEntries:
                        filteredLogEntries[key] = [];
                    filteredLogEntries[key].append(entry)
                    filteredCount += 1

    if filteredCount > 100:
        howMany= input('There are {n} entries. Display all/some/none (a/s/n)? '.format(n=filteredCount))
        if howMany == 'a':
            printLog(filteredLogEntries, abridge=False, ordered=ordered)
        elif howMany == 's':
            printLog(filteredLogEntries, abridge=True, ordered=ordered)
    else:
        printLog(filteredLogEntries, abridge=False, ordered=ordered)

    print()

    summarizeLog(logEntries)


#errorList = re.findall('([Ee]rror in [a-zA-Z\ ]*)([^\|]*)\|\|', logText, flags=(re.MULTILINE | re.DOTALL))
#for errorState, errorDescription in errorList:
#    errorParts = errorDescription.split()
#    errorName = errorParts[-1]
#    print(errorState, errorName)
r'''
|| *********************************** /\ AW BUFFERING /\ ********************************************
|| Update trigger to stop now
|| msg=, exitFlag=False
|| *********************************** /\ AT ANALYZING /\ ********************************************
|| VW_19355735 - partially missed trigger by 2.0140631198883057 seconds, which is 60.421908702126345 frames!
|| msg=, exitFlag=False
|| *********************************** /\ VW_19355735 BUFFERING /\ ********************************************
|| VA_19355735 ERROR STATE. Error messages:


|| Error in ACQUIRING state

Traceback (most recent call last):
  File "C:\Users\GLab\Documents\PyVAQ\StateMachineProcesses.py", line 2565, in run
    self.imageQueue.put(imageResult, metadata={'frameTime':frameTime})
  File "C:\Users\GLab\Documents\PyVAQ\SharedImageQueue.py", line 113, in put
    self.metadataQueue.put(metadata, block=False)
  File "C:\Users\GLab\AppData\Local\Programs\Python\Python37\lib\multiprocessing\queues.py", line 83, in put
    raise Full
queue.Full

|| msg=, exitFlag=False
|| *********************************** /\ VA_19355735 ERROR /\ ********************************************
|| AW - Sending audio filename to merger
|| msg=, exitFlag=False
|| *********************************** /\ AW WRITING /\ ********************************************
|| Send new trigger!
'''
