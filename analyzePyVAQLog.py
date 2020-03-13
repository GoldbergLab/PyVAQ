import os, re
import os.path
import pprint
import numpy as np
import copy
import sys

def parseLog(logText):
    rawLogEntries = re.split(r'\|\| (?:[a-zA-Z0-9\-\_]* \- )?\*+ \/\\ ([a-zA-Z0-9 \_]*) \/\\ \**', logText, flags=re.MULTILINE)
    logEntries = {}
    for k in range(len(rawLogEntries)//2):
        index = k
        state = rawLogEntries[2*k+1].strip()
        content = rawLogEntries[2*k].strip().splitlines()
        if state not in logEntries:
            logEntries[state] = []
        logEntries[state].append((index, state, content))
    return logEntries

def summarizeLog(logEntries, filteredLogEntries):
    print("SUMMARY OF LOG MESSAGES:")
    lines = []
    allKeys = list(set(logEntries.keys()) | set(filteredLogEntries.keys()))
    keyHeader = 'Entry type'
    countHeader = 'Total #'
    filtCountHeader = 'Filtered #'
    maxKeyLength = len(keyHeader)
    maxCountLength = max(len(countHeader), len(filtCountHeader))
    for key in allKeys:
        if key in logEntries:
            count = str(len(logEntries[key]))
        else:
            count = '.'
        if key in filteredLogEntries:
            filtCount = str(len(filteredLogEntries[key]))
        else:
            filtCount = '.'
        lines.append([key, count, filtCount])
        maxKeyLength = max(maxKeyLength, len(key))
        maxCountLength = max(maxCountLength, len(count), len(filtCount))

    tableSpec = '{key:{fill}<{keyWidth}}{filtCount:{fill}<{countWidth}}{count:{fill}<{countWidth}}'
    fillChar = ' '
    print(tableSpec.format(key=keyHeader, count=countHeader, filtCount=filtCountHeader, fill=fillChar, keyWidth=maxKeyLength+1, countWidth=maxCountLength+1))
    for line in lines:
        key, count, filtCount = line
        print(tableSpec.format(key=key, count=count, filtCount=filtCount, fill=fillChar, keyWidth=maxKeyLength+1, countWidth=maxCountLength+1))

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

if len(sys.argv) > 1:
    # Don't use most recent log, look back N logs, N given by the first commandline argument
    lookback = int(sys.argv[1])
else:
    lookback = 0

#root = r'C:\Users\Goldberg\Documents\PyVAQ'
logFolder = os.path.join(root, 'logs')
logs = sorted(os.listdir(logFolder))
lastLog = os.path.join(logFolder, logs[-1-lookback])
print('Analyzing log:', lastLog)
with open(lastLog, 'r') as f:
    logText = f.read();

# afreq = float(re.findall('actual audio frequency\: *([0-9\.]*)', logText)[0])
# vfreq = float(re.findall('actual video frequency\: *([0-9\.]*)', logText)[0])
# sampList = [int(sampNum) for sampNum in re.findall('\# samples\:([0-9]*)', logText)]
# frameList = [int(frameNum) for frameNum in re.findall('\# frames\:([0-9]*)', logText)]
# imageIDList = [int(imageID) for imageID in re.findall('Image ID\:([0-9]*)', logText)]
# if len(sampList) * len(frameList) > 0:
#     chunkSize = 1000
#     sampT = np.array(range(len(sampList))) / (afreq / chunkSize)
#     frameT = np.array(range(len(frameList))) / (vfreq)
#
#     frameInterp = np.interp(sampT, frameT, frameList)
#     print('Max audio/video chunk/frame discrepancy:')
#     print('Expected: <', max([chunkSize/afreq, 1/vfreq]))
#     print('Actual:    ', max((np.array(sampList) * vfreq / afreq)  - frameInterp) * (1/vfreq))
#
#     print('Dropped frames:', max(abs((1 + np.array(imageIDList)) - np.array(frameList))))

logEntries = parseLog(logText)
filterList = []
filteredLogEntries = copy.deepcopy(logEntries)
ordered = False

summarizeLog(logEntries, filteredLogEntries)
print()

while True:
    printEntries = True
    filtInput = input("Enter a filtering command ('h' for help): ")
    try:
        filtType, filt = filtInput.split(' ', maxsplit=1)
    except ValueError:
        filtType = filtInput
        filt = ''
    newFilteredLogEntries = copy.deepcopy(filteredLogEntries)
    if filtType == 'i':
        # INDEX FILTERING
        newFilteredLogEntries = {}
        filterList.append(filtInput)
        cnt, rad = [int(i) for i in filt.strip().split(' ')]
        for key in filteredLogEntries:
            for entry in filteredLogEntries[key]:
                index, state, content = entry
                if abs(index - cnt) <= rad:
                    if key not in newFilteredLogEntries:
                        newFilteredLogEntries[key] = [];
                    newFilteredLogEntries[key].append(entry)
    elif filtType == 'rx':
        filterList.append(filtInput)
        filtRegex = re.compile(filt, flags=re.MULTILINE)
        groups = filtRegex.groups + 1
        extractedData = []
        matches = []
        maxMatchLengths = [0 for k in range(groups)]
        for key in filteredLogEntries.keys():
            for entry in filteredLogEntries[key]:
                index, state, content = entry
                matches += re.finditer(filtRegex, '\n'.join(content))
        for match in matches:
            extractedDatum = []
            for k in range(groups):
                extractedDatum.append(match.group(k))
                maxMatchLengths[k] = max(maxMatchLengths[k], len(extractedDatum[k])+2)
            extractedData.append(extractedDatum)

        print(''.join(['{data['+str(k)+']:{width['+str(k)+']}}' for k in range(groups)]).format(data=['Match'] + ['Group {k}'.format(k=k) for k in range(groups)], width=maxMatchLengths))
        for extractedDatum in extractedData:
            print(''.join(['{data['+str(k)+']:{width['+str(k)+']}}' for k in range(groups)]).format(data=extractedDatum, width=maxMatchLengths))
    elif filtType == 'r':
        # REGEX FILTERING
        newFilteredLogEntries = {}
        filterList.append(filtInput)
        negate = False
        if filt.split(' ')[0] == 'not':
            filt = filt.split(' ', maxsplit=1)[1]
            negate = True
        filtRegex = re.compile(filt, flags=re.MULTILINE)
        for key in filteredLogEntries.keys():
            for entry in filteredLogEntries[key]:
                index, state, content = entry
                if negate ^ bool(re.search(filtRegex, '\n'.join(content))):
                    if key not in newFilteredLogEntries:
                        newFilteredLogEntries[key] = []
                    newFilteredLogEntries[key].append(entry)
    elif filtType == 't':
        # ENTRY TYPE REGEX FILTERING
        newFilteredLogEntries = {}
        filterList.append(filtInput)
        filtRegex = re.compile(filt)
        for key in filteredLogEntries.keys():
            if re.search(filtRegex, key):
                newFilteredLogEntries[key] = []
                for entry in filteredLogEntries[key]:
                    newFilteredLogEntries[key].append(entry)
    elif filtType == 'c':
        printEntries = False
        filterList = []
        filteredLogEntries = copy.deepcopy(logEntries)
        newFilteredLogEntries = copy.deepcopy(filteredLogEntries)
        print('Clearing all filters')
    elif filtType == 'o':
        ordered = not ordered
        if ordered:
            print('Index ordering is now on')
        else:
            print('Index ordering is now off')
    elif filtType == 'h':
        printEntries = False
        # Help with filtering:
        helptext = '''
Filter syntax:
    Regex filtering:
        r REGEX
            REGEX = a regular expression to filter by
        r not REGEX
            Displays all entries that do NOT match the regex
    Index filtering:
        i MID RAD
            Displays all entries numnbers from MID - RAD to MID + RAD
    Toggle ordering
        o
        Toggle between printing ordered by time index, or sorted by log type
    Display help
        h
        Print help text'''
        print(helptext)
        print()
    else:
        printEntries = False
        print('Filter command not recognized. Type "h" for help.')
        continue

    filteredLogEntries = copy.deepcopy(newFilteredLogEntries)

    if printEntries:
        filteredCount = sum([len(filteredLogEntries[key]) for key in filteredLogEntries])

        if filteredCount > 100:
            howMany = input('There are {n} entries. Display all/some/none (a/s/n)? '.format(n=filteredCount))
            if howMany == 'a':
                printLog(filteredLogEntries, abridge=False, ordered=ordered)
            elif howMany == 's':
                printLog(filteredLogEntries, abridge=True, ordered=ordered)
        else:
            printLog(filteredLogEntries, abridge=False, ordered=ordered)

        print()

    summarizeLog(logEntries, filteredLogEntries)
    print()

    if len(filterList) > 0:
        print('Filters:')
        for filt in filterList:
            print('\t'+filt)
        print()


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
