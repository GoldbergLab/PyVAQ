import os, re
import os.path
import numpy as np


root = r'C:\Users\Goldberg\Documents\PyVAQ'
logFolder = os.path.join(root, 'logs')
logs = sorted(os.listdir(logFolder))
lastLog = os.path.join(logFolder, logs[-1])
with open(lastLog, 'r') as f:
    logText = f.read();

afreq = float(re.findall('actual audio frequency\: *([0-9\.]*)', logText)[0])
vfreq = float(re.findall('actual video frequency\: *([0-9\.]*)', logText)[0])
sampList = [int(sampNum) for sampNum in re.findall('\# samples\:([0-9]*)', logText)]
frameList = [int(frameNum) for frameNum in re.findall('\# frames\:([0-9]*)', logText)]
imageIDList = [int(imageID) for imageID in re.findall('Image ID\:([0-9]*)', logText)]
chunkSize = 1000
sampT = np.array(range(len(sampList))) / (afreq / chunkSize)
frameT = np.array(range(len(frameList))) / (vfreq)

frameInterp = np.interp(sampT, frameT, frameList)
print('Max audio/video chunk/frame discrepancy:')
print('Expected: <', max([chunkSize/afreq, 1/vfreq]))
print('Actual:    ', max((np.array(sampList) * vfreq / afreq)  - frameInterp) * (1/vfreq))

print('Dropped frames:', max(abs((1 + np.array(imageIDList)) - np.array(frameList))))
print('Errors:', re.findall('[Ee]rror', logText))
