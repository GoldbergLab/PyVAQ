import subprocess
import sys
import shutil
import numpy as np
from matplotlib import pyplot as plt
from scipy.io import savemat
from pathlib import Path
from numpy.core.records import fromarrays
import time

# A script to process a video using one or more ROIs to detect which ROI appears to have the most motion in it.
# This script depends on ffmpeg and ffprobe being installed on the system and available on the system path
#
# Usage:
#
# python findMotion.py path/to/input/video/file/or/directory --roi 100:200x300:400 --roi 250:350x300:400 -m *.avi
#
# This would seach the given directory for avi files, and analyze the motion found in the two ROIs.
#   The first ROI spans x coordinates 100 to 200 and y coordinates 300 to 400.
#   The second ROI spans x coordinates 250 to 350 and y coordinates 300 to 400.
#
# The output will be stored in the same folder as the videos as a file called "motionTracking.mat"
# A validation directory will also be created containing a labeled tracking validation video
#

FFMPEG_EXE = shutil.which('ffmpeg')
FFPROBE_EXE = shutil.which('ffprobe')

def parseROI(roiString):
    # Format: X0:X1xY0:Y1
    return tuple([tuple([int(coord) for coord in subString.split(':')]) for subString in roiString.split('x')])

def moving_average_other(a, n=3):
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

def moving_average(a, n=3):
    return np.convolve(a, np.ones(n), mode='same')

def createSubtitleEntry(idx, t0, t1, text):
    entry = []
    entry.append(str(idx) + '\n')
    entry.append(getTimeStamp(t0) + ' --> ' + getTimeStamp(t1) + '\n')
    entry.append(text + '\n')
    entry.append('\n')
    return entry

def getTimeStamp(totalSeconds):
    milliseconds = int(totalSeconds*1000) % 1000
    seconds = int(totalSeconds) % 60
    minutes = int(totalSeconds // 60) % 60
    hours = int(totalSeconds // (60*60))
    return '{hours:02d}:{minutes:02d}:{seconds:02d},{millis:03d}'.format(hours=hours, minutes=minutes, seconds=seconds, millis=milliseconds)

def displayHelp():
    print('')
    print('findMotion.py')
    print('')
    print('     A script for identifying motion in video ROIs (regions of interest).')
    print('')
    print('Example usage:')
    print('     python findMotion.py -h')
    print('         Display this help message and exit')
    print('')
    print('     python findMotion.py path/to/directory -m *.avi --roi 100:150x55:85')
    print('         Process all files matching the pattern *.avi within the')
    print('         given directory, and look for motion in a single ROI defined')
    print('         by the rectangle spanning from pixel (100, 55) to (150, 85)')
    print('')
    print('     python findMotion.py path/to/video.avi --roi 100:150x55:85')
    print('         Process a single video at the given path, and look for')
    print('         motion in a single ROI defined by the rectangle spanning')
    print('         from pixel (100, 55) to (150, 85)')
    print('')
    print('     python findMotion.py path/to/directory -m *.avi --roi 100:150x55:85 --roi 250:320x110:180')
    print('         Process all files matching the pattern *.avi within the')
    print('         given directory, and look for motion in two ROIs, one')
    print('         defined by the rectangle spanning from pixel (100, 55) to')
    print('         (150, 85), and the other from (250, 110) to (320, 180')

if __name__ == '__main__':
    startTime = time.time()

    # Define some variables/parameters
    frameRate = 30
    hysteresisThreshold = 0.25  # Min confidence required to switch ROI tracks
    ROIs = []
    pattern = None
    batchRun = False
    videoDir = None

    if len(sys.argv) == 1:
        displayHelp()
        sys.exit()

    # Parse input arguments (video path, ROIs, etc)
    for argNum, arg in enumerate(sys.argv):
        if argNum == 0:
            continue
        elif arg == '-h':
            displayHelp()
            sys.exit()
        elif argNum == 1:
            videoPath = Path(arg)
        elif arg == '-m':
            pattern = sys.argv[argNum+1]
        else:
            if arg == '--roi':
                ROIs.append(parseROI(sys.argv[argNum+1]))

    if videoPath.is_dir():
        # User has passed in a diretory. Loop over all files in it.
        batchRun = True
        videoDir = videoPath
        if pattern is None:
            raise ValueError('If you specify a directory to batch-process videos, you must supply a filename pattern with the -m flag. For example: "-m *.avi"')
        videoList = sorted(videoDir.glob(pattern))
        # videoList = videoList[:10]
    else:
        # User just passed in a single file
        videoList = [videoPath]

    # Set up variables to hold data for processing
    allTracking = np.array([], dtype='uint8')
    allMotion = {}
    allVideoIndices = []
    roiImageStack = {}
    lastROIImageStack = {}

    # Factor by which to downsample images before processing, for speed
    downsampleFactor = 4

    # Loop over video list
    for videoNum, videoPath in enumerate(videoList):
        # Prepare to query video size
        ffprobeCommand = [
            FFPROBE_EXE,
            '-v', 'error',
            '-select_streams', 'v:0',
            '-count_packets', '-show_entries',
            'stream=nb_read_packets', '-show_entries', 'stream=width,height',
            '-of', 'csv=p=0', videoPath
        ]

        # Query video size using ffprobe
        ffprobeProc = subprocess.Popen(
            ffprobeCommand,
            stdout=subprocess.PIPE)

        videoWidth, videoHeight, numFrames = [int(d) for d in (ffprobeProc.stdout.readline()).decode('utf-8').split(',')]

        print('Motion tracking video #{videoNum} of {numVideos}: {path}'.format(videoNum=videoNum+1, numVideos=len(videoList), path=videoPath))

        # Loop over requested ROIs and load video data
        for roiNum, roi in enumerate(ROIs):
            print('    Loading ROI #{roiNum}'.format(roiNum=roiNum))
            roiX0 = roi[0][0]
            roiY0 = roi[1][0]
            roiX1 = roi[0][1]
            roiY1 = roi[1][1]
            roiWidth = roiX1-roiX0
            roiHeight = roiY1-roiY0
            newROIWidth = roiWidth//downsampleFactor
            newROIHeight = roiHeight//downsampleFactor
            cropString = 'crop={w}:{h}:{x}:{y},scale={newW}:{newH}'.format(w=roiWidth, h=roiHeight, x=roiX0, y=roiY0, newW=newROIWidth, newH=newROIHeight)
            roiWidth = newROIWidth
            roiHeight = newROIHeight

            # Prepare to load ROI video data
            ffmpegCommand = [FFMPEG_EXE,
                '-i', videoPath,
                '-an',
                '-v', 'error',
                '-f', 'image2pipe',
                '-filter:v', cropString,
                '-pix_fmt', 'rgb24',
                '-vcodec', 'rawvideo', '-']

            # Load video ROI data using ffmpeg
            ffmpegProc = subprocess.Popen(
                ffmpegCommand,
                stdout=subprocess.PIPE,
                bufsize=10**8)

            print('    Processing ROI #{roiNum}'.format(roiNum=roiNum))

            roiImageStack[roi] = np.zeros((roiHeight, roiWidth, 3, numFrames), dtype='int16')
            for frameNum in range(numFrames):
                raw_image = ffmpegProc.stdout.read(roiWidth*roiHeight*3)
                roiImage = np.frombuffer(raw_image, dtype='uint8')
                roiImage = roiImage.reshape((roiHeight, roiWidth, 3))
                roiImageStack[roi][:, :, :, frameNum] = roiImage

            ffmpegProc.terminate()

            # Extract preliminary motion signal in ROI data
            if roi not in lastROIImageStack:
                # Prepend one duplicate starting frame so diff will have same # of frames, for analysis convenience
                roiImageStack[roi] = np.append(roiImageStack[roi][:, :, :, [0]], roiImageStack[roi], axis=3)
            else:
                # Prepend one frame from the previous video so diff will have same # of frames, for analysis convenience
                roiImageStack[roi] = np.append(lastROIImageStack[roi][:, :, :, [-1]], roiImageStack[roi], axis=3)
            # Analyze amount of motion in frames
            newMotion = np.abs(np.diff(roiImageStack[roi])).mean(axis=(0, 1, 2))
            lastROIImageStack[roi] = roiImageStack[roi]
            if roi not in allMotion:
                allMotion[roi] = newMotion
            else:
                allMotion[roi] = np.append(allMotion[roi], newMotion)

        allVideoIndices.append((len(allMotion[ROIs[0]]) - len(newMotion), len(allMotion[ROIs[0]])))

        print()

    # Smooth motion data
    allMotion = {roi:moving_average(allMotion[roi], n=20) for roi in ROIs}

    # Extract preliminary ROI choices from ROI data
    allMotionStacked = np.stack([allMotion[roi] for roi in ROIs], axis=1)
    motionSort = np.sort(allMotionStacked, axis=1)
    confidence = abs((motionSort[:, -1] - motionSort[:, -2]) / motionSort[:, -1])
    allTracking = np.argmax(allMotionStacked, axis=1).astype(dtype='uint8')

    # Apply hysteresis threshold to decrease spurious ROI bouncing
    currentROINum = allTracking[0]
    numHysteresisSwaps = 0
    for k, newROINum in enumerate(allTracking):
        if newROINum != currentROINum:
            # Tracked ROI has changed
            if confidence[k] < hysteresisThreshold:
                # confidence level is not high enough - keep previously tracked ROI
                allTracking[k] = currentROINum
                numHysteresisSwaps = numHysteresisSwaps + 1
            else:
                # confidence levl is high enough - switch ROI
                currentROINum = newROINum
    print('Number of hysteresis switches: {n}'.format(n=numHysteresisSwaps))

    # Output summary for whole run
    output = {}
    output['ROIs'] = [fromarrays(roi[0]+roi[1], names=['x0', 'x1', 'y0', 'y1']) for roi in ROIs]
    output['motion'] = [allMotion[roi] for roi in allMotion]
    output['tracking'] = allTracking
    output['videoDir'] = str(videoDir)
    output['videoList'] = np.array([str(path) for path in videoList], dtype='object')
    output['videoIndices'] = allVideoIndices
    output['confidence'] = confidence
    matPath = videoPath.parents[0] / ('motionTracking.mat')
    savemat(matPath, output)
    endTime = time.time()
    print('Elapsed time: {t}'.format(t=endTime-startTime))

    # Create validation video with subtitles
    validationDirectory = videoPath.parents[0] / 'validation'
    validationDirectory.mkdir(exist_ok=True)
    print('Created validation directory:')
    print(validationDirectory)
    tempFileList = validationDirectory / 'tempFileList.txt';
    with open(tempFileList, 'w') as f:
        f.writelines(['file \'{path}\'\n'.format(path=videoPath) for videoPath in videoList])

    print('Created validation temp file list:')
    print(tempFileList)

    validationSubtitles = validationDirectory / 'motionTrackingValidation.srt'
    currentROINum = allTracking[0]
    trackingSegment = [0, None]
    subtitleCount = 1
    with open(validationSubtitles, 'w') as f:
        for k in range(len(allTracking)):
            if (currentROINum != allTracking[k]) or (k == len(allTracking)-1):
                trackingSegment[1] = k-1
                f.writelines(createSubtitleEntry(subtitleCount, trackingSegment[0]/frameRate, trackingSegment[1]/frameRate, str(currentROINum)))
                trackingSegment[0] = k
                trackingSegment[1] = None
                currentROINum = allTracking[k]
                subtitleCount = subtitleCount + 1

    print('Created validation subtitle file: ')
    print(validationSubtitles)

    validationDownsampleFactor = 4

    validationVideo = validationDirectory / 'motionTrackingValidation.avi'

    print('Creating validation video:')
    print(validationVideo)

    ffmpegCommand = [FFMPEG_EXE,
        '-f', 'concat',
        '-y', '-v', 'error',
        '-safe', '0',
        '-i', str(tempFileList),
        '-filter:v', 'scale={w}:{h}'.format(w=videoWidth//validationDownsampleFactor, h=videoHeight//validationDownsampleFactor),
        '-c:v', 'libx264', '-crf', '25',
        '-c:a', 'copy',
        '-c:s', str(validationSubtitles),
        str(validationVideo)]

    ffmpegProc = subprocess.Popen(
        ffmpegCommand)

# plt.figure(0)
# plt.plot(maxVals)
#
# for roiNum, roi in enumerate(ROIs):
#     plt.figure(roiNum+1)
#     plt.plot(roiTraces[roi])
# #    plt.imshow(roiImages[roi][:, :, :])
# plt.show()
