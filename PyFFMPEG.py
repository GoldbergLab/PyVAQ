import shutil
import subprocess
import numpy as np
import cv2
import time

FFMPEG_EXE = shutil.which('ffmpeg')
FFPROBE_EXE = shutil.which('ffprobe')

def getVideoSize(videoPath):
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
    return videoWidth, videoHeight, numFrames

def loadVideo(videoPath, numFramesOut=None, cropBox=None, size=None, pixFmt='rgb24'):
    # cropBox: (x, y, w, h)
    # size: (w, h)
    videoWidth, videoHeight, numFrames = getVideoSize(videoPath)

    if numFramesOut is None:
        numFramesout = numFrames

    filterList = []
    outWidth = videoWidth
    outHeight = videoHeight
    if cropBox is not None:
        filterList.append('crop={w}:{h}:{x}:{y}'.format(w=cropBox[2], h=cropBox[3], x=cropBox[0], y=cropBox[1]))
        outWidth = cropBox[2]
        outHeight = cropBox[3]
    if size is not None:
        filterList.append('scale={newW}:{newH}'.format(newW=size[0], newH=size[1]))
        outWidth = size[0]
        outHeight = size[1]

    filterString = ','.join(filterList)

    # Prepare to load ROI video data
    ffmpegCommand = [FFMPEG_EXE,
        '-i', videoPath,
        '-an',
        '-v', 'error',
        '-f', 'image2pipe'] + \
        ['-filter:v', filterString] * (len(filterList) > 0) + \
        ['-pix_fmt', pixFmt,
        '-vcodec', 'rawvideo', '-']

    # Load video ROI data using ffmpeg
    ffmpegProc = subprocess.Popen(
        ffmpegCommand,
        stdout=subprocess.PIPE,
        bufsize=10**8)

    video = np.zeros((outHeight, outWidth, 3, numFrames), dtype='uint8')
    for frameNum in range(numFrames):
        flatImage = ffmpegProc.stdout.read(outWidth*outHeight*3)
        image = np.frombuffer(flatImage, dtype='uint8')
        image = image.reshape((outHeight, outWidth, 3))
        video[:, :, :, frameNum] = image

    ffmpegProc.terminate()

    return video

def extractMotion(video):
    meanFrame = np.expand_dims(video.mean(axis=3), axis=3)
    stdFrame = np.expand_dims(video.std(axis=3), axis=3)
    zScoredVideo = np.mean(np.abs(video - meanFrame) / stdFrame, axis=2)
    zScoreThreshold = 2.5
    thresholdedVideo = ((zScoredVideo > zScoreThreshold) * 255).astype('uint8')
    return thresholdedVideo

def playVideo(video, fps=None):
    if fps is None:
        fps = 30
    print('Playing video...press \'q\' to stop...')
    for frameNum in range(v.shape[-1]):
        cv2.imshow('Video', video[..., frameNum])
        time.sleep(1 / fps)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print('Stopping at user request!')
            cv2.destroyAllWindows()
            break
    print('...done playing video.')

if __name__ == '__main__':
    print('loading video...')
    v = loadVideo(r'D:\Test Data\ZF Courtship\test data 2\test_21159096_2023-03-29-10-36-26-791048_1508.avi', numFramesOut=50, size=(1600, 1100))
    print('done loading')

    emv = extractMotion(v)
    v[emv == 255] == [255, 0, 0]

    playVideo(extractMotion(v), fps=5)
